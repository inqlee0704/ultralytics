# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from tqdm import tqdm

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics3d import Metric3D
from ultralytics.utils.plotting3d import Plot3D
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.metrics import box_iou_3d
from ultralytics.utils.ops import Profile
import time
from pathlib import Path
import torchvision

class BaseValidator3D:
    """Base validator for 3D object detection."""
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize 3D validator."""
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.pbar = pbar
        self.args = args
        self.args.task = "detect3d"
        self.args.iou_thres = 0.5
        # self.metrics = Metric3D(nc=6)
        # self.metrics = Metric3D(nc=args.nc)
        self.plot = Plot3D()
        
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes 3D validation process."""
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half = self.device.type != "cpu"
            model = AutoBackend(
                weights=model or self.args.model,
                device=self.device,
                dnn=self.args.dnn,
                fp16=self.args.half
            )
            self.device = model.device
            self.args.half = model.fp16
            model.eval()
            model.warmup()  # warmup model

        dt = [Profile(device=self.device) for _ in range(4)]
        bar = TQDM(self.dataloader, desc="Validating", total=len(self.dataloader))
        self.seen = 0
        # self.metrics.reset()
        self.init_metrics(de_parallel(model))
        
        for batch_i, batch in enumerate(bar):
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"])

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            
            # Visualization
            if self.args.plots and batch_i < 3:
                self.plot.plot_3d_predictions(batch, preds, batch_i, self.save_dir)

        # Calculate statistics
        # stats = self.metrics.evaluate(iou_thres=self.args.iou_thres)
        stats = self.get_stats()
        self.speed = dict(zip(['preprocess', 'inference', 'loss', 'postprocess'],
                            (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))

        # Print results
        LOGGER.info(f"Speed: {self.speed['preprocess']:.1f}ms preprocess, "
                    f"{self.speed['inference']:.1f}ms inference, "
                    f"{self.speed['loss']:.1f}ms loss, "
                    f"{self.speed['postprocess']:.1f}ms postprocess per volume")
        # LOGGER.info(f"mAP@{self.args.iou_thres:.2f}: {stats[0]:.4f}")

        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        return stats 

    def preprocess(self, batch):
        """Preprocess batch data."""
        batch["volume"] = batch["volume"].to(self.device, non_blocking=True)
        batch["labels"] = batch["labels"].to(self.device)
        return batch
        
    def postprocess(self, preds):
        """Apply NMS and filter predictions."""
        return self.non_max_suppression_3d(
            preds,
            conf_thres=self.args.conf_thres,
            iou_thres=self.args.iou_thres,
            max_det=self.args.max_det
        )
    
    @staticmethod
    def non_max_suppression_3d(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """Perform NMS on 3D predictions."""
        bs = prediction.shape[0]
        nc = prediction.shape[2] - 6  # number of classes
        xc = prediction[..., 6:].max(2) > conf_thres  # candidates
        
        # Settings
        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        
        output = [torch.zeros((0, 7))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            
            if not x.shape[0]:
                continue
                
            # Compute conf
            x[:, 6:] *= x[:, 5:6]  # conf = obj_conf * cls_conf
            
            # Box (center x, center y, center z, depth, height, width)
            box = x[:, :6]
            conf = x[:, 6:].max(1, keepdim=True)[0]  # conf = max conf
            j = x[:, 6:].argmax(1, keepdim=True)  # class
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            
            # Filter by class
            if (x[:, 7] != j).any():
                x = x[x[:, 7] == j.view(-1)]  # separate by class
            
            # Check shape
            n = x.shape[0]
            if not n:
                continue
            elif n > max_nms:
                x = x[x[:, 6].argsort(descending=True)[:max_nms]]  # sort by confidence
            
            # Batched NMS
            c = x[:, 7:8] * max_wh  # classes
            boxes, scores = x[:, :6], x[:, 6]  # boxes, scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            
            output[xi] = x[i]
            
        return output
    
    def update_metrics(self, preds, batch):
        """Update metrics."""
        for si, pred in enumerate(preds):
            labels = batch["labels"][batch["labels"][:, 0] == si][:, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            
            # Metrics
            tp, conf, pred_cls, target_cls = [], [], [], []
            for *box, conf, cls in pred:
                if nl:
                    # Find best matching label
                    iou, j = box_iou_3d(torch.tensor(box).unsqueeze(0), labels[:, :6]).max(0)
                    if iou > self.args.iou_thres:
                        tp.append(1)
                        conf.append(conf)
                        pred_cls.append(cls)
                        target_cls.append(labels[j, 6])
                    else:
                        tp.append(0)
                        conf.append(conf)
                        pred_cls.append(cls)
                        target_cls.append(-1)
                        
            self.metrics.process(
                torch.tensor(tp, device=self.device),
                torch.tensor(conf, device=self.device),
                torch.tensor(pred_cls, device=self.device),
                torch.tensor(target_cls, device=self.device)
            )
    
    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.metrics.reset()
        
        pbar = tqdm(self.dataloader, desc="Validating", total=len(self.dataloader))
        self.model.eval()
        
        for batch in pbar:
            batch = self.preprocess(batch)
            with torch.no_grad():
                preds = self.model(batch["volume"])
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            
            # Visualization
            if self.args.save_visualizations:
                self.plot_predictions(batch, preds)
        
        stats = self.metrics.evaluate(iou_thres=self.args.iou_thres)
        LOGGER.info(f"mAP@{self.args.iou_thres:.2f}: {stats[0]:.4f}")
        return stats 

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}