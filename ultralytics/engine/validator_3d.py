# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .validator import BaseValidator
import torch
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.utils import LOGGER

class BaseValidator3D(BaseValidator):
    """
    Base class for 3D validation tasks extending BaseValidator.
    
    Implements 3D-specific validation methods and metrics while maintaining compatibility
    with the original validation pipeline.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize 3D validator with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "detect3d"
        self.nt_per_class = None
        self.nt_per_volume = None
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def preprocess(self, batch):
        """Preprocesses batch of volumes."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for 3D detection."""
        val = self.data.get(self.args.split, "")  # validation path
        self.class_map = list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and not self.training
        self.names = model.names
        self.nc = len(model.names)
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_vol=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics."""
        return ("%22s" + "%11s" * 6) % ("Class", "Volumes", "Instances", "Box(P", "R", "mAP50", "mAP50-95")

    def _prepare_batch(self, si, batch):
        """Prepares a batch of volumes and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        if len(cls):
            bbox = bbox * torch.tensor(imgsz, device=self.device)[[2, 1, 0, 2, 1, 0]]
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz}

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_volume = np.bincount(stats["target_vol"].astype(int), minlength=self.nc)
        stats.pop("target_vol", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """Prints validation results."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_volume[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

    def plot_val_samples(self, batch, ni):
        """Plot validation batch if args.plots."""
        pass  # Implement in derived class

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input volumes if args.plots."""
        pass  # Implement in derived class 