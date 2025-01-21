# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from pathlib import Path
from ultralytics.utils.plotting3d import Plot3D

class BasePredictor3D:
    """Base predictor for 3D object detection."""
    
    def __init__(self, model=None, args=None):
        """Initialize predictor with model and arguments."""
        self.model = model
        self.args = args
        self.plot = Plot3D()
        
    def preprocess(self, volume):
        """Preprocess input volume."""
        # Normalize
        volume = volume.float() / 255.0
        
        # Add batch dimension
        if len(volume.shape) == 3:
            volume = volume.unsqueeze(0)
            
        return volume.to(self.device)
    
    def postprocess(self, preds):
        """Postprocess predictions."""
        return self.non_max_suppression_3d(
            preds,
            conf_thres=self.args.conf_thres,
            iou_thres=self.args.iou_thres,
            max_det=self.args.max_det
        )
    
    def predict(self, source):
        """Run inference on input source."""
        self.model.eval()
        results = []
        
        # Load source
        if isinstance(source, (str, Path)):
            source = torch.load(source)
        
        # Preprocess
        volume = self.preprocess(source)
        
        # Inference
        with torch.no_grad():
            preds = self.model(volume)
        
        # Postprocess
        preds = self.postprocess(preds)
        
        # Visualize if needed
        if self.args.visualize:
            self.plot.plot_predictions(
                volume[0],
                preds[0],
                save_path=self.args.save_dir / "pred.png"
            )
        
        return preds 