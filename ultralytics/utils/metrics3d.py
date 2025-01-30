# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np
from .loss3d import bbox3d_iou

class Metric3D:
    """3D Object Detection Metrics."""
    
    def __init__(self, nc=80):
        """Initialize 3D metrics module."""
        self.nc = nc
        self.stats = []  # list to store stats per image
        self.ap = np.zeros((nc, ))  # AP per class
        self.ap_class = []  # AP class indices
        
    def process(self, tp, conf, pred_cls, target_cls):
        """Process batch statistics for AP calculation."""
        self.stats.append((tp, conf, pred_cls, target_cls))
        
    def compute_ap(self, recall, precision):
        """Compute Average Precision using 11-point interpolation."""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        
        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        
        # 11-point interpolation
        x = np.linspace(0, 1, 11)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        
        return ap
        
    def evaluate(self, iou_thres=0.5):
        """
        Compute 3D mAP and other metrics.
        
        Args:
            iou_thres (float): IoU threshold for considering true positives
            
        Returns:
            tuple: (mAP, precision, recall, f1-score)
        """
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(stats) == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        tp, conf, pred_cls, target_cls = stats
        
        # Sort by confidence
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        
        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        
        # Compute metrics for each class
        ap = np.zeros((self.nc, ))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]
            n_p = i.sum()
            
            if n_p == 0 or n_l == 0:
                continue
                
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = tp[i].cumsum()
            
            # Recall
            recall = tpc / (n_l + 1e-16)
            
            # Precision
            precision = tpc / (tpc + fpc)
            
            # AP from recall-precision curve
            ap[ci] = self.compute_ap(recall, precision)
            
        self.ap = ap
        self.ap_class = unique_classes
        
        # Calculate metrics
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        mean_ap = ap.mean()
        
        return mean_ap, precision[-1], recall[-1], f1[-1] 