# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

def bbox3d_iou(box1, box2):
    """
    Calculate 3D IoU between two bounding boxes.
    
    Args:
        box1 (torch.Tensor): (N, 6) First box (x,y,z,d,h,w)
        box2 (torch.Tensor): (N, 6) Second box (x,y,z,d,h,w)
        
    Returns:
        torch.Tensor: IoU values (N,)
    """
    # Convert to corners format
    b1_mins = box1[..., :3] - box1[..., 3:] / 2
    b1_maxs = box1[..., :3] + box1[..., 3:] / 2
    b2_mins = box2[..., :3] - box2[..., 3:] / 2
    b2_maxs = box2[..., :3] + box2[..., 3:] / 2
    
    # Intersection
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxs = torch.min(b1_maxs, b2_maxs)
    intersect_wdh = torch.clamp(intersect_maxs - intersect_mins, min=0)
    intersect = intersect_wdh[..., 0] * intersect_wdh[..., 1] * intersect_wdh[..., 2]
    
    # Union
    b1_wdh = box1[..., 3:].prod(dim=-1)
    b2_wdh = box2[..., 3:].prod(dim=-1)
    union = b1_wdh + b2_wdh - intersect
    
    return intersect / (union + 1e-7)

class Loss3D(nn.Module):
    """YOLOv8 3D Loss."""
    
    def __init__(self):
        """Initialize 3D loss module."""
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.box_weight = 7.5  # box loss gain
        self.cls_weight = 0.5  # cls loss gain
        
    def forward(self, preds, targets):
        """Calculate 3D detection loss."""
        box_loss = torch.zeros(1, device=targets.device)
        cls_loss = torch.zeros(1, device=targets.device)
        
        # Target boxes (xyzdwh), class
        tbox, tcls = targets[:, :6], targets[:, 6]
        
        # Predicted boxes
        pbox = preds[:, :6]
        pcls = preds[:, 6:]
        
        # Box loss (IoU loss)
        iou = bbox3d_iou(pbox, tbox)
        box_loss = (1.0 - iou).mean()
        
        # Classification loss
        if pcls.shape[1]:
            cls_loss = self.bce(pcls, tcls)
            
        loss = self.box_weight * box_loss + self.cls_weight * cls_loss
        return loss, torch.cat((box_loss, cls_loss)).detach() 