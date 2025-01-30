# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plot3D:
    """3D visualization tools."""
    
    def __init__(self):
        """Initialize 3D plotting."""
        self.colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    def plot_3d_box(self, ax, box, color='r', label=None):
        """Plot a 3D bounding box."""
        x, y, z, d, h, w = box
        
        # Calculate vertices
        x_corners = [x-w/2, x+w/2, x+w/2, x-w/2, x-w/2, x+w/2, x+w/2, x-w/2]
        y_corners = [y-h/2, y-h/2, y+h/2, y+h/2, y-h/2, y-h/2, y+h/2, y+h/2]
        z_corners = [z-d/2, z-d/2, z-d/2, z-d/2, z+d/2, z+d/2, z+d/2, z+d/2]
        
        # Plot vertices
        ax.scatter(x_corners, y_corners, z_corners, c=color)
        
        # Plot edges
        for i in range(4):
            ax.plot([x_corners[i], x_corners[i+4]], 
                   [y_corners[i], y_corners[i+4]], 
                   [z_corners[i], z_corners[i+4]], c=color)
            ax.plot([x_corners[i], x_corners[(i+1)%4]], 
                   [y_corners[i], y_corners[(i+1)%4]], 
                   [z_corners[i], z_corners[(i+1)%4]], c=color)
            ax.plot([x_corners[i+4], x_corners[((i+1)%4)+4]], 
                   [y_corners[i+4], y_corners[((i+1)%4)+4]], 
                   [z_corners[i+4], z_corners[((i+1)%4)+4]], c=color)
        
        if label:
            ax.text(x, y, z, label)
    
    def plot_predictions(self, volume, boxes, labels=None, scores=None, save_path=None):
        """Plot volume and predicted boxes."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot volume
        x, y, z = np.where(volume > 0.5)  # threshold volume
        ax.scatter(x, y, z, c='gray', alpha=0.1, s=1)
        
        # Plot boxes
        for i, box in enumerate(boxes):
            color = self.colors[i % len(self.colors)]
            label = f"{labels[i]}: {scores[i]:.2f}" if labels is not None and scores is not None else None
            self.plot_3d_box(ax, box, color=color, label=label)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 