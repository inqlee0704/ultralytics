# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
import torch.nn as nn

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel, DetectionModel3D, yaml_model_load
from ultralytics.utils import ROOT, yaml_load, LOGGER
from ultralytics.engine.predictor3d import BasePredictor3D
# from ultralytics.engine.trainer3d import BaseTrainer3D


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes


class YOLO3D(Model):
    """YOLO3D object detection model."""

    def __init__(self, model='yolo11-3d.yaml', task='detect3d', verbose=True):
        """Initialize YOLO3D model."""
        # Set initial task
        self.task = task
        
        # Initialize parent class with our settings
        super().__init__(model=model, task=self.task, verbose=verbose)
        # Set initial overrides
        self.overrides = {
            'model': model,
            'task': self.task,
            'mode': 'train',
            'imgsz': 640,
            'data': None,
            'device': '',
            'verbose': verbose,
            'cfg': model if str(model).endswith('.yaml') else None,
        }
        
        
        # Merge any additional overrides from parent class
        self.overrides.update(getattr(self.model, 'args', {}))

    def _load(self, weights: str, task: str = None):
        """Load a YOLO3D model."""
        suffix = Path(weights).suffix
        if suffix == '.yaml':
            self._new(weights, task or self.task)
        else:
            super()._load(weights, task or self.task)

    def _new(self, cfg: str, task=None):
        """Create a new YOLO3D model from YAML file."""
        self.task = task or self.task or 'detect3d'
        cfg_dict = yaml_model_load(cfg)
        
        if self.task not in self.task_map:
            raise ValueError(f"Task '{self.task}' not found in task map. Available tasks: {list(self.task_map.keys())}")
            
        self.model = self.task_map[self.task]['model'](cfg_dict, verbose=self.overrides.get('verbose', True))
        self.model.task = self.task
        self.model.names = cfg_dict.get('names', {i: f'class{i}' for i in range(cfg_dict.get('nc', 80))})
        self.model.transforms = None
        
        # Update overrides with model configuration
        if hasattr(self.model, 'args'):
            self.overrides.update(self.model.args)
            
        return self

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        from ultralytics.engine.trainer3d import BaseTrainer3D
        from ultralytics.engine.validator3d import BaseValidator3D
        return {
            'detect3d': {
                'model': DetectionModel3D,
                'trainer': BaseTrainer3D,
                'validator': BaseValidator3D,
                'predictor': BasePredictor3D,
            }
        }

    def _check_is_pytorch_model(self):
        """Check if the model is a PyTorch model."""
        return isinstance(self.model, (DetectionModel3D, nn.Module))
