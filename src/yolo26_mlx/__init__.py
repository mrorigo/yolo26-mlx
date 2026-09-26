"""Ground-up YOLO26 for Apple MLX. No PyTorch."""

from .config import load_config, scale_config
from .tasks import DetectionModel, build_model

__all__ = ["DetectionModel", "__version__", "build_model", "load_config", "scale_config"]

__version__ = "0.1.0"
