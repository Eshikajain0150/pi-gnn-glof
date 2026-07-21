"""Physics-informed multimodal graph learning for GLOF hazard forecasting."""

from .config import load_config
from .models.pignn import PIGNN

__all__ = ["PIGNN", "load_config"]
__version__ = "0.1.0"
