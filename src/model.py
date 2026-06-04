"""Deprecated: use app.py instead.

Example:
    python app.py --model tooth --image path/to/xray.jpg
    python app.py --model pano --image path/to/pano.jpg --backend local
"""

from src.inference_http import run_http_inference
from src.inference_local import run_local_inference

__all__ = ["run_http_inference", "run_local_inference"]
