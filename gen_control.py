from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

import webui_api


@dataclass
class SDParams:
    """StableDiffusion parameters"""

    prompt: str
    init_images: List[str]
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    batch_size: int = 1
    steps: int = 20
    cfg_scale: int = 7
    controlnet: Optional[Dict] = None


class GenControl:
    """Controls the generation of images with StableDiffusion"""
