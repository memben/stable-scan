from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image


@dataclass
class SDParams:
    """StableDiffusion parameters"""

    prompt: str
    init_image: Image = None
    mask: np.ndarray = None  # set to 1 to keep pixels, 0 to discard
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: int = 7
    controlnet: Optional[Dict] = None


def generate(webui_url: str, params: SDParams) -> Image:
    """Generate an image using the webui api, uses init_image if provided"""
    import webui_api

    use_img2img = params.init_image is not None
    payload = (
        webui_api.img2img_payload(params)
        if use_img2img
        else webui_api.txt2img_payload(params)
    )
    if params.controlnet is not None:
        payload = webui_api.inject_controlnet_payload(payload, params)

    img = None
    if use_img2img:
        img = webui_api.generate_img2img(webui_url, payload)
    else:
        img = webui_api.generate_txt2img(webui_url, payload)

    return img
