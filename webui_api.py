import base64
import io

import requests
from PIL import Image

url = "http://127.0.0.1:7860"


def encode_image_to_base64(image):
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="PNG")
    image_byte_array = image_byte_array.getvalue()
    encoded_image = base64.b64encode(image_byte_array).decode("utf-8")
    return encoded_image


def img2img_payload(prompt: str, img: Image, depth: Image):
    encoded_image = encode_image_to_base64(img)
    encoded_depth = encode_image_to_base64(depth)
    payload = {
        "prompt": prompt,
        "init_images": [encoded_image],
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "batch_size": 1,
        "steps": 20,
        "cfg_scale": 7,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": encoded_depth,
                        "model": "control_v11f1p_sd15_depth [cfd03158]",
                    }
                ]
            }
        },
    }
    return payload


def generate_img2img(prompt: str, img: Image, depth: Image):
    """Generates an image from a prompt, rgb image, and depth image
    using StableDiffusion 1.5 and ControlNet 1.1."""
    payload = img2img_payload(prompt, img, depth)
    response = requests.post(url=f"{url}/sdapi/v1/img2img", json=payload)
    r = response.json()
    result = r["images"][0]
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    return image
