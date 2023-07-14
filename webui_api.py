import base64
import io

import requests
from PIL import Image

from gen_control import SDParams


def encode_image_to_base64(image):
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="PNG")
    image_byte_array = image_byte_array.getvalue()
    encoded_image = base64.b64encode(image_byte_array).decode("utf-8")
    return encoded_image


def txt2img_payload(params: SDParams):
    payload = {
        "prompt": params.prompt,
        "width": params.width,
        "height": params.height,
        "steps": params.steps,
        "cfg_scale": params.cfg_scale,
        # "sampler_name": "UniPC",
    }

    return payload


def img2img_payload(params: SDParams):
    assert params.init_image is not None
    _txt2img_payload = txt2img_payload(params)
    encoded_image = encode_image_to_base64(params.init_image)
    payload = {
        **_txt2img_payload,
        "init_images": [encoded_image],
        # "do_not_save_samples": False,
    }

    return payload


def inject_controlnet_payload(payload: str, params: SDParams):
    assert params.controlnet is not None

    args = []
    if params.controlnet["depth"] is not None:
        encoded_depth = encode_image_to_base64(params.controlnet["depth"])
        args.append(
            {
                "input_image": encoded_depth,
                "model": "control_v11f1p_sd15_depth [cfd03158]",
            }
        )

    if args:
        payload["alwayson_scripts"] = {"controlnet": {"args": args}}

    return payload


def _decode_response(response):
    r = response.json()
    result = r["images"][0]
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    return image


def generate_txt2img(url, payload):
    response = requests.post(url=f"{url}/sdapi/v1/txt2img", json=payload)
    return _decode_response(response)


def generate_img2img(url, payload):
    print(url)
    # save payload to a a .json file
    with open("payload.json", "w") as f:
        f.write(str(payload))
    response = requests.post(url=f"{url}/sdapi/v1/img2img", json=payload)
    return _decode_response(response)


# Uuse python3 webui_api.py > payload.json && sed -i '' "s/'/\"/g" payload.json for debugging
if __name__ == "__main__":
    params = SDParams(
        prompt="A painting of a cat",
        init_image=Image.open("cat.png"),
        width=512,
        height=512,
        steps=20,
        cfg_scale=7,
        controlnet={"depth": Image.open("depth.png")},
    )
    payload = img2img_payload(params)
    inject_controlnet_payload(payload, params)
    print(payload)
