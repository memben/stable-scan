from PIL import Image
import webui_api


def generate_img(img: Image, depth: Image):
    img.show(title="Image")
    depth.show(title="Depth Image")
    response = input("Would you like to proceed? (Y/N): ").upper()
    if response == 'Y':
        prompt = input("Enter prompt: ")
        print(f"Generating {prompt}...")
    else:
        print("Cancelled")
        return
    result = webui_api.generate_img2img(prompt, img, depth)
    print("Done!")
    return result