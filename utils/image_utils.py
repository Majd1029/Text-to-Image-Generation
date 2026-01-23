import os
from datetime import datetime

OUTPUT_DIR = "outputs/generated_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_image(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"generated_{timestamp}.png")
    image.save(path)
    return path