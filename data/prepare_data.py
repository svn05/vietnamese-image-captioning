"""Prepare image captioning evaluation data.

Downloads sample images and creates reference captions for evaluation.

Usage:
    python data/prepare_data.py
"""

import os
import json
import numpy as np
from PIL import Image


DATA_DIR = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.join(os.path.dirname(DATA_DIR), "examples")


def generate_sample_images(output_dir=None, n_images=10):
    """Generate sample images for testing the pipeline.

    Creates simple synthetic images with known content for evaluation.

    Args:
        output_dir: Where to save images.
        n_images: Number of sample images.
    """
    if output_dir is None:
        output_dir = EXAMPLES_DIR
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    scenes = [
        {"name": "beach", "colors": [(30, 120, 220), (200, 180, 120)], "ref_vi": "bãi biển với cát và nước xanh"},
        {"name": "forest", "colors": [(34, 100, 34), (0, 128, 0)], "ref_vi": "rừng cây xanh tươi tốt"},
        {"name": "city", "colors": [(128, 128, 128), (80, 80, 100)], "ref_vi": "phong cảnh thành phố"},
        {"name": "sunset", "colors": [(255, 100, 50), (255, 200, 100)], "ref_vi": "hoàng hôn trên biển"},
        {"name": "mountain", "colors": [(100, 100, 140), (200, 200, 220)], "ref_vi": "núi non hùng vĩ"},
    ]

    references = []

    for i in range(min(n_images, len(scenes))):
        scene = scenes[i % len(scenes)]
        img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Gradient background
        for y in range(224):
            ratio = y / 224
            c1 = np.array(scene["colors"][0])
            c2 = np.array(scene["colors"][1])
            img[y, :] = (c1 * (1 - ratio) + c2 * ratio).astype(np.uint8)

        # Add noise
        img = np.clip(img.astype(np.int16) + np.random.randint(-10, 10, img.shape), 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img)
        img_path = os.path.join(output_dir, f"sample_{scene['name']}.jpg")
        pil_img.save(img_path)

        references.append({
            "image": img_path,
            "caption": scene["ref_vi"],
        })

    # Save references
    ref_path = os.path.join(DATA_DIR, "references.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(references)} sample images in {output_dir}")
    print(f"References saved to {ref_path}")


if __name__ == "__main__":
    generate_sample_images()
