"""Rank captions using CLIP image-text similarity.

Uses OpenAI's CLIP model to rank generated captions by their
visual-semantic alignment with the input image.

Usage:
    python rank_captions.py --image path/to/image.jpg --captions "a dog" "a cat" "a bird"
"""

import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """Load CLIP model for image-text matching.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        model, processor, device tuple.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor, device


def rank_captions(image, captions, model, processor, device):
    """Rank captions by CLIP similarity to image.

    Computes cosine similarity between the image embedding and
    each caption's text embedding to find the best match.

    Args:
        image: PIL Image object.
        captions: List of caption strings.
        model: CLIP model.
        processor: CLIP processor.
        device: torch device.

    Returns:
        List of (caption, score) tuples sorted by score descending.
    """
    if not captions:
        return []

    inputs = processor(
        text=captions,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Normalized image and text embeddings
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarities = (image_embeds @ text_embeds.T).squeeze(0)

    ranked = sorted(
        zip(captions, similarities.cpu().numpy()),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked


def select_best_caption(image, captions, model, processor, device):
    """Select the highest-scoring caption.

    Args:
        image: PIL Image.
        captions: List of candidate captions.
        model: CLIP model.
        processor: CLIP processor.
        device: torch device.

    Returns:
        Best caption string and its score.
    """
    ranked = rank_captions(image, captions, model, processor, device)
    if ranked:
        return ranked[0]
    return ("", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Rank captions with CLIP")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--captions", nargs="+", required=True)
    args = parser.parse_args()

    model, processor, device = load_clip_model()
    image = Image.open(args.image).convert("RGB")

    ranked = rank_captions(image, args.captions, model, processor, device)

    print(f"\nImage: {args.image}")
    print("Ranked captions:")
    for i, (caption, score) in enumerate(ranked, 1):
        bar = "█" * int(score * 30)
        print(f"  {i}. [{score:.4f}] {caption} {bar}")


if __name__ == "__main__":
    main()
