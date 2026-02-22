"""Generate image captions using BLIP.

Uses Salesforce's BLIP model to generate multiple English captions
for input images as the first stage of the captioning pipeline.

Usage:
    python generate_captions.py --image path/to/image.jpg
"""

import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_blip_model(model_name="Salesforce/blip-image-captioning-large"):
    """Load BLIP captioning model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        model, processor, device tuple.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor, device


def generate_captions(
    image, model, processor, device,
    num_captions=5, max_length=50,
    num_beams=5, temperature=1.0,
):
    """Generate multiple captions for an image.

    Uses diverse beam search to produce varied captions.

    Args:
        image: PIL Image object.
        model: BLIP model.
        processor: BLIP processor.
        device: torch device.
        num_captions: Number of captions to generate.
        max_length: Maximum caption length in tokens.
        num_beams: Number of beams for beam search.
        temperature: Sampling temperature.

    Returns:
        List of caption strings.
    """
    # Conditional captioning (with text prompt)
    prompts = [
        "a photograph of",
        "this image shows",
        "in this picture",
        "a scene depicting",
        "the image contains",
    ]

    captions = set()

    # Unconditional captioning
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=max(num_beams, num_captions),
            num_return_sequences=min(num_captions, num_beams),
            early_stopping=True,
        )
    for ids in output_ids:
        caption = processor.decode(ids, skip_special_tokens=True).strip()
        captions.add(caption)

    # Conditional captioning with different prompts
    for prompt in prompts[:num_captions]:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        captions.add(caption)

        if len(captions) >= num_captions:
            break

    return list(captions)[:num_captions]


def main():
    parser = argparse.ArgumentParser(description="Generate image captions with BLIP")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--num-captions", type=int, default=5)
    args = parser.parse_args()

    model, processor, device = load_blip_model()
    image = Image.open(args.image).convert("RGB")

    captions = generate_captions(
        image, model, processor, device,
        num_captions=args.num_captions,
    )

    print(f"\nImage: {args.image}")
    print(f"Generated {len(captions)} captions:")
    for i, cap in enumerate(captions, 1):
        print(f"  {i}. {cap}")


if __name__ == "__main__":
    main()
