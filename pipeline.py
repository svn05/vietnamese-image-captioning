"""End-to-end Vietnamese image captioning pipeline.

Orchestrates the full pipeline: BLIP caption generation → CLIP ranking
→ NLLB translation to Vietnamese.

Usage:
    python pipeline.py --image path/to/image.jpg
    python pipeline.py --image path/to/image.jpg --num-captions 10
"""

import argparse
import time
from PIL import Image

from generate_captions import load_blip_model, generate_captions
from rank_captions import load_clip_model, rank_captions
from translate import load_nllb_model, translate_to_vietnamese


class VietnameseCaptioningPipeline:
    """End-to-end pipeline for generating Vietnamese image captions.

    Pipeline stages:
    1. BLIP generates multiple English captions
    2. CLIP ranks captions by image-text similarity
    3. NLLB translates the best caption to Vietnamese
    """

    def __init__(self):
        print("Loading models...")
        start = time.time()

        print("  Loading BLIP (caption generation)...")
        self.blip_model, self.blip_processor, self.device = load_blip_model()

        print("  Loading CLIP (caption ranking)...")
        self.clip_model, self.clip_processor, _ = load_clip_model()
        self.clip_model = self.clip_model.to(self.device)

        print("  Loading NLLB (translation)...")
        self.nllb_model, self.nllb_tokenizer, _ = load_nllb_model()
        self.nllb_model = self.nllb_model.to(self.device)

        elapsed = time.time() - start
        print(f"Models loaded in {elapsed:.1f}s\n")

    def caption(self, image, num_captions=5, return_all=False):
        """Generate Vietnamese caption for an image.

        Args:
            image: PIL Image object.
            num_captions: Number of candidate captions to generate.
            return_all: If True, return all stages' outputs.

        Returns:
            Vietnamese caption string (or dict with all outputs if return_all=True).
        """
        # Stage 1: Generate English captions with BLIP
        english_captions = generate_captions(
            image, self.blip_model, self.blip_processor, self.device,
            num_captions=num_captions,
        )

        # Stage 2: Rank captions with CLIP
        ranked = rank_captions(
            image, english_captions,
            self.clip_model, self.clip_processor, self.device,
        )

        best_english = ranked[0][0] if ranked else english_captions[0]
        best_score = ranked[0][1] if ranked else 0.0

        # Stage 3: Translate best caption to Vietnamese with NLLB
        vietnamese_caption = translate_to_vietnamese(
            best_english, self.nllb_model, self.nllb_tokenizer, self.device,
        )

        if return_all:
            return {
                "english_captions": english_captions,
                "ranked_captions": ranked,
                "best_english": best_english,
                "clip_score": float(best_score),
                "vietnamese_caption": vietnamese_caption,
            }

        return vietnamese_caption


def main():
    parser = argparse.ArgumentParser(description="Vietnamese image captioning pipeline")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--num-captions", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pipeline = VietnameseCaptioningPipeline()
    image = Image.open(args.image).convert("RGB")

    result = pipeline.caption(image, num_captions=args.num_captions, return_all=True)

    print(f"Image: {args.image}")

    if args.verbose:
        print(f"\nGenerated English captions:")
        for i, cap in enumerate(result["english_captions"], 1):
            print(f"  {i}. {cap}")

        print(f"\nCLIP-ranked captions:")
        for i, (cap, score) in enumerate(result["ranked_captions"], 1):
            print(f"  {i}. [{score:.4f}] {cap}")

    print(f"\nBest English: {result['best_english']} (CLIP: {result['clip_score']:.4f})")
    print(f"Vietnamese:   {result['vietnamese_caption']}")


if __name__ == "__main__":
    main()
