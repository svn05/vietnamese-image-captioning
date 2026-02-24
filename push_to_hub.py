"""Push fine-tuned BLIP model to HuggingFace Hub.

Usage:
    python push_to_hub.py
    python push_to_hub.py --model-dir outputs/model --repo sanvo/vietnamese-image-captioning
"""

import argparse
import json
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import HfApi


def push_to_hub(model_dir="outputs/model", repo_id="sanvo/vietnamese-image-captioning"):
    """Push fine-tuned BLIP model to HuggingFace Hub."""
    print(f"Loading model from {model_dir}...")
    model = BlipForConditionalGeneration.from_pretrained(model_dir)
    processor = BlipProcessor.from_pretrained(model_dir)

    # Load training history for metrics
    history_path = os.path.join(model_dir, "training_history.json")
    metrics = ""
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        final = history[-1]
        best_val = min(h["val_loss"] for h in history)
        metrics = f"""
## Training Results

| Metric | Value |
|--------|-------|
| Final Train Loss | {final['train_loss']:.4f} |
| Best Val Loss | {best_val:.4f} |
| Epochs | {len(history)} |
"""

    model_card = f"""---
language:
  - en
  - vi
tags:
  - image-captioning
  - blip
  - vietnamese
  - vision-language
license: mit
datasets:
  - nlphuji/flickr30k
pipeline_tag: image-to-text
---

# Vietnamese Image Captioning (Fine-tuned BLIP)

Fine-tuned BLIP model for the Vietnamese image captioning pipeline.
The model generates English captions which are then translated to Vietnamese using NLLB-200.

## Pipeline Architecture

1. **BLIP** (this model) — generates English captions from images
2. **CLIP** — ranks multiple caption candidates by image-text similarity
3. **NLLB-200** — translates the best caption to Vietnamese

## Fine-tuning Details

- **Base Model**: Salesforce/blip-image-captioning-base
- **Dataset**: Flickr30k (2000 image-caption pairs)
- **Strategy**: Vision encoder frozen, text decoder fine-tuned
- **Device**: Apple MPS (Metal Performance Shaders)
{metrics}

## Usage

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("{repo_id}")
model = BlipForConditionalGeneration.from_pretrained("{repo_id}")

image = Image.open("example.jpg")
inputs = processor(images=image, return_tensors="pt")
output = model.generate(**inputs, max_length=50)
caption = processor.decode(output[0], skip_special_tokens=True)
print(caption)
```
"""

    print(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id, commit_message="Upload fine-tuned BLIP captioning model")
    processor.push_to_hub(repo_id, commit_message="Upload processor")

    # Upload model card
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )

    print(f"Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--repo", default="sanvo/vietnamese-image-captioning")
    args = parser.parse_args()
    push_to_hub(args.model_dir, args.repo)
