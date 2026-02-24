"""Fine-tune BLIP for image captioning on Flickr30k.

Fine-tunes the text decoder of BLIP while keeping the vision encoder frozen,
improving caption quality for the Vietnamese captioning pipeline.

Usage:
    python train.py
    python train.py --epochs 3 --batch-size 4 --lr 5e-5
"""

import argparse
import os
import time
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class CaptionDataset(Dataset):
    """Dataset for image captioning fine-tuning."""

    def __init__(self, data, processor, max_length=64):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")
        caption = item["caption"]

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze batch dim
        input_ids = encoding["input_ids"].squeeze(0)
        pixel_values = encoding["pixel_values"].squeeze(0)
        attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)

        # Labels: same as input_ids, with padding tokens set to -100
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def download_flickr30k(max_samples=2000):
    """Download Flickr30k dataset from HuggingFace.

    Returns:
        List of dicts with 'image' and 'caption' keys.
    """
    from datasets import load_dataset

    print("Downloading Flickr30k dataset...")
    try:
        dataset = load_dataset(
            "nlphuji/flickr30k",
            split="test",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Flickr30k failed: {e}")
        print("Trying COCO captions instead...")
        dataset = load_dataset(
            "yerevann/coco-karpathy",
            split="train",
            trust_remote_code=True,
        )

    data = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        img = item.get("image")
        if img is None:
            continue

        # Get caption - different datasets have different formats
        caption = None
        if "caption" in item:
            cap = item["caption"]
            caption = cap[0] if isinstance(cap, list) else cap
        elif "sentences" in item:
            caption = item["sentences"][0] if item["sentences"] else None
        elif "text" in item:
            caption = item["text"]

        if caption and img:
            data.append({"image": img, "caption": caption})

    print(f"Loaded {len(data)} image-caption pairs")
    return data


def train(args):
    """Fine-tune BLIP on captioning data."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Use base model for fine-tuning (smaller, fits on MPS)
    model_name = "Salesforce/blip-image-captioning-base"
    print(f"Loading {model_name}...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # Freeze vision encoder, only train text decoder
    for param in model.vision_model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model = model.to(device)

    # Load data
    data = download_flickr30k(max_samples=args.max_samples)
    if not data:
        print("ERROR: No training data available!")
        return

    # Split train/val
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = CaptionDataset(train_data, processor)
    val_dataset = CaptionDataset(val_data, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Training loop
    best_val_loss = float("inf")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg = total_loss / (step + 1)
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)} | Loss: {avg:.4f} | {elapsed:.0f}s")

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {elapsed:.0f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP for image captioning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="outputs/model")
    args = parser.parse_args()

    train(args)
