"""Translate English captions to Vietnamese using NLLB.

Uses Meta's NLLB-200 model for high-quality English-to-Vietnamese
translation as the final stage of the captioning pipeline.

Usage:
    python translate.py --text "A dog running on the beach"
"""

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_nllb_model(model_name="facebook/nllb-200-distilled-600M"):
    """Load NLLB translation model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        model, tokenizer, device tuple.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def translate_to_vietnamese(
    text, model, tokenizer, device,
    src_lang="eng_Latn", tgt_lang="vie_Latn",
    max_length=128, beam_size=5,
):
    """Translate text from English to Vietnamese.

    Args:
        text: English text to translate.
        model: NLLB model.
        tokenizer: NLLB tokenizer.
        device: torch device.
        src_lang: Source language NLLB code.
        tgt_lang: Target language NLLB code.
        max_length: Maximum output length.
        beam_size: Beam search width.

    Returns:
        Vietnamese translation string.
    """
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            num_beams=beam_size,
            max_length=max_length,
            early_stopping=True,
        )

    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translation.strip()


def translate_captions(captions, model, tokenizer, device):
    """Translate a list of English captions to Vietnamese.

    Args:
        captions: List of English caption strings.
        model: NLLB model.
        tokenizer: NLLB tokenizer.
        device: torch device.

    Returns:
        List of Vietnamese translations.
    """
    translations = []
    for caption in captions:
        vi_caption = translate_to_vietnamese(caption, model, tokenizer, device)
        translations.append(vi_caption)
    return translations


def main():
    parser = argparse.ArgumentParser(description="Translate captions to Vietnamese")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer, device = load_nllb_model()
    translation = translate_to_vietnamese(args.text, model, tokenizer, device)

    print(f"English:    {args.text}")
    print(f"Vietnamese: {translation}")


if __name__ == "__main__":
    main()
