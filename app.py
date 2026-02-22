"""Gradio interactive demo for Vietnamese image captioning.

Upload an image to get a Vietnamese caption generated through the
BLIP → CLIP → NLLB pipeline.

Usage:
    python app.py
"""

import gradio as gr
from PIL import Image

from generate_captions import load_blip_model, generate_captions
from rank_captions import load_clip_model, rank_captions
from translate import load_nllb_model, translate_to_vietnamese


# Load all models at startup
print("Loading models for Vietnamese Image Captioning...")
blip_model, blip_processor, device = load_blip_model()
clip_model, clip_processor, _ = load_clip_model()
clip_model = clip_model.to(device)
nllb_model, nllb_tokenizer, _ = load_nllb_model()
nllb_model = nllb_model.to(device)
print("All models loaded.\n")


def caption_image(image, num_captions=5):
    """Generate Vietnamese caption for uploaded image.

    Args:
        image: PIL Image from Gradio.
        num_captions: Number of candidate captions.

    Returns:
        Vietnamese caption, English caption, ranked captions details.
    """
    if image is None:
        return "No image provided.", "", ""

    image = image.convert("RGB")

    # Stage 1: BLIP generates English captions
    english_captions = generate_captions(
        image, blip_model, blip_processor, device,
        num_captions=num_captions,
    )

    # Stage 2: CLIP ranks captions
    ranked = rank_captions(image, english_captions, clip_model, clip_processor, device)
    best_english = ranked[0][0] if ranked else english_captions[0]

    # Stage 3: NLLB translates to Vietnamese
    vietnamese = translate_to_vietnamese(best_english, nllb_model, nllb_tokenizer, device)

    # Format ranked captions for display
    ranked_text = "\n".join(
        f"{i+1}. [{score:.3f}] {cap}"
        for i, (cap, score) in enumerate(ranked)
    )

    return vietnamese, best_english, ranked_text


demo = gr.Interface(
    fn=caption_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of Candidate Captions"),
    ],
    outputs=[
        gr.Textbox(label="Vietnamese Caption (Chú thích tiếng Việt)"),
        gr.Textbox(label="Best English Caption"),
        gr.Textbox(label="All Ranked Captions (CLIP scores)", lines=6),
    ],
    title="Vietnamese Multi-Modal Image Captioning",
    description=(
        "Upload an image to generate a Vietnamese caption.\n\n"
        "**Pipeline:** BLIP (caption generation) → CLIP (ranking) → NLLB (translation)\n\n"
        "1. **BLIP** generates multiple English captions\n"
        "2. **CLIP** ranks them by image-text similarity\n"
        "3. **NLLB** translates the best caption to Vietnamese"
    ),
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
