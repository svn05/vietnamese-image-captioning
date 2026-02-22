# Vietnamese Multi-Modal Image Captioning

An end-to-end vision-language pipeline generating **Vietnamese image captions** using **BLIP** for caption generation, **CLIP** for ranking, and **NLLB** for English-to-Vietnamese translation. Evaluated with **BLEU/METEOR** metrics. Deployed as an interactive **Gradio** demo.

## Pipeline

```
Input Image → BLIP (generate N English captions)
           → CLIP (rank by image-text similarity)
           → NLLB (translate best caption to Vietnamese)
           → Vietnamese Caption
```

## Features

- **BLIP** generates diverse captions via conditional + unconditional captioning
- **CLIP** ranks candidates by visual-semantic alignment (cosine similarity)
- **NLLB-200** translates the highest-scoring caption to Vietnamese
- **BLEU/METEOR** evaluation on reference captions
- **Gradio** interactive demo: upload image, get Vietnamese caption

## Setup

```bash
git clone https://github.com/svn05/vietnamese-image-captioning.git
cd vietnamese-image-captioning
pip install -r requirements.txt
```

## Usage

### Full pipeline
```bash
python pipeline.py --image path/to/image.jpg --verbose
python pipeline.py --image path/to/image.jpg --num-captions 10
```

### Individual stages
```bash
# Generate English captions
python generate_captions.py --image photo.jpg --num-captions 5

# Rank captions with CLIP
python rank_captions.py --image photo.jpg --captions "a dog" "a cat on a mat"

# Translate to Vietnamese
python translate.py --text "A dog running on the beach"
```

### Evaluate with BLEU/METEOR
```bash
python evaluate.py
python evaluate.py --predictions results.json --references refs.json
```

### Run Gradio demo
```bash
python app.py
```

### Generate sample data
```bash
python data/prepare_data.py
```

## Project Structure

```
vietnamese-image-captioning/
├── pipeline.py            # End-to-end captioning pipeline
├── generate_captions.py   # BLIP caption generation
├── rank_captions.py       # CLIP-based caption ranking
├── translate.py           # NLLB English→Vietnamese translation
├── evaluate.py            # BLEU/METEOR evaluation
├── app.py                 # Gradio interactive demo
├── data/
│   └── prepare_data.py    # Sample data generation
├── examples/              # Sample images (generated)
├── requirements.txt
└── README.md
```

## Models Used

| Stage | Model | Purpose |
|-------|-------|---------|
| Caption Generation | [BLIP-large](https://huggingface.co/Salesforce/blip-image-captioning-large) | Generate English image captions |
| Caption Ranking | [CLIP-ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | Rank captions by visual similarity |
| Translation | [NLLB-200-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | Translate English → Vietnamese |

## Tech Stack

- **BLIP** (Salesforce) — Vision-language captioning model
- **CLIP** (OpenAI) — Image-text contrastive model
- **NLLB-200** (Meta) — Multilingual neural machine translation
- **HuggingFace Transformers** — Model loading and inference
- **NLTK** — BLEU/METEOR metric computation
- **Gradio** — Interactive web demo
