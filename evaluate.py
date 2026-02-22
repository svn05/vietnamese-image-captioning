"""Evaluate captioning quality with BLEU and METEOR metrics.

Usage:
    python evaluate.py --predictions results.json --references references.json
"""

import argparse
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


# Download NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


def compute_bleu(references, hypotheses, n=4):
    """Compute corpus-level BLEU score.

    Args:
        references: List of reference caption lists (each ref is a list of tokens).
        hypotheses: List of hypothesis captions (each is a list of tokens).
        n: Maximum n-gram order.

    Returns:
        BLEU score (0 to 1).
    """
    # Tokenize if strings
    ref_tokens = []
    hyp_tokens = []

    for ref_list, hyp in zip(references, hypotheses):
        if isinstance(ref_list, str):
            ref_list = [ref_list]
        ref_tokens.append([r.split() if isinstance(r, str) else r for r in ref_list])
        hyp_tokens.append(hyp.split() if isinstance(hyp, str) else hyp)

    smoother = SmoothingFunction().method1
    weights = tuple([1.0 / n] * n)
    score = corpus_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=smoother)
    return score


def compute_meteor(references, hypotheses):
    """Compute average METEOR score.

    Args:
        references: List of reference strings.
        hypotheses: List of hypothesis strings.

    Returns:
        Average METEOR score (0 to 1).
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if isinstance(ref, list):
            ref = ref[0]
        ref_tokens = ref.split() if isinstance(ref, str) else ref
        hyp_tokens = hyp.split() if isinstance(hyp, str) else hyp
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_captions(predictions_file=None, references_file=None):
    """Evaluate caption quality.

    Args:
        predictions_file: JSON file with predicted captions.
        references_file: JSON file with reference captions.

    Returns:
        Dict with BLEU and METEOR scores.
    """
    # Sample evaluation data if no files provided
    if predictions_file is None:
        print("No prediction files provided. Running sample evaluation...\n")

        # Simulated Vietnamese captions
        references = [
            ["một chú chó đang chạy trên bãi biển"],
            ["phong cảnh núi non với hồ nước xanh"],
            ["một nhóm người đang ngồi trong quán cà phê"],
            ["con mèo nằm ngủ trên ghế sofa"],
            ["thành phố về đêm với ánh đèn rực rỡ"],
        ]

        hypotheses = [
            "một chú chó chạy trên bãi biển",
            "phong cảnh núi với hồ nước",
            "nhóm người ngồi trong quán cà phê",
            "con mèo đang ngủ trên sofa",
            "thành phố vào ban đêm với đèn sáng",
        ]
    else:
        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        with open(references_file, "r", encoding="utf-8") as f:
            references_data = json.load(f)

        hypotheses = [p["caption"] for p in predictions]
        references = [[r["caption"]] for r in references_data]

    # Compute metrics
    bleu1 = compute_bleu(references, hypotheses, n=1)
    bleu2 = compute_bleu(references, hypotheses, n=2)
    bleu3 = compute_bleu(references, hypotheses, n=3)
    bleu4 = compute_bleu(references, hypotheses, n=4)
    meteor = compute_meteor(
        [r[0] if isinstance(r, list) else r for r in references],
        hypotheses,
    )

    results = {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "METEOR": meteor,
    }

    print("=" * 50)
    print("VIETNAMESE IMAGE CAPTIONING — EVALUATION")
    print("=" * 50)
    print(f"Samples: {len(hypotheses)}")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate captioning with BLEU/METEOR")
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--references", type=str, default=None)
    args = parser.parse_args()

    evaluate_captions(args.predictions, args.references)
