import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import logging
from rich.logging import RichHandler
import pandas as pd
from pathlib import Path
import torch
from datasets import load_dataset
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from embed.lavida_embed import create_lavida_embed_model

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


def build_index(outputs, embedding_model, batch_size=1):
    """
    Build embeddings index for all outputs for efficient similarity search.
    Returns: (embeddings, output_texts)
    """
    all_embeddings = []

    for i in tqdm(
        range(0, len(outputs), batch_size), desc="Building embeddings index"
    ):
        batch = outputs[i : i + batch_size]
        emb = embedding_model.embed_text(batch)
        all_embeddings.append(emb.cpu())
    return torch.stack(all_embeddings, dim=0)


def batch_find_best_matches(
    input_texts, output_embeddings, output_texts, embedding_model, batch_size=1
):
    """
    Compute all input embeddings in batch, then cosine similarity with output embeddings.
    Returns: list of (best_output, score) for each input.
    """
    import torch

    all_input_embeds = []
    for i in tqdm(
        range(0, len(input_texts), batch_size), desc="Encoding input embeddings"
    ):
        batch = input_texts[i : i + batch_size]
        emb = embedding_model.embed_text(batch)
        all_input_embeds.append(emb.cpu())
    input_embeddings = torch.stack(all_input_embeds, dim=0)  # (N, D)
    # Normalize
    input_embeddings = torch.nn.functional.normalize(input_embeddings, dim=-1)
    output_embs = torch.nn.functional.normalize(output_embeddings, dim=-1)
    # Cosine similarity matrix (N, M)
    scores = torch.matmul(input_embeddings, output_embs.t())
    log.info(f"scores: {scores}")
    best_indices = scores.argmax(dim=1)
    best_scores = scores.max(dim=1).values
    results = [
        (output_texts[idx], score.item())
        for idx, score in zip(best_indices, best_scores)
    ]
    return results


def evaluate_accuracy(results_csv_path, outputs):
    results_df = pd.read_csv(results_csv_path)
    correct = 0
    total = len(results_df)
    for i, row in results_df.iterrows():
        if row["best_output"] == outputs[i]:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    log.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(
        description="LaViDa WebQA Retrieval Evaluation"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Only evaluate existing CSV results",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="webqa_lavida_results.csv",
        help="CSV file to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (test/train/validation)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="jacklishufan/lavida-llada-v1.0-instruct",
        help="LaViDa model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype to use",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=True,
        help="Debug mode: only use first 20 samples",
    )
    args = parser.parse_args()

    # Load WebQA dataset from HuggingFace
    log.info(
        f"Loading WebQA dataset from HuggingFace (suolyer/webqa), split={args.split}..."
    )
    dataset = load_dataset("suolyer/webqa", split=args.split)
    inputs = dataset["input"]
    outputs = dataset["output"]

    # Debug mode: only use first 20 samples
    args.debug = False
    if args.debug:
        log.info("Debug mode enabled: using only first 20 samples")
        inputs = inputs[:20]
        outputs = outputs[:20]

    if args.eval_only:
        log.info(f"Evaluating only, using CSV: {args.csv}")
        evaluate_accuracy(args.csv, outputs)
        return

    # Otherwise, run full pipeline
    log.info(f"Loading LaViDa model from {args.model_path}...")
    embedding_model = create_lavida_embed_model(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    log.info("Building output embeddings index...")
    output_embeddings = build_index(outputs, embedding_model)
    log.info(f"Output embeddings shape: {output_embeddings.shape}")

    # Compute all input embeddings and match in batch
    batch_results = batch_find_best_matches(
        inputs, output_embeddings, outputs, embedding_model
    )

    results = []
    for input_text, (best_output, score) in zip(inputs, batch_results):
        results.append(
            {"input": input_text, "best_output": best_output, "score": score}
        )
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.csv, index=False)
    log.info(f"Results saved to {args.csv}")
    evaluate_accuracy(args.csv, outputs)


if __name__ == "__main__":
    main()
