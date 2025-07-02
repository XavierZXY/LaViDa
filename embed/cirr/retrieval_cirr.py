import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.logging import RichHandler
from tqdm import tqdm

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


def cir_recall_at_k(scores, labels, k):
    """
    Compute recall at k for CIRR dataset
    :param scores: similarity scores between queries and candidates (nb_queries, nb_candidates)
    :param labels: target indices for each query
    :param k: number of candidates to consider
    :return: recall at k averaged over all queries
    """
    num_queries = scores.size(0)
    recalls = []
    for i in range(num_queries):
        top_k_indices = torch.topk(scores[i], k=k, largest=True).indices
        recalls.append(int(labels[i] in top_k_indices))
    return sum(recalls) / num_queries


def main(
    model_path: str = "jacklishufan/lavida-llada-v1.0-instruct",
    batch_size: int = 4,
    device: str = "cuda",
    num_samples: int = None,
    debug: bool = False,
):
    log.info("Initializing LaViDa embedding model...")
    embed_model = create_lavida_embed_model(
        model_path=model_path, device=device, torch_dtype="bfloat16"
    )

    log.info("Loading CIRR dataset...")
    dataset = load_dataset("royokong/cirr_val")
    img_dataset = load_dataset("royokong/cirr_imgs")
    dataset = dataset["val"]
    img_dataset = img_dataset["val"]

    if debug or num_samples is not None:
        sample_size = num_samples if num_samples is not None else 10
        dataset = dataset.select(range(sample_size))
        img_dataset = img_dataset.select(range(sample_size))

    log.info(f"Dataset size: {len(dataset)}")
    log.info(f"Image dataset size: {len(img_dataset)}")

    # Embed all candidate images first
    log.info("Embedding candidate images...")
    images = [item["img"] for item in img_dataset]
    images_embs = []
    for start in tqdm(range(0, len(images), batch_size)):
        end = start + batch_size
        batch_imgs = images[start:end]
        with torch.no_grad():
            embs = embed_model.embed_image(batch_imgs)
            embs = F.normalize(embs, dim=-1)
        images_embs.append(embs.cpu().float())
    images_emb = torch.stack(images_embs, dim=0)
    images_ids = img_dataset["id"]
    log.info(f"Image embeddings shape: {images_emb.shape}")

    # Embed text+image queries
    log.info("Embedding text+image queries...")
    retrieve_emb = []
    target_ids = dataset["target_id"]

    for i in tqdm(range(0, len(dataset), batch_size)):
        end = min(i + batch_size, len(dataset))
        batch = dataset.select(range(i, end))

        # Prepare text prompts
        texts = [
            f"Modify this image with '{caption}', describe the modified image in one sentence:"
            for caption in batch["caption"]
        ]

        # Get candidate images for this batch
        candidate_images = [item["candidate"] for item in batch]

        with torch.no_grad():
            # Use LaViDa's embed_fused method for text+image fusion
            fused_embs = embed_model.embed_fused(texts, candidate_images)
            fused_embs = F.normalize(fused_embs, dim=-1)

        retrieve_emb.append(fused_embs.cpu().float())

    retrieve_emb = torch.stack(retrieve_emb, dim=0)
    log.info(f"Query embeddings shape: {retrieve_emb.shape}")

    # Compute similarity scores
    scores = torch.matmul(retrieve_emb, images_emb.T)
    log.info(f"Similarity matrix shape: {scores.shape}")

    # Create labels mapping
    labels = []
    for target_id in target_ids:
        labels.append(images_ids.index(target_id))

    # Remove reference image from candidates (CIRR specific)
    if not debug:
        mask_indices = [
            images_ids.index(candidate_id)
            for candidate_id in dataset["candidate_id"]
        ]
        for i, mask_idx in enumerate(mask_indices):
            scores[i][mask_idx] = -1

    # Compute recall metrics
    r_at_1 = cir_recall_at_k(scores, labels, 1)
    r_at_3 = cir_recall_at_k(scores, labels, 3)
    r_at_5 = cir_recall_at_k(scores, labels, 5)
    r_at_10 = cir_recall_at_k(scores, labels, 10)

    metrics = {
        "recall@1": r_at_1,
        "recall@3": r_at_3,
        "recall@5": r_at_5,
        "recall@10": r_at_10,
    }

    log.info("CIRR Retrieval metrics:")
    for k, v in metrics.items():
        log.info(f"{k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LaViDa CIRR Retrieval")
    parser.add_argument(
        "--model_path",
        type=str,
        default="jacklishufan/lavida-llada-v1.0-instruct",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=True,
        help="Use only the first 10 samples for quick debugging.",
    )
    args = parser.parse_args()
    args.debug = False
    if args.debug:
        args.num_samples = 20

    main(
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        num_samples=args.num_samples,
        debug=args.debug,
    )
