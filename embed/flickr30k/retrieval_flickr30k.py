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


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_images
    )
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(
        dim=(1, 2)
    )
    recall_at_k = nb_true_positive / nb_positive
    return recall_at_k


def batchify(func, X, Y, batch_size, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end]
        y = Y[start:end]
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def main(
    model_path: str = "jacklishufan/lavida-llada-v1.0-instruct",
    batch_size: int = 4,
    device: str = "cuda",
    num_samples: int = None,
):
    log.info("Initializing LaViDa embedding model...")
    embed_model = create_lavida_embed_model(
        model_path=model_path, device=device, torch_dtype="bfloat16"
    )

    log.info("Loading flickr30k test dataset...")
    dataset = load_dataset("royokong/flickr30k_test", split="test")
    dataset = dataset.rename_column("text", "caption")
    dataset = dataset.rename_column("image", "img")
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))
    dataset = dataset.map(lambda x: {"caption": " ".join(x["caption"])})

    # Embed images
    log.info("Embedding images...")
    images = [item["img"] for item in dataset]
    images_embs = []
    for start in tqdm(range(0, len(images), batch_size)):
        end = start + batch_size
        batch_imgs = images[start:end]
        with torch.no_grad():
            embs = embed_model.embed_image(batch_imgs)
            embs = F.normalize(embs, dim=-1)
        images_embs.append(embs.cpu().float())
    images_emb = torch.stack(images_embs, dim=0)
    log.info(f"Image embeddings shape: {images_emb.shape}")

    # Embed texts
    log.info("Embedding texts...")
    texts = [item["caption"] for item in dataset]
    texts_embs = []
    for start in tqdm(range(0, len(texts), batch_size)):
        end = start + batch_size
        batch_texts = texts[start:end]
        with torch.no_grad():
            embs = embed_model.embed_text(batch_texts)
            embs = F.normalize(embs, dim=-1)
        texts_embs.append(embs.cpu().float())
    texts_emb = torch.stack(texts_embs, dim=0)
    log.info(f"Text embeddings shape: {texts_emb.shape}")

    if texts_emb.ndim == 1:
        texts_emb = texts_emb.unsqueeze(0)
    if images_emb.ndim == 1:
        images_emb = images_emb.unsqueeze(0)

    # Compute similarity scores (cosine similarity)
    scores = torch.matmul(texts_emb, images_emb.T)
    log.info(f"Similarity matrix shape: {scores.shape}")

    # Create positive pairs matrix (diagonal is True)
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), torch.arange(len(scores))] = True

    metrics = {}
    recall_k_list = [1, 5, 10]
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (
            (batchify(recall_at_k, scores, positive_pairs, 64, k=recall_k) > 0)
            .float()
            .mean()
            .item()
        )
        metrics[f"text_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    recall_at_k, scores.T, positive_pairs.T, 64, k=recall_k
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )

    log.info("Retrieval metrics:")
    for k, v in metrics.items():
        log.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LaViDa Flickr30k Retrieval")
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
        default=False,
        help="Use only the first 20 samples for quick debugging.",
    )
    args = parser.parse_args()

    args.debug = False
    args.batch_size = 1
    if args.debug:
        args.num_samples = 20
    main(
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        num_samples=args.num_samples,
    )
