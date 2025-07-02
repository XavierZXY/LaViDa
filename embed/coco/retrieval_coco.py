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
    """
    Compute the recall at k for each sample
    :param scores: compatibility score between text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_images
    )
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(
        dim=(1, 2)
    )
    # compute recall at k
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
    debug: bool = False,
):
    log.info("Initializing LaViDa embedding model...")
    embed_model = create_lavida_embed_model(
        model_path=model_path, device=device, torch_dtype="bfloat16"
    )

    log.info("Loading COCO test dataset...")
    dataset = load_dataset("royokong/coco_test", split="test")
    dataset = dataset.rename_column("text", "caption")
    dataset = dataset.rename_column("image", "img")

    # For COCO dataset, each image has multiple captions (5 captions per image)
    dataset = dataset.map(lambda x: {"caption": x["caption"][:5]}, num_proc=4)

    if debug:
        dataset = dataset.select(range(10))
    elif num_samples is not None:
        dataset = dataset.select(range(num_samples))

    log.info(f"COCO dataset loaded with {len(dataset)} images")
    log.info(f"Sample caption structure: {dataset[0]['caption']}")

    # Embed images first
    log.info("Start embedding images...")
    images = [item["img"] for item in dataset]
    images_embs = []
    for start in tqdm(range(0, len(images), batch_size)):
        end = start + batch_size
        batch_imgs = images[start:end]
        with torch.no_grad():
            embs = embed_model.embed_image(batch_imgs)
            embs = F.normalize(embs, dim=-1)
            assert embs.isnan().sum() == 0, "nan in emb after norm"
        images_embs.append(embs.cpu().float())
    images_emb = torch.stack(images_embs)[: len(dataset)]
    log.info(f"Image embeddings shape: {images_emb.shape}")

    # log.info("Cleared GPU memory after image embeddings.")

    # Process texts - for COCO dataset, each image has multiple captions
    log.info("Start embedding texts...")
    text_embs = []
    for start in tqdm(range(0, len(dataset), batch_size)):
        end = start + batch_size
        batch = dataset[start:end]

        # Flatten the captions list for processing
        flat_captions = []
        for captions in batch["caption"]:
            flat_captions.extend(captions)

        with torch.no_grad():
            embs = embed_model.embed_text(flat_captions)
            embs = F.normalize(embs, dim=-1)
        text_embs.append(embs.cpu().float())
    text_emb = torch.cat(text_embs)
    log.info(f"Text embeddings shape: {text_emb.shape}")

    log.info("Cleared GPU memory after text embeddings.")

    assert text_emb.isnan().sum().item() == 0, "nan in retrieve emb"
    assert images_emb.isnan().sum().item() == 0, "nan in images emb"

    total_captions = sum(len(item["caption"]) for item in dataset)
    log.info(f"Total captions in COCO dataset: {total_captions}")
    log.info(f"Expected text_emb shape: {total_captions}")

    # Get the score for each text and image pair
    scores = text_emb @ images_emb.t()
    log.info(f"Scores shape: {scores.shape}")

    # Create positive pairs matrix
    # For COCO dataset, each image has multiple captions (5 captions per image)
    # Calculate how many captions we have in total
    total_captions = sum(len(item["caption"]) for item in dataset)
    # Create mapping from caption index to image index
    caption_to_image = []
    for i, item in enumerate(dataset):
        for _ in range(len(item["caption"])):
            caption_to_image.append(i)

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    for i, img_idx in enumerate(caption_to_image):
        positive_pairs[i, img_idx] = True

    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size = 64
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    recall_at_k,
                    scores,
                    positive_pairs,
                    batch_size,
                    k=recall_k,
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )
        metrics[f"text_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    recall_at_k,
                    scores.T,
                    positive_pairs.T,
                    batch_size,
                    k=recall_k,
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

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LaViDa COCO Retrieval")
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
    args.batch_size = 1
    if args.debug:
        args.num_samples = 10

    main(
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        num_samples=args.num_samples,
        debug=args.debug,
    )
