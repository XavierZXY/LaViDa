import logging
import pytest
from rich.logging import RichHandler
from datasets import load_dataset
import torch
import torch.nn.functional as F
from PIL import Image
from embed.lavida_embed import create_lavida_embed_model

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


@pytest.mark.parametrize("num_samples", [100])
def test_flickr30k_image_text_similarity_lavida(num_samples):
    """
    Test the similarity between the first N images and their corresponding texts in the flickr30k dataset using the LaViDa embedding model.
    """
    model_path = "jacklishufan/lavida-llada-v1.0-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Initializing LaViDa embedding model...")
    embed_model = create_lavida_embed_model(
        model_path=model_path, device=device, torch_dtype="bfloat16"
    )

    log.info(
        f"Loading flickr30k test dataset and selecting first {num_samples} samples..."
    )
    dataset = load_dataset("royokong/flickr30k_test", split="test")
    dataset = dataset.rename_column("text", "caption")
    dataset = dataset.rename_column("image", "img")
    dataset = dataset.select(range(num_samples))
    dataset = dataset.map(lambda x: {"caption": " ".join(x["caption"])})

    images_embs = []
    texts_embs = []
    log.info("Embedding images and texts...")
    for i in range(num_samples):
        img = dataset[i]["img"]
        text = dataset[i]["caption"]
        # log.info(f"text: {text}")
        # img is a PIL Image
        img_emb = embed_model.embed_image(img)
        img_emb = F.normalize(img_emb, dim=-1)
        images_embs.append(img_emb.cpu().float())
        text_emb = embed_model.embed_text([text])
        text_emb = F.normalize(text_emb, dim=-1)
        texts_embs.append(text_emb.cpu().float())
        # log.info(f"Embedded sample {i + 1}/{num_samples}")

    images_emb = torch.stack(images_embs, dim=0)
    texts_emb = torch.stack(texts_embs, dim=0)

    log.info(f"Image embeddings shape: {images_emb.shape}")
    log.info(f"Text embeddings shape: {texts_emb.shape}")

    # Compute similarity scores (cosine similarity)
    similarity_scores = torch.matmul(texts_emb, images_emb.T)
    log.info(f"Similarity matrix shape: {similarity_scores.shape}")
    log.info(f"similarity_scores: {similarity_scores}")

    # Check that diagonal (corresponding pairs) are the highest in their row
    correct_matches = 0
    for i in range(num_samples):
        row = similarity_scores[i]
        max_idx = torch.argmax(row).item()
        is_correct = max_idx == i
        correct_matches += int(is_correct)
        log.info(
            f"Sample {i}: Max index {max_idx}, Correct: {is_correct}, Score: {row[i]:.4f}"
        )

    accuracy = correct_matches / num_samples
    log.info(f"Top-1 accuracy for {num_samples} samples: {accuracy:.2%}")
    assert accuracy > 0.5, f"Top-1 accuracy {accuracy:.2%} is too low!"
