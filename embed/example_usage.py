#!/usr/bin/env python3
"""
Example usage of LaViDa embedding model.

This script demonstrates how to use the LaViDa embedding model to generate
embeddings for text, images, and text+image combinations.
"""

import torch
import logging
from rich.logging import RichHandler
from PIL import Image
import numpy as np
from rich import traceback

traceback.install(show_locals=False)
# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")

from lavida_embed import create_lavida_embed_model


def create_sample_image(width=224, height=224):
    """Create a sample image for demonstration."""
    # Create a simple gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [
                int(255 * i / height),  # Red gradient
                int(255 * j / width),  # Green gradient
                128,  # Blue constant
            ]

    # Convert to PIL Image
    pil_image = Image.fromarray(img_array)

    # Convert to tensor (C, H, W) format
    img_tensor = (
        torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
    )

    return img_tensor


def main():
    """Main function demonstrating LaViDa embedding usage."""
    log.info("=== LaViDa Embedding Model Example ===")

    # Configuration
    model_path = "jacklishufan/lavida-llada-v1.0-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Using device: {device}")
    log.info(f"Model path: {model_path}")

    # Create the embedding model
    log.info("Loading LaViDa embedding model...")
    embed_model = create_lavida_embed_model(
        model_path=model_path, device=device, torch_dtype="bfloat16"
    )

    log.info("Model loaded successfully!")
    log.info(f"Embedding dimension: {embed_model.get_embedding_dim()}")

    # Example 1: Text embedding
    log.info("\n--- Example 1: Text Embedding ---")
    text_examples = [
        "A beautiful sunset over the ocean.",
        "A cat sitting on a windowsill.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for i, text in enumerate(text_examples):
        log.info(f"Text {i + 1}: '{text}'")
        embeddings = embed_model.embed_text(text)
        log.info(f"  Embedding shape: {embeddings.shape}")
        log.info(f"  Embedding norm: {torch.norm(embeddings).item():.4f}")

    # Example 2: Image embedding
    log.info("\n--- Example 2: Image Embedding ---")
    # sample_image = create_sample_image()
    sample_image = Image.open("images/dog.png")
    # log.info(f"Sample image shape: {sample_image.shape}")

    image_embeddings = embed_model.embed_image(sample_image)
    log.info(f"Image embedding shape: {image_embeddings.shape}")
    log.info(f"Image embedding norm: {torch.norm(image_embeddings).item():.4f}")

    # Example 3: Text + Image fusion
    log.info("\n--- Example 3: Text + Image Fusion ---")
    text = "A colorful gradient image with red and green patterns."

    fusion_embeddings = embed_model.embed_fused(text, sample_image)
    log.info(f"Text: '{text}'")
    log.info(f"Fusion embedding shape: {fusion_embeddings.shape}")
    log.info(
        f"Fusion embedding norm: {torch.norm(fusion_embeddings).item():.4f}"
    )

    # Example 4: Batch processing
    log.info("\n--- Example 4: Batch Processing ---")
    batch_texts = [
        "First example text for batch processing.",
        "Second example text for batch processing.",
        "Third example text for batch processing.",
    ]

    batch_images = [Image.open("images/dog.png") for _ in range(3)]

    # Batch text embeddings
    batch_text_embeddings = embed_model.embed_text(batch_texts)
    log.info(f"Batch text embeddings shape: {batch_text_embeddings.shape}")

    # Batch image embeddings
    batch_image_embeddings = embed_model.embed_image(batch_images)
    log.info(f"Batch image embeddings shape: {batch_image_embeddings.shape}")

    # Batch fusion embeddings
    batch_fusion_embeddings = embed_model.embed_fused(batch_texts, batch_images)
    log.info(f"Batch fusion embeddings shape: {batch_fusion_embeddings.shape}")

    # Example 5: Using forward method
    log.info("\n--- Example 5: Using Forward Method ---")

    # Text only
    text_forward = embed_model(text="Using the forward method with text only.")
    log.info(f"Forward text embedding shape: {text_forward.shape}")

    # Image only
    image_forward = embed_model(images=sample_image)
    log.info(f"Forward image embedding shape: {image_forward.shape}")

    # Text + Image
    fusion_forward = embed_model(
        text="Using forward method with text and image.",
        images=sample_image,
    )
    log.info(f"Forward fusion embedding shape: {fusion_forward.shape}")

    # Example 6: Similarity comparison
    log.info("\n--- Example 6: Similarity Comparison ---")

    # Create embeddings for similar and different content
    similar_text1 = "A red car driving on the highway."
    similar_text2 = "A red automobile traveling on the road."
    different_text = "A blue bird flying in the sky."

    similar_emb1 = embed_model.embed_text(similar_text1)
    similar_emb2 = embed_model.embed_text(similar_text2)
    different_emb = embed_model.embed_text(different_text)

    # Calculate cosine similarities
    def cosine_similarity(a, b):
        return torch.dot(a.flatten(), b.flatten()) / (
            torch.norm(a) * torch.norm(b)
        )

    sim_similar = cosine_similarity(similar_emb1, similar_emb2).item()
    sim_different1 = cosine_similarity(similar_emb1, different_emb).item()
    sim_different2 = cosine_similarity(similar_emb2, different_emb).item()

    log.info(f"Similarity between similar texts: {sim_similar:.4f}")
    log.info(
        f"Similarity between similar and different text 1: {sim_different1:.4f}"
    )
    log.info(
        f"Similarity between similar and different text 2: {sim_different2:.4f}"
    )

    log.info("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
