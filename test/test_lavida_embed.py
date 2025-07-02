import torch
import pytest
from PIL import Image
import numpy as np
from typing import List
import logging
from rich.logging import RichHandler
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from embed.lavida_embed import LaViDaEmbedModel, create_lavida_embed_model

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")


def create_dummy_pil_image(height: int = 224, width: int = 224) -> Image.Image:
    """Create a dummy PIL image for testing."""
    # Create a random numpy array and convert to PIL Image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def create_dummy_pil_images(
    batch_size: int, height: int = 224, width: int = 224
) -> List[Image.Image]:
    """Create a batch of dummy PIL images for testing."""
    return [create_dummy_pil_image(height, width) for _ in range(batch_size)]


class TestLaViDaSimilarity:
    """Test class for LaViDa similarity functionality."""

    @pytest.fixture
    def embed_model(self):
        """Create embedding model fixture."""
        try:
            model = create_lavida_embed_model(
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype="bfloat16",
            )
            log.info(
                f"Model loaded successfully with hidden size: {model.hidden_size}"
            )
            return model
        except Exception as e:
            log.warning(f"Could not load model: {e}")
            log.warning("Skipping tests that require model loading")
            return None

    def test_text_similarity(self, embed_model):
        """Test text similarity functionality."""
        if embed_model is None:
            pytest.skip("Model not available")

        log.info("=== Testing Text Similarity ===")

        # Test data
        texts1 = [
            "核电占发电量比例最大的是哪个国家?",
            "郑州是那个省的",
            "深圳合租记主题曲叫什么",
        ]
        texts2 = [
            "法国。在世界主要工业大国中，法国核电的比例最高，核电占国家总发电量的78%，位居世界第二，日本的核电比例为40%，德国为33%，韩国为30%，美国为22%",  # Same as first text
            "河南。郑州是河南省省会城市，周边有洛阳、开封、新郑、新密、许昌等城市",  # Similar to second text
            "从爱发落。电视剧《深圳合租记》的主题曲是罗志祥演唱的《从爱发落》，片尾曲是罗志祥演唱的《再见陌生人》，插曲是毛俊杰演唱的《怎样的女人》",  # Similar to third text
        ]

        # Test cosine similarity
        similarity_matrix = embed_model.text_similarity(
            texts1, texts2, "cosine"
        )

        assert similarity_matrix.shape == (3, 3)
        assert not torch.isnan(similarity_matrix).any()

        # Diagonal should have highest similarity for same texts
        # assert similarity_matrix[0, 0] > similarity_matrix[0, 1]
        # assert similarity_matrix[0, 0] > similarity_matrix[0, 2]

        # All cosine similarities should be in [-1, 1] range
        assert torch.all(similarity_matrix >= -1) and torch.all(
            similarity_matrix <= 1
        )

        log.info(f"Text similarity matrix shape: {similarity_matrix.shape}")
        log.info(f"Text similarity scores:\n{similarity_matrix}")
        log.info("Text similarity test passed!")

    def test_image_similarity(self, embed_model):
        """Test image similarity functionality."""
        if embed_model is None:
            pytest.skip("Model not available")

        log.info("=== Testing Image Similarity ===")

        # Create test images
        images1 = [Image.open("images/dog.png"), Image.open("images/cat.jpg")]
        images2 = [Image.open("images/cat.jpg"), Image.open("images/dog.png")]

        # Test cosine similarity
        similarity_matrix = embed_model.image_similarity(
            images1, images2, "cosine"
        )

        assert similarity_matrix.shape == (2, 2), (
            f"similarity_matrix shape: {similarity_matrix.shape}"
        )
        assert not torch.isnan(similarity_matrix).any(), (
            f"similarity_matrix: {similarity_matrix}"
        )

        # All cosine similarities should be in [-1, 1] range
        assert torch.all(similarity_matrix >= -1) and torch.all(
            similarity_matrix <= 1
        ), f"similarity_matrix: {similarity_matrix}"

        log.info(f"Image similarity matrix shape: {similarity_matrix.shape}")
        log.info(f"Image similarity scores:\n{similarity_matrix}")
        log.info("Image similarity test passed!")

    def test_text_image_mixed_similarity(self, embed_model):
        """Test text-image mixed similarity functionality."""
        if embed_model is None:
            pytest.skip("Model not available")

        log.info("=== Testing Text-Image Mixed Similarity ===")

        # Test data
        texts = [
            "A dog is running in the park.",
            "A bird is flying.",
            "A cat is sitting on a chair.",
        ]
        images = [
            Image.open("images/dog.jpg"),
            Image.open("images/bird.jpg"),
            Image.open("images/cat.jpg"),
        ]

        # Test text-image similarity
        text_image_similarity = embed_model.text_image_similarity(
            texts, images, "cosine"
        )

        assert text_image_similarity.shape == (3, 3), (
            f"text_image_similarity shape: {text_image_similarity.shape}"
        )
        assert not torch.isnan(text_image_similarity).any(), (
            f"text_image_similarity: {text_image_similarity}"
        )

        # All cosine similarities should be in [-1, 1] range
        assert torch.all(text_image_similarity >= -1) and torch.all(
            text_image_similarity <= 1
        ), f"text_image_similarity: {text_image_similarity}"

        log.info(
            f"Text-image similarity matrix shape: {text_image_similarity.shape}"
        )
        log.info(f"Text-image similarity scores:\n{text_image_similarity}")

        # # Test fused similarity (text-image pairs)
        # log.info("=== Testing Fused Similarity ===")
        # pairs1 = [
        #     ("A cat is sitting on a chair.", create_dummy_pil_image()),
        #     ("A dog is running in the park.", create_dummy_pil_image()),
        # ]
        # pairs2 = [
        #     (
        #         "A cat is sitting on a chair.",
        #         create_dummy_pil_image(),
        #     ),  # Similar to first pair
        #     ("A bird is flying in the sky.", create_dummy_pil_image()),
        # ]

        # fused_similarity = embed_model.fused_similarity(
        #     pairs1, pairs2, "cosine"
        # )

        # assert fused_similarity.shape == (2, 2)
        # assert not torch.isnan(fused_similarity).any()

        # # All cosine similarities should be in [-1, 1] range
        # assert torch.all(fused_similarity >= -1) and torch.all(
        #     fused_similarity <= 1
        # )

        # # First pair should be more similar to first pair in second set
        # assert fused_similarity[0, 0] > fused_similarity[0, 1]

        # log.info(f"Fused similarity matrix shape: {fused_similarity.shape}")
        # log.info(f"Fused similarity scores:\n{fused_similarity}")
        # log.info("Text-image mixed similarity test passed!")


def demo_similarity():
    """Demonstrate similarity functionality of the LaViDa embedding model."""
    log.info("=== LaViDa Similarity Demo ===")

    try:
        # Create the embedding model
        embed_model = create_lavida_embed_model(
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="bfloat16",
        )

        log.info(
            f"Model loaded successfully with hidden size: {embed_model.hidden_size}"
        )

        # Demo 1: Text Similarity
        log.info("\n=== Text Similarity Demo ===")
        texts1 = [
            "A cat is sitting on a chair.",
            "A dog is running in the park.",
        ]
        texts2 = [
            "A cat is sitting on a chair.",  # Same as first text
            "A bird is flying in the sky.",
        ]

        text_similarity = embed_model.text_similarity(texts1, texts2)
        log.info(f"Text similarity matrix shape: {text_similarity.shape}")
        log.info(f"Text similarity scores:\n{text_similarity}")

        # Demo 2: Image Similarity
        log.info("\n=== Image Similarity Demo ===")
        images1 = create_dummy_pil_images(2)
        images2 = create_dummy_pil_images(2)

        image_similarity = embed_model.image_similarity(images1, images2)
        log.info(f"Image similarity matrix shape: {image_similarity.shape}")
        log.info(f"Image similarity scores:\n{image_similarity}")

        # Demo 3: Text-Image Mixed Similarity
        log.info("\n=== Text-Image Mixed Similarity Demo ===")

        # Text-image similarity
        text_image_similarity = embed_model.text_image_similarity(
            texts1, images1
        )
        log.info(
            f"Text-image similarity matrix shape: {text_image_similarity.shape}"
        )
        log.info(f"Text-image similarity scores:\n{text_image_similarity}")

        # Fused similarity
        pairs1 = [
            ("A cat is sitting on a chair.", create_dummy_pil_image()),
            ("A dog is running in the park.", create_dummy_pil_image()),
        ]
        pairs2 = [
            ("A cat is sitting on a chair.", create_dummy_pil_image()),
            ("A bird is flying in the sky.", create_dummy_pil_image()),
        ]

        fused_similarity = embed_model.fused_similarity(pairs1, pairs2)
        log.info(f"Fused similarity matrix shape: {fused_similarity.shape}")
        log.info(f"Fused similarity scores:\n{fused_similarity}")

        log.info("\nDemo completed successfully!")

    except Exception as e:
        log.error(f"Demo failed: {e}")


if __name__ == "__main__":
    # Run the demo
    demo_similarity()
