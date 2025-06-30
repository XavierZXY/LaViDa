import torch
import pytest
from PIL import Image
import numpy as np
from typing import List
import logging
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")

from lavida_embed import LaViDaEmbedModel, create_lavida_embed_model


def create_dummy_image(
    height: int = 224, width: int = 224, channels: int = 3
) -> torch.Tensor:
    """Create a dummy image tensor for testing."""
    return torch.randn(channels, height, width)


def create_dummy_images(
    batch_size: int, height: int = 224, width: int = 224, channels: int = 3
) -> torch.Tensor:
    """Create a batch of dummy image tensors for testing."""
    return torch.randn(batch_size, channels, height, width)


class TestLaViDaEmbedModel:
    """Test class for LaViDa embedding model."""

    @pytest.fixture
    def model_path(self):
        """Model path fixture - replace with actual model path for testing."""
        # This should be replaced with an actual LaViDa model path
        return "path/to/your/lavida/model"

    @pytest.fixture
    def embed_model(self, model_path):
        """Create embedding model fixture."""
        try:
            model = create_lavida_embed_model(
                model_path=model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype="bfloat16",
            )
            return model
        except Exception as e:
            log.warning(f"Could not load model from {model_path}: {e}")
            log.warning("Skipping tests that require model loading")
            return None

    def test_model_initialization(self, embed_model):
        """Test model initialization."""
        if embed_model is None:
            pytest.skip("Model not available")

        assert embed_model is not None
        assert hasattr(embed_model, "hidden_size")
        assert hasattr(embed_model, "tokenizer")
        assert hasattr(embed_model, "model")
        assert hasattr(embed_model, "image_processor")

        log.info(
            f"Model initialized with hidden size: {embed_model.hidden_size}"
        )

    def test_text_embedding_single(self, embed_model):
        """Test single text embedding."""
        if embed_model is None:
            pytest.skip("Model not available")

        text = "This is a test sentence for embedding."

        embeddings = embed_model.embed_text(text)

        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(f"Single text embedding shape: {embeddings.shape}")

    def test_text_embedding_batch(self, embed_model):
        """Test batch text embedding."""
        if embed_model is None:
            pytest.skip("Model not available")

        texts = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ]

        embeddings = embed_model.embed_text(texts)

        assert embeddings.shape[0] == len(texts)  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(f"Batch text embedding shape: {embeddings.shape}")

    def test_image_embedding_single(self, embed_model):
        """Test single image embedding."""
        if embed_model is None:
            pytest.skip("Model not available")

        # Create dummy image
        image = create_dummy_image()

        embeddings = embed_model.embed_image(image)

        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(f"Single image embedding shape: {embeddings.shape}")

    def test_image_embedding_batch(self, embed_model):
        """Test batch image embedding."""
        if embed_model is None:
            pytest.skip("Model not available")

        # Create dummy images
        images = create_dummy_images(batch_size=3)

        embeddings = embed_model.embed_image(images)

        assert embeddings.shape[0] == 3  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(f"Batch image embedding shape: {embeddings.shape}")

    def test_text_image_fusion_single(self, embed_model):
        """Test single text-image fusion embedding."""
        if embed_model is None:
            pytest.skip("Model not available")

        text = "A beautiful sunset over the mountains."
        image = create_dummy_image()

        embeddings = embed_model.embed_text_and_image(text, image)

        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(
            f"Single text-image fusion embedding shape: {embeddings.shape}"
        )

    def test_text_image_fusion_batch(self, embed_model):
        """Test batch text-image fusion embedding."""
        if embed_model is None:
            pytest.skip("Model not available")

        texts = [
            "A cat sitting on a chair.",
            "A dog running in the park.",
            "A bird flying in the sky.",
        ]
        images = create_dummy_images(batch_size=3)

        embeddings = embed_model.embed_text_and_image(texts, images)

        assert embeddings.shape[0] == 3  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(f"Batch text-image fusion embedding shape: {embeddings.shape}")

    def test_forward_method_text_only(self, embed_model):
        """Test forward method with text only."""
        if embed_model is None:
            pytest.skip("Model not available")

        text = "Testing the forward method with text."

        embeddings = embed_model(text=text)

        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(
            f"Forward method text-only embedding shape: {embeddings.shape}"
        )

    def test_forward_method_image_only(self, embed_model):
        """Test forward method with image only."""
        if embed_model is None:
            pytest.skip("Model not available")

        image = create_dummy_image()

        embeddings = embed_model(images=image)

        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(
            f"Forward method image-only embedding shape: {embeddings.shape}"
        )

    def test_forward_method_text_image(self, embed_model):
        """Test forward method with text and image."""
        if embed_model is None:
            pytest.skip("Model not available")

        text = "Testing the forward method with text and image."
        image = create_dummy_image()

        embeddings = embed_model(text=text, images=image)

        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[1] == embed_model.hidden_size
        assert not torch.isnan(embeddings).any()

        log.info(
            f"Forward method text-image embedding shape: {embeddings.shape}"
        )

    def test_forward_method_no_inputs(self, embed_model):
        """Test forward method with no inputs (should raise error)."""
        if embed_model is None:
            pytest.skip("Model not available")

        with pytest.raises(ValueError, match="Either text or images"):
            embed_model()

    def test_embedding_dimension(self, embed_model):
        """Test getting embedding dimension."""
        if embed_model is None:
            pytest.skip("Model not available")

        dim = embed_model.get_embedding_dim()
        assert dim == embed_model.hidden_size
        assert isinstance(dim, int)
        assert dim > 0

        log.info(f"Embedding dimension: {dim}")


def demo_usage():
    """Demonstrate usage of the LaViDa embedding model."""
    log.info("=== LaViDa Embedding Model Demo ===")

    # Replace with your actual model path
    model_path = "path/to/your/lavida/model"

    try:
        # Create the embedding model
        embed_model = create_lavida_embed_model(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="bfloat16",
        )

        log.info(
            f"Model loaded successfully with hidden size: {embed_model.hidden_size}"
        )

        # Example 1: Text embedding
        text = "The quick brown fox jumps over the lazy dog."
        text_embeddings = embed_model.embed_text(text)
        log.info(f"Text embedding shape: {text_embeddings.shape}")

        # Example 2: Image embedding
        dummy_image = create_dummy_image()
        image_embeddings = embed_model.embed_image(dummy_image)
        log.info(f"Image embedding shape: {image_embeddings.shape}")

        # Example 3: Text + Image fusion
        fusion_embeddings = embed_model.embed_text_and_image(text, dummy_image)
        log.info(f"Fusion embedding shape: {fusion_embeddings.shape}")

        # Example 4: Using forward method
        forward_embeddings = embed_model(text=text, images=dummy_image)
        log.info(f"Forward method embedding shape: {forward_embeddings.shape}")

        # Example 5: Batch processing
        texts = ["First text", "Second text", "Third text"]
        images = create_dummy_images(batch_size=3)

        batch_text_embeddings = embed_model.embed_text(texts)
        batch_image_embeddings = embed_model.embed_image(images)
        batch_fusion_embeddings = embed_model.embed_text_and_image(
            texts, images
        )

        log.info(f"Batch text embeddings shape: {batch_text_embeddings.shape}")
        log.info(
            f"Batch image embeddings shape: {batch_image_embeddings.shape}"
        )
        log.info(
            f"Batch fusion embeddings shape: {batch_fusion_embeddings.shape}"
        )

        log.info("Demo completed successfully!")

    except Exception as e:
        log.error(f"Demo failed: {e}")
        log.info(
            "Please make sure to replace 'model_path' with an actual LaViDa model path"
        )


if __name__ == "__main__":
    # Run the demo
    demo_usage()
