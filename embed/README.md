# LaViDa Embedding Model

A LaViDa-based embedding model that extracts the last layer hidden states and averages them to create embeddings for text, image, and text+image inputs.

## Features

- **Text Embedding**: Generate embeddings for text inputs
- **Image Embedding**: Generate embeddings for image inputs  
- **Text+Image Fusion**: Generate fused embeddings for text and image combinations
- **Batch Processing**: Support for batch processing of multiple inputs
- **Flexible Input**: Support for single inputs or lists of inputs
- **Attention Masking**: Proper handling of padding tokens using attention masks

## Architecture

The embedding model is based on the LaViDa (Language-Vision Diffusion) model architecture:

1. **Model Loading**: Loads a pre-trained LaViDa model using the LLaVA model builder
2. **Hidden State Extraction**: Extracts the last layer hidden states from the transformer
3. **Averaging**: Averages the hidden states across the sequence dimension to create fixed-size embeddings
4. **Attention Masking**: Uses attention masks to properly handle padding tokens during averaging

## Installation

Make sure you have the required dependencies installed:

```bash
pip install torch transformers pillow numpy rich pytest
```

## Usage

### Basic Usage

```python
from embed.lavida_embed import create_lavida_embed_model

# Create the embedding model
embed_model = create_lavida_embed_model(
    model_path="path/to/your/lavida/model",
    device="cuda",  # or "cpu"
    torch_dtype="bfloat16"
)

# Text embedding
text = "A beautiful sunset over the ocean."
text_embeddings = embed_model.embed_text(text)

# Image embedding
image_tensor = torch.randn(3, 224, 224)  # (channels, height, width)
image_embeddings = embed_model.embed_image(image_tensor)

# Text + Image fusion
fusion_embeddings = embed_model.embed_text_and_image(text, image_tensor)
```

### Using the Forward Method

The model also supports a flexible forward method that automatically determines the embedding type:

```python
# Text only
text_emb = embed_model(text="Some text")

# Image only  
image_emb = embed_model(images=image_tensor)

# Text + Image
fusion_emb = embed_model(text="Some text", images=image_tensor)
```

### Batch Processing

```python
# Batch text embeddings
texts = ["First text", "Second text", "Third text"]
batch_text_embeddings = embed_model.embed_text(texts)

# Batch image embeddings
batch_images = torch.stack([image1, image2, image3])
batch_image_embeddings = embed_model.embed_image(batch_images)

# Batch fusion embeddings
batch_fusion_embeddings = embed_model.embed_text_and_image(texts, batch_images)
```

## API Reference

### LaViDaEmbedModel

#### `__init__(model_path, device="cuda", torch_dtype="bfloat16", **kwargs)`

Initialize the LaViDa embedding model.

**Parameters:**
- `model_path` (str): Path to the LaViDa model
- `device` (str): Device to load the model on ("cuda" or "cpu")
- `torch_dtype` (str): Torch data type ("bfloat16", "float16", etc.)
- `**kwargs`: Additional arguments passed to the model loader

#### `embed_text(text, max_length=None)`

Generate embeddings for text input.

**Parameters:**
- `text` (str or List[str]): Input text or list of texts
- `max_length` (int, optional): Maximum sequence length

**Returns:**
- `torch.Tensor`: Text embeddings of shape (batch_size, hidden_size)

#### `embed_image(images)`

Generate embeddings for image input.

**Parameters:**
- `images` (torch.Tensor or List[torch.Tensor]): Input images of shape (batch_size, channels, height, width) or list of images

**Returns:**
- `torch.Tensor`: Image embeddings of shape (batch_size, hidden_size)

#### `embed_text_and_image(text, images, max_length=None)`

Generate fused embeddings for text and image input.

**Parameters:**
- `text` (str or List[str]): Input text or list of texts
- `images` (torch.Tensor or List[torch.Tensor]): Input images
- `max_length` (int, optional): Maximum sequence length

**Returns:**
- `torch.Tensor`: Fused text-image embeddings of shape (batch_size, hidden_size)

#### `forward(text=None, images=None, max_length=None)`

Forward pass that automatically determines the embedding type based on inputs.

**Parameters:**
- `text` (str or List[str], optional): Text input
- `images` (torch.Tensor or List[torch.Tensor], optional): Image input
- `max_length` (int, optional): Maximum sequence length for text

**Returns:**
- `torch.Tensor`: Embeddings of shape (batch_size, hidden_size)

#### `get_embedding_dim()`

Get the embedding dimension.

**Returns:**
- `int`: The embedding dimension (hidden size of the model)

### Factory Function

#### `create_lavida_embed_model(model_path, device="cuda", torch_dtype="bfloat16", **kwargs)`

Factory function to create a LaViDa embedding model.

**Parameters:**
- `model_path` (str): Path to the LaViDa model
- `device` (str): Device to load the model on
- `torch_dtype` (str): Torch data type
- `**kwargs`: Additional arguments passed to the model

**Returns:**
- `LaViDaEmbedModel`: LaViDaEmbedModel instance

## Examples

### Simple Example

```python
from embed.lavida_embed import create_lavida_embed_model
import torch

# Create model
embed_model = create_lavida_embed_model("path/to/model")

# Generate embeddings
text = "Hello, world!"
text_emb = embed_model.embed_text(text)
print(f"Text embedding shape: {text_emb.shape}")
```

### Advanced Example

```python
from embed.lavida_embed import create_lavida_embed_model
import torch
from PIL import Image
import torchvision.transforms as transforms

# Create model
embed_model = create_lavida_embed_model("path/to/model")

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("image.jpg")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Generate embeddings
text = "A cat sitting on a chair"
text_emb = embed_model.embed_text(text)
image_emb = embed_model.embed_image(image_tensor)
fusion_emb = embed_model.embed_text_and_image(text, image_tensor)

# Calculate similarities
similarity = torch.cosine_similarity(text_emb, fusion_emb, dim=1)
print(f"Text-Fusion similarity: {similarity.item():.4f}")
```

## Testing

Run the tests using pytest:

```bash
cd embed
pytest test_lavida_embed.py -v
```

## Demo

Run the example script:

```bash
cd embed
python example_usage.py
```

Make sure to update the `model_path` variable in the script with your actual LaViDa model path.

## Notes

- The model requires a pre-trained LaViDa model checkpoint
- Images should be in the format (channels, height, width) with values in [0, 1]
- The embedding dimension depends on the LaViDa model's hidden size
- The model automatically handles padding and attention masking
- All embeddings are normalized by averaging across the sequence dimension

## Requirements

- PyTorch
- Transformers
- Pillow
- NumPy
- Rich (for logging)
- Pytest (for testing)

## License

This code follows the same license as the LaViDa model. 