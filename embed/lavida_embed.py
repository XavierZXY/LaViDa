import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
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

from transformers import AutoTokenizer, AutoConfig
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llada import LlavaLladaForMaskedDiffusion


class LaViDaEmbedModel(nn.Module):
    """
    LaViDa-based embedding model that extracts the last layer hidden states
    and averages them to create embeddings for text, image, and text+image inputs.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.torch_dtype = torch_dtype
        self.model_path = model_path

        # Load the LaViDa model
        log.info(f"Loading LaViDa model from {model_path}")
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name="llava_llada",
                device_map=device,
                torch_dtype=torch_dtype,
                **kwargs,
            )
        )

        # Set model to evaluation mode
        self.model.eval()

        # Get the hidden size from the model
        self.hidden_size = self.model.config.hidden_size

        # Debug: Check model structure
        log.info(f"Model type: {type(self.model)}")
        log.info(f"Model has 'model' attribute: {hasattr(self.model, 'model')}")
        if hasattr(self.model, "model"):
            log.info(f"Model.model type: {type(self.model.model)}")
            log.info(
                f"Model.model has 'transformer' attribute: {hasattr(self.model.model, 'transformer')}"
            )
            if hasattr(self.model.model, "transformer"):
                transformer = self.model.model.transformer
                log.info(
                    f"Transformer embedding dimension: {transformer.wte.embedding_dim}"
                )
                log.info(
                    f"Transformer hidden size: {getattr(transformer, 'd_model', 'unknown')}"
                )
        log.info(
            f"Model has 'prepare_inputs_labels_for_multimodal' method: {hasattr(self.model, 'prepare_inputs_labels_for_multimodal')}"
        )
        log.info(
            f"Model has 'get_vision_tower' method: {hasattr(self.model, 'get_vision_tower')}"
        )
        if hasattr(self.model, "get_vision_tower"):
            vision_tower = self.model.get_vision_tower()
            log.info(
                f"Vision tower hidden size: {getattr(vision_tower, 'hidden_size', 'unknown')}"
            )
        log.info(
            f"Model has 'mm_projector' attribute: {hasattr(self.model, 'mm_projector')}"
        )

        log.info(
            f"LaViDa embedding model loaded with hidden size: {self.hidden_size}"
        )

    def _extract_last_layer_hidden_states(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        modalities: Optional[List[str]] = ["image"],
    ) -> torch.Tensor:
        """
        Extract the last layer hidden states from the LaViDa model.

        Args:
            input_ids: Input token IDs
            inputs_embeds: Pre-computed input embeddings
            images: Input images
            attention_mask: Attention mask
            image_sizes: Image sizes for unpad processing
            modalities: List of modalities (e.g., ["image"])

        Returns:
            Last layer hidden states of shape (batch_size, seq_len, hidden_size)
        """
        with torch.no_grad():
            log.info(
                f"Extracting hidden states - input_ids: {input_ids is not None}, images: {images is not None}"
            )

            # For LaViDa models, we need to access the internal model structure
            # to get the hidden states before the final projection
            if hasattr(self.model, "model") and hasattr(
                self.model.model, "transformer"
            ):
                transformer = self.model.model.transformer
                log.info("Using direct transformer access method")

                # Handle multimodal input (text + image)
                if input_ids is not None and images is not None:
                    log.info("Processing multimodal input (text + image)")
                    # Use the model's multimodal preparation method
                    if hasattr(
                        self.model, "prepare_inputs_labels_for_multimodal"
                    ):
                        # Create dummy labels for the multimodal preparation
                        batch_size = input_ids.shape[0]
                        seq_len = input_ids.shape[1]
                        dummy_labels = torch.full(
                            (batch_size, seq_len),
                            -100,
                            dtype=torch.long,
                            device=input_ids.device,
                        )

                        # Prepare multimodal inputs
                        prepared_inputs = (
                            self.model.prepare_inputs_labels_for_multimodal(
                                input_ids=input_ids,
                                position_ids=None,
                                attention_mask=attention_mask,
                                past_key_values=None,
                                labels=dummy_labels,
                                images=images,
                                modalities=modalities,
                                image_sizes=image_sizes,
                                return_inputs=True,
                            )
                        )

                        # Extract the prepared inputs
                        (
                            input_ids,
                            position_ids,
                            attention_mask,
                            past_key_values,
                            inputs_embeds,
                            labels,
                            new_input_ids,
                        ) = prepared_inputs

                        # Use the prepared input embeddings
                        x = inputs_embeds
                        log.info(
                            f"Prepared multimodal embeddings shape: {x.shape}"
                        )
                    else:
                        # Fallback: process text and image separately and concatenate
                        log.warning(
                            "Multimodal preparation not available, using fallback method"
                        )
                        x = transformer.wte(input_ids)
                else:
                    # Get input embeddings for text-only or image-only
                    if inputs_embeds is not None:
                        x = inputs_embeds
                        log.info(
                            f"Using provided input embeddings, shape: {x.shape}"
                        )
                    elif input_ids is not None:
                        x = transformer.wte(input_ids)
                        log.info(f"Using token embeddings, shape: {x.shape}")
                    elif images is not None:
                        log.info("Processing image-only input")
                        # For image-only input, we need to process images through the vision tower
                        # and then provide the embeddings directly to the model
                        if hasattr(self.model, "get_vision_tower"):
                            vision_tower = self.model.get_vision_tower()
                            image_features = vision_tower(images)
                            log.info(
                                f"Vision tower output shape: {image_features.shape}"
                            )

                            # Check if we need to project the image features
                            if hasattr(self.model, "mm_projector"):
                                image_features = self.model.mm_projector(
                                    image_features
                                )
                                log.info(
                                    f"Projected image features shape: {image_features.shape}"
                                )

                            # For image-only input, we need to create a sequence
                            # The image features should be flattened and used as input
                            if (
                                len(image_features.shape) == 3
                            ):  # (batch_size, seq_len, hidden_size)
                                x = image_features
                            elif (
                                len(image_features.shape) == 2
                            ):  # (batch_size, hidden_size)
                                # Add sequence dimension
                                x = image_features.unsqueeze(
                                    1
                                )  # (batch_size, 1, hidden_size)
                            else:
                                # Flatten spatial dimensions if needed
                                batch_size = image_features.shape[0]
                                x = image_features.view(
                                    batch_size, -1, image_features.shape[-1]
                                )

                            log.info(f"Final image input shape: {x.shape}")

                            # For image-only input, we need to skip the embedding layer
                            # and go directly to the transformer blocks
                            # Apply embedding dropout
                            x = transformer.emb_drop(x)
                            log.info(
                                f"After embedding dropout shape: {x.shape}"
                            )

                            # Run through all transformer blocks to get the final output
                            log.info(
                                f"Running through {len(transformer.blocks)} transformer blocks"
                            )
                            for i, block in enumerate(transformer.blocks):
                                x, _ = block(
                                    x, attention_bias=None, use_cache=False
                                )
                                if i % 10 == 0:  # Log every 10th block
                                    log.info(
                                        f"Block {i} output shape: {x.shape}"
                                    )

                            # Apply final layer norm to get the last layer hidden states
                            last_layer_hidden_states = transformer.ln_f(x)
                            log.info(
                                f"Final hidden states shape: {last_layer_hidden_states.shape}"
                            )

                            return last_layer_hidden_states
                        else:
                            raise ValueError(
                                "Cannot process images without vision tower"
                            )
                    else:
                        raise ValueError(
                            "Either input_ids, inputs_embeds, or images must be provided"
                        )

                # Apply embedding dropout (only for text and multimodal inputs)
                if input_ids is not None or inputs_embeds is not None:
                    x = transformer.emb_drop(x)
                    log.info(f"After embedding dropout shape: {x.shape}")

                    # Run through all transformer blocks to get the final output
                    log.info(
                        f"Running through {len(transformer.blocks)} transformer blocks"
                    )
                    for i, block in enumerate(transformer.blocks):
                        x, _ = block(x, attention_bias=None, use_cache=False)
                        if i % 10 == 0:  # Log every 10th block
                            log.info(f"Block {i} output shape: {x.shape}")

                    # Apply final layer norm to get the last layer hidden states
                    last_layer_hidden_states = transformer.ln_f(x)
                    log.info(
                        f"Final hidden states shape: {last_layer_hidden_states.shape}"
                    )

            else:
                # Fallback: try the standard forward method
                log.warning("Using fallback method to get hidden states")
                outputs = self.model(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    images=images,
                    image_sizes=image_sizes,
                    modalities=modalities,
                    output_hidden_states=True,
                    return_dict=True,
                )
                log.info(f"outpus dict is \n {outputs.__dict__}")

                if (
                    hasattr(outputs, "hidden_states")
                    and outputs.hidden_states is not None
                ):
                    last_layer_hidden_states = outputs.hidden_states[-1]
                    log.info(
                        f"Fallback hidden states shape: {last_layer_hidden_states.shape}"
                    )
                else:
                    # Last resort: use logits (not ideal)
                    log.warning("Using logits as proxy for hidden states")
                    last_layer_hidden_states = outputs.logits
                    log.info(f"Logits shape: {last_layer_hidden_states.shape}")

            return last_layer_hidden_states

    def _average_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Average the hidden states across the sequence dimension.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Averaged embeddings of shape (batch_size, hidden_size)
        """
        if attention_mask is not None:
            # Use attention mask to exclude padding tokens
            # attention_mask: 1 for valid tokens, 0 for padding
            mask = attention_mask.unsqueeze(
                -1
            ).float()  # (batch_size, seq_len, 1)
            masked_hidden_states = hidden_states * mask
            # Sum across sequence dimension and divide by number of valid tokens
            sum_hidden_states = masked_hidden_states.sum(
                dim=1
            )  # (batch_size, hidden_size)
            num_valid_tokens = mask.sum(dim=1)  # (batch_size, 1)
            # Avoid division by zero
            num_valid_tokens = torch.clamp(num_valid_tokens, min=1.0)
            averaged_embeddings = sum_hidden_states / num_valid_tokens
        else:
            # Simple average across sequence dimension
            averaged_embeddings = hidden_states.mean(
                dim=1
            )  # (batch_size, hidden_size)

        return averaged_embeddings

    def embed_text(
        self, text: Union[str, List[str]], max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate embeddings for text input.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length

        Returns:
            Text embeddings of shape (batch_size, hidden_size)
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        log.info("Starting to extract hidden states")
        # Extract hidden states
        hidden_states = self._extract_last_layer_hidden_states(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Average the hidden states
        embeddings = self._average_hidden_states(hidden_states, attention_mask)

        return embeddings

    def embed_image(
        self, images: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Generate embeddings for image input.

        Args:
            images: Input images of shape (batch_size, channels, height, width) or list of images

        Returns:
            Image embeddings of shape (batch_size, hidden_size)
        """
        if isinstance(images, list):
            # Stack images if they are in a list
            images = torch.stack(images)

        # Ensure images are on the correct device
        images = images.to(self.device)

        # Extract hidden states (no input_ids, only images)
        hidden_states = self._extract_last_layer_hidden_states(images=images)

        # Average the hidden states
        embeddings = self._average_hidden_states(hidden_states)

        return embeddings

    def embed_fused(
        self,
        text: Union[str, List[str]],
        images: Union[torch.Tensor, List[torch.Tensor]],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate fused embeddings for text and image input.

        Args:
            text: Input text or list of texts
            images: Input images of shape (batch_size, channels, height, width) or list of images
            max_length: Maximum sequence length

        Returns:
            Fused text-image embeddings of shape (batch_size, hidden_size)
        """
        if isinstance(text, str):
            text = [text]

        if isinstance(images, list):
            images = torch.stack(images)

        # Ensure images are on the correct device
        images = images.to(self.device)

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # Extract hidden states for text + image
        hidden_states = self._extract_last_layer_hidden_states(
            input_ids=input_ids, attention_mask=attention_mask, images=images
        )

        # Average the hidden states
        embeddings = self._average_hidden_states(hidden_states, attention_mask)

        return embeddings

    def forward(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass that automatically determines the embedding type based on inputs.

        Args:
            text: Optional text input
            images: Optional image input
            max_length: Maximum sequence length for text

        Returns:
            Embeddings of shape (batch_size, hidden_size)
        """
        if text is not None and images is not None:
            # Text + Image fusion
            return self.embed_fused(text, images, max_length)
        elif text is not None:
            # Text only
            return self.embed_text(text, max_length)
        elif images is not None:
            # Image only
            return self.embed_image(images)
        else:
            raise ValueError("Either text or images (or both) must be provided")

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.hidden_size


def create_lavida_embed_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    **kwargs,
) -> LaViDaEmbedModel:
    """
    Factory function to create a LaViDa embedding model.

    Args:
        model_path: Path to the LaViDa model
        device: Device to load the model on
        torch_dtype: Torch data type
        **kwargs: Additional arguments passed to the model

    Returns:
        LaViDaEmbedModel instance
    """
    return LaViDaEmbedModel(
        model_path=model_path, device=device, torch_dtype=torch_dtype, **kwargs
    )
