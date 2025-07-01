import torch
import torch.nn as nn
from typing import Union, List, Optional
import logging
from rich.logging import RichHandler
import copy
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.language_model.llada.generate import generate as llada_generate

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


class LaViDaEmbedModel(nn.Module):
    def __init__(
        self,
        model_path: str = "jacklishufan/lavida-llada-v1.0-instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        vision_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_path = model_path

        # Default vision kwargs following embed_example.py
        self.vision_kwargs = dict(
            mm_vision_tower="google/siglip-so400m-patch14-384",
            mm_resampler_type=None,
            mm_projector_type="mlp2x_gelu",
            mm_hidden_size=1152,
            use_mm_proj=True,
        )
        if vision_kwargs is not None:
            self.vision_kwargs.update(vision_kwargs)

        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name="llava_llada",
                device_map=device,
                torch_dtype=torch_dtype,
                vision_kwargs=self.vision_kwargs,
                **kwargs,
            )
        )

        # Setup model following embed_example.py
        self.model.eval()
        self.model.tie_weights()
        self.model.to(
            torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        )
        self.hidden_size = self.model.config.hidden_size

        log.info(
            f"Loaded LaViDa model from {model_path} with hidden size {self.hidden_size}"
        )

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask
            summed = masked_hidden.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts
        else:
            return hidden_states.mean(dim=1).squeeze(0)

    def embed_text(
        self, text: Union[str, List[str]], max_length: Optional[int] = None
    ) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        embeddings = []
        for t in text:
            # Use conversation template following embed_example.py
            conv = copy.deepcopy(conv_templates["llada"])
            conv.append_message(
                conv.roles[0], t + " conclusion this in one sentence."
            )
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # log.info(f"Text prompt: {prompt}")

            # Tokenize using the same approach as embed_example.py
            input_ids = (
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )

            # log.info(f"input_ids shape: {input_ids.shape}")
            # log.info(f"input_ids: {input_ids}")

            # Use model.generate() following embed_example.py approach
            with torch.no_grad():
                hidden_states = self.model.generate(
                    input_ids,
                    embedd_flag=True,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=8,  # Just get the last hidden state
                    block_length=8,
                    step_ratio=1.0,
                    tokenizer=self.tokenizer,
                    prefix_lm=True,
                    verbose=False,
                )
                # Get the last hidden state
                # last_hidden = hidden_states[-1]

            # log.info(f"last_hidden shape: {hidden_states.shape}")

            # embedding = self._mean_pool(hidden_states[:, 44:-7])
            embedding = self._mean_pool(hidden_states)
            embeddings.append(embedding)

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.stack(embeddings)

    def embed_image(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        embeddings = []
        for image in images:
            # Process image following embed_example.py
            image_tensor = process_images(
                [image], self.image_processor, self.model.config
            )
            image_tensor = [
                _image.to(dtype=torch.bfloat16, device=self.device)
                for _image in image_tensor
            ]

            # Use conversation template with image token
            conv = copy.deepcopy(conv_templates["llada"])
            question = (
                DEFAULT_IMAGE_TOKEN + "\nDescribe the image in one sentence."
            )
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # log.info(f"Image prompt: {prompt_question}")

            # Tokenize following embed_example.py
            input_ids = (
                tokenizer_image_token(
                    prompt_question,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )

            image_sizes = [image.size]

            # Use model.generate() following embed_example.py
            with torch.no_grad():
                hidden_states = self.model.generate(
                    input_ids,
                    embedd_flag=True,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=8,  # Just get the last hidden state
                    block_length=8,
                    step_ratio=1.0,
                    tokenizer=self.tokenizer,
                    prefix_lm=True,
                    verbose=False,
                )
                last_hidden = hidden_states[:, 42:-7]
                last_hidden = hidden_states

            embedding = self._mean_pool(last_hidden)
            embeddings.append(embedding)

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.stack(embeddings)

    def embed_fused(
        self,
        text: Union[str, List[str]],
        images: Union[Image.Image, List[Image.Image]],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        if isinstance(images, Image.Image):
            images = [images]

        embeddings = []
        for t, image in zip(text, images):
            # Process image
            image_tensor = process_images(
                [image], self.image_processor, self.model.config
            )
            image_tensor = [
                _image.to(dtype=torch.bfloat16, device=self.device)
                for _image in image_tensor
            ]

            # Use conversation template with image and text
            conv = copy.deepcopy(conv_templates["llada"])
            question = DEFAULT_IMAGE_TOKEN + "\n" + t
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            log.info(f"Fused prompt: {prompt_question}")

            # Tokenize
            input_ids = (
                tokenizer_image_token(
                    prompt_question,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )

            image_sizes = [image.size]

            # Use model.generate()
            with torch.no_grad():
                hidden_states = self.model.generate(
                    input_ids,
                    embedd_flag=True,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=64,
                    block_length=64,
                    step_ratio=1.0,
                    tokenizer=self.tokenizer,
                    prefix_lm=True,
                    verbose=False,
                )
                last_hidden = hidden_states[:, 44:-7]

            log.info(f"last_hidden shape: {last_hidden.shape}")

            embedding = self._mean_pool(last_hidden)
            embeddings.append(embedding)

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.stack(embeddings)

    def forward(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        if text is not None and images is not None:
            return self.embed_fused(text, images, max_length)
        elif text is not None:
            return self.embed_text(text, max_length)
        elif images is not None:
            return self.embed_image(images)
        else:
            raise ValueError("Either text or images (or both) must be provided")

    def get_embedding_dim(self) -> int:
        return self.hidden_size

    def compute_similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        similarity_type: str = "cosine",
    ) -> torch.Tensor:
        """
        Compute similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (N, hidden_size)
            embeddings2: Second set of embeddings (M, hidden_size)
            similarity_type: Type of similarity ("cosine", "euclidean", "dot")

        Returns:
            Similarity matrix (N, M)
        """
        # Normalize embeddings for cosine similarity
        if similarity_type == "cosine":
            emb1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
            emb2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
            similarity = torch.mm(emb1_norm, emb2_norm.t())
        elif similarity_type == "dot":
            similarity = torch.mm(embeddings1, embeddings2.t())
        elif similarity_type == "euclidean":
            # Convert to distance, then to similarity (1 / (1 + distance))
            dist = torch.cdist(embeddings1, embeddings2, p=2)
            similarity = 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")

        return similarity

    def text_similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]],
        similarity_type: str = "cosine",
    ) -> torch.Tensor:
        """
        Compute similarity between two sets of text embeddings.

        Args:
            texts1: First set of texts
            texts2: Second set of texts
            similarity_type: Type of similarity ("cosine", "euclidean", "dot")

        Returns:
            Similarity matrix
        """
        embeddings1 = self.embed_text(texts1)
        embeddings2 = self.embed_text(texts2)
        log.info(f"embeddings1 shape: {embeddings1.shape}")
        log.info(f"embeddings2 shape: {embeddings2.shape}")
        return self.compute_similarity(
            embeddings1, embeddings2, similarity_type
        )

    def image_similarity(
        self,
        images1: Union[Image.Image, List[Image.Image]],
        images2: Union[Image.Image, List[Image.Image]],
        similarity_type: str = "cosine",
    ) -> torch.Tensor:
        """
        Compute similarity between two sets of image embeddings.

        Args:
            images1: First set of images
            images2: Second set of images
            similarity_type: Type of similarity ("cosine", "euclidean", "dot")

        Returns:
            Similarity matrix
        """
        embeddings1 = self.embed_image(images1)
        embeddings2 = self.embed_image(images2)
        return self.compute_similarity(
            embeddings1, embeddings2, similarity_type
        )

    def text_image_similarity(
        self,
        texts: Union[str, List[str]],
        images: Union[Image.Image, List[Image.Image]],
        similarity_type: str = "cosine",
    ) -> torch.Tensor:
        """
        Compute similarity between text and image embeddings.

        Args:
            texts: Set of texts
            images: Set of images
            similarity_type: Type of similarity ("cosine", "euclidean", "dot")

        Returns:
            Similarity matrix
        """
        text_embeddings = self.embed_text(texts)
        image_embeddings = self.embed_image(images)
        return self.compute_similarity(
            text_embeddings, image_embeddings, similarity_type
        )

    def fused_similarity(
        self,
        text_image_pairs1: List[tuple],
        text_image_pairs2: List[tuple],
        similarity_type: str = "cosine",
    ) -> torch.Tensor:
        """
        Compute similarity between two sets of text-image fusion embeddings.

        Args:
            text_image_pairs1: List of (text, image) tuples for first set
            text_image_pairs2: List of (text, image) tuples for second set
            similarity_type: Type of similarity ("cosine", "euclidean", "dot")

        Returns:
            Similarity matrix
        """
        # Extract texts and images from pairs
        texts1, images1 = zip(*text_image_pairs1)
        texts2, images2 = zip(*text_image_pairs2)

        embeddings1 = self.embed_fused(texts1, images1)
        embeddings2 = self.embed_fused(texts2, images2)
        return self.compute_similarity(
            embeddings1, embeddings2, similarity_type
        )


def create_lavida_embed_model(
    model_path: str = "jacklishufan/lavida-llada-v1.0-instruct",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    vision_kwargs: Optional[dict] = None,
    **kwargs,
) -> LaViDaEmbedModel:
    return LaViDaEmbedModel(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
        vision_kwargs=vision_kwargs,
        **kwargs,
    )
