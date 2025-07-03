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
import torch

# sys.path.append(str(Path(__file__).parent.parent))
from llava.model.builder import load_pretrained_model

from embed.lavida_embed import LaViDaEmbedModel, create_lavida_embed_model

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")


def test_lavida_simple_text():
    model_path = "jacklishufan/lavida-llada-v1.0-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava_llada",
        device_map=device,
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()

    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(inputs)
    log.info(outputs.__dict__)
