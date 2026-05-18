"""
Shared utilities: seeding, config loading, logging, device detection.
"""

import os
import random
import logging

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        level=level,
    )
    # Suppress noisy HuggingFace logs
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    return logging.getLogger('hateval')


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
        return torch.device('cuda')
    print("No GPU found — using CPU (training will be slow)")
    return torch.device('cpu')


def model_shortname(model_name: str) -> str:
    """Convert model name to a filesystem-safe short identifier."""
    return (
        model_name
        .replace('/', '_')
        .replace('-', '_')
        .lower()
    )
