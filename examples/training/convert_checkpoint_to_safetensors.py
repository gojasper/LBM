#!/usr/bin/env python3
"""
Convert PyTorch Lightning checkpoint to safetensors format for inference.

This script converts the large training checkpoints (~14GB) that include optimizer state
and training metadata to lightweight safetensors files (~5GB) with just model weights.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint_to_safetensors(
    checkpoint_path: str,
    output_dir: str,
    config_path: str = None,
):
    """
    Convert a PyTorch Lightning checkpoint to safetensors format.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        output_dir: Directory to save the converted files
        config_path: Path to config.yaml (if None, will look in checkpoint directory)
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract model state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise ValueError("No 'state_dict' found in checkpoint")
    
    # Remove "model." prefix from keys and filter out LPIPS loss weights
    logger.info("Cleaning state dict - removing 'model.' prefix and filtering training-only weights")
    cleaned_state_dict = {}
    model_prefix = "model."
    
    for key, value in state_dict.items():
        if key.startswith(model_prefix):
            new_key = key[len(model_prefix):]
            # Skip LPIPS loss weights that are only used during training
            if new_key.startswith("lpips_loss."):
                logger.debug(f"Skipping training-only weight: {new_key}")
                continue
            # Clone tensors to break memory sharing (fixes safetensors shared memory error)
            cleaned_state_dict[new_key] = value.clone()
        else:
            # Skip LPIPS loss weights that don't have model prefix
            if key.startswith("lpips_loss."):
                logger.debug(f"Skipping training-only weight: {key}")
                continue
            # Keep keys that don't have the model prefix
            cleaned_state_dict[key] = value.clone()
    
    # Save as safetensors
    safetensors_path = output_dir / "model.safetensors"
    logger.info(f"Saving safetensors to {safetensors_path}")
    save_file(cleaned_state_dict, safetensors_path)
    
    # Handle config.yaml
    if config_path is None:
        # Look for config.yaml in the same directory as checkpoint
        config_path = checkpoint_path.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        output_config_path = output_dir / "config.yaml"
        logger.info(f"Copying config from {config_path} to {output_config_path}")
        shutil.copy2(config_path, output_config_path)
    else:
        logger.warning(f"Config file not found at {config_path}")
        logger.info("You may need to manually create config.yaml for inference")
    
    # Log size comparison
    original_size = checkpoint_path.stat().st_size / (1024**3)  # GB
    new_size = safetensors_path.stat().st_size / (1024**3)  # GB
    
    logger.info(f"Conversion complete!")
    logger.info(f"Original checkpoint: {original_size:.2f} GB")
    logger.info(f"Safetensors file: {new_size:.2f} GB")
    logger.info(f"Size reduction: {((original_size - new_size) / original_size * 100):.1f}%")
    logger.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Lightning checkpoint to safetensors format"
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to the .ckpt file to convert"
    )
    parser.add_argument(
        "--output_dir", 
        required=True,
        help="Directory to save the converted files"
    )
    parser.add_argument(
        "--config_path",
        help="Path to config.yaml (optional, will look in checkpoint directory if not provided)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_checkpoint_to_safetensors(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            config_path=args.config_path,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
