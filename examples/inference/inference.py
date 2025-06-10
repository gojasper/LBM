import argparse
import logging
import os

import torch
from PIL import Image

from lbm.inference import evaluate, get_model

PATH = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--source_image", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument(
    "--model_name",
    type=str,
    default="normals",
    choices=["normals", "depth", "relighting"],
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to local model directory (overrides model_name if provided)",
)


args = parser.parse_args()


def main():
    # Use custom model path if provided
    if args.model_path:
        logging.info(f"Loading LBM model from custom path: {args.model_path}")
        model = get_model(args.model_path, torch_dtype=torch.bfloat16, device="cuda")
    
    # Otherwise use model_name with HF hub or local cache
    elif not os.path.exists(os.path.join(PATH, "ckpts", f"{args.model_name}")):
        logging.info(f"Downloading {args.model_name} LBM model from HF hub...")
        model = get_model(
            f"jasperai/LBM_{args.model_name}",
            save_dir=os.path.join(PATH, "ckpts", f"{args.model_name}"),
            torch_dtype=torch.bfloat16,
            device="cuda",
        )

    else:
        model_dir = os.path.join(PATH, "ckpts", f"{args.model_name}")
        logging.info(f"Loading {args.model_name} LBM model from local...")
        model = get_model(model_dir, torch_dtype=torch.bfloat16, device="cuda")

    source_image = Image.open(args.source_image).convert("RGB")

    output_image = evaluate(model, source_image, args.num_inference_steps)

    os.makedirs(args.output_path, exist_ok=True)

    source_image.save(os.path.join(args.output_path, "source_image.jpg"))
    output_image.save(os.path.join(args.output_path, "output_image.jpg"))


if __name__ == "__main__":
    main()
