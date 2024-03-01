import argparse
import torch

from inference.inference import generate_and_store_captions
from models import RSDiXClipCap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--processor", type=str, default="openai/clip-vit-base-patch32",
                        help="Processor from CLIPProcessor.from_pretrained to preprocess data")
    parser.add_argument('--out_path', type=str, default=f'_inferenceimages/',
                        help='path to store the generated captions in json format')
    parser.add_argument('--use_beam_search', action="store_true",
                        help='whether to use beam search during inference')
    args = parser.parse_args()

    # Load the params
    print("Loaded args: ", args)
    # Load the model
    print(f"Loading CLIPCap model from checkpoint: {args.checkpoint_path}")
    model = RSDiXClipCap.load_from_checkpoint(args.checkpoint_path)
    print("Loaded CLIPCap model")

    # Generate the captions
    generate_and_store_captions(model=model, args=args)

    del model
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
