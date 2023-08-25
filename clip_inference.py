import argparse
import torch

from inference.inference import generate_and_store_clip_embeddings
from models import CLIPWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--processor", type=str, default="openai/clip-vit-base-patch32",
                        help="Processor from CLIPProcessor.from_pretrained to preprocess data")
    parser.add_argument('--out_path', type=str, default=f'_clipinferenceimages/',
                        help='path to store the generated captions in json format')
    args = parser.parse_args()

    # Load the params
    print("Loaded args: ", args)
    # Load the model
    print(f"Loading CLIP model from checkpoint: {args.checkpoint_path}")
    model = CLIPWrapper.load_from_checkpoint(args.checkpoint_path)
    print("Loaded CLIP model")

    # Generate the captions
    generate_and_store_clip_embeddings(clip_model=model, args=args)

    del model
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
