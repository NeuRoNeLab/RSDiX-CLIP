import json
import os

import torch
from tqdm import tqdm
from transformers import CLIPProcessor

from datasets import CaptioningDataset
from models import CLIPCapWrapper, CLIPWrapper
from models.clipcap import generate_caption
from utils import IMAGE_FIELD


@torch.no_grad()
def generate_and_store_captions(model: CLIPCapWrapper, args, checkpoint_interval: int = 50):
    """
    Generates and stores captions for images in a dataset using a pre-trained image-captioning model.

    Args:
        model (CLIPCapWrapper): The image-captioning model for generating captions.
        args (argparse.Namespace): The command-line arguments containing the following:
            - annotations_file (str): The path to the dataset file containing the annotations.
            - img_dir (str): Directory where all the images are contained.
            - processor (str): Processor from CLIPProcessor.from_pretrained to preprocess data.
            - out_path (str): The output directory to save the generated captions.
            - use_beam_search (bool): Whether to use beam search for text generation.
        checkpoint_interval (int, optional): The interval to store generated captions. Defaults to 50.
    """
    # Load ImageFolder dataset
    ds = CaptioningDataset(annotations_file=args.annotations_file, img_dir=args.img_dir)
    print(f"Dataset - with annotations file: '{args.annotations_file} and image directory: {args.img_dir}' - loaded.")
    preprocessor = CLIPProcessor.from_pretrained(args.processor)
    # Initialize caption buffer and start inference
    captions_buffer = {}
    progress_bar = tqdm(range(0, len(ds)), desc=f"Performing inference on dataset "
                                                f"- with annotations file: '{args.annotations_file} "
                                                f"and image directory: {args.img_dir}' -'...")
    os.makedirs(args.out_path, exist_ok=True)  # make the out dir if it doesn't exist
    for i in progress_bar:
        # Load PIL image
        img, _ = ds[i]

        # Preprocess data
        img = preprocessor(images=img, return_tensors="pt")[IMAGE_FIELD].to(model.device)

        # Get the caption
        caption = generate_caption(imgs=img,
                                   clip_encoder=model.clip_encoder,
                                   tokenizer=model.gpt2_tokenizer,
                                   model=model.clipcap,
                                   use_beam_search=args.use_beam_search)

        # Save the caption in the buffer
        filename = ds.img_captions.iloc[i, 0]['filename']
        captions_buffer[filename] = caption

        # Store the captions every 'checkpoint_interval' iterations
        if (i + 1) % checkpoint_interval == 0:
            with open(f"{args.out_path}_{i - checkpoint_interval + 1}_{i}.json", "w") as fp:
                json.dump(captions_buffer, fp, indent=2)
            captions_buffer = {}
            print(f"\n\nStored captions from {i - checkpoint_interval + 1} to {i} to path: "
                  f"'{args.out_path}_{i - checkpoint_interval + 1}_{i}.json'.\n\n")

    # Store remaining captions
    if len(captions_buffer) > 0:
        with open(f"{args.out_path}_{len(ds) - (len(ds) % checkpoint_interval)}_{len(ds) - 1}.json", "w") as fp:
            json.dump(captions_buffer, fp, indent=2)
        print(f"\n\nStored captions from {len(ds) - (len(ds) % checkpoint_interval)} to {len(ds) - 1} to path: "
              f"'{args.out_path}_{len(ds) - (len(ds) % checkpoint_interval)}_{len(ds) - 1}.json'.\n\n")

    print(f"Inference on dataset - with annotations file: '{args.annotations_file} and image directory: "
          f"{args.img_dir}' - completed.")


@torch.no_grad()
def generate_and_store_clip_embeddings(clip_model: CLIPWrapper, args):
    """
    Extracts CLIP embeddings for images in a given dataset and saves them as individual files.

    Args:
        clip_model (CLIPWrapper): The CLIP model used for encoding images.
        args (argparse.Namespace): The command-line arguments containing the following:
            - annotations_file (str): The path to the dataset file containing the annotations.
            - img_dir (str): Directory where all the images are contained.
            - processor (str): Processor from CLIPProcessor.from_pretrained to preprocess data.
            - out_path (str): The output directory to save the embeddings.
    """
    # Load ImageFolder dataset
    ds = CaptioningDataset(annotations_file=args.annotations_file, img_dir=args.img_dir)
    print(f"Dataset - with annotations file: '{args.annotations_file} and image directory: {args.img_dir}' - loaded.")

    # Load CLIP model and GPT-2 tokenizer
    preprocessor = CLIPProcessor.from_pretrained(args.processor)

    progress_bar = tqdm(range(0, len(ds)),
                        desc=f"Performing CLIP embedding extraction on dataset "
                             f"- with annotations file: '{args.annotations_file} "
                             f"and image directory: {args.img_dir}' -...")

    os.makedirs(args.out_path, exist_ok=True)  # make the out dir if it doesn't exist
    for i in progress_bar:
        # Load PIL image
        img, _ = ds[i]

        img = preprocessor(images=img, return_tensors="pt")[IMAGE_FIELD]

        # Get the embedding
        image_embedding = get_image_embedding(img, clip_model, preprocessor).cpu()

        # Store the embedding
        filename = ds.img_captions.iloc[i, 0]['filename'].split(".")[0].split("/")[-1] + ".pt"
        torch.save(image_embedding, os.path.join(args.out_path, filename))

    print(f"CLIP inference on dataset - with annotations file: '{args.annotations_file} "
          f"and image directory: {args.img_dir}' - completed.")


@torch.no_grad()
def get_image_embedding(imgs, clip_model) -> torch.Tensor:
    """
    Computes the CLIP embeddings for a given image or a list of images.

    Args:
        imgs (torch.Tensor): The input image(s) for embedding extraction.
        clip_model (CLIPWrapper): The CLIP model used for encoding images.

    Returns:
        torch.Tensor: The CLIP embeddings for the input image(s).
    """

    imgs = imgs.to(clip_model.device)

    if isinstance(imgs, list):
        emb_list = []
        for img in imgs:
            img = img.to(clip_model.device)
            image_embedding = clip_model.encode_image(img)
            emb_list.append(image_embedding)
        image_embedding = torch.stack(emb_list)
    else:
        image_embedding = clip_model.encode_image(imgs)

    return image_embedding
