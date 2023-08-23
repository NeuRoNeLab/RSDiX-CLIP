import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as t
from torchvision.io import read_image
from tqdm import tqdm
from transformers import CLIPProcessor

from models import CLIPWrapper
from evaluation.utils import get_eval_images, get_model_basename, get_classes

K_VALUES = [1, 3, 5, 10]


def predict_image(img_file, model, processor, eval_sentences, classes_names, k, imgs_dir):
    """
    Predicts classes for an input image.

    Args:
        img_file (str): The filename of the image.
        model (CLIPWrapper): The pre-trained CLIP model.
        processor (CLIPProcessor): The CLIP processor for data preprocessing.
        eval_sentences (list of str): Evaluation sentences corresponding to class names.
        classes_names (list of str): List of class names.
        k (int): The number of top predictions to consider.
        imgs_dir (str): Directory containing the evaluation images.

    Returns:
        str: The true label of the image.
        list of tuple: A list of top-K predicted class-label and probability pairs.
        """
    label = img_file.split("_")[0]
    img_file = os.path.join(imgs_dir, img_file)

    img_ext = img_file.split(".")[-1]
    if img_ext != 'jpeg' and img_file != 'png':
        eval_image = t.PILToTensor()(Image.open(img_file))
    else:
        eval_image = read_image(img_file)

    inputs = processor(images=eval_image, text=eval_sentences, truncation=True, padding="max_length",
                       return_tensors="pt")

    # move inputs to current device
    for key in inputs.keys():
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(model.device)

    outputs = model(inputs, return_loss=False)
    probs = outputs.logits_per_image.softmax(dim=1).cpu().detach().numpy()
    probs_np = np.asarray(probs)[0]
    probs_npi = np.argsort(-probs_np)
    predictions = [(classes_names[i], probs_np[i]) for i in probs_npi[0:k]]

    return label, predictions


def predict(model, processor, eval_images, classes_names, model_scores_file, imgs_dir):
    """
    Predicts classes for a list of evaluation images using a CLIP model and computes scores.

    Args:
        model (CLIPWrapper): The pre-trained CLIP model.
        processor (CLIPProcessor): The CLIP processor for data preprocessing.
        eval_images (list of str): List of image filenames to evaluate.
        classes_names (list of str): List of class names.
        model_scores_file (str): Path to the file to store prediction scores.
        imgs_dir (str): Directory containing the evaluation images.
    """
    print("Generating predictions...")
    eval_sentences = [f"Aerial photograph of {cn}" for cn in classes_names]
    images_predicted = 0

    with open(model_scores_file, "w") as msf:
        for eval_image in tqdm(eval_images):
            label, predictions = predict_image(eval_image, model, processor, eval_sentences,
                                               classes_names, max(K_VALUES), imgs_dir)

            msf.write("{:s}\t{:s}\t{:s}\n".format(eval_image, label, "\t".join(["{:s}\t{:.5f}".format(c, p)
                                                                                for c, p in predictions])))
            images_predicted += 1

    print(f"{images_predicted} images evaluated, COMPLETED!")


def compute_scores(scores_file, model_scores_file, model_basename):
    """
    Computes final accuracy scores based on prediction results.

    Args:
        scores_file (str): Path to the file to store final scores.
        model_scores_file (str): Path to the file containing prediction scores.
        model_basename (str): Basename of the model being evaluated.
    """
    print("Computing final scores...")
    num_examples = 0
    correct_k = [0] * len(K_VALUES)

    with open(model_scores_file, "r") as msf:
        for line in msf:
            cols = line.strip().split('\t')
            label = cols[1]
            preds = []
            for i in range(2, 22, 2):
                preds.append(cols[i])
            for kid, k in enumerate(K_VALUES):
                preds_k = set(preds[0:k])
                if label in preds_k:
                    correct_k[kid] += 1
            num_examples += 1

    scores_k = [ck / num_examples for ck in correct_k]
    print("\t".join(["score@{:d}".format(k) for k in K_VALUES]))
    print("\t".join(["{:.3f}".format(s) for s in scores_k]))

    with open(scores_file, "a") as sf:
        sf.write("{:s}\t{:s}\n".format(model_basename, "\t".join(["{:.3f}".format(s) for s in scores_k])))


def main(args):
    print("Evaluating CLIP: Starting evaluation...")
    print(f"Loading checkpoint: {args.model_pth} and processor: {args.processor}")

    model = CLIPWrapper.load_from_checkpoint(args.model_pth)
    processor = CLIPProcessor.from_pretrained(args.processor)

    model_scores_file = os.path.join(args.scores_dir, get_model_basename(args.model_pth)) + ".tsv"

    classes_names = get_classes(imgs_dir=args.imgs_dir)
    eval_images = get_eval_images(annotations_file=args.annotations_file)

    predict(model, processor, eval_images, classes_names, model_scores_file, args.imgs_dir)
    compute_scores(os.path.join(args.scores_dir, args.scores_file), model_scores_file,
                   get_model_basename(args.model_pth))

    print("Evaluation COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scores_dir", type=str, default=os.path.join(os.getcwd(), "eval_results"))
    parser.add_argument("--scores_file", type=str, default="scores.tsv")
    parser.add_argument("--model_pth", type=str, help="Path of the model to evaluate", required=True)
    parser.add_argument("--processor", type=str, default="openai/clip-vit-base-patch32",
                        help="Processor from CLIPProcessor.from_pretrained to preprocess data")
    parser.add_argument("--annotations_file", type=str, required=True)
    parser.add_argument("--imgs_dir", type=str, required=True)

    main(parser.parse_args())
