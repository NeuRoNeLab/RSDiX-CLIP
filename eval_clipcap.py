import argparse
import os
from typing import Dict

from aac_metrics.functional import sbert_sim, rouge_l, bleu, meteor
from tqdm import tqdm
from transformers import CLIPProcessor

from datasets import CaptioningDataset
from models import CLIPCapWrapper
from models.clipcap import generate_caption
from utils import METEOR, SBERT_SIM, ROUGE_L, BLEU, MAX_BLEU, MIN_BLEU, \
    IMAGE_FIELD, CLIP_MAX_LENGTH, get_model_basename


def eval_model(ds: CaptioningDataset, model: CLIPCapWrapper, preprocessor: CLIPProcessor, args) \
        -> Dict[str, float]:
    """
    Evaluates the performance of the CLIPCapWrapper on a given dataset.

    Args:
        ds (datasets.CaptioningDataset): The dataset to evaluate the model on.
        model (CLIPCapWrapper): The CLIPCapWrapper to be evaluated.
        preprocessor (CLIPProcessor): The CLIPProcessor to preprocess the image with.
        args (argparse.Namespace): The command-line arguments containing the following:
            - metrics (List[str]): List of evaluation metrics to compute (e.g., METEOR, SBERT_SIM, ROUGE_L, BLEU1,
                BLEU2, etc.).
            - use_beam_search (bool): Whether to use beam search for text generation.

    Returns:
        Dict[str, float]: A dictionary containing the average value of each evaluation metric computed on the dataset.
    """

    # Initialize metrics dict and start evaluation
    avg_metrics = {metric: 0.0 for metric in args.metrics}
    no_meteor_count = 0
    progress_bar = tqdm(range(0, len(ds)), desc=f"Evaluating model, current metrics: {avg_metrics}")
    for i in progress_bar:

        img = preprocessor(images=ds[i][0], truncation=True, padding="max_length", max_length=CLIP_MAX_LENGTH,
                           return_tensors="pt")[IMAGE_FIELD].to(model.device)
        # get row from pd dataset to extract all captions
        row = ds.img_captions.iloc[i, 0]
        reference_captions = [[sentence['raw'] for sentence in row["sentences"]]]

        # Get the caption
        preds = generate_caption(imgs=img,
                                 clip_encoder=model.clip_encoder,
                                 tokenizer=model.gpt2_tokenizer,
                                 model=model.clipcap,
                                 use_beam_search=args.use_beam_search)

        if METEOR in avg_metrics:
            try:
                value, _ = meteor(candidates=preds, mult_references=reference_captions,
                                  java_path=os.getenv("JAVA_HOME"))
                value = value[METEOR].item()
                avg_metrics[METEOR] = avg_metrics[METEOR] + 1 / (i + 1 - no_meteor_count) * (
                            value - avg_metrics[METEOR])
            except ValueError as e:
                print(f"Meteor could not be computed due to error {e.with_traceback(None)} "
                      f"on the couple: ({preds}, {reference_captions}). "
                      f"Increasing the no_meteor_count to {no_meteor_count}")
                no_meteor_count += 1

        if SBERT_SIM in avg_metrics:
            value, _ = sbert_sim(candidates=preds, mult_references=reference_captions)
            value = value[SBERT_SIM].item()
            avg_metrics[SBERT_SIM] = avg_metrics[SBERT_SIM] + 1 / (i + 1) * (value - avg_metrics[SBERT_SIM])
        if ROUGE_L in avg_metrics:
            value, _ = rouge_l(candidates=preds, mult_references=reference_captions)
            value = value[ROUGE_L].item()
            avg_metrics[ROUGE_L] = avg_metrics[ROUGE_L] + 1 / (i + 1) * (value - avg_metrics[ROUGE_L])
        for j in range(MIN_BLEU, MAX_BLEU + 1):
            bleu_j = f"{BLEU}{j}"
            if bleu_j in avg_metrics:
                value, _ = bleu(candidates=preds, mult_references=reference_captions, n=j)
                value = value[bleu_j].item()
                avg_metrics[bleu_j] = avg_metrics[bleu_j] + 1 / (i + 1) * (value - avg_metrics[bleu_j])

        progress_bar.set_description(f"Evaluating model, current metrics: {avg_metrics}")

    return avg_metrics


def main(args):
    print("Evaluating CLIP-CAP: Starting evaluation...")
    print(f"Loading checkpoint: {args.model_pth} and processor: {args.processor}")

    model = CLIPCapWrapper.load_from_checkpoint(args.model_pth)
    preprocessor = CLIPProcessor.from_pretrained(args.processor)
    ds = CaptioningDataset(annotations_file=args.annotations_file, img_dir=args.imgs_dir, train=False)

    avg_metrics = eval_model(ds=ds, model=model, preprocessor=preprocessor, args=args)
    metrics_str = ""

    for metric, value in avg_metrics.items():
        metrics_str += "{:s}\t{:.3f}\t".format(metric, value)

    with open(os.path.join(args.scores_dir, args.scores_file), "a") as msf:
        msf.write("{:s}\t{:s}\n".format(get_model_basename(args.model_pth), metrics_str))

    print("Evaluation COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scores_dir", type=str, default=os.path.join(os.getcwd(), "eval_results"))
    parser.add_argument("--scores_file", type=str, default="clip_cap_scores.tsv")
    parser.add_argument("--model_pth", type=str, help="Path of the model to evaluate", required=True)
    parser.add_argument("--processor", type=str, default="openai/clip-vit-base-patch32",
                        help="Processor from CLIPProcessor.from_pretrained to preprocess data")
    parser.add_argument("--annotations_file", type=str, required=True)
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--use_beam_search", type=bool, default=False)
    parser.add_argument('--metrics', nargs='*',
                        default=[ROUGE_L, SBERT_SIM, f'{BLEU}1', f'{BLEU}2', f'{BLEU}3', f'{BLEU}4'],
                        help='the metrics to use during evaluation')

    main(parser.parse_args())
