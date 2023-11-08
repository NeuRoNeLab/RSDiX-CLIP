import argparse
import os
from typing import Dict

from aac_metrics.functional import sbert_sim, rouge_l, bleu, meteor
from lightning import seed_everything
from tqdm import tqdm
from transformers import CLIPProcessor

from evaluation.utils import get_model_basename, get_splits_for_evaluation
from models import CLIPCapWrapper
from models.clipcap import generate_caption
from utils import METEOR, SBERT_SIM, ROUGE_L, BLEU, MAX_BLEU, MIN_BLEU, \
    IMAGE_FIELD, CLIP_MAX_LENGTH, RAW_CAPTION_FIELD


def eval_model(model: CLIPCapWrapper, preprocessor: CLIPProcessor, args) \
        -> Dict[str, float]:
    """
    Evaluates the performance of the CLIPCapWrapper on a given dataset.

    Args:
        model (CLIPCapWrapper): The CLIPCapWrapper to be evaluated.
        preprocessor (CLIPProcessor): The CLIPProcessor to preprocess the image with.
        args (argparse.Namespace): The command-line arguments containing the following:
            - metrics (List[str]): List of evaluation metrics to compute (e.g., METEOR, SBERT_SIM, ROUGE_L, BLEU1,
                BLEU2, etc.).
            - use_beam_search (bool): Whether to use beam search for text generation.

    Returns:
        Dict[str, float]: A dictionary containing the average value of each evaluation metric computed on the dataset.
    """

    # Set global seed
    seed_everything(args.seed)

    ds = get_splits_for_evaluation(args.annotations_files, args.img_dirs, args.splits, not args.no_splits)

    # Initialize metrics dict and start evaluation
    avg_metrics = {metric: 0.0 for metric in args.metrics}
    no_meteor_count = 0
    progress_bar = tqdm(range(0, len(ds)), desc=f"Evaluating model, current metrics: {avg_metrics}")
    for i in progress_bar:
        img = preprocessor(images=ds[i][IMAGE_FIELD], truncation=True, padding="max_length", max_length=CLIP_MAX_LENGTH,
                           return_tensors="pt")[IMAGE_FIELD].to(model.device)
        reference_captions = ds[i][RAW_CAPTION_FIELD]

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
                no_meteor_count += 1
                print(f"Meteor could not be computed due to error {e.with_traceback(None)} "
                      f"on the couple: ({preds}, {reference_captions}). "
                      f"Increasing the no_meteor_count to {no_meteor_count}")

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

    if args.no_splits is False and len(args.splits) != len(args.annotations_files):
        raise Exception("The number of splits must match the number of annotations files")

    print(f"Loading checkpoint: {args.model_pth} and processor: {args.processor}")

    model = CLIPCapWrapper.load_from_checkpoint(args.model_pth)
    preprocessor = CLIPProcessor.from_pretrained(args.processor)

    avg_metrics = eval_model(model=model, preprocessor=preprocessor, args=args)
    metrics_str = ""

    for metric, value in avg_metrics.items():
        metrics_str += "{:s}\t{:.3f}\t".format(metric, value)

    with open(os.path.join(args.scores_dir, args.scores_file), "a") as msf:
        msf.write("{:s}\t{:s}\n".format(get_model_basename(args.model_pth), metrics_str))

    print("Evaluation COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores_dir", type=str, default=os.path.join(os.getcwd(), "eval_results"))
    parser.add_argument("--scores_file", type=str, default="clip_cap_scores.tsv")
    parser.add_argument("--model_pth", type=str, help="Path of the model to evaluate", required=True)
    parser.add_argument("--processor", type=str, default="openai/clip-vit-base-patch32",
                        help="Processor from CLIPProcessor.from_pretrained to preprocess data")
    parser.add_argument("--use_beam_search", default=False, action="store_true")
    parser.add_argument("--metrics", nargs='*',
                        default=[METEOR, ROUGE_L, SBERT_SIM, f'{BLEU}1', f'{BLEU}2', f'{BLEU}3', f'{BLEU}4'],
                        help='the metrics to use during evaluation')
    parser.add_argument("--no_splits", default=False, action="store_true")
    parser.add_argument("--export_captions", default=False, action="store_true")
    parser.add_argument("--annotations_files", nargs='*',
                        default=["./data/RSICD/dataset_rsicd.json", "./data/UCMD/dataset_ucmd.json",
                                 "./data/RSITMD/dataset_rsitmd.json", "./data/NAIS/dataset_nais.json"])
    parser.add_argument("--img_dirs", nargs='*',
                        default=["./data/RSICD/RSICD_images", "./data/UCMD/UCMD_images", "./data/RSITMD/RSITMD_images",
                                 "./data/NAIS/NAIS_images"])
    parser.add_argument("--splits", nargs='*',
                        default=["val", "test", "test", "test"])

    main(parser.parse_args())
