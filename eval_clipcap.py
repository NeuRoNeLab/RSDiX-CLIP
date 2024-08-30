import argparse
import json
import os

from lightning import seed_everything
from tqdm import tqdm
from transformers import CLIPProcessor

from evaluation.utils import get_model_basename, get_splits_for_evaluation, compute_captioning_metrics
from models import RSDiXClipCap
from models.clipcap import generate_caption
from utils import IMAGE_FIELD, CLIP_MAX_LENGTH, RAW_CAPTION_FIELD, ALLOWED_METRICS, load_model_checkpoint, METEOR


def export_metrics(avg_metrics, scores_dir, scores_file, model_basename):
    metrics_str = ""

    for metric, value in avg_metrics.items():
        metrics_str += "{:s}\t{:.3f}\t".format(metric, value)

    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    scores_file = os.path.join(scores_dir, scores_file)
    mode = "w" if not os.path.exists(scores_file) else "a"

    with open(scores_file, mode) as msf:
        msf.write("{:s}\t{:s}\n".format(model_basename, metrics_str))


def eval_model(model: RSDiXClipCap, preprocessor: CLIPProcessor, args):
    """
    Evaluates the performance of the CLIPCapWrapper on a given dataset.

    Args:
        model (RSDClipCap): The CLIPCapWrapper to be evaluated.
        preprocessor (CLIPProcessor): The CLIPProcessor to preprocess the image with.
        args (argparse.Namespace): The command-line arguments containing the following:
            - metrics (List[str]): List of evaluation metrics to compute (e.g., METEOR, SBERT_SIM, ROUGE_L, BLEU1,
                BLEU2, etc.).
            - use_beam_search (bool): Whether to use beam search for text generation.
    """

    # Set global seed
    seed_everything(args.seed)

    ds = get_splits_for_evaluation(args.annotations_files, args.img_dirs, args.splits, not args.no_splits)
    # Initialize metrics dict and start evaluation
    args.captions = []
    progress_bar = tqdm(range(0, len(ds)),
                        postfix=args.avg_metrics if not args.export_captions_file else None,
                        desc=f"Evaluating model, current metrics" \
                            if not args.export_captions_file and not args.no_evaluation
                        else "Generating captions")

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

        if not args.no_evaluation:
            args.avg_metrics = compute_captioning_metrics(preds=preds, reference_captions=reference_captions,
                                                          avg_metrics=args.avg_metrics, i=i)
            progress_bar.set_postfix({key: "{:.3f}".format(value) for key, value in args.avg_metrics.items() \
                                      if key != "no_meteor_count"})

        if args.export_captions_file:
            args.captions.append(
                {"filename": ds[i]["filename"], "preds": preds, "reference_captions": reference_captions})


def main(args):
    args.avg_metrics = {metric: 0.0 for metric in args.metrics}
    args.avg_metrics["no_meteor_count"] = 0

    if METEOR in args.metrics:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.import_captions_file is None:
        if args.model_pth is None:
            raise Exception("`import_captions_file` and `model_pth` can not be None at the same time. "
                            "Make sure you pass one of them")

        if not args.no_splits and len(args.splits) != len(args.annotations_files):
            raise Exception("The number of splits must match the number of annotations files")

        if args.export_captions_file and not args.export_captions_file.endswith(".json"):
            raise Exception("No `export_captions_file` was passed. Make sure you pass a valid file JSON path.")

        if not args.export_captions_file and args.no_evaluation:
            raise Exception("`export_captions_file` cannot be None while `no_evaluation` is True.")

        print("Evaluating CLIP-CAP: Starting evaluation...")

        if not args.no_splits and len(args.splits) == 1:
            args.splits = args.splits[0]

        if len(args.annotations_files) == 1:
            args.annotations_files = args.annotations_files[0]

        if len(args.img_dirs) == 1:
            args.img_dirs = args.img_dirs[0]

        print(f"Loading checkpoint: {args.model_pth} and processor: {args.processor}")

        model = load_model_checkpoint(RSDiXClipCap, args.model_pth)
        preprocessor = CLIPProcessor.from_pretrained(args.processor)
        model_basename = args.model_basename if args.model_basename is not None else get_model_basename(args.model_pth)

        eval_model(model=model, preprocessor=preprocessor, args=args)

        if not args.no_evaluation:
            export_metrics(avg_metrics=args.avg_metrics, scores_dir=args.scores_dir, scores_file=args.scores_file,
                           model_basename=model_basename)

        if args.export_captions_file:
            args.captions[len(args.captions) - 1] = {"model_basename": model_basename}

            with open(args.export_captions_file, "w") as export_file:
                json.dump(args.captions, export_file)

            print(f"Captions exported to: {args.export_captions_file}")
    else:
        print(f"Evaluating CLIP-CAP: Starting evaluation on imported file: {args.import_captions_file}...")

        with open(args.import_captions_file) as json_file:
            data = json.load(json_file)

        model_basename = data[len(data) - 1]["model_basename"]
        print(f"Model basename: {model_basename}")

        progress_bar = tqdm(range(0, len(data) - 1), desc=f"Evaluating model, current metrics",
                            postfix=args.avg_metrics)
        for i in progress_bar:
            args.avg_metrics = compute_captioning_metrics(preds=data[i]["preds"],
                                                          reference_captions=data[i]["reference_captions"],
                                                          avg_metrics=args.avg_metrics, i=i)
            progress_bar.set_postfix({key: "{:.3f}".format(value) for key, value in args.avg_metrics.items() \
                                      if key != "no_meteor_count"})

        del args.avg_metrics["no_meteor_count"]
        export_metrics(avg_metrics=args.avg_metrics, scores_dir=args.scores_dir, scores_file=args.scores_file,
                       model_basename=model_basename)

    print("Evaluation COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores_dir", type=str, default=os.path.join(os.getcwd(), "eval_results"),
                        help="Directory to store evaluation scores")
    parser.add_argument("--scores_file", type=str, default="clip_cap_scores.tsv",
                        help="Name of the scores file. It will be saved under the scores_dir")
    parser.add_argument("--model_pth", type=str, help="Path of the model to evaluate", default=None)
    parser.add_argument("--model_basename", type=str, default=None,
                        help="The model basename that will be saved along with the scores")
    parser.add_argument("--processor", type=str, default="openai/clip-vit-base-patch32",
                        help="Processor from CLIPProcessor.from_pretrained to preprocess data")
    parser.add_argument("--use_beam_search", default=False, action="store_true")
    parser.add_argument("--metrics", nargs='*',
                        default=ALLOWED_METRICS,
                        help='The metrics to use during evaluation')
    parser.add_argument("--no_splits", default=False, action="store_true")
    parser.add_argument("--no_evaluation", default=False, action="store_true")
    parser.add_argument("--export_captions_file", default=None, type=str)
    parser.add_argument("--import_captions_file", type=str, default=None)
    parser.add_argument("--annotations_files", nargs='*',
                        default=["./data/RSICD/dataset_rsicd.json", "./data/UCMD/dataset_ucmd.json",
                                 "./data/RSITMD/dataset_rsitmd.json", "./data/NAIS/dataset_nais.json",
                                 "./data/NWPU-Captions/dataset_nwpu.json"])
    parser.add_argument("--img_dirs", nargs='*',
                        default=["./data/RSICD/RSICD_images", "./data/UCMD/UCMD_images", "./data/RSITMD/RSITMD_images",
                                 "./data/NAIS/NAIS_images", "./data/NWPU-Captions/NWPU-Captions_images"])
    parser.add_argument("--splits", nargs='*',
                        default=["test", "test", "test", "test", "test"])

    main(parser.parse_args())
