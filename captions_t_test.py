import argparse
import copy
import json
import math
import os
import random

import nlpaug.augmenter.word as naw
import numpy as np
from lightning import seed_everything
from nltk.tokenize import RegexpTokenizer
from scipy import stats
from tqdm import tqdm

from eval_clipcap import export_metrics
from evaluation.utils import get_classes, compute_captioning_metrics
from utils import SBERT_SIM, BLEU, ROUGE_L


def main(args):
    seed_everything(args.seed)

    with open(args.annotations_file) as f:
        annotations = json.load(f)

    avg_metrics = {metric: {"mean": 0.0, "var": 0.0, "means": [], "vars": []} for metric in args.metrics}

    tokenizer = RegexpTokenizer(r'\w+')

    modified_annotations = copy.deepcopy(annotations)
    forbidden_words = get_classes(imgs_dir=args.imgs_dir)

    forbidden_words.extend(["near", "next"])

    # perform synonym replacement
    progress_bar = tqdm(range(0, len(modified_annotations["images"])), desc="Performing synonym replacement")
    for i in progress_bar:
        for sentence in modified_annotations["images"][i]["sentences"]:
            aug = naw.SynonymAug(aug_src=args.aug_src, aug_p=random.uniform(args.aug_p_min, args.aug_p_max),
                                 aug_min=args.aug_min, aug_max=args.aug_max, stopwords=forbidden_words)
            # og_sentence = copy.deepcopy(sentence)
            sentence["raw"] = aug.augment(sentence["raw"])[0]
            sentence["tokens"] = tokenizer.tokenize(sentence["raw"])

    # save modified annotations_file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    annotations_file_basename = os.path.basename(args.annotations_file)
    modified_annotations_filename = os.path.splitext(os.path.basename(annotations_file_basename))[0] + "_synonyms.json"

    with open(os.path.join(args.output_dir, modified_annotations_filename), "w") as fp:
        json.dump(modified_annotations, fp)

    progress_bar = tqdm(range(0, len(annotations["images"])), desc="Computing metrics", postfix=avg_metrics)

    for img_idx in progress_bar:
        for sentence_idx, sentence in enumerate(annotations["images"][img_idx]["sentences"]):
            avg_metrics = compute_captioning_metrics(
                preds=[modified_annotations["images"][img_idx]["sentences"][sentence_idx]["raw"]],
                reference_captions=[[sentence["raw"]]],
                avg_metrics=avg_metrics, i=sentence_idx, compute_var=True)

            progress_bar.set_postfix(
                {key: {"mean": "{:.3f}".format(avg_metrics[key]["mean"]),
                       "var": "{:.3f}".format(avg_metrics[key]["var"])} \
                 for key in avg_metrics.keys() if key != "no_meteor_count"})

    annotations_file_basename = os.path.splitext(annotations_file_basename)[0]

    means = {k: v.get("mean") for k, v in avg_metrics.items()}
    stds = {k: math.sqrt(v.get("var")) for k, v in avg_metrics.items()}

    # H0: mu_1 >= mu_2 where mu_1 is the mean of the n-gram score (n=1,2,3,4) and mu_2
    # is the mean of the sbert_sim score
    # H1: mu_1 < mu_2
    hypotheses = {metric: {"p_value": 2.0, "accepted": 0} for metric in avg_metrics if metric != SBERT_SIM}

    if args.normalize_sbert_sim:
        as_array = np.asarray(avg_metrics[SBERT_SIM]["means"])
        avg_metrics[SBERT_SIM]["means"] = list((as_array + 1) / 2)  # minmax scaling [-1, 1]

    for metric in hypotheses:
        hypotheses[metric]["p_value"] = stats.ttest_ind(avg_metrics[SBERT_SIM]["means"],
                                                        avg_metrics[metric]["means"]).pvalue
        hypotheses[metric]["accepted"] = int(hypotheses[metric]["p_value"] / 2 < args.alpha)

    export_metrics(means, args.output_dir, args.output_file,
                   annotations_file_basename + "_mean")
    export_metrics(stds, args.output_dir, args.output_file,
                   annotations_file_basename + "_std")

    with open(os.path.join(args.output_dir, annotations_file_basename + "_hypotheses.json"), "w") as fp:
        json.dump(hypotheses, fp, indent=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations_file", type=str)
    parser.add_argument("--imgs_dir", type=str)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.getcwd(), "captions_t_test_results"))
    parser.add_argument("--output_file", type=str, default="captions_t_test_results.tsv")
    parser.add_argument("--aug_src", type=str, default="wordnet")
    parser.add_argument("--aug_min", type=int, default=1)
    parser.add_argument("--aug_max", type=int, default=10)
    parser.add_argument("--aug_p_min", type=float, default=0.3)
    parser.add_argument("--aug_p_max", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--normalize_sbert_sim", default=False, action="store_true")
    parser.add_argument("--metrics", nargs='*',
                        default=[SBERT_SIM, ROUGE_L, f'{BLEU}1', f'{BLEU}2', f'{BLEU}3', f'{BLEU}4'])

    main(parser.parse_args())
