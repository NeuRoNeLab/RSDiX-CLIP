import json
import os

from aac_metrics.functional import sbert_sim, rouge_l, bleu, meteor
from typing import List, Union

from datasets import CaptioningDataModule, CaptioningDataset
from utils import IMAGE_FIELD, RAW_CAPTION_FIELD, METEOR, SBERT_SIM, ROUGE_L, BLEU, MAX_BLEU, MIN_BLEU


def get_model_basename(model):
    """
    Extracts the base name of a model file.

    Args:
        model (str): The file path to the mode

    Returns:
        str: The base name of the model file.
    """
    return '.'.join(model.split(os.sep)[-1].split(".")[:-1])


def get_eval_images(annotations_file: str) -> List[str]:
    """Returns all the images in the dataset which name follows the pattern: 'class_number.ext'
        Example:
            - airport_344.jpg will be taken into consideration
            - 0001.jpg will be ignored"""
    print("Retrieving evaluation images...")

    annotations_file_ext = annotations_file.split(".")[-1]
    if annotations_file_ext != "json":
        raise Exception(f"annotations_file type: '{annotations_file_ext}' not supported. JSON only is supported.")

    with open(annotations_file, "r") as f:
        data = json.loads(f.read())

    eval_images = [image["filename"] for image in data["images"] if image["split"] == "test"]
    print(f"Retrieved {len(eval_images)} images")
    return eval_images


def get_classes(imgs_dir: str) -> List[str]:
    """Returns all the classes contained in the dataset"""

    print("Retrieving classes...")
    class_names = set()

    # Iterate over the directory and its subdirectories
    for root, dirs, files in os.walk(imgs_dir):
        if dirs:
            # Use subdirectory names instead of filenames
            class_names.update(name.lower() for name in dirs)
            break
        else:
            for filename in files:
                if filename.find("_") > -1:
                    class_names.add(filename.lower().split("_")[0])

    class_names = sorted(list(class_names))

    print(f"Retrieved {len(class_names)} classes")
    return class_names


def get_split_images(ds, ds_indices, sets):
    """
    Retrieve images and captions from a dataset based on indices and append them to a list of sets.

    Args:
        ds: The dataset.
        ds_indices: Indices to retrieve data from the dataset.
        sets: List of sets to which images and captions will be appended.
    """
    for idx in ds_indices:
        row = ds.img_captions.iloc[idx, 0]
        img = ds[idx][0]
        captions = [[sentence['raw'] for sentence in row["sentences"]]]
        sets.append({"filename": os.path.join(ds.img_dir, row["filename"]),
                     IMAGE_FIELD: img, RAW_CAPTION_FIELD: captions})


def get_splits_for_evaluation(annotations_files: Union[str, List[str]], img_dirs: Union[str, List[str]],
                              splits: Union[str, List[str]], use_splits: bool):
    """
    Get data splits for evaluation based on annotations files, image directories, and specified splits.

    Args:
        annotations_files (Union[str, List[str]]): Paths to annotations files.
        img_dirs (Union[str, List[str]]): Paths to image directories.
        splits (Union[str, List[str]]): Splits to retrieve data for (e.g., "val" or "test").
        use_splits (bool): Whether to use the splits or not.

    Returns:
        List[dict]: List of sets containing image and caption data for evaluation.
    """
    datamodule = CaptioningDataModule(annotations_files=annotations_files, img_dirs=img_dirs) if use_splits else \
        CaptioningDataModule(annotations_files=annotations_files, img_dirs=img_dirs, train_split_percentage=0,
                             val_split_percentage=0)  # load everything into the test dataloader if use_splits is False

    dataloaders = []
    if use_splits and isinstance(splits, list):
        if "val" in splits:
            datamodule.setup("fit")
            dataloaders.append(datamodule.val_dataloader())
        if "test" in splits:
            datamodule.setup("test")
            dataloaders.append(datamodule.test_dataloader())
    elif use_splits:
        if splits == "val":
            datamodule.setup("fit")
            dataloaders.append(datamodule.val_dataloader())
        else:
            datamodule.setup("test")
            dataloaders.append(datamodule.test_dataloader())
    else:
        datamodule.setup("test")
        dataloaders = [datamodule.test_dataloader()]

    if isinstance(annotations_files, list):
        datasets = [CaptioningDataset(annotations_file=annotations_files[i], img_dir=img_dirs[i])
                    for i in range(len(annotations_files))]
    else:
        datasets = [CaptioningDataset(annotations_file=annotations_files, img_dir=img_dirs)]

    datasets_indices = []
    if use_splits and isinstance(splits, list):
        for idx, split in enumerate(splits):
            dataloader_idx = 0 if split == "val" else 1
            dataloader_idx = dataloader_idx if len(dataloaders) > dataloader_idx else dataloader_idx - 1
            datasets_indices.append(dataloaders[dataloader_idx].dataset.datasets[idx].indices)
    else:
        if use_splits:
            datasets_indices.append(dataloaders[0].dataset.indices)
        else:
            datasets_indices.append(dataloaders[0].dataset.indices)

    val_sets = []

    for _ in range(len(datasets_indices)):
        get_split_images(datasets[_], datasets_indices[_], val_sets)

    return val_sets


def compute_captioning_metrics(preds: list[str], reference_captions: list[list[str]], avg_metrics: dict,
                               i: int, no_meteor_count: int):
    if METEOR in avg_metrics:
        try:
            value, _ = meteor(candidates=preds, mult_references=reference_captions)
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

    return avg_metrics
