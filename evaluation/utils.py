import json
import os

from typing import List, Union

from datasets import CaptioningDataModule, CaptioningDataset
from utils import IMAGE_FIELD, RAW_CAPTION_FIELD


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

    eval_images = [image["filename"] for image in data["images"] if image["split"] == "test" and
                   image["filename"].find("_") > 0]
    print(f"Retrieved {len(eval_images)} images")
    return eval_images


def get_classes(imgs_dir: str) -> List[str]:
    """Returns all the classes contained in the dataset"""

    print("Retrieving classes...")
    class_names = sorted(list(set([image_name.split("_")[0] for image_name in os.listdir(imgs_dir)
                                   if image_name.find("_") > -1])))
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
        sets.append({IMAGE_FIELD: img, RAW_CAPTION_FIELD: captions})


def get_splits_for_evaluation(annotations_files: Union[str, List[str]], img_dirs: Union[str, List[str]],
                              splits: Union[str, List[str]]):
    """
    Get data splits for evaluation based on annotations files, image directories, and specified splits.

    Args:
        annotations_files (Union[str, List[str]]): Paths to annotations files.
        img_dirs (Union[str, List[str]]): Paths to image directories.
        splits (Union[str, List[str]]): Splits to retrieve data for (e.g., "val" or "test").

    Returns:
        List[dict]: List of sets containing image and caption data for evaluation.
    """
    datamodule = CaptioningDataModule(annotations_files=annotations_files, img_dirs=img_dirs)
    dataloaders = []
    if isinstance(splits, list):
        if "val" in splits:
            datamodule.setup("fit")
            dataloaders.append(datamodule.val_dataloader())
        if "test" in splits:
            datamodule.setup("test")
            dataloaders.append(datamodule.test_dataloader())
    else:
        if splits == "val":
            datamodule.setup("fit")
            dataloaders.append(datamodule.val_dataloader())
        else:
            datamodule.setup("test")
            dataloaders.append(datamodule.test_dataloader())

    if isinstance(annotations_files, list):
        datasets = [CaptioningDataset(annotations_file=annotations_files[i], img_dir=img_dirs[i])
                    for i in range(len(annotations_files))]
    else:
        datasets = [CaptioningDataset(annotations_file=annotations_files, img_dir=img_dirs)]

    datasets_indices = []
    if isinstance(splits, list):
        for idx, split in enumerate(splits):
            dataloader_idx = 0 if split == "val" else 1
            datasets_indices.append(dataloaders[dataloader_idx].dataset.datasets[idx].indices)
    else:
        dataloader_idx = 0 if splits == "val" else 1
        datasets_indices.append(dataloaders[dataloader_idx].dataset.indices)

    val_sets = []

    for _ in range(len(datasets_indices)):
        get_split_images(datasets[_], datasets_indices[_], val_sets)

    return val_sets
