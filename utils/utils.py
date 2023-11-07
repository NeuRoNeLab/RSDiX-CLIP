import json
import os
from typing import Tuple

import torch.cuda
import xmltodict


def get_splits(n_instances: int, train_split_percentage: float, val_split_percentage: float) -> Tuple[int, int, int]:
    """
    Calculate dataset splits based on specified percentages.

    Args:
        n_instances (int): Total number of instances.
        train_split_percentage (float): Percentage of instances for the training split.
        val_split_percentage (float): Percentage of instances for the validation split.

    Returns:
        Tuple[int, int, int]: Number of instances for training, validation, and test splits.
    """
    train_split = int(n_instances * train_split_percentage / 100)
    remaining_split = n_instances - train_split
    val_split = remaining_split - int(n_instances * val_split_percentage / 100)
    test_split = remaining_split - val_split

    # If no test set is required, then test_split is just remainder, that we can add to the train
    if train_split_percentage + val_split_percentage >= 100.0:
        train_split = train_split + test_split
        test_split = 0

    return train_split, val_split, test_split


def nais_to_json(annotations_file: str, json_file_name: str = "dataset_nais"):
    """
    Convert NAIS dataset annotations from XML to JSON format.

    Args:
        annotations_file (str): Path to the XML annotations file.
        json_file_name (str): Name of the output JSON file.
    """
    with open(annotations_file) as f:
        data_dict = xmltodict.parse(f.read())

    data_dict = data_dict["annotations"]
    images = {"images": []}

    for image in data_dict["image"]:
        image_data = {"filename": image["@name"], "imgid": int(image["@id"])}
        sentences = []
        for mask in image["mask"]:
            sentences.append({"raw": mask["@label"]})
        image_data["sentences"] = sentences
        images["images"].append(image_data)

    # get annotations_file directory
    data_dir = os.path.dirname(annotations_file)
    with open(f"{data_dir}/{json_file_name}.json", "w") as f:
        json.dump(images, f, indent=4)


class ListWrapper(list):
    """
    A custom list class that supports device assignment.
    """
    def __init__(self, initial_list=None):
        """
        Initialize the ListWrapper

        Args:
            initial_list (list): Initial list to populate the object.
        """
        if initial_list is None:
            super().__init__()
        else:
            super().__init__(initial_list)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    def to(self, device):
        self._device = device
        return self


def separate_rsicd_test_images(annotations_file: str, test_output_file: str = "dataset_rsicd_test.json"):
    """
    Separate test images from RSICD dataset and create a separate JSON file for test images.

    Args:
        annotations_file (str): Path to the JSON annotations file.
        test_output_file (str): Name of the output JSON file for test images.
    """
    data = []
    with open(annotations_file) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    test_images = {"images": [], "dataset": data["dataset"]}
    new_data = {"images": [], "dataset": data["dataset"]}

    for idx, img in enumerate(data["images"]):
        if img["split"] == "test":
            test_images["images"].append(img)
        else:
            new_data["images"].append(img)

    # overwrite existing dataset
    with open(annotations_file, "w") as json_file:
        json.dump(new_data, json_file)

    with open(test_output_file, "w") as json_file:
        json.dump(test_images, json_file)


def separate_nwpu_test_images(annotations_file: str, test_output_file: str = "dataset_nwpu_test.json"):
    """
        Separate test images from NWPU-Captions dataset and create a separate JSON file for test images.

        Args:
            annotations_file (str): Path to the JSON annotations file.
            test_output_file (str): Name of the output JSON file for test images.
    """
    data = []
    with open(annotations_file, encoding="utf8") as json_file:
        data = json.load(json_file)

    train_data = {"images": [], "dataset": "NWPU-Captions"}
    test_data = {"images": [], "dataset": "NWPU-Captions"}

    # the dataset is structure as follows:
    # category:
    #   [
    #       "filename": "category_1",
    #       "split": "train",
    #       "raw": "raw sentence 1",
    #       "raw_1": "raw sentence 2",
    #       ...
    #   ]
    for category in data.keys():
        for category_row in data[category]:
            row = {
                "filename": category_row["filename"],
                "imgid": category_row["imgid"],
                "split": category_row["split"],
                "sentences": [{"raw": category_row[raw_key]} for raw_key in category_row.keys() if raw_key.startswith("raw")]
            }
            if row["split"] == "test":
                test_data["images"].append(row)
            else:
                train_data["images"].append(row)

        # overwrite existing dataset
    with open(annotations_file, "w") as json_file:
        json.dump(train_data, json_file)

    with open(test_output_file, "w") as json_file:
        json.dump(test_data, json_file)
