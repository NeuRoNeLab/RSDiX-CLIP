import json
import os
from typing import Tuple

import torch.cuda
import xmltodict


def get_splits(n_instances: int, train_split_percentage: float, val_split_percentage: float) -> Tuple[int, int, int]:
    train_split = int(n_instances * train_split_percentage / 100)
    remaining_split = n_instances - train_split
    test_split = remaining_split - int(n_instances * val_split_percentage / 100)
    val_split = remaining_split - test_split

    # If no test set is required, then test_split is just remainder, that we can add to the train
    if train_split_percentage + val_split_percentage >= 100.0:
        train_split = train_split + test_split
        test_split = 0

    return train_split, val_split, test_split


def nais_to_json(annotations_file: str, json_file_name: str = "dataset_nais"):
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
    def __init__(self, initial_list=None):
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


if __name__ == "__main__":
    # separate rsicd test images from the others
    data = []
    with open("./data/RSICD/dataset_rsicd.json") as json_file:
        for line in json_file:
            data.append(json.loads(line))
    test_images = {"images": [], "dataset": data["dataset"]}
    new_data = {"images": [], "dataset": data["dataset"]}

    for idx, image in enumerate(data["images"]):
        if image["split"] == "test":
            test_images["images"].append(image)
        else:
            new_data["images"].append(image)

    # overwrite existing dataset
    with open("./data/RSICD/dataset_rsicd.json", "w") as json_file:
        json.dump(new_data, json_file)

    with open("./data/RSICD/dataset_rsicd_test.json", "w") as json_file:
        json.dump(test_images, json_file)
