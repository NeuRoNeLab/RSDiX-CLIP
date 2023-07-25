import json
import os
import torch.cuda
import xmltodict


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
