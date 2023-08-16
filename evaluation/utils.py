import json
import os

from typing import List


def get_model_basename(model):
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
