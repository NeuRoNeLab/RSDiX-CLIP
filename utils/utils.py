import json
import os
import xmltodict

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


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


class RemoteSensingLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "es")
        parser.add_lightning_class_args(ModelCheckpoint, "model_chk")
        parser.set_defaults({
            "trainer.devices": -1, "trainer.accelerator": "cuda", "trainer.max_epochs": 32, "trainer.precision": 32,
            "es.patience": 3, "es.stopping_threshold": None, "es.divergence_threshold": None, "es.monitor": "val_loss",
            "model_chk.filename": "clip-rsicd-{epoch:02d}-{val_loss:.2f}", "model_chk.monitor": "val_loss"
        })

        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("es.mode", "model_chk.mode")
        parser.link_arguments("es.monitor", "model_chk.monitor")
