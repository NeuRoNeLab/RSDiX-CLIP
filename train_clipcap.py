import os

from lightning import Trainer  # noqa: F401
from lightning.pytorch.cli import LightningCLI

from datasets import CaptioningDataModule
from models import RSDClipCap
from utils import enable_matmul_precision


def cli_main():
    LightningCLI(model_class=RSDClipCap, datamodule_class=CaptioningDataModule,
                 save_config_kwargs={"overwrite": True, "config_filename": "clipcap_config_CLI.yaml"},
                 parser_kwargs={"fit": {"default_config_files": ["clipcap_config.yaml"]}})


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    enable_matmul_precision()
    cli_main()
