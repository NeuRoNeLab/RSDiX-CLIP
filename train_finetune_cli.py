from lightning.pytorch.cli import LightningArgumentParser
from lightning import Trainer  # noqa: F401

from datasets import CaptioningDataModule  # noqa: F401
from models import CLIPWrapper
from utils import RemoteSensingLightningCLI


def cli_main(parser):
    cli = RemoteSensingLightningCLI(model_class=CLIPWrapper, datamodule_class=CaptioningDataModule)
    cli.add_arguments_to_parser(parser)


if __name__ == "__main__":
    cli_main(LightningArgumentParser())
