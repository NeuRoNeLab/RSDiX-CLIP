from lightning import Trainer  # noqa: F401
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from datasets import CaptioningDataModule  # noqa: F401
from models import CLIPWrapper
from utils import RemoteSensingLightningCLI


def cli_main(parser):
    cli = LightningCLI(model_class=CLIPWrapper, datamodule_class=CaptioningDataModule,
                       save_config_kwargs={"overwrite": True})
    cli.add_arguments_to_parser(parser)


if __name__ == "__main__":
    cli_main(LightningArgumentParser())
