from lightning import Trainer  # noqa: F401
from lightning.pytorch.cli import LightningCLI

from datasets import CaptioningDataModule
from models import CLIPCapWrapper


def cli_main():
    LightningCLI(model_class=CLIPCapWrapper, datamodule_class=CaptioningDataModule,
                 save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
