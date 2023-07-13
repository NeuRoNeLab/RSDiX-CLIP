import os

from argparse import ArgumentParser

from models import CLIPWrapper
from datasets import CaptioningDataModule

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner


def main(args):
    model = CLIPWrapper(batch_size=args.batch_size)
    datamodule = CaptioningDataModule(
        ["./data/RSICD/dataset_rsicd.json", "./data/UCMD/dataset_ucmd.json", "./data/RSITMD/dataset_rsitmd.json",
         "./data/NAIS/dataset_nais.json"], ["./data/RSICD/RSICD_images", "./data/UCMD/UCMD_images",
                                            "./data/RSITMD/RSITMD_images", "./data/NAIS/NAIS_images"],
        num_workers=args.num_workers, batch_size=args.batch_size)
    trainer = Trainer(default_root_dir=args.default_root_dir)

    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    # Pick point based on plot, or get suggestion
    optimal_lr = tuner.lr_find(model, datamodule=datamodule).suggestion()
    optimal_batch_size = tuner.scale_batch_size(model, datamodule=datamodule)

    # save to  file
    with open(f"{args.results_file}", "w") as f:
        f.write(f"optimal_lr: {optimal_lr}\n")
        f.write(f"optimal_batch_size: {optimal_batch_size}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--default_root_dir", type=str,
                        default=os.path.join(os.getcwd(), "lightning_logs/tuner_checkpoints"),
                        help="Trainer's default root dir. The directory where the tuner's checkpoints will be saved")
    parser.add_argument("--results_file", type=str, default="results.txt",
                        help="File where tuner's results will be saved")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)

    main(parser.parse_args())
