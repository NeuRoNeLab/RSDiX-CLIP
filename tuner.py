import os
from argparse import ArgumentParser

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner

from datasets import CaptioningDataModule
from models import CLIPWrapper


def main(args):
    # suppress hugging face tokenizers warning
    if args.num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = '1'

    model = CLIPWrapper(batch_size=args.batch_size)
    datamodule = CaptioningDataModule(
        ["./data/dataset_rsicd.json", "./data/dataset_ucmd.json", "./data/dataset_rsitmd.json",
         "./data/dataset_nais.json"], ["./data/images", "./data/images",
                                       "./data/images", "./data/images"], batch_size=args.batch_size,
        num_workers=args.num_workers)
    trainer = Trainer(default_root_dir=args.default_root_dir, max_epochs=5, log_every_n_steps=1)

    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    # Pick point based on plot, or get suggestion
    lr_finder = tuner.lr_find(model, datamodule=datamodule)
    optimal_lr = lr_finder.suggestion()

    # save to  file
    with open(f"{args.results_file}", "w") as f:
        f.write(f"optimal_lr: {optimal_lr}\n")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--default_root_dir", type=str,
                        default=os.path.join(os.getcwd(), "lightning_logs/tuner_checkpoints"),
                        help="Trainer's default root dir. The directory where the tuner's checkpoints will be saved")
    parser.add_argument("--results_file", type=str, default="results.txt",
                        help="File where tuner's results will be saved")
    parser.add_argument("--mode", type=str, default="power",
                        help="Batch Size Finder mode. Supported 'power' and 'binsearch'")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)

    main(parser.parse_args())
