import torch

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoTokenizer, AutoModel

from models import CustomCLIPWrapper
from datasets.captioningDataset import CaptioningDataModule


def main(hparams):
    img_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if hparams.batch_size < 1:
        hparams.batch_size = 128

    seed_everything(hparams.seed, workers=True)

    early_stopping = EarlyStopping(monitor=hparams.monitor_metric, mode=hparams.mode,
                                   stopping_threshold=hparams.stopping_threshold,
                                   divergence_threshold=hparams.divergence_threshold)
    model_checkpoint = ModelCheckpoint(dirpath=hparams.dirpath, monitor=hparams.monitor_metric,
                                       save_on_train_epoch_end=hparams.save_on_train_epoch_end, mode=hparams.mode,
                                       filename=hparams.filename, save_top_k=hparams.save_top_k)

    trainer = Trainer(precision=hparams.precision,
                      accelerator=hparams.accelerator, devices=hparams.devices,
                      max_epochs=hparams.max_epochs, callbacks=[early_stopping, model_checkpoint])

    dm = CaptioningDataModule(annotations_file=hparams.annotations_file, img_dir=hparams.img_dir,
                              custom_tokenizer=tokenizer, batch_size=hparams.batch_size,
                              num_workers=hparams.num_workers, shuffle=hparams.shuffle)

    model = CustomCLIPWrapper(model_name=hparams.model_name, image_encoder=img_encoder, text_encoder=txt_encoder,
                              minibatch_size=hparams.batch_size, kl_coeff=hparams.kl_coeff,
                              learning_rate=hparams.learning_rate, avg_word_embs=True)

    trainer.fit(model, train_dataloaders=dm)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help='the integer value seed for global random state in Lightning')

    # model's arguments
    parser.add_argument('--model_name', type=str, default='RN50')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--kl_coeff', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=0)

    # trainer's arguments
    parser.add_argument('--devices', type=int, default=-1)
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--max_epochs', type=int, default=32)
    parser.add_argument('--precision', type=int, default=32)

    # datamodule's arguments
    parser.add_argument('--annotations_file', type=str, required=True, help='annotation file of your training data')
    parser.add_argument('--img_dir', type=str, required=True, help='image directory of your training data')
    parser.add_argument('--train_split', type=float, default=80, help='percentage split for the training set')
    parser.add_argument('--val_split', type=float, default=10, help='percentage split for the validation set')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')

    # early stopping arguments
    parser.add_argument('--patience', type=int, default=3,
                        help='number of checks with no improvement after which training will be stopped')
    parser.add_argument('--stopping_threshold', type=float, default=None,
                        help='Stop training immediately once the monitored quantity reaches this threshold')
    parser.add_argument('--divergence_threshold', type=float, default=None,
                        help='Stop training as soon as the monitored quantity becomes worse than this threshold')

    # checkpointing arguments
    parser.add_argument('--dirpath', type=str, default='./checkpoints/', help='directory to save the model file')
    parser.add_argument('--save_on_train_epoch_end', type=bool, default=True,
                        help='Whether to run checkpointing at the end of the training epoch')
    parser.add_argument('--filename', type=str, default='clip-rsicd-{epoch:02d}-{val_loss:.2f}',
                        help='checkpoint filename. Can contain named formatting options to be auto-filled')
    parser.add_argument('--save_top_k', type=int, default=5,
                        help='the best k models according to the quantity monitored will be saved')

    # used both in model and datamodule
    parser.add_argument('--batch_size', type=int, default=128, help='size of the batch')

    # used both for early stopping and checkpointing
    parser.add_argument('--mode', type=str, default='max',
                        help='number of checks with no improvement after which training will be stopped')
    parser.add_argument('--monitor_metric', type=str, default='val_loss', help='metric to monitor for early stopping '
                                                                               'and checkpointing')

    args = parser.parse_args()

    main(args)
