import torch

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
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
        hparams.batch_size = 512

    seed_everything(hparams.seed, workers=True)

    trainer = Trainer(precision=hparams.precision, accelerator=hparams.accelerator, devices=hparams.devices,
                      max_epochs=hparams.max_epochs)

    dm = CaptioningDataModule(annotations_file=hparams.annotations_file, img_dir=hparams.img_dir,
                              custom_tokenizer=tokenizer, batch_size=hparams.batch_size,
                              num_workers=hparams.num_workers, shuffle=hparams.shuffle)

    model = CustomCLIPWrapper(model_name=hparams.model_name, image_encoder=img_encoder, text_encoder=txt_encoder,
                              minibatch_size=hparams.batch_size, kl_coeff=hparams.kl_coeff,
                              learning_rate=hparams.learning_rate, avg_word_embs=True)

    trainer.fit(model, train_dataloaders=dm)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default='RN50')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--kl_coeff', type=float, default=1.0)
    parser.add_argument('--devices', type=int, default=-1)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--max_epochs', type=int, default=32)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--annotations_file', type=str, required=True, help='annotation file of your training data')
    parser.add_argument('--img_dir', type=str, required=True, help='image directory of your training data')
    parser.add_argument('--train_split', type=float, default=80, help='percentage split for the training set')
    parser.add_argument('--val_split', type=float, default=10, help='percentage split for the validation set')
    parser.add_argument('--batch_size', type=int, default=512, help='size of the batch')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    main(args)
