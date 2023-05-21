import torch

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

from models import CustomCLIPWrapper
from datasets.rsicd import RSICD


def main(hparams):
    img_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = 512

    trainer = Trainer(precision=hparams.precision, accelerator=hparams.accelerator, devices=hparams.devices,
                      max_epochs=hparams.max_epochs)
    dm = RSICD("./data/RSICD/dataset_rsicd.json", "./data/RSICD/RSICD_images")
    model = CustomCLIPWrapper(model_name=hparams.model_name, image_encoder=img_encoder, text_encoder=txt_encoder,
                              minibatch_size=hparams.minibatch_size, avg_word_embs=True)
    trainer.fit(model, train_dataloaders=dm)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default="RN50")
    parser.add_argument('--minibatch_size', type=int, default=512)
    parser.add_argument('--devices', type=int, default=-1)
    parser.add_argument('--accelerator', type=str, default="cuda")
    parser.add_argument('--max_epochs', type=int, default=32)
    parser.add_argument('--precision', type=int, default=32)

    args = parser.parse_args()

    main(args)
