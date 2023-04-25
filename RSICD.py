import os
import random

import pandas as pd
import torch
import spacy
import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset
from torchvision import transforms as t
from torchvision.io import read_image


class RSICD(Dataset):
    """ Remote Sensing Image Captioning Dataset. """

    def __init__(self, annotations_file, img_dir, img_transform=None, caption_transform=None,
                 transform=None, target_transform=None):
        """
            Arguments:
                annotations_file (string): Path to the json file containing the annotations.
                img_dir (string): Directory with all the images.
                img_transform (callable, optional): Optional transform to be applied on an image in order to
                    perform data augmentation. If None, random transformations will be applied.
                caption_transform (callable, optional): Optional transform to be applied on a caption in order to
                    perform data augmentation. If None, synonym replacement will be applied.
                transform (callable, optional): Optional transform to be applied on an image.
                target_transform (callable, optional): Optional transform to be applied on a caption.
        """
        self.img_captions = pd.read_json(annotations_file)
        self.img_dir = img_dir

        if img_transform:
            self.img_transform = img_transform
        else:
            self.img_transform = t.RandomApply(torch.nn.ModuleList(
                [t.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3.85)),
                 t.RandomRotation(degrees=(0, 180)), t.RandomHorizontalFlip(p=0.5), t.RandomVerticalFlip(p=0.5)]))

        self.caption_transform = caption_transform
        self.transform = transform
        self.target_transform = target_transform

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.img_captions)

    def __getitem__(self, idx):
        """
            Returns a dictionary containing the image and the caption.
            Arguments:
                idx (int, Tensor): The index of the item to return.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.img_captions.iloc[idx, 0]
        img_name = os.path.join(self.img_dir, row['filename'])
        image = read_image(img_name).to(self.device)
        # get a random sentence from the five sentences associated to each image
        sentence = row["sentences"][random.randint(0, len(row["sentences"]) - 1)]
        caption = sentence["raw"]

        image = self.img_transform(image)
        if self.transform:
            image = self.transform(image)

        if self.caption_transform:
            caption = self.caption_transform(caption)
        else:
            # apply synonym replacement
            nlp = spacy.load("en_core_web_lg")
            aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="substitute", aug_min=0,
                                            aug_max=len(sentence["tokens"]), stopwords=nlp.Defaults.stop_words,
                                            device=self.device)
            caption = aug.augment(caption)[0]

        if self.target_transform:
            caption = self.target_transform(caption)

        return {"image": image, "caption": caption}
