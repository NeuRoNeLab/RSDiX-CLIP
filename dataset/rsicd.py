import os
import random

import pandas as pd
import torch
import spacy
import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset
from torchvision.io import read_image

from .constants import DEFAULT_TRANSFORMS, IMAGE_FIELD, CAPTION_FIELD


class RSICD(Dataset):
    """ Remote Sensing Image Captioning Dataset. """

    def __init__(self, annotations_file, img_dir, img_transform=None, target_transform=None, bert_text_augmentation=True):
        """
            Arguments:
                annotations_file (string): Path to the json file containing the annotations.
                img_dir (string): Directory with all the images.
                img_transform (callable, optional): Optional transform to be applied on an image in order to
                    perform data augmentation. If None, random transformations will be applied.
                target_transform (callable, optional): Optional transform to be applied on a caption.
                bert_text_augmentation (bool): Boolean to check if to use bert text augmentation or not.
                    True by default implies that bert will be used to augment text data.
        """
        self.img_captions = pd.read_json(annotations_file)
        self.img_dir = img_dir

        if img_transform:
            self.img_transform = img_transform
        else:
            self.img_transform = DEFAULT_TRANSFORMS

        self.target_transform = target_transform
        self.bert_text_augmentation = bert_text_augmentation
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

        if self.bert_text_augmentation:
            # apply synonym replacement
            nlp = spacy.load("en_core_web_lg")
            caption = augment_text(data=caption, model_path="bert-base-uncased", action="substitute", aug_min=0,
                                   aug_max=len(sentence["tokens"]), stopwords=nlp.Defaults.stop_words,
                                   device=self.device)[0]
            # augment data with bert insert action
            caption = augment_text(data=caption, model_path="bert-base-uncased", action="insert", aug_min=0,
                                   aug_max=4, stopwords=nlp.Defaults.stop_words,
                                   device=self.device)[0]

        if self.target_transform:
            caption = self.target_transform(caption)

        return {IMAGE_FIELD: image, CAPTION_FIELD: caption}


def augment_text(data, model_path, action, aug_min, aug_max, stopwords, device='cpu'):
    aug = naw.ContextualWordEmbsAug(model_path=model_path, action=action, aug_min=aug_min, aug_max=aug_max,
                                    stopwords=stopwords, device=device)

    return aug.augment(data)
