import torch

from datasets import CaptioningDataset
from torch.utils.data import ConcatDataset

from transformers import GPT2Tokenizer

if __name__ == "__main__":
    annotations_files = ["./data/RSICD/dataset_rsicd.json", "./data/dataset_ucmd.json", "./data/dataset_rsitmd.json",
                         "./data/dataset_nais.json"]
    imgs_dir = ["./data/images", "./data/images", "./data/images", "./data/images"]

    datasets = []
    for i in range(len(annotations_files)):
        # Create the i-th dataset
        captioning_dataset = CaptioningDataset(annotations_file=annotations_files[i],
                                               img_dir=imgs_dir[i],
                                               img_transform=None,
                                               target_transform=None,
                                               train=True)
        datasets.append(captioning_dataset)

    dataset = ConcatDataset(datasets)
    tokens = []
    all_len = []
    max_seq_len = 0
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    for sample in dataset:
        tokens.append([torch.tensor(tokenizer.encode(c), dtype=torch.int64) for c in sample[1]])
        seq_lenghts = [ct.shape[0] for ct in tokens[-1]]
        all_len.extend(seq_lenghts)
        max_seq_len = max(max_seq_len, *seq_lenghts)

    all_len = torch.tensor(all_len).float()
    max_seq_len = min(int(all_len.mean() + all_len.std() * 10), max_seq_len)

