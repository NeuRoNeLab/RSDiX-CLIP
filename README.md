<p align="center">
    <img width="550" src="https://github.com/angelonazzaro/remote-sensing-captioning-transformer/assets/58223071/c4c67028-9474-4ebb-9670-f001b2f207f6" alt="NeuRoNeLab logo">
</p>
<h3 align="center">
 Remote Sensing Captioning Transformer
</h3>
<p align="center">
 A Transformer-based remote sensing image captioning codebase.
</p>
<p align="center">
 <a href="#"><img src="https://img.shields.io/github/contributors/NeuRoNeLab/remote-sensing-captioning-transformer?style=for-the-badge" alt="Contributors"/></a>
 <img src="https://img.shields.io/github/last-commit/NeuRoNeLab/remote-sensing-captioning-transformer?style=for-the-badge" alt="last commit">
</p>
<p align="center">
 <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs Welcome"/></a>
 <a href="#"><img src="https://img.shields.io/github/languages/top/NeuRoNeLab/remote-sensing-captioning-transformer?style=for-the-badge" alt="Languages"/></a>
</p>

<br>

# Remote Sensing Captioning Transformer
Welcome to the Remote Sensing Captioning Transformer project! This repository hosts a Transformer-based solution for generating captions for remote sensing images.
Specifically, this project utilizes two models: 
1. A fine-tuned [CLIP transformer model](https://huggingface.co/transformers/model_doc/clip.html#transformers.CLIPModel)
2. A fine-tuned CLIP-based captioning model similar to CLIPCap[^4] that integrates the fine-tuned CLIP transformer model's vision encoder. 

This README provides a concise overview of the training and evaluation procedures, along with detailed instructions for installing project prerequisites and executing the embedding extraction script, training script, and evaluation/inference scripts.

# Table of Contents
- [Training details](#training-details)
  - [Datasets](#datasets)
  - [Augmentation Strategy](#augmentation-strategy)
  - [Evaluation Results](#evaluation-results)
      - [CLIP Evaluation Results](#clip-evaluation-results)
      - [CLIPCap Evaluation Results](#clipcap-evaluation-results)
- [Installation Guide](#installation-guide)
  - [Installation Python](#installing-python)
  - [Creating the Virtual Environment](#creating-the-virtual-environment)
  - [Installing Requirements](#installing-requirements)
  - [Cloning the Repository](#cloning-the-repository)
- [Models, weights and additional inference data](#models-weights-and-additional-inference-data)
- [Training and fine-tuning](#training-and-fine-tuning)
  - [Training and fine-tuning CLIP](#training-and-fine-tuning-clip)
  - [Training and fine-tuning CLIPCap](#training-and-fine-tuning-clipcap)
  - [Running Bayesian Optimization](#running-bayesian-optimization)
- [Evaluating](#evaluating)
  - [Evaluating CLIP](#evaluating-clip)
  - [Evaluating CLIPCap](#evaluating-clipcap)
- [Inference](#inference)
  - [Running the CLIP Remote Sensing Inference Script](#running-the-clip-remote-sensing-inference-script)
  - [Running the CLIPCap Remote Sensing Inference Script](#running-the-clipcap-remote-sensing-inference-script) 
- [Acknowledgements and references](#acknowledgements-and-references)

# Training details 
Both models were training using [Lightning](https://lightning.ai/) and [Pytorch](https://pytorch.org/) on GPU. More specifically, the models were training on an **NVIDIA Geforce RTX 4090** with Tensor Cores on `cuda 11.8`. The training process involved fine-tuning the models with remote sensing image and text data, using advanced techniques like **Bayesian Optimization** for hyperparameter tuning. 
## Datasets
The following datasets were used in the training process: 
1. [The Remote Sensing Image Captioning Dataset (RSICD)](https://github.com/201528014227051/RSICD_optimal)
2. [UC Merced Land Use Dataset (UCMD)](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
3. [The Remote Sensing Image-Text Match Dataset (RSITMD)](https://github.com/xiaoyuan1996/AMFMN)
4. A custom dataset containing 1533 images and five captions per image. 

To ensure consistency, all images were resized to 224x224 pixels. The first model utilized image-caption pairs, while the second model employed a single image and a list of five captions.

## Augmentation Strategy
To enhance the generalization of our models and mitigate the risk of overfitting, we employed both image and text augmentation techniques, following a strategy similar to the approach used by the arampacha team[^2]. 

For image augmentation, we applied transformations directly within PyTorch's Torchvision package. These transformations included: **RandomRotation**, **RandomVerticalFlip**, **RandomHorizontalFlip**, **ColorJitter**, **RandomResizedCrop** and a customized implementation of **AdjustSharpness**.

Text augmentation, on the other hand, was carried out offline through a process called backtranslation, utilizing  the [Marian MT](https://huggingface.co/transformers/model_doc/marian.html) family of translation models, specifically the [ROMANCE models from Helsinki-NLP](https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE). Each augmentation corresponded to backtranslation through a different pair of language models.

All data augmentation operations were applied probabilistically using a binomial distribution.

## Evaluation Results

### CLIP Evaluation Results 

For evaluating the CLIP model we used the same strategy as the arampacha team[^2],  which is detailed as follows:

We used a subset of the RSICD test set with file names that specified that the image belonged to one of 30 image categories. Evaluation was done by comparing the CLIP encoding of each image with CLIP encodings of each of 30 synthetic caption sentences of the form `"An aerial photograph of {category}"`. Categories corresponding to captions with the top k scores (for k=1, 3, 5, and 10) were compared with the "label" category indicated by the image name. The score is 1 if the top-k predicted classes contained the label category (for k=1, 3, 5, and 10), otherwise the score is 0. The scores are averaged over the entire set of evaluation images and reported for various values of k, as shown below.


| Model-name                                                                                             | k=1       | k=3       | k=5       | k=10      |
|--------------------------------------------------------------------------------------------------------|-----------|-----------|-----------|-----------|
| clip-all-datasets-epoch=16-val_loss=1.27                                                               | **0.908** | **0.985** | **0.993** | **0.999** |

### CLIPCap Evaluation Results
The CLIPCap model was evaluated by using the following metrics made available by [aac-metrics](https://github.com/Labbeti/aac-metrics):

| Metric    | Origin              | Range   | Short description                                 |
|:----------|:--------------------|:--------|:--------------------------------------------------|
| BLEU      | machine translation | [0, 1]  | Precision of n-grams                              |
| ROUGE-L   | machine translation | [0, 1]  | FScore of the longest common subsequence          |
| SBERT-SIM | audio captioning    | [-1, 1] | Cosine-similarity of **Sentence-BERT embeddings** |

Initially, the model was first evaluated on each dataset individually and subsequently on a collective evaluation encompassing all datasets.  The average value in the following table represent the **moving average** of each metric: 

| Model-name                                                | Metric    | Range   | Average Value |
|:----------------------------------------------------------|:----------|:--------|:--------------|
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsicd-only  | BLEU_1    | [0, 1]  | 0.628         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsicd-only  | BLEU_2    | [0, 1]  | 0.527         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsicd-only  | BLEU_3    | [0, 1]  | 0.452         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsicd-only  | BLEU_4    | [0, 1]  | 0.375         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsicd-only  | ROUGE-L   | [0, 1]  | 0.590         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsicd-only  | SBERT-SIM | [-1, 1] | 0.794         |
|                                                           |           |         |               |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-ucmd-only   | BLEU_1    | [0, 1]  | 0.767         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-ucmd-only   | BLEU_2    | [0, 1]  | 0.731         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-ucmd-only   | BLEU_3    | [0, 1]  | 0.688         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-ucmd-only   | BLEU_4    | [0, 1]  | 0.629         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-ucmd-only   | ROUGE-L   | [0, 1]  | 0.750         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-ucmd-only   | SBERT-SIM | [-1, 1] | 0.809         |
|                                                           |           |         |               |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsitmd-only | BLEU_1    | [0, 1]  | 0.451         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsitmd-only | BLEU_2    | [0, 1]  | 0.257         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsitmd-only | BLEU_3    | [0, 1]  | 0.147         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsitmd-only | BLEU_4    | [0, 1]  | 0.091         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsitmd-only | ROUGE-L   | [0, 1]  | 0.351         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428-rsitmd-only | SBERT-SIM | [-1, 1] | 0.600         |
|                                                           |           |         |               |
| clip-cap-all-datasets-epoch=18-val_loss=1.428             | BLEU_1    | [0, 1]  | 0.558         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428             | BLEU_2    | [0, 1]  | 0.447         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428             | BLEU_3    | [0, 1]  | 0.369         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428             | BLEU_4    | [0, 1]  | 0.301         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428             | ROUGE-L   | [0, 1]  | 0.513         |
| clip-cap-all-datasets-epoch=18-val_loss=1.428             | SBERT-SIM | [-1, 1] | 0.707         |

You can access the trained model weights and their corresponding configurations at the [Models, weights and additional inference data](#models-weights-and-additional-inference-data) section.

# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.9` or higher.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).
## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 
## Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```
3. After installing the requirements, you need to install additional dependencies using the following command to install the captioning metrics (only necessary if you are going to use the CLIPCap evaluation script, skip this if you are interested in inference or CLIP only):
```shell
aac-metrics-download
```
## Cloning the Repository 
To clone this repository, first navigate to your project directory:
```shell 
cd <project-directory>
```
Once you are inside your project directory, extract the `.zip` project files you downloaded using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/NeuRoNeLab/remote-sensing-captioning-transformer.git
```

# Models, weights and additional inference data
Trained model weights and additional inference data/results can be found at the following MEGA link:

   - https://mega.nz/folder/VjtDiDbK#syVkf1wF6T3L3ODtZ0jYUQ

Specifically:
   
   - The `CLIP-best-model` directory contains the weights and configuration file to load the CLIP Remote Sensing 
     model.

   - The `CLIPCap-best-model` contains the weights and configuration file to load the CLIP Remote Sensing 
     model. It also contains the `_inferenceimages` directory, which holds the captioning inference results in JSON format.

   - The `RSICD-test-set-file` directory contains the RSICD annotations file containing only the images of the test set of RSICD.

# Training and fine-tuning 
To train and finetune the models, please check out the sections below. 
## Training and fine-tuning CLIP
To train and finetune CLIP, follow the instructions below: 

1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Run `train_finetune_clip.py` file with the desired parameters. This script will train the CLIP model using the specified parameters. The parameters are mainly classified into **model parameters** and **data parameters**.
   1. Here are the available **model parameters**:
      
      -  `--model.model (str)`: The pre-trained CLIP model to use. Defaults to "openai/clip-vit-base-patch32".
      -  `--model.lr (Optional[float])`: The learning rate for the optimizer. If not provided, it attempts to use the
                    learning rate from the model's configuration.
      -  `--model.alpha (float)`: Trade-off factor between the contrastive loss and self-distillation loss.
                    Defaults to 0.5 for equal contributions from both losses.
      -  `--model.ema_decay (float)`: Exponential Moving Average (EMA) decay factor for the teacher model. Controls the
                    adaptation rate of the teacher model.
      -  `--model.weight_decay (float)`: Weight decay applied to model parameters during optimization.
      -  `--model.start_factor (float)`: Starting factor for the learning rate schedule during linear warm-up.
      -  `--model.end_factor (float)`: Ending factor for the learning rate schedule during linear warm-up.
      -  `--model.total_iters (int)`: Total number of iterations over which linear warm-up is applied.
      -  `--model.use_warmup (str)`: Specifies whether to use warm-up for learning rate scheduling.
                    Choose between "cosine" or "linear."
      -  `--model.warmup_steps (int)`: Number of warm-up steps.
      -  `--model.eps (float)`: A small epsilon value added to prevent division by zero when normalizing embeddings.
      -  `--model.betas (tuple[float, float])`: Beta coefficients for the Adam optimizer.
                    Control exponential moving averages of gradient and squared gradient.
      -  `--model.sinkhorn_lambda (float)`: Parameter used in Sinkhorn distance computation for self-distillation.
      -  `--model.sinkhorn_iter (int)`: Number of iterations for Sinkhorn distance computation.
      -  `--model.ii_coeff (float)`: Coefficient used in computing teacher targets for self-distillation.
      -  `--model.tt_coeff (float)`: Coefficient used in computing teacher targets for self-distillation.
      -  `--model.remove_diag (bool)`: Flag to determine whether to remove diagonal elements when computing teacher
                    targets.
   2. Here are the available **data parameters**:
      
      - `--data.annotations_files (Union[str, List[str]])`: Path or Paths to the file(s) containing the annotations.
      - `--data.img_dirs (Union[str, List[str]])`: Directory or Directories with all the images.
      - `--data.additional_test_annotation_files (Optional[List[Optional[str]]])`: Optional list of paths to additional
                test annotation files. Defaults to None.
      - `--data.img_transform (callable, optional)`: Optional transforms to be applied on an image for data augmentation.
                If None, random transformations will be applied. Defaults to None.
      - `--data.target_transform (callable, optional)`: Optional transforms to be applied on a caption. Defaults to None.
      - `--data.train_split_percentage (float)`: The training set split percentage. If smaller than 100, the remaining
                will be divided between the validation and test set. Defaults to TRAIN_SPLIT_PERCENTAGE.
      - `--data.val_split_percentage (float)`: The validation set split percentage. If train_split + val_split is smaller
                than 100, the remaining will be used to split the train set. Defaults to VAL_SPLIT_PERCENTAGE.
      - `--data.batch_size (int)`: The batch size of each dataloader. Defaults to BATCH_SIZE.
      - `--data.num_workers (int,` optional): The number of workers in the DataLoader. Defaults to 0.
      - `--data.augment_image_data (bool)`: Whether to apply transforms to augment image data. Defaults to False.
      - `--data.augment_text_data (bool)`: Whether to apply transforms to augment text data. Defaults to False.
      - `--data.shuffle (bool, optional)`: Whether to have shuffling behavior during sampling. Defaults to False.
      - `--data.processor (str)`: The CLIPProcessor to preprocess the batches. Defaults to None.
      - `--data.use_gpt2_tokenizer (bool)`: Whether to use GPT2-Tokenizer for tokenization. True if training ClipCap.
4. Here is an example command to run this script:
```shell
    python train_finetune_clip.py fit --data.annotations_file data/RSICD/dataset_rsicd.json --data.img_dirs data/RSICD/RSICD_images --model.lr 1e-05 --model.weight_decay 0.01
```
4. Alternatively, you can modify the parameters values in the `clip_config.yaml` and run the following command:
```shell 
   python train_finetune_clip.py fit --config clip_config.yaml 
```
5. To configure the callable parameters, you must specify the class path either through the CLI or within the `clip_config.yaml` file. For example: 
```shell 
  python train_finetune_clip.py fit --config clip_config.yaml --data.img_transform torchvision.transforms.Pad
```
6. For major information, please refer to [Configure hyperparameters from the CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).
## Training and fine-tuning CLIPCap 
To train and finetune CLIP, follow the instructions below: 

1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Run `train_finetune_clip.py` file with the desired parameters. This script will train the CLIP model using the specified parameters. The parameters are mainly classified into **model parameters** and **data parameters**.
   1. Here are the available **model parameters**:
      
       - `--model.prefix_length (int)`: Length of the prefix token used for text generation.
       - `--model.clip_length (Optional[int])`: Length of the CLIP context window. If None, it uses the default context length.
       - `--model.prefix_size (int)`: Size of the prefix embedding layer.
       - `--model.num_layers (int)`: Number of layers for the text-to-text (T2T) mapping.
       - `--model.mapping_type (MappingType)`: Type of the mapping function (MLP or Linear).
       - `--model.dropout_transformer (float)`: Dropout rate for the T2T transformer.
       - `--model.dropout_gpt2 (Optional[float])`: Dropout rate for the GPT2-based caption decoder.
       - `--model.clipcap_lr (float)`: Learning rate for the CLIPCap model.
       - `--model.clipcap_weight_decay (float)`: Weight decay for the CLIPCap model.
       - `--model.clipcap_warmup_steps (int)`: Number of warm-up steps for learning rate scheduling.
       - `--model.model (str)`: Pre-trained CLIP model to use.
       - `--model.lr (Optional[float])`: Learning rate for the CLIP model.
       - `--model.alpha (float)`: Alpha parameter for the Sinkhorn-Knopp algorithm.
       - `--model.ema_decay (float)`: Exponential moving average decay for model parameters.
       - `--model.weight_decay (float)`: Weight decay for the optimizer.
       - `--model.start_factor (float)`: Start factor for learning rate scheduling.
       - `--model.end_factor (float)`: End factor for learning rate scheduling.
       - `--model.total_iters (int)`: Total number of iterations for learning rate scheduling.
       - `--model.use_warmup (str)`: Warm-up strategy for the learning rate scheduler.
       - `--model.warmup_steps (int)`: Number of warm-up steps for learning rate scheduling.
       - `--model.eps (float)`: Epsilon value for numerical stability in Sinkhorn-Knopp.
       - `--model.betas (tuple[float, float])`: Beta values for the AdamW optimizer.
       - `--model.sinkhorn_lambda (float)`: Lambda parameter for the Sinkhorn-Knopp algorithm.
       - `--model.sinkhorn_iter (int)`: Number of iterations for Sinkhorn-Knopp.
       - `--model.ii_coeff (float)`: Coefficient for the image-image matching loss.
       - `--model.tt_coeff (float)`: Coefficient for the text-text matching loss.
       - `--model.remove_diag (bool)`: Whether to remove the diagonal of the similarity matrix.
       - `--model.load_from_checkpoint (bool)`: Whether to load the CLIP model from a checkpoint.
       - `--model.checkpoint_path (str)`: Path to the CLIP model checkpoint.
       - `--model.metrics (Union[`str, list]): Evaluation metrics for the model.
       - `--model.use_beam_search (bool)`: Whether to use beam search for text generation.
       - `--model.tokenizer (str)`: Pre-trained tokenizer for text generation.
       - `--model.pad_token (str)`: Token used for padding sequences. If None, the EOS token is used for padding.
       - `--model.every_n_batches (int)`: Frequency of computing evaluation metrics.
       - `--model.freeze_clip_encoder (bool)`: Whether to freeze the CLIP encoder during training.
    2. Here are the available **data parameters**:
       
       - `--data.annotations_files (Union[str, List[str]])`: Path or Paths to the file(s) containing the annotations.
       - `--data.img_dirs (Union[str, List[str]])`: Directory or Directories with all the images.
       - `--data.additional_test_annotation_files (Optional[List[Optional[str]]])`: Optional list of paths to additional
                   test annotation files. Defaults to None.
       - `--data.img_transform (callable, optional)`: Optional transforms to be applied on an image for data augmentation.
                   If None, random transformations will be applied. Defaults to None.
       - `--data.target_transform (callable, optional)`: Optional transforms to be applied on a caption. Defaults to None.
       - `--data.train_split_percentage (float)`: The training set split percentage. If smaller than 100, the remaining
                   will be divided between the validation and test set. Defaults to 80.
       - `--data.val_split_percentage (float)`: The validation set split percentage. If train_split + val_split is smaller
                   than 100, the remaining will be used to split the train set. Defaults to 10.
       - `--data.batch_size (int)`: The batch size of each dataloader. Defaults to 512.
       - `--data.num_workers (int, optional)`: The number of workers in the DataLoader. Defaults to 0.
       - `--data.augment_image_data (bool)`: Whether to apply transforms to augment image data. Defaults to False.
       - `--data.augment_text_data (bool)`: Whether to apply transforms to augment text data. Defaults to False.
       - `--data.shuffle (bool, optional)`: Whether to have shuffling behavior during sampling. Defaults to False.
       - `--data.processor (str)`: The CLIPProcessor to preprocess the batches. Defaults to None.
       - `--data.use_gpt2_tokenizer (bool)`: Whether to use GPT2-Tokenizer for tokenization. True if training ClipCap.
4. Here is an example command to run this script:
```shell
    python train_clipcap.py fit --data.annotations_file data/RSICD/dataset_rsicd.json --data.img_dirs data/RSICD/RSICD_images --model.clipcap_lr 1e-05 --model.clipcap_weight_decay 0.01
```
4. Alternatively, you can modify the parameters values in the `clip_config.yaml` and run the following command:
```shell 
   python train_clipcap.py fit --config clipcap_config.yaml 
```
5. To configure the callable parameters, you must specify the class path either through the CLI or within the `clipcap_config.yaml` file. For example: 
```shell 
  python train_clipcap.py fit --config clipcap_config.yaml --data.img_transform torchvision.transforms.Pad
```
6. For major information, please refer to [Configure hyperparameters from the CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).
## Running Bayesian Optimization
To train and finetune one of the two models leveraging bayesian optimization, follow the instructions below:

1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``
   
2. Ensure to have a `grid_file.yaml` that adheres to the structure shown in `clip_grid.yaml` and `clipcap_grid.yaml` files. Please note that the values of the parameters you want to finetune with must be **comma seprated values with no white spaces**.
   
3. Run `bayesian_optimization.py` with the desired parameters: 
    - `--default_root_dir (str)`: Path where the model's logs directory will be created. 
    - `--logs_dir (str):` Path where the model's logs will be saved. 
    - `--grid_file (str)`: Path to the file containing the hyperparameter research space
    - `--n_iter (int)`: Number of total iterations. Defaults to 10. 
    - `--n_init_points (int)`: Number of initial random configuration to test. Defaults to 5.
    - `--random_state (int)`: The random state to set. Defaults to: 42.  

4. Alternatively, you can run `grid_search.py` (or `grid_search.sh` if you are using a Linux-based system) to perform classic **grid search**. 

# Evaluating
To evaluate the models, please check out the sections below. 
## Evaluating CLIP
To evaluate the CLIP model, please:

1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Run `eval_clip.py` with the desired parameters: 
    - `--scores_dir (str)`: Path where the model's results directory will be created. Defaults to "eval_results". 
    - `--scores_file (str):` Path where the model's results will be saved. Defaults to "scores.tsv".
    - `--model_pth (str)`: Path to the model checkpoint to evaluate.
    - `--processor (str)`: CLIPProcessor to use to preprocess data. Defaults to "openai/clip-vit-base-patch32". 
    - `--annotations_file (str)`: Annotations file of the dataset to evaluate the model on.
    - `--img_dir (str)`: Directory containing the images of dataset to evaluate the model on.

## Evaluating CLIPCap
To evaluate the CLIP model, please:

1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Run `eval_clipcap.py` with the desired parameters: 
    - `--seed (int)`: The global seed to set. Defaults to 42. 
    - `--scores_dir (str)`: Path where the model's results directory will be created. Defaults to "eval_results". 
    - `--scores_file (str):` Path where the model's results will be saved. Defaults to "clip_cap_scores". 
    - `--model_pth (str)`: Path to the model checkpoint to evaluate.
    - `--processor (str)`: CLIPProcessor to use to preprocess data. Defaults to "openai/clip-vit-base-patch32". 
    - `--use_beam_search (bool)`: Whether to use beam search for caption generation. Defaults to false. 
    - `--metrics (list[str])`: The metrics to evaluate the model on. Defaults to ["meteor", "rouge_l", "sbert_sim", "bleu_1", "bleu_2", "bleu_3", "bleu_4"]
    - `--annotations_files (Union[str, List[str]])`: Path or Paths to the file(s) containing the annotations.
    - `--img_dirs (Union[str, List[str]])`: Directory or Directories with all the images.
    - `--splits (Union[str, List[str]])`: Split or splits to considerate for evaluation. Can be "val" or "test". 

# Inference

## Running the CLIP Remote Sensing Inference Script
To extract image embeddings using the CLIP Remote Sensing model, follow the instructions below:

1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Run `clip_inference.py` with the desired parameters. This script will extract image embeddings using the CLIP Remote Sensing model and the specified parameters: 

    `--annotations_file (str)`: Annotations file of the dataset to evaluate the model on.

    - `--img_dir (str)`: Directory containing the images of dataset  to evaluate the model on.

    - `--checkpoint_path (str)` : Path to the trained CLIP weights.

    - `--out_path (str)` The path to store the generated captions in JSON format. Defaults to "_inferenceimages/".

    - `--processor (str)`: Specify the CLIPProcessor model version to use for preprocessing data. Defaults to "openai/clip-vit-base-patch32".

## Running the CLIPCap Remote Sensing Inference Script
To generate captions using the CLIPCap trained model, follow the instructions below:

1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Run `clipcap_inference.py` with the desired parameters. This script will generate captions using the trained model and the specified parameters:
    - `--annotations_file (str)`: Annotations file of the dataset to evaluate the model on.

    - `--img_dir (str)`: Directory containing the images of dataset  to evaluate the model on.

    - `--checkpoint_path (str)` : Path to the trained CLIPCap weights.

    - `--out_path (str)` The path to store the generated captions in JSON format. Defaults to "_inferenceimages/".

    - `--processor (str)`: Specify the CLIPProcessor model version to use for preprocessing data. Defaults to "openai/clip-vit-base-patch32".

    - `--use_beam_search (bool)`: Specify whether to use beam search during inference (not using will use top-p sampling, which seems to work better with this model).
# Acknowledgements and references
A special thank must be delivered to the train-CLIP[^1], CLIP-rsicd[^2] and CapDec[^3] repositories contributors, which have been a fundamental building block of this project's codebase.

[^1]: [train-CLIP](https://github.com/Zasder3/train-CLIP)
[^2]: [CLIP-rsicd](https://github.com/arampacha/CLIP-rsicd)
[^3]: [CapDec repository](https://github.com/DavidHuji/CapDec)
[^4]: [CLIPCap paper](https://arxiv.org/abs/2111.09734)
