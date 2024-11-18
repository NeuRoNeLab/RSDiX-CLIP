<p align="center">
    <img width="550" src="https://github.com/angelonazzaro/remote-sensing-captioning-transformer/assets/58223071/c4c67028-9474-4ebb-9670-f001b2f207f6" alt="NeuRoNeLab logo">
</p>
<h3 align="center">
 RSDiX: Addressing Intra-Class Similarity in Remote Sensing with Self-Distillation
</h3>
<p align="center">
 Official implementation of the RSDiX: Addressing Intra-Class Similarity in Remote Sensing with Self-Distillation paper. 
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

# Table of Contents

1. [RSDiX: Addressing Intra-Class Similarity in Remote Sensing with Self-Distillation](#rsdix-addressing-intra-class-similarity-in-remote-sensing-with-self-distillation)
2. [Datasets](#datasets)
3. [Data Availability](#data-availablity)
4. [Results](#results)
   - [RSDiX-CLIP Comparison Results](#rsdix-clip-comparison-results)
   - [RSDiX-CLIPCap Comparison Results](#rsdix-clipcap-comparison-results)
   - [RSDiX-SigLIP Comparison Results](#rsdix-siglip-comparison-results)
   - [RSDiX-SigLIPCap Comparison Results](#rsdix-siglipcap-comparison-results)
   - [RSDiX-CLIP-SBERT Comparison Results](#rsdix-clip-sbert-comparison-results)
   - [RSDiX-CLIPCap-SBERT Comparison Results](#rsdix-clipcap-sbert-comparison-results)
5. [Models' weights](#models-weights)
6. [Installation Guide](#installation-guide)
   - [Installing Python](#installing-python)
   - [Cloning the Repository](#cloning-the-repository)
   - [Creating the Virtual Environment](#creating-the-virtual-environment)
   - [Installing Requirements](#installing-requirements)
7. [Training and Fine-Tuning](#training-and-fine-tuning)
   - [Training and fine-tuning RSDiX-CLIP](#training-and-fine-tuning-rsdix-clip)
   - [Training and fine-tuning RSDiX-CLIPCap](#training-and-fine-tuning-rsdix-clipcap)
   - [Running Bayesian Optimization](#running-bayesian-optimization)
8. [Evaluating](#evaluating)
   - [Evaluating RSDiX-CLIP](#evaluating-rsdix-clip)
   - [Evaluating RSDiX-CLIPCap](#evaluating-rsdix-clipcap)
9. [Inference](#inference)
   - [Running the RSDiX-CLIP Remote Sensing Inference Script](#running-the-rsdix-clip-remote-sensing-inference-script)
   - [Running the RSDiX-CLIPCap Remote Sensing Inference Script](#running-the-rsdix-clipcap-remote-sensing-inference-script)
10. [Acknowledgements and references](#acknowledgements-and-references)

# RSDiX: Addressing Intra-Class Similarity in Remote Sensing with Self-Distillation

Remote sensing (RS) imagery serves as a crucial information source for diverse applications, including environmental monitoring, urban planning, defense, and security. Despite facing challenges such as spatial/spectral, temporal variability, or lack of quality annotated data, recent years have witnessed the development of deep learning methods for RS image processing. Such models often integrate linguistic information to enrich semantic understanding, showing potential in tasks such as zero-shot classification, detection, retrieval, and captioning of satellite images, to generate labels or descriptions with limited data. However, existing methods for these tasks encounter limitations, including low-quality captions, poor linguistic variety, similar captions for different images, noisy captions, and unreliable evaluation. In this work, we try to overcome some of these limitations by combining the power of pre-trained models with advanced training techniques such as self-distillation:

1. To tackle the issue of intra-class similarity in RS image datasets, we introduce *RSDiX-CLIP*, a fine-tuned version of CLIP with an additional self-distillation objective. We propose *RSDiX-CLIPCap*, a family of captioning models that use the fine-tuned RSDiX-CLIP encoder and a transformer mapper network from CLIPCap. Our models achieve superior/competitive performance against state-of-the-art (SOTA) methods across multiple zero-shot RS image classification and captioning datasets. Furthermore, we explore the impact of mixed distillation strategies and alternative contrastive learning frameworks, introducing the *RSDiX-CLIP-S-BERT* and *RSDiX-SigLIP* families.

2. We present *Sentinel-2 Land-cover Captioning Dataset* (S2LCD), a novel RS captioning dataset with 1533 Sentinel-2 images with several land cover/use and human influence and 7665 wide-vocabulary, detailed captions.

3. We challenge, within the domain of RS images, *N*-gram-based metrics, such as the BLEU score building upon prior research to provide additional evidence of their susceptibility to inherent bias and inaccuracy. A statistical sensitivity/robustness comparison on perturbed captions is used to advocate for more reliable alternative metrics Sentence-BERT-Similarity.

## Datasets 

**RSICD**: One of the largest datasets for RSIC containing RSI collected from Google Earth, Baidu Map, MapABC, and Tianditu. It contains 10,921 remote sensing images with a fixed size of 224 x 224 pixels with various resolutions, each annotated with 5 captions, accounting for 54,605 descriptions in total. The dataset covers a wide range of scenes, objects, and scenarios and has one of the largest diversity rates amongst RSI datasets.

**UCMD**: Consists of 2,100 images belonging to 21 classes (100 images per class), and each image, measuring 256 x 256 pixels, is associated with 5 captions, containing 10,500 descriptions in total. It contains images from urban areas only with high spatial resolution.

**RSITMD**: A recently introduced dataset containing 4,743 images and 5 captions per image, presenting 23,715 total descriptions. Unlike traditional RS image-text datasets, it presents more scene changes and fine-grained captions.

**NWPU-Captions**: A recent RS dataset, comprising 31,500 256 x 256 images and 157,500 captions (5 per each image), manually annotated by experienced volunteers. It offers a substantial scale and a broad representation of intricate scenes, providing a wealth of diverse vocabulary and sentence structures.

**S2LCD**: The proposed *Sentinel-2 Land-cover Captioning Dataset* encompasses 1533 image patches (224x224 pixels) created from Sentinel-2 L2A images, ensuring diversity in land cover/use (forests, mountains, agriculture, urban areas, all with varying human influence). Each patch has 5 captions (7665 in total) with wide vocabulary (natural language and EAGLES lexicon) and attention to detail. This dataset is used in captioning experiments only due to the peculiar caption structure of some images: a few captions describe only partial image elements while others capture complementary details. This makes them less suitable for contrastive image-text losses.

## Data Availablity

**The S2LCD dataset is distributed under the [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) license. You can download the dataset from this repository in the file named "S2LCD.zip".**

## Results 

### RSDiX-CLIP Comparison Results

| Dataset      | RSDiX-CLIP (B/32)    | RSDiX-CLIP (B/16)    | RSDiX-CLIP (L/14)    | RemoteCLIP (B/32) | RemoteCLIP (L/14) | GeoRSCLIP (B/32) | GeoRSCLIP (H/14) | RS-CLIP (B/32) |
|-------------------|---------|---------|---------|-----------------------|-----------------------|-----------------------|-----------------------|---------------------|
| RSICD        | **96.00**| 94.40   | 92.90   | -                     | -                     | -                     | -                     | -                   |
| RSI-CB128    | 27.30   | 35.00   | **38.60**| 24.18                 | 37.22                 | -                     | -                     | -                   |
| RSI-CB256    | 45.90   | 45.40   | 48.30   | 39.50                 | **52.82**             | -                     | -                     | -                   |
| WHU-earth    | 65.70   | 75.20   | **78.50**| 63.12                 | 70.83                 | -                     | -                     | -                   |
| EuroSAT-RGB  | 42.60   | 51.70   | 51.10   | 35.96                 | 59.94                 | 61.49                 | **67.47**             | -                   |
| MLRSNet      | 65.20   | 71.60   | **73.50**| 59.28                 | 66.32                 | -                     | -                     | -                   |
| PatternNet   | 59.40   | 67.30   | **72.80**| 57.71                 | 68.75                 | -                     | -                     | -                   |
| RESISC45     | **93.20**| 94.60   | **95.60**| 70.33                 | 79.84                 | 71.89                 | 73.83                 | 85.44               |
| AID          | **95.10**| 92.60   | 91.10   | 91.30                 | 87.90                 | 73.72                 | 76.33                 | 79.56               |
| RSSCN7       | **78.60**| 77.40   | 77.30   | 68.57                 | 72.32                 | -                     | -                     | -                   |
| OPTIMAL-31   | 96.40   | **98.00**| 95.50   | 77.96                 | 90.05                 | -                     | -                     | -                   |
| RSC11        | 77.80   | **78.20**| 73.90   | 64.94                 | 74.90                 | -                     | -                     | -                   |
| WHU-RS19     | **98.40**| 98.30   | 96.30   | 96.12                 | 94.66                 | -                     | -                     | **99.10**           |
|**Average (sub. A):** 70.47  | 73.78  | **74.60** | 62.41  | 71.30  | -  | -  | -  | -
|**Average (sub. B):** 76.97  | **79.63**  | 79.27  | 66.19  | 75.89  | 69.03  | 69.54  | -  | -
|**Average (sub. C):** **95.57**  | 95.17  | 94.33  | 85.92  | 87.47  | -  | -  | 88.03  | -

Top-1 accuracy results comparison of our RSDiX-CLIP models with current SOTA methods. $A, B, C$ refer to subsets of datasets on which results are available for RemoteCLIP, GeoRSCLIP and RS-CLIP, respectively.

### RSDiX-CLIPCap Comparison Results

| Model | Dataset | M ↑ | SBS ↑ | S ↑ | R ↑ | B-1 ↑ | B-2 ↑ | B-3 ↑ | B-4 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VLAD-LSTM | RSICD | 0.205 | - | - | 0.433 | 0.500 | 0.320 | 0.232 | 0.178 |
|  | UCMD | 0.346 | - | - | 0.652 | 0.702 | 0.609 | 0.550 | 0.503 |
| Text-a-a | RSICD | 0.292 | - | 0.389 | 0.527 | 0.651 | 0.513 | 0.414 | 0.336 |
|  | UCMD | 0.330 | - | 0.358 | 0.627 | 0.711 | 0.625 | 0.554 | 0.497 |
| SAT (LAM-TL) | RSICD | 0.330 | - | 0.471 | 0.591 | 0.679 | 0.562 | 0.478 | 0.415 |
|  | UCMD | 0.488 | - | 0.513 | 0.793 | 0.821 | 0.786 | 0.753 | 0.723 |
| Adaptive (LAM-TL) | RSICD | 0.326 | - | 0.467 | 0.585 | 0.676 | 0.555 | 0.471 | 0.408 |
|  | UCMD | 0.510 | - | 0.535 | 0.826 | 0.857 | 0.812 | 0.775 | 0.743 |
| TCE | RSICD | 0.344 | - | - | 0.669 | 0.761 | 0.636 | 0.547 | 0.479 |
|  | UCMD | 0.478 | - | - | 0.757 | 0.821 | 0.762 | 0.714 | 0.670 |
| Word-sentence | RSICD | 0.320 | - | - | 0.626 | 0.724 | 0.586 | 0.493 | 0.425 |
|  | UCMD | 0.440 | - | - | 0.713 | 0.838 | 0.762 | 0.704 | 0.656 |
| Recurrent-Attn. | RSICD | 0.363 | - | 0.472 | 0.669 | 0.773 | 0.665 | 0.578 | 0.506 |
|  | UCMD | 0.457 | - | 0.489 | 0.807 | 0.852 | 0.793 | 0.743 | 0.698 |
| GVFGA + LSGA | RSICD | 0.329 | - | 0.468 | 0.593 | 0.678 | 0.560 | 0.478 | 0.417 |
|  | UCMD | 0.444 | - | 0.485 | 0.785 | 0.832 | 0.766 | 0.710 | 0.660 |
| SVM-D CONC | RSICD | 0.230 | - | - | 0.456 | 0.600 | 0.435 | 0.336 | 0.269 |
|  | UCMD | 0.370 | - | - | 0.688 | 0.765 | 0.695 | 0.642 | 0.594 |
| Structured-Attn. | RSICD | 0.329 | - | - | 0.571 | 0.702 | 0.561 | 0.465 | 0.393 |
|  | UCMD | 0.463 | - | - | 0.814 | 0.854 | 0.804 | 0.757 | 0.715 |
| JTTS | RSICD | 0.377 | - | 0.488 | 0.682 | 0.789 | 0.680 | 0.589 | 0.514 |
|  | UCMD | 0.491 | - | 0.523 | 0.836 | 0.870 | 0.822 | 0.779 | 0.738 |
| CSMLF | RSICD | 0.213 | - | 0.199 | 0.446 | 0.576 | 0.386 | 0.283 | 0.222 |
|  | UCMD | 0.132 | - | 0.076 | 0.393 | 0.436 | 0.273 | 0.186 | 0.121 |
|  | NWPU | 0.320 | - | 0.265 | 0.578 | 0.717 | 0.590 | 0.509 | 0.440 |
| SM-Att+LSTM | RSICD | 0.342 | - | - | 0.580 | 0.750 | 0.631 | 0.538 | 0.459 |
|  | UCMD | 0.435 | - | - | 0.763 | 0.815 | 0.759 | 0.694 | 0.647 |
|  | NWPU | 0.330 | - | 0.276 | 0.593 | 0.739 | 0.617 | 0.532 | 0.468 |
| MLCA-Net | RSICD | 0.351 | - | 0.444 | 0.638 | 0.750 | 0.631 | 0.538 | 0.459 |
|  | UCMD | 0.435 | - | 0.473 | 0.772 | 0.826 | 0.770 | 0.717 | 0.668 |
|  | NWPU | 0.337 | - | 0.285 | 0.601 | 0.745 | 0.624 | 0.541 | 0.478 |
| RSDiX-CLIPCap (best) | RSICD | 0.598 | 0.817 | 0.632 | 0.659 | 0.685 | 0.611 | 0.545 | 0.487 |
|  | UCMD | 0.800 | 0.859 | 0.639 | 0.797 | 0.828 | 0.806 | 0.781 | 0.744 |
|  | NWPU | 0.527 | 0.720 | 0.320 | 0.656 | 0.761 | 0.667 | 0.571 | 0.476 |

Comparison of RSDiX-CLIPCap results with SOTA methods on RSICD, UCMD and NWPU datasets. Scores: METEOR, S-SBERT-Sim, SPICE, ROUGE-L, BLEU-{1, 2, 3, 4}.

### RSDiX-SigLip Comparison Results 

| Dataset         | RSDiX-CLIP (B/16) | RSDiX-SigLIP (B/16) |
|-----------------|-------------------|---------------------|
| RSICD          | 94.40            | **95.60**          |
| RSI-CB128      | 35.00            | **37.90**          |
| RSI-CB256      | 45.40            | **50.30**          |
| WHU-earth      | 75.20            | **76.30**          |
| EuroSAT-RGB    | **51.70**        | 42.00              |
| MLRSNet        | 71.60            | **74.40**          |
| PatternNet     | 67.30            | **68.50**          |
| RESISC45       | 94.60            | **94.90**          |
| AID            | 92.60            | **95.10**          |
| RSSCN7         | 77.40            | **78.00**          |
| OPTIMAL-31     | 98.00            | **98.70**          |
| RSC11          | 78.20            | **81.20**          |
| WHU-RS19       | **98.30**        | 96.00              |
| **Average Top-1** ↑ | 72.43            | **76.06**          |
| **SBS-MSE** ↓    | **0.058**        | 0.219              |

Top-1 accuracy and S-BERT-Sim MSE results comparison of our RSDiX-CLIP-B/16 model with our RSDiX-SigLIP-B/16 model.

### RSdiX-SigLipCap Comparison Results 

| Model                      | M ↑ (🔒) | M ↑ (🔓) | SBS ↑ (🔒) | SBS ↑ (🔓) | S ↑ (🔒) | S ↑ (🔓) |
|----------------------------|----------|----------|------------|------------|----------|----------|
| RSDiX-SigLIPCap-P/16-B     | 0.374    | **0.394** | **0.599**  | 0.597      | 0.322    | **0.345** |
| RSDiX-SigLIPCap-P/16-M     | 0.347    | 0.283    | 0.570      | 0.438      | 0.305    | 0.235    |
| RSDiX-SigLIPCap-P/16-L     | 0.351    | 0.345    | 0.436      | 0.523      | 0.288    | 0.284    |
| RSDiX-SigLIPCap-P/16-XL    | 0.313    | 0.305    | 0.517      | 0.441      | 0.265    | 0.243    |

Average performances of our RSDiX-SigLIPCap models. Columns represent average METEOR, S-BERT-Sim, and SPICE scores. 🔒/🔓 denote our model variants trained with a locked/unlocked encoder and the size indicator (B, M, L, XL) denotes the GPT-2 decoder model.

### RSDiX-CLIP-SBERT Comparison Results

| Dataset         | RSDiX-CLIP (B/32) | RSDiX-CLIP (B/16) | RSDiX-CLIP (L/14) | RSDiX-CLIP-S-BERT (B/32) | RSDiX-CLIP-S-BERT (B/16) | RSDiX-CLIP-S-BERT (L/14) |
|-----------------|-------------------|-------------------|-------------------|--------------------------|--------------------------|--------------------------|
| RSICD          | 96.00            | 94.40            | 92.90            | 94.00                   | 93.70                   | **96.60**               |
| RSI-CB128      | 27.30            | 35.00            | 38.60            | 29.70                   | 29.50                   | **40.20**               |
| RSI-CB256      | 45.90            | 45.40            | 48.30            | 45.10                   | 45.00                   | 50.10                   |
| WHU-earth      | 65.70            | 75.20            | **78.50**         | 69.30                   | 61.30                   | 76.90                   |
| EuroSAT-RGB    | 42.60            | 51.70            | 51.10            | **56.60**               | 47.50                   | 51.50                   |
| MLRSNet        | 65.20            | 71.60            | **73.50**         | 66.00                   | 68.00                   | 72.90                   |
| PatternNet     | 59.40            | 67.30            | **72.80**         | 59.40                   | 60.40                   | 68.40                   |
| RESISC45       | 93.20            | 94.60            | **95.60**         | 93.50                   | 91.30                   | 95.20                   |
| AID            | **95.10**        | 92.60            | 91.10            | 92.70                   | 92.50                   | 94.90                   |
| RSSCN7         | **78.60**        | 77.40            | 77.30            | 75.70                   | 76.40                   | 73.80                   |
| OPTIMAL-31     | 96.40            | 98.00            | 95.50            | 97.30                   | 95.10                   | **98.40**               |
| RSC11          | 77.80            | 78.20            | 73.90            | 76.30                   | 76.90                   | **82.30**               |
| WHU-RS19       | 98.40            | 98.30            | 96.30            | 95.80                   | 96.70                   | **98.90**               |
| **Average Top-1** ↑ | 70.47            | 73.78            | 74.60            | 73.18                   | 71.87                   | **76.93**               |
| **SBS-MSE** ↓   | 0.104            | **0.058**         | 0.078            | 0.155                   | 0.161                   | 0.139                   |


Top-1 accuracy and S-BERT-Sim MSE results comparison of our RSDiX-CLIP and RSDiX-CLIP-S-BERT models.

### RSDiX-CLIPCap-SBERT Comparison Results

| Model                          | M ↑ (🔒) | M ↑ (🔓) | SBS ↑ (🔒) | SBS ↑ (🔓) | S ↑ (🔒) | S ↑ (🔓) |
|--------------------------------|----------|----------|------------|------------|----------|----------|
| RSDiX-CLIPCap-SBERT-B/32-B     | 0.407    | -        | 0.724      | -          | 0.495    | -        |
| RSDiX-CLIPCap-SBERT-B/32-M     | 0.424    | 0.477    | 0.763      | 0.755      | 0.516    | 0.545    |
| RSDiX-CLIPCap-SBERT-B/32-L     | 0.467    | -        | 0.777      | -          | 0.543    | -        |
| **RSDiX-CLIPCap-SBERT-B/16-B** | 0.504    | -        | 0.736      | -          | 0.550    | -        |
| RSDiX-CLIPCap-SBERT-B/16-M     | 0.445    | 0.478    | 0.775      | 0.750      | 0.535    | 0.543    |
| RSDiX-CLIPCap-SBERT-B/16-L     | 0.442    | 0.553    | 0.719      | **0.798**  | 0.509    | 0.590    |
| RSDiX-CLIPCap-SBERT-L/14-B     | 0.438    | **0.600** | 0.755      | 0.792      | 0.523    | **0.616** |

Average performances of our RSDiX-CLIPCap-SBERT models. Columns represent average METEOR, S-BERT-Sim, and SPICE scores. 🔒/🔓 denote our model variants trained with a locked/unlocked encoder and the size indicator (B, M, L, XL) denotes the GPT-2 decoder model.


# Models' weights 
Trained model weights can be found at the following MEGA link:

   - https://mega.nz/folder/N2di0BKZ#KhNe70VIg9a2mjoopo6A9A

Specifically:
   
   - The `RSDiX-CLIP` directory contains the weights for the RSDiX-CLIP family.

   - The `RSDiX-CLIPCap` contains the weights for the RSDiX-CLIPCap family.

# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.9` or higher.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).
## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/NeuRoNeLab/remote-sensing-captioning-transformer.git
```

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
# Training and fine-tuning 
To train and finetune the models, please check out the sections below. 
## Training and fine-tuning RSDiX-CLIP
To train and finetune RSDiX-CLIP, run `train_finetune_clip.py` file with the desired parameters. The parameters are mainly classified into **model parameters** and **data parameters**.
   1. Here are the available **model parameters**:
        - **`model`** (*type: str, default: `"openai/clip-vit-base-patch32"`*):  
      The pre-trained CLIP model to use. Defaults to `"openai/clip-vit-base-patch32"`.

        - **`lr`** (*type: Optional[float], default: None*):  
      The learning rate for the optimizer. If not provided, the model's configuration is used.
    
        - **`alpha`** (*type: float, default: 0.5*):  
          Trade-off factor between the contrastive loss and self-distillation loss. Defaults to `0.5`.
        
        - **`ema_decay`** (*type: float, default: 0.999*):  
          Exponential Moving Average (EMA) decay factor for the teacher model.
        
        - **`weight_decay`** (*type: float, default: 0.1*):  
          Weight decay applied during optimization.
        
        - **`start_factor`** (*type: float, default: 0.3333*):  
          Starting factor for the learning rate schedule during linear warm-up.
        
        - **`end_factor`** (*type: float, default: 1.0*):  
          Ending factor for the learning rate schedule during linear warm-up.
        
        - **`total_iters`** (*type: int, default: 5*):  
          Total number of iterations for linear warm-up.
        
        - **`use_warmup`** (*type: str, default: `"cosine"`*):  
          Specifies the warm-up strategy for learning rate scheduling. Options: `"cosine"` or `"linear"`.
        
        - **`warmup_steps`** (*type: int, default: 0*):  
          Number of warm-up steps.
        
        - **`eps`** (*type: float, default: 1e-08*):  
          Small epsilon added to prevent division by zero when normalizing embeddings.
        
        - **`betas`** (*type: tuple[float, float], default: `BETAS`*):  
          Beta coefficients for the Adam optimizer.
        
        - **`sinkhorn_lambda`** (*type: float, default: 0.1*):  
          Parameter for Sinkhorn distance computation in self-distillation.
        
        - **`sinkhorn_iter`** (*type: int, default: 4*):  
          Number of iterations for Sinkhorn distance computation.
        
        - **`ii_coeff`** (*type: float, default: 1.0*):  
          Coefficient for computing teacher targets for self-distillation.
        
        - **`tt_coeff`** (*type: float, default: 1.0*):  
          Coefficient for computing teacher targets for self-distillation.
        
        - **`remove_diag`** (*type: bool, default: False*):  
          Flag to remove diagonal elements during teacher target computation.
        
        - **`checkpoint_path`** (*type: str, default: None*):  
          Path to the CLIP model checkpoint.
        
        - **`use_sentence_bert_as_teacher`** (*type: bool, default: False*):  
          Use Sentence-BERT as a teacher model.
        
        - **`freeze_sentence_bert`** (*type: bool, default: True*):  
          Whether to freeze the Sentence-BERT model during training.
        
        - **`sentence_bert_model`** (*type: str, default: None*):  
          Path or name of the Sentence-BERT model to use as a teacher.
        
        - **`use_sigmoid_loss`** (*type: bool, default: False*):  
          Use a sigmoid-based loss function.

   3. Here are the available **data parameters**:
      
        - **`annotations_files`** (*type: Union[str, List[str]]*):  
          Path(s) to the annotation file(s).
    
        - **`img_dirs`** (*type: Union[str, List[str]]*):  
          Path(s) to the image directory/directories.
        
        - **`additional_test_annotation_files`** (*type: Optional[List[Optional[str]]], default: None*):  
          Additional annotation files for testing.
        
        - **`img_transform`** (*type: None, default: None*):  
          Transformation applied to image data.
        
        - **`target_transform`** (*type: None, default: None*):  
          Transformation applied to target/label data.
        
        - **`train_split_percentage`** (*type: float, default: `TRAIN_SPLIT_PERCENTAGE`*):  
          Percentage of data used for training split.
        
        - **`val_split_percentage`** (*type: float, default: `VAL_SPLIT_PERCENTAGE`*):  
          Percentage of data used for validation split.
        
        - **`batch_size`** (*type: int, default: `BATCH_SIZE`*):  
          Number of samples per batch.
        
        - **`num_workers`** (*type: int, default: 0*):  
          Number of worker threads for data loading.
        
        - **`augment_image_data`** (*type: bool, default: False*):  
          Whether to apply data augmentation to images.
        
        - **`augment_text_data`** (*type: bool, default: False*):  
          Whether to apply data augmentation to text.
        
        - **`shuffle`** (*type: bool, default: False*):  
          Whether to shuffle the dataset during loading.
        
        - **`processor`** (*type: str, default: None*):  
          Processor to use for data preprocessing.
        
        - **`use_gpt2_tokenizer`** (*type: bool, default: False*):  
          Whether to use the GPT-2 tokenizer for text processing. True if training CLIPCap. 
        
4. Here is an example command to run this script:
```shell
    python train_finetune_rsidx_clip.py fit --data.annotations_file data/RSICD/dataset_rsicd.json --data.img_dirs data/RSICD/RSICD_images --model.lr 1e-05 --model.weight_decay 0.01
```
4. Alternatively, you can modify the parameters values in the `clip_config.yaml` and run the following command:
```shell 
   python train_finetune_rsidx_clip.py fit --config clip_config.yaml 
```
5. To configure the callable parameters, you must specify the class path either through the CLI or within the `clip_config.yaml` file. For example: 
```shell 
  python train_finetune_rsidx_clip.py fit --config clip_config.yaml --data.img_transform torchvision.transforms.Pad
```
6. For major information, please refer to [Configure hyperparameters from the CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).
## Training and fine-tuning RSDiX-CLIPCap 

To train and finetune RSDiX-CLIPCap, run `train_finetune_clip.py` file with the desired parameters. The parameters are mainly classified into **model parameters** and **data parameters**.
   1. Here are the available **model parameters**:

         - **`prefix_length`** (*type: int*):  
          The length of the prefix input.
            
        - **`clip_length`** (*type: Optional[int], default: None*):  
          The length of the CLIP feature input.
        
        - **`prefix_size`** (*type: int, default: 512*):  
          The size of the prefix embedding.
        
        - **`num_layers`** (*type: int, default: 8*):  
          Number of layers in the transformer.
        
        - **`mapping_type`** (*type: MappingType, default: `MappingType.MLP`*):  
          The type of mapping layer to use.
        
        - **`dropout_transformer`** (*type: float, default: 0.0*):  
          Dropout rate applied in the transformer.
        
        - **`dropout_gpt2`** (*type: Optional[float], default: None*):  
          Dropout rate for GPT-2 model layers.
        
        - **`clipcap_lr`** (*type: float, default: 1e-3*):  
          Learning rate for the ClipCap model.
        
        - **`clipcap_weight_decay`** (*type: float, default: 0.1*):  
          Weight decay for the ClipCap optimizer.
        
        - **`clipcap_warmup_steps`** (*type: int, default: 5000*):  
          Number of warm-up steps for ClipCap learning rate scheduler.
        
        - **`model`** (*type: str, default: "openai/clip-vit-base-patch32"*):  
          The pre-trained CLIP model to use.
        
        - **`lr`** (*type: Optional[float], default: None*):  
          Learning rate for the optimizer.
        
        - **`alpha`** (*type: float, default: 0.5*):  
          Trade-off factor between contrastive loss and self-distillation loss.
        
        - **`ema_decay`** (*type: float, default: 0.999*):  
          Exponential Moving Average (EMA) decay factor for the teacher model.
        
        - **`weight_decay`** (*type: float, default: 0.1*):  
          Weight decay applied to model parameters.
        
        - **`start_factor`** (*type: float, default: 0.3333333333333333*):  
          Starting factor for linear warm-up in learning rate scheduling.
        
        - **`end_factor`** (*type: float, default: 1.0*):  
          Ending factor for linear warm-up in learning rate scheduling.
        
        - **`total_iters`** (*type: int, default: 5*):  
          Total iterations for linear warm-up.
        
        - **`use_warmup`** (*type: str, default: "cosine"*):  
          Warm-up strategy for learning rate scheduling, options are "cosine" or "linear."
        
        - **`warmup_steps`** (*type: int, default: 0*):  
          Number of warm-up steps.
        
        - **`eps`** (*type: float, default: 1e-08*):  
          Small epsilon value to prevent division by zero during normalization.
        
        - **`betas`** (*type: tuple[float, float], default: BETAS*):  
          Beta coefficients for the Adam optimizer.
        
        - **`sinkhorn_lambda`** (*type: float, default: 0.1*):  
          Parameter for Sinkhorn distance computation in self-distillation.
        
        - **`sinkhorn_iter`** (*type: int, default: 4*):  
          Number of iterations for Sinkhorn distance computation.
        
        - **`ii_coeff`** (*type: float, default: 1.0*):  
          Coefficient for inter-instance teacher target computation.
        
        - **`tt_coeff`** (*type: float, default: 1.0*):  
          Coefficient for teacher-target computation.
        
        - **`remove_diag`** (*type: bool, default: False*):  
          Flag to remove diagonal elements in teacher-target computation.
        
        - **`checkpoint_path`** (*type: str, default: None*):  
          Path to the general model checkpoint.
        
        - **`clip_checkpoint_path`** (*type: str, default: None*):  
          Path to the CLIP model checkpoint.
        
        - **`clipcap_checkpoint_path`** (*type: str, default: None*):  
          Path to the ClipCap model checkpoint.
        
        - **`metrics`** (*type: Union[str, list], default: `ALLOWED_METRICS`*):  
          List of metrics to evaluate the model.
        
        - **`use_beam_search`** (*type: bool, default: False*):  
          Whether to use beam search during inference.
        
        - **`gpt_model`** (*type: str, default: "gpt2"*):  
          The GPT model to use for captioning.
        
        - **`pad_token`** (*type: str, default: None*):  
          Padding token to use during inference.
        
        - **`every_n_batches`** (*type: int, default: 10*):  
          Frequency of evaluation or logging in terms of number of batches.
        
        - **`freeze_clip_encoder`** (*type: bool, default: True*):  
          Whether to freeze the weights of the CLIP encoder.
    
2. Here are the available **data parameters**:
       
    - **`annotations_files`** (*type: Union[str, List[str]]*):  
          Path(s) to the annotation file(s).
    
    - **`img_dirs`** (*type: Union[str, List[str]]*):  
          Path(s) to the image directory/directories.
        
    - **`additional_test_annotation_files`** (*type: Optional[List[Optional[str]]], default: None*):  
      Additional annotation files for testing.
    
    - **`img_transform`** (*type: None, default: None*):  
      Transformation applied to image data.
    
    - **`target_transform`** (*type: None, default: None*):  
      Transformation applied to target/label data.
    
    - **`train_split_percentage`** (*type: float, default: `TRAIN_SPLIT_PERCENTAGE`*):  
      Percentage of data used for training split.
    
    - **`val_split_percentage`** (*type: float, default: `VAL_SPLIT_PERCENTAGE`*):  
      Percentage of data used for validation split.
    
    - **`batch_size`** (*type: int, default: `BATCH_SIZE`*):  
      Number of samples per batch.
    
    - **`num_workers`** (*type: int, default: 0*):  
      Number of worker threads for data loading.
    
    - **`augment_image_data`** (*type: bool, default: False*):  
      Whether to apply data augmentation to images.
    
    - **`augment_text_data`** (*type: bool, default: False*):  
      Whether to apply data augmentation to text.
    
    - **`shuffle`** (*type: bool, default: False*):  
      Whether to shuffle the dataset during loading.
    
    - **`processor`** (*type: str, default: None*):  
      Processor to use for data preprocessing.
    
    - **`use_gpt2_tokenizer`** (*type: bool, default: False*):  
      Whether to use the GPT-2 tokenizer for text processing. True if training CLICap. 
     
4. Here is an example command to run this script:
```shell
    python train_rsidx_clipcap.py fit --data.annotations_file data/RSICD/dataset_rsicd.json --data.img_dirs data/RSICD/RSICD_images --model.clipcap_lr 1e-05 --model.clipcap_weight_decay 0.01
```
4. Alternatively, you can modify the parameters values in the `clip_config.yaml` and run the following command:
```shell 
   python train_rsidx_clipcap.py fit --config clipcap_config.yaml 
```
5. To configure the callable parameters, you must specify the class path either through the CLI or within the `clipcap_config.yaml` file. For example: 
```shell 
  python train_rsidx_clipcap.py fit --config clipcap_config.yaml --data.img_transform torchvision.transforms.Pad
```
6. For major information, please refer to [Configure hyperparameters from the CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).
## Running Bayesian Optimization
To train and finetune one of the two models leveraging bayesian optimization, follow the instructions below:
   
1. Ensure to have a `grid_file.yaml` that adheres to the structure shown in `clip_grid.yaml` and `clipcap_grid.yaml` files. Please note that the values of the parameters you want to finetune with must be **comma seprated values with no white spaces**.
   
2. Run `bayesian_optimization.py` with the desired parameters: 

- **`default_root_dir`** (*type: str, default: `os.getcwd()`*):  
  The root directory for storing outputs, defaulting to the current working directory.

- **`logs_dir`** (*type: str, default: "lightning_logs"*):  
  Directory where the training logs will be stored. Default is `lightning_logs`.

- **`grid_file`** (*type: str, default: "clip_grid.yaml"*):  
  Path to the grid file for Bayesian optimization. Defaults to `clip_grid.yaml`.

- **`n_iter`** (*type: int, default: 10*):  
  The number of iterations for optimization or training.

- **`n_init_points`** (*type: int, default: 5*):  
  The number of initial points to sample for Bayesian optimization.

- **`random_state`** (*type: int, default: 42*):  
  Random seed for reproducibility.

- **`bayesian_runs_export_file`** (*type: str, default: "bayesian_runs.json"*):  
  Path to export the results of the Bayesian optimization runs. Defaults to `bayesian_runs.json`.

- **`bayesian_runs_import_file`** (*type: str, default: None*):  
  Path to import previously saved Bayesian optimization results.

3. Alternatively, you can run `grid_search.py` (or `grid_search.sh` if you are using a Linux-based system) to perform classic **grid search**. 

# Evaluating
To evaluate the models, please check out the sections below. 
## Evaluating RSDiX-CLIP
To evaluate the RSDiX-CLIP model, run `eval_clip.py` with the desired parameters: 

- **`scores_dir`** (*type: str, default: `os.path.join(os.getcwd(), "eval_results")`*):  
  Directory to save evaluation results.

- **`scores_file`** (*type: str, default: "scores.tsv"*):  
  Name of the file where evaluation scores will be saved.

- **`model_pth`** (*type: str, required: True*):  
  Path to the model to evaluate.

- **`processor`** (*type: str, default: "openai/clip-vit-base-patch32"*):  
  Processor (e.g., `CLIPProcessor.from_pretrained`) used to preprocess data.

- **`annotations_file`** (*type: str, required: True*):  
  Path to the annotations file containing image-caption pairs.

- **`imgs_dir`** (*type: str, required: True*):  
  Directory containing images for evaluation.

- **`model_basename`** (*type: str, default: None*):  
  Optional basename of the model (if applicable).

## Evaluating RSDiX-CLIPCap
To evaluate the RSDiX-CLIPCap model, run `eval_clipcap.py` with the desired parameters: 

- **`seed`** (*type: int, default: 42*):  
  Random seed for reproducibility.

- **`scores_dir`** (*type: str, default: `os.path.join(os.getcwd(), "eval_results")`*):  
  Directory where evaluation scores will be stored.

- **`scores_file`** (*type: str, default: "clip_cap_scores.tsv"*):  
  Name of the file where evaluation scores will be saved. Located in the directory specified by `scores_dir`.

- **`model_pth`** (*type: str, default: None*):  
  Path to the model that will be evaluated.

- **`model_basename`** (*type: str, default: None*):  
  Basename for the model, used for naming the saved evaluation scores.

- **`processor`** (*type: str, default: "openai/clip-vit-base-patch32"*):  
  Processor (e.g., `CLIPProcessor.from_pretrained`) used for data preprocessing.

- **`use_beam_search`** (*type: bool, default: False*):  
  Flag to indicate whether beam search should be used during evaluation.

- **`metrics`** (*type: list of str, default: `ALLOWED_METRICS`*):  
  The metrics to be used during evaluation. If not provided, the default `ALLOWED_METRICS` will be used.

- **`no_splits`** (*type: bool, default: False*):  
  Flag to disable splitting the dataset during evaluation.

- **`no_evaluation`** (*type: bool, default: False*):  
  Flag to skip evaluation.

- **`export_captions_file`** (*type: str, default: None*):  
  Path to export the generated captions in a file.

- **`import_captions_file`** (*type: str, default: None*):  
  Path to import previously generated captions.

- **`annotations_files`** (*type: list of str, default:*):  
  List of annotation files for evaluation, by default:
  - `./data/RSICD/dataset_rsicd.json`
  - `./data/UCMD/dataset_ucmd.json`
  - `./data/RSITMD/dataset_rsitmd.json`
  - `./data/NAIS/dataset_nais.json`
  - `./data/NWPU-Captions/dataset_nwpu.json`

- **`img_dirs`** (*type: list of str, default:*):  
  List of image directories for evaluation, by default:
  - `./data/RSICD/RSICD_images`
  - `./data/UCMD/UCMD_images`
  - `./data/RSITMD/RSITMD_images`
  - `./data/NAIS/NAIS_images`
  - `./data/NWPU-Captions/NWPU-Captions_images`

- **`splits`** (*type: list of str, default: ["test", "test", "test", "test", "test"]*):  
  List of dataset splits to use for evaluation, by default using "test" for each dataset.


### Warnings 

Please note that when assessing the captioning model using Meteor metric and SPICE-based metrics on a Windows or MacOS device, you may expect some errors as stated in the [aac-metrics repository](https://github.com/Labbeti/aac-metrics). 
In order to avoid any complications, we recommend you to set up your environment on a Linux system. 
If you are not able to do so, you may export the predicted captions into a `.json` file and perform the evaluation on another machine.

### Errors 
When computing the Meteor metric, in case you encounter the following exception: 
```python
METEOR: could not convert string to float: '' on the couple: ([...], [[...]])
```
Try changing the locale settings to **US**. For more information, follow [here](https://github.com/Labbeti/aac-metrics/issues/9).

# Inference

## Running the RSDiX-CLIP Remote Sensing Inference Script
To extract image embeddings using the CLIP Remote Sensing model, run `clip_inference.py` with the desired parameters:

- **`annotations_file`** (*type: str, required: True*):  
  Path to the annotations file containing image-caption pairs.

- **`img_dir`** (*type: str, required: True*):  
  Directory containing the images to process.

- **`checkpoint_path`** (*type: str, required: True*):  
  Path to the model checkpoint to load for inference.

- **`processor`** (*type: str, default: "openai/clip-vit-base-patch32"*):  
  The processor (e.g., `CLIPProcessor.from_pretrained`) used to preprocess data.

- **`out_path`** (*type: str, default: "_clipinferenceimages/"*):  
  Path to save the generated captions in JSON format.

## Running the RSDiX-CLIPCap Remote Sensing Inference Script
To generate captions using the RSDiX-CLIPCap trained model, run `clipcap_inference.py` with the desired parameters. This script will generate captions using the trained model and the specified parameters:
- **`annotations_file`** (*type: str, required: True*):  
  Path to the annotations file containing image-caption pairs.

- **`img_dir`** (*type: str, required: True*):  
  Directory containing the images to process.

- **`checkpoint_path`** (*type: str, required: True*):  
  Path to the model checkpoint to load for inference.

- **`processor`** (*type: str, default: "openai/clip-vit-base-patch32"*):  
  The processor (e.g., `CLIPProcessor.from_pretrained`) used to preprocess data.

- **`out_path`** (*type: str, default: "_inferenceimages/"*):  
  Path to save the generated captions in JSON format.

- **`use_beam_search`** (*action: store_true*):  
  Whether to use beam search during inference (Setting it to *False* will use top-p sampling, which seems to work better for this model).


# Acknowledgements and references
The authors would like to sincerely thank the NAIS Solutions staff for helping with the realization of the S2LCD dataset.

This research was partly supported in part by PON Campania project AIWEO - Artificial Intelligence Wizard for Earth Observation, CUP: B67H22002780007, SURF: 22012BP000000049. 

Furthermore, a special thank must be delivered to the train-CLIP[^1], CLIP-rsicd[^2] and CapDec[^3] repositories contributors, which have been a fundamental building block of this project's codebase.

[^1]: [train-CLIP](https://github.com/Zasder3/train-CLIP)
[^2]: [CLIP-rsicd](https://github.com/arampacha/CLIP-rsicd)
[^3]: [CapDec repository](https://github.com/DavidHuji/CapDec)
[^4]: [CLIPCap paper](https://arxiv.org/abs/2111.09734)
