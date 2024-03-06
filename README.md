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

# RSDiX: Addressing Intra-Class Similarity in Remote Sensing with Self-Distillation

Remote sensing (RS) imagery serves as a crucial information source for diverse applications, including environmental monitoring, urban planning, defense, and security. Despite facing challenges such as spatial/spectral, temporal variability, or lack of quality annotated data, recent years have witnessed the development of deep learning methods for RS image processing. Such models often integrate linguistic information to enrich semantic understanding, showing potential in tasks such as zero-shot classification, detection, retrieval, and captioning of satellite images, to generate labels or descriptions with limited data. However, existing methods for these tasks encounter limitations, including low-quality captions, poor linguistic variety, similar captions for different images, noisy captions, and unreliable evaluation. In this work, we try to overcome some of these limitations by combining the power of pre-trained models with advanced training techniques such as self-distillation:

1. To tackle the issue of intra-class similarity in RS image datasets, we introduce *RSDiX-CLIP*, a fine-tuned version of CLIP  with an additional self-distillation objective. We propose *RSDiX-CLIPCap*, a family of captioning models that use the fine-tuned RSDiX-CLIP encoder and a transformer mapper network from CLIPCap. Our models achieve superior/competitive performance against state-of-the-art (SOTA) methods across multiple zero-shot RS image classification and captioning datasets.

2. We present *Sentinel-2 Land-cover Captioning Dataset* (S2LCD), a novel RS captioning dataset with 1533 Sentinel-2 images with several land cover/use and human influence and 7665 wide-vocabulary, detailed captions.

3. We challenge, within the domain of RS images, *N*-gram-based metrics, such as the BLEU score building upon prior research to provide additional evidence of their susceptibility to inherent bias and inaccuracy. A statistical sensitivity/robustness comparison on perturbed captions is used to advocate for more reliable alternative metrics Sentence-BERT-Similarity.

## Datasets 

**RSICD**: One of the largest datasets for RSIC containing RSI collected from Google Earth, Baidu Map, MapABC, and Tianditu. It contains 10,921 remote sensing images with a fixed size of 224 x 224 pixels with various resolutions, each annotated with 5 captions, accounting for 54,605 descriptions in total. The dataset covers a wide range of scenes, objects, and scenarios and has one of the largest diversity rates amongst RSI datasets.

**UCMD**: Consists of 2,100 images belonging to 21 classes (100 images per class), and each image, measuring 256 x 256 pixels, is associated with 5 captions, containing 10,500 descriptions in total. It contains images from urban areas only with high spatial resolution.

**RSITMD**: A recently introduced dataset containing 4,743 images and 5 captions per image, presenting 23,715 total descriptions. Unlike traditional RS image-text datasets, it presents more scene changes and fine-grained captions.

**NWPU-Captions**: A recent RS dataset, comprising 31,500 256 x 256 images and 157,500 captions (5 per each image), manually annotated by experienced volunteers. It offers a substantial scale and a broad representation of intricate scenes, providing a wealth of diverse vocabulary and sentence structures.

**S2LCD**: The proposed *Sentinel-2 Land-cover Captioning Dataset* encompasses 1533 image patches (224x224 pixels) created from Sentinel-2 L2A images, ensuring diversity in land cover/use (forests, mountains, agriculture, urban areas, all with varying human influence). Each patch has 5 captions (7665 in total) with wide vocabulary (natural language and EAGLES lexicon [@eagleslexicon]) and attention to detail. This dataset is used in captioning experiments only due to the peculiar caption structure of some images: a few captions describe only partial image elements while others capture complementary details. This makes them less suitable for contrastive image-text losses.


## RSDiX-CLIP Comparison Results

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

## RSDiX-CLIPCap Comparison Results

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
      
        - `--model` (type: str, default: "openai/clip-vit-base-patch32"): The pre-trained CLIP model to use. Defaults to "openai/clip-vit-base-patch32".

        - `--lr` (type: Optional[float], default: None): The learning rate for the optimizer. If not provided, it attempts to use the learning rate from the model's configuration.
    
        - `--alpha` (type: float, default: 0.5): Trade-off factor between the contrastive loss and self-distillation loss. Defaults to 0.5 for equal contributions from both losses.
    
        - `--ema_decay` (type: float, default: 0.999): Exponential Moving Average (EMA) decay factor for the teacher model. Controls the adaptation rate of the teacher model.
    
        - `--weight_decay` (type: float, default: 0.1): Weight decay applied to model parameters during optimization.
    
        - `--start_factor` (type: float, default: 0.3333333333333333): Starting factor for the learning rate schedule during linear warm-up.
    
        - `--end_factor` (type: float, default: 1.0): Ending factor for the learning rate schedule during linear warm-up.
    
        - `--total_iters` (type: int, default: 5): Total number of iterations over which linear warm-up is applied.
    
        - `--use_warmup` (type: str, default: "cosine"): Specifies whether to use warm-up for learning rate scheduling. Choose between "cosine" or "linear."
    
        - `--warmup_steps` (type: int, default: 0): Number of warm-up steps.
    
        - `--eps` (type: float, default: 1e-08): A small epsilon value added to prevent division by zero when normalizing embeddings.
    
        - `--betas` (type: tuple[float, float], default: BETAS): Beta coefficients for the Adam optimizer. Control exponential moving averages of gradient and squared gradient.
    
        - `--sinkhorn_lambda` (type: float, default: 0.1): Parameter used in Sinkhorn distance computation for self-distillation.
    
        - `--sinkhorn_iter` (type: int, default: 4): Number of iterations for Sinkhorn distance computation.
    
        - `--ii_coeff` (type: float, default: 1.0): Coefficient used in computing teacher targets for self-distillation.
    
        - `--tt_coeff` (type: float, default: 1.0): Coefficient used in computing teacher targets for self-distillation.
    
        - `--remove_diag` (type: bool, default: False): Flag to determine whether to remove diagonal elements when computing teacher targets.
    
        - `--checkpoint_path` (type: str, default: None): Path to the CLIP model checkpoint.

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

      - `--prefix_length` (type: int): Length of the prefix token used for text generation.
        
      - `--clip_length` (type: Optional[int], default: None): Length of the CLIP context window. If None, it uses the default context length.
        
      - `--prefix_size` (type: int): Size of the prefix embedding layer.
        
      - `--num_layers` (type: int): Number of layers for the text-to-text (T2T) mapping.
        
      - `--mapping_type` (type: MappingType, default: MappingType.MLP): Type of the mapping function (MLP or Linear).
        
      - `--dropout_transformer` (type: float): Dropout rate for the T2T transformer.
        
      - `--dropout_gpt2` (type: Optional[float], default: None): Dropout rate for the GPT2-based caption decoder.
        
      - `--clipcap_lr` (type: float, default: 1e-3): Learning rate for the CLIPCap model.
        
      - `--clipcap_weight_decay` (type: float, default: 0.1): Weight decay for the CLIPCap model.
        
      - `--clipcap_warmup_steps` (type: int, default: 5000): Number of warm-up steps for learning rate scheduling.
        
      - `--model` (type: str, default: "openai/clip-vit-base-patch32"): Pre-trained CLIP model to use.
        
      - `--lr` (type: Optional[float]): Learning rate for the CLIP model.
        
      - `--alpha` (type: float, default: 0.5): Alpha parameter for the Sinkhorn-Knopp algorithm.
        
      - `--ema_decay` (type: float, default: 0.999): Exponential moving average decay for model parameters.
        
      - `--weight_decay` (type: float, default: 0.1): Weight decay for the optimizer.
        
      - `--start_factor` (type: float, default: 0.3333333333333333): Start factor for learning rate scheduling.
        
      - `--end_factor` (type: float, default: 1.0): End factor for learning rate scheduling.
        
      - `--total_iters` (type: int, default: 5): Total number of iterations for learning rate scheduling.
        
      - `--use_warmup` (type: str, default: "cosine"): Warm-up strategy for the learning rate scheduler.
        
      - `--warmup_steps` (type: int, default: 0): Number of warm-up steps for learning rate scheduling.
        
      - `--eps` (type: float, default: 1e-08): Epsilon value for numerical stability in Sinkhorn-Knopp.
        
      - `--betas` (type: tuple[float, float], default: BETAS): Beta values for the AdamW optimizer. Control exponential moving averages of gradient and squared gradient.
        
      - `--sinkhorn_lambda` (type: float, default: 0.1): Lambda parameter for the Sinkhorn-Knopp algorithm.
        
      - `--sinkhorn_iter` (type: int, default: 4): Number of iterations for Sinkhorn-Knopp.
        
      - `--ii_coeff` (type: float, default: 1.0): Coefficient for the image-image matching loss.
        
      - `--tt_coeff` (type: float, default: 1.0): Coefficient for the text-text matching loss.
        
      - `--remove_diag` (type: bool, default: False): Whether to remove the diagonal of the similarity matrix.
        
      - `--checkpoint_path` (type: str): Path to the model checkpoint.
        
      - `--clip_checkpoint_path` (type: str): Path to the CLIP model checkpoint.
        
      - `--clipcap_checkpoint_path` (type: str): Path to the CLIPCap model checkpoint.
        
      - `--metrics` (type: Union[str, list]): Evaluation metrics for the model.
        
      - `--use_beam_search` (type: bool, default: False): Whether to use beam search for text generation.
        
      - `--gpt_model` (type: str, default: "gpt2"): The GPT-2 model to use to generate the captions.
        
      - `--pad_token` (type: str): Token used for padding sequences. If None, the EOS token is used for padding.
        
      - `--every_n_batches` (type: int, default: 10): Frequency of computing evaluation metrics.
        
      - `--freeze_clip_encoder` (type: bool, default: True): Whether to freeze the CLIP encoder during training.
    
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
- `--default_root_dir` (str): The root directory for the experiment. Default is the current working directory.
  
- `--logs_dir` (str): The directory where logs are stored. Default is "lightning_logs".

- `--grid_file` (str): Path to the YAML grid file specifying hyperparameter ranges. Default is "clip_grid.yaml".

- `--n_iter` (int): Number of optimization iterations. Default is 10.

- `--n_init_points` (int): Number of initial points for Bayesian Optimization. Default is 5.

- `--random_state` (int): Random seed for reproducibility. Default is 42.

- `--bayesian_runs_export_file` (str): Path to the JSON file to export Bayesian optimization results. Default is "bayesian_runs.json".

- `--bayesian_runs_import_file` (str): Path to the JSON file to import previous Bayesian optimization results. Default is None.

3. Alternatively, you can run `grid_search.py` (or `grid_search.sh` if you are using a Linux-based system) to perform classic **grid search**. 

# Evaluating
To evaluate the models, please check out the sections below. 
## Evaluating RSDiX-CLIP
To evaluate the RSDiX-CLIP model, run `eval_clip.py` with the desired parameters: 

- `--scores_dir (str)`: Path where the model's results directory will be created. Defaults to "eval_results".
  
- `--scores_file (str):` Path where the model's results will be saved. Defaults to "scores.tsv".
  
- `--model_pth (str)`: Path to the model checkpoint to evaluate.
  
- `--processor (str)`: CLIPProcessor to use to preprocess data. Defaults to "openai/clip-vit-base-patch32".
  
- `--annotations_file (str)`: Annotations file of the dataset to evaluate the model on.
  
- `--img_dir (str)`: Directory containing the images of dataset to evaluate the model on.

## Evaluating RSDiX-CLIPCap
To evaluate the RSDiX-CLIPCap model, run `eval_clipcap.py` with the desired parameters: 

- `--seed` (int): Global seed for reproducibility. Default is 42.

- `--scores_dir` (str): Directory to store evaluation scores. Default is "eval_results".

- `--scores_file` (str): Name of the scores file. It will be saved under the `scores_dir`. Default is "clip_cap_scores.tsv".

- `--model_pth` (str): Path of the model checkpoint to evaluate. Required.

- `--model_basename` (str): The model basename that will be saved along with the scores. Default is None.

- `--processor` (str): Processor from CLIPProcessor.from_pretrained to preprocess data. Default is "openai/clip-vit-base-patch32".

- `--use_beam_search` (bool): Whether to use beam search for text generation. Default is False.

- `--metrics` (List[str]): List of evaluation metrics to compute (e.g., METEOR, SBERT_SIM, ROUGE_L, BLEU1, BLEU2, etc.).
  Default includes all allowed metrics.

- `--no_splits` (bool): If set, disables splitting of datasets. Default is False.

- `--no_evaluation` (bool): If set, skips the evaluation step. Default is False.

- `--export_captions_file` (str): Path to export generated captions in JSON format. Default is None.

- `--import_captions_file` (str): Path to import captions for evaluation from a JSON file. Default is None.

- `--annotations_files` (List[str]): List of paths to annotation files for different datasets.
  Default includes datasets like RSICD, UCMD, RSITMD, NAIS, NWPU-Captions.

- `--img_dirs` (List[str]): List of paths to image directories corresponding to the annotation files.
  Default includes directories for datasets like RSICD, UCMD, RSITMD, NAIS, NWPU-Captions.

- `--splits` (List[str]): List of splits to evaluate for each dataset. Default is ["val", "test", "test", "test", "test"].


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

- `--annotations_file (str)`: Annotations file of the dataset to evaluate the model on.

- `--img_dir (str)`: Directory containing the images of dataset  to evaluate the model on.

- `--checkpoint_path (str)` : Path to the trained CLIP weights.

- `--out_path (str)` The path to store the generated captions in JSON format. Defaults to "_inferenceimages/".

- `--processor (str)`: Specify the CLIPProcessor model version to use for preprocessing data. Defaults to "openai/clip-vit-base-patch32".

## Running the RSDiX-CLIPCap Remote Sensing Inference Script
To generate captions using the RSDiX-CLIPCap trained model, run `clipcap_inference.py` with the desired parameters. This script will generate captions using the trained model and the specified parameters:
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
