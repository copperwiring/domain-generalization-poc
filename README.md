# GOAL
This code analyzes domain embeddings to understand the usability of CLIP for images frm different domains.

# SET-UP

`git clone https://github.com/copperwiring/domain-generalization-poc`

## Dataset
Donwoad the DomainNet dataset from here: http://ai.bu.edu/M3SDA/ The dataset contains many images so feel free to randomly sample few images (200) from each domain. The domains used for the code are:
- Clipart
- Infograph
- Painting
- Quickdraw
- Real
- Sketch

The directory structure for groundtruth and prediction is:

    .
    ├── ...
    ├── gt                    
    │   ├── get.csv
    └── pred
        ├── pred_domain.csv


# Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

(Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.)

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
```

## Get CLIP domain predictions

`python get_pred_features_domain_net.py `

## Plot Confusion Matrix
 `python3 get_confuson_matrix.py`
