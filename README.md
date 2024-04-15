# FedMGP PyTorch Implementation

This repository contains PyTorch implementation code.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```

## Pretrain models
Our method  loads pre-trained ViT locally. You can remove the following two lines of code from main.py to switch to online loading:
```
pretrained_cfg = create_model(args.model).default_cfg
pretrained_cfg['file']='pretrain_model/ViT-B_16.npz'
```

## Data preparation
If you already have CIFAR-100 or 5-Datasets (MNIST, Fashion-MNIST, NotMNIST, CIFAR10, SVHN), pass your dataset path to  `--data-path`.


The datasets aren't ready, change the download argument in `datasets.py` as follows

**CIFAR-100**
```
datasets.CIFAR100(download=True)
```

**5-Datasets**
```
datasets.CIFAR10(download=True)
MNIST_RGB(download=True)
FashionMNIST(download=True)
NotMNIST(download=True)
SVHN(download=True)
```

## Training
To train a model via command line:

Single node with single gpu

'--data_name' can be chosen from ['cifar100','5datasets','office_home']
config_file can be chosen from['cifar100_delay','five_datasets_delay']

```
python main.py \
       cifar100_delay \   #config_file
       --model vit_base_patch16_224 \
       --batch-size 4 \
       --data-path local_datasets/ \
       --output_dir ./output \
       --data_name cifar100
```

Specially for one dataset, if you want to use 'office_home' as the dataset:
```
python main.py \
       five_datasets_delay \   #config_file
       --model vit_base_patch16_224 \
       --batch-size 4 \
       --data-path local_datasets/ \
       --output_dir ./output \
       --data_name office_home

```



## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


