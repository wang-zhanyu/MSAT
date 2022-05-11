# MSAT

Source code for the paper "A Medical Semantic-Assisted Transformer for
Radiographic Report Generation"

<br>

<p align="center">
  <img src="./images/framework.jpg" alt="overview of the proposed framework" width="800">
  <br>
  <b>Figure</b>: A overview of the proposed framework
</p>


## Requirements
* Python 3
* CUDA 10
* numpy
* tqdm
* easydict
* psutil
* [PyTorch](http://pytorch.org/) (>1.0)
* [torchvision](http://pytorch.org/)

## Data preparation
1. Download the [mimic_cxr dataset](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and extract clip features
```
python tools/extract_clip_feature.py --annotation mimic_cxr/annotation.json --save_path ./data/feature/mimic_clip16_att_512
```

2. Convert reports to tokens and save it to data/mimic folder.
```
python tools/build_vocab.py --annotation mimic_cxr/annotation.json --save_path data/mimic
```

## Training
### Train MSAT model
```
python main.py --folder experiments/V1
```

### Train MSAT model using reinforcement learning
```
python main.py --folder experiments/V1_rl --resume experiments/V1/snapshot/{best_model}.pth
```

