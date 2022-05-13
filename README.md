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
* CUDA 11
* tqdm
* easydict
* psutil
* ftfy
* regex
* tqdm
* PyTorch=1.7.1
* torchvision

## Data preparation
1. Download the [mimic_cxr dataset](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and extract clip features
```
python tools/extract_clip_feature.py --annotation mimic_cxr/annotation.json --save_path ./data/feature/mimic_clip16_att_512
```

2. Convert reports to tokens and save it to data/mimic folder using the following script. Or download from [here](https://drive.google.com/file/d/1tUJlC_yJ7Tq-VdK76yGxOmFBVeVSDmiz/view?usp=sharing)
```
python tools/build_vocab.py --annotation mimic_cxr/annotation.json --save_path data/mimic --radgraph data/mimic/MIMIC-CXR_graphs.json
```

3. Download metric package from [here](https://drive.google.com/file/d/1OcOwa73e0u1GggrrgDMaAXt9IOaLTYrs/view?usp=sharing) and unzip it into MSAT folder.

## Training
### Train MSAT model
```
python main.py --folder experiments/V1
```

### Train MSAT model using reinforcement learning
```
python main.py --folder experiments/V1_rl --resume experiments/V1/snapshot/{best_model}.pth
```

## Acknowledgements
Thanks the contribution of [image-captioning](https://github.com/JDAI-CV/image-captioning), [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and awesome PyTorch team.