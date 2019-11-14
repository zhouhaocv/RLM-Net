# Visual Relationship Detection with Relative Location Mining
## Introduction
This repository contains the pytorch codes in our ACM MM 2019 paper "[Visual Relationship Detection with Relative Location Mining](https://arxiv.org/abs/1911.00713)", which is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
## Installation
**Requirement(ours)**
- pytorch,v1.0.1
- torchvision,v0.2.2
- CUDA, v9.0
- python3.6
- matlab
### 1.install dependencies
```bash
pip install yacs scipy tqdm
export INSTALL_DIR=$PWD
```
### 2.install pycocotools
```bash
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```
### 3.install RLM-Net
```bash
cd $INSTALL_DIR
git clone https://github.com/zhouhaocv/RLM-Net.git
cd RLM-Net
python setup.py build develop
```
### 4.download datasets(VRD）
- See details from [here](https://cs.stanford.edu/people/ranjaykrishna/vrd/)

- put "json_dataset" and "sg_dataset" in vrd/data/vrd/.

### 5.training
```bash
cd $INSTALL_DIR
cd vrd

a) generate vrd json
python gen_vrd_json.py

b) train the first stage(~4 hours)
python train_RLM_proposing_stage.py

c) train the second stage(~4.5 hours)
python train_RLM_predicate_stage.py
```
### 6.testing
```bash
python test_RLM_Net.py
```
evaluate with [predicate_step](https://github.com/zhouhaocv/RLM-Net/blob/master/vrd/eval/predicate_step.m) and [relation_phrase_step](https://github.com/zhouhaocv/RLM-Net/blob/master/vrd/eval/relation_phrase_step.m).
## Contact
If you have any problems, you can email to zhouhao_0039@sjtu.edu.cn.
