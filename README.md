# Description
Source code of Paper "IntTower: the Next Generation of  Two-Tower Model for Pre-ranking System"
sadasdasd
## Easy to use
``` shell
pip install -r requirements.txt
python train_movielens_IntTower.py 
```
## Source code of Paper "IntTower: the Next Generation of  Two-Tower Model for Pre-ranking System" 
![avatar](./figure/model.PNG)
# Contents
- [Contents](#contents)
- [IntTower Description](#IntTower-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Performance](#performance)

# [IntTower Description](#contents)

The proposed model, IntTower (short for Interaction enhanced Two-Tower), consists of Light-SE, FE-Block and CIR modules. 
Specifically, lightweight Light-SE module is used to identify the importance of different features and obtain refined feature representations in each tower. FE-Block module performs fine-grained and early feature interactions to capture the interactive signals between user and item towers explicitly and CIR module leverages a contrastive interaction regularization to further enhance the interactions implicitly.

[Paper](https://dl.acm.org/doi/abs/10.1145/3459637.3481915): Xiangyang Li*, Bo Chen*, Huifeng Guo, Jingjie Li, Chenxu Zhu, Xiang Long, Yichao Wang, Wei Guo, Longxia Mao, Jinxing Liu, Zhenhua Dong, Ruiming Tang. IntTower: the Next Generation of Two-Tower Model for
Pre-Ranking System

# [Dataset](#contents)

- [Movie-Lens-1M](https://grouplens.org/datasets/movielens/1m/)
- [Amazon(eletro)](https://jmcauley.ucsd.edu/data/amazon/)
- [Alibaba ads](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)

# [Environment Requirements](#contents)

- Hardware（CPU/GPU）
    - Prepare hardware environment with CPU or GPU processor.
- Framework
- Requirements
  - deepctr==0.9.0
  - deepctr_torch==0.2.7
  - deepmatch==0.2.0
  - keras==2.8.0
  - matplotlib==3.5.2
  - numpy==1.21.4
  - pandas==1.4.2
  - pytorch_lightning==1.6.3
  - scikit_learn==1.1.1
  - tensorflow==2.8.0
  - torch==1.10.0
  - torchkeras==3.0.2
  - torchsummary==1.5.1
  - torchvision==0.12.0
  - tqdm==4.51.0
  - xgboost==1.6.1


# [Quick Start](#contents)


- running on CPU or GPU

  ```python
  # run training and evaluation example
  python train_movielens_IntTower.py
  ```
 # [Performance](#contents)
 ![avatar](./figure/performance.PNG)


## cite our work
```
@article{li2021exploring,
  title={Exploring text-transformers in aaai 2021 shared task: Covid-19 fake news detection in english},
  author={Li, Xiangyang and Xia, Yu and Long, Xiang and Li, Zheng and Li, Sujian},
  journal={arXiv preprint arXiv:2101.02359},
  year={2021}
}
```

