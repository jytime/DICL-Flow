## DICL-Flow
This repository contains the code for our NeurIPS 2020 paper [Displacement-Invariant Matching Cost Learning for Accurate Optical Flow Estimation](https://papers.nips.cc/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html).

## Requirements

PyTorch 1.1, CUDA 9.0, and other dependencies (e.g., torchvision, tqdm). You can create a virtual environment and use pip to install the requirements. You may also install the dependencies manually.

```shell
conda create -n dicl python=3.6
conda activate dicl
pip install -r requirements.txt
```


### Dataset

We conduct experiments on the [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [MPI Sintel](http://sintel.is.tue.mpg.de/), and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) optical flow datasets.

My dataset structure is listed below. You can put data at any place and set the --data flag of main.py correspondingly.

```Shell
├── dataset
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D_subset
        ├── train
        ├── val
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI2012
        ├── testing
        ├── training
    ├── KITTI2015
        ├── testing
        ├── training
```

### Pretrained Weights

Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1y2ISM5veD3K9D0CGJg9qEGeyaQHgLy1y?usp=sharing).


### Training


### Evaluation

You can evaluate a trained model like below, where ```-e``` indicates evaluation:

```Shell
# Sintel Dataset
python main.py -b 1 -e --pretrained pretrained/ckpt_sintel.pth.tar --cfg cfgs/dicl4_sintel.yml \
--data /Path/To/Sintel/Dataset --exp_dir /Path/To/Save/Log --dataset mpi_sintel_clean 

# KITTI Dataset
python main.py -b 1 -e --pretrained pretrained/ckpt_kitti.pth.tar --cfg cfgs/dicl5_kitti.yml \
--data /Path/To/KITTI/Dataset --exp_dir /Path/To/Save/Log --dataset KITTI
```





### Citation

```
@article{wang2020displacement,
  title={Displacement-Invariant Matching Cost Learning for Accurate Optical Flow Estimation},
  author={Wang, Jianyuan and Zhong, Yiran and Dai, Yuchao and Zhang, Kaihao and Ji, Pan and Li, Hongdong},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

### Acknowledgment

Our codes were developed on the basis of [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) and [VCN](https://github.com/gengshan-y/VCN). After the paper acceptation, we update the part of Dataloader, learning from [RAFT](https://github.com/princeton-vl/RAFT). Thanks a lot for their works.

