## DICL-Flow
This repository contains the code for our NeurIPS 2020 paper [Displacement-Invariant Matching Cost Learning for Accurate Optical Flow Estimation](https://papers.nips.cc/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html).


```
@article{wang2020displacement,
  title={Displacement-Invariant Matching Cost Learning for Accurate Optical Flow Estimation},
  author={Wang, Jianyuan and Zhong, Yiran and Dai, Yuchao and Zhang, Kaihao and Ji, Pan and Li, Hongdong},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

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
        ├── training
        ├── test
    ├── KITTI2012
        ├── training
        ├── testing
    ├── KITTI2015
        ├── training
        ├── testing
```

### Pretrained Weights

Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1y2ISM5veD3K9D0CGJg9qEGeyaQHgLy1y?usp=sharing).


### Training

As discussed in our paper, the training on the FlyingChairs dataset is divided into three phases for a good initialization (only single phase on other three datasets). We assume using eight GPUs while empirically the number of GPUs does not affect the final performance. You can use the commands like below:

```Shell
# Chair 0 phase, only simple augmentation is applied in the first phase.
python main.py -b 64 --lr 0.001 --epochs 120 --exp_dir dicl0_chair --cfg cfgs/dicl0_chair.yml \
--data /Path/To/FlyingChairs --dataset flying_chairs

# Chair 1 phase, using context network now.
python main.py -b 64 --lr 0.001 --epochs 120 --exp_dir dicl1_chair --cfg ../cfgs/dicl1_chair.yml \
--pretrained /Path/To/dicl0/checkpoint_best.pth.tar --data /Path/To/FlyingChairs --dataset flying_chairs

# Chair 2 phase, using Displacement Aware Projection layer now.
# Drop the learning rate by half at epoch 10.
python main.py -b 64 --lr 0.001 --epochs 120 --exp_dir dicl2_chair --cfg ../cfgs/dicl2_chair.yml --milestones 10 \
--pretrained /Path/To/dicl1/checkpoint_best.pth.tar --data /Path/To/FlyingChairs --dataset flying_chairs
```

For other datasets, just use one command, e.g.,

```Shell
python main.py -b 16 --lr 0.00025 --epochs 50 --pretrained /Path/To/dicl2/checkpoint_best.pth.tar
--exp_dir dicl3_thing --cfg cfgs/dicl3_thing.yml --data /Path/To/FlyingThings --dataset flying_things
```

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


### Acknowledgment

Our codes were developed on the basis of [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) and [VCN](https://github.com/gengshan-y/VCN). After the paper acceptation, we update the part of Dataloader, learning from [RAFT](https://github.com/princeton-vl/RAFT). Thanks a lot for their excellent works.

