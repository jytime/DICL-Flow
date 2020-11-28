# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import cv2
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, FlowAugmentorKITTI
from config import cfg, cfg_from_file, save_config_to_file
from PIL import Image
import torchvision.transforms as transforms
import flow_transforms
from imageio import imread

class CombinedDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        length = 0 
        for i in range(len(self.datasets)):
            length += len(self.datsaets[i])
        return length

    def __getitem__(self, index):
        i = 0
        for j in range(len(self.datasets)):
            if i + len(self.datasets[j]) >= index:
                yield self.datasets[j][index-i]
                break
            i += len(self.datasets[j])

    def __add__(self, other):
        self.datasets.append(other)
        return self

class FlowDataset(data.Dataset):
    def __init__(self, args, image_size=None, do_augument=False, return_path= False,return_img_path= False):
        self.image_size = image_size
        self.do_augument = do_augument
        self.return_path = return_path
        self.return_img_path = return_img_path

        if self.do_augument:
            self.augumentor = FlowAugmentor(self.image_size)

        self.flow_list = []
        self.image_list = []

    def __getitem__(self, index):


        np.random.seed()

        if index!=(index % len(self.image_list)): assert NotImplementedError
        index = index % len(self.image_list)
        flow = frame_utils.read_gen(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        flow = np.array(flow).astype(np.float32)

        if self.do_augument:
            img1, img2, flow = self.augumentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img1 = (img1/255-0.5)/0.5
        img2 = (img2/255-0.5)/0.5


        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.ones_like(flow[0])

        if self.return_img_path: return [img1,img2], flow,valid, self.image_list[index][0]        
        if self.return_path: return [img1,img2], flow, self.flow_list[index]
        return [img1,img2], flow,valid

    def __len__(self):
        return len(self.image_list)

    def __add(self, other):
        return CombinedDataset([self, other])


class FlyingChairs(FlowDataset):
    def __init__(self, args, image_size=None, do_augument=True, root='datasets/FlyingChairs_release/data',mode='train'):
        super(FlyingChairs, self).__init__(args, image_size, do_augument)
        self.root = root
        if do_augument:
            self.augumentor.min_scale = -0.2
            self.augumentor.max_scale = 1.0

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('FlyingChairs_train_val.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (mode=='train' and xid==1) or (mode=='val' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingChairs_SimpleAug(data.Dataset):
    def __init__(self, args, root='datasets/FlyingChairs_release/data',mode='train'):

        # Normalize images to [-1,1]
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        target_transform = transforms.Compose([
            flow_transforms.ArrayToTensor()])

        # Simple Aug
        if mode=='train':
            co_transform = flow_transforms.Compose([
                flow_transforms.RandomTranslate(cfg.RANDOM_TRANS),
                flow_transforms.RandomCrop((cfg.CROP_SIZE[0],cfg.CROP_SIZE[1])),
                flow_transforms.RandomVerticalFlip(),
                flow_transforms.RandomHorizontalFlip() ])
        else:
            co_transform = None

        self.root = root
        self.transform = input_transform
        self.target_transform = target_transform
        self.co_transform = co_transform


        images = []
        for flow_map in sorted(glob.glob(os.path.join(root,'*_flow.flo'))):
            flow_map = os.path.basename(flow_map)
            root_filename = flow_map[:-9]
            img1 = root_filename+'_img1.ppm'
            img2 = root_filename+'_img2.ppm'
            if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
                continue
            images.append([[img1,img2],flow_map])

        train_list, test_list = split2list(root,'FlyingChairs_train_val.txt')
        if mode=='train':
            self.path_list = train_list
        else:
            self.path_list = test_list

    def loader(self, root, path_imgs, path_flo):
        imgs = [os.path.join(root,path) for path in path_imgs]
        flo = os.path.join(root,path_flo)
        return [imread(img).astype(np.float32)[:,:,:3] for img in imgs],load_flo(flo)


    def __getitem__(self, index):
        inputs, target = self.path_list[index]
        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        valid = torch.ones(target[0].shape)
        return inputs, target,valid

    def __len__(self):
        return len(self.path_list)



class MpiSintel(FlowDataset):
    def __init__(self, args, image_size=None, do_augument=True, root='datasets/Sintel/training', dstype='clean'):
        super(MpiSintel, self).__init__(args, image_size, do_augument)
        if do_augument:
            self.augumentor.min_scale = -0.2
            self.augumentor.max_scale = 0.7

        self.root = root
        self.dstype = dstype

        flow_dir = 'flow'
        assert(os.path.isdir(os.path.join(self.root,flow_dir)))
        img_dir = dstype 
        assert(os.path.isdir(os.path.join(self.root,img_dir)))

        images = []
        for flow_map in sorted(glob(os.path.join(self.root,flow_dir,'*','*.flo'))):
            flow_map = os.path.relpath(flow_map,os.path.join(self.root,flow_dir))

            scene_dir, filename = os.path.split(flow_map)
            no_ext_filename = os.path.splitext(filename)[0]
            prefix, frame_nb = no_ext_filename.split('_')
            frame_nb = int(frame_nb)
            img1 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
            img2 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb + 1))
            flow_map = os.path.join(flow_dir,flow_map)
            if not (os.path.isfile(os.path.join(self.root,img1)) or os.path.isfile(os.path.join(self.root,img2))):
                continue
            images.append([[img1,img2],flow_map])
        
        # Use split2list just to ensure the same data structure; actually we do not split here
        tbd_list, _ = split2list(images, split=1.1, default_split=1.1,order=True)

        self.flow_list = []
        self.image_list = []
        for i in range(len(tbd_list)):
            self.flow_list.append(os.path.join(root,tbd_list[i][1]))
            im1 = os.path.join(root,tbd_list[i][0][0])
            im2 = os.path.join(root,tbd_list[i][0][1])
            self.image_list.append([im1, im2])





class SceneFlow(FlowDataset):
    def __init__(self, args, image_size, do_augument=True, root='datasets',mode='train',
            dstype='frames_cleanpass', use_flyingthings=True, use_monkaa=False, use_driving=False):
        
        super(SceneFlow, self).__init__(args, image_size, do_augument)
        self.root = root
        self.dstype = dstype
        self.mode = mode

        if do_augument:
            self.augumentor.min_scale = -0.2
            self.augumentor.max_scale = 0.8

        if use_flyingthings:
            self.add_flyingthings()
        
    def add_flyingthings(self):
        if self.mode=='train':
            # filter some hard cases for training, the same as: https://github.com/gengshan-y/VCN
            images_train = []
            filepath = self.root + '/train/'
            exc_list = ['0004117.flo','0003149.flo','0001203.flo','0003147.flo','0003666.flo','0006337.flo','0006336.flo','0007126.flo','0004118.flo',]

            left_fold  = 'image_clean/left/'
            flow_noc   = 'flow/left/into_future/'
            train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

            l0_trainlf  = [filepath+left_fold+img.replace('flo','png') for img in train]
            l1_trainlf = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainlf]
            flow_trainlf = [filepath+flow_noc+img for img in train]


            exc_list = ['0003148.flo','0004117.flo','0002890.flo','0003149.flo','0001203.flo','0003666.flo','0006337.flo','0006336.flo','0004118.flo',]

            left_fold  = 'image_clean/right/'
            flow_noc   = 'flow/right/into_future/'
            train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

            l0_trainrf = [filepath+left_fold+img.replace('flo','png') for img in train]
            l1_trainrf = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainrf]
            flow_trainrf = [filepath+flow_noc+img for img in train]


            exc_list = ['0004237.flo','0004705.flo','0004045.flo','0004346.flo','0000161.flo','0000931.flo','0000121.flo','0010822.flo',
            '0004117.flo','0006023.flo','0005034.flo','0005054.flo','0000162.flo','0000053.flo','0005055.flo','0003147.flo','0004876.flo','0000163.flo','0006878.flo',]

            left_fold  = 'image_clean/left/'
            flow_noc   = 'flow/left/into_past/'
            train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

            l0_trainlp  = [filepath+left_fold+img.replace('flo','png') for img in train]
            l1_trainlp = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(-1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainlp]
            flow_trainlp = [filepath+flow_noc+img for img in train]

            exc_list = ['0003148.flo','0004705.flo','0000161.flo','0000121.flo','0004117.flo','0000160.flo','0005034.flo',
            '0005054.flo','0000162.flo','0000053.flo','0005055.flo','0003147.flo','0001549.flo','0000163.flo','0006336.flo','0001648.flo','0006878.flo',]

            left_fold  = 'image_clean/right/'
            flow_noc   = 'flow/right/into_past/'
            train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

            l0_trainrp  = [filepath+left_fold+img.replace('flo','png') for img in train]
            l1_trainrp = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(-1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainrp]
            flow_trainrp = [filepath+flow_noc+img for img in train]

            if cfg.HALF_THINGS:
                l0_train = l0_trainlf  + l0_trainlp 
                l1_train = l1_trainlf  + l1_trainlp 
                flow_train = flow_trainlf + flow_trainlp 
            else:
                l0_train = l0_trainlf + l0_trainrf + l0_trainlp + l0_trainrp
                l1_train = l1_trainlf + l1_trainrf + l1_trainlp + l1_trainrp
                flow_train = flow_trainlf + flow_trainrf + flow_trainlp + flow_trainrp

            for num in range(len(l1_train)):
                self.image_list.append([l0_train[num],l1_train[num]])
                self.flow_list.append(flow_train[num])
        elif self.mode=='val':
            images_val =[]
            filepath = self.root+'/val/'
            exc_list = ['9999999.flo']
            left_fold  = 'image_clean/left/'
            flow_noc   = 'flow/left/into_future/'
            val = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

            l0_vallf  = [filepath+left_fold+img.replace('flo','png') for img in val]
            l1_vallf = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_vallf]
            flow_vallf = [filepath+flow_noc+img for img in val]
            for num in range(len(l0_vallf)):
                self.image_list.append([l0_vallf[num],l1_vallf[num]])
                self.flow_list.append(flow_vallf[num])
        else:
            raise NotImplementedError


class KITTI(FlowDataset):
    def __init__(self, args, image_size=None,return_path=False, do_augument=True, is_val=False, do_pad=False, split=True, logger=None,root='datasets/KITTI'):
        super(KITTI, self).__init__(args, image_size, do_augument)
        self.root = root
        self.is_val = is_val
        self.do_pad = do_pad
        self.logger = logger
        self.return_path = return_path

        if self.do_augument:
            self.augumentor = FlowAugmentorKITTI(self.image_size, min_scale=-0.2, max_scale=0.5,logger=logger)

        flows = sorted(glob(os.path.join(root, 'training', 'flow_occ/*_10.png')))
        images1 = sorted(glob(os.path.join(root, 'training', 'image_2/*_10.png')))
        images2 = sorted(glob(os.path.join(root, 'training', 'image_2/*_11.png')))

        for i in range(len(flows)):
            self.flow_list += [flows[i]]
            self.image_list += [[images1[i], images2[i]]]


    def __getitem__(self, index):

        np.random.seed()

        index = index % len(self.image_list)
        frame_id = self.image_list[index][0]
        frame_id = frame_id.split('/')[-1]

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])

        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
            
        if self.do_augument:
            img1, img2, flow, valid = self.augumentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()


        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()

        if self.do_pad:
            ht, wd = img1.shape[1:]
            pad_ht = (((ht // 128) + 1) * 128 - ht) % 128
            pad_wd = (((wd // 128) + 1) * 128 - wd) % 128
            pad_ht1 = [0, pad_ht]
            pad_wd1 = [pad_wd//2, pad_wd - pad_wd//2]
            pad = pad_wd1 + pad_ht1

            img1 = img1.view(1, 3, ht, wd)
            img2 = img2.view(1, 3, ht, wd)
            flow = flow.view(1, 2, ht, wd)
            valid = valid.view(1, 1, ht, wd)

            img1 = torch.nn.functional.pad(img1, pad, mode='replicate')
            img2 = torch.nn.functional.pad(img2, pad, mode='replicate')
            flow = torch.nn.functional.pad(flow, pad, mode='constant', value=0)
            valid = torch.nn.functional.pad(valid, pad, mode='replicate', value=0)

            img1 = img1.view(3, ht+pad_ht, wd+pad_wd)
            img2 = img2.view(3, ht+pad_ht, wd+pad_wd)
            flow = flow.view(2, ht+pad_ht, wd+pad_wd)
            valid = valid.view(ht+pad_ht, wd+pad_wd)

        img1 = (img1/255-0.5)/0.5
        img2 = (img2/255-0.5)/0.5

        if self.return_path:
            return [img1,img2], flow, valid, frame_id

        return [img1,img2],flow, valid 


class KITTI12(FlowDataset):
    def __init__(self, args, return_path=False, image_size=None, do_augument=True, is_val=False, do_pad=False, split=True, logger=None,root='datasets/KITTI'):
        super(KITTI12, self).__init__(args, image_size, do_augument)
        self.root = root
        self.is_val = is_val
        self.do_pad = do_pad
        self.logger = logger
        self.return_path = return_path

        if self.do_augument:
            self.augumentor = FlowAugmentorKITTI(self.image_size, min_scale=-0.2, max_scale=0.5,logger=logger)

        flows = sorted(glob(os.path.join(root, 'training', 'flow_occ/*_10.png')))
        images1 = sorted(glob(os.path.join(root, 'training', 'colored_0/*_10.png')))
        images2 = sorted(glob(os.path.join(root, 'training', 'colored_0/*_11.png')))

        for i in range(len(flows)):
            self.flow_list += [flows[i]]
            self.image_list += [[images1[i], images2[i]]]


    def __getitem__(self, index):

        np.random.seed()

        index = index % len(self.image_list)
        frame_id = self.image_list[index][0]
        frame_id = frame_id.split('/')[-1]

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]

        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])


        if self.do_augument:
            img1, img2, flow, valid = self.augumentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()

        if self.do_pad:
            ht, wd = img1.shape[1:]
            pad_ht = (((ht // 128) + 1) * 128 - ht) % 128
            pad_wd = (((wd // 128) + 1) * 128 - wd) % 128
            pad_ht1 = [0, pad_ht]
            pad_wd1 = [pad_wd//2, pad_wd - pad_wd//2]
            pad = pad_wd1 + pad_ht1

            img1 = img1.view(1, 3, ht, wd)
            img2 = img2.view(1, 3, ht, wd)
            flow = flow.view(1, 2, ht, wd)
            valid = valid.view(1, 1, ht, wd)

            img1 = torch.nn.functional.pad(img1, pad, mode='replicate')
            img2 = torch.nn.functional.pad(img2, pad, mode='replicate')
            flow = torch.nn.functional.pad(flow, pad, mode='constant', value=0)
            valid = torch.nn.functional.pad(valid, pad, mode='replicate', value=0)

            img1 = img1.view(3, ht+pad_ht, wd+pad_wd)
            img2 = img2.view(3, ht+pad_ht, wd+pad_wd)
            flow = flow.view(2, ht+pad_ht, wd+pad_wd)
            valid = valid.view(ht+pad_ht, wd+pad_wd)

        img1 = (img1/255-0.5)/0.5
        img2 = (img2/255-0.5)/0.5

        if self.return_path:
            return [img1,img2], flow, valid, frame_id

        return [img1,img2],flow, valid 


def split2list(images, split, default_split=1.1,order = False):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        # assert(len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0,1,len(images)) < split
    else:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    
    if (not isinstance(split, str)) and (order==True):
        if isinstance(split, float):
            check_split = split
        else:
            check_split = default_split

        split_values = np.ones(len(images))==1
        split_values[int(len(images)*check_split):]=False
    
    if len(split_values)!=len(images):
        import pdb;pdb.set_trace()

    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    
    return train_samples, test_samples



