import os
import argparse
import shutil
import time
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import flow_transforms
import models
from loss_functions import *
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from config import cfg, cfg_from_file, save_config_to_file
from utils import frame_utils
from utils.flow_viz import flow_to_image
from utils.log import create_logger

import datasets

################################################################


# search for model and dataset names
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='Parser for training and testing',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--data',help='path to dataset')
parser.add_argument('--data-kitti12',help='path to kitti2012, if necessary', default=None)
parser.add_argument('--cfg', dest='cfg', default=None, type=str, help='path to config file')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs')

group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,help='test-val split file')
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')

parser.add_argument('--milestones', default=[300,500,1000], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--eval_freq', default=1, type=int,
                    help='evaluation frequency')
parser.add_argument('--no-eval', action='store_true',
                    help='do not conduct validation ')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--reuse_optim', action='store_true',
                    help='reuse optimizer ')
parser.add_argument('--eval_sintel', action='store_true',
                    help='eval on sintel ')
parser.add_argument('--out_dir', default=None,
                    help='path to save model')
parser.add_argument('--exp_dir', default='default',
                    help='path to save model')
parser.add_argument('--visual_all', action='store_true',
                    help='eval on sintel ')
parser.add_argument('--clip', default=-1, type=float)


best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    global args, best_EPE, save_path
    args = parser.parse_args()

    # Load config file
    if args.cfg is not None:
        cfg_from_file(args.cfg)
        assert cfg.TAG == os.path.splitext(os.path.basename(args.cfg))[0], 'TAG name should be file name'

    # Build save_path, which can be specified by out_dir and exp_dir
    save_path = '{},{}epochs{},b{},lr{}'.format(
        'dicl_wrapper', args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',args.batch_size,args.lr)
    
    save_path = os.path.join(args.exp_dir,save_path)
    if args.out_dir is not None:
        outpath = os.path.join(args.out_dir,args.dataset)
    else:
        outpath = args.dataset
    save_path = os.path.join(outpath,save_path)

    if not os.path.exists(outpath): os.makedirs(outpath)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # Create logger
    log_file = os.path.join(save_path, 'log.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    logger.info('=> will save everything to {}'.format(save_path))


    # Print settings
    for _,key in enumerate(args.__dict__):
        logger.info(args.__dict__[key])
    save_config_to_file(cfg, logger=logger)
    logger.info(args.pretrained)

    # Set random seed
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    eval_writer = SummaryWriter(os.path.join(save_path,'eval'))


    logger.info("=> fetching img pairs in '{}'".format(args.data))



    ########################## DATALOADER ##########################
    if args.dataset == 'flying_chairs':
        if cfg.SIMPLE_AUG:
            train_dataset = datasets.FlyingChairs_SimpleAug(args,root=args.data)
            test_dataset = datasets.FlyingChairs_SimpleAug(args,root=args.data,mode='val')
        else:
            train_dataset = datasets.FlyingChairs(args, image_size=cfg.CROP_SIZE,root=args.data)
            test_dataset = datasets.FlyingChairs(args,root=args.data,mode='val',do_augument=False)
    elif args.dataset == 'flying_things':
        train_dataset= datasets.SceneFlow(args, image_size=cfg.CROP_SIZE,root=args.data, dstype='frames_cleanpass',mode='train')
        test_dataset = datasets.SceneFlow(args, image_size=cfg.CROP_SIZE,root=args.data, dstype='frames_cleanpass',mode='val',do_augument=False)
    elif args.dataset == 'mpi_sintel_clean' or args.dataset =='mpi_sintel_final':
        clean_dataset = datasets.MpiSintel(args, image_size=cfg.CROP_SIZE,root=args.data, dstype='clean')
        final_dataset = datasets.MpiSintel(args, image_size=cfg.CROP_SIZE,root=args.data, dstype='final')
        train_dataset = torch.utils.data.ConcatDataset([clean_dataset]+ [final_dataset])
        if args.dataset == 'mpi_sintel_final':
            test_dataset = datasets.MpiSintel(args, do_augument=False,image_size=None,root=args.data, dstype='final')
        else:
            test_dataset = datasets.MpiSintel(args, do_augument=False,image_size=None,root=args.data, dstype='clean')              
    elif args.dataset == 'KITTI':
        train_dataset = datasets.KITTI(args, image_size=cfg.CROP_SIZE,root=args.data, is_val=False,logger=logger)     
        if args.data_kitti12 is not None:
            train_dataset12 = datasets.KITTI12(args, image_size=cfg.CROP_SIZE,root=args.data_kitti12, is_val=False,logger=logger)     
            train_dataset = torch.utils.data.ConcatDataset([train_dataset]+[train_dataset12])
        test_dataset = datasets.KITTI(args, root=args.data,do_augument=False, is_val=True, do_pad=False)      
    else:
        raise NotImplementedError


    logger.info('Training with %d image pairs' % len(train_dataset))

    logger.info('Testing with %d image pairs' % len(test_dataset))

    gpuargs = {'num_workers': args.workers, 'drop_last' : cfg.DROP_LAST}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, **gpuargs)

    if 'KITTI' in args.dataset:
        # We set batch size to 1 since KITTI images have different sizes
        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,num_workers=args.workers, pin_memory=True, shuffle=False)
    else:
        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size,num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.pretrained))
        pretrained_dict = torch.load(args.pretrained)

        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}

    model = models.__dict__['dicl_wrapper'](None)

    assert(args.solver in ['adam', 'sgd'])
    logger.info('=> setting {} solver'.format(args.solver))


    if args.solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=cfg.WEIGHT_DECAY,
                                     betas=(cfg.MOMENTUM, cfg.BETA))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=cfg.WEIGHT_DECAY,
                                    momentum=cfg.MOMENTUM)

    if args.pretrained:
        if 'state_dict' in pretrained_dict.keys():
            model.load_state_dict(pretrained_dict['state_dict'],strict=False)
        else:
            model.load_state_dict(pretrained_dict,strict=False)

        if args.reuse_optim:
            try:
                optimizer.load_state_dict(pretrained_dict['optimizer_state'])
            except:
                logger.info('do not have optimizer state')
        del pretrained_dict
        torch.cuda.empty_cache()    

    model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model=model.cuda()

    # Evaluation
    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(val_loader, model, 0, None, eval_writer,logger=logger)
        return

    # Learning rate schedule
    milestones =[]
    for num in range(len(args.milestones)):
        milestones.append(int(args.milestones[num]))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    

    ###################################### Training  ######################################
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, train_writer,logger=logger)
        scheduler.step()
        
        train_writer.add_scalar('lr',optimizer.param_groups[0]['lr'],epoch)
        train_writer.add_scalar('avg_loss',train_loss,epoch)

        if epoch%args.eval_freq==0 and not args.no_eval:
            with torch.no_grad():
                EPE = validate(val_loader, model, epoch, output_writers,eval_writer,logger=logger)
            eval_writer.add_scalar('mean_EPE', EPE, epoch)

            if best_EPE < 0:
                best_EPE = EPE

            if EPE<best_EPE:
                best_EPE = EPE
                ckpt_best_file = 'checkpoint_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'dicl_wrapper',
                    'state_dict': model.module.state_dict(),
                    'optimizer_state':optimizer.state_dict(),
                    'best_EPE': EPE
                }, False, filename=ckpt_best_file) 
            logger.info('Epoch: [{0}] Best EPE: {1}'.format(epoch, best_EPE))

        # Skip at least 5 epochs to save memory
        save_freq = max(args.eval_freq, 5)
        if epoch%save_freq==0:
            ckpt_file = 'checkpoint_'+str(epoch)+'.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'dicl_wrapper',
                'state_dict': model.module.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'best_EPE': best_EPE
            }, False, filename=ckpt_file)




def train(train_loader, model, optimizer, epoch, train_writer,logger):
    global n_iter, args

    # Recorders
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epe_rec = AverageMeter()
    loss_records = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),
                AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),
                AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()


    for i, (input1,target,valid) in enumerate(train_loader):
        if i >= epoch_size:
            break
        # measure data loading time
        data_time.update(time.time() - end)


        # load imgs and targets to device, and check if there is NaN or Inf
        target = target.to(device)
        target = target[:,0:2,:,:]    # flows have 2 dims
        input1 = torch.cat(input1,1).to(device)
        if torch.isinf(target).any() or (target!=target).any():
            continue

        if cfg.CLAMP_INPUT:   input1 = torch.clamp(input1,-1,1)


        # compute output   
        output = model(input1)
        # compute loss
        loss,loss_list,epe = MultiScale_UP(output,target,weight=cfg.MultiScale_W,loss_type=cfg.LOSS_TYPE,extra_mask=valid,valid_range=cfg.VALID_RANGE,removezero=args.removezero)

        # check if there is NaN or Inf in loss. If so, debug
        if torch.isinf(loss).any() or (loss!=loss).any(): import pdb;pdb.set_trace()

        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        for loss_level in range(len(loss_list)): loss_records[loss_level].update(loss_list[loss_level].item(),target.size(0))
        
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        if epe is not None: epe_rec.update(epe.item(),target.size(0)) 

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()

        if args.clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #### FOR VISUAL
        b, _, h, w = target.size()

        if (type(output) is not tuple) and (type(output) is not set):
            output = {output}


        with torch.no_grad():
            if i % args.print_freq == 0:
                # Visual the input and output of batch 0 
                # Convert imgs from [-1,1] to [0,255]
                im1_vis=0.5 + (input1[0,:3])*0.5;im2_vis=0.5 + (input1[0,3:])*0.5
                im1_vis= torch.clamp(im1_vis,0,1);im2_vis= torch.clamp(im2_vis,0,1)

                train_writer.add_image(('left'+str(0)),im1_vis,n_iter)
                train_writer.add_image(('right'+str(0)),im2_vis,n_iter)

                cur_valid = valid[0] if valid is not None else None
                
                for level, cur_flow in enumerate(output):
                    up_flow = F.interpolate(cur_flow, (h,w), mode='bilinear', align_corners=True)
                    up_flow[:,0,:,:] = up_flow[:,0,:,:]*(w/cur_flow.shape[3])
                    up_flow[:,1,:,:] = up_flow[:,1,:,:]*(h/cur_flow.shape[2])
                    realflow_vis = flow_to_image(up_flow[0].cpu().detach().numpy().transpose(1,2,0),cur_valid)/255
                    realflow_vis = realflow_vis.transpose(2,0,1)
                    train_writer.add_image(('flow'+str(level)),realflow_vis,n_iter)

                target_vis = flow_to_image(target[0].cpu().detach().numpy().transpose(1,2,0),cur_valid)/255
                target_vis = target_vis.transpose(2,0,1)
                train_writer.add_image(('gt_flow'+str(0)),target_vis,n_iter)

                logger.info('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}'
                      .format(epoch, i, epoch_size, batch_time,data_time, losses.avg))
        n_iter += 1


    train_writer.add_scalar('epe', epe_rec.avg, epoch)
    for loss_level in range(len(loss_list)):
        train_writer.add_scalar('level_'+str(loss_level)+'_loss', loss_records[loss_level].avg, epoch)
    return losses.avg


def validate(val_loader, model, epoch, output_writers,test_writer,logger):
    global args

    batch_time = AverageMeter(); flow2_EPEs = AverageMeter()

    epe_records = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),
                AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),
                AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]

    # switch to evaluate mode
    model.eval()

    end = time.time()
    out_list = []

    for i, (input1, target,valid) in enumerate(val_loader):
        target = target.to(device)            
        input1 = torch.cat(input1,1).to(device)
        raw_shape = input1.shape

        # Pad the input images to N*128
        height_new = int(np.ceil(raw_shape[2]/cfg.MIN_SCALE)*cfg.MIN_SCALE)
        width_new  = int(np.ceil(raw_shape[3]/cfg.MIN_SCALE)*cfg.MIN_SCALE)
        padding = (0, width_new-raw_shape[3], 0, height_new-raw_shape[2])

        if cfg.PAD_BY_CONS:
            input1 = torch.nn.functional.pad(input1, padding, "constant", cfg.PAD_CONS)
        else:
            input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)

        # If there is no valid mask, compute all pixels
        if valid is None:  valid = torch.ones(target.shape)[:,0,:,:].type_as(target)

        target = torch.nn.functional.pad(target, padding, "constant", 0)
        valid = torch.nn.functional.pad(valid, padding, "constant", 0)
        target = target[:,0:2,:,:]


        if cfg.CLAMP_INPUT:
            input1 = torch.clamp(input1,-1,1)

        # Compute outputs
        output = model(input1)


        b, _, h, w = target.size()
        if type(output) is not tuple:
            flow1 = output; output = {output}
        else:
            flow1 = output[0]

        # Upsample the highest resolution flow to raw resolution
        realflow = F.interpolate(flow1, (h,w), mode='bilinear', align_corners=True)
        realflow[:,0,:,:] = realflow[:,0,:,:]*(w/flow1.shape[3])
        realflow[:,1,:,:] = realflow[:,1,:,:]*(h/flow1.shape[2])


        if 'KITTI' in args.dataset:
            # Compute both epe and outlier percentage for KITTI
            flow_pr = realflow
            flow_gt = target
            epe = torch.sum((flow_pr - flow_gt)**2, dim=1).sqrt()
            mag = torch.sum(flow_gt**2, dim=1).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid.contiguous().view(-1) > 0 
            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            flow2_EPE = epe[val].mean()
            out_list.append(out[val].cpu().numpy())
        else:
            flow2_EPE = realEPE(realflow, target, sparse=cfg.SPARSE,extra_mask=valid)


        if torch.isinf(flow2_EPE).any() or (flow2_EPE!=flow2_EPE).any():
            import pdb;pdb.set_trace()

        flow2_EPEs.update(flow2_EPE.item(), target.size(0))


        # Visualization
        if args.visual_all or (i%100==0):
            cur_valid = valid[0] if valid is not None else None
            im1_vis=0.5 + (input1[0,:3])*0.5
            im2_vis=0.5 + (input1[0,3:])*0.5
            im1_vis= torch.clamp(im1_vis,0,1)
            im2_vis= torch.clamp(im2_vis,0,1)

            test_writer.add_image(('left'),im1_vis,i)
            test_writer.add_image(('right'),im2_vis,i)

            target_vis = flow_to_image(target[0].cpu().detach().numpy().transpose(1,2,0),cur_valid)/255
            target_vis = target_vis.transpose(2,0,1)
            test_writer.add_image(('gt_flow'),target_vis,i)
            
        for level, cur_flow in enumerate(output):
            # Visual outputs of each level
            up_flow = F.interpolate(cur_flow, (h,w), mode='bilinear', align_corners=True)
            up_flow[:,0,:,:] = up_flow[:,0,:,:]*(w/cur_flow.shape[3])
            up_flow[:,1,:,:] = up_flow[:,1,:,:]*(h/cur_flow.shape[2])
            cur_EPE = realEPE(up_flow, target, sparse=cfg.SPARSE,valid_range = cfg.VALID_RANGE[level],extra_mask=valid)
            epe_records[level].update(cur_EPE.item(), target.size(0))
            if args.visual_all or (i%100==0):
                cur_valid = valid[0] if valid is not None else None
                realflow_vis = flow_to_image(up_flow[0].cpu().detach().numpy().transpose(1,2,0),cur_valid)/255
                realflow_vis = realflow_vis.transpose(2,0,1)
                test_writer.add_image(('flow'+str(level)),realflow_vis,i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('EVAL: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))

    # Print each level outputs
    if len(out_list)>1:
        out_list = np.concatenate(out_list)
        logger.info("KITTI * Out: %f" % (100*np.mean(out_list)))

    logger.info('EVAL: * EPE {:.3f}'.format(flow2_EPEs.avg))
    for level in range(len(output)):
        logger.info('EVAL: Level {} Valid EPE {:.3f}'.format(level,epe_records[level].avg))

    return flow2_EPEs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


if __name__ == '__main__':
    main()
