import argparse
import time
import os
import numpy as np
from collections import OrderedDict

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, loss
from mxboard import SummaryWriter    

from multi_pose.models import build_model
from multi_pose.datasets.coco_data import get_loader
from multi_pose.utils import AverageMeter


# Hyper-params
parser = argparse.ArgumentParser(description='MXNet rtpose Training')
parser.add_argument('--data-dir', default='./data/dataset/COCO/images', type=str, metavar='DIR', help='path to where coco images stored') 
parser.add_argument('--mask-dir', default='./data/dataset/COCO/mask', type=str, metavar='DIR', help='path to where coco images stored')    
parser.add_argument('--logdir', default='logs', type=str, metavar='DIR', help='path to where tensorboard log restore')                                       
parser.add_argument('--json_path', default='./data/dataset/COCO/COCO.json', type=str, metavar='PATH', help='path to where coco images stored')                                      
parser.add_argument('--model-path', default='./model_checkpoints', type=str, metavar='DIR', help='path to where the model saved') 
parser.add_argument('--load-model', default='', type=str, help='which model to load') 
parser.add_argument('--log-key', default='new_experiment', type=str, help='Which key to use for mxboard') 
parser.add_argument('--trunk', default='resnet18_v1b', type=str, help='Which backend to use [mobilenet, vgg19]') 
parser.add_argument('--dtype', default='float32', type=str, help='Which precision to use') 
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--epochs-pre', default=5, type=int,help='number of total epochs to run before fine-tuning')
parser.add_argument('--epochs-ft', default=5, type=int,help='number of total epochs to run for finetuning')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float, help='weight decay (default: 1e-4)')  
parser.add_argument('--gpu-id', help='which gpu to use', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--input-size', default=384, type=int, help='Input size M (MxM) default 384')
parser.add_argument('--print-freq', default=20, type=int, help='number of iterations to print the training statistics')


def build_names():
    names = []
    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def get_loss(saved_for_loss, heat_temp, heat_weight,vec_temp, vec_weight):

    names = build_names()
    saved_for_log = OrderedDict()
    total_loss = 0
    loss_fn = gluon.loss.L2Loss()
    for j in range(len(saved_for_loss)//2):
        pred1 = saved_for_loss[2 * j] * vec_weight 
        gt1 = vec_temp * vec_weight
        pred2 = saved_for_loss[2 * j + 1] * heat_weight 
        gt2 = heat_weight * heat_temp

        # Compute losses
        loss1 = loss_fn(pred1, gt1)
        loss2 = loss_fn(pred2, gt2)

        total_loss = total_loss + loss1
        total_loss = total_loss + loss2
        saved_for_log[names[2 * j]] = loss1.mean().asscalar()
        saved_for_log[names[2 * j + 1]] = loss2.mean().asscalar()

    saved_for_log['max_ht'] = saved_for_loss[-1][:, 0:-1, :, :].asnumpy().max()
    saved_for_log['min_ht'] = saved_for_loss[-1][:, 0:-1, :, :].asnumpy().min()
    saved_for_log['max_paf'] = saved_for_loss[-2].asnumpy().max()
    saved_for_log['min_paf'] = saved_for_loss[-2].asnumpy().min()

    return total_loss, saved_for_log
         
def run_epoch(iterator, model, epoch, writer, is_train=True, trainer_trunk=None, trainer_pose=None, dtype='float32', ctx=mx.cpu(), print_freq=20, log_key=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.cast(dtype)
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    end = time.time()
    
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(iterator):
        img = img.as_in_context(ctx).astype(dtype, copy=False)
        heatmap_target = heatmap_target.as_in_context(ctx).astype(dtype, copy=False)
        heat_mask = heat_mask.as_in_context(ctx).astype(dtype, copy=False)
        paf_target = paf_target.as_in_context(ctx).astype(dtype, copy=False)
        paf_mask = paf_mask.as_in_context(ctx).astype(dtype, copy=False)
                
        with autograd.record(is_train):
            # compute output
            out = model(img)
            if type(out[0]) == tuple or type(out[0]) == list: # vgg19 or mobilenet
                total_loss, saved_for_log = get_loss(out[1], heatmap_target, heat_mask,
                       paf_target, paf_mask)
            else: # resnet
                #print(list(out))
                total_loss, saved_for_log = get_loss(list(out), heatmap_target, heat_mask,
                       paf_target, paf_mask)
        
        for name,_ in saved_for_log.items():
            meter_dict[name].update(saved_for_log[name], img.shape[0])
        losses.update(total_loss.astype('float32').mean().asscalar(), img.shape[0])

        if is_train:
            total_loss.backward()
            if trainer_trunk is not None:
                trainer_trunk.step(img.shape[0])
            trainer_pose.step(img.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0 and is_train:
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(iterator)))
            print('Data time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format( batch_time=batch_time))
            print('Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))
            writer.add_scalar('data/max_ht', {log_key:meter_dict['max_ht'].avg}, i+epoch*len(iterator))
            writer.add_scalar('data/max_paf', {log_key:meter_dict['max_paf'].avg}, i+epoch*len(iterator))
            writer.add_scalar('data/loss', {log_key:losses.avg}, i+epoch*len(iterator)),
            for name in saved_for_log:
                print('{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=meter_dict[name]))
            writer.flush()
            print()
    return losses.avg

def train(args):
    
    data_dir = args.data_dir
    mask_dir = args.mask_dir
    logdir = args.logdir
    json_path = args.json_path
    model_path = args.model_path
    load_model = args.load_model
    lr = args.lr
    epochs_ft = args.epochs_ft
    epochs_pre = args.epochs_pre
    wd = args.weight_decay
    batch_size = args.batch_size
    print_freq = args.print_freq
    log_key = args.log_key
    model_trunk = args.trunk
    dtype = args.dtype
    input_size = args.input_size

    ctx = ctx = mx.gpu(args.gpu_id) if args.gpu_id > -1 else mx.cpu()
    params_transform = dict()
    params_transform['mode'] = 5
    # === aug_scale ===
    params_transform['scale_min'] = 0.5
    params_transform['scale_max'] = 1.1
    params_transform['scale_prob'] = 1
    params_transform['target_dist'] = 0.6
    # === aug_rotate ===
    params_transform['max_rotate_degree'] = 40

    # ===
    params_transform['center_perterb_max'] = 40

    # === aug_flip ===
    params_transform['flip_prob'] = 0.5

    params_transform['np'] = 56
    params_transform['sigma'] = 7.0
    params_transform['limb_width'] = 1.

    downsample = 8 if model_trunk == 'mobilenet' or model_trunk == 'vgg19' else 4

    print("Loading dataset...")
    # load data
    train_data = get_loader(json_path, data_dir, mask_dir, 
                            input_size, downsample,batch_size, params_transform = params_transform, shuffle=True, training=True, num_workers=8)
    print('train dataset len: {}'.format(len(train_data._dataset)))

    # validation data
    valid_data = get_loader(json_path, data_dir, mask_dir, input_size,
                                 downsample, training=False,
                                 batch_size=batch_size, params_transform = params_transform, shuffle=False, num_workers=8)
    print('val dataset len: {}'.format(len(valid_data._dataset)))

    # model
    model = build_model(trunk=model_trunk, pretrained_ctx=ctx, is_train=True, num_joints=19)
    if load_model != '':
        model.load_parameters(os.path.join(model_path, load_model))
    model.hybridize(static_alloc=True, static_shape=True)
    
    #logs
    log_key=model_trunk
    writer = SummaryWriter(logdir=logdir)   

    # Different training schemes
    if model_trunk == 'vgg19':
        trainer_trunk = gluon.Trainer(model.model0.collect_params('.*CPM.*'), 'adam', {'learning_rate':lr})
        trainer_pose = gluon.Trainer(model.collect_params('block.*'), 'adam', {'learning_rate':lr}) 
    elif model_trunk =='mobilenet':
        trainer_pose = gluon.Trainer(model.collect_params('block.*'), 'adam', {'learning_rate':lr}) 
        trainer_trunk = None
    elif 'resnet' in model_trunk:
        trainer_trunk = gluon.Trainer(model.collect_params('.*resnet.*'), 'adam', {'learning_rate':lr})
        trainer_pose = gluon.Trainer(model.collect_params('.*final.*'), 'adam', {'learning_rate':lr})                                                                               

    # Training
    for epoch in range(epochs_pre):
        # train for one epoch
        train_loss = run_epoch(train_data, model, epoch, writer, is_train=True, trainer_trunk=trainer_trunk, trainer_pose=trainer_pose, ctx=ctx, dtype=dtype, print_freq=print_freq, log_key=log_key)  
        model.save_parameters(os.path.join(model_path, log_key+'_'+model_trunk+'_pose_'+str(epoch)+'.params'))
        # evaluate on validation set
        val_loss = run_epoch(valid_data, model, epoch, writer, is_train=False, ctx=ctx, dtype=dtype, print_freq=print_freq, log_key=log_key)  

        writer.add_scalar('epoch/train_loss', {log_key: train_loss}, epoch)
        writer.add_scalar('epoch/val_loss', {log_key: val_loss}, epoch)       


    # Fine tuning
    if model_trunk == 'vgg19':
        trainer_trunk = gluon.Trainer(model.model0.collect_params('.*vgg19_.*'), 'adam', {'learning_rate':lr*0.1, 'wd':wd})
        trainer_pose = gluon.Trainer(model.collect_params('block.*'), 'adam', {'learning_rate':lr*0.1, 'wd':wd}) 
    elif model_trunk =='mobilenet':
        trainer_trunk =  gluon.Trainer(model.model0.collect_params('.*mobilenet.*'), 'adam', {'learning_rate':lr*0.1, 'wd':wd})
        trainer_pose = gluon.Trainer(model.collect_params('block.*'), 'adam', {'learning_rate':lr*0.1,  'wd':wd}) 
    elif 'resnet' in model_trunk:
        trainer_trunk = gluon.Trainer(model.collect_params('.*resnet.*'), 'adam', {'learning_rate':lr*0.1})
        trainer_pose = gluon.Trainer(model.collect_params('.*final.*'), 'adam', {'learning_rate':lr*0.1})   
        
    log_key += '_ft'        

    for epoch in range(epochs_pre, epochs_pre+epochs_ft):
        # train for one epoch
        train_loss = run_epoch(train_data, model, epoch, writer, is_train=True, trainer_trunk=trainer_trunk, trainer_pose=trainer_pose, dtype=dtype, ctx=ctx,print_freq=print_freq, log_key=log_key)  
        model.save_parameters(os.path.join(model_path, log_key+'_'+model_trunk+'_pose_ft_'+str(epoch)+'_'+dtype+'.params'))
        # evaluate on validation set
        val_loss = run_epoch(valid_data, model, epoch, writer, is_train=False, ctx=ctx, dtype=dtype, print_freq=print_freq, log_key=log_key)

        writer.add_scalar('epoch_ft/train_loss', {log_key: train_loss}, epoch)
        writer.add_scalar('epoch_ft/val_loss', {log_key: val_loss}, epoch)                                                 
    writer.close()    

    
if __name__ == '__main__':
    
    args = parser.parse_args()  
    print(args)
    train(args)
