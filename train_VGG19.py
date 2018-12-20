import argparse
import time
import os
import numpy as np
from collections import OrderedDict

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, loss

from network.rtpose_vgg import get_model
from training.datasets.coco import get_loader
from mxboard import SummaryWriter    

# Hyper-params
parser = argparse.ArgumentParser(description='MXNet rtpose Training')
parser.add_argument('--data_dir', default='./training/dataset/COCO/images', type=str, metavar='DIR',
                    help='path to where coco images stored') 
parser.add_argument('--mask_dir', default='./training/dataset/COCO/mask', type=str, metavar='DIR',
                    help='path to where coco images stored')    
parser.add_argument('--logdir', default='logs', type=str, metavar='DIR',
                    help='path to where tensorboard log restore')                                       
parser.add_argument('--json_path', default='./training/dataset/COCO/COCO.json', type=str, metavar='PATH',
                    help='path to where coco images stored')                                      
parser.add_argument('--model_path', default='./model_checkpoints', type=str, metavar='DIR',
                    help='path to where the model saved') 
parser.add_argument('--load_model', default='', type=str,
                    help='which model to load') 
parser.add_argument('--log_key', default='new_experiment', type=str,
                    help='Which key to use for mxboard') 
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
 
parser.add_argument('--epochs-pre', default=5, type=int, metavar='N',
                    help='number of total epochs to run before fine-tuning')
parser.add_argument('--epochs-ft', default=5, type=int, metavar='N',
                    help='number of total epochs to run for finetuning')

parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  
parser.add_argument('-o', '--optim', default='sgd', type=str)
#Device options
parser.add_argument('--gpu_ids', dest='gpu_ids', help='which gpu to use', nargs="+",
                    default=[0], type=int)
                    
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')

args = parser.parse_args()  
print(args)
data_dir = args.data_dir
mask_dir = args.mask_dir
logdir = args.logdir
json_path = args.json_path
model_path = args.model_path
load_model = args.load_model
lr = args.lr
momentum = args.momentum
epochs_ft = args.epochs_ft
epochs_pre = args.epochs_pre
wd = args.weight_decay        
optim = args.optim
gpuIDs = args.gpu_ids
batch_size = args.batch_size
print_freq = args.print_freq
log_key = args.log_key

ctx = [mx.gpu(e) for e in args.gpu_ids] if args.gpu_ids[0] != -1 else [mx.cpu()]
ctx = ctx[0] # single GPU for now
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
        
def build_names():
    names = []
    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def get_loss(saved_for_loss, heat_temp, heat_weight,
               vec_temp, vec_weight):

    names = build_names()
    saved_for_log = OrderedDict()
    loss_fn = gluon.loss.L2Loss()
    total_loss = 0

    for j in range(6):
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
         
def run_epoch(iterator, model, epoch, is_train=True, trainer_vgg=None, trainer_pose=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    end = time.time()
    
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(iterator):
        img = img.as_in_context(ctx)
        heatmap_target = heatmap_target.as_in_context(ctx)
        heat_mask = heat_mask.as_in_context(ctx)
        paf_target = paf_target.as_in_context(ctx)
        paf_mask = paf_mask.as_in_context(ctx)
                
        with autograd.record(is_train):
            # compute output
            _,saved_for_loss = model(img)

            total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
                   paf_target, paf_mask)

            for name,_ in meter_dict.items():
                meter_dict[name].update(saved_for_log[name], img.shape[0])
            losses.update(total_loss.mean().asscalar(), img.shape[0])

        if is_train:
            total_loss.backward()
            trainer_vgg.step(img.shape[0])
            trainer_pose.step(img.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0 and is_train:
#            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(iterator)))
#            print('Data time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format( batch_time=batch_time))
             print('Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))
             writer.add_scalar('data/max_ht', {log_key:meter_dict['max_ht'].avg}, i+epoch*len(iterator))
             writer.add_scalar('data/max_paf', {log_key:meter_dict['max_paf'].avg}, i+epoch*len(iterator))
             writer.add_scalar('data/loss', {log_key:losses.avg}, i+epoch*len(iterator)),
#            for name, value in meter_dict.items():
#                print('{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value))
             writer.flush()
    return losses.avg
    

print("Loading dataset...")
# load data
train_data = get_loader(json_path, data_dir,
                        mask_dir, 368, 8,
                        'vgg', batch_size, params_transform = params_transform, 
                        shuffle=True, training=True, num_workers=8)
print('train dataset len: {}'.format(len(train_data._dataset)))

# validation data
valid_data = get_loader(json_path, data_dir, mask_dir, 368,
                            8, preprocess='vgg', training=False,
                            batch_size=batch_size, params_transform = params_transform, shuffle=False, num_workers=8)
print('val dataset len: {}'.format(len(valid_data._dataset)))

# model

model = get_model(trunk='vgg19')
model.collect_params().reset_ctx(ctx)
if load_model != '':
    model.load_parameters(os.path.join(model_path, load_model))
model.hybridize()

# Fix the VGG pre-trained weights for now
trainer_vgg = gluon.Trainer(model.model0.collect_params('.*CPM.*'), 'sgd', {'learning_rate':lr, 'momentum': momentum, 'wd':wd})
trainer_pose = gluon.Trainer(model.collect_params('block.*'), 'sgd', {'learning_rate':lr, 'momentum': momentum, 'wd':wd}) 
                                                                                          
writer = SummaryWriter(logdir=logdir)       
for epoch in range(epochs_pre):
    # train for one epoch
    train_loss = run_epoch(train_data, model, epoch, is_train=True, trainer_vgg=trainer_vgg, trainer_pose=trainer_pose)
    model.save_parameters(os.path.join(model_path, log_key+'_vgg_pose_'+str(epoch)+'.params'))
    # evaluate on validation set
    val_loss = run_epoch(valid_data, model, epoch, is_train=False)  
                  
    writer.add_scalar('epoch/train_loss', {log_key: train_loss}, epoch)
    writer.add_scalar('epoch/val_loss', {log_key: val_loss}, epoch)       
    
if optim == 'sgd':
    trainer_vgg = gluon.Trainer(model.model0.collect_params(), 'sgd', {'learning_rate':lr, 'momentum': momentum, 'wd':wd})
    trainer_pose = gluon.Trainer(model.collect_params('block.*'), 'sgd', {'learning_rate':lr, 'momentum': momentum, 'wd':wd}) 
elif optim == 'adam':
    trainer_vgg = gluon.Trainer(model.model0.collect_params(), 'adam', {'learning_rate':lr, 'wd':wd})
    trainer_pose = gluon.Trainer(model.collect_params('block.*'), 'adam', {'learning_rate':lr, 'wd':wd}) 
else:
    raise "Unknown optim " + optim
log_key += '_ft'        

for epoch in range(epochs_pre, epochs_pre+epochs_ft):
    # train for one epoch
    train_loss = run_epoch(train_data, model, epoch, is_train=True, trainer_vgg=trainer_vgg, trainer_pose=trainer_pose)
    model.save_parameters(os.path.join(model_path, log_key+'_vgg_pose_'+str(epoch)+'.params'))
    # evaluate on validation set
    val_loss = run_epoch(valid_data, model, epoch, is_train=False)  
                                 
    writer.add_scalar('epoch_ft/train_loss', {log_key: train_loss}, epoch)
    writer.add_scalar('epoch_ft/val_loss', {log_key: val_loss}, epoch)                                                                

        
writer.close()    
