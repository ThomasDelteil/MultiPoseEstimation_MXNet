import argparse
import os

from multi_pose.coco_eval import run_eval
from multi_pose.models import build_model
import mxnet as mx


# Hyper-params
parser = argparse.ArgumentParser(description='MXNet rtpose Evaluation')
parser.add_argument('--data-dir', default='./data/dataset/COCO/images', type=str, metavar='DIR', help='path to where coco images stored') 
parser.add_argument('--anno-dir', default='./data/dataset/COCO/', type=str, metavar='DIR', help='path to where coco annotation stored')    
parser.add_argument('--vis-dir', default='./data/dataset/COCO/vis', type=str, metavar='DIR', help='path to store vis')                                       
parser.add_argument('--image-list', default='./data/image_info_val2014_1k.txt', type=str, metavar='DIR', help='path to annotations')                                      
parser.add_argument('--model-path', default='./model_checkpoints', type=str, metavar='DIR', help='path to where the model saved') 
parser.add_argument('--load-model', default='resnet18_v1brefactor_resnet18_v1b_pose_0.params', type=str, help='which model to load') 
parser.add_argument('--trunk', default='resnet18_v1b', type=str, help='Which backend to use [mobilenet, vgg19, resnet18_v1b]') 
parser.add_argument('--gpu-id', default=0, help='which gpu to use', type=int)
parser.add_argument('--input-size', default=384, type=int, help='Input size M (MxM) default 384')
parser.add_argument('--num-joints', default=19, type=int, help='Number of joints')


def evaluate(args):
    ctx = mx.gpu(args.gpu_id) if args.gpu_id > -1 else mx.cpu()
    num_joints = args.num_joints
    model = build_model(trunk=args.trunk, pretrained_ctx=ctx, is_train=False, num_joints=num_joints)
    model.load_parameters(os.path.join(args.model_path, args.load_model), ctx=ctx)
    model.hybridize(static_shape=True, static_alloc=True)
    
    downsampling = 8 if args.trunk == 'mobilenet' or args.trunk == 'vgg19' else 4

    run_eval(image_dir=args.data_dir, anno_dir =args.anno_dir, vis_dir =args.vis_dir, image_list_txt=args.image_list, 
             model=model, ctx=ctx, input_size=args.input_size, num_joints=args.num_joints, downsampling=downsampling)


if __name__ == '__main__':
    args = parser.parse_args()  
    print(args)
    evaluate(args)
