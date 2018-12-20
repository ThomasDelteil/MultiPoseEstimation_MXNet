from evaluate.coco_eval import run_eval
from network.rtpose_vgg import get_model
import mxnet as mx

def evaluate(ctx = mx.gpu(3), weight_path='./model_checkpoints/vgg_pose_ft_6.params'):
    
    model = get_model(trunk='vgg19')
    model.collect_params().reset_ctx(ctx)
    model.load_parameters(weight_path, ctx=ctx)
    model.hybridize(static_shape=True, static_alloc=True)
    run_eval(image_dir= './training/dataset/COCO/images/', anno_dir = './training/dataset/COCO/', vis_dir = './training/dataset/COCO/vis',
        image_list_txt='./evaluate/image_info_val2014_1k.txt', 
        model=model, preprocess='vgg', ctx=ctx)


