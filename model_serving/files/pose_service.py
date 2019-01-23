import cv2
import mxnet as mx
from mxnet import gluon, nd
import os

ctx=mx.gpu(0)

SIZE = 368
LEFT_WRIST_INDEX = 4
RIGHT_WRIST_INDEX = 7

class PoseEstimationService(object):
    """
    Pose Estimation Service
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params):
        symbol_file = 'export_mobilenet_heatmap-symbol.json'
        param_file = 'export_mobilenet_heatmap-0000.params'

        gpu_id = params.system_properties.get("gpu_id")
        model_dir = params.system_properties.get("model_dir")
        param_file_path = os.path.join(model_dir, param_file)
        symbol_file_path = os.path.join(model_dir, symbol_file)
        

        self.transform = mx.gluon.data.vision.transforms.Compose(
            [
                mx.gluon.data.vision.transforms.Resize(SIZE, True),
                mx.gluon.data.vision.transforms.CenterCrop(SIZE, cv2.INTER_CUBIC),
                mx.gluon.data.vision.transforms.ToTensor(),
                mx.gluon.data.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        
        if not os.path.isfile(param_file_path):
            raise OSError("Parameter file not found {}".format(param_file_path))
        if not os.path.isfile(symbol_file_path):
            raise OSError("Symbol file not found {}".format(symbol_file_path))
        
        self.ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)

        self.net = gluon.nn.SymbolBlock.imports(
            symbol_file=symbol_file_path,
            input_names=['data'],
            ctx=self.ctx)
        self.net.load_parameters(param_file_path, ctx=self.ctx, ignore_extra=True)
        self.initialized = True
        self.net.hybridize(static_shape=True, static_alloc=True)
        self.net(mx.nd.ones((1, 3, SIZE, SIZE), ctx=self.ctx)) #warm-up


    def preprocess(self, data):
        """
        Pre-process text to a encode it to a form, that gives spatial information to the CNN
        """
        img = data[0].get("body")
        img = mx.image.imdecode(img, 1, True)
        img = self.transform(img)
        data = img.as_in_context(self.ctx).expand_dims(axis=0)
        
        return data

    def inference(self, data):
        # Call forward/hybrid_forward
        heatmap = self.net(data)
        return heatmap
    
    def postprocess(self, data):
        # Post process and output the most likely category
        left_wrist = data[0, LEFT_WRIST_INDEX, :, :].expand_dims(axis=2).as_in_context(mx.cpu())
        right_wrist = data[0, RIGHT_WRIST_INDEX, :, :].expand_dims(axis=2).as_in_context(mx.cpu())
        left_wrist_hr = mx.image.imresize(left_wrist, SIZE, SIZE, interp=cv2.INTER_CUBIC)
        right_wrist_hr = mx.image.imresize(right_wrist, SIZE, SIZE, interp=cv2.INTER_CUBIC)
        lw_index = left_wrist_hr.asnumpy().argmax()
        lw_y = lw_index // SIZE
        lw_x = lw_index % SIZE

        rw_index = right_wrist_hr.asnumpy().argmax()
        rw_y = rw_index // SIZE
        rw_x = rw_index % SIZE
        return [{'lw_x': lw_x / float(SIZE),
                 'lw_y': lw_y / float(SIZE),
                 'rw_x': rw_x / float(SIZE),
                 'rw_y': rw_y / float(SIZE)}]
    
    def predict(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        return self.postprocess(data)


svc = PoseEstimationService()

def pose_inference(data, context):
    res = ""
    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        res = svc.predict(data)

    return res
