from collections import OrderedDict

import mxnet
import mxnet as mx
from mxnet import gluon, autograd, nd
import mxnet.gluon.nn as nn


def make_stage(prefix, blocks):
    stage = nn.HybridSequential(prefix=prefix+'_')
    with stage.name_scope():
        for i, block in enumerate(blocks):
            for k, v in block.items():
                if 'pool' in k:
                    stage.add(nn.MaxPool2D(prefix=k, pool_size=v[0], strides=v[1],
                                            padding=v[2]))
                else:
                    activation = "relu" if i < len(blocks) - 1 else None
                    if i > 0:
                        stage.add(nn.BatchNorm())
                        stage.add(nn.Dropout(rate=0.2))
                    stage.add(nn.Conv2D(prefix=k, channels=v[1],
                                       kernel_size=v[2], strides=v[3],
                                       padding=v[4], activation=activation))

    return stage

def make_vgg19_block():
    """Builds the vgg19 using pre-trained data
    """
    vgg19_block = nn.HybridSequential(prefix='vgg_19_')
    vgg19 = gluon.model_zoo.vision.vgg19(pretrained=True)
    with vgg19_block.name_scope():
        for i in range(23):
                vgg19_block.add(vgg19.features[i])
        vgg19_block.add(nn.Conv2D(prefix='conv4_3_CPM', channels=256, kernel_size=3, padding=1))
        vgg19_block.add(nn.Conv2D(prefix='conv4_4_CPM', channels=128, kernel_size=3, padding=1))
    vgg19_block[-2:].initialize(mx.init.Normal(0.01))
    return vgg19_block

def make_mobilenet_block():
    """Builds the vgg19 using pre-trained data
    """
    mobilenet_block = nn.HybridSequential(prefix='mobilenet')
    mobilenet = gluon.model_zoo.vision.mobilenet_v2_0_5(pretrained=True)
    with mobilenet_block.name_scope():
        for i in range(9):
                mobilenet_block.add(mobilenet.features[i])
        #mobilenet_block.add(nn.Conv2D(prefix='conv4_3_CPM', channels=256, kernel_size=1, padding=1))
        #mobilenet_block.add(nn.Conv2D(prefix='conv4_4_CPM', channels=128, kernel_size=1, padding=1))
    #mobilenet_block[-1:].initialize(mx.init.Normal(0.01))
    return mobilenet_block

def get_model(trunk='vgg19', is_train=True):
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    models = {}

    if trunk == 'vgg19':
        models['block0'] = make_vgg19_block()
    elif trunk == 'mobilenet':
        models['block0'] = make_mobilenet_block()


    for k, v in blocks.items():
        models[k] = make_stage(k, list(v))

    class RTPose(nn.HybridBlock):
        def __init__(self, model_dict, is_train=True):
            super(RTPose, self).__init__()
            with self.name_scope():
                self.model0 = model_dict['block0']
                self.model1_1 = model_dict['block1_1']
                self.model2_1 = model_dict['block2_1']
                self.model3_1 = model_dict['block3_1']
                self.model4_1 = model_dict['block4_1']
                self.model5_1 = model_dict['block5_1']
                self.model6_1 = model_dict['block6_1']

                self.model1_2 = model_dict['block1_2']
                self.model2_2 = model_dict['block2_2']
                self.model3_2 = model_dict['block3_2']
                self.model4_2 = model_dict['block4_2']
                self.model5_2 = model_dict['block5_2']
                self.model6_2 = model_dict['block6_2']

            self._initialize_weights_norm()
            self.is_train = is_train

        def hybrid_forward(self, F, x):


            out1 = self.model0(x)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = F.concat(out1_1, out1_2, out1, dim=1)


            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = F.concat(out2_1, out2_2, out1, dim=1)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = F.concat(out3_1, out3_2, out1, dim=1)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = F.concat(out4_1, out4_2, out1, dim=1)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = F.concat(out5_1, out5_2, out1, dim=1)


            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            if self.is_train:
                saved_for_loss = []
                saved_for_loss.append(out1_1)
                saved_for_loss.append(out1_2)
                saved_for_loss.append(out2_1)
                saved_for_loss.append(out2_2)
                saved_for_loss.append(out3_1)
                saved_for_loss.append(out3_2)
                saved_for_loss.append(out4_1)
                saved_for_loss.append(out4_2)
                saved_for_loss.append(out5_1)
                saved_for_loss.append(out5_2)
                saved_for_loss.append(out6_1)
                saved_for_loss.append(out6_2)
                return (out6_1, out6_2), saved_for_loss
            return (out6_1, out6_2)

        def _initialize_weights_norm(self):
            
            self.collect_params('block.*bias').initialize(mx.init.Zero())
            self.collect_params('block.*weight').initialize(mx.init.Normal(0.01))
            self.collect_params('block.*batchnorm.*').initialize(mx.init.Zero())

    model = RTPose(models, is_train=is_train)
    return model
