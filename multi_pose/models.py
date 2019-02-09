from collections import OrderedDict

import gluoncv as gcv
from gluoncv.model_zoo import get_model
import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import initializer, gluon, autograd, nd
import mxnet.gluon.nn as nn
import numpy as np

class MultiPoseResNet(HybridBlock):
    # Taken from https://github.com/hetong007/gluon-cv pose-estimation branch
    # Credits to hetong007
    
    def __init__(self, base_name='resnet50_v1b',
                 pretrained_base=False, pretrained_ctx=cpu(),
                 num_joints=19,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1, deconv_with_bias=False, **kwargs):
        super(MultiPoseResNet, self).__init__(**kwargs)

        base_network = get_model(base_name, pretrained=pretrained_base, ctx=pretrained_ctx)

        self.resnet = nn.HybridSequential()
        if base_name.endswith('v1'):
            for layer in ['features']:
                self.resnet.add(getattr(base_network, layer))
        else:
            for layer in ['conv1', 'bn1', 'relu', 'maxpool',
                          'layer1', 'layer2', 'layer3', 'layer4']:
                self.resnet.add(getattr(base_network, layer))

        self.deconv_with_bias = deconv_with_bias

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        self.final_layer = nn.Conv2D(
            channels=num_joints*3,
            kernel_size=final_conv_kernel,
            strides=1,
            prefix='final_',
            padding=1 if final_conv_kernel == 3 else 0,
            weight_initializer=initializer.Normal(0.001),
            bias_initializer=initializer.Zero()
        )

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'

        layer = nn.HybridSequential(prefix='final_')
        with layer.name_scope():
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i])

                planes = num_filters[i]
                layer.add(
                    nn.Conv2DTranspose(
                        channels=planes,
                        kernel_size=kernel,
                        strides=2,
                        padding=padding,
                        output_padding=output_padding,
                        use_bias=self.deconv_with_bias,
                        weight_initializer=initializer.Normal(0.001),
                        bias_initializer=initializer.Zero()))
                layer.add(nn.BatchNorm(gamma_initializer=initializer.One(),
                                       beta_initializer=initializer.Zero()))
                layer.add(nn.Activation('relu'))
                self.inplanes = planes

        return layer

    def hybrid_forward(self, F, x):
        x = self.resnet(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        
        x = x.split(3, axis=1)
        vec = F.concat(x[0],x[1], dim=1)
        kp = x[2]
        
        return vec, kp
    
    
class RTPose(nn.HybridBlock):
    def __init__(self, trunk, is_train=True, pretrained_ctx=[mx.cpu()], num_joints=19):
        super(RTPose, self).__init__()
        
        
        self.is_train = is_train
        self.ctx = pretrained_ctx
                 
        blocks = {}

        # Stage 1
        blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L1': [512, num_joints*2, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L2': [512, num_joints, 1, 1, 0]}]

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = [
                {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
                {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_L1' % i: [128, num_joints*2, 1, 1, 0]}
            ]

            blocks['block%d_2' % i] = [
                {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
                {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_L2' % i: [128, num_joints, 1, 1, 0]}
            ]

        models = {}

        if trunk == 'vgg19':
            models['block0'] = self._make_vgg19_block()
        elif trunk == 'mobilenet':
            models['block0'] = self._make_mobilenet_block()


        for k, v in blocks.items():
            models[k] = self._make_stage(k, list(v))
        
        with self.name_scope():
            self.model0 = models['block0']
            self.model1_1 = models['block1_1']
            self.model2_1 = models['block2_1']
            self.model3_1 = models['block3_1']
            self.model4_1 = models['block4_1']
            self.model5_1 = models['block5_1']
            self.model6_1 = models['block6_1']

            self.model1_2 = models['block1_2']
            self.model2_2 = models['block2_2']
            self.model3_2 = models['block3_2']
            self.model4_2 = models['block4_2']
            self.model5_2 = models['block5_2']
            self.model6_2 = models['block6_2']
        self._initialize_weights_norm()
        
        
    def _make_stage(self, prefix, blocks):
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

    def _make_vgg19_block(self):
        """Builds the vgg19 using pre-trained data
        """
        vgg19_block = nn.HybridSequential(prefix='vgg_19_')
        vgg19 = gluon.model_zoo.vision.vgg19(pretrained=True, ctx=self.ctx)
        with vgg19_block.name_scope():
            for i in range(23):
                    vgg19_block.add(vgg19.features[i])
            vgg19_block.add(nn.Conv2D(prefix='conv4_3_CPM', channels=256, kernel_size=3, padding=1))
            vgg19_block.add(nn.Conv2D(prefix='conv4_4_CPM', channels=128, kernel_size=3, padding=1))
        vgg19_block[-2:].initialize(mx.init.Normal(0.01), ctx=self.ctx)
        return vgg19_block

    def _make_mobilenet_block(self):
        """Builds the mobilenet using pre-trained data
        """
        mobilenet_block = nn.HybridSequential(prefix='mobilenet')
        mobilenet = gluon.model_zoo.vision.mobilenet_v2_0_5(pretrained=True, ctx=self.ctx)
        with mobilenet_block.name_scope():
            for i in range(9):
                    mobilenet_block.add(mobilenet.features[i])
        return mobilenet_block


    def _initialize_weights_norm(self):
        
        self.collect_params('block.*bias').initialize(mx.init.Zero(), ctx=self.ctx)
        self.collect_params('block.*weight').initialize(mx.init.Normal(0.01), ctx=self.ctx)
        self.collect_params('block.*batchnorm.*').initialize(mx.init.Zero(), ctx=self.ctx)
        
    
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


def build_model(trunk='vgg19', pretrained_ctx=[mx.cpu()], is_train=True, num_joints=19):
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet' or 'resnet18'
    Returns: Module, the defined model
    """
    
    if 'resnet' in trunk:
        model = MultiPoseResNet(base_name=trunk, pretrained_base=True, pretrained_ctx=pretrained_ctx, num_joints=num_joints)
        model.deconv_layers.initialize(ctx=pretrained_ctx)
        model.final_layer.initialize(ctx=pretrained_ctx)
        return model

    if trunk == 'mobilenet' or trunk == 'vgg19':    
        model = RTPose(trunk, is_train=is_train, pretrained_ctx=pretrained_ctx, num_joints=num_joints)
        return model
    

    

