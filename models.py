# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from .autofocus import AutofocusDouble

from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet


class WNet3d(BaseNet):
    """
    Implementation of WNet in
    Wang et al., Automatic Brain Tumor Segmentation using Cascaded Anisotropic
    Convolutional Neural Networks
    https://arxiv.org/abs/1709.00382

    Taken from NiftyNet and adapted with kernels of size 3x3x3 and 1x1x1
    to create a 3D WNet version.
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='selu',
                 name='WNet3d'):

        super(WNet3d, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.acti_func = acti_func
        self.base_chns = [32, 32, 32, 32]

    def layer_op(self, images, is_training):
        block1_1 = ResBlock(self.base_chns[0],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block1_1')

        block1_2 = ResBlock(self.base_chns[0],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block1_2')

        block2_1 = ResBlock(self.base_chns[1],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block2_1')

        block2_2 = ResBlock(self.base_chns[1],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block2_2')

        block3_1 = ResBlock(self.base_chns[2],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            dilation_rates=[[1, 1, 1], [1, 1, 1]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block3_1')

        block3_2 = ResBlock(self.base_chns[2],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            dilation_rates=[[2, 2, 2], [2, 2, 2]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block3_2')

        block3_3 = ResBlock(self.base_chns[2],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            dilation_rates=[[3, 3, 3], [3, 3, 3]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block3_3')

        block4_1 = ResBlock(self.base_chns[3],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            dilation_rates=[[3, 3, 3], [3, 3, 3]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block4_1')

        block4_2 = ResBlock(self.base_chns[3],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            dilation_rates=[[2, 2, 2], [2, 2, 2]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block4_2')

        block4_3 = ResBlock(self.base_chns[3],
                            kernels=[[3, 3, 3], [3, 3, 3]],
                            dilation_rates=[[1, 1, 1], [1, 1, 1]],
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='block4_3')

        fuse1 = ConvolutionalLayer(self.base_chns[0],
                                   kernel_size=[1, 1, 1],
                                   padding='VALID',
                                   w_initializer=self.initializers['w'],
                                   w_regularizer=self.regularizers['w'],
                                   b_initializer=self.initializers['b'],
                                   b_regularizer=self.regularizers['b'],
                                   acti_func=self.acti_func,
                                   with_bn=self.acti_func != 'selu',
                                   name='fuse1')

        downsample1 = ConvolutionalLayer(self.base_chns[0],
                                         kernel_size=[3, 3, 3],
                                         stride=[2, 2, 2],
                                         padding='SAME',
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         b_initializer=self.initializers['b'],
                                         b_regularizer=self.regularizers['b'],
                                         acti_func=self.acti_func,
                                         with_bn=self.acti_func != 'selu',
                                         name='downsample1')

        fuse2 = ConvolutionalLayer(self.base_chns[1],
                                   kernel_size=[1, 1, 1],
                                   padding='VALID',
                                   w_initializer=self.initializers['w'],
                                   w_regularizer=self.regularizers['w'],
                                   b_initializer=self.initializers['b'],
                                   b_regularizer=self.regularizers['b'],
                                   acti_func=self.acti_func,
                                   with_bn=self.acti_func != 'selu',
                                   name='fuse2')

        downsample2 = ConvolutionalLayer(self.base_chns[1],
                                         kernel_size=[3, 3, 3],
                                         stride=[2, 2, 2],
                                         padding='SAME',
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         b_initializer=self.initializers['b'],
                                         b_regularizer=self.regularizers['b'],
                                         acti_func=self.acti_func,
                                         with_bn=self.acti_func != 'selu',
                                         name='downsample2')

        fuse3 = ConvolutionalLayer(self.base_chns[2],
                                   kernel_size=[1, 1, 1],
                                   padding='VALID',
                                   w_initializer=self.initializers['w'],
                                   w_regularizer=self.regularizers['w'],
                                   b_initializer=self.initializers['b'],
                                   b_regularizer=self.regularizers['b'],
                                   acti_func=self.acti_func,
                                   with_bn=self.acti_func != 'selu',
                                   name='fuse3')

        fuse4 = ConvolutionalLayer(self.base_chns[3],
                                   kernel_size=[1, 1, 1],
                                   padding='VALID',
                                   w_initializer=self.initializers['w'],
                                   w_regularizer=self.regularizers['w'],
                                   b_initializer=self.initializers['b'],
                                   b_regularizer=self.regularizers['b'],
                                   acti_func=self.acti_func,
                                   with_bn=self.acti_func != 'selu',
                                   name='fuse4')

        feature_expand1 = ConvolutionalLayer(
            self.base_chns[1],
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding='SAME',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            b_initializer=self.initializers['b'],
            b_regularizer=self.regularizers['b'],
            acti_func=self.acti_func,
            with_bn=self.acti_func != 'selu',
            name='feature_expand1')

        feature_expand2 = ConvolutionalLayer(
            self.base_chns[2],
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding='SAME',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            b_initializer=self.initializers['b'],
            b_regularizer=self.regularizers['b'],
            acti_func=self.acti_func,
            with_bn=self.acti_func != 'selu',
            name='feature_expand2')

        feature_expand3 = ConvolutionalLayer(
            self.base_chns[3],
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding='SAME',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            b_initializer=self.initializers['b'],
            b_regularizer=self.regularizers['b'],
            acti_func=self.acti_func,
            with_bn=self.acti_func != 'selu',
            name='feature_expand3')

        centra_slice1 = TensorSliceLayer(margin=2)
        centra_slice2 = TensorSliceLayer(margin=1)
        pred_up1 = DeconvolutionalLayer(self.num_classes,
                                        kernel_size=[3, 3, 3],
                                        stride=[2, 2, 2],
                                        padding='SAME',
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        b_initializer=self.initializers['b'],
                                        b_regularizer=self.regularizers['b'],
                                        acti_func=self.acti_func,
                                        with_bn=self.acti_func != 'selu',
                                        name='pred_up1')
        pred_up2_1 = DeconvolutionalLayer(self.num_classes * 2,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up2_1')
        pred_up2_2 = DeconvolutionalLayer(self.num_classes * 2,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up2_2')
        pred_up3_1 = DeconvolutionalLayer(self.num_classes * 4,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up3_1')
        pred_up3_2 = DeconvolutionalLayer(self.num_classes * 4,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up3_2')

        final_pred = ConvLayer(self.num_classes,
                               kernel_size=[3, 3, 3],
                               padding='SAME',
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               b_initializer=self.initializers['b'],
                               b_regularizer=self.regularizers['b'],
                               name='final_pred')

        f1 = images
        f1 = block1_1(f1, is_training=is_training)
        f1 = block1_2(f1, is_training=is_training)
        f1 = fuse1(f1, is_training=is_training)
        f1 = downsample1(f1, is_training=is_training)
        if self.base_chns[0] != self.base_chns[1]:
            f1 = feature_expand1(f1, is_training=is_training)
        f1 = block2_1(f1, is_training=is_training)
        f1 = block2_2(f1, is_training=is_training)
        f1 = fuse2(f1, is_training=is_training)

        f2 = downsample2(f1, is_training=is_training)
        if self.base_chns[1] != self.base_chns[2]:
            f2 = feature_expand2(f2, is_training=is_training)
        f2 = block3_1(f2, is_training=is_training)
        f2 = block3_2(f2, is_training=is_training)
        f2 = block3_3(f2, is_training=is_training)
        f2 = fuse3(f2, is_training=is_training)

        f3 = f2
        if self.base_chns[2] != self.base_chns[3]:
            f3 = feature_expand3(f3, is_training)
        f3 = block4_1(f3, is_training=is_training)
        f3 = block4_2(f3, is_training=is_training)
        f3 = block4_3(f3, is_training=is_training)
        f3 = fuse4(f3, is_training=is_training)

        p1 = f1
        p1 = pred_up1(p1, is_training=is_training)

        p2 = f2
        p2 = pred_up2_1(p2, is_training=is_training)
        p2 = pred_up2_2(p2, is_training=is_training)

        p3 = pred_up3_1(f3, is_training=is_training)
        p3 = pred_up3_2(p3, is_training=is_training)

        cat = tf.concat([p1, p2, p3], axis=4, name='concate')
        pred = final_pred(cat)
        return pred


class AutofocusNet(BaseNet):
    """
    Implementation of Autofocus Net in
    Stefani, A., Rahmat, R., Harris-Birtill, D. (2020). Autofocus Net: Auto-focused 3D CNN for Brain Tumour Segmentation.
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='AutofocusNet'):

        super(AutofocusNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.acti_func = acti_func
        # base channels for 4 residual blocks
        self.base_chns = [32, 32, 32, 32]
        # base channels for 6 autofocus blocks
        self.af_base_chns = [[32, 32], [32, 32], [32, 32],
                             [32, 32], [32, 32], [32, 32]]
        # input channels for 6 autofocus blocks, needed for the attention model
        # inside an autofocus layer
        self.af_in_chns = [[32, 32], [32, 32], [32, 32],
                           [32, 32], [32, 32], [32, 32]]

    def layer_op(self, images, is_training):
        block1_1 = ResBlock(self.base_chns[0],
                            kernels = [[3, 3, 3], [3, 3, 3]],
                            acti_func = self.acti_func,
                            w_initializer = self.initializers['w'],
                            w_regularizer = self.regularizers['w'],
                            name = 'block1_1')

        block1_2 = ResBlock(self.base_chns[1],
                            kernels = [[3, 3, 3], [3, 3, 3]],
                            acti_func = self.acti_func,
                            w_initializer = self.initializers['w'],
                            w_regularizer = self.regularizers['w'],
                            name = 'block1_2')

        block2_1 = ResBlock(self.base_chns[2],
                            kernels = [[3, 3, 3], [3, 3, 3]],
                            acti_func = self.acti_func,
                            w_initializer = self.initializers['w'],
                            w_regularizer = self.regularizers['w'],
                            name = 'block2_1')

        block2_2 = ResBlock(self.base_chns[3],
                            kernels = [[3, 3, 3], [3, 3, 3]],
                            acti_func = self.acti_func,
                            w_initializer = self.initializers['w'],
                            w_regularizer = self.regularizers['w'],
                            name = 'block2_2')

        block3_1 = AutofocusDouble(self.af_base_chns[0],
                                   n_input_chns = self.af_in_chns[0],
                                   kernel_size = [[3,3,3], [3,3,3]],
                                   dilation_list = [2,4,8,12],
                                   num_branches = 4,
                                   acti_func = self.acti_func,
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   name = 'block3_1')

        block3_2 = AutofocusDouble(self.af_base_chns[1],
                                   n_input_chns = self.af_in_chns[1],
                                   kernel_size = [[3,3,3], [3,3,3]],
                                   dilation_list = [2,4,8,12],
                                   num_branches = 4,
                                   acti_func = self.acti_func,
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   name = 'block3_2')

        block3_3 = AutofocusDouble(self.af_base_chns[2],
                                   n_input_chns = self.af_in_chns[2],
                                   kernel_size = [[3,3,3], [3,3,3]],
                                   dilation_list = [2,4,8,12],
                                   num_branches = 4,
                                   acti_func = self.acti_func,
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   name = 'block3_3')

        block4_1 = AutofocusDouble(self.af_base_chns[3],
                                   n_input_chns = self.af_in_chns[3],
                                   kernel_size = [[3,3,3], [3,3,3]],
                                   dilation_list = [2,4,8,12],
                                   num_branches = 4,
                                   acti_func = self.acti_func,
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   name = 'block4_1')

        block4_2 = AutofocusDouble(self.af_base_chns[4],
                                   n_input_chns = self.af_in_chns[4],
                                   kernel_size = [[3,3,3], [3,3,3]],
                                   dilation_list = [2,4,8,12],
                                   num_branches = 4,
                                   acti_func = self.acti_func,
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   name = 'block4_2')

        block4_3 = AutofocusDouble(self.af_base_chns[5],
                                   n_input_chns = self.af_in_chns[5],
                                   kernel_size = [[3,3,3], [3,3,3]],
                                   dilation_list = [2,4,8,12],
                                   num_branches = 4,
                                   acti_func = self.acti_func,
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   name = 'block4_3')

        fuse1 = ConvolutionalLayer(self.base_chns[0],
                                   kernel_size = [1, 1, 1],
                                   padding = 'VALID',
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   b_initializer = self.initializers['b'],
                                   b_regularizer = self.regularizers['b'],
                                   acti_func = self.acti_func,
                                   with_bn = self.acti_func != 'selu',
                                   name = 'fuse1')

        downsample1 = ConvolutionalLayer(self.base_chns[0],
                                         kernel_size = [3, 3, 3],
                                         stride = [2, 2, 2],
                                         padding = 'SAME',
                                         w_initializer = self.initializers['w'],
                                         w_regularizer = self.regularizers['w'],
                                         b_initializer = self.initializers['b'],
                                         b_regularizer = self.regularizers['b'],
                                         acti_func = self.acti_func,
                                         with_bn = self.acti_func != 'selu',
                                         name = 'downsample1')

        fuse2 = ConvolutionalLayer(self.base_chns[1],
                                   kernel_size = [1, 1, 1],
                                   padding = 'VALID',
                                   w_initializer = self.initializers['w'],
                                   w_regularizer = self.regularizers['w'],
                                   b_initializer = self.initializers['b'],
                                   b_regularizer = self.regularizers['b'],
                                   acti_func = self.acti_func,
                                   with_bn = self.acti_func != 'selu',
                                   name = 'fuse2')

        downsample2 = ConvolutionalLayer(self.base_chns[1],
                                         kernel_size=[3, 3, 3],
                                         stride=[2, 2, 2],
                                         padding='SAME',
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         b_initializer=self.initializers['b'],
                                         b_regularizer=self.regularizers['b'],
                                         acti_func=self.acti_func,
                                         with_bn=self.acti_func != 'selu',
                                         name='downsample2')

        fuse3 = ConvolutionalLayer(self.base_chns[2],
                                   kernel_size=[1, 1, 1],
                                   padding='VALID',
                                   w_initializer=self.initializers['w'],
                                   w_regularizer=self.regularizers['w'],
                                   b_initializer=self.initializers['b'],
                                   b_regularizer=self.regularizers['b'],
                                   acti_func=self.acti_func,
                                   with_bn=self.acti_func != 'selu',
                                   name='fuse3')

        fuse4 = ConvolutionalLayer(self.base_chns[3],
                                   kernel_size=[1, 1, 1],
                                   padding='VALID',
                                   w_initializer=self.initializers['w'],
                                   w_regularizer=self.regularizers['w'],
                                   b_initializer=self.initializers['b'],
                                   b_regularizer=self.regularizers['b'],
                                   acti_func=self.acti_func,
                                   with_bn=self.acti_func != 'selu',
                                   name='fuse4')

        feature_expand1 = ConvolutionalLayer(
            self.base_chns[1],
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding='SAME',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            b_initializer=self.initializers['b'],
            b_regularizer=self.regularizers['b'],
            acti_func=self.acti_func,
            with_bn=self.acti_func != 'selu',
            name='feature_expand1')

        feature_expand2 = ConvolutionalLayer(
            self.base_chns[2],
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding='SAME',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            b_initializer=self.initializers['b'],
            b_regularizer=self.regularizers['b'],
            acti_func=self.acti_func,
            with_bn=self.acti_func != 'selu',
            name='feature_expand2')

        feature_expand3 = ConvolutionalLayer(
            self.base_chns[3],
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding='SAME',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            b_initializer=self.initializers['b'],
            b_regularizer=self.regularizers['b'],
            acti_func=self.acti_func,
            with_bn=self.acti_func != 'selu',
            name='feature_expand3')

        centra_slice1 = TensorSliceLayer(margin=2)
        centra_slice2 = TensorSliceLayer(margin=1)
        pred_up1 = DeconvolutionalLayer(self.num_classes,
                                        kernel_size=[3, 3, 3],
                                        stride=[2, 2, 2],
                                        padding='SAME',
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        b_initializer=self.initializers['b'],
                                        b_regularizer=self.regularizers['b'],
                                        acti_func=self.acti_func,
                                        with_bn=self.acti_func != 'selu',
                                        name='pred_up1')
        pred_up2_1 = DeconvolutionalLayer(self.num_classes * 2,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up2_1')
        pred_up2_2 = DeconvolutionalLayer(self.num_classes * 2,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up2_2')
        pred_up3_1 = DeconvolutionalLayer(self.num_classes * 4,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up3_1')
        pred_up3_2 = DeconvolutionalLayer(self.num_classes * 4,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          padding='SAME',
                                          w_initializer=self.initializers['w'],
                                          w_regularizer=self.regularizers['w'],
                                          b_initializer=self.initializers['b'],
                                          b_regularizer=self.regularizers['b'],
                                          acti_func=self.acti_func,
                                          with_bn=self.acti_func != 'selu',
                                          name='pred_up3_2')

        final_pred = ConvLayer(self.num_classes,
                               kernel_size=[3, 3, 3],
                               padding='SAME',
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               b_initializer=self.initializers['b'],
                               b_regularizer=self.regularizers['b'],
                               name='final_pred')

        f1 = images
        f1 = block1_1(f1, is_training=is_training)
        f1 = block1_2(f1, is_training=is_training)
        f1 = fuse1(f1, is_training=is_training)
        f1 = downsample1(f1, is_training=is_training)
        if self.base_chns[0] != self.base_chns[1]:
            f1 = feature_expand1(f1, is_training=is_training)
        f1 = block2_1(f1, is_training=is_training)
        f1 = block2_2(f1, is_training=is_training)
        f1 = fuse2(f1, is_training=is_training)

        f2 = downsample2(f1, is_training=is_training)
        if self.base_chns[1] != self.base_chns[2]:
            f2 = feature_expand2(f2, is_training=is_training)
        f2 = block3_1(f2, is_training=is_training)
        f2 = block3_2(f2, is_training=is_training)
        f2 = block3_3(f2, is_training=is_training)
        f2 = fuse3(f2, is_training=is_training)

        f3 = f2
        if self.base_chns[2] != self.base_chns[3]:
            f3 = feature_expand3(f3, is_training)
        f3 = block4_1(f3, is_training=is_training)
        f3 = block4_2(f3, is_training=is_training)
        f3 = block4_3(f3, is_training=is_training)
        f3 = fuse4(f3, is_training=is_training)

        p1 = f1
        p1 = pred_up1(p1, is_training=is_training)

        p2 = f2
        p2 = pred_up2_1(p2, is_training=is_training)
        p2 = pred_up2_2(p2, is_training=is_training)
        p3 = pred_up3_1(f3, is_training=is_training)
        p3 = pred_up3_2(p3, is_training=is_training)

        cat = tf.concat([p1, p2, p3], axis=4, name='concate')
        pred = final_pred(cat)

        return pred


class ResBlock(TrainableLayer):
    """
    This class define a high-resolution block with residual connections
    kernels - specify kernel sizes of each convolutional layer
            - e.g.: kernels=(5, 5, 5) indicates 3 conv layers of kernel_size 5
    with_res - whether to add residual connections to bypass the conv layers
    """

    def __init__(self,
                 n_output_chns,
                 kernels=[[1, 3, 3], [1, 3, 3]],
                 strides=[[1, 1, 1], [1, 1, 1]],
                 dilation_rates=[[1, 1, 1], [1, 1, 1]],
                 acti_func='selu',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.n_output_chns = n_output_chns
        if hasattr(kernels, "__iter__"):  # a list of layer kernel_sizes
            assert (len(kernels) == len(strides))
            assert (len(kernels) == len(dilation_rates))
            self.kernels = kernels
            self.strides = strides
            self.dilation_rates = dilation_rates
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
            self.strides = [strides]
            self.dilation_rates = [dilation_rates]
        self.acti_func = acti_func
        self.with_res = with_res

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        if self.acti_func == 'selu':
            return self.layer_op_selu(input_tensor, is_training)
        else:
            return self.layer_op_prelu(input_tensor, is_training)

    def layer_op_prelu(self, input_tensor, is_training):
        output_tensor = input_tensor
        for i in range(len(self.kernels)):
            # create parameterised layers
            bn_op = BNLayer(regularizer=self.regularizers['w'],
                            name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=self.kernels[i],
                                stride=self.strides[i],
                                dilation=self.dilation_rates[i],
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            # connect layers
            output_tensor = bn_op(output_tensor, is_training)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)
        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor

    def layer_op_selu(self, input_tensor, is_training):
        output_tensor = input_tensor
        for i in range(len(self.kernels)):
            # create parameterised layers
            #             bn_op = BNLayer(regularizer=self.regularizers['w'],
            #                             name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=self.kernels[i],
                                stride=self.strides[i],
                                dilation=self.dilation_rates[i],
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            # connect layers
            #             output_tensor = bn_op(output_tensor, is_training)
            output_tensor = conv_op(output_tensor)
            output_tensor = acti_op(output_tensor)

        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor


class TensorSliceLayer(TrainableLayer):
    """
    extract the central part of a tensor
    """

    def __init__(self, margin=1, regularizer=None, name='tensor_extract'):
        self.layer_name = name
        super(TensorSliceLayer, self).__init__(name=self.layer_name)
        self.margin = margin

    def layer_op(self, input_tensor):
        input_shape = input_tensor.get_shape().as_list()
        begin = [0] * len(input_shape)
        begin[1] = self.margin
        size = input_shape
        size[1] = size[1] - 2 * self.margin
        output_tensor = tf.slice(input_tensor, begin, size, name='slice')
        return output_tensor
