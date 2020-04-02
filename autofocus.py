# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.residual_unit import ResidualUnit
from niftynet.network.base_net import BaseNet


class AutofocusSingle(TrainableLayer):
    """
        Implementation of autofocus layer presented in:
        [1] Qin et al., Autofocus Layer for Semantic Segmentation
        https://arxiv.org/abs/1805.08403

        This class defines a block with two layers: The first layer is a
        standard convolutional layer, the second layer is an autofocus layer.

        ### Building blocks:
            [Conv]      - standard convolutional layer
            [AF]        - autofocus block with attention model and K parallel convs
            [Att_1]     - 3x3x3 convolutional layer
            [Att_2]     - 1x1x1 convolutional layer
            [CONV_rX]   - 3x3x3 convolutional layer with dilation rate r1..r4
            [X]         - multiply output of [BN] with corresponding attention weight
                          calculated with the attention model in the first step
            [+]         - weighted summation of activations from parallel convs

        ### Diagram:
            Overview
            (Input)---->[Conv]---->[AF]---->(Output)

            Zoom into an [AF] (attention model)
            (Input)-->[Att_1]-->[PReLU]-->[Att_2]-->[Softmax]-->(Attention weights)

            Zoom into an [AF] (K=4 parallel convs)
            (Input)---->[CONV_r1]---->[BN]---->[X]---->[+]---->(Output)
                    |-->[CONV_r2]---->[BN]---->[X]--|
                    |-->[CONV_r3]---->[BN]---->[X]--|
                    `-->[CONV_r4]---->[BN]---->[X]--´

        ### Hint:
            Instead of creating K parallel dilated convolutions, we are using simply
            a single convolution with K dilated input tensors as input to simulate
            the parallelism.

        ### Constraints:
            - Image spatial window size must be divisible (modulo) by all K dilation
            rates throughout the network
            - The spatial window size of the input to an autofocus layer must be
            divisible by all defined K dilation rates (see example below)

        ### Examples:
            - Appropriate configuration for training:
            image spatial window size = 96 (for fixed dilation rates = 2, 4, 8, 12)
            - Appropriate configuration for inference:
            image spatial window size = 144 (for fixed dilation rates above)
            - For example, if the network downsamples the spatial window size twice
            from 144 to 72 to 36, the dilation rates (2, 4, 8, 12) won't hold for
            size 36 because (36%8)!=0

        ### Defaults:
            - Default values adapted from the original paper [1]
            - K=4 (num_branches) parallel convolutions
            - Kernel size 3x3x3 for parallel convolutions
            - Kernel size 3x3x3 for first attention convolution [Att_1]
            - Kernel size 1x1x1 for second attention convolution [Att_2]
            - Dilation rates = [2, 6, 10, 14] for 4 parallel convolutions
            - Dilation rate = 2 for first standard convolutional layer
            - ReLU activation function
    """

    def __init__(self,
                 n_output_chns = [50, 50],
                 n_input_chns = [40, 50]
                 kernel_size = [[3, 3, 3], [3, 3, 3]],
                 dilation_rates = [[2, 2, 2], [2, 6, 10, 14]]
                 stride = [1, 1, 1],
                 num_branches = 4,
                 acti_func = 'relu'
                 with_res = True,
                 w_initializer = None,
                 w_regularizer = None,
                 b_initializer = None,
                 b_regularizer = None,
                 name = 'Autofocus_Single'):

        """
        :param n_output_chns: List<int>, output channels for both layers
        :param n_input_chns: List<int>, input channels for both layers
        :param kernel_size: List<int>, kernel size for both layers
        :param dilation_rates: List<int>, dilation rates for first conv and K parallel convs
        :param stride: List<int>, stride for first conv layer
        :param num_branches: int, number (K) of parallel convolutions
        :param acti_func: String, activation function to use
        :param with_res: boolean, whether or not to use residual connection
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param name: String, layer name
        """

        super(AutofocusUnit, self).__init__(name = name)

        if hasattr(kernel_size, "__iter__"):  # a list of layer kernel_sizes
            assert (len(kernel_size) == len(strides))
            self.kernel_size = kernel_size
            self.stride = stride
            self.dilation_rates = dilation_rates
        else:  # is a single number (indicating single layer)
            self.kernel_size = [kernel_size]
            self.stride = [stride]
            self.dilation_rates = [dilation_rates]

        # layer properties
        self.n_output_chns = n_output_chns
        self.n_input_chns = n_input_chns
        self.num_branches = num_branches
        self.dilation = dilation_rates
        self.acti_func = acti_func
        self.with_res = with_res
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training):
        """
        :param input_tensor: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :return: tensor, output of the autofocus block
        """
        output_tensor = input_tensor

        ########################################################################
        # 1: Create standard convolutional layer with BN and Activation.
        ########################################################################
        conv_1 = ConvLayer(n_output_chns = self.n_output_chns[0],
                            kernel_size = self.kernel_size[0],
                            stride = self.stride,
                            dilation = self.dilation_rates[0],
                            w_initializer = self.initializers['w'],
                            w_regularizer = self.regularizers['w'],
                            name = 'conv_1')

        bn_1 = BNLayer(regularizer=self.regularizers['w'],
                        name='bn_1')

        acti_1 = ActiLayer(func=self.acti_func,
                            regularizer=self.regularizers['w'],
                            name='acti_1')

        ########################################################################
        # 2: Create single autofocus layer.
        ########################################################################
        # A convolution without feature norm and activation.
        conv_2 = ConvLayer(n_output_chns = self.n_output_chns[1],
                           kernel_size = self.kernel_size[1],
                           padding='SAME',
                           dilation = 1,
                           w_initializer = self.initializers['w'],
                           w_regularizer = self.regularizers['w'],
                           name = 'conv_2')

        # Create two conv layers for the attention model. The output of the
        # attention model will be needed for the K parallel conv layers.

        # First convolutional layer of the attention model (conv l,1).
        conv_att_21 = ConvLayer(n_output_chns = int(self.n_input_chns[1]/2),
                                kernel_size = self.kernel_size[1],
                                padding = 'SAME',
                                w_initializer = self.initializers['w'],
                                w_regularizer = self.regularizers['w'],
                                name = 'af_conv_att_21')

        # Second convolutional layer of the attention model (conv l,2).
        conv_att_22 = ConvLayer(n_output_chns = self.num_branches,
                                kernel_size = [1, 1, 1],
                                padding = 'SAME',
                                w_initializer = self.initializers['w'],
                                w_regularizer = self.regularizers['w'],
                                name = 'af_conv_att_22')

        # Batch norm (BN) layer for each of the K parallel convolutions
        bn_layer_2 = []
        for i in range(self.num_branches):
            bn_layer_2.append(BNLayer(regularizer = self.regularizers['w'],
                                      name = 'bn_layer_2'))

        # Activation function used in the first attention model
        acti_op_2 = ActiLayer(func = self.acti_func,
                              regularizer = self.regularizers['w'],
                              name = 'af_acti_op_2')

        ########################################################################
        # 3: Create other parameterised layers
        ########################################################################
        acti_op = ActiLayer(func = self.acti_func,
                            regularizer = self.regularizers['w'],
                            name = 'acti_op')

        ########################################################################
        # 4: Connect layers
        ########################################################################
        # connect layers
        output_tensor = conv_1(output_tensor, is_training)
        output_tensor = bn_1(output_tensor)
        output_tensor = acti_1(output_tensor)

        # compute attention weights for the K parallel conv layers in the single
        # autofocus convolutional layer
        feature_2 = output_tensor
        att_2 = acti_op_2(conv_att_21(feature_2))
        att_2 = conv_att_22(att_2)
        att_2 = tf.nn.softmax(att_2, axis=1)

        # Create K dilated tensors as input to the autofocus layer. This
        # simulates the K parallel convolutions with different dilation
        # rates. Doing it this way ensures the required weight sharing.
        dilated_tensor_2 = []
        for i in range(self.num_branches):
            dilated_2 = output_tensor
            with DilatedTensor(dilated_2, dilation_factor = self.dilation_rates[1][i]) as dilated:
                dilated.tensor = conv_2(dilated.tensor)
                dilated.tensor = bn_layer_2[i](dilated.tensor, is_training)
            dilated.tensor = dilated.tensor * att_2[:,:,:,:,i:(i+1)]
            dilated_tensor_2.append(dilated.tensor)
        output_tensor = tf.add_n(dilated_tensor_2)

        # make residual connection using ElementwiseLayer with SUM
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)

        # apply the last ReLU activation
        output_tensor = acti_op(output_tensor)

        return output_tensor


class AutofocusDouble(TrainableLayer):
    """
    Implementation of autofocus layer presented in:
    [1] Qin et al., Autofocus Layer for Semantic Segmentation
    https://arxiv.org/abs/1805.08403

    This class defines an autofocus block with two autofocus layers. Each layer
    consists of an attention model and K parallel convolutions.

    ### Building blocks:
        [AF]        - autofocus block with attention model and K parallel convs
        [Att_1]     - 3x3x3 convolutional layer
        [Att_2]     - 1x1x1 convolutional layer
        [CONV_rX]   - 3x3x3 convolutional layer with dilation rate r1..r4
        [X]         - multiply output of [BN] with corresponding attention weight
                      calculated with the attention model in the first step
        [+]         - weighted summation of activations from parallel convs

    ### Diagram:
        Overview
        (Input)---->[AF]---->[AF]---->(Output)

        Zoom into an [AF] (attention model)
        (Input)-->[Att_1]-->[PReLU]-->[Att_2]-->[Softmax]-->(Attention weights)

        Zoom into an [AF] (K=4 parallel convs)
        (Input)---->[CONV_r1]---->[BN]---->[X]---->[+]---->(Output)
                |-->[CONV_r2]---->[BN]---->[X]--|
                |-->[CONV_r3]---->[BN]---->[X]--|
                `-->[CONV_r4]---->[BN]---->[X]--´

    ### Hint:
        Instead of creating K parallel dilated convolutions, we are using simply
        a single convolution with K dilated input tensors as input to simulate
        the parallelism.

    ### Constraints:
        - Image spatial window size must be divisible (modulo) by all K dilation
        rates throughout the network
        - The spatial window size of the input to an autofocus layer must be
        divisible by all defined K dilation rates (see example below)

    ### Examples:
        - Appropriate configuration for training:
        image spatial window size = 96 (for fixed dilation rates = 2, 4, 8, 12)
        - Appropriate configuration for inference:
        image spatial window size = 144 (for fixed dilation rates above)
        - For example, if the network downsamples the spatial window size twice
        from 144 to 72 to 36, the dilation rates (2, 4, 8, 12) won't hold for
        size 36 because (36%8)!=0

    ### Defaults:
        - Default values adapted from the original paper [1]
        - K=4 (num_branches) parallel convolutions
        - Kernel size 3x3x3 for parallel convolutions
        - Kernel size 3x3x3 for first attention convolution [Att_1]
        - Kernel size 1x1x1 for second attention convolution [Att_2]
        - Dilation rates = [2, 6, 10, 14] for 4 parallel convolutions
        - ReLU activation function
    """

    def __init__(self,
                 n_output_chns = [50, 50],
                 n_input_chns = [40, 50],
                 kernel_size = [[3, 3, 3], [3, 3, 3]],
                 dilation_list = [2, 6, 10, 14],
                 strides = [[1, 1, 1], [1, 1, 1]],
                 num_branches = 4,
                 acti_func = 'relu',
                 with_res = True,
                 w_initializer = None,
                 w_regularizer = None,
                 b_initializer = None,
                 b_regularizer = None,
                 name = 'Autofocus_Double'):

        """
        :param n_output_chns: List<int>, output channel for both layers
        :param n_input_chns: List<int>, input channels for both layers
        :param kernel_size: List<List<int>>, kernel size for both layers
        :param dilation_list: List<int>, dilation rates for K parallel convs
        :param strides: List<List<int>>, strides for both layers
        :param num_branches: int, number (K) of parallel convolutions
        :param acti_func: String, activation function to use
        :param with_res: boolean, whether or not to use residual connection
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param name: String, layer name
        """

        super(AFBlock, self).__init__(name = name)

        if hasattr(kernel_size, "__iter__"):  # a list of layer kernel_sizes
            assert (len(kernel_size) == len(strides))
            self.kernel_size = kernel_size
            self.strides = strides
            self.dilation_list = dilation_list
        else:  # is a single number (indicating single layer)
            self.kernel_size = [kernel_size]
            self.strides = [strides]
            self.dilation_list = [dilation_list]

        # layer properties
        self.n_output_chns = n_output_chns
        self.n_input_chns = n_input_chns
        self.num_branches = num_branches
        self.acti_func = acti_func
        self.with_res = with_res
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training):
        """
        :param input_tensor: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :return: tensor, output of the autofocus block
        """
        output_tensor = input_tensor

        ########################################################################
        # 1: Create first of two autofocus layer of autofocus block.
        ########################################################################
        # A convolution without feature norm and activation.
        conv_1 = ConvLayer(n_output_chns = self.n_output_chns[0],
                           kernel_size = self.kernel_size[0],
                           padding='SAME',
                           dilation = 1,
                           w_initializer = self.initializers['w'],
                           w_regularizer = self.regularizers['w'],
                           name = 'conv_1')

        # Create two conv layers for the attention model. The output of the
        # attention model will be needed for the K parallel conv layers.

        # First convolutional layer of the attention model (conv l,1).
        conv_att_11 = ConvLayer(n_output_chns = int(self.n_input_chns[0]/2),
                                kernel_size = self.kernel_size[0],
                                padding = 'SAME',
                                w_initializer = self.initializers['w'],
                                w_regularizer = self.regularizers['w'],
                                name = 'conv_att_11')

        # Second convolutional layer of the attention model (conv l,2).
        conv_att_12 = ConvLayer(n_output_chns = self.num_branches,
                                kernel_size = [1, 1, 1],
                                padding = 'SAME',
                                w_initializer = self.initializers['w'],
                                w_regularizer = self.regularizers['w'],
                                name = 'conv_att_12')

        # Batch norm (BN) layer for each of the K parallel convolutions
        bn_layer_1 = []
        for i in range(self.num_branches):
            bn_layer_1.append(BNLayer(regularizer = self.regularizers['w'],
                                      name = 'bn_layer_1_{}'.format(i)))

        # Activation function used in the first attention model
        acti_op_1 = ActiLayer(func = self.acti_func,
                              regularizer = self.regularizers['w'],
                              name = 'acti_op_1')

        ########################################################################
        # 2: Create second of two autofocus layer of autofocus block.
        ########################################################################
        # A convolution without feature norm and activation.
        conv_2 = ConvLayer(n_output_chns = self.n_output_chns[1],
                           kernel_size = self.kernel_size[1],
                           padding='SAME',
                           dilation = 1,
                           w_initializer = self.initializers['w'],
                           w_regularizer = self.regularizers['w'],
                           name = 'conv_2')

        # Create two conv layers for the attention model. The output of the
        # attention model will be needed for the K parallel conv layers.
        # First convolutional layer of the attention model (conv l,1).
        conv_att_21 = ConvLayer(n_output_chns = int(self.n_input_chns[1]/2),
                                kernel_size = self.kernel_size[1],
                                padding = 'SAME',
                                w_initializer = self.initializers['w'],
                                w_regularizer = self.regularizers['w'],
                                name = 'conv_att_21')

        # Second convolutional layer of the attention model (conv l,2).
        conv_att_22 = ConvLayer(n_output_chns = self.num_branches,
                                kernel_size = [1, 1, 1],
                                padding = 'SAME',
                                w_initializer = self.initializers['w'],
                                w_regularizer = self.regularizers['w'],
                                name = 'conv_att_22')

        # Batch norm (BN) layer for each of the K parallel convolutions
        bn_layer_2 = []
        for i in range(self.num_branches):
            bn_layer_2.append(BNLayer(regularizer = self.regularizers['w'],
                                      name = 'bn_layer_2_{}'.format(i)))

        # Activation function used in the second attention model
        acti_op_2 = ActiLayer(func = self.acti_func,
                              regularizer = self.regularizers['w'],
                              name = 'acti_op_2')

        ########################################################################
        # 3: Create other parameterised layers
        ########################################################################
        acti_op = ActiLayer(func = self.acti_func,
                            regularizer = self.regularizers['w'],
                            name = 'acti_op')

        ########################################################################
        # 4: Connect layers
        ########################################################################
        # compute attention weights for the K parallel conv layers in the first
        # autofocus convolutional layer
        feature_1 = output_tensor
        att_1 = acti_op_1(conv_att_11(feature_1))
        att_1 = conv_att_12(att_1)
        att_1 = tf.nn.softmax(att_1, axis=1)

        # Create K dilated tensors as input to the autofocus layer. This
        # simulates the K parallel convolutions with different dilation
        # rates. Doing it this way ensures the required weight sharing.
        dilated_tensor_1 = []
        for i in range(self.num_branches):
            dilated_1 = output_tensor
            with DilatedTensor(dilated_1, dilation_factor = self.dilation_list[i]) as dilated:
                dilated.tensor = conv_1(dilated.tensor)
                dilated.tensor = bn_layer_1[i](dilated.tensor, is_training)
            dilated.tensor = dilated.tensor * att_1[:,:,:,:,i:(i+1)]
            dilated_tensor_1.append(dilated.tensor)
        output_tensor = tf.add_n(dilated_tensor_1)
        output_tensor = acti_op(output_tensor)

        # compute attention weights for the K parallel conv layers in the second
        # autofocus convolutional layer
        feature_2 = output_tensor
        att_2 = acti_op_2(conv_att_21(feature_2))
        att_2 = conv_att_22(att_2)
        att_2 = tf.nn.softmax(att_2, axis=1)

        # Create K dilated tensors as input to the autofocus layer. This
        # simulates the K parallel convolutions with different dilation
        # rates. Doing it this way ensures the required weight sharing.
        dilated_tensor_2 = []
        for i in range(self.num_branches):
            dilated_2 = output_tensor
            with DilatedTensor(dilated_2, dilation_factor = self.dilation_list[i]) as dilated:
                dilated.tensor = conv_2(dilated.tensor)
                dilated.tensor = bn_layer_2[i](dilated.tensor, is_training)
            dilated.tensor = dilated.tensor * att_2[:,:,:,:,i:(i+1)]
            dilated_tensor_2.append(dilated.tensor)
        output_tensor = tf.add_n(dilated_tensor_2)

        # make residual connection using ElementwiseLayer with SUM
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)

        # apply the last ReLU activation
        output_tensor = acti_op(output_tensor)
        print("output_tensor:", output_tensor)

        return output_tensor
