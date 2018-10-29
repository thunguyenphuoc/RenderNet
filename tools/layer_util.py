import numpy as np
import tensorflow.contrib.layers as layers
import tensorflow as tf

"""
Helper functions to build up a CNN
"""

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x, name=name)

def prelu(x, trainable=True, alpha = None):
    """
    Creating parametric ReLU with variable alpha

    :param weight_name: name of the weight to load
    :param weight_dict: the dictionary that is used to stored all the weight
    """
    if alpha is None:
        alpha = tf.get_variable(
            name='alpha',
            shape=x.get_shape()[-1],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
            trainable=trainable)
    else:
        print("Load alpha")
        alpha = tf.get_variable(name = 'alpha', initializer = alpha, dtype = tf.float32, trainable = trainable)

    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

def get_weight(weight_name, weight_dict):
    """
    Return a numpy array of weights from a pretrained model given the layer name.
    Return None if training from scratch (no pretrained weights to use)

    :param weight_name: name of the weight to load
    :param weight_dict: the dictionary that is used to stored all the weight
    """
    if weight_dict is None:
        return None
    else:
        return weight_dict.get(weight_name)  # returns None if name is not found in dictionary

def res_block_3d(input, out_channels=64, scope = 'res_block', kernel=[3, 3, 3], stride=[1, 1, 1], weight_dict=None):
    """
    Create residual block comprises of a 3D convolution, Relu, and another 3D convolution.
    The input if concatnated with the output of the second convolution

    """
    with tf.variable_scope(scope):
        net = tf.nn.relu(conv3d(input, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="con1_3X3",
                                weight_initializer=get_weight(scope + 'con1_3X3_weights', weight_dict),
                                bias_initializer=get_weight(scope + 'con1_3X3_biases', weight_dict),
                                weight_initializer_type=tf.contrib.layers.xavier_initializer()))

        net = conv3d(net, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="conv2_3x3",
                     weight_initializer=get_weight(scope + 'conv2_3x3_weights', weight_dict),
                     bias_initializer=get_weight(scope + 'conv2_3x3_biases', weight_dict),
                     weight_initializer_type=tf.contrib.layers.xavier_initializer())

    return tf.add(tf.cast(net, tf.float32), tf.cast(input, tf.float32))


def res_block_2d(input, out_channels=64, scope = 'res_block', kernel=[3, 3], stride=[1, 1], weight_dict=None):
    """
    Create residual block comprises of a 2D convolution, Relu, and another 2D convolution.
    The input if concatnated with the output of the second convolution

    """
    with tf.variable_scope(scope):
        net = tf.nn.relu(conv2d(input, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="con1_3X3",
                                weight_initializer=get_weight(scope + 'con1_3X3_weights', weight_dict),
                                bias_initializer=get_weight(scope + 'con1_3X3_biases', weight_dict),
                                weight_initializer_type=tf.contrib.layers.xavier_initializer()))

        net = conv2d(net, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="conv2_3x3",
                     weight_initializer=get_weight(scope + 'conv2_3x3_weights', weight_dict),
                     bias_initializer=get_weight(scope + 'conv2_3x3_biases', weight_dict),
                     weight_initializer_type=tf.contrib.layers.xavier_initializer())

    return tf.add(tf.cast(net, tf.float32), tf.cast(input, tf.float32))

# def binary_activation(x, threshold):
#     cond = tf.less(x, tf.constant(threshold))
#     out = tf.where(cond, tf.fill(tf.shape(x),1e-2), tf.fill(tf.shape(x),(1-(1e-2))))
#     return out

def keep_prob(dropout, train):
    """
    Returning a dropout rate or no dropout depending on the training mode (train or no train)
    :param dropout: Drop out probability
    :param  train: train mode (tf.binary, True or False)
    :return dropout probability
    """
    return tf.cond(train, lambda: tf.constant(dropout), lambda: tf.constant(1.))

def bias_variable(shape, bias_initializer=None, trainable=True):
    """
    Creating bias (to be used with convolution layer)
    :param  shape: Desired shape of the bias tensor
    :param  bias_initilizer: Initilizer option for bias. If bias_initializer =None, a constant (0.001) is used
    :param  trainable:  whether the tensor can be trained or not (useful when loading pretrained model for testing
                    of further training)
    """
    if bias_initializer is None :
        return tf.get_variable(name='biases', initializer=tf.constant(0.001, shape=shape), trainable = trainable)
    else:
        return tf.get_variable(name='biases', initializer=bias_initializer, trainable = trainable)


def conv2d(input_, num_outputs, kernel_size=[4,4], stride=[1,1], pad='SAME', if_bias=True, trainable=True, reuse=False,
           scope='conv2d', weight_initializer=None, bias_initializer=None,
           weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    """
    Creating a 2D convolution (to be used with bias_variable)
    :param  shape: Desired shape of the bias tensor
    :param  bias_initilizer: Initilizer option for bias. If bias_initializer =None, a constant (0.001) is used
    :param  trainable:  whether the tensor can be trained or not (useful when loading pre-trained model for testing
                    of further training)
    """
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size +  [input_.get_shape()[-1]] + [num_outputs],
                                initializer = weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer = weight_initializer,
                                dtype = tf.float32, trainable=trainable)

        conv = tf.nn.conv2d(input_, w,
                            padding = pad,
                            strides = [1] + stride + [1])

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv=conv + bias_variable([num_outputs], trainable=trainable)
            else:
                print("Loading biases")
                conv=conv + bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)

        return conv


def conv2d_transpose(x, num_outputs, kernel_size = (4,4), stride= (1,1), pad='SAME', if_bias=True,
                     reuse=False, scope = "conv2d_transpose", trainable = True, weight_initializer=None,
                     bias_initializer=None, weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    """
    Creating a 2D up-convolution (to be used with bias_variable)
    :param  shape: Desired shape of the bias tensor
    :param  bias_initilizer: Initilizer option for bias. If bias_initializer =None, a constant (0.001) is used
    :param  trainable:  whether the tensor can be trained or not (useful when loading pre-trained model for testing
                    of further training)
    """
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size + [num_outputs] + [x.get_shape()[-1]],
                                initializer=weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer=weight_initializer,
                                dtype = tf.float32, trainable=trainable)

        output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], num_outputs]

        conv_trans = tf.nn.conv2d_transpose(x, w,
                                            output_shape = output_shape,
                                            strides = [1] + stride + [1],
                                            padding = pad)

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv_trans = conv_trans + bias_variable([num_outputs], trainable=trainable)
            else:
                print("Load biases")
                conv_trans = conv_trans + bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)


        return conv_trans

def conv3d(input_, num_outputs, pad = "SAME", reuse = False, kernel_size = [4,4,4], stride = [2,2,2], if_bias= True,
           trainable = True, scope = "conv3d", weight_initializer=None, bias_initializer=None, weight_initializer_type = tf.random_normal_initializer(stddev=0.02)):
    """
    Creating a 3D convolution (to be used with bias_variable)
    :param  shape: Desired shape of the bias tensor
    :param  bias_initilizer: Initilizer option for bias. If bias_initializer =None, a constant (0.001) is used
    :param  trainable:  whether the tensor can be trained or not (useful when loading pre-trained model for testing
                    of further training)
    """
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initialise weight")
            w = tf.get_variable(name='weights',
                                trainable=trainable,
                                shape=kernel_size + [input_.get_shape()[-1]] + [num_outputs],
                                initializer=weight_initializer_type,
                                dtype=tf.float32)
        else:
            print("Loading weight")
            w = tf.get_variable(name='weights',
                                trainable=trainable,
                                initializer=weight_initializer,
                                dtype=tf.float32)

        conv = tf.nn.conv3d(input_, w,
                            padding = pad,
                            strides = [1] + stride + [1])

        if if_bias:
            if bias_initializer is None:
                print("Initialise bias")
                conv = conv + bias_variable([num_outputs], trainable=trainable)
            else:
                print("Loading bias")
                conv = conv + bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)

        return conv



def conv3d_transpose(x, num_output, kernel_size = (4,4), stride= (1,1), pad='SAME', if_bias=True,
                     reuse=False, scope = "conv3d_transpose", trainable = True, weight_initializer=None,
                     bias_initializer=None, weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    """
    Creating a 3D up-convolution (to be used with bias_variable)
    :param  shape: Desired shape of the bias tensor
    :param  bias_initilizer: Initilizer option for bias. If bias_initializer =None, a constant (0.001) is used
    :param  trainable:  whether the tensor can be trained or not (useful when loading pre-trained model for testing
                    of further training)
    """
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size + [num_output] + [x.get_shape().as_list()[-1]],
                                initializer=weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer=weight_initializer,
                                dtype = tf.float32, trainable=trainable)
        print("W " + str(w.get_shape()))
        output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], tf.shape(x)[3] * stride[2], num_output]

        conv_trans = tf.nn.conv3d_transpose(x, w,
                                            output_shape = output_shape,
                                            strides = [1] + stride + [1],
                                            padding = pad)

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv_trans = conv_trans + bias_variable([num_output], trainable=trainable)
            else:
                print("Load biases")
                conv_trans = conv_trans + bias_variable([num_output], trainable=trainable, bias_initializer=bias_initializer)


        return conv_trans

def fully_connected(input_, output_size, reuse = False, scope = 'fully_connected', if_bias=True,
                    weight_initializer=None, bias_initializer=None, trainable = True,
                    weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    """
    Creating a fully-connected layer (to be used with bias_variable)
    :param  shape: Desired shape of the bias tensor
    :param  bias_initilizer: Initilizer option for bias. If bias_initializer =None, a constant (0.001) is used
    :param  trainable:  whether the tensor can be trained or not (useful when loading pre-trained model for testing
                    of further training)
    """
    print(scope)
    if (type(input_)== np.ndarray):
        shape = input_.shape
    else:
        shape = input_.get_shape().as_list()
        # shape = tf.shape(input_)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            matrix = tf.get_variable("weights", [shape[1], output_size], initializer=weight_initializer_type, dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            matrix = tf.get_variable("weights", initializer=weight_initializer, dtype=tf.float32, trainable=trainable)

        fc = tf.matmul(input_, matrix)
        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                fc = fc + bias_variable([output_size], trainable=trainable)
            else:
                print("Load biases")
                fc = fc + bias_variable([output_size], bias_initializer, trainable=trainable)
        return fc

