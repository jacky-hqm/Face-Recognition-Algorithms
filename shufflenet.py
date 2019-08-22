# -*- coding:UTF-8 -*-
import os
import numpy as np
from model_desc import ModelDesc
import tensorflow as tf
import logger


slim = tf.contrib.slim
logging = logger._getlogger('shufflnet')

MEAN = [103.94, 116.78, 123.68]
NORMALIZER = 0.017

def GlobalAvgPooling(x, data_format='NHWC'):
    """
    Global average pooling as in the paper `Network In Network
    <http://arxiv.org/abs/1312.4400>`_.
    Args:
        x (tf.Tensor): a NHWC tensor.
    Returns:
        tf.Tensor: a NC tensor named ``output``.
    """
    assert x.shape.ndims == 4
    assert data_format in ['NHWC', 'NCHW']
    axis = [1, 2] if data_format == 'NHWC' else [2, 3]
    return tf.reduce_mean(x, axis, name='output')

def decode_tfrecord_to_data_label(tf_record_list, num_classes, tf_shape, feature_shape, batch_size, num_epochs=None,normal=False):
    filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=num_epochs, shuffle=True)  # 读取文件名队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    label = tf.cast(features['label'], tf.int32)
    # name = features['name']
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, tf_shape)
    img = tf.image.resize_images(img, (feature_shape[0], feature_shape[1]))
    if normal:
        img = tf.cast(img, tf.float32) * (1. / 255)  # - 0.5
    else:
        img = tf.cast(img, tf.float32)
    img_path_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size,
                                                         capacity=3000, num_threads=2,
                                                         min_after_dequeue=1000, allow_smaller_final_batch=True)
    label_batch = tf.one_hot(label_batch, depth=num_classes)
    return img_path_batch, label_batch

def shufflnet_arg_scope(is_training=True,
                        weight_decay=0.00004,
                        stddev=0.09,
                        regularize_depthwise=False,
                        batch_norm_decay=0.995,
                        batch_norm_epsilon=0.001):
    """Defines the default MobilenetV1 arg scope.

    Args:
      is_training: Whether or not we're training the model. If this is set to
        None, the parameter is not added to the batch_norm arg_scope.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.

    Returns:
      An `arg_scope` to use for the mobilenet v1 model.
    """
    batch_norm_params = {
        'center': True,
        'scale': True,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES]
    }
    if is_training is not None:
        batch_norm_params['is_training'] = is_training

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc


############################################################################################################
# Convolution layer Methods
def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o


def grouped_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0, bias=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups
        conv_side_layers = [
            conv2d(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, bias, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training) for i in
            range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a


def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    return conv_a


############################################################################################################
############################################################################################################
# ShuffleNet unit methods

def shufflenet_unit(name, x, w=None, num_groups=1, group_conv_bottleneck=True, num_filters=16, stride=(1, 1),
                    l2_strength=0.0, bias=0.0, batchnorm_enabled=True, is_training=True, fusion='add'):
    # Paper parameters. If you want to change them feel free to pass them as method parameters.
    activation = tf.nn.relu

    with tf.variable_scope(name) as scope:
        residual = x
        bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[
            3].value) // 4

        if group_conv_bottleneck:
            bottleneck = grouped_conv2d('Gbottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                        padding='VALID',
                                        num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                        activation=activation,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = channel_shuffle('channel_shuffle', bottleneck, num_groups)
        else:
            bottleneck = conv2d('bottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                padding='VALID', l2_strength=l2_strength, bias=bias, activation=activation,
                                batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = bottleneck
        padded = tf.pad(shuffled, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        depthwise = depthwise_conv2d('depthwise', x=padded, w=None, stride=stride, l2_strength=l2_strength,
                                     padding='VALID', bias=bias,
                                     activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
        if stride == (2, 2):
            residual_pooled = avg_pool_2d(residual, size=(3, 3), stride=stride, padding='SAME')
        else:
            residual_pooled = residual

        if fusion == 'concat':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters - residual.get_shape()[3].value,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
        elif fusion == 'add':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            residual_match = residual_pooled
            # This is used if the number of filters of the residual block is different from that
            # of the group convolution.
            if num_filters != residual_pooled.get_shape()[3].value:
                residual_match = conv2d('residual_match', x=residual_pooled, w=None, num_filters=num_filters,
                                        kernel_size=(1, 1),
                                        padding='VALID', l2_strength=l2_strength, bias=bias, activation=None,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(group_conv1x1 + residual_match)
        else:
            raise ValueError("Specify whether the fusion is \'concat\' or \'add\'")


def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output


############################################################################################################
############################################################################################################
# Fully Connected layer Methods

def __dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
              bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        __variable_summaries(bias)
        output = tf.nn.bias_add(tf.matmul(x, w), bias)
        return output


def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength,
                              bias=bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o


############################################################################################################
############################################################################################################
# Pooling Methods

def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='pooling'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)


def avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding=padding,
                          name=name)


############################################################################################################
############################################################################################################
# Utilities for layers
def __variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w


# Summaries for variables
def __variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    pass
    # with tf.name_scope('summaries'):
    #     mean = tf.reduce_mean(var)
    #     tf.summary.scalar('mean', mean)
    #     with tf.name_scope('stddev'):
    #         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #     tf.summary.scalar('stddev', stddev)
    #     tf.summary.scalar('max', tf.reduce_max(var))
    #     tf.summary.scalar('min', tf.reduce_min(var))
    #     tf.summary.histogram('histogram', var)


############################################################################################################


def make_stage(x, output_channels, stage=2, repeat=3, num_groups=3, l2_strength=4e-5, bias=0.0, batchnorm_enabled=True,
               is_training=True):
    if 2 <= stage <= 4:
        stage_layer = shufflenet_unit('stage' + str(stage) + '_0', x=x, w=None,
                                      num_groups=num_groups,
                                      group_conv_bottleneck=not (stage == 2),
                                      num_filters=output_channels[str(num_groups)][stage - 2],
                                      stride=(2, 2),
                                      fusion='concat', l2_strength=l2_strength,
                                      bias=bias,
                                      batchnorm_enabled=batchnorm_enabled,
                                      is_training=is_training)

        for i in range(1, repeat + 1):
            stage_layer = shufflenet_unit('stage' + str(stage) + '_' + str(i),
                                          x=stage_layer, w=None,
                                          num_groups=num_groups,
                                          group_conv_bottleneck=True,
                                          num_filters=output_channels[str(num_groups)][stage - 2],
                                          stride=(1, 1),
                                          fusion='add',
                                          l2_strength=l2_strength,
                                          bias=bias,
                                          batchnorm_enabled=batchnorm_enabled,
                                          is_training=is_training)
        return stage_layer
    else:
        raise ValueError("Stage should be from 2 -> 4")


def shufflnet_base(inputs, output_channels, stage_repeats, num_groups, is_training=True,
                   l2_strength=4e-5, bias=0.0, batchnorm_enabled=True,
                   reuse=None, scope=None):
    end_points = {}
    with tf.variable_scope(scope, 'Base', [inputs], reuse=reuse):
        x_padded = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        conv1 = conv2d('conv1', x=x_padded, w=None, num_filters=output_channels['conv1'], kernel_size=(3, 3),
                       stride=(2, 2), l2_strength=l2_strength, bias=bias,
                       batchnorm_enabled=batchnorm_enabled, is_training=is_training,
                       activation=tf.nn.relu, padding='VALID')
        end_points['conv1'] = conv1
        logging.debug('conv1 :%s', conv1)
        padded = tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
        max_pool = max_pool_2d(padded, size=(3, 3), stride=(2, 2), name='max_pool')
        end_points['maxpool1'] = max_pool
        logging.debug('max_pool :%s', max_pool)

        # stage
        stage2 = make_stage(max_pool, output_channels, stage=2, repeat=stage_repeats[2 - 2], num_groups=num_groups,
                            l2_strength=l2_strength, bias=bias, batchnorm_enabled=batchnorm_enabled,
                            is_training=is_training)
        end_points['stage2'] = stage2
        logging.debug('stage2 :%s', stage2)
        stage3 = make_stage(stage2, output_channels, stage=3, repeat=stage_repeats[3 - 2], num_groups=num_groups,
                            l2_strength=l2_strength, bias=bias, batchnorm_enabled=batchnorm_enabled,
                            is_training=is_training)
        end_points['stage3'] = stage3
        logging.debug('stage3 :%s', stage3)
        stage4 = make_stage(stage3, output_channels, stage=4, repeat=stage_repeats[4 - 2], num_groups=num_groups,
                            l2_strength=l2_strength, bias=bias, batchnorm_enabled=batchnorm_enabled,
                            is_training=is_training)
        end_points['stage4'] = stage4
        logging.debug('stage4 :%s', stage4)

        global_pool = GlobalAvgPooling(stage4)
        end_points['global_pool'] = global_pool
        logging.debug('global_pool :%s', global_pool.shape)
        return global_pool, end_points


def _do_base_logits(inputs, num_groups, stage_repeats,
                    phase_train=True,
                    l2_strength=4e-5, bias=0.0, batchnorm_enabled=True,
                    bottleneck_layer_flag=False, bottleneck_layer_size=128,
                    reuse=None, scope=None
                    ):
    output_channels = {
        'conv1': 24,
        '1': [144, 288, 576],
        '2': [200, 400, 800],
        '3': [240, 480, 960],
        '4': [272, 544, 1088],
        '8': [384, 768, 1536],
    }
    # stage_repeats = [3, 7, 3]
    # num_groups = 1
    # logging.debug('inputs shape %s', inputs.shape)


    #注释
    # with tf.name_scope('Preprocessing'):
    #     red, green, blue = tf.split(inputs, num_or_size_splits=3, axis=3)
    #     preprocessed_input = tf.concat([tf.subtract(blue, MEAN[0]) * NORMALIZER,
    #                                     tf.subtract(green, MEAN[1]) * NORMALIZER,
    #                                     tf.subtract(red, MEAN[2]) * NORMALIZER], 3)
    # logging.debug('after preprocessing shape %s', preprocessed_input.shape)

    base_net, end_points = shufflnet_base(inputs, output_channels, stage_repeats, num_groups, is_training=phase_train,
                                          l2_strength=l2_strength,
                                          bias=bias, batchnorm_enabled=batchnorm_enabled, reuse=reuse, scope=scope)
    logging.debug('base_net %s', base_net)

    base_net = slim.flatten(base_net)
    end_points['PreLogitsFlatten'] = base_net
    if bottleneck_layer_flag:
        base_net = slim.fully_connected(base_net, bottleneck_layer_size, activation_fn=None,
                                        scope='Bottleneck', reuse=False)
    return base_net, end_points


class NetAlgo(ModelDesc):
    def __init__(self, parser):
        super(NetAlgo, self).__init__()
        self.train_public_args(parser)
        self.train_private_args(parser)
        self.loss = {}
        self.accuracy = {}

    # 初始化参数
    def train_private_args(self, parser):
        # 设定各个训练独有的参数
        private_train = parser.add_argument_group('Private', 'Private train parser')
        private_train.add_argument('--l2_strength', type=float, default=4e-5,
                                   help='the l2_strength of shufflenet')
        private_train.add_argument('--bias', type=float, default=0.0,
                                   help='the bias of shufflenet')
        private_train.add_argument('--bottleneck-layer-flag', type=bool, default=False,
                                   help='bottleneck_layer_flag of shufflenet')
        private_train.add_argument('--bottleneck-layer-size', type=int, help='num of net size before full_connected')

        private_train.add_argument('--batchnorm-enabled', type=bool, default=True, help='batchnorm enable flag ')

        private_train.add_argument('--num-groups', type=int, help='the group num of shufflenet')
        private_train.add_argument('--stage-repeats', type=int, help='stage_repeats')

        # private_train.add_argument('--bn-decay', type=float, default=0.995, help='the bn_decay of mobilenet_v1')
        # private_train.add_argument('--bn-epsilon', type=float, default=0.001,
        #                            help='the bn-epsilon of mobilenet_v1')
        # private_train.add_argument('--weight-decay', type=float, default=0.0,
        #                            help='the weight-decay of mobilenet_v1')

        private_train.add_argument('--data-train-tfrecords', type=str, help='train data tfrecords path')
        private_train.add_argument('--data-train-tfrecords-shape', help='train data img shape')
        private_train.add_argument('--data-train-tfrecord', type=str, help='train data tfrecord path')
        private_train.add_argument('--data-val-img', type=str, help='val data img path')
        private_train.add_argument('--data-val-tfrecord', type=str, help='val data tfrecord path')
        private_train.add_argument('--log-path', type=str, help='log path')
        return private_train

    def _do_parse_params(self, parameter):
        '''
        将参数转换成内部变量
        '''
        self.network = parameter.network
        self.mode_type = parameter.mode_type
        if self.mode_type == 'train':
            self.phase_train = True
            logging.info('Train Mode')
        else:
            self.phase_train = False
            logging.info('Test Mode')
        self.num_classes = parameter.num_classes
        self.img_width = parameter.img_width
        self.img_height = parameter.img_height
        self.img_channel = parameter.img_channel
        self.img_shape = (self.img_height, self.img_width, self.img_channel)
        self.data_train_tfrecords = parameter.data_train_tfrecords
        self.data_train_tfrecords_shape = parameter.data_train_tfrecords_shape
        self.num_epochs = parameter.num_epochs
        self.lr = parameter.lr
        self.batch_size = parameter.batch_size
        self.bottleneck_layer_flag = parameter.bottleneck_layer_flag
        self.bottleneck_layer_size = parameter.bottleneck_layer_size
        self.l2_strength = parameter.l2_strength
        self.bias = parameter.bias
        self.batchnorm_enabled = parameter.batchnorm_enabled
        self.num_groups = parameter.num_groups
        self.stage_repeats = parameter.stage_repeats
        # self.dropout_keep_prob = parameter.dropout_keep_prob
        # self.bn_decay = parameter.bn_decay
        # self.bn_epsilon = parameter.bn_epsilon
        # self.weight_decay = parameter.weight_decay
        self.fix_params = parameter.fix_params
        self.log_path = parameter.log_path
        self.gpus = parameter.gpus
        self.fine_tune = parameter.fine_tune
        self.disp_batches = parameter.disp_batches
        self.snapshot_iters = parameter.snapshot_iters
        self.model_prefix = parameter.model_prefix

    # 输入
    def _do_inputs(self):
        inputs_placeholder = tf.placeholder(tf.float32, [None, self.img_height, self.img_width,
                                                         self.img_channel], 'inputs_placeholder')
        labels_placeholder = tf.placeholder(tf.int32, [None, self.num_classes], 'labels_placeholder')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train_placeholder')
        return inputs_placeholder, labels_placeholder, phase_train_placeholder

    # 最后分类层
    def _do_custom_net(self, base_net, end_points):
        logits = slim.fully_connected(base_net, self.num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      weights_regularizer=slim.l2_regularizer(1e-5),
                                      scope='logits', reuse=False)

        end_points['logits'] = logits
        softmax = tf.nn.softmax(logits, name='softmax')
        end_points['softmax'] = softmax
        return logits, end_points

    # loss 损失函数
    def _do_cost(self, logits, labels):
        logging.info('-------- Cost --------')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        self.loss['total_loss'] = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('total_loss', self.loss['total_loss'])

    # 优化器
    def _do_optimizer(self):
        logging.info('-------- Optimizer --------')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # 固定参数
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.fix_params is not None:
            # 指定固定多少参数
            self.train_vars = all_vars[:self.fix_params]
        else:
            self.train_vars = all_vars[:]
        #设置学习率是衰退的
        decay_lr = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=3000, decay_rate=0.9,
                                              staircase=True)
        tf.summary.scalar("lr", decay_lr)

        op_handel = tf.train.AdamOptimizer(learning_rate=decay_lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
        with tf.control_dependencies(update_ops):
            self.optimizer = op_handel.minimize(self.loss['total_loss'],
                                                global_step=self.global_step,
                                                )  # var_list=self.train_vars

    # 准确率计算
    def _do_evaluate(self, logits, labels):
        logging.info('-------- Evaluate --------')
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        self.accuracy['total_accuracy'] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('total_accuracy', self.accuracy['total_accuracy'])

    # 构建训练网络
    def build_graph(self, parameter):
        # 传入参数
        self._do_parse_params(parameter)
        with tf.variable_scope(self.network):
            # 生成 input
            self.inputs_placeholder, self.labels_placeholder, self.phase_train_placeholder = self._do_inputs()
            logging.debug('input  %s', self.inputs_placeholder)
            logging.debug('label  %s', self.labels_placeholder)
            logging.debug('phase  %s', self.phase_train_placeholder)
            # 归一化
            input_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.inputs_placeholder)

            base_net, end_points = _do_base_logits(input_image, num_groups=self.num_groups,
                                                   stage_repeats=self.stage_repeats,
                                                   phase_train=self.phase_train_placeholder,
                                                   l2_strength=self.l2_strength, bias=self.bias,
                                                   batchnorm_enabled=self.batchnorm_enabled,
                                                   bottleneck_layer_flag=self.bottleneck_layer_flag,
                                                   bottleneck_layer_size=self.bottleneck_layer_size,
                                                   )
            feature = tf.identity(base_net, name='feature')
            end_points['feature'] = feature
            logging.debug('feature  %s', feature)
            # 结果
            self.logits, self.end_points = self._do_custom_net(base_net, end_points)
            logging.debug('logits  %s', self.logits)

    # loss 训练op构建
    def build_train_op(self):
        self._do_cost(self.logits, self.labels_placeholder)
        self._do_optimizer()
        self._do_evaluate(self.logits, self.labels_placeholder)

    # 训练
    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus  # 使用 GPU
        train_img_data_batch, train_label_batch = decode_tfrecord_to_data_label([self.data_train_tfrecords],
                                                                                num_classes=self.num_classes,
                                                                                tf_shape=self.data_train_tfrecords_shape,
                                                                                feature_shape=self.img_shape,
                                                                                batch_size=self.batch_size,
                                                                                num_epochs=self.num_epochs)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_path, sess.graph)
            self.saver = tf.train.Saver(max_to_keep=3)#最多保存3个模型
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            # 看有没有模型需要重新加载,初始化参数
            if self.fine_tune:
                # 迁移学习
                logging.info('fine_tune restore')
                all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)#获取所有
                # for v in all_vars:
                #     print(v.name)
                # print("-----------------------------------")
                restore_vars = all_vars[:-2]  # -2
                sess.run(init_op)
                saver = tf.train.Saver(restore_vars)
                saver.restore(sess, self.fine_tune_model)
            else:
                # num_files, files = self.find_previous(os.path.join(self.log_path, self.model_prefix))
                # if num_files == 0:
                #     logging.info('Init No Models')
                #     sess.run(tf.global_variables_initializer())
                # else:
                #     # 加载最后那个模型
                #     logging.info('Restore %s', files[-1])
                #     # 获取Iter
                #     _, start_step = files[-1].split('iter_')
                #     sess.run(tf.global_variables_initializer())
                #     self.saver.restore(sess, files[-1])
                #     tf.assign(self.global_step, int(start_step))
                ckpt = tf.train.get_checkpoint_state(self.log_path)
                if ckpt and ckpt.model_checkpoint_path:
                    logging.info('Restore %s', ckpt.model_checkpoint_path)
                    sess.run(init_op)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    logging.info('Init No Models')
                    sess.run(init_op)

            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                logging.info('start train...')
                while not coord.should_stop():
                    # 准备数据
                    batch_img_path, batch_img_label = sess.run([train_img_data_batch, train_label_batch])
                    # 训练
                    _, summary, loss_out, acc = sess.run(
                        [self.optimizer, self.summary_op, self.loss['total_loss'], self.accuracy['total_accuracy']],
                        feed_dict={self.inputs_placeholder: batch_img_path,
                                   self.labels_placeholder: batch_img_label,
                                   self.phase_train_placeholder: self.phase_train})
                    run_step = sess.run(self.global_step)
                    self.writer.add_summary(summary, run_step)
                    if run_step % self.disp_batches == 0:
                        logging.info("global_step " + str(run_step) + ", Minibatch Loss= " + \
                                     "{:.6f}".format(loss_out) + ", Training Accuracy= " + \
                                     "{:.5f}".format(acc))

                        val_acc = sess.run([self.accuracy['total_accuracy']],
                                           feed_dict={self.inputs_placeholder: batch_img_path,
                                                      self.labels_placeholder: batch_img_label,
                                                      self.phase_train_placeholder: False})
                        logging.info("val_acc " + "{:.5f}".format(val_acc[0]))

                        # print("global_step " + str(run_step) + ", Iter " + str(i) + "/" + str(
                        #     idx * self.config.TRAIN.BATCH_SIZE) + ", Minibatch Loss= " + \
                        #       "{:.6f}".format(loss_out) + ", Training Accuracy= " + \
                        #       "{:.5f}".format(acc))

                    # 保存临时模型
                    if run_step % self.snapshot_iters == 0:
                        # 每snapshot个n_epoch保存一下模型
                        self.saver.save(sess, os.path.join(self.log_path, self.model_prefix + '_iter_' + str(run_step)))

            except tf.errors.OutOfRangeError:
                # OutOfRangeError is thrown when epoch limit per
                # tf.train.limit_epochs is reached.
                logging.info('Stopping Training.')
            finally:
                # 最终保存下模型
                self.writer.close()
                self.saver.save(sess, os.path.join(self.log_path, self.model_prefix))
                coord.request_stop()

            coord.join(threads)


if __name__ == '__main__':
    import argparse

    net_name = 'shufflenet'
    # 设定参数
    parser = argparse.ArgumentParser(description="train",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common_model = NetAlgo(parser)

    # 设定确切的值
    parser.set_defaults(
        # public
        mode_type='train',  # 指定运行模式 train 、predict 、feature
        network=net_name,
        num_classes=17,
        img_width=121,  # 图片宽度
        img_height=121,  # 图片高度
        img_channel=3,  # 图片通道数
        num_epochs=1,  # 训练epoch 个数
        lr=0.001,  # 设置学习率
        bottleneck_layer_flag=False,
        bottleneck_layer_size=128,
        l2_strength=4e-5,
        bias=0.0,
        num_groups=1,
        stage_repeats=[3, 7, 3],
        # bn_decay=0.9,
        batch_size=10,  # 设定batch size
        # dropout_keep_prob=1.0,
        fix_params=None,
        data_train_tfrecords=r'D:\svm\tfrecords\tfrecords_C4_N17998_(171, 171)_img_data.tfrecords',
        data_train_tfrecords_shape=(100, 100, 3),
        log_path=r'./models/shufflenet3',
        gpus='0',
        disp_batches=10,  # 显示logging信息
        snapshot_iters=100,  # 保存临时模型
        model_prefix='tf_shufflenet'
    )
    # 生成参数
    args = parser.parse_args()
    with tf.Graph().as_default():
        common_model.build_graph(args)
        common_model.build_train_op()
        common_model.train()
