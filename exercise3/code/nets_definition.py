from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}


    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
        conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

        print("Conv1 shape")
        print(conv1.get_shape())

    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    #print("Conv 1")
    #print(x.get_shape())

    #180x180x24
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2)
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)

    print("Block One dim ")
    print(x)

    DB2_skip_connection = x
    #90x90x32
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)

    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    #45x45x96
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)

    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x
    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)

    print("Block Four dim ")
    print(x)

    #23x23x320
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)

    print("Block Four dim ")
    print(x)


    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        #input is features named 'x'

        # TODO(1.1) - incorporate a upsample function which takes the features of x
        # and produces 120 output feature maps, which are 16x bigger in resolution than
        # x. Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up5

        factor = 16
        k_size = 2*factor - factor % 2
        current_up5 = TransitionUp_elu(input=x, filters=120, strideN=factor, name='test')
        cropped = crop(current_up5, conv1)

        End_maps_decoder1 = slim.conv2d(cropped, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        #input is features named 'x'

        # TODO (2.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps


        # TODO (2.2) - incorporate a upsample function which takes the features from TODO (2.1)
        # and produces 120 output feature maps, which are 8x bigger in resolution than
        # TODO (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3

        current_up3 = TransitionUp_elu(input=x, filters=120, strideN=2, name='current_up3')
        print("\nshape after Transition", current_up3.shape)

        current_up3 = crop(current_up3, DB4_skip_connection)
        print("\nshape after cropping", current_up3.shape)

        current_up3 = Concat_layers(current_up3, DB4_skip_connection)
        print("\nshape after concat", current_up3.shape)

        current_up3 = Convolution(current=current_up3, out_features=256, kernel_size=3, name='current_up31')
        print("\nshape after convolution", current_up3.shape)


        current_up3 = TransitionUp_elu(current_up3, filters=120, strideN=8, name='current_up32')
        print("\nshape after transition2", current_up3.shape)

        current_up3 = crop(current_up3, self.tgt_image)
        print("\nshape after second cropping", current_up3.shape)

        End_maps_decoder1 = slim.conv2d(current_up3, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        #input is features named 'x'

        # TODO (3.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        current_up4 = TransitionUp_elu(input=x, filters=120, strideN=2, name='current_up4')
        print("\nshape after Transition", current_up4.shape)

        current_up4 = crop(current_up4, DB4_skip_connection)
        print("\nshape after cropping", current_up4.shape)

        current_up4 = Concat_layers(current_up4, DB4_skip_connection)
        print("\nshape after concat", current_up4.shape)

        current_up4 = Convolution(current=current_up4, out_features=256, kernel_size=3, name='current_up41')
        print("\nshape after convolution", current_up4.shape)

        # TODO (3.2) - Repeat TODO(3.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.
        current_up4 = TransitionUp_elu(input=current_up4, filters=160, strideN=2, name='current_up42')
        print("\nshape after second Transition", current_up4.shape)

        current_up4 = crop(current_up4, DB3_skip_connection)
        print("\nshape after second cropping", current_up4.shape)

        current_up4 = Concat_layers(current_up4, DB3_skip_connection)
        print("\nshape after second concat", current_up4.shape)

        #current_up4 = Convolution(current=current_up4, out_features=256, kernel_size=3, name='current_up31')
        #print("\nshape after second convolution", current_up4.shape)



        # TODO (3.3) - incorporate a upsample function which takes the features from TODO (3.2)
        # and produces 120 output feature maps which are 4x bigger in resolution than
        # TODO (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4

        current_up4 = TransitionUp_elu(input=current_up4, filters=120, strideN=4, name='current_up43')
        print("\nshape after third transition", current_up4.shape)

        current_up4 = crop(current_up4, self.tgt_image)
        print("\nshape after third cropping", current_up4.shape)

        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration
    if self.configuration == 4:

        ######################################################################################
        ######################################### DECODER Full #############################################



        # TODO (4.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        current_up6 = TransitionUp_elu(input=x, filters=120, strideN=2, name='current_up6')
        print("\nshape after Transition", current_up6.shape)

        current_up6 = crop(current_up6, DB4_skip_connection)
        print("\nshape after cropping", current_up6.shape)

        current_up6 = Concat_layers(current_up6, DB4_skip_connection)
        print("\nshape after concat", current_up6.shape)

        current_up6 = Convolution(current=current_up6, out_features=256, kernel_size=3, name='current_up61')
        print("\nshape after convolution", current_up6.shape)

        # TODO (4.2) - Repeat TODO(4.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        current_up6 = TransitionUp_elu(input=current_up6, filters=160, strideN=2, name='current_up62')
        print("\nshape after second Transition", current_up6.shape)

        current_up6 = crop(current_up6, DB3_skip_connection)
        print("\nshape after second cropping", current_up6.shape)

        current_up6 = Concat_layers(current_up6, DB3_skip_connection)
        print("\nshape after second concat", current_up6.shape)

        # TODO (4.3) - Repeat TODO(4.2) now producing 96 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.

        current_up6 = TransitionUp_elu(input=current_up6, filters=96, strideN=2, name='current_up63')
        print("\nshape after second Transition", current_up6.shape)

        current_up6 = crop(current_up6, DB2_skip_connection)
        print("\nshape after second cropping", current_up6.shape)

        current_up6 = Concat_layers(current_up6, DB2_skip_connection)
        print("\nshape after second concat", current_up6.shape)

        # TODO (4.4) - incorporate a upsample function which takes the features from TODO(4.3)
        # and produce 120 output feature maps which are 2x bigger in resolution than
        # TODO(4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4

        current_up6 = TransitionUp_elu(input=current_up6, filters=120, strideN=2, name='current_up64')
        print("\nshape after second Transition", current_up6.shape)

        current_up6 = crop(current_up6, self.tgt_image)
        print("\nshape after second cropping", current_up6.shape)

        End_maps_decoder1 = slim.conv2d(current_up6, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    return Reshaped_map
