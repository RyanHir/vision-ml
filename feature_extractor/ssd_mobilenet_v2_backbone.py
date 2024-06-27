#! /usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from .models.mobilenet_v2 import MobileNetV2

def SSDMobileNetV2Backbone(input_shape, name=None):
    mobilenet = keras.applications.MobileNetV2(input_shape[1:], weights=None)
    if 0:
        conv2d_11_pointwise = mobilenet.get_layer("block_13_expand_relu").output
        conv2d_13_pointwise = mobilenet.get_layer("out_relu").output
    else:
        conv2d_11_pointwise = mobilenet.get_layer("block_5_expand_relu").output
        conv2d_13_pointwise = mobilenet.get_layer("block_13_expand_relu").output
    print(conv2d_11_pointwise)
    print(conv2d_13_pointwise)

    backbone = keras.Model(
            inputs=mobilenet.inputs,
            outputs=[conv2d_11_pointwise, conv2d_13_pointwise],
            name=name)
    return backbone
