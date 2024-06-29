#! /usr/bin/env python3

import tensorflow.keras as keras

def SSDMobileNetV2Backbone(input_shape, name=None):
    mobilenet = keras.applications.MobileNetV2(input_shape[1:], weights=None, include_top=False)

    conv2d_11_pointwise = mobilenet.get_layer("block_5_expand_relu").output
    conv2d_13_pointwise = mobilenet.get_layer("block_12_expand_relu").output

    backbone = keras.Model(
            inputs=mobilenet.inputs,
            outputs=[conv2d_11_pointwise, conv2d_13_pointwise],
            name=name)
    return backbone

