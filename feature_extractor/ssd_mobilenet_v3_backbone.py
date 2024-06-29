#! /usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def SSDMobileNetV3Backbone(input_shape, name=None, *args, **kwargs):
    mobilenet = keras.applications.MobileNetV3Large(input_shape[1:], weights=None, include_top=False, include_preprocessing=False, *args, **kwargs)
    for i, layer in enumerate(mobilenet.layers):
        print(f"layer_{i}", layer.name, layer.output.shape)
    conv2d_11_pointwise = mobilenet.layers[83].output
    conv2d_13_pointwise = mobilenet.layers[167].output
    # conv2d_13_pointwise = mobilenet.get_layer("expanded_conv_5/depthwise").output

    backbone = keras.Model(
            inputs=mobilenet.inputs,
            outputs=[conv2d_11_pointwise, conv2d_13_pointwise],
            name=name)
    return backbone
