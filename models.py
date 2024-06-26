from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import math

from anchors import Anchor, create_ssd_anchors

LAST_REDUCE_BOXES = True
LAST_NUM_BOXES = 3 if LAST_REDUCE_BOXES else 6

class Bottleneck(keras.Model):
    def __init__(self, expansion, filters, strides, name, alpha=1):
        super(Bottleneck, self).__init__(name=name)
        self.filters = filters
        self.strides = strides
        self.expansion = expansion
        self.output_channels = alpha * filters
        self.alpha = alpha
        self.out = None

    def build(self, input_shape):
        self.d = int(input_shape[3])
        # Expand
        self.expand = layers.Conv2D(
                filters = self.expansion*self.d,
                kernel_size = 1,
                use_bias = False,
                name="expand")
        self.expand_bn = layers.BatchNormalization(name="expand_BN")
        self.expand_relu = layers.ReLU(6.0, name="expand_ReLU")
        
        # Conv
        self.conv = layers.DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same', use_bias=False, name="conv")
        self.conv_bn = layers.BatchNormalization(name="conv_BN")
        self.conv_relu = layers.ReLU(6.0, name="conv_ReLU")

        # Projection
        self.project = layers.Conv2D(filters=self.filters, kernel_size=1, use_bias=False, name="contract")
        self.project_bn = layers.BatchNormalization(name="contract_BN")

        self.residual = layers.Add(name="residual")

    def call(self, inputs):
        x = self.expand(inputs)
        x = self.expand_bn(x)
        x = self.expand_relu(x)
        self.out = x

        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.conv_relu(x)

        x = self.project(x)
        x = self.project_bn(x)

        if self.output_channels == self.d and self.strides == 1:
            x = self.residual([inputs, x])
        return x

class MobileNetV2(keras.Model):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
                filters = 32,
                kernel_size = 3,
                strides = (2, 2),
                padding='valid',
                use_bias = False)
        self.bn = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name="conv_bn")
        self.relu = layers.ReLU(6.0, name="conv_relu")

        self.bn1 = Bottleneck(1, 16, 1, name="BN1")
        
        self.bn2 = Bottleneck(6, 24, 2, name="BN2_1")
        self.bn3 = Bottleneck(6, 24, 1, name="BN2_2")
        
        self.bn4 = Bottleneck(6, 32, 2, name="BN3_1")
        self.bn5 = Bottleneck(6, 32, 1, name="BN3_2")
        self.bn6 = Bottleneck(6, 32, 1, name="BN3_3")
        
        self.bn5 = Bottleneck(6, 64, 2, name="BN4_1")
        self.bn6 = Bottleneck(6, 64, 1, name="BN4_2")
        self.bn7 = Bottleneck(6, 64, 1, name="BN4_3")
        self.bn8 = Bottleneck(6, 64, 1, name="BN4_4")
        
        self.bn9  = Bottleneck(6, 96, 1, name="BN5_1")
        self.bn10 = Bottleneck(6, 96, 1, name="BN5_2")
        self.bn11 = Bottleneck(6, 96, 1, name="BN5_3")
        
        self.bn12 = Bottleneck(6, 160, 2, name="BN6_1")
        self.bn13 = Bottleneck(6, 160, 1, name="BN6_2")
        self.bn14 = Bottleneck(6, 160, 1, name="BN6_3")
        
        self.bn15 = Bottleneck(6, 320, 1, name="BN7")
        self.conv2 = layers.Conv2D(filters = 1280, kernel_size=1, strides=(1,1), use_bias=False)
        self.avgpool = layers.AveragePooling2D(
                pool_size = (7,7))
        self.conv3 = layers.Conv2D(filters=self.num_classes, kernel_size=1, strides=(1,1), use_bias=False)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.bn3(x)
        x = self.bn4(x)
        x = self.bn5(x)
        x = self.bn6(x)
        x = self.bn7(x)
        x = self.bn8(x)
        x = self.bn9(x)
        x = self.bn10(x)
        x = self.bn11(x)
        x = self.bn12(x)
        x = self.bn13(x)
        x = self.bn14(x)
        x = self.bn15(x)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        return x

@keras.utils.register_keras_serializable('vision', name="Loss")
class SSD(keras.Model):
    def __init__(self, classes = 10, ratios=[1.0, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0], ratios_last=None, *args, **kwargs):
        super(SSD, self).__init__(*args, **kwargs)
        if ratios_last is None:
            ratios_last = [1.0, 2.0, 0.5]
        self.classes = classes + 4
        self.ratios = ratios
        self.ratiosLast = ratios_last
        self.featureMaps = 6
        self.layerSize = [2**3, 2**4, 2**5, 2**6, 2**7, 2**8]

        self.numBoxes = [len(ratios) + 1 for _ in range(self.featureMaps)]
        self.numBoxes[0] = len(ratios_last)

    def build(self, input_shape):
        anchors = self.gen_anchors(input_shape[1:3])
        # self.mobilenet = keras.applications.MobileNetV2(input_shape[1:4], alpha=1.0, include_top=False, classes=self.classes, weights=None)
        self.mobilenet = MobileNetV2(k=self.classes)
        self.mobilenet.build(input_shape)
        # for layer in self.mobilenet.layers[-7:]:
        #     layer.trainable=False
        # for layer in self.mobilenet.layers[-8].layers[2:-1]:
        #     layer.trainable=False
        self.features = [None for _ in range(self.featureMaps)]

        self.conv1_1 = layers.Conv2D(256,1,name='SSD_conv_1_1')
        self.conv1_2 = layers.Conv2D(512,3,strides=(2,2),padding='same',name='SSD_conv_1_2')
        
        self.conv2_1 = layers.Conv2D(128,1,name='SSD_conv_2_1')
        self.conv2_2 = layers.Conv2D(256,3,strides=(2,2),padding='same',name='SSD_conv_2_2')
        
        self.conv3_1 = layers.Conv2D(128,1,name='SSD_conv_3_1')
        self.conv3_2 = layers.Conv2D(256,3,strides=(1,1),name='SSD_conv_3_2')
        
        self.conv4_1 = layers.Conv2D(128,1,name='SSD_conv_4_1')
        self.conv4_2 = layers.Conv2D(256,2,strides=(1,1),name='SSD_conv_4_2') # changed the kernel size to 2 since the output of the previous layer has width 3

        self.conv = []
        self.resh = []
        for i, anchor in enumerate(anchors):
            name1 = f"Classification_{i}"
            print(anchor.shape)
            self.conv.append(layers.Conv2D(self.numBoxes[i] * self.classes, 3, padding='same'))
            self.resh.append(layers.Reshape((anchor.shape[0], self.classes)))

    def call(self, x):
        x = self.mobilenet(x)
        print(x)
        self.features[0] = self.mobilenet.get_layer("BN4_1").out
        self.features[1] = self.mobilenet.get_layer("BN5_3").out
        self.features[2] = self.conv1_2(self.conv1_1(self.features[1]))
        self.features[3] = self.conv2_2(self.conv2_1(self.features[2]))
        self.features[4] = self.conv3_2(self.conv3_1(self.features[3]))
        self.features[5] = self.conv4_2(self.conv4_1(self.features[4]))

        results = [None for _ in range(self.featureMaps)]
        for i in range(self.featureMaps):
            x = self.conv[i](self.features[i])
            print(x.shape)
            x = self.resh[i](x)
            results[i] = x
        x = layers.concatenate(results, axis = -2)
        return x

    def gen_anchors(self, img_size, min_scale = 0.1, max_scale = 0.95):
        # a = Anchor(
        #         min_level=3,
        #         max_level=8,
        #         num_scales=1,
        #         aspect_ratios=self.ratios,
        #         anchor_size=4,
        #         image_size=img_size,
        #         )
        # boxes = []
        # for data in a.multilevel_boxes:
        #     print(data.shape)
        #     data = tf.reshape(data, (-1, self.classes))
        #     boxes.append(data)
        # return boxes
        gen = create_ssd_anchors(min_scale=min_scale, max_scale=max_scale,
                                 num_layers=self.featureMaps,
                                 scales=[],
                                 aspect_ratios=self.ratios,
                                 aspect_ratios_last=self.ratiosLast,
                                 interpolated_scale_aspect_ratio = 1.0,
                                 reduce_boxes_in_lowest_layer = LAST_REDUCE_BOXES,
                                 base_anchor_size = [1.0, 1.0],
                                 anchor_strides = [],
                                 anchor_offsets = [])
        layers = [(img_size[0] / v, img_size[1] / v) for v in self.layerSize]
        print(layers)
        layers = [(math.ceil(x), math.ceil(y)) for (x, y) in layers]
        return gen.generate(layers, im_width=img_size[0], im_height=img_size[1])
    def model(self, input_shape):
        if not self.built:
            self.build(input_shape)
        input_ = keras.Input(input_shape)
        return keras.Model(inputs=input_, outputs=self.call(input_))

