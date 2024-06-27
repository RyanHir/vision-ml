from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import math

from anchors import Anchor, create_ssd_anchors
from feature_extractor.ssd_mobilenet_v2_backbone import SSDMobileNetV2Backbone

LAST_REDUCE_BOXES = True
LAST_NUM_BOXES = 3 if LAST_REDUCE_BOXES else 6


def _make_divisible(v: float, divisor: int, min_value = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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
        self.backbone = SSDMobileNetV2Backbone(input_shape)
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
            self.conv.append(layers.Conv2D(self.numBoxes[i] * self.classes, 3, padding='same'))
            self.resh.append(layers.Reshape((anchor.shape[0], self.classes)))

    def call(self, x):
        out1, out2 = self.backbone(x)
        self.features[0] = out1
        self.features[1] = out2
        self.features[2] = self.conv1_2(self.conv1_1(self.features[1]))
        self.features[3] = self.conv2_2(self.conv2_1(self.features[2]))
        self.features[4] = self.conv3_2(self.conv3_1(self.features[3]))
        self.features[5] = self.conv4_2(self.conv4_1(self.features[4]))

        results = [None for _ in range(self.featureMaps)]
        for i in range(self.featureMaps):
            x = self.conv[i](self.features[i])
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
        layers = [(math.ceil(x), math.ceil(y)) for (x, y) in layers]
        return gen.generate(layers, im_width=img_size[0], im_height=img_size[1])
    def model(self, input_shape):
        if not self.built:
            self.build(input_shape)
        input_ = keras.Input(input_shape)
        return keras.Model(inputs=input_, outputs=self.call(input_))

