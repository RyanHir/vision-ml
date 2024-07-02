from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import math

from anchors import Anchor, create_ssd_anchors
from feature_extractor.ssd_mobilenet_v2_feature_extractor import SSDMobileNetV2FeatureExtractor

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
    def __init__(self, classes = 10, *args, **kwargs):
        super(SSD, self).__init__(*args, **kwargs)
        self.classes = classes + 4
        self.ratios = [1.0, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0]
        self.ratiosLast = [1.0, 2.0, 0.5]
        self.featureMaps = 6

        layerStart = 3
        layerStop = layerStart + self.featureMaps
        self.layerSize = [2**n for n in range(layerStart, layerStop)]

        self.numBoxes = [len(self.ratios) + 1 for _ in range(self.featureMaps)]
        self.numBoxes[0] = len(self.ratiosLast)

    def build(self, input_shape):
        self.feature_extractor = SSDMobileNetV2FeatureExtractor(min_depth=32, depth_multiplier=1.0)

        self.conv = []
        self.resh = []
        for numBox in self.numBoxes:
            self.conv.append(layers.Conv2D(numBox * self.classes, 3, padding='same'))
            self.resh.append(layers.Reshape((-1, self.classes)))

    def call(self, x):
        features = self.feature_extractor(x)
        results = [None for _ in range(len(features))]
        for i, feature in enumerate(features):
            x = self.conv[i](feature)
            x = self.resh[i](x)
            results[i] = x
        x = layers.concatenate(results, axis = -2)
        return {"bbox": x[..., 0:4], "cls": x[..., 4:]}

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
                                 reduce_boxes_in_lowest_layer = True,
                                 base_anchor_size = [1.0, 1.0],
                                 anchor_strides = [],
                                 anchor_offsets = [])
        layers = [(img_size[0] / v, img_size[1] / v) for v in self.layerSize]
        layers = [(math.ceil(x), math.ceil(y)) for (x, y) in layers]
        anchors = gen.generate(layers, im_width=img_size[0], im_height=img_size[1])
        return anchors
