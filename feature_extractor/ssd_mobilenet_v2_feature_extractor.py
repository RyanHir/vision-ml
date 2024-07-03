import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from .ssd_mobilenet_v2_backbone import SSDMobileNetV2Backbone

def _build_feature_extractors(from_layers, layer_depths, depth_multiplier, min_depth, conv_kernel_size: int = 3):
    base_from_layer = ""
    networks = []

    def depth_fn(depth):
        depth = int(depth * depth_multiplier)
        return max(depth, min_depth)

    for index, (from_layer, layer_depth) in enumerate(zip(from_layers, layer_depths)):
        network = []
        if from_layer:
            base_from_layer = from_layer
            networks.append(None)
        else:
            layer_name = "{}_1_conv2d_{}_1x1_{}".format(
                base_from_layer,
                index,
                depth_fn(layer_depth // 2))
            network.append(layers.Conv2D(
                depth_fn(layer_depth // 2),
                [1,1],
                padding="same",
                strides=1,
                name=layer_name + "_conv"))
            network.append(layers.BatchNormalization(name=layer_name + "_batchnorm"))
            network.append(layers.Lambda(tf.nn.relu6, name=layer_name))


            layer_name = "{}_2_conv2d_{}_{}x{}_s2_{}".format(
                base_from_layer,
                index,
                conv_kernel_size, conv_kernel_size, depth_fn(layer_depth))
            network.append(layers.Conv2D(
                depth_fn(layer_depth),
                [conv_kernel_size, conv_kernel_size],
                padding="same",
                strides=2,
                name=layer_name + "_conv"))
            network.append(layers.BatchNormalization(name=layer_name + "_batchnorm"))
            network.append(layers.Lambda(tf.nn.relu6, name=layer_name))
            networks.append(network)
    return networks

class SSDMobileNetV2FeatureExtractor(keras.Model):
    def __init__(self, min_depth, depth_multiplier = 1.0, layers = 6):
        super().__init__()
        self.min_depth = min_depth
        self.depth_multiplier = depth_multiplier
        self.from_layers = ["layer_1", "layer_2", "", "", "", "", "", ""][:layers]
        self.layer_depths = [-1, -1, 512, 256, 256, 128, 128, 64][:layers]
    def build(self, input_size):
        self.backbone = SSDMobileNetV2Backbone(input_size)
        self.feature_networks = _build_feature_extractors(self.from_layers, self.layer_depths, self.depth_multiplier, self.min_depth)
    def call(self, inputs):
        image_features = self.backbone(inputs)
        image_features = {
            "layer_1": image_features[0],
            "layer_2": image_features[1],
            }
        feature_maps = []
        for index, from_layer in enumerate(self.from_layers):
            if from_layer:
                feature_map = image_features[from_layer]
            else:
                feature_map = feature_maps[-1]
                for layer in self.feature_networks[index]:
                    feature_map = layer(feature_map)
            feature_maps.append(feature_map)
        return feature_maps

