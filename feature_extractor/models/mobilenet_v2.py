import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class ReLU6(keras.layers.Layer):
    def call(self, data):
        return tf.nn.relu6(data)

class MobileNetV2(keras.Model):
    def __init__(self, num_classes=1000, *args, **kwargs):
        super(MobileNetV2, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
                filters = 32,
                kernel_size = 3,
                strides = (2, 2),
                padding='valid',
                use_bias = False)
        self.bn = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name="conv_bn")
        self.relu = ReLU6(name="conv_relu")

        self.bn1 = Bottleneck(1, 16, 1, name="BN1")

        self.bn2 = Bottleneck(6, 24, 2, name="BN2_1")
        self.bn3 = Bottleneck(6, 24, 1, name="BN2_2")

        self.bn4 = Bottleneck(6, 32, 2, name="BN3_1")
        self.bn5 = Bottleneck(6, 32, 1, name="BN3_2")
        self.bn6 = Bottleneck(6, 32, 1, name="BN3_3")

        self.bn7 = Bottleneck(6, 64, 2, name="BN4_1")
        self.bn8 = Bottleneck(6, 64, 1, name="BN4_2")
        self.bn9 = Bottleneck(6, 64, 1, name="BN4_3")
        self.bn10 = Bottleneck(6, 64, 1, name="BN4_4")

        self.bn11 = Bottleneck(6, 96, 1, name="BN5_1")
        self.bn12 = Bottleneck(6, 96, 1, name="BN5_2")
        self.bn13 = Bottleneck(6, 96, 1, name="BN5_3")

        self.bn14 = Bottleneck(6, 160, 2, name="BN6_1")
        self.bn15 = Bottleneck(6, 160, 1, name="BN6_2")
        self.bn16 = Bottleneck(6, 160, 1, name="BN6_3")

        self.bn17 = Bottleneck(6, 320, 1, name="BN7")
        self.conv2 = layers.Conv2D(filters = 1280, kernel_size=1, strides=(1,1), use_bias=False)
        self.avgpool = layers.AveragePooling2D(pool_size = (7,7))
        self.conv3 = layers.Conv2D(filters=self.num_classes, kernel_size=1, strides=(1,1), use_bias=False)

    def call(self, inputs):
        x = self.conv1(inputs)
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
        x = self.bn16(x)

        x = self.bn17(x)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        return x

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
        self.expand_relu = ReLU6(name="expand_ReLU")
        
        # Conv
        self.conv = layers.DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same', use_bias=False, name="conv")
        self.conv_bn = layers.BatchNormalization(name="conv_BN")
        self.conv_relu = ReLU6(name="conv_ReLU")

        # Projection
        self.project = layers.Conv2D(filters=self.filters, kernel_size=1, use_bias=False, name="contract")
        self.project_bn = layers.BatchNormalization(name="contract_BN")

        self.residual = layers.Add(name="residual")

    def call(self, inputs):
        x = self.expand(inputs)
        x = self.expand_bn(x)
        x = self.expand_relu(x)

        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.conv_relu(x)

        x = self.project(x)
        x = self.project_bn(x)
        
        if self.output_channels == self.d and self.strides == 1:
            x = self.residual([inputs, x])
        return x

