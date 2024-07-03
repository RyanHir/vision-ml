import tensorflow as tf
import tensorflow.lite as tflite
from argparse import ArgumentParser

from models import SSD

args = ArgumentParser()
args.add_argument("weights", type=str)
args.add_argument("out", type=str)
args.add_argument("-c", "--classes", type=int, default=80)
args = args.parse_args()

model = SSD(classes=args.classes, disable_tensor_split=True)
model.load_weights(args.weights)

inputs = tf.keras.Input((320, 320, 3), batch_size=1)
outputs = model(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

converter = tflite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
model = converter.convert()

with open(args.out, "wb") as f:
    f.write(model)

print(args)

