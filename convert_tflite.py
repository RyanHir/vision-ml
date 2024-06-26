import tensorflow as tf
import tensorflow.lite as tflite
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("weights", type=str)
args.add_argument("out", type=str)
args = args.parse_args()

model = tf.keras.models.load_model(args.weights, compile=False)
model(tf.keras.Input((320, 320, 3)))

converter = tflite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
model = converter.convert()

with open(args.out, "wb") as f:
    f.write(model)

print(args)

