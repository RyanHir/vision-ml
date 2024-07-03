import tensorflow as tf
import tensorflow.lite as tflite
from argparse import ArgumentParser

from models import SSD

args = ArgumentParser()
args.add_argument("weights", type=str)
args.add_argument("out", type=str)
args.add_argument("-c", "--classes", type=int, default=80)
args.add_argument("--imgsz", nargs="+", type=int, default=[320, 320])
args = args.parse_args()

if len(args.imgsz) == 1:
    img_w = img_h = args.imgsz[0]
else:
    img_w, img_h = args.imgsz

model = SSD(classes=args.classes, disable_tensor_split=True)
model.load_weights(args.weights)

inputs = tf.keras.Input((img_h, img_w, 3), batch_size=1)
outputs = model(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

converter = tflite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float32]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

model = converter.convert()

with open(args.out, "wb") as f:
    f.write(model)

