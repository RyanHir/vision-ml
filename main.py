from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

# tf.compat.v1.disable_eager_execution()

from anchors import GridAnchorGenerator, create_ssd_anchors
from models import SSD

CLASSES = 80
BATCH_SIZE = 24
EPOCHS = 300
ITERS = 200000
BASE_LEARNING_RATE = 1e-4
IMG_SIZE = (320, 320) 

COCO_FEATURE_MAP = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
}

def decode_tfrecord_feature(feature):
    features = tf.io.parse_single_example(feature, features=COCO_FEATURE_MAP)
    img = tf.image.decode_image(features["image/encoded"], channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    # assert tf.reduce_min(img) >= 0.0 and tf.reduce_max(img) <= 1.0

    # img /= img / 255.0
    sample_size = tf.shape(img)[0:2]
    img = tf.image.resize(img, IMG_SIZE)
    features["image/encoded"] = img

    y = [
        tf.sparse.to_dense(features['image/object/bbox/xmin'], default_value=0),
        tf.sparse.to_dense(features['image/object/bbox/ymax'], default_value=0),
        tf.sparse.to_dense(features['image/object/bbox/xmin'], default_value=0),
        tf.sparse.to_dense(features['image/object/bbox/ymax'], default_value=0),
    ]
    y = tf.reshape(y, (-1, 4))
#     y /= tf.repeat(tf.cast(sample_size[::-1], tf.float32), 2)
    # y *= (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1])
    features["image/object/bbox"] = y
    features["image/object/class/label"] = tf.sparse.to_dense(features["image/object/class/label"])
    return features

def prepare_samples(anchors):
    def func(features):
        bbox = features["image/object/bbox"]
        cls  = features["image/object/class/label"]

        num = bbox.shape[0]
        y1 = tf.zeros((anchors.shape[0], 4), dtype=tf.float32)
        y2 = tf.zeros((anchors.shape[0], 1), dtype=tf.float32)
        if num is not None:
            for i in range(num):
                box_id = tf.cast(best_iou(bbox[i]), tf.uint16)
                y1[box_id, :] = bbox[i]
                y2[box_id, 0] = cls[i]
        return features["image/encoded"], {"bbox": y1, "cls": y2} 
    return func

def read_tfrecord(path, batch_size, epochs, anchors):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dataset = tf.data.TFRecordDataset([path])
    dataset = dataset.map(decode_tfrecord_feature)
    dataset = dataset.map(prepare_samples(anchors))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset

def iou(box1, box2):
    box1 = tf.cast(box1[..., 0:4], tf.float64)
    box2 = tf.cast(box2[..., 0:4], tf.float64)
    # find the left and right co-ordinates of the edges. Min should be less than Max for non zero overlap
    xmin = tf.maximum(box1[:,0],box2[:,0])
    ymin = tf.maximum(box1[:,1],box2[:,1])
    xmax = tf.minimum(box1[:,2],box2[:,2])
    ymax = tf.minimum(box1[:,3],box2[:,3])

    intersection = tf.abs(tf.maximum(xmax - xmin,0) * tf.maximum(ymax - ymin,0))
    boxArea1 = tf.abs((box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]))
    boxArea2 = tf.abs((box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]))
    unionArea = boxArea1 + boxArea2 - intersection
    # assert tf.math.reduce_all(unionArea > 0).numpy()
    iou = (intersection + 1e-7) / (unionArea + 1e-7)

    return iou

def iou_loss(real, pred):
    return 1.0 - keras.ops.mean(iou(real, pred))

def best_iou(anchors, searchBox):
    return np.argwhere(iou(np.matlib.repmat(searchBox,anchors.shape[0],1), anchors) > 0.5)

@keras.utils.register_keras_serializable('vision', name="SmoothL1")
def smooth_l1(x, y):
    absdiff = tf.abs(tf.cast(x[..., :4], tf.float64) - tf.cast(y[..., :4], tf.float64))
    sqrdiff = absdiff**2
    loss = tf.where(
            absdiff < 1.0,
            0.5 * sqrdiff,
            absdiff - 0.5)
    return tf.math.reduce_mean(loss, axis=-1)

def main():

    model = SSD(classes=CLASSES)
    anchors = model.gen_anchors(img_size=IMG_SIZE)
    anchors = tf.concat(anchors, axis=0)
    anchors /= (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1])
    base_learning_rate = 0.001

    train_dataset = read_tfrecord("./train.tfrecord", BATCH_SIZE, EPOCHS, anchors)
    test_dataset  = read_tfrecord("./validate.tfrecord", BATCH_SIZE, EPOCHS, anchors)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=BASE_LEARNING_RATE),
                  loss = {
                      "bbox": smooth_l1,
                      "cls": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      },
                  metrics = {
                      "bbox": "mse",
                      }
                  )
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        steps_per_epoch=ITERS // EPOCHS,
                        validation_data = test_dataset,
                        validation_steps=ITERS // EPOCHS,
                        callbacks=[keras.callbacks.TensorBoard(log_dir="train/logs"),
                                   keras.callbacks.ModelCheckpoint("train/ckpt/model_{epoch}.keras"),
                                   ]
                        )

if __name__ == "__main__":
    main()

