from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

import keras_cv

# tf.compat.v1.disable_eager_execution()

from models import SSD

CLASSES = 80
BATCH_SIZE = 1
EPOCHS = 32
IMG_W = 224 
IMG_H = 224 

def iou(bbox1, bbox2):
    box1 = box1.astype(np.float64)
    box2 = box2.astype(np.float64)
    # find the left and right co-ordinates of the edges. Min should be less than Max for non zero overlap
    xmin = np.maximum(box1[:,0],box2[:,0])
    xmax = np.minimum(box1[:,2],box2[:,2])
    ymin = np.maximum(box1[:,1],box2[:,1])
    ymax = np.minimum(box1[:,3],box2[:,3])

    intersection = np.abs(np.maximum(xmax-xmin,0) * np.maximum(ymax-ymin,0))
    boxArea1 = np.abs((box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]))
    boxArea2 = np.abs((box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]))
    unionArea = boxArea1 + boxArea2 - intersection
    assert (unionArea > 0).all()
    iou = intersection / unionArea

    return iou


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
    img = tf.image.decode_jpeg(features["image/encoded"], channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_W, IMG_H))
    features["image/encoded"] = img

    y = [
        tf.sparse.to_dense(features['image/object/bbox/xmin'], default_value=0),
        tf.sparse.to_dense(features['image/object/bbox/ymax'], default_value=0),
        tf.sparse.to_dense(features['image/object/bbox/xmin'], default_value=0),
        tf.sparse.to_dense(features['image/object/bbox/ymax'], default_value=0),
    ]
    y = tf.reshape(y, (-1, 4))
    features["image/object/bbox"] = y
    features["image/object/class/label"] = tf.sparse.to_dense(features["image/object/class/label"])
    return features

def prepare_samples(features):
    bbox = features["image/object/bbox"]
    cls  = features["image/object/class/label"]
    cls = tf.keras.utils.to_categorical(cls, CLASSES)
    return features["image/encoded"], {"bbox": bbox, "cls": cls}

def read_tfrecord(path, batch_size, epochs):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dataset = tf.data.TFRecordDataset([path])
    dataset = dataset.map(decode_tfrecord_feature)
    dataset = dataset.map(prepare_samples)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    return dataset

def smooth_l1(true, pred):
    abs_loss = tf.abs(true - pred)
    sqr_loss = 0.5 * (true - pred) ** 2
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sqr_loss, abs_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)

def class_loss(logits, label):
    """Calculate category losses."""
    label = tf.one_hot(label, ops.shape(logits)[-1], Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
    weight = tf.ones_like(logits)
    pos_weight = tf.ones_like(logits)
    sigmiod_cross_entropy = tf.binary_cross_entropy_with_logits(logits, label, weight.astype(ms.float32), pos_weight.astype(ms.float32))
    sigmoid = tf.sigmoid(logits)
    label = label.astype(tf.float32)
    p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
    modulating_factor = tf.pow(1 - p_t, 2.0)
    alpha_weight_factor = label * 0.75 + (1 - label) * (1 - 0.75)
    focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
    return focal_loss

def main():
    model = SSD(classes=CLASSES+5)
    base_learning_rate = 0.001

    train_dataset = read_tfrecord("./train.tfrecord", BATCH_SIZE, EPOCHS)
    test_dataset  = read_tfrecord("./validate.tfrecord", BATCH_SIZE, EPOCHS)

    # for batch in train_dataset:
    #    print(batch)

    anchors = model.gen_anchors(img_size=IMG_W)
    print(anchors)
    exit()
    y = model(keras.Input((IMG_H, IMG_W, 3)))
    model.summary()
    print(y)

    model.compile(optimizer="adam",
                  # loss = loss_fn,
                  loss = {'bbox': keras_cv.losses.GIoULoss("xyxy"), 'conf': 'binary_crossentropy', 'cls' : 'categorical_crossentropy'},
                  metrics = {'cls' : 'accuracy', 'bbox' : 'mse' }
                  )
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data = test_dataset,
                        )

if __name__ == "__main__":
    main()

