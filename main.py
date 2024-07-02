from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

import albumentations as A

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
        tf.sparse.to_dense(features['image/object/bbox/xmin']),
        tf.sparse.to_dense(features['image/object/bbox/ymin']),
        tf.sparse.to_dense(features['image/object/bbox/xmax']),
        tf.sparse.to_dense(features['image/object/bbox/ymax']),
    ]
    y = tf.stack(y, axis=-1)
    features["image/object/bbox"] = y
    features["image/object/class/label"] = tf.sparse.to_dense(features["image/object/class/label"])
    return features

AUGMENT = A.Compose([
    A.Rotate(limit=70, min_area=1),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HorizontalFlip(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=["labels"]))

def augment(features):
    @tf.numpy_function(Tout=[tf.float32, tf.float32, tf.int64])
    def fn(image, bboxes, labels):
        global AUGMENT
        mask = np.bitwise_and(bboxes[:, 0] != bboxes[:, 2], bboxes[:, 1] != bboxes[:, 3])
        bboxes = bboxes[mask]
        labels = labels[mask]
        res = AUGMENT(image=image, bboxes=bboxes, labels=labels)
        return np.array(res["image"], dtype=np.float32), np.array(res["bboxes"], dtype=np.float32), np.array(res["labels"], dtype=np.int64)
    results = fn(features["image/encoded"], features["image/object/bbox"], features["image/object/class/label"])
    features["image/encoded"] = results[0]
    features["image/object/bbox"] = results[1]
    features["image/object/class/label"] = results[2]
    return features


def prepare_samples(anchors):
    def func(features):
        image = features["image/encoded"]
        bbox  = features["image/object/bbox"]
        cls   = features["image/object/class/label"]

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
    dataset = dataset.repeat()
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(prepare_samples(anchors), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
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

