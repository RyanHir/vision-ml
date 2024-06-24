from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

#import keras_cv

# tf.compat.v1.disable_eager_execution()

from anchors import GridAnchorGenerator, create_ssd_anchors
from models import SSD

CLASSES = 80
BATCH_SIZE = 16
EPOCHS = 100
ITERS = 200000
BASE_LEARNING_RATE=0.004
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
    img = tf.image.decode_jpeg(features["image/encoded"], channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
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

def prepare_samples(anchors):
    def func(features):
        bbox = features["image/object/bbox"]
        cls  = features["image/object/class/label"]

        num = bbox.shape[0]
        y = tf.zeros((anchors.shape[0], 5), dtype=tf.float32)
        if num is not None:
            for i in range(num):
                box_id = tf.cast(best_iou(bbox[i]), tf.uint16)
                y[box_id, :4] = bbox[i]
                y[box_id, 4]  = cls[i]
        else:
            print("b")
        return features["image/encoded"], y 
    return func

def read_tfrecord(path, batch_size, epochs, anchors):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dataset = tf.data.TFRecordDataset([path])
    print(dataset.cardinality().numpy())
    dataset = dataset.map(decode_tfrecord_feature)
    dataset = dataset.map(prepare_samples(anchors))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset

def smooth_l1(true, pred):
    abs_loss = tf.abs(true - pred)
    sqr_loss = 0.5 * (true - pred) ** 2
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sqr_loss, abs_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)

def iou(box1, box2):
  box1 = tf.cast(box1, tf.float64)
  box2 = tf.cast(box2, tf.float64)
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
    return 1.0 - tf.math.reduce_mean(iou(real, pred))

def best_iou(anchors, searchBox):
    return np.argwhere(iou(np.matlib.repmat(searchBox,anchors.shape[0],1), anchors) > 0.5)

def smoothL1(x,y,label):
    diff = keras.backend.abs(x-y) #* K.switch(label == 10, label*1.0/BOXES, label)
    result = keras.backend.switch(diff < 1, 0.5 * diff**2, diff - 0.5)
    return keras.backend.mean(result)

def confidenceLoss(y,label):
    unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
    # class_weights = tf.constant([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/BOXES]]*BOXES])
    # weights = tf.reduce_sum(class_weights * y, axis = -1)
    # weighted_loss = unweighted_loss * weights
    return keras.backend.mean(unweighted_loss)
     

def Loss(gt,y):
    print(y)
    # shape of y is n * BOXES * output_channels
    # shape of gt is n * BOXES * 5 
    loss = 0
    # localisation loss
    loss += smoothL1(y[:,:,0:4], gt[:,:,0:4], gt[:,:,0:1])
    # confidence loss
    loss += confidenceLoss(y[:,:,4:], tf.cast(gt[:,:,0],tf.int32))
    return loss
    # return tf.math.log(loss)


def main():
    model = SSD(classes=CLASSES)
    anchors = model.gen_anchors(img_size=IMG_SIZE)
    anchors = tf.concat(anchors, axis=0)
    base_learning_rate = 0.001

    train_dataset = read_tfrecord("./train.tfrecord", BATCH_SIZE, EPOCHS, anchors)
    test_dataset  = read_tfrecord("./validate.tfrecord", BATCH_SIZE, EPOCHS, anchors)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=BASE_LEARNING_RATE),
                  loss = Loss,
                  # loss = {'bbox': iou_loss, 'conf': 'binary_crossentropy', 'cls' : 'categorical_crossentropy'},
                  # metrics = [iou]
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

