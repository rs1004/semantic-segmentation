import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.metrics import sparse_categorical_accuracy
from functools import partial
from config import Config

CONFIG = Config()


def get_metrics():
    metrics = []
    for i in range(CONFIG.CLASS_NUM):
        f = partial(sparse_class_average_iou, class_id=i)
        f.__name__ = f'sparse_class_average_iou_class{i}'
        metrics.append(f)
    metrics.append(sparse_mean_iou)
    metrics.append(sparse_categorical_accuracy)


def _convert(y_true, y_pred):
    B, H, W, _ = y_true.shape
    y_true = K.one_hot(tf.reshape(y_true, (B, H, W)), CONFIG.CLASS_NUM)
    y_pred = K.one_hot(K.argmax(y_pred), y_pred.shape[-1])
    return y_true, y_pred


def sparse_class_average_iou(y_true, y_pred, class_id):
    y_true, y_pred = _convert(y_true=y_true, y_pred=y_pred)

    intersection = y_pred[:, :, :, class_id] * y_true[:, :, :, class_id]
    union = K.maximum(y_pred[:, :, :, class_id], y_true[:, :, :, class_id])

    iou = K.sum(intersection, axis=(1, 2)) / K.sum(union + K.epsilon(), axis=(1, 2))

    return K.mean(iou)


def sparse_mean_iou(y_true, y_pred):
    y_true, y_pred = _convert(y_true=y_true, y_pred=y_pred)

    intersection = y_pred * y_true
    union = K.maximum(y_pred, y_true)

    ious = K.sum(intersection, axis=(1, 2)) / K.sum(union + K.epsilon(), axis=(1, 2))
    average_ious = K.mean(ious, axis=0)

    return K.mean(average_ious)
