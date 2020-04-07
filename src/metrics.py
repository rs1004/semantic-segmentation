import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.metrics import sparse_categorical_accuracy
from functools import partial
from config import Config

CONFIG = Config()


def get_metrics():
    metrics = []
    # for i in range(CONFIG.CLASS_NUM):
    #     f = partial(sparse_class_average_iou, class_id=i)
    #     f.__name__ = f'average_iou_class_{i}'
    #     metrics.append(f)
    metrics.append(sparse_mean_iou)
    metrics.append(sparse_f1_score)
    metrics.append(sparse_categorical_accuracy)

    return metrics


def _convert(y_true, y_pred):
    y_true = K.one_hot(K.cast(K.max(y_true, axis=-1), 'int32'), CONFIG.CLASS_NUM)
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


def sparse_mean_average_precision(y_true, y_pred):
    y_true, y_pred = _convert(y_true=y_true, y_pred=y_pred)

    tp = K.sum(y_true * y_pred, axis=(1, 2))
    fp = K.sum(K.cast(K.equal(y_pred - y_true, 1.), K.floatx()), axis=(1, 2))

    precisions = tp / (tp + fp + K.epsilon())
    average_precisions = K.mean(precisions, axis=0)
    return K.mean(average_precisions)


def sparse_mean_average_recall(y_true, y_pred):
    y_true, y_pred = _convert(y_true=y_true, y_pred=y_pred)

    tp = K.sum(y_true * y_pred, axis=(1, 2))
    fn = K.sum(K.cast(K.equal(y_pred - y_true, -1.), K.floatx()), axis=(1, 2))

    recalls = tp / (tp + fn + K.epsilon())
    average_recalls = K.mean(recalls, axis=0)
    return K.mean(average_recalls)


def sparse_f1_score(y_true, y_pred):
    y_true, y_pred = _convert(y_true=y_true, y_pred=y_pred)
    tp = K.sum(y_true * y_pred, axis=(1, 2))
    fp = K.sum(K.cast(K.equal(y_pred - y_true, 1.), K.floatx()), axis=(1, 2))
    fn = K.sum(K.cast(K.equal(y_pred - y_true, -1.), K.floatx()), axis=(1, 2))

    precision = K.mean(K.mean(tp / (tp + fp + K.epsilon()), axis=0))
    recall = K.mean(K.mean(tp / (tp + fn + K.epsilon()), axis=0))

    return 2 * precision * recall / (precision + recall)
