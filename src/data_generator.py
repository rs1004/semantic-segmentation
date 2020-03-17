from set_config import set_config
import tensorflow as tf
import os
import json

PARALLEL_NUM = os.cpu_count()


class DataGenerator:
    def __init__(self, data_type):
        set_config()
        self.data_type = data_type
        self.tfrecord_dir = os.environ['TFRECORD_DIR']
        self.batch_size = int(os.environ['BATCH_SIZE'])
        self.set_data_spec()

    def set_data_spec(self):
        with open(f'{self.tfrecord_dir}/{self.data_type}/spec.json', 'r') as f:
            d = json.load(f)

        self.H = d['height']
        self.W = d['width']
        self.data_length = d['data_length']

    def get_one_shot_iterator(self):
        files = tf.io.gfile.glob(f'{self.tfrecord_dir}/{self.data_type}/*.tfrecord')
        dataset_iterator = (
            tf.data.TFRecordDataset(files, num_parallel_reads=PARALLEL_NUM)
            .map(self.parse, num_parallel_calls=PARALLEL_NUM)
            .map(self.augumentation, num_parallel_calls=PARALLEL_NUM)
            .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=100))
            .batch(self.batch_size).prefetch(self.batch_size)
        )

        return dataset_iterator

    def parse(self, example_proto):
        W, H = self.W, self.H
        features = {
            'id': tf.io.FixedLenFeature([], tf.int64),
            'image_data': tf.io.FixedLenFeature([], tf.string),
            'label_data': tf.io.FixedLenFeature([], tf.string)
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image = tf.image.decode_png(parsed_features['image_data'], channels=3)
        image = tf.reshape(tf.cast(image, tf.float32), (H, W, 3))
        image /= 255.0
        label = tf.image.decode_png(parsed_features['label_data'], channels=1)
        label = tf.reshape(label, (H, W, 1))

        return image, label

    def augumentation(self, image, label):
        # random flip
        dice = tf.random.uniform([1], minval=-1, maxval=1)
        image = tf.cond(dice >= 0, lambda: image, lambda: tf.image.flip_left_right(image))
        label = tf.cond(dice >= 0, lambda: label, lambda: tf.image.flip_left_right(label))

        return image, label
