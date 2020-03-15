import tensorflow as tf

TFRECORD_DIR = '/work/data/tfrecord'
H = 512
W = 1024
PARALLEL_NUM = 8
BATCH_SIZE = 32


class DataGenerator:
    def __init__(self, data_type):
        self.data_type = data_type

    def get_one_shot_iterator(self):
        files = tf.io.gfile.glob(f'{TFRECORD_DIR}/{self.data_type}/*')
        dataset_iterator = (
            tf.data.TFRecordDataset(files, num_parallel_reads=PARALLEL_NUM)
            .map(self.parse, num_parallel_calls=PARALLEL_NUM)
            .map(self.augumentation, num_parallel_calls=PARALLEL_NUM)
            .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=100))
            .batch(BATCH_SIZE).prefetch(BATCH_SIZE)
        )

        return dataset_iterator

    def parse(self, example_proto):
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

        sample = {
            'id': parsed_features['id'],
            'image': image,
            'label': label
        }

        return sample

    def augumentation(self, sample):
        image = sample['image']
        label = sample['label']

        # random flip
        dice = tf.random.uniform([1], minval=-1, maxval=1)
        image = tf.cond(dice >= 0, lambda: image, lambda: tf.image.flip_left_right(image))
        label = tf.cond(dice >= 0, lambda: label, lambda: tf.image.flip_left_right(label))

        sample['image'] = image
        sample['label'] = label

        return sample
