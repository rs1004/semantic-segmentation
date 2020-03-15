from tqdm import tqdm
from glob import glob
import tensorflow as tf
import numpy as np
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
PROCESS_NUM = 2
DATA_DIR = '/work/data'
TFRECORD_DIR = f'{DATA_DIR}/tfrecord' 


def get_path_list(data_type):
    image_path_list = sorted(glob(f'{DATA_DIR}/image/{data_type}/*'))
    label_path_list = sorted(glob(f'{DATA_DIR}/label/{data_type}/*'))
    global DATA_LENGTH
    DATA_LENGTH = len(image_path_list)
    return image_path_list, label_path_list


def read_image(file_path):
    with tf.io.gfile.GFile(file_path, 'rb') as f:
        data = f.read()
    return data


def get_shard_list(l, n):
    return l[int((DATA_LENGTH / PROCESS_NUM) * n):int((DATA_LENGTH / PROCESS_NUM) * (n + 1))]


def get_tf_example(id, image_data, label_data):
    feature = {
        'id': _int64_feature(id),
        'image_data': _bytes_feature(image_data),
        'label_data': _bytes_feature(label_data)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    for t in ['train', 'val']:
        image_path_list, label_path_list = get_path_list(data_type=t)

        assert len(image_path_list) == len(label_path_list), f'list length is mismatch: image => {len(image_path_list)}, label => {len(label_path_list)}'

        # create tfrecord
        os.makedirs(f'{TFRECORD_DIR}/{t}', exist_ok=True)
        for n in range(PROCESS_NUM):
            result_file_path = f'{TFRECORD_DIR}/{t}/{n+1:04}-of-{PROCESS_NUM:04}.tfrecord'
            shard_image_path_list = get_shard_list(l=image_path_list, n=n)
            shard_label_path_list = get_shard_list(l=label_path_list, n=n)

            with tf.io.TFRecordWriter(result_file_path) as writer:
                for i in tqdm(range(len(shard_image_path_list)), desc=f'write to {result_file_path}'):
                    example = get_tf_example(
                        id=i, image_data=read_image(shard_image_path_list[i]), label_data=read_image(shard_label_path_list[i]))
                    writer.write(example.SerializeToString())
