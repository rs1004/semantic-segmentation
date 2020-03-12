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
    img_list = sorted(glob(f'{DATA_DIR}/img/{data_type}/*'))
    anno_list = sorted(glob(f'{DATA_DIR}/anno/{data_type}/*'))
    global DATA_LENGTH
    DATA_LENGTH = len(img_list)
    return img_list, anno_list


def read_image(file_path):
    with tf.io.gfile.GFile(file_path, 'rb') as f:
        data = f.read()
    return data


def get_shard_list(l, n):
    return l[int((DATA_LENGTH / PROCESS_NUM) * n):int((DATA_LENGTH / PROCESS_NUM) * (n + 1))]


def get_tf_example(id, img_path, anno_path):
    feature = {
        'id': _int64_feature(id),
        'img_data': _bytes_feature(img_path),
        'anno_data': _bytes_feature(anno_path)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    for t in ['train', 'val']:
        img_list, anno_list = get_path_list(data_type=t)

        assert len(img_list) == len(anno_list), f'list length is mismatch: img => {len(img_list)}, anno => {len(anno_list)}'

        # create tfrecord
        os.makedirs(f'{TFRECORD_DIR}/{t}', exist_ok=True)
        for n in range(PROCESS_NUM):
            result_file_path = f'{TFRECORD_DIR}/{t}/{n+1:04}-of-{PROCESS_NUM:04}.tfrecord'
            shard_img_list = get_shard_list(l=img_list, n=n)
            shard_anno_list = get_shard_list(l=anno_list, n=n)

            with tf.io.TFRecordWriter(result_file_path) as writer:
                for i in tqdm(range(len(shard_img_list)), desc=f'write to {result_file_path}'):
                    example = get_tf_example(
                        id=i, img_path=read_image(shard_img_list[i]), anno_path=read_image(shard_anno_list[i]))
                    writer.write(example.SerializeToString())
