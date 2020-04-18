from tqdm import tqdm
from config import Config
import tensorflow as tf
import json

CONFIG = Config()


def get_path_list(data_type):
    image_path_list = sorted((CONFIG.IMAGE_DIR / data_type).glob('*'))
    label_path_list = sorted((CONFIG.LABEL_DIR / data_type).glob('*'))
    global DATA_LENGTH
    DATA_LENGTH = len(image_path_list)
    return image_path_list, label_path_list


def read_image(file_path):
    with tf.io.gfile.GFile(file_path.as_posix(), 'rb') as f:
        data = f.read()
    return data


def get_shard_list(l, n):
    return l[int((DATA_LENGTH / CONFIG.TFRECORD_FILE_NUM) * n):int((DATA_LENGTH / CONFIG.TFRECORD_FILE_NUM) * (n + 1))]


def get_spec_example(height, width, data_length):
    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'data_length': _int64_feature(data_length)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def get_data_example(id, image_data, label_data):
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
    for t in ['train', 'test', 'val']:
        image_path_list, label_path_list = get_path_list(data_type=t)

        assert len(image_path_list) == len(label_path_list), f'list length is mismatch: image => {len(image_path_list)}, label => {len(label_path_list)}'

        # create tfrecord
        (CONFIG.TFRECORD_DIR / t).mkdir(parents=True, exist_ok=True)

        # spec
        spec_file_path = CONFIG.TFRECORD_DIR / t / 'spec.json'
        print(f'write to {spec_file_path.resolve()}')
        with open(spec_file_path, 'w') as f:
            spec = {
                "height":  CONFIG.H,
                "width": CONFIG.W,
                "data_length": DATA_LENGTH
            }
            json.dump(spec, f, indent=2)

        # data
        for n in range(CONFIG.TFRECORD_FILE_NUM):
            result_file_path = CONFIG.TFRECORD_DIR / t / f'{n+1:04}-of-{CONFIG.TFRECORD_FILE_NUM:04}.tfrecord'
            shard_image_path_list = get_shard_list(l=image_path_list, n=n)
            shard_label_path_list = get_shard_list(l=label_path_list, n=n)

            with tf.io.TFRecordWriter(result_file_path.as_posix()) as writer:
                for i in tqdm(range(len(shard_image_path_list)), desc=f'write to {result_file_path}'):
                    example = get_data_example(
                        id=i, image_data=read_image(shard_image_path_list[i]), label_data=read_image(shard_label_path_list[i]))
                    writer.write(example.SerializeToString())
