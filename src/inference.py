from data_generator import DataGenerator
from model import UNet
from config import Config
from PIL import Image, ImageDraw
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import json
import argparse


CONFIG = Config()


def get_label_map():
    with open(CONFIG.ROOT_DIR / 'labelmap.json', 'r') as f:
        label_map = json.load(f)

    label_map = {d['id']: d['colors'][0] for d in label_map.values()}
    return label_map


def get_concat_h(image, pred, label, shape, title_height=11):
    height, width = shape
    dst = Image.new('RGB', (3 * width, height + title_height), 'white')
    dst.paste(image, (0, title_height))
    dst.paste(pred, (width, title_height))
    dst.paste(label, (2 * width, title_height))

    draw = ImageDraw.Draw(dst)
    draw.text((0, 0), 'original image', fill=(0, 0, 0, 0))
    draw.text((width, 0), 'prediction', fill=(0, 0, 0, 0))
    draw.text((2 * width, 0), 'answer', fill=(0, 0, 0, 0))

    return dst


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('num_of_outputs', type=int)
    args = parser.parse_args()

    # output => (original image, prediction, answer)
    data_gen_val = DataGenerator('val', is_train=False)
    ds_val = data_gen_val.get_one_shot_iterator()
    model = UNet(input_shape=(data_gen_val.H, data_gen_val.W, 3), class_num=CONFIG.CLASS_NUM)
    label_map = get_label_map()

    H, W = data_gen_val.H, data_gen_val.W
    counter = 0
    (CONFIG.RESULT_DIR / 'inference').mkdir(parents=True, exist_ok=True)
    for image, label in tqdm(ds_val.take(args.num_of_outputs // CONFIG.BATCH_SIZE), total=args.num_of_outputs // CONFIG.BATCH_SIZE):
        pred = tf.math.argmax(model(image), axis=-1, output_type=tf.int64)
        for i in range(CONFIG.BATCH_SIZE):
            img = (image[i].numpy() * 255.0).astype(np.uint8)
            prd = np.asarray([label_map[id] for id in pred[i].numpy().reshape(-1)], dtype=np.uint8).reshape(H, W, 3)
            lbl = np.asarray([label_map[id] for id in label[i].numpy().reshape(-1)], dtype=np.uint8).reshape(H, W, 3)
            get_concat_h(
                image=Image.fromarray(img),
                pred=Image.fromarray(prd),
                label=Image.fromarray(lbl),
                shape=(H, W)
            ).save(CONFIG.RESULT_DIR / 'inference' / f'{counter:06}.png')
            counter += 1
