from glob import glob
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import json
import numpy as np

ROOT_DIR = '/work'
DATA_DIR = '/work/data'
PROCESS_NUM = cpu_count()
H = 512
W = 1024


def set_label_map():
    with open(f'{ROOT_DIR}/labelmap.json', 'r') as f:
        label_map = json.load(f)
    global LABEL_MAP
    LABEL_MAP = {f'{r}_{g}_{b}': i for i, k in enumerate(label_map.keys()) for r, g, b in label_map[k]}


def preprocess_image(src_dst_path):
    src_path, dst_path = src_dst_path
    img = Image.open(src_path).convert('RGB').resize((W, H))
    img.save(dst_path)


def preprocess_label(src_dst_path):
    src_path, dst_path = src_dst_path
    img = np.asarray(Image.open(src_path).convert('RGB').resize((W, H), resample=Image.NEAREST))
    img = Image.fromarray(
        np.array(
            object=[LABEL_MAP[f'{r}_{g}_{b}'] for (r, g, b) in img.reshape(-1, 3).tolist()],
            dtype=np.uint8
        ).reshape(H, W)
    )
    img.save(dst_path)


if __name__ == '__main__':
    # initialize
    set_label_map()

    for t in ['train', 'val']:
        # preprocess: image => downsize
        src_path_list = sorted(glob(f'{DATA_DIR}/leftImg8bit/{t}/*/*'))
        dst_path_list = [f'{DATA_DIR}/image/{t}/X{i:05}.png' for i in range(len(src_path_list))]

        os.makedirs(f'{DATA_DIR}/image/{t}', exist_ok=True)
        with Pool(processes=PROCESS_NUM) as p:
            m = p.imap(preprocess_image, zip(src_path_list, dst_path_list))
            list(tqdm(m, desc=f'preprocess {t} image files', total=len(src_path_list)))

        # preprocess: label => rgb2lbl, downsize
        src_path_list = sorted(glob(f'{DATA_DIR}/gtFine/{t}/*/*_color.png'))
        dst_path_list = [f'{DATA_DIR}/label/{t}/Y{i:05}.png' for i in range(len(src_path_list))]

        os.makedirs(f'{DATA_DIR}/label/{t}', exist_ok=True)
        with Pool(processes=PROCESS_NUM) as p:
            m = p.imap(preprocess_label, zip(src_path_list, dst_path_list))
            list(tqdm(m, desc=f'preprocess {t} label files', total=len(src_path_list)))
