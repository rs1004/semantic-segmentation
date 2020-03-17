from glob import glob
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from set_config import set_config
import os
import json
import numpy as np

PARALLEL_NUM = os.cpu_count()


def set_label_map():
    with open(f'{os.environ["LABELMAP_PATH"]}', 'r') as f:
        label_map = json.load(f)
    global LABEL_MAP
    LABEL_MAP = {f'{r}_{g}_{b}': i for i, k in enumerate(label_map.keys()) for r, g, b in label_map[k]}


def preprocess_image(src_dst_path):
    W, H = int(os.environ['W']), int(os.environ['H'])
    src_path, dst_path = src_dst_path
    img = Image.open(src_path).convert('RGB').resize((W, H))
    img.save(dst_path)


def preprocess_label(src_dst_path):
    W, H = int(os.environ['W']), int(os.environ['H'])
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
    set_config()
    set_label_map()
    data_dir = os.environ['DATA_DIR']

    for t in ['train', 'val']:
        # preprocess: image => downsize
        src_path_list = sorted(glob(f'{data_dir}/leftImg8bit/{t}/*/*'))
        dst_path_list = [f'{data_dir}/image/{t}/X{i:05}.png' for i in range(len(src_path_list))]

        os.makedirs(f'{data_dir}/image/{t}', exist_ok=True)
        with Pool(processes=PARALLEL_NUM) as p:
            m = p.imap(preprocess_image, zip(src_path_list, dst_path_list))
            list(tqdm(m, desc=f'preprocess {t} image files', total=len(src_path_list)))

        # preprocess: label => rgb2lbl, downsize
        src_path_list = sorted(glob(f'{data_dir}/gtFine/{t}/*/*_color.png'))
        dst_path_list = [f'{data_dir}/label/{t}/Y{i:05}.png' for i in range(len(src_path_list))]

        os.makedirs(f'{data_dir}/label/{t}', exist_ok=True)
        with Pool(processes=PARALLEL_NUM) as p:
            m = p.imap(preprocess_label, zip(src_path_list, dst_path_list))
            list(tqdm(m, desc=f'preprocess {t} label files', total=len(src_path_list)))
