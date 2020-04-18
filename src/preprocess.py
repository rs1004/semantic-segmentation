from pathlib import Path
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from config import Config
import json
import numpy as np

CONFIG = Config()


def set_label_map():
    with open(CONFIG.ROOT_DIR / 'labelmap.json', 'r') as f:
        label_map = json.load(f)
    global LABEL_MAP
    LABEL_MAP = {f'{r}_{g}_{b}': k['id'] for k in label_map.values() for r, g, b in k['colors']}


def preprocess_image(src_dst_path):
    W, H = CONFIG.W, CONFIG.H
    src_path, dst_path = src_dst_path
    img = Image.open(src_path).convert('RGB').resize((W, H))
    img.save(dst_path)


def preprocess_label(src_dst_path):
    W, H = CONFIG.W, CONFIG.H
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

    for t in ['train', 'test', 'val']:
        # preprocess: image => downsize
        src_path_list = sorted((CONFIG.DATA_DIR / 'leftImg8bit' / t).glob('*/*'))
        dst_path_list = (CONFIG.IMAGE_DIR / t / f'{i:05}.png' for i in range(len(src_path_list)))

        (CONFIG.IMAGE_DIR / t).mkdir(parents=True, exist_ok=True)
        with Pool(processes=CONFIG.PARALLEL_NUM) as p:
            m = p.imap(preprocess_image, zip(src_path_list, dst_path_list))
            list(tqdm(m, desc=f'preprocess {t} image files', total=len(src_path_list)))

        # preprocess: label => rgb2lbl, downsize
        src_path_list = sorted((CONFIG.DATA_DIR / 'gtFine' / t).glob('*/*_color.png'))
        dst_path_list = (CONFIG.LABEL_DIR / t / f'{i:05}.png' for i in range(len(src_path_list)))

        (CONFIG.LABEL_DIR / t).mkdir(parents=True, exist_ok=True)
        with Pool(processes=CONFIG.PARALLEL_NUM) as p:
            m = p.imap(preprocess_label, zip(src_path_list, dst_path_list))
            list(tqdm(m, desc=f'preprocess {t} label files', total=len(src_path_list)))
