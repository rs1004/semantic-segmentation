from pathlib import Path
from multiprocessing import cpu_count


class Config:
    def __init__(self):
        # image spec
        self.H = 512
        self.W = 1024
        self.CLASS_NUM = 8

        # process spec
        self.PARALLEL_NUM = cpu_count()
        self.TFRECORD_FILE_NUM = 2
        self.BATCH_SIZE = 16
        self.EPOCHS = 100
        self.SAVE_PERIODS = 5

        # path
        self.ROOT_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT_DIR / 'data'
        self.IMAGE_DIR = self.DATA_DIR / 'image'
        self.LABEL_DIR = self.DATA_DIR / 'label'
        self.TFRECORD_DIR = self.DATA_DIR / 'tfrecord'
        self.RESULT_DIR = self.ROOT_DIR / 'result'
