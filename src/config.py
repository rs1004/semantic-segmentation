from pathlib import Path


class Config:
    def __init__(self):
        # image spec
        self.H = 256
        self.W = 512

        # process spec
        self.TFRECORD_FILE_NUM = 2
        self.BATCH_SIZE = 32

        # path
        self.ROOT_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT_DIR / 'data'
        self.IMAGE_DIR = self.DATA_DIR / 'image'
        self.LABEL_DIR = self.DATA_DIR / 'label'
        self.TFRECORD_DIR = self.DATA_DIR / 'tfrecord'
