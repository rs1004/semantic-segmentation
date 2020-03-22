from pathlib import Path


class Config:
    def __init__(self):
        self.H = 256
        self.W = 512
        self.ROOT_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT_DIR / 'data'
        self.TFRECORD_DIR = self.ROOT_DIR / 'data' / 'tfrecord'
        self.BATCH_SIZE = 32
