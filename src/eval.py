from data_generator import DataGenerator
from model import UNet
from config import Config

CONFIG = Config()

data_gen_val = DataGenerator('val')
ds_val = data_gen_val.get_one_shot_iterator()

model = UNet(input_shape=(data_gen_val.H, data_gen_val.W, 3), class_num=CONFIG.CLASS_NUM)

model.evaluate(ds=ds_val, steps=data_gen_val.data_length // CONFIG.BATCH_SIZE)