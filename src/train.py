from data_generator import DataGenerator
from model import SampleNet
from config import Config

CONFIG = Config()

data_gen_train = DataGenerator('train')
ds_train = data_gen_train.get_one_shot_iterator()

model = SampleNet(input_shape=(data_gen_train.H, data_gen_train.W, 3), class_num=CONFIG.CLASS_NUM)

model.optimize(ds=ds_train, epochs=CONFIG.EPOCHS, steps=data_gen_train.data_length // CONFIG.BATCH_SIZE)
