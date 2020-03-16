import tensorflow as tf
import math
from data_generator import DataGenerator

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(400, 800, 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(strides=2),
    tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

data_gen_train = DataGenerator('train')
data_gen_val = DataGenerator('val')

ds_train = data_gen_train.get_one_shot_iterator()

model.fit(ds_train, epochs=5, steps_per_epoch=math.ceil(data_gen_train.data_length/data_gen_train.batch_size))

ds_val = data_gen_val.get_one_shot_iterator()

model.evaluate(ds_val, steps=math.ceil(data_gen_val.data_length/data_gen_val.batch_size))