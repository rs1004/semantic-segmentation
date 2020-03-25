import tensorflow as tf
import re
from pathlib import Path
from config import Config
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

CONFIG = Config()


class UNet:
    def __init__(self, input_shape, class_num):
        self.input_shape = input_shape
        self.class_num = class_num
        self.model = self.create_model()

    def __call__(self, x):
        return self.model.predict(x)

    def optimize(self, ds, epochs, steps, resume=True):
        ckpt_path = CONFIG.RESULT_DIR / 'model-{epoch:04d}.ckpt'
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path.as_posix(),
            save_weights_only=True,
            verbose=1,
            period=CONFIG.SAVE_PERIODS
        )
        if resume and ckpt_path.parent.exists():
            latest = tf.train.latest_checkpoint(CONFIG.RESULT_DIR)
            self.model.load_weights(latest)
            initial_epoch = int(re.findall(r'\d{4}', latest)[0])
        else:
            initial_epoch = 0

        self.model.fit(ds, initial_epoch=initial_epoch, epochs=initial_epoch + epochs, steps_per_epoch=steps, callbacks=[ckpt_callback])

    def create_model(self):
        inputs = Input(shape=(self.input_shape))

        conv1_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        conv1_2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(conv1_1)
        pool1 = MaxPool2D(pool_size=2, strides=2)(conv1_2)

        conv2_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(pool1)
        conv2_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(conv2_1)
        pool2 = MaxPool2D(pool_size=2, strides=2)(conv2_2)

        conv3_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(pool2)
        conv3_2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(conv3_1)
        pool3 = MaxPool2D(pool_size=2, strides=2)(conv3_2)

        conv4_1 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(pool3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(conv4_1)
        pool4 = MaxPool2D(pool_size=2, strides=2)(conv4_2)

        conv5_1 = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')(pool4)
        conv5_2 = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')(conv5_1)
        concated1 = concatenate([conv4_2, Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='same')(conv5_2)])

        conv6_1 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(concated1)
        conv6_2 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(conv6_1)
        concated2 = concatenate([conv3_2, Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='same')(conv6_2)])

        conv7_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(concated2)
        conv7_2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(conv7_1)
        concated3 = concatenate([conv2_2, Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding='same')(conv7_2)])

        conv8_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(concated3)
        conv8_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(conv8_1)
        concated4 = concatenate([conv1_2, Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='same')(conv8_2)])

        conv9_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(concated4)
        conv9_2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(conv9_1)
        logits = Conv2D(filters=self.class_num, kernel_size=1, padding='same', activation='softmax')(conv9_2)

        model = Model(inputs=inputs, outputs=logits)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

        return model
