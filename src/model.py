import tensorflow as tf
import re
from pathlib import Path
from config import Config
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from metrics import get_metrics


CONFIG = Config()


class UNet:
    def __init__(self, input_shape, class_num, resume=True):
        self.input_shape = input_shape
        self.class_num = class_num

        self.initialize_model(resume=resume)

    def __call__(self, x):
        return self.model.predict(x, steps=1)

    def optimize(self, ds, epochs, steps):
        self.model.fit(
            ds,
            initial_epoch=self.model_latest_epoch,
            epochs=epochs,
            steps_per_epoch=steps,
            callbacks=self.callbacks
        )

    def evaluate(self, ds, steps):
        scores = self.model.evaluate(ds, verbose=2, steps=steps)
        return dict(zip(self.model.metrics_names, scores))

    def initialize_model(self, resume):
        self.create_model()
        if resume and CONFIG.RESULT_DIR.exists():
            latest = tf.train.latest_checkpoint(CONFIG.RESULT_DIR)
            self.model.load_weights(latest)
            self.model_latest_epoch = int(re.findall(r'\d{4}', latest)[0])
        else:
            self.model_latest_epoch = 0
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=(CONFIG.RESULT_DIR / 'model-{epoch:04d}.ckpt').as_posix(),
                monitor='loss',
                save_weights_only=True,
                save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=1e-4,
                patience=5),
            tf.keras.callbacks.TensorBoard(
                log_dir=CONFIG.RESULT_DIR,
                histogram_freq=1,
                write_graph=True,
                write_images=True)
            ]

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
            optimizer='Adam',
            loss=sparse_categorical_crossentropy,
            metrics=get_metrics()
        )

        self.model = model
