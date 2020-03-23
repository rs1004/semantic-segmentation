import tensorflow as tf
import re
from pathlib import Path
from config import Config

CONFIG = Config()


class SampleNet:
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
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            # tf.keras.layers.MaxPooling2D(strides=2),
            tf.keras.layers.Conv2D(filters=self.class_num, kernel_size=1, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

        return model
