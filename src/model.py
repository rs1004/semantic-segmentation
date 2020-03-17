import tensorflow as tf


class SampleNet:
    def __init__(self, input_shape, class_num):
        self.input_shape = input_shape
        self.class_num = class_num
        self.model = self.create_model()

    def __call__(self, x):
        return self.model.predict(x)

    def optimize(self, ds, epochs, steps, resume=True):
        # TODO： check pointを参照する処理
        self.model.fit(ds, epochs=epochs, steps_per_epoch=steps)

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
