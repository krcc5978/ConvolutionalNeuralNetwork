from keras.initializers import TruncatedNormal, Constant
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential


def conv2d(filters, kernel_size, strides=(1, 1), padding='same', bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters, kernel_size, strides=strides, padding=padding,
        activation='relu', kernel_initializer=trunc, bias_initializer=cnst, **kwargs
    )


def dense(units, activation='tanh'):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units, activation=activation,
        kernel_initializer=trunc, bias_initializer=cnst,
    )


class AlexNet:

    def __init__(self, input_shape, label):
        self.model = Sequential()

        # 第1畳み込み層
        self.model.add(conv2d(96, 11, strides=(4, 4), bias_init=0, input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(BatchNormalization())

        # 第２畳み込み層
        self.model.add(conv2d(256, 5, bias_init=1))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(BatchNormalization())

        # 第３~5畳み込み層
        self.model.add(conv2d(384, 3, bias_init=0))
        self.model.add(conv2d(384, 3, bias_init=1))
        self.model.add(conv2d(256, 3, bias_init=1))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(BatchNormalization())

        # 密結合層
        self.model.add(Flatten())
        self.model.add(dense(4096))
        self.model.add(Dropout(0.5))
        self.model.add(dense(4096))
        self.model.add(Dropout(0.5))

        # 読み出し層
        self.model.add(Dense(label, activation='softmax'))

    def get_model(self):
        return self.model
