from keras.layers import Activation, Conv2D, Dense, MaxPooling2D, Dropout, Input, BatchNormalization, add, \
    GlobalAveragePooling2D
from keras.models import Model


def _shortcut(inputs, residual):
    # _keras_shape[3] チャンネル数
    n_filters = residual._keras_shape[3]

    # inputs と residual とでチャネル数が違うかもしれない。
    # そのままだと足せないので、1x1 conv を使って residual 側のフィルタ数に合わせている
    shortcut = Conv2D(n_filters, (1, 1), strides=(1, 1), padding='valid')(inputs)

    # 2つを足す
    return add([shortcut, residual])


def _resblock(n_filters, strides=(1, 1)):
    def f(input):
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = Conv2D(n_filters, (3, 3), strides=strides,
                   kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.4)(x)
        x = Conv2D(n_filters, (3, 3), strides=strides,
                   kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)

        return _shortcut(input, x)

    return f


class ResNet:

    def __init__(self, input_shape, label):
        # モデルの定義

        inputs = Input(shape=input_shape)
        x = Conv2D(32, (7, 7), strides=(1, 1),
                   kernel_initializer='he_normal', padding='same')(inputs)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = _resblock(n_filters=64)(x)
        x = Activation('relu')(x)
        x = _resblock(n_filters=64)(x)
        x = Activation('relu')(x)
        x = _resblock(n_filters=64)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = _resblock(n_filters=128)(x)
        x = Activation('relu')(x)
        x = _resblock(n_filters=128)(x)
        x = Activation('relu')(x)
        x = _resblock(n_filters=128)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1000, kernel_initializer='he_normal', activation='sigmoid')(x)
        x = Dense(label, kernel_initializer='he_normal', activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=x)

    def get_model(self):
        return self.model
