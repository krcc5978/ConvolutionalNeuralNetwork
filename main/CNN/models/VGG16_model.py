from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential


class VGG16:

    def __init__(self, input_shape, label):
        self.model = Sequential()

        self.model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Conv2D(filters=128, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Conv2D(filters=256, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Conv2D(filters=512, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Conv2D(filters=512, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3),
                              activation='relu',
                              strides=(1, 1), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(4096))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dropout(rate=0.5))

        # 分類したい人数を入れる
        self.model.add(Dense(label))
        self.model.add(Activation('softmax'))

    def get_model(self):
        return self.model
