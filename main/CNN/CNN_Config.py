import numpy as np
from keras import optimizers
from main.CNN.callback_method import model_checkpoint


class CNN_Config:

    def __init__(self, input_shape, label_count, use_model_name='ResNet'):
        """
        :param input_shape: 入力サイズ
        :param label_count: ラベル数
        :param use_model_name: 使用する学習モデル
        """
        if use_model_name == 'ResNet':
            from main.CNN.models.ResNet_model import ResNet as use_model
        elif use_model_name == 'AlexNet':
            from main.CNN.models.AlexNet_model import AlexNet as use_model
        elif use_model_name == 'VGG':
            from main.CNN.models.VGG16_model import VGG16 as use_model
        else:
            from main.CNN.models.load_model import load_model as use_model

        self.model = use_model(input_shape, label_count).get_model()

    def load_weight(self, weight_path):
        """
        :param weight_path: 使用する重みファイルのパス
        :return:
        """
        self.model.load_weights(weight_path)

    def recognition(self, image, batch_size=1):
        """
        :param image: 画像データ
        :param batch_size: バッチサイズ
        :return: 認証結果
        """
        test_image_array = np.array(image)
        return self.model.predict(test_image_array, batch_size=batch_size)

    def trainning_start(self, train_data, vali_data, num_train, num_val, batch_size, model_save_path='./'):
        """
        :param train_data: 学習データのgenerator
        :param vali_data: 検証データのgenerator
        :param num_train: 学習データの数
        :param num_val: 検証データの数
        :param batch_size: バッチサイズ
        :param model_save_path: 学習モデルの保存場所
        :return:
        """

        # オプティマイザーの設定
        optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-4)

        # 使用するモデルのコンパイル
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # モデルの保存
        model_json_str = self.model.to_json()
        open(model_save_path + 'face_model.json', 'w').write(model_json_str)

        # コールバック関数の宣言
        checkpoint = model_checkpoint('./logs/000/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                      'val_loss',
                                      True,
                                      True,
                                      3)

        # 学習
        self.model.fit_generator(train_data,
                                 steps_per_epoch=max(1, num_train // batch_size),
                                 validation_data=vali_data,
                                 validation_steps=max(1, num_val // batch_size),
                                 epochs=10000,
                                 initial_epoch=0,
                                 callbacks=[checkpoint]
                                 )

        # 結果の出力
        self.model.save_weights('./main/face_model_weights.h5')
