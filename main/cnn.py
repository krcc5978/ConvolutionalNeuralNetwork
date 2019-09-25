import numpy as np
from main.const_value import *
from main.CNN.CNN_Config import CNN_Config
from main.CNN.utils import make_teacher_data, data_generator, image_open


def train():
    # CNNの設定作成
    cnn_config = CNN_Config(input_shape, len(teacher_directory_list), use_model_name)

    # 教師データの作成、分割
    train_data, val_data = make_teacher_data(base_path, teacher_directory_list)

    # generatorの作成
    train_generator = data_generator(train_data, batch_size)
    val_generator = data_generator(val_data, batch_size)

    # 学習の開始
    cnn_config.trainning_start(train_generator, val_generator, len(train_data), len(val_data), batch_size)


def predict():
    # CNNの設定作成
    cnn_config = CNN_Config(input_shape, len(teacher_directory_list), use_model_name)

    # 重みの読み込み
    cnn_config.load_weight(weight_path)

    # リサイズサイズ
    width = input_shape[0]
    height = input_shape[1]

    # テストデータの作成
    test_data, _ = make_teacher_data(base_path, teacher_directory_list, 0)

    # テストデータの件数分認証を行う
    for predict_data in test_data:
        # 画像の読み込み、リサイズ
        image = image_open(predict_data[0], width, height)

        # 認証結果をnumpy形式で取得
        result = np.array(cnn_config.recognition([image])[0])

        # 認証結果のインデックスとスコアの最大値を取得
        result_index = np.argmax(result)
        result_max = np.max(result)

        # 正解データのインデックスとスコアの最大値の取得
        answer = np.array(predict_data[1])
        answer_index = np.argmax(answer)
        answer_max = np.max(answer)
        print('-----------------------------------------------------------------')
        print('answer index : {} \t answer max : {}'.format(str(answer_index), str(answer_max)))
        print('result index : {} \t result max : {}'.format(str(result_index), str(result_max)))
        print('-----------------------------------------------------------------')


if __name__ == '__main__':
    """
    学習 → train
    認証 → predict
    """
    train()
    # predict()
