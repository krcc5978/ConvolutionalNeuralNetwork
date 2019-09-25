# 入力サイズ
input_shape = (224, 224, 3)

# バッチサイズ
batch_size = 32

# 入力データ格納場所
base_path = 'D:\\data\\商用利用不可\\動物\\'
teacher_directory_list = ['1', '2']

# 重みファイル出力場所
weight_path = './logs/000/ep003-loss10.074-val_loss1.537.h5'

# 使用するCNNモデル
use_model_name = 'AlexNet'
