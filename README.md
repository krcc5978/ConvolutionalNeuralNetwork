# CounvolutionalNeuralNetwork
画像認識用AI

## ファイル構造
```
D:.
│
└─main
    │  cnn.py
    │  const_value.py
    │
    ├─CNN
    │  │  callback_method.py
    │  │  CNN_Config.py
    │  │  utils.py
    │  │
    │  └─models
    │  　  　  AlexNet_model.py
    │  　  　  load_model.py
    │  　  　  ResNet_model.py
    │  　  　  VGG16_model.py
    │
    └─logs
    　  └─000
    
```

`cnn.py` ： 認証/学習を開始するファイル <br>
`const_value.py` : 定数定義ファイル <br>
`callback_method.py` : 学習時のコールバックメソッドが定義されているファイル <br>
`CNN_Config.py` : CNNの設定ファイル <br>
`utils.py` : CNNので使用する関数が格納されているファイル <br>
`models`: 各モデルが定義されている`.py`が格納されているディレクトリ


## 認証/学習方法
`const_value.py 8～9行目`で読み込ませたいデータが格納されているディレクトリを選択
```
（例）
base_path = 'D:\\data\\動物\\'
teacher_directory_list = ['猫', '犬', '馬']
```