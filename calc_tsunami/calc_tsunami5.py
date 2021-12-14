# 目的：最大振幅から津波の高さを推測（回帰）
# 入力データ：地震加速度波形の最大振幅・・・バンドパスフィルター済み
# 　　　　　　訓練：テスト＝８：２
# 正解データ：津波の高さ
# 正解と出力をプロット

import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# ===========================================================================================================================================
# 使用データファイルの取得
files = glob.glob("../datasets/tsunami/4/merged.dat_b4.2016")


# ===========================================================================================================================================
# 正解データ
# ---------------------------------------------------------------------------------------------------------------------------------
# 津波データ全体をリスト化
tsunami_data = []
data_row = []
i = 0
for file in files:
    f=open(file,'r')
    while True:
        data=f.readline()
        if data == '':
            break
        if 'lat,lon,st_lat,st_lon,1-3s,3-9s,9-27s,27-81s,tsunami_h(m),date' in data:
            while True:
                data2 = f.readline()
                if data2 == '':
                    break
                data3 = data2.split(",")
                if data3[4] != 'null':
                    seismic_dis1 = float(data3[4])
                    seismic_dis2 = float(data3[5])
                    seismic_dis3 = float(data3[6])
                    seismic_dis4 = float(data3[7])
                    tsunami_h = float(data3[8])
                    data3_list = [seismic_dis1 ,seismic_dis2 ,seismic_dis3 ,seismic_dis4 ,tsunami_h]
                    tsunami_data.append(data3_list)

# ---------------------------------------------------------------------------------------------------------------------------------
# 津波の高さ(正解データ)をリスト化
correct_data = []
for t in tsunami_data:
    correct_data.append([t[4]])

# ===========================================================================================================================================
# 入力データ
input_data = []
for t in tsunami_data:
    input_data.append([t[0], t[1], t[2], t[3]])


# ===========================================================================================================================================
# 訓練データとテストデータに分ける(訓練：テスト＝８：２)
# (リストは配列に変換)
index = list(range(len(input_data)))
random.shuffle(index)

index_train = index[:round(len(input_data)*8/10)]
index_test = index[round(len(input_data)*8/10):]

input_data = np.array(input_data)
correct_data = np.array(correct_data)

input_train = input_data[index_train , : ]
input_test = input_data[index_test , : ]
correct_train = correct_data[index_train]
correct_test = correct_data[index_test]

n_train = len(input_train)
n_test = len(input_test)



# ===========================================================================================================================================
# ===========================================================================================================================================
# ハイパーパラメータの定義
n_in = len(input_train[0])
n_mid = 25
n_out = 1
wb_width = 0.1
#下げてみる？
eta = 0.01
epoch = 10
batch_size = 4
interval = epoch / 10


# ===========================================================================================================================================
# 各層のクラスの定義
class BaseLayer:
    def __init__(self ,n_upper ,n):
        self.w = wb_width * np.random.randn(n_upper ,n)
        self.b = wb_width * np.random.randn(n)
    
    def update(self ,eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

class MiddleLayer(BaseLayer):
    def forward(self ,x):
        self.x = x
        u = np.dot(x ,self.w) + self.b
        self.y = 1 / (1 + np.exp(-u))
        print("Middle Forward")
    
    def backward(self ,grad_y):
        delta = grad_y * (1 - self.y) * self.y
        self.grad_w = np.dot(self.x.T ,delta)
        self.grad_b = np.sum(delta ,axis = 0)
        self.grad_x = np.dot(delta ,self.w.T)
        print("Middle Back")

class OutputLayer(BaseLayer):
    def forward(self ,x):
        self.x = x
        u = np.dot(x ,self.w) + self.b
        self.y = u
        print("Out Forward")

    def backward(self ,t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T ,delta)
        self.grad_b = np.sum(delta ,axis=0)
        self.grad_x = np.dot(delta ,self.w.T)
        print("Out Back")



# ===========================================================================================================================================
# 各層のインスタンスの生成
middle_layer_1 = MiddleLayer(n_in ,n_mid)
middle_layer_2 = MiddleLayer(n_mid ,n_mid)
output_layer = OutputLayer(n_mid ,n_out)


# ===========================================================================================================================================
# 関数の定義
def forward_propagation(x):
    middle_layer_1.forward(x)
    middle_layer_2.forward(middle_layer_1.y)
    output_layer.forward(middle_layer_2.y)

def backpropagation(t):
    output_layer.backward(t)
    middle_layer_2.backward(output_layer.grad_x)
    middle_layer_1.backward(middle_layer_2.grad_x)

def update_wb():
    middle_layer_1.update(eta)
    middle_layer_2.update(eta)
    output_layer.update(eta)

def get_error(t ,batch_size ):
    return 1.0 / 2.0 * np.sum(np.square(output_layer.y - t)) / batch_size


# ===========================================================================================================================================
# 誤差の記録と学習
# ---------------------------------------------------------------------------------------------------------------------------------
# 誤差の記録
train_error_x=[]
train_error_y=[]
test_error_x=[]
test_error_y=[]

n_batch = n_train // batch_size
for i in range(epoch):
    forward_propagation(input_train)
    error_train = get_error(correct_train ,n_train )
    forward_propagation(input_test)
    error_test = get_error(correct_test ,n_test )

    train_error_x.append(i)
    train_error_y.append(error_train)
    test_error_x.append(i)
    test_error_y.append(error_test)

    if i%interval==0:
        print("Epoch:"+str(i+1)+"/"+str(epoch),
            "Error_train:"+str(error_train),
            "Error_test:"+str(error_test))
    # ---------------------------------------------------------------------------------------------------------------------------------
    # 学習
    index_random=np.arange(n_train)
    np.random.shuffle(index_random)

    # ミニバッチ学習
    for j in range(n_batch):
        mb_index = index_random[j*batch_size:(j+1)*batch_size]
        x = input_train[mb_index , : ]
        t = correct_train[mb_index]

        forward_propagation(x)
        backpropagation(t)
        update_wb()


# ===========================================================================================================================================
# 誤差の記録をグラフ表示
plt.plot(train_error_x,train_error_y,label="Train")
plt.plot(test_error_x,test_error_y,label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()


# ===========================================================================================================================================
# 正解率の測定
forward_propagation(input_train)
count_train = np.sum(np.argmax(output_layer.y ,axis = 1) == np.argmax(correct_train ,axis = 1))
forward_propagation(input_test)
count_test = np.sum(np.argmax(output_layer.y ,axis = 1) == np.argmax(correct_test ,axis = 1))

print("Accuracy Train:" + str(count_train / n_train * 100)+"%" ,"Accuracy Test:" + str(count_test / n_test * 100) + "%")


# ===========================================================================================================================================
# 正解と出力のプロット
output_x=output_layer.y
correct_y=correct_test

plt.scatter(correct_y ,output_x)
plt.ylabel("out put")
plt.xlabel("correct")
plt.grid(True)
plt.show()