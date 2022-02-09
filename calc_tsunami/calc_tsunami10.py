# 目的：最大振幅から津波の高さを推測（回帰）
# 入力データ：地震加速度波形の最大振幅・・・バンドパスフィルター済み1つと震央距離
# 　　　　　　訓練：テスト＝8:2
# 正解データ：津波の高さ。低いものは使用しない。
# 正解と出力をプロット
# 勝間田教授のoutファイル14個を使用
#
# 地形による補正をする
# 2011データは省く
# 震央距離はオフセットをとる
# 最大振幅は対数を取る・・スケールの小さい部分を拡大する
# 震央距離は負にする

import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import math
import time

# ===========================================================================================================================================
# 使用データファイルの取得
# files = glob.glob("../datasets/tsunami/omit_2011/*")
files = glob.glob("../datasets/tsunami/omit_2011_b3/*")

# ＊１（75行目）ファイル初期化
f_reset = open("../datasets/tsunami/6/data","w")
f_reset.write("")

# ===========================================================================================================================================
# 正解データ
# ---------------------------------------------------------------------------------------------------------------------------------
# 津波データ全体をリスト化
tsunami_data = []
dis_list = []
sesmic_dis1_list = []
sesmic_dis2_list = []
sesmic_dis3_list = []
i = 0
for file in files:
    f=open(file,'r')
    j = 0
    # while True:
    data=f.readline()
    j += 1
    if data == '':
        break
    if j == 1:
        while True:
            data2 = f.readline()
            if data2 == '':
                break
            data3 = data2.split(",")
            if data3[4] != 'null':
                lat = float(data3[0])
                lon = float(data3[1])
                st_lat = float(data3[2])
                st_lon = float(data3[3])
                seismic_dis1 = float(data3[4])
                seismic_dis2 = float(data3[5])
                seismic_dis3 = float(data3[6])
                tsunami_h = float(data3[7])
                date = float(data3[8])
                
                pole_radius = 6356752.314245 # 極半径
                equator_radius = 6378137.0 # 赤道半径

                lat_rad = math.radians(lat)
                lon_rad = math.radians(lon)
                st_lat_rad = math.radians(st_lat)
                st_lon_rad = math.radians(st_lon)
                
                lat_difference = lat_rad - st_lat_rad # 緯度差
                lon_difference = lon_rad - st_lon_rad # 経度差
                lat_average = (lat_rad + st_lat_rad) / 2 # 平均緯度

                e2 = (math.pow(equator_radius, 2) - math.pow(pole_radius, 2)) / math.pow(equator_radius, 2) # 第一離心率^2
                w = math.sqrt(1- e2 * math.pow(math.sin(lat_average), 2))
                m = equator_radius * (1 - e2) / math.pow(w, 3) # 子午線曲率半径
                n = equator_radius / w # 卯酉線曲半径
                dis = math.sqrt(math.pow(m * lat_difference, 2) + math.pow(n * lon_difference * math.cos(lat_average), 2)) / 1000# 距離計測
                # if 100 <= dis <= 200:
                # if tsunami_h >= 10:
                # リアス式海岸の場合、振幅を1.5倍とみなす

                # if 42.8 <= st_lat <= 43.2 and 144.0 <= st_lon <= 144.5 :
                #     tsunami_h = tsunami_h / 3
                # elif 43.0 <= st_lat <= 43.3 and 144.3 <= st_lon <= 145.2 :
                #     tsunami_h = tsunami_h / 3
                # elif 33.5 <= st_lat <= 33.9 and 135.9 <= st_lon <= 136.1 :
                #     tsunami_h = tsunami_h / 3
                # elif 38.1 <= st_lat <= 38.5 and 140.8 <= st_lon <= 141.0 :
                #     tsunami_h = tsunami_h / 3

                data3_list = [seismic_dis1 ,seismic_dis2 ,seismic_dis3 ,tsunami_h]
                tsunami_data.append(data3_list)
                dis_list.append(dis)
                sesmic_dis1_list.append(seismic_dis1)
                sesmic_dis2_list.append(seismic_dis2)
                sesmic_dis3_list.append(seismic_dis3)
                # ＊１使用した震央距離と最大振幅と津波高さを別ファイルに出力
                f2 = open("../datasets/tsunami/6/data","a")
                # f2.write("\n"+str(lat)+"    "+str(lon)+"    "+str(st_lat)+"    "+str(st_lon)+"    "+str(dis)+"    "+str(seismic_dis1)+"   "+str(seismic_dis2)+"    "+str(seismic_dis3)+"   "+str(tsunami_h))
                f2.write("\n"+str(lat)+"    "+str(lon)+"    "+str(st_lat)+"    "+str(st_lon)+"    "+str(dis)+"    "+str(seismic_dis1)+"   "+str(seismic_dis2)+"    "+str(seismic_dis3)+"   "+str(tsunami_h)+"   "+str(date))

# ===========================================================================================================================================
# データの前処理
# ---------------------------------------------------------------------------------------------------------------------------------
# 震央距離にオフセットを取って負にする
ave_dis = np.average(dis_list,axis=0)
std_dis = np.std(dis_list,axis=0)
dis_list = -(dis_list - ave_dis) / std_dis
# ---------------------------------------------------------------------------------------------------------------------------------
# 震央距離に対数を取って負にする
# dis = -np.log(dis)
# ---------------------------------------------------------------------------------------------------------------------------------
# 最大振幅にオフセットを取る
ave_sesmic_dis1 = np.average(sesmic_dis1_list,axis=0)
std_sesmic_dis1 = np.std(sesmic_dis1_list,axis=0)
sesmic_dis1_list = (sesmic_dis1_list - ave_sesmic_dis1) / std_sesmic_dis1

ave_sesmic_dis2 = np.average(sesmic_dis2_list,axis=0)
std_sesmic_dis2 = np.std(sesmic_dis2_list,axis=0)
sesmic_dis2_list = (sesmic_dis2_list - ave_sesmic_dis2) / std_sesmic_dis2

ave_sesmic_dis3 = np.average(sesmic_dis3_list,axis=0)
std_sesmic_dis3 = np.std(sesmic_dis3_list,axis=0)
sesmic_dis3_list = (sesmic_dis3_list - ave_sesmic_dis3) / std_sesmic_dis3

# ---------------------------------------------------------------------------------------------------------------------------------
# 最大振幅に対数を取る
# sesmic_dis1_list = np.log(sesmic_dis1_list)
# sesmic_dis2_list = np.log(sesmic_dis2_list)
# sesmic_dis3_list = np.log(sesmic_dis3_list)

print(len(tsunami_data))
time.sleep(2)
# ---------------------------------------------------------------------------------------------------------------------------------
# 津波の高さ(正解データ)をリスト化
correct_data = []
for t in tsunami_data:
    correct_data.append([t[3]])

# ===========================================================================================================================================
# 入力データ
input_data = []
i = 0
for t in tsunami_data:
    # input_data.append([dis_list[i], sesmic_dis1_list[i], sesmic_dis2_list[i], sesmic_dis3_list[i]])
    input_data.append([sesmic_dis1_list[i], sesmic_dis2_list[i], sesmic_dis3_list[i]])
    # input_data.append([dis_list[i], sesmic_dis1_list[i], sesmic_dis2_list[i]])
    # input_data.append([dis_list[i], sesmic_dis3_list[i]])
    i += 1


# ===========================================================================================================================================
# 訓練データとテストデータに分ける(訓練：テスト＝8:2)
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
n_mid = 200
n_out = 1
wb_width = 0.1
eta = 0.001
epoch = 12000
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
        # self.y = np.where(u <= 0, 0, u)
        # self.y = u
        # print("mid for")
    
    def backward(self ,grad_y):
        delta = grad_y * (1 - self.y) * self.y
        self.grad_w = np.dot(self.x.T ,delta)
        self.grad_b = np.sum(delta ,axis = 0)
        self.grad_x = np.dot(delta ,self.w.T)
        # print("mid back")
        # print(self.grad_x)

class OutputLayer(BaseLayer):
    def forward(self ,x):
        self.x = x
        u = np.dot(x ,self.w) + self.b
        self.y = u
        # print("out for")

    def backward(self ,t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T ,delta)
        self.grad_b = np.sum(delta ,axis=0)
        self.grad_x = np.dot(delta ,self.w.T)
        # print("out back")
        # print(self.grad_x)



# ===========================================================================================================================================
# 各層のインスタンスの生成
middle_layer_1 = MiddleLayer(n_in ,n_mid)
middle_layer_2 = MiddleLayer(n_mid ,n_mid)
output_layer = OutputLayer(n_mid ,n_out)


# ===========================================================================================================================================
# 関数の定義
def forward_propagation(x):
    middle_layer_1.forward(x)
    # middle_layer_2.forward(middle_layer_1.y)
    # output_layer.forward(middle_layer_2.y)
    # 中間層1層
    output_layer.forward(middle_layer_1.y)


def backpropagation(t):
    output_layer.backward(t)
    # middle_layer_2.backward(output_layer.grad_x)
    # middle_layer_1.backward(middle_layer_2.grad_x)
    # 中間層1層
    middle_layer_1.backward(output_layer.grad_x)

def update_wb():
    middle_layer_1.update(eta)
    # middle_layer_2.update(eta)
    output_layer.update(eta)

def get_error(t ,batch_size ):
    return 1.0 / 2.0 * np.sum(np.square(output_layer.y - t)) / batch_size
    # return np.abs(np.sum(output_layer.y - t) / batch_size)


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

# =============================================================
def get_error2(y, t  ):
    return 1.0 / 2.0 * np.sum(np.square(y - t))

def calc(x):
    return 100/250*x+20

test_error_x2=[]
test_error_y2=[]
count=0
for i in range(len(correct_y)):
    error_test = get_error2(calc(correct_y[i][0]),correct_y[i][0]  )
    test_error_x2.append(i)
    test_error_y2.append(error_test)
    count=count+error_test

ave=count/len(correct_y)
print(ave)
plt.scatter(test_error_x2,test_error_y2)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.grid(True)
plt.show()
exit()

# ================================================================
from numpy.linalg import solve
xval = []
yval = []

for i in range(len(correct_y)):    
    xval.append(correct_y[i-1][0])
    yval.append(output_x[i-1][0])
#---------------------------------------------------
a = []
for i in range(len(xval)):
    a.append([xval[i], 1.0])

a_arr = np.array(a)
b_arr = np.dot(a_arr.T , a_arr)
d_arr = np.dot(a_arr.T,yval)
m_arr = solve(b_arr,d_arr)
print("m=",m_arr)
#---------------------------------------------------
x_p = np.array([np.min(xval),np.max(xval)])
y_p = m_arr[0] * x_p + m_arr[1]
plt.title("data plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(xval,yval,"o")
# plt.savefig("lsqs_fig01.png")
plt.plot(x_p,y_p,color="red")
# plt.savefig("lsqs_fig02.png")
plt.show()

