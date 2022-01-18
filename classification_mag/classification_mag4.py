# 目的：地震加速度波形からマグニチュード6 or 7を分類（分類）
# 入力データ：フーリエ変換した地震加速度波形
# 正解データ：マグニチュード6 or 7

import numpy as np
import matplotlib.pyplot as plt
import glob
import random

#データ処理
#拡張子がUDのファイルの震源地、マグニチュードを配列に格納
files=glob.glob("../datasets/mag/datasets_plot/*")
# files=glob.glob("../datasets/mag/datasets_use2/*")
n_in=6000



print('====================================================================')
print('[インデックス、緯度、経度、マグニチュード]----------------------')
lat_lon=[]
i=0
for file in files:
    # print(file)
    f=open(file,'r')
    while True:
        data=f.readline()
        if 'Lat.' in data:
            ev_lat = float(data[19-1:30])
            # print("lat=",ev_lat)
            lat_lon.append([i+1,ev_lat,'tmp','tmp'])
        elif 'Long.' in data:
            ev_lon = float(data[19-1:30])
            # print("lon=",ev_lon)
            lat_lon[i][2]=ev_lon
        elif 'Mag.' in data:
            mag=float(data[19-1:30])
            f.close() 
            # print("mag=",mag)
            lat_lon[i][3]=mag
            i+=1
            break
print("len(lat_lon)=",len(lat_lon))
print("len=",len(lat_lon))
for ll in lat_lon:
    print("lat=",ll[1],"lon=",ll[2],"mag=",ll[3])
print('マグニチュード7以上の個数計算、インデックス取得------------------------')
i=0
j=0
indexes_m6=[]
indexes_m7=[]
for ll in lat_lon:
    if ll[3]>=7.0:
        i+=1
        indexes_m7.append(ll[0]-1)
    else:
        indexes_m6.append(ll[0]-1)
print(i)
print("index of M7 ",indexes_m7)
print("index of M6 ",indexes_m6)

print('=============================================================')
print('入力データ----------------------------------------------------')
lat_lon_input=[]
sdata = []
fcount=0
# for file in files:
#     fcount+=1
#     print('ファイル番号')
#     print(fcount)
#     f=open(file,'r')
#     print('f open')
#     while True:
#         data=f.readline()
#         print('data f readline')
#         if data == '':
#             break
#         if 'Memo.' in data:
#             while True:
#                 data2=f.readline()
                
#                 print('data2 f readline')
#                 if data2 == '':
#                     break
#                 data3 =data2.split()
            
#                 n = len(data3)
            
#                 cnt=0
#                 for i in range(n):
#                     val = float(data3[i])
#                     sdata.append(val)
#                     print('sdata append')
#                     cnt+=1
#                     if cnt > 1000:
#                         break
#             lat_lon_input.append(sdata)
#             print('append')
#             print(sdata)
#             sdata=[]
#             break
#     f.close()
#     print('f close')

for file in files:
    fcount+=1
    print('ファイル番号')
    print(fcount)
    f=open(file,'r')
    print('f open',file)
    while True:
        data=f.readline()
        # print('data f readline')
        if data == '':
            break
        if 'Memo.' in data:
            ndt = 0
            break_flag=0
            while True:
                data2 = f.readline()
                # print('data2 f readline')
                if data2 == '':
                    break
                data3 =data2.split()
                n = len(data3)
                for i in range(n):
                    val = float(data3[i])
                    sdata.append(val)
                    # print('sdata append')
                    ndt+=1
                    if ndt >= n_in:
                        # print('n_inを越しました。')
                        break_flag=1
                        break
                if break_flag==1:
                    # print('break flagが立っています')
                    break
            lat_lon_input.append(sdata)
            # print('append')
            # print(sdata)
            print('f close data num=',len(sdata))
            sdata=[]
            break
    f.close()

print('入力データの数----------------------------------------------------')
print(len(lat_lon_input[0]))


#ファイルごとに標準化
input_offset=[]
for lli in lat_lon_input:
    ave_input=np.average(lli,axis=0)
    std_input=np.std(lli,axis=0)
    lli=(lli-ave_input)/std_input
    print("ave=",np.average(lli,axis=0))
    print("std=",np.std(lli,axis=0)) 
    input_offset.append(lli)

print('標準化済み----------------------------------------------------')
# print(input_offset)
# print(len(input_offset[0]))

# print("len(1)=",len(input_offset[1]))
# print("len(2)=",len(input_offset[2]))



lat_lon_input=input_offset
lat_lon_input=np.array(lat_lon_input)
print(lat_lon_input)

print('フーリエ変換----------------------------------------------------')
# fft_data = np.abs(np.fft.rfft(lat_lon_input))
# lat_lon_input = fft_data
# print(fft_data)

# lat_lon_input=np.array(lat_lon_input)
#初めて1以上が現れるまで要素を削除し続ける。
#1以上が現れたらbreakする。
# lat_lon_input_more_than_1=[]
# for io in input_offset:
#     delete_range=0
#     for i in io:
#         if i<1.0:
#             delete_range+=1
#         else:
#             lat_lon_input_more_than_1.append(io[delete_range:])
#             break

# #要素数の配列を取得し、最大値を取得する。
# sum_array=[]
# for llimt1 in lat_lon_input_more_than_1:
#     sum=len(llimt1)
#     sum_array.append(sum)
# sum_min=min(sum_array)

# #要素数の最大値までをスライスする。
# lat_lon_input_completed=[]
# for llimt1 in lat_lon_input_more_than_1:
#     lat_lon_input_completed.append(llimt1[:sum_min])

# i=0
# print('len llic')
# for llic in lat_lon_input_completed:
#     print(len(llic))
#     i+=1
#     if i>10:
#         break

# #lat_lon_inputの上書き
# lat_lon_input=lat_lon_input_completed
# lat_lon_input=np.array(lat_lon_input)

# print('========================================================')
print('正解データ----------------------------------------------------')
lat_lon_correct=[]
j=0
for ll in lat_lon:
    if ll[3]<7.0:
        lat_lon_correct.append([1,0])
    else:
        lat_lon_correct.append([0,1])
    j+=1
lat_lon_correct=np.array(lat_lon_correct)
print("correct=",lat_lon_correct)
# print('正解データの数----------------------------------------------------')
print('正解データの数=',len(lat_lon_correct))


# print('===============================================================')
#配列の要素数を取得
print('震源数----------------------------------------------------')
ll_len=len(lat_lon)
# ll_len=100
print(ll_len)
select_num=ll_len
#------------------------------------------------------------------
n_data=ll_len
#要素数からランダムで~個の数を選ぶ
# use_file_index=random.sample(range(ll_len),select_num)
# print('選んだインデックス----------------------------------------------------')
# print(use_file_index)

#マグニチュードの配列を取得
mag=[]
for ll in lat_lon:
    mag.append(ll[3])
print("mag=",mag)
# print('==============================================================')
#訓練リストとテストリストに分ける
i=0
input_train=[]
input_test=[]
correct_train=[]
correct_test=[]

print("indexes_m7:",indexes_m7)
mag_test=[]
for im7 in indexes_m7:
    i+=1
    print("i=",i)
    #if i>=9:
    if i>=132*4/5:
        input_test.append(lat_lon_input[im7])
        correct_test.append(lat_lon_correct[im7])
        mag_test.append(mag[im7])
    else:
        input_train.append(lat_lon_input[im7])
        correct_train.append(lat_lon_correct[im7])
j=0
print("indexes_m6:",indexes_m6)
for im6 in indexes_m6:
    j+=1
    print("j=",j)    
    #if j>=9:
    if j>=132*4/5:    
        input_test.append(lat_lon_input[im6])
        correct_test.append(lat_lon_correct[im6])
        mag_test.append(mag[im6])
    else:
        input_train.append(lat_lon_input[im6])
        correct_train.append(lat_lon_correct[im6])
print("len input_train=",len(input_train))
print("len input_test=",len(input_test))
print("len correct_train=",len(correct_train))
print("len correct_test=",len(correct_test))

# print(input_train)
# print(input_test)
# print(correct_train)
# print(correct_test)
# exit()


# print('==============================================================')
# #波形プロット
i=0
for it in input_train:
    if correct_train[i][1]==1:
        title='input_train:M7'
    else:
        title='input_train:M6'
    plt.plot(it)
    plt.title(title)
    plt.legend()
    plt.show()
    i+=1

i=0
for it in input_test:
    if correct_test[i][1]==1:
        title='input_test:M7'
    else:
        title='input_test:M6'
    plt.plot(it)
    plt.title(title)
    plt.legend()
    plt.show()
    i+=1
print('plot end')
exit()
# print('===============================================================')
# input_train=lat_lon_input[:int(select_num//2)]
# input_test=lat_lon_input[int(select_num//2):]
# correct_train=lat_lon_correct[:int(select_num//2)]
# correct_test=lat_lon_correct[int(select_num//2):]

# print('訓練入力----------------------------------------------------')
# # print(input_train)
# print('テスト入力----------------------------------------------------')
# # print(input_test)
# print('訓練正解----------------------------------------------------')
# print(correct_train)
# print('テスト正解----------------------------------------------------')
# print(correct_test)

n_train=len(input_train)
n_test=len(input_test)
# print('訓練データ数----------------------------------------------------')
print('訓練データ数=',n_train)
# print('テストデータ数----------------------------------------------------')
print('テストデータ数=',n_test)

# n_in = 3000
# n_in=sum_min
# n_mid=300
# n_mid=100
n_mid=25
# n_mid=200

n_out=2

wb_width=0.1
eta=0.01
epoch=300
epoch=1000
epoch=400
batch_size=8
batch_size=10
batch_size=4
interval=100

class BaseLayer:
    def __init__(self,n_upper,n):
        print("baselayer init")
        self.w=wb_width*np.random.randn(n_upper,n)
        self.b=wb_width*np.random.randn(n)

        # self.h_w=np.zeros((n_upper,n))+1e-8
        # self.h_b=np.zeros(n)+1e-8

    def update(self,eta):
        print("update eta=",eta)
        w_max=np.abs(np.amax(self.w))
        b_max=np.abs(np.amax(self.b))
        print("w_max=",w_max,"b_max=",b_max)
        if w_max < 1.0e+20 and b_max < 1.0e+20:
            # print("update bef w:",self.w)
            # print("update self.grad_w:",self.grad_w)
            self.w-=eta*self.grad_w
            # print("update aft w:",self.w)
	    
            # print("update bef b:",self.b)	
            self.b-=eta*self.grad_b
            # print("update aft b:",self.b)
        else:
            r_val=w_max/1.0e20
            if r_val < b_max/1.0e20:
                r_val=b_max/1.0e20
            print("r_val=",r_val)	    
            print("w_max max(grad_w=",np.amax(self.grad_w))	    
            self.w-=eta*self.grad_w
            self.b-=eta*self.grad_b
        w_max=np.abs(np.amax(self.w))
        b_max=np.abs(np.amax(self.b))
        print("after update w_max=",w_max,"b_max=",b_max)
        if w_max > 1.0e+20 or b_max > 1.0e+20:
            r_val=w_max/1.0e20
            if r_val < b_max/1.0e20:
                r_val=b_max/1.0e20
            print("after update r_val=",r_val)	    


class MiddleLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        print('中間層入力')
        self.u=np.dot(x,self.w)+self.b
        w_max=np.abs(np.amax(self.u))
        print("middle forward w_max=",w_max)	
        if w_max > 1.0e20:
            r_val=w_max/1.0e20
            self.u=self.u/r_val	
        print("middle forward max(self.u)=",np.max(self.u))	
        # print("forward self.u",self.u)	
        # self.y=np.where(self.u<=0,0,self.u)
        self.y = 1 / (1 + np.exp(-self.u))
        print('中間層出力')
        # print(self.y)
        # print("middle forward success")

    def backward(self,grad_y):
        delta=grad_y*np.where(self.u<=0,0,self.u)
        print('中間層出力勾配')
        print("max(self.x)=",np.amax(self.x))
        print("max(delta)=",np.amax(delta))	
        self.grad_w=np.dot(self.x.T,delta)
        print("max(grad_w)=",np.amax(self.grad_w))	
        self.grad_b=np.sum(delta,axis=0)

        self.grad_x=np.dot(delta,self.w.T)
        print('中間層入力勾配')
        # print(self.grad_x)
        print("middle backward success")

class OutputLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        #print('出力層入力')
        print('output layer')
        print("b=",self.b)
        u=np.dot(x,self.w)+self.b
        print("u",u)
        #print("type of u:",type(u))
        u_max=np.amax(u)
        print("u_max=",u_max)
        print("u=",u)
        u_s=u-u_max	
        print("u_s 1 =",u_s)
        u_min=np.amin(u_s)
        if u_min < -30:	
            i1=0	
            for uu1 in u_s:
                i2=0	
                for uu2 in uu1:
                    if uu2 < -30:		
                        u_s[i1][i2]=-30-i1-i2
                    i2+=1
                i1+=1		    
        print("u_s 2 =",u_s)
        self.y=np.exp(u_s)/np.sum(np.exp(u_s),axis=1,keepdims=True)
        # self.y=np.where(u<=0,0,u)
        print('出力層出力')
        # print("out:",self.y)
        print("out forward success")

    def backward(self,t):
        print("output backward t=",t)
        print("output backward self.y=",self.y)
        delta=self.y-t
        print('正解誤差')
        # print("backward delta:",delta)

        print("output backward max self.x=",np.amax(self.x))
        print("output backward max delta=",np.amax(delta))
        self.grad_w=np.dot(self.x.T,delta)
        print("output backward max grad_w=",np.amax(self.grad_w))
        self.grad_b=np.sum(delta,axis=0)

        self.grad_x=np.dot(delta,self.w.T)
        print('出力層入力勾配')
        # print(self.grad_x)
        print("out backward success")

# define neural network
print("define neural network")
middle_layer1=MiddleLayer(n_in,n_mid)
middle_layer2=MiddleLayer(n_mid,n_mid)
middle_layer3=MiddleLayer(n_mid,n_mid)
output_layer=OutputLayer(n_mid,n_out)

# define forward process
print("define forward")
def forward_propagation(x):
    middle_layer1.forward(x)
    middle_layer2.forward(middle_layer1.y)
    middle_layer3.forward(middle_layer2.y)
    output_layer.forward(middle_layer3.y)

print("define back")
def backpropagation(t):
    print("backward output_layer")
    output_layer.backward(t)
    print("backward middle_layer2")
    middle_layer3.backward(output_layer.grad_x)
    print("backward middle_layer1")
    middle_layer2.backward(middle_layer3.grad_x)
    middle_layer1.backward(middle_layer2.grad_x)

def update_wb():
    print("update middle_layer1")
    middle_layer1.update(eta)
    print("update middle_layer2")
    middle_layer2.update(eta)
    middle_layer3.update(eta)
    print("update output")
    output_layer.update(eta)

def get_error(t,batch_size):
    print("get_error val=",-np.sum(t*np.log(output_layer.y+1e-7))/batch_size)
    print(np.log(output_layer.y+1e-7))
    print(t)
    print(len(t))
    return -np.sum(t*np.log(output_layer.y+1e-7))/batch_size

train_error_x=[]
train_error_y=[]
test_error_x=[]
test_error_y=[]

print("start of learning")
print("n_train=",n_train)
print("batch_size=",batch_size)
n_batch=n_train//batch_size
print("n_batch=",n_batch)
for i in range(epoch):
    print('エポック：'+str(i))
    print('誤差計算-----------------------------------------')
    forward_propagation(input_train)
    error_train=get_error(correct_train,n_train)
    forward_propagation(input_test)
    error_test=get_error(correct_test,n_test)

    train_error_x.append(i)
    train_error_y.append(error_train)
    test_error_x.append(i)
    test_error_y.append(error_test)

    if i%interval==0:
        print("Epoch:"+str(i+1)+"/"+str(epoch),
            "Error_train:"+str(error_train),
            "Error_test:"+str(error_test))

    index_random=np.arange(n_train)
    np.random.shuffle(index_random)
    print('学習-----------------------------------------------')
    for j in range(n_batch):
        print(str(j+1)+'バッチ目')
        mb_index=index_random[j*batch_size:(j+1)*batch_size]
        x=[]
        t=[]
        for m in mb_index:
            x.append(input_train[m])
            t.append(correct_train[m])
        x=np.array(x)
        t=np.array(t)
        # print('x--------------------------------------------')
        # print(x)
        forward_propagation(x)
        backpropagation(t)
        update_wb()

plt.plot(train_error_x,train_error_y,label="Train")
plt.plot(test_error_x,test_error_y,label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

forward_propagation(input_train)
count_train = np.sum(np.argmax(output_layer.y ,axis = 1) == np.argmax(correct_train ,axis = 1))
forward_propagation(input_test)
count_test = np.sum(np.argmax(output_layer.y ,axis = 1) == np.argmax(correct_test ,axis = 1))

print("Accuracy Train:"+str(count_train/n_train*100)+"%",
    "Accuracy Test:"+str(count_test/n_test*100)+"%")


forward_propagation(input_test)
print('結果------------------------------------------------------')
print(output_layer.y)
print('マグニチュード------------------------------------------------------')
print(mag_test)
print('テスト正解------------------------------------------------------')
print(correct_test)
