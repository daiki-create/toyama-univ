import numpy as np
import matplotlib.pyplot as plt
import glob
import random

#データ処理
#拡張子がUDのファイルの震源地、マグニチュードを配列に格納
# files=glob.glob("datasets_test/*")
# files=glob.glob("datasets_test2/*")
files=glob.glob("datasets_use/*")

print('============================================================================================================')
print('[インデックス、緯度、経度、マグニチュード]----------------------------------------------------')
lat_lon=[]
i=0
for file in files:
    f=open(file,'r')
    while True:
        data=f.readline()
        if 'Lat.' in data:
            ev_lat = float(data[19-1:30])
            print(ev_lat)
            lat_lon.append([i+1,ev_lat,'tmp','tmp'])
        elif 'Long.' in data:
            ev_lon = float(data[19-1:30])
            print(ev_lon)
            lat_lon[i][2]=ev_lon
        elif 'Mag.' in data:
            mag=float(data[19-1:30])
            f.close() 
            print(mag)
            lat_lon[i][3]=mag
            i+=1
            break
print('マグニチュード7以上の個数計算、インデックス取得----------------------------------------------------')
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
print(indexes_m7)
print(indexes_m6)


print('============================================================================================================')
print('入力データ----------------------------------------------------')
lat_lon_input=[]
sdata = []
fcount=0
for file in files:
    fcount+=1
    print('ファイル番号')
    print(fcount)
    f=open(file,'r')
    print('f open')
    while True:
        data=f.readline()
        print('data f readline')
        if data == '':
            break
        if 'Memo.' in data:
            while True:
                data2=f.readline()
                print('data2 f readline')
                if data2 == '':
                    break
                data3 =data2.split()
                n = len(data3)
                for i in range(n):
                    val = float(data3[i])
                    sdata.append(val)
                    print('sdata append')
            lat_lon_input.append(sdata)
            print('append')
            print(sdata)
            sdata=[]
            break
    f.close()
    print('f close')
#ファイルごとに標準化
input_offset=[]
lat_lon_input=np.array(lat_lon_input)
for i in lat_lon_input:
    ave_input=np.average(i,axis=0)
    std_input=np.std(i,axis=0)
    i=(i-ave_input)/std_input
    input_offset.append(i)
lat_lon_input=input_offset
print(lat_lon_input)
print('入力データの数----------------------------------------------------')
print(len(lat_lon_input))

#初めて1以上が現れるまで要素を削除し続ける。
#1以上が現れたらbreakする。
lat_lon_input_more_than_1=[]
for lli in lat_lon_input:
    delete_range=0
    for l in lli:
        if l<1.0:
            delete_range+=1
        else:
            lat_lon_input_more_than_1.append(lli[delete_range:])
            break
print('last index delete_range')
print(delete_range)

#要素数の配列を取得し、最大値を取得する。
sum_array=[]
for llimt1 in lat_lon_input_more_than_1:
    sum=len(llimt1)
    sum_array.append(sum)
sum_min=min(sum_array)
print('sum_array')
print(sum_array)
print('sum_min')
print(sum_min)

#要素数の最大値までをスライスする。
lat_lon_input_completed=[]
for llimt1 in lat_lon_input_more_than_1:
    lat_lon_input_completed.append(llimt1[:sum_min])

i=0
print('len llic')
for llic in lat_lon_input_completed:
    print(len(llic))

#lat_lon_inputの上書き
lat_lon_input=lat_lon_input_completed

print('============================================================================================================')
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
print(lat_lon_correct)
print('正解データの数----------------------------------------------------')
print(len(lat_lon_correct))


print('============================================================================================================')
#配列の要素数を取得
print('震源数----------------------------------------------------')
ll_len=len(lat_lon)
# ll_len=100
print(ll_len)
select_num=ll_len
#---------------------------------------------------------------------------------------------------------------------------------------------------------
n_data=ll_len
#要素数からランダムで~個の数を選ぶ
# use_file_index=random.sample(range(ll_len),select_num)
# print('選んだインデックス----------------------------------------------------')
# print(use_file_index)

#マグニチュードの配列を取得
mag=[]
for ll in lat_lon:
    mag.append(ll[3])

print('============================================================================================================')
#訓練リストとテストリストに分ける
i=0
input_train=[]
input_test=[]
correct_train=[]
correct_test=[]

mag_test=[]
for im7 in indexes_m7:
    i+=1
    if i>=9:
        input_test.append(lat_lon_input[im7])
        correct_test.append(lat_lon_correct[im7])
        mag_test.append(mag[im7])
    else:
        input_train.append(lat_lon_input[im7])
        correct_train.append(lat_lon_correct[im7])
j=0
for im6 in indexes_m6:
    j+=1
    if j>=9:
        input_test.append(lat_lon_input[im6])
        correct_test.append(lat_lon_correct[im6])
        mag_test.append(mag[im6])
    else:
        input_train.append(lat_lon_input[im6])
        correct_train.append(lat_lon_correct[im6])
print(input_train)
print(input_test)
print(correct_train)
print(correct_test)


print('============================================================================================================')
#波形プロット
i=0
for it in input_train:
    if correct_train[i][1]==1:
        title='input_train：M7'
    else:
        title='input_train：M6'
    plt.plot(it)
    plt.title(title)
    plt.legend()
    plt.show()
    i+=1

i=0
for it in input_test:
    if correct_test[i][1]==1:
        title='input_test M7'
    else:
        title='input_test M6'
    plt.plot(it)
    plt.title(title)
    plt.legend()
    plt.show()
    i+=1