import glob
import random
import shutil

#本番用
files=glob.glob("datasets_ud_tohoku/*")

#テスト用
# select_num=10
# files=glob.glob("test/*")

#空の2次元配列を作成
lat_lon=[]
i=0
j=0

#震源情報だけ抽出（ダブりがないように)
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

ll_len=len(lat_lon)

print('ダブりありの震源数----------------------------------------------------')
print(ll_len)

#二次元配列の重複を削除
for i in range(ll_len):
    for j in range (ll_len):
        if lat_lon[i-1][1]==lat_lon[j-1][1] and lat_lon[i-1][2]==lat_lon[j-1][2] and i<j:
            lat_lon[i-1][0]='delete'
print('deletフラグ----------------------------------------------------')
print(lat_lon)

print('ダブりなし----------------------------------------------------')
tmp = []
while lat_lon:
    e = lat_lon.pop()
    if e[0]  != 'delete':
        tmp.append(e)

while tmp:
    lat_lon.append(tmp.pop())

print(lat_lon)

#乱数で使用する震源を選びだす（配列）
#配列の要素数を取得
ll_len=len(lat_lon)
print('震源数----------------------------------------------------')
print(ll_len)
select_num=ll_len

#要素数からランダムで~個の数を選ぶ
use_file_index=random.sample(range(ll_len),select_num)
print('選んだインデックス----------------------------------------------------')
print(use_file_index)

#使用するファイルを呼び出してリスト出力
use_file=[]
for u in use_file_index:
    index=lat_lon[u][0]
    use_file.append(files[index])

#研究に使用するデータをフォルダに格納
for file in use_file:
    shutil.move(file,'datasets_use2')
    break

print('使用ファイル----------------------------------------------------')
print(use_file)

#学習ファイルリストとテストファイルリストに分ける
train_file=use_file[:int(select_num//2)]
test_file=use_file[int(select_num//2):]

print('学習ファイル----------------------------------------------------')
print(train_file)
print('テストファイル----------------------------------------------------')
print(test_file)




print('マグニチュード7以上----------------------------------------------------')
cnt_m7=0
for u in use_file_index:
    mag=lat_lon[u][3]
    if mag>7.0:
        cnt_m7+=1
print(cnt_m7)
