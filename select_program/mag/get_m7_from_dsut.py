import glob
import random
import shutil
import numpy as np

files=glob.glob("../../datasets/mag/datasets_ud_tohoku/*")

lat_lon=[]
st_lat_lon=[]
mags_7=[]
mags_6=[]

i=0
n=10

xh=[]
yh=[]
zh=[]
xs=[]
ys=[]
zs=[]

cos_delta=[]
dis=[]
max_dis=300

for file in files:
    f=open(file,'r')
    while True:
        data=f.readline()
        if 'Lat.' in data:
            ev_lat = float(data[19-1:30])            
            lat_lon.append([i+1,ev_lat,'tmp','tmp'])
        elif 'Long.' in data:
            ev_lon = float(data[19-1:30])
            lat_lon[i][2]=ev_lon
        elif 'Mag.' in data:
            mag=float(data[19-1:30])
            lat_lon[i][3]=mag

            i+=1
            f.close()
            print("f close")
            break
    # if i>=n:
    #     break

i=0
for file in files:
    f=open(file,'r')
    while True:
        data=f.readline()
        if 'Mag.' in data:
            mag=float(data[19-1:30])
            st_lat_lon.append([i+1,'tmp','tmp',mag])
        elif 'Station Lat.' in data:
            st_lat = float(data[19-1:30])
            st_lat_lon[i][1]=st_lat
        elif 'Station Long.' in data:
            st_lon = float(data[19-1:30])
            st_lat_lon[i][2]=st_lon

            i+=1
            f.close()
            print("f close")
            break
    # if i>=n:
    #     break

# print(lat_lon)
# print(st_lat_lon)

# 震源
for l in range(len(lat_lon)):
    # イギリス方向の赤道面の軸
    xh.append(np.cos(lat_lon[l][1])*np.cos(lat_lon[l][2]))
    # 東経90度の赤道面の軸
    yh.append(np.cos(lat_lon[l][1])*np.sin(lat_lon[l][2]))
    # 北極向きの軸
    zh.append(np.sin(lat_lon[l][1]))
# print(xh)
# print(yh)
# print(zh)

# 観測点
for l in range(len(lat_lon)):
    # イギリス方向の赤道面の軸
    xs.append(np.cos(st_lat_lon[l][1])*np.cos(st_lat_lon[l][2]))
    # 東経90度の赤道面の軸
    ys.append(np.cos(st_lat_lon[l][1])*np.sin(st_lat_lon[l][2]))
    # 北極向きの軸
    zs.append(np.sin(st_lat_lon[l][1]))
# print(xs)
# print(ys)
# print(zs)

# 観測点と震源の中心角delta
for l in range(len(lat_lon)):
    cos_delta=(xh[l]*xs[l]+yh[l]*ys[l]+zh[l]*zs[l])

    if (6367*cos_delta<=max_dis and 6367*cos_delta>=0) or (6367*cos_delta>=max_dis and 6367*cos_delta<0):
        # 震央距離dis
        dis.append(6367*cos_delta)
    # 震央距離300km以上のときlat_lonとst_lat_lonからデータを削除
    else:
        lat_lon[l]="del"
        st_lat_lon[l]="del"

print("震央距離")
# print(dis)

print(len(dis))

for ll in lat_lon[:]:
    if ll=="del":
        lat_lon.remove(ll)

for sll in st_lat_lon[:]:
    if sll=="del":
        st_lat_lon.remove(sll)

print(len(lat_lon))
print(len(st_lat_lon))

for ll in lat_lon:
    if ll[3]>=7.0:
        mags_7.append(ll)

i=0
for ll in lat_lon:
    if ll[3]<7.0:
        mags_6.append(ll)
        i+=1
        if i>=len(mags_7):
            break

i=0
for file in files:
    for m7 in mags_7:
        if m7[0]==i+1:
            shutil.copy(file,'../../datasets/mag/datasets_use2')
            break
    for m6 in mags_6:
        if m6[0]==i+1:
            shutil.copy(file,'../../datasets/mag/datasets_use2')
            break
    i+=1

print(len(mags_7))
