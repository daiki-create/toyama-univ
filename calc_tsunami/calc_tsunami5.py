# 目的：周波数帯ごとにプロット

import numpy as np
import matplotlib.pyplot as plt
import glob

# ===========================================================================================================================================
# 使用データファイルの取得
files = glob.glob("../datasets/tsunami/tsunami/2/*")


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
        if 'seisic_lat,seismic_lon,seismic_dis(1-3s,3-9s,9-27s,27-81s,cm),tsunami_lat,tsunami_lon,tsunami_h(m),distance(km)' in data:
            while True:
                data2 = f.readline()
                if data2 == '':
                    break
                data3 = data2.split()

                seismic_lat = data3[0].split(',')[0]
                seismic_lon = data3[0].split(',')[1]
                seismic_dis1 = float(data3[1].split(',')[0])
                seismic_dis2 = float(data3[2].split(',')[0])
                seismic_dis3 = float(data3[3].split(',')[0])
                seismic_dis4 = float(data3[4].split(',')[0])
                tsunami_lat = data3[5].split(',')[0]
                tsunami_lon = data3[5].split(',')[1]
                tsunami_h = float(data3[6].split(',')[0])
                distance = data3[7]
                new_data3 = [seismic_lat ,seismic_lon ,seismic_dis1 ,seismic_dis2 ,seismic_dis3 ,seismic_dis4 ,tsunami_lat ,tsunami_lon ,tsunami_h ,distance]

                tsunami_data.append(new_data3)
# ---------------------------------------------------------------------------------------------------------------------------------
# 津波の高さ(正解データ)をリスト化
y = []
for t in tsunami_data:
    y.append([t[8]])

# ===========================================================================================================================================
# 入力データ
x1 = []
x2 = []
x3 = []
x4 = []
for t in tsunami_data:
    x1.append(t[2])
    x2.append(t[3])
    x3.append(t[4])
    x4.append(t[5])

# ===========================================================================================================================================

plt.scatter(x1,y,c='red')
plt.scatter(x2,y,c='blue')
plt.scatter(x3,y,c='green')
plt.scatter(x4,y,c='yellow')

plt.ylabel("correct")
plt.xlabel("input")
plt.grid(True)
plt.show()