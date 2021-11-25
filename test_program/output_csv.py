# 目的：作成したcsvファイルを行ごとにスプリットして出力する。

import glob

# ===========================================================================================================================================
# 使用データファイルの取得
files = glob.glob("../datasets/tsunami/3/merged.dat_b4.1998")


# ===========================================================================================================================================
for file in files:
    f = open(file,'r')
    while True:
        row = f.readline()
        if row == '':
            break
        if 'lat,lon,st_lat,st_lon,1-3s,3-9s,9-27s,27-81s,tsunami_h(m),date' in row:
            while True:
                row2 = f.readline()
                if row2 == '':
                    break
                data = row2.split(',')
                print(data)
                exit()
