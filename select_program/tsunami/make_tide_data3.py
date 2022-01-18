# 目的：2016年福島の潮汐データファイルにバンドパスフィルタをかけた地震波の振幅最大値を収めること
# 　　　　サンプリングレート、スケーリングファクターも取得する
# 潮汐データ:2016/11/22 5:59 37.35,141.60のみ
# k-netデータ:2016/11/22 5:59のみ

import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import time
import shutil
# import warnings

# warnings.filterwarnings('ignore') 

files_tide_pass = glob.glob("../../datasets/tsunami/3/merged.dat_b4.2016")
files_ud_pass = glob.glob("../../datasets/mag/1611220559/*")

# 震源も観測点も一致するk-netデータを格納するリスト
match_1611220559 = []

for ftp in files_tide_pass:
    ft = open(ftp,"r")
    while True:
        line_t = ft.readline()
        if line_t == "":
            break
        if "lat,lon,st_lat,st_lon,1-3s,3-9s,9-27s,tsunami_h(cm),date" in line_t:
            while True:
                line_t2 = ft.readline()
                if line_t2 == "":
                    break
                data_t = line_t2.split(",")
                # リストの中から年～日がdateと一致するファイル名のリストを変数に格納
                files_ud_match_date = []
                for fup in files_ud_pass:
                    if fup[36:46] == data_t[8][0:10]:
                        files_ud_match_date.append(fup)                      
                # 日時の一致するUDファイルがなければnullを挿入
                data_t[4] = "null"
                data_t[5] = "null"
                data_t[6] = "null"
                if len(files_ud_match_date) == 0:
                    data_t_insert = str(data_t[0]) + "," +str(data_t[1]) + "," + str(data_t[2]) + "," + str(data_t[3]) + "," + str(data_t[4]) + "," + str(data_t[5]) + "," + str(data_t[6]) + "," + str(data_t[7]) + "," + str(data_t[8])
                    with open("../../datasets/tsunami/4/merged.dat_b4.2016", mode="a", encoding="utf-8") as f:
                        f.write(data_t_insert)
                # 日時の一致するUDファイルがあれば・・
                else:
                    for fumd in files_ud_match_date:
                        fu = open(fumd,"r")
                        while True:
                            line_u = fu.readline()
                            if line_u == "":
                                break
                            if 'Station Lat.' in line_u:
                                st_lat = float(line_u[19-1:30][0:4])
                            elif 'Station Long.' in line_u:
                                st_lon = float(line_u[19-1:30][0:5])
                            elif 'Lat.' in line_u:
                                lat = float(line_u[19-1:30][0:4])
                            elif 'Long.' in line_u:
                                lon = float(line_u[19-1:30][0:5])
                        fu.close()
                        # 震源と観測点が一致ならば・・
                        if float(data_t[0][0:4])-0.4 <= lat <= float(data_t[0][0:4])+0.4 and float(data_t[1][0:5])-0.4 <= lon <= float(data_t[1][0:5])+0.4:
                            if (float(data_t[2][0:4])-0.2 <= st_lat <= float(data_t[2][0:4])+0.2) and (float(data_t[3][0:5])-0.2 <= st_lon <= float(data_t[3][0:5])+0.2):
                                print("\n観測点も一致")
                                print("k-net:" + str(st_lat) + "," + str(st_lon) + "\n潮汐:" + str(data_t[2][0:4]) + "," + str(data_t[3][0:5]))
                                fu = open(fumd,"r")
                                sesmic_a = []
                                while True:
                                    line_u2 = fu.readline()
                                    if line_u2 == "":
                                        break
                                    if "Sampling Freq(Hz)" in line_u2:
                                        srate_hz = line_u2[19-1:30]
                                        srate = float(srate_hz.replace('Hz',''))
                                    if "Scale Factor" in line_u2:
                                        scale_factor_str = line_u2[19-1:40]
                                        scale_factor_list = scale_factor_str.split("(gal)/")
                                        sf_denominator = float(scale_factor_list[1])
                                        sf_numerator = float(scale_factor_list[0])
                                        scale_factor =  sf_numerator / sf_denominator
                                    if "Memo." in line_u2:
                                        while True:
                                            line_u_memo = fu.readline()
                                            if line_u_memo == "":
                                                break
                                            sesmic_a_list = line_u_memo.split()
                                            for sal in sesmic_a_list:
                                                sesmic_a.append(float(sal) * scale_factor)
                                        # オフセット
                                        ave_sa=np.average(sesmic_a)
                                        sesmic_a = sesmic_a - ave_sa
                                        # フーリエ変換                                    
                                        # rfftは複素共役を省略
                                        # fftは省略なし
                                        fft_data = np.fft.rfft(sesmic_a)
                                        freqList = np.fft.fftfreq(len(fft_data), 1.0 / srate)
                                        # 積分して変位にする
                                        index = np.where(freqList > 0)
                                        for i in index:
                                            omega0 = 2.0 * np.pi * freqList[i] * (0 + 1j)
                                            fft_data[i] = fft_data[i] / omega0 / omega0
                                        f_cut_low = [1/3, 1/9, 1/27]
                                        f_cut_high = [1/1, 1/3, 1/9]
                                        for i in range(3):
                                            fft_data_cppy = np.fft.rfft(sesmic_a)
                                            freqList = np.fft.fftfreq(len(fft_data_cppy), 1.0 / srate)
                                            index = np.where(freqList > 0)
                                            for j in index:
                                                omega0 = 2.0 * np.pi * freqList[j] * (0 + 1j)
                                                fft_data_cppy[j] = fft_data_cppy[j] / omega0 / omega0
                                            # plt.plot(np.real(fft_data))
                                            # plt.show()
                                            index2 = np.where(freqList < f_cut_low[i])
                                            for j in index2:
                                                fft_data_cppy[j] = 0
                                            index3 = np.where(freqList > f_cut_high[i])
                                            for j in index3:
                                                fft_data_cppy[j] = 0
                                            irfft_data = np.fft.irfft(fft_data_cppy)
                                            max_fft_data = max(irfft_data)
                                            data_t[i+4] = max_fft_data
                                            # plt.plot(np.real(irfft_data))
                                            # plt.show()
                                print("\nmatch lat and lon!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                match_1611220559.append(fumd)
                                break
                            else:
                                print("\nしかし観測点は不一致")
                                print("k-net:" + str(st_lat) + "," + str(st_lon) + "\n潮汐:" + str(data_t[2][0:4]) + "," + str(data_t[3][0:5]))
                        else:
                            print("\n観測点すら不一致")
                            print("k-net:" + str(lat) + "," + str(lon) + "\n潮汐:" + str(data_t[0][0:4]) + "," + str(data_t[1][0:5]))

                    # 行データを形成
                    data_t_insert = str(data_t[0]) + "," +str(data_t[1]) + "," + str(data_t[2]) + "," + str(data_t[3]) + "," + str(data_t[4]) + "," + str(data_t[5]) + "," + str(data_t[6]) + "," + str(data_t[7]) + "," + str(data_t[8])
                    if data_t[4] != "null":
                        with open("../../datasets/tsunami/4/merged.dat_b4.2016", mode="a", encoding="utf-8") as f:
                            f.write(data_t_insert)

print(match_1611220559)
set(match_1611220559)
for m1 in match_1611220559:
    shutil.move(m1,"../../datasets/mag/match_1611220559_3")
