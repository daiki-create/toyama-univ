# 目的：1998年の潮汐データファイルにバンドパスフィルタをかけた地震波の振幅最大値を収めること

import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import time
# import warnings

# warnings.filterwarnings('ignore') 

files_tide_pass = glob.glob("../../datasets/tsunami/3/*")
files_ud_pass = glob.glob("../../datasets/mag/datasets_ud/*")

for ftp in files_tide_pass:
    ft = open(ftp,"r")
    while True:
        line_t = ft.readline()
        if line_t == "":
            break
        if "lat,lon,st_lat,st_lon,1-3s,3-9s,9-27s,27-81s,tsunami_h(m),date" in line_t:
            while True:
                line_t2 = ft.readline()
                if line_t2 == "":
                    break
                data_t = line_t2.split(",")
                # リストの中から年～日がdateと一致するファイル名のリストを変数に格納
                files_ud_match_date = []
                for fup in files_ud_pass:
                    if fup[37:43] == data_t[9][0:6]:
                        files_ud_match_date.append(fup)                                      
                # 日時の一致するUDファイルがなければnullを挿入
                data_t[4] = "null"
                data_t[5] = "null"
                data_t[6] = "null"
                data_t[7] = "null"
                if len(files_ud_match_date) == 0:
                    data_t_insert = str(data_t[0]) + "," +str(data_t[1]) + "," + str(data_t[2]) + "," + str(data_t[3]) + "," + str(data_t[4]) + "," + str(data_t[5]) + "," + str(data_t[6]) + "," + str(data_t[7]) + "," + str(data_t[8]) + "," + str(data_t[9])
                    with open("../../datasets/tsunami/4/merged.dat_b4.1998", mode="a", encoding="utf-8") as f:
                        f.write("\n"+data_t_insert)
                # 日時の一致するUDファイルがあれば・・
                else:
                    for fumd in files_ud_match_date:
                        fu = open(fumd,"r")
                        while True:
                            line_u = fu.readline()
                            if line_u == "":
                                break
                            if "Lat." in line_u:
                                lat = float(line_u[19-1:30][0:2])
                            if "Long." in line_u:
                                lon = float(line_u[19-1:30][0:2])
                            if "Station Lat." in line_u:
                                st_lat = float(line_u[19-1:30][0:2])
                            if "Station Long." in line_u:
                                st_lon = float(line_u[19-1:30][0:2])
                        fu.close()
                        # 震源と観測点が一致
                        if (int(data_t[0][0:2])-2 <= lat <= int(data_t[0][0:2])+2) and (int(data_t[1][0:2])-2 <= lon <= int(data_t[1][0:2])+2) and (int(data_t[2][0:2])-2 <= st_lat <= int(data_t[2][0:2])+2) and (int(data_t[3][0:2])-2 <= st_lon <= int(data_t[3][0:2])+2):
                            fu = open(fumd,"r")
                            sesmic_a = []
                            while True:
                                line_u2 = fu.readline()
                                if line_u2 == "":
                                    break
                                if "Memo." in line_u2:
                                    while True:
                                        line_u_memo = fu.readline()
                                        if line_u_memo == "":
                                            break
                                        sesmic_a_list = line_u_memo.split()
                                        for sal in sesmic_a_list:
                                            sesmic_a.append(float(sal))
                                    # オフセット
                                    ave_sa=np.average(sesmic_a)
                                    sesmic_a = sesmic_a - ave_sa
                                    # フーリエ変換
                                    srate = 100
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
                            print("k-net:" + str(lat) + ",潮汐:" + str(data_t[0][0:2]))
                            break
                        else:
                            print("\nnot match")
                            print("k-net:" + str(lat) + ",潮汐:" + str(data_t[0][0:2]))
                    data_t_insert = str(data_t[0]) + "," +str(data_t[1]) + "," + str(data_t[2]) + "," + str(data_t[3]) + "," + str(data_t[4]) + "," + str(data_t[5]) + "," + str(data_t[6]) + "," + str(data_t[7]) + "," + str(data_t[8]) + "," + str(data_t[9])
                    with open("../../datasets/tsunami/4/merged.dat_b4.1998", mode="a", encoding="utf-8") as f:
                        f.write(data_t_insert)
