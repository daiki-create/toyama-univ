# ⓶
#     0.UDファイル名リストを取得[files_ud]
#     1.潮汐ファイルをopen[f_tide]
#     2.行を変数に格納(while True)[line]
#     3.スプリットでlat,lon,st_lat,st_lon,dateを変数に格納[lat,lon,st_lat,st_lon,data]
#     //4.潮汐ファイルをclose
#     6.リストの中から7~16文字目（年～分の10の位）がdateと一致するファイル名のリストを変数に格納[files_ud_match_date]
#     8.リストのUDファイルをopen（whie True）[f_ud]
#     8.lat,lon,st_lat,st_lonを変数(接頭文字k-)に格納（それぞれ小数第二位までの浮動小数を取得）[k-lat,k-lon,k-st_lat,k-st_lon,k-data]
#     　→ 潮汐データ側の小数第二位が繰り上がっている可能性があるので、小数第一位に変更する可能性あり
#     9a.各緯度経度が一致している場合、memo下のリスト（地震波）を変数に格納[sesmic_a]
#         標準化[sesmic_a_offset]
#         フーリエ変換[sesmic_a_fourier]
#         2階積分して変位にする[sesmic_x_fourier]
#         1.周波数1-3s,3-9s,9-27s,27-81sでカットしてをかけてそれぞれのリストを変数に格納[sesmic_x_fourier_1_3,・・]
#         それぞれ時間領域に逆変換[sesmic_x_1_3,・・]
#         2.それぞれのリストの最大値を変数に格納[sesmic_x_max_1_3,・・]
#         3.UDファイルをclose
#         //4.先ほどcloseした潮汐ファイルをopen
#         5.5~8列目に9a-2で取得した変数を記入
#         //6.breakで2に戻る
#     9b.各緯度経度が一致していない場合、UDファイルをclose
#         //してbreakで8に戻る

# 目的：潮汐データファイルにバンドパスフィルタをかけた地震波の振幅最大値を収めること

import numpy as np
import matplotlib.pyplot as plt
import glob
import random

files_tide = glob.glob("../../datasets/tsunami/3/merged.dat_b4.1998")
files_ud = glob.glob("../../datasets/mag/datasets_ud/*")
print(files_ud)
print(len(files_ud))
print(files_ud[0])
print(files_ud[0][38:46])
