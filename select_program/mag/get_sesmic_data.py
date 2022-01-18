import os
import shutil
import glob

#東北地方の緯度経度を定義(仮)
#観測点が遠すぎるとゴミみたいなグラフになる
# min_st_lat=0
# max_st_lat=1
# min_st_lon=0
# max_st_lon=1

#基本は震源だけでいい
#36,37からでいい、日本海溝沿いから
min_ev_lat=37
max_ev_lat=41
min_ev_lon=138
max_ev_lon=145

#datasetsから拡張子UDのみを選択してdatasets_udに格納
# new_path = "datasets_ud"
# new_path2="datasets_ud_lat_tohoku"
# new_path3="datasets_ud_tohoku"
# if not os.path.exists(new_path):
    # os.mkdir(new_path)
# if not os.path.exists(new_path2):
#     os.mkdir(new_path2)
# if not os.path.exists(new_path3):
#     os.mkdir(new_path3)

# files=glob.glob("datasets/*.UD")
# for file in files:
#     shutil.move(file,'datasets_ud')

#一つ一つファイルを開いて2,3,7,8行目の値を配列sesmic_infoに格納
# DIR = '/datasets_ud'
# ud_sum=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
# print(ud_sum)
# for i in range(1,ud_sum+1):
#     open()
# files=glob.glob("datasets_ud/*")
# for file in files:
#     f=open(file,'r')
#     while True:
#         data=f.readline()
#         # if 'Station Lat.' in data:
#         #     st_lat = float(data[19-1:30])
#         #     print(st_lat)
#         # if 'Station Long.' in data:
#         #     st_lon = float(data[19-1:30])
#         #     print(st_lon)
#         if 'Lat.' in data:
#             ev_lat = float(data[19-1:30])
#             print(ev_lat)
#             if(ev_lat >= min_ev_lat and ev_lat<=max_ev_lat):
#                 print('東北かも')
#                 f.close() 
#                 shutil.move(file,'datasets_ud_lat_tohoku')
#                 break
#             else:
#                 f.close() 
#                 print('東北じゃない')
#                 break           

files=glob.glob("../../datasets/mag/datasets_ud/*")
for file in files:
    f=open(file,'r')
    while True:
        data=f.readline()
        if data == "":
            break
        if 'Lat.' in data:
            ev_lat = float(data[19-1:30])
        elif 'Long.' in data:
            ev_lon = float(data[19-1:30])
    f.close()
    if(min_ev_lon <= ev_lon <= max_ev_lon and min_ev_lat <= ev_lat <= max_ev_lat):
        print('東北です!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        shutil.move(file,'../../datasets/mag/datasets_ud_tohoku')
    else:
        print('東北じゃない')
        
print('ok')
