import numpy as np
import matplotlib.pyplot as plt 
# import parse
from datetime import datetime as dt
n_sz = 10000
sdata = [] 
#sdata = [0] * n_sz
ndt = 0

f = open("../../datasets/mag/datasets_test/AKT0010108140511.UD","r")

# datalist = f.readlines()
# print (datalist[0],end='')
# print (datalist[1],end='')
# print (datalist[2],end='')
while True:
  data = f.readline()
  if data == '':
    break
  #lower(data)
  #print (data,end='')
  if 'Origin Time' in data:
    print(data[19-1:38-1])
    origin_time = dt.strptime(data[19-1:38-1], '%Y/%m/%d %H:%M:%S')
    print(origin_time )
    # 2016/11/22 05:59:00
  elif 'Station Lat.' in data:
    st_lat = float(data[19-1:30])
    print('st_lat=',ev_lat)
  elif 'Station Long.' in data:
    st_lon = float(data[19-1:30])
    print('st_lon',ev_lon)
  elif 'Lat.' in data:
    ev_lat = float(data[19-1:30])
    print('ev_lat=',ev_lat)
  elif 'Long.' in data:
    ev_lon = float(data[19-1:30])
    print('ev_lon=',ev_lon)
  elif 'Depth' in data:
    ev_dep = float(data[19-1:30])
    print('ev_dep=',ev_dep)
  elif 'Mag.' in data:
    ev_mag = float(data[19-1:30])
    print('ev_mag=',ev_mag)
  #elif 'station code' in data      YMT016
  elif 'Station Height(m)' in data:
    st_hei = float(data[19-1:30])
    print('st_hei=',st_hei)
  elif 'Record Time' in data:
    record_time = dt.strptime(data[19-1:38-1], '%Y/%m/%d %H:%M:%S')
    print('record_time=', record_time)
  #elif 'sampling freq(hz)' in data' 100Hz
  elif 'Duration Time(s)' in data:
    duration = float(data[19-1:30])
    print('duration=',duration)
  #elif 'dir.' in data '              U-D
  #elif 'scale factor' in data     7845(gal)/8223790
  #elif 'max. acc. (gal)' in data '   1.815
  #elif 'last correction' in data    2016/11/22 06:00:34
  elif 'Memo.' in data:
    ndt = 0
    while True:
      data2 = f.readline()
      if data2 == '':
        break
      data3 =data2.split() 
      #print(data3)
      n = len(data3)
      #print('n=',n)
      for i in range(n):
        val = float(data3[i])
        #print('i=',i,' ',val)
        #print('ndt=',ndt)
        #print('len=',len(sdata))
        #sdata[ndt] = val
        sdata.append(val)
        ndt+=1
        if ndt >= n_sz:
          break
    print(st_lat)
    print(ev_lat)
    exit()
    break

f.close()
print('ndt=',ndt)
plt.plot(sdata)
plt.show()
