# 目的：read lineの挙動を確認

import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import time

files_tide_pass = glob.glob("../datasets/tsunami/3/*")

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
                print(line_t2)