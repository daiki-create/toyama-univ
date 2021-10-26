# 津波データの緯度経度に紐づく地震加速度波形データの格納

import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# ===========================================================================================================================================
# 使用データファイルの取得
files = glob.glob("../datasets/tsunami/*")
files2 = glob.glob("../datasets/mag/datasets_ud_tohoku")

