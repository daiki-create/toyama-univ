# 目的：2つ以上のファイルが同時にopenできるかの確認

import glob

# ===========================================================================================================================================
# 使用データファイルの取得
files = glob.glob("../datasets/mag/datasets_ud/datasets_ud/*")


# ===========================================================================================================================================
i = 0
f = []
for file in files:
    f.append(file)
    i += 1
    if i == 2:
        break
f1 = open(f[0],'r')
f2 = open(f[1],'r')
print(f1)
print(f2)
print('pythonは同時に2つ以上のファイルが開けます')
