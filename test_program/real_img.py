# 目的：虚数演算の練習

import cmath

a = 1 + 2j
b = 2 + 3j

# 実部、虚部
print(a.real)
print(a.imag)

# 四則演算
print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 絶対値
print(abs(a))

# 極座標との相互変換
print(cmath.polar(a))
print(cmath.rect(1, cmath.pi/2))