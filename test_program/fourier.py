import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 正弦波のデータ作成
    f = 1000
    rate = 44100
    T = np.arange(0, 0.01, 1 / rate)
    s = []
    for t in T:
        v = np.sin(2 * np.pi * f * t)
        s.append(v)

    plt.plot(T, s)
    plt.xlabel('Time')
    plt.ylabel('Gain')
    plt.show()

    # フーリエ変換
    fft_data = np.abs(np.fft.rfft(s))
    freqList = np.fft.rfftfreq(len(s), 1.0 / rate)  # 横軸
    plt.loglog(freqList, 10 * np.log(fft_data))
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.show()

    # 逆フーリエ変換
    r = np.fft.irfft(fft_data, len(T))
    plt.plot(T, r)
    plt.xlabel('Time')
    plt.ylabel('Gain')
    plt.show()