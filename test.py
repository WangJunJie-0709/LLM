import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def cfc60_filter(data, sample_rate):
    """
    应用CFC60滤波器到数据上

    参数:
    data (numpy array): 输入信号数据
    sample_rate (int): 数据的采样率 (Hz)

    返回:
    numpy array: 滤波后的信号
    """
    # CFC60滤波器的截止频率为1000 Hz
    cutoff_freq = 1000.0

    # 计算奈奎斯特频率
    nyquist_freq = 0.5 * sample_rate

    # 计算归一化截止频率
    normal_cutoff = cutoff_freq / nyquist_freq

    # 设计一个2阶巴特沃斯低通滤波器
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)

    # 应用滤波器到数据上
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


# 示例用法
if __name__ == "__main__":
    # 生成样本信号
    sample_rate = 10000  # 采样率 (Hz)
    t = np.linspace(0, 1.0, sample_rate)
    signal_data = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    signal_data_noise = signal_data.copy()
    # 添加噪声
    noise = 0.5 * np.random.normal(size=t.shape)
    signal_data_noise += noise

    # 应用CFC60滤波器
    filtered_signal = cfc60_filter(signal_data_noise, sample_rate)

    # 绘制原始和滤波后的信号
    plt.figure()
    plt.plot(t, signal_data, label='signal')
    # plt.plot(t, signal_data_noise, label='noised signal', linewidth=2)
    plt.plot(t, filtered_signal, label='processed signal', linewidth=2)
    plt.legend()
    plt.xlabel('time(s)')
    plt.ylabel('width')
    plt.title('CFC60')
    plt.show()
