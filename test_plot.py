"""
纯测试用画图脚本，不依赖项目代码。
用于快速验证绘图环境和常用库。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 全局中文字体设置（根据本机可用字体自动匹配）
plt.rcParams["font.sans-serif"] = ["SimHei", "PingFang SC", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def peak_then_decay(x, base_hour, peak=1.0, sigma=2.0):
    """
    baseHour 之前 y 递增，baseHour 时达峰，之后 y 递减并逐渐趋于 0。
    使用高斯型曲线实现。
    """
    return peak * np.exp(-((x - base_hour) ** 2) / (2 * sigma**2))


def score(H, S, alpha=1.3, beta=0.9, gamma=0.01):
    """
    原始公式（保持不变的结构）：

    Score = log(1+H) * H^alpha / (gamma + H^alpha + S^beta)
            + sqrt(S) * S^beta / (gamma + H^alpha + S^beta)

    其中 H, S ∈ [0, 1]。
    这里仅通过调整 alpha, beta, gamma 的数值，
    让「低 H + 高 S 的长尾 item」得分相对更高，同时保证 Score ∈ (0, 1)。
    """
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)

    denom = gamma + H**alpha + S**beta
    term_h = np.log1p(H) * (H**alpha) / denom
    term_s = np.sqrt(S) * (S**beta) / denom
    return term_h + term_s


def main():
    # ---------- 参数：热度主导，约 7:3 偏好（H:S） ----------
    alpha = 1.3   # 略微强化高 H 的区分度，但不过分陡峭
    beta = 0.9    # 让 S 有贡献，但相对 H 略弱，符合 7:3 直觉
    gamma = 0.01   # 避免分母为0

    # 在 [0, 1] × [0, 1] 上取网格点
    h = np.linspace(0.0, 1.0, 120)
    s = np.linspace(0.0, 1.0, 120)
    H, S = np.meshgrid(h, s)
    Z = score(H, S, alpha=alpha, beta=beta, gamma=gamma)

    # 3D 曲面图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(H, S, Z, cmap="viridis", edgecolor="none", alpha=0.9)

    ax.set_xlabel("H")
    ax.set_ylabel("S")
    ax.set_zlabel("Score")
    ax.set_title("Score(H, S) 的 3D 曲面图")
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label="Score")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
