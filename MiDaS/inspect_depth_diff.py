
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.fftpack import fft2, fftshift, ifftshift, ifft2

# 👉 加载原始深度图
raw = np.load("output/example-midas_v21_small_256.npy")

# 👉 滤波函数：中值 + 傅里叶低通
def apply_median_and_fourier_filter(depth, median_size=5, cutoff=0.1):
    medianed = median_filter(depth, size=median_size)

    f = fft2(medianed)
    fshift = fftshift(f)
    rows, cols = depth.shape
    crow, ccol = rows // 2, cols // 2
    r = int(cutoff * min(rows, cols))
    mask = np.zeros_like(depth)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

    f_filtered = fshift * mask
    f_ishift = ifftshift(f_filtered)
    filtered = np.abs(ifft2(f_ishift))
    return filtered

# 👉 得到滤波后的深度图
filtered = apply_median_and_fourier_filter(raw)

# 👀 显示原图、滤波图和差值图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(raw, cmap='inferno')
axs[0].set_title("Raw Depth")

axs[1].imshow(filtered, cmap='inferno')
axs[1].set_title("Filtered Depth")

diff = raw - filtered
axs[2].imshow(diff, cmap='bwr', vmin=-10, vmax=10)
axs[2].set_title("Difference (Raw - Filtered)")

for ax in axs:
    ax.axis("off")

# 🔍 点击时显示深度值和差值
def onclick(event):
    if event.inaxes not in axs:
        return
    x, y = int(event.xdata), int(event.ydata)
    d_raw = raw[y, x]
    d_filtered = filtered[y, x]
    print(f"(x={x}, y={y}) → Raw: {d_raw:.4f}, Filtered: {d_filtered:.4f}, Δ={d_raw - d_filtered:.4f}")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()
