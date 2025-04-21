import numpy as np
import matplotlib.pyplot as plt

# 读取深度图（.npy）
depth = np.load("output/example-midas_v21_small_256.npy")

# 回调函数：鼠标点击显示深度值
def onclick(event):
    x, y = int(event.xdata), int(event.ydata)
    value = depth[y, x]
    print(f"Clicked at (x={x}, y={y}) → depth={value:.4f}")

# 显示热力图
plt.imshow(depth, cmap="inferno")
plt.title("Click to show depth value")
plt.colorbar()
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
