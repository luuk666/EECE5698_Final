
import numpy as np
import matplotlib.pyplot as plt

# 加载滤波后的深度图（也可以换成原始的）
depth = np.load("output/example-midas_v21_small_256.npy")

# 用户标定相关信息
calibration_point = []
scale_factor = None

def onclick(event):
    global calibration_point, scale_factor

    x, y = int(event.xdata), int(event.ydata)
    clicked_depth = depth[y, x]

    if not calibration_point:
        print(f"[标定] 请点击已知物体位置：({x}, {y}) → 相对深度 = {clicked_depth:.4f}")
        real_distance = float(input("请输入该物体的真实距离（单位：cm）: "))
        scale_factor = real_distance / clicked_depth
        calibration_point = [x, y]
        print(f"→ 已建立尺度：scale = {scale_factor:.4f} cm/unit
")
    else:
        real_estimated = clicked_depth * scale_factor
        print(f"[测距] 点击点 ({x}, {y}) → 相对深度 = {clicked_depth:.4f} → 估算真实距离 = {real_estimated:.2f} cm")

# 显示深度图
plt.imshow(depth, cmap='inferno')
plt.title("点击图像：先选标定点，再选任意测距点")
plt.colorbar()
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
