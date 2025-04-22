
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 👇 修改为你自己的数据
# 格式：[("描述", 实际距离cm, 相对深度值)]
measurements = [
    ("A4纸中心", 29.7, 0.312),
    ("瓶子",     45.0, 0.471),
    ("显示器",   60.0, 0.607),
    ("书本",     35.0, 0.382),
    ("键盘",     50.0, 0.538)
]

# === 标定 ===
# 比如你用的是 A4纸，它的实际距离是 29.7 cm，对应相对深度是 0.312
scale_point = measurements[0]
scale_factor = scale_point[1] / scale_point[2]

# === 计算 ===
labels = []
gt = []
pred = []

for label, real_dist, depth_val in measurements:
    if label != scale_point[0]:  # 跳过标定点
        est_dist = depth_val * scale_factor
        labels.append(label)
        gt.append(real_dist)
        pred.append(est_dist)
        print(f"{label} → GT: {real_dist:.2f} cm | 估算: {est_dist:.2f} cm")

mae = mean_absolute_error(gt, pred)
rmse = mean_squared_error(gt, pred, squared=False)
print(f"\n📊 MAE: {mae:.2f} cm | RMSE: {rmse:.2f} cm")

# === 可视化 ===
x = np.arange(len(labels))
plt.figure(figsize=(8,5))
plt.bar(x - 0.2, gt, width=0.4, label="Ground Truth")
plt.bar(x + 0.2, pred, width=0.4, label="Estimated")
plt.xticks(x, labels, rotation=20)
plt.ylabel("Distance (cm)")
plt.title("Real vs Estimated Distance")
plt.legend()
plt.tight_layout()
plt.savefig("output/distance_error.png")
plt.show()
