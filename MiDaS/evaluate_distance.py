
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 格式: (名称, 实际距离cm, 滤波后深度值)
measurements = [
    ("Vitamin", 30.0, 289.5337),
    ("Theanine", 50.0, 561.2368),
    ("Magtein", 70.0, 785.0587)
]


# 用第一个点进行标定（维生素）
scale_point = measurements[0]
scale_factor = scale_point[1] / scale_point[2]

# 输出换算比例
print(f"🔧 标定点: {scale_point[0]} — scale = {scale_factor:.6f} cm/unit\n")

# 初始化
labels, gt, pred = [], [], []

# 计算每个点的估算距离
for name, real_dist, depth in measurements:
    est = depth * scale_factor
    print(f"{name}: GT = {real_dist:.2f} cm | 估算 = {est:.2f} cm")
    if name != scale_point[0]:  # 排除标定点
        labels.append(name)
        gt.append(real_dist)
        pred.append(est)

# 误差评估
mae = mean_absolute_error(gt, pred)
rmse = mean_squared_error(gt, pred) ** 0.5
print(f"\n📊 MAE: {mae:.2f} cm | RMSE: {rmse:.2f} cm")

# 可视化
x = np.arange(len(labels))
plt.figure(figsize=(7, 4))
plt.bar(x - 0.2, gt, width=0.4, label="Ground Truth")
plt.bar(x + 0.2, pred, width=0.4, label="Estimated")
plt.xticks(x, labels)
plt.ylabel("Distance (cm)")
plt.title("Ground Truth vs Estimated Distance")
plt.legend(["Ground Truth", "Estimated"])

plt.tight_layout()
plt.savefig("output/distance_error.png")
plt.show()
