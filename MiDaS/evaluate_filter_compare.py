
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 格式: (名称, 实际距离cm, 滤波前深度值, 滤波后深度值)
measurements = [
    ("Vitamin", 30.0, 289.4317, 286.5337),
    ("Theanine", 50.0, 561.4529, 549.2368),
    ("Magtein", 70.0, 784.2333, 778.0587)
]

# 用滤波后值的第一个点（Vitamin）作为标定
ref_real = measurements[0][1]
ref_depth = measurements[0][3]
scale = ref_real / ref_depth
print(f"🔧 标定点: {measurements[0][0]} — scale = {scale:.6f} cm/unit\n")

labels, gt, raw_pred, filt_pred = [], [], [], []

# 计算估算距离
for name, real, raw_d, filt_d in measurements:
    raw_est = raw_d * scale
    filt_est = filt_d * scale
    labels.append(name)
    gt.append(real)
    raw_pred.append(raw_est)
    filt_pred.append(filt_est)
    print(f"{name}: GT = {real:.2f} cm | Raw = {raw_est:.2f} cm | Filtered = {filt_est:.2f} cm")

# MAE / RMSE 评估（排除标定点）
gt_eval = gt[1:]
raw_eval = raw_pred[1:]
filt_eval = filt_pred[1:]

mae_raw = mean_absolute_error(gt_eval, raw_eval)
rmse_raw = mean_squared_error(gt_eval, raw_eval) ** 0.5
mae_filt = mean_absolute_error(gt_eval, filt_eval)
rmse_filt = mean_squared_error(gt_eval, filt_eval) ** 0.5

print(f"\n📊 Raw  MAE: {mae_raw:.2f} cm | RMSE: {rmse_raw:.2f} cm")
print(f"📊 Filtered MAE: {mae_filt:.2f} cm | RMSE: {rmse_filt:.2f} cm")

# 绘图
x = np.arange(len(labels))
plt.figure(figsize=(9, 5))
plt.bar(x - 0.25, gt, width=0.25, label="Ground Truth")
plt.bar(x, raw_pred, width=0.25, label="Raw Estimate")
plt.bar(x + 0.25, filt_pred, width=0.25, label="Filtered Estimate")
plt.xticks(x, labels)
plt.ylabel("Distance (cm)")
plt.title("Ground Truth vs Raw vs Filtered Depth Estimate")
plt.legend()
plt.tight_layout()
plt.savefig("output/distance_filter_compare.png")
plt.show()
