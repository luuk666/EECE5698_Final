import os
import numpy as np
import cv2
import open3d as o3d

# === 路径设置 ===
image_folder = "images"
depth_folder = "output"
output_file = "output/combined_pointcloud.ply"

z_scale = 100  # Z轴放大系数，增强立体感
all_pcd = []

# === 相机视角设置（平移 + Y轴旋转）===
translations = [-0.3, 0.0, 0.3]  # 假设三张图的相机横向位移
rotations_y_deg = [15, 0, -15]   # 向左、正前、向右拍摄
rotations_y = [np.radians(a) for a in rotations_y_deg]

# === 三张图分别处理 ===
for idx, name in enumerate(["1", "2", "3"]):
    print(f"Processing {name}.jpg")

    # --- 读取图像 & 深度图 ---
    img_path = os.path.join(image_folder, f"{name}.jpg")
    depth_path = os.path.join(depth_folder, f"{name}.npy")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)

    # --- 归一化深度 + 放大Z ---
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth *= z_scale
    img = cv2.resize(img, (depth.shape[1], depth.shape[0]))

    # --- 构造点云数据 ---
    h, w = depth.shape
    cx, cy = w / 2, h / 2
    points, colors = [], []

    for v in range(h // 4, h * 3 // 4):
        for u in range(w // 4, w * 3 // 4):
            z = depth[v, u]
            if z == 0 or np.isnan(z): continue
            x = (u - cx) * z
            y = (v - cy) * z
            points.append([x, -y, -z])
            colors.append(img[v, u] / 255.0)

    # --- 创建 Open3D 点云对象 ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # --- 视角变换（旋转 + 平移模拟相机运动） ---
    T = np.eye(4)
    T[0, 3] = translations[idx] * z_scale  # X方向平移
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, rotations_y[idx], 0))
    T[:3, :3] = R
    pcd.transform(T)

    all_pcd.append(pcd)

# === 合并所有点云 & 显示 ===
combined = all_pcd[0]
for p in all_pcd[1:]:
    combined += p

# 可视化
o3d.visualization.draw_geometries([combined])

# 可选保存
o3d.io.write_point_cloud(output_file, combined)
print(f"✅ 点云已保存为：{output_file}")
