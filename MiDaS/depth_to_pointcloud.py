import numpy as np
import open3d as o3d
import cv2

# === 路径设置 ===
depth_path = "output/example-midas_v21_small_256.npy"   # MiDaS 输出的深度图（npy）
img_path = "example.jpg"                                # 原图（用于着色）

# === 加载图像和深度图 ===
depth = np.load(depth_path)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (depth.shape[1], depth.shape[0]))

# === 步骤1：归一化深度并放大 Z 值 ===
depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
depth *= 100  # 放大Z以增强立体感

# === 步骤2：仅保留中间区域点云（去除背景干扰） ===
h, w = depth.shape
points = []
colors = []

for v in range(h // 4, h * 3 // 4):
    for u in range(w // 4, w * 3 // 4):
        z = depth[v, u]
        if z == 0 or np.isnan(z):
            continue
        x = (u - w / 2) * z
        y = (v - h / 2) * z
        points.append([x, -y, -z])  # 注意：反转Y/Z轴让结构“朝向观察者”
        colors.append(img[v, u] / 255.0)

# === 步骤3：构造点云对象 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
pcd.estimate_normals()

# === 步骤4：可视化点云 ===
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Bonian 3D MiDaS PointCloud Viewer')
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.point_size = 2.0
opt.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
vis.run()
vis.destroy_window()

# === 可选：保存为 .ply 文件（可用于 MeshLab/Blender 打开） ===
o3d.io.write_point_cloud("output/final_pointcloud.ply", pcd)
print("✅ 点云已保存为：output/final_pointcloud.ply")
