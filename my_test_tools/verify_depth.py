import numpy as np
import json
import trimesh
import os
import argparse

def back_project(depth_map, K, Rt):
    """
    将深度图反投影为 3D 点云 (World Coordinates)
    """
    H, W = depth_map.shape

    # 1. 创建像素坐标网格
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # 2. 筛选有效深度 (非0值)
    valid_mask = depth_map > 0
    z = depth_map[valid_mask]
    u = i[valid_mask]
    v = j[valid_mask]

    # 3. 反投影到相机坐标系 (OpenCV Convention: X right, Y down, Z forward)
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # P_cam = [x, y, z]
    P_cam = np.stack([x, y, z], axis=-1) # (N, 3)

    # 4. 转回世界坐标系
    # P_cam = R * P_world + t
    # => P_world = R_inv * (P_cam - t)
    # 对于旋转矩阵 R，R_inv = R.T

    R = Rt[:3, :3]
    t = Rt[:3, 3]

    P_world = (P_cam - t) @ R # 注意矩阵乘法顺序，或者用 R.T @ (P_cam - t).T
    # 验证一下形状：(N, 3) @ (3, 3) -> (N, 3) 正确

    return P_world

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 请修改为你实际的路径
    root_dir = "/root/dust3r/my_test_tools/facescape_test"
    target_idx = 0  # 你想验证的图片索引 (例如第0张)
    # ===========================================

    ply_path = os.path.join(root_dir, "2_smile.ply")
    params_path = os.path.join(root_dir, "params.json")
    npy_path = os.path.join(root_dir, "2_smile_depth", f"{target_idx}.npy")

    if not os.path.exists(npy_path):
        print(f"错误: 找不到深度图文件 {npy_path}，请先运行渲染脚本。")
        exit()

    print(f"正在验证索引 {target_idx} ...")

    # 加载参数
    with open(params_path, 'r') as f:
        params = json.load(f)

    K = np.array(params[f"{target_idx}_K"])
    Rt = np.array(params[f"{target_idx}_Rt"])

    # 加载深度图
    depth_map = np.load(npy_path)
    print(f"深度图加载成功，形状: {depth_map.shape}, 有效点数: {np.count_nonzero(depth_map)}")

    # 反投影
    print("正在反投影回 3D 空间...")
    points = back_project(depth_map, K, Rt)

    # 保存为 PLY 用于对比
    output_ply = "reconstructed_check.ply"
    pcd = trimesh.points.PointCloud(points)
    pcd.export(output_ply)

    print(f"\n验证文件已生成: {output_ply}")
    print("-" * 30)
    print("【如何判断是否完美？】")
    print(f"1. 请下载 '{output_ply}' 和原始的 '1_neutral.ply'")
    print("2. 在 MeshLab 中打开这两个文件")
    print("3. 它们应该**严丝合缝**地重叠在一起。")
    print("4. 如果重叠完美，说明你的深度图就是最高精度的。")