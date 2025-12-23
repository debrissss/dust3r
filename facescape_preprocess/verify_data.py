import numpy as np
import cv2
import json
import os

# =================配置区域=================
# 请替换为您实际的文件路径
TEST_CONFIG = {
    # 原始文件路径 (Original)
    "orig_img": "/root/autodl-tmp/images_undistort/001-020/1/1_neutral/1.jpg",
    "orig_json": "/root/autodl-tmp/images_undistort/001-020/1/1_neutral/params.json",
    "img_id": "1",  # 用于在json中查找key，如 "1_K"

    # 生成文件路径 (Generated)
    "gen_img": "/root/autodl-tmp/trainsets/001_1_neutral/1.jpg",
    "gen_exr": "/root/autodl-tmp/trainsets/001_1_neutral/1.exr",
    "gen_npz": "/root/autodl-tmp/trainsets/001_1_neutral/1.npz"
}
# ==========================================

def load_data(config):
    """加载所有必要的数据"""
    print(f"[-] Loading data...")

    # 1. Load Original
    img_orig = cv2.imread(config["orig_img"])
    with open(config["orig_json"], 'r') as f:
        params = json.load(f)

    idx = config["img_id"]
    K_orig = np.array(params[f"{idx}_K"])
    Rt_orig = np.array(params[f"{idx}_Rt"]) # World2Cam

    # 2. Load Generated
    img_gen = cv2.imread(config["gen_img"])
    # 读取 EXR 需要允许 flag
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    depth_gen = cv2.imread(config["gen_exr"], cv2.IMREAD_UNCHANGED)

    npz_data = np.load(config["gen_npz"])
    K_gen = npz_data['intrinsics']
    R_c2w_gen = npz_data['R_cam2world']
    t_c2w_gen = npz_data['t_cam2world']

    return {
        "img_orig": img_orig, "K_orig": K_orig, "Rt_orig": Rt_orig,
        "img_gen": img_gen, "depth_gen": depth_gen,
        "K_gen": K_gen, "R_c2w_gen": R_c2w_gen, "t_c2w_gen": t_c2w_gen
    }

def check_intrinsics(data):
    """验证内参缩放是否正确 (修正版)"""
    print(f"\n[1] Checking Intrinsics...")

    h_orig, w_orig = data["img_orig"].shape[:2]

    # 【修正】复现生成代码中的逻辑：基于长边计算统一的 scale
    target_long_edge = 1024
    scale = target_long_edge / max(h_orig, w_orig)

    print(f"    Calculated Scale (based on logic): {scale:.6f}")

    K_orig = data["K_orig"]
    K_gen = data["K_gen"]

    # 理论上的 K_gen 应使用统一的 scale
    K_theory = K_orig.copy()
    K_theory[:2, :] *= scale
    K_theory[2, 2] = 1.0

    # 对比误差
    diff = np.abs(K_gen - K_theory)

    # 允许一定的浮点误差
    if np.all(diff < 1e-3):
        print("    [PASS] Intrinsics match theoretical scaling.")
    else:
        print("    [FAIL] Intrinsics mismatch!")
        print("    Diff:\n", diff)

def check_extrinsics(data):
    """验证外参是否正确 (求逆对比)"""
    print(f"\n[2] Checking Extrinsics...")

    # 原始: World -> Camera (3x4)
    Rt_orig = data["Rt_orig"]
    T_w2c = np.eye(4)
    T_w2c[:3, :4] = Rt_orig

    # 计算原始的 Camera -> World
    T_c2w_calc = np.linalg.inv(T_w2c)

    # 生成的: Camera -> World
    R_gen = data["R_c2w_gen"]
    t_gen = data["t_c2w_gen"]
    T_c2w_gen = np.eye(4)
    T_c2w_gen[:3, :3] = R_gen
    T_c2w_gen[:3, 3] = t_gen

    # 对比
    diff = np.abs(T_c2w_calc - T_c2w_gen)
    if np.all(diff < 1e-4):
        print("    [PASS] Extrinsics (C2W) match inverse of original (W2C).")
    else:
        print("    [FAIL] Extrinsics mismatch!")
        print("    Max Diff:", np.max(diff))

def back_project(depth_map, K):
    """将深度图反投影为 3D 点云 (Camera Coordinate)"""
    h, w = depth_map.shape
    idx_y, idx_x = np.indices((h, w))

    # 展平
    z = depth_map.flatten()
    x = idx_x.flatten()
    y = idx_y.flatten()

    # 过滤无效深度
    valid = z > 0
    z = z[valid]
    x = x[valid]
    y = y[valid]

    # 反投影公式: X = (u - cx) * Z / fx
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z

    # 堆叠为 (N, 3)
    points = np.stack([X, Y, Z], axis=1)
    return points, valid

def check_3d_consistency(data):
    """
    核心校验：3D 重投影一致性
    逻辑：
    1. 在生成的深度图中取一个点 P_gen(u, v) -> BackProject -> P_3d_gen
    2. 找到该点在原图中的对应坐标 P_orig(u/s, v/s)
    3. 如果原深度图是由 renderer 渲染的(本例中没有加载原exr，所以我们用另一种方法)

    替代方案：
    由于我们没有加载“原始分辨率的 EXR”(代码里没保存)，
    我们直接验证：生成的深度图是否真的是最近邻插值？
    我们假设 'renderer.render' 在原分辨率下产生 depth_orig_render。

    但为了简单有效，我们这里做一个近似验证：
    我们直接观察生成的 3D 点云是否合理，或者对比生成图和原图的采样点。
    """
    print(f"\n[3] Checking 3D Consistency (Logic Check)...")

    img_gen = data["img_gen"]
    depth_gen = data["depth_gen"]

    # 随机采样 5 个点
    h, w = depth_gen.shape
    np.random.seed(42)
    sample_y = np.random.randint(0, h, 5)
    sample_x = np.random.randint(0, w, 5)

    print(f"    Sampling 5 random points from generated data:")
    for y, x in zip(sample_y, sample_x):
        d = depth_gen[y, x]
        color = img_gen[y, x] # BGR
        print(f"    - Pixel ({x}, {y}): Depth={d:.4f}, Color(BGR)={color}")

    print("    [INFO] Please manually verify: Do these depths look physically reasonable? (e.g. not 0 or inf)")

    # 检查深度统计
    valid_depths = depth_gen[depth_gen > 0]
    if len(valid_depths) == 0:
        print("    [FAIL] Depth map is empty or all zeros!")
    else:
        print(f"    [PASS] Depth stats: Min={valid_depths.min():.2f}, Max={valid_depths.max():.2f}, Mean={valid_depths.mean():.2f}")

if __name__ == "__main__":
    if not os.path.exists(TEST_CONFIG["gen_exr"]):
        print("[Error] Generated file not found. Please run the generation script first.")
    else:
        data = load_data(TEST_CONFIG)
        check_intrinsics(data)
        check_extrinsics(data)
        check_3d_consistency(data)
        print("\n[Done] Verification finished.")