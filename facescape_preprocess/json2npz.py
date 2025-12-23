import json
import numpy as np
import os

def process_and_convert_params(target_dir, json_path):
    # 1. 读取原始 JSON 文件
    print(f"正在读取参数文件: {json_path}")
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. 提取所有索引 (例如 "0", "1", "35" 等)
    indices = set()
    for key in data.keys():
        if "_" in key:
            idx = key.split('_')[0]
            indices.add(idx)

    print(f"JSON 中共包含 {len(indices)} 组相机参数。")
    print(f"正在过滤并处理位于以下路径的图片: {target_dir}")
    print(f"匹配规则: [ID].jpg (例如 0.jpg)")

    count = 0
    skipped_not_found = 0

    # 3. 遍历每一组参数
    for idx in indices:
        # --- 修改点：不再读取 _ori，而是直接用 ID 构建文件名 ---
        image_name = f"{idx}.jpg"

        # --- 路径匹配 ---
        # 检查这张图片 (例如 0.jpg) 是否在用户指定的目标文件夹里
        image_full_path = os.path.join(target_dir, image_name)
        if not os.path.exists(image_full_path):
            # 如果找不到对应 ID 的图片，跳过
            skipped_not_found += 1
            continue

        # --- 步骤 1 & 3: 提取参数并进行坐标系转换 ---

        # A. 获取内参 K (3x3)
        if f"{idx}_K" not in data:
            print(f"警告: 索引 {idx} 缺少 K 参数，跳过。")
            continue

        K_list = data[f"{idx}_K"]
        intrinsics = np.array(K_list, dtype=np.float32)

        # B. 获取 Rt (World-to-Camera) (3x4)
        if f"{idx}_Rt" not in data:
            print(f"警告: 索引 {idx} 缺少 Rt 参数，跳过。")
            continue

        Rt_list = data[f"{idx}_Rt"]
        w2c_matrix = np.eye(4, dtype=np.float64) # 构建 4x4 矩阵
        w2c_matrix[:3, :] = np.array(Rt_list, dtype=np.float64)

        # C. 坐标系逆转: w2c -> c2w
        # DUST3R 需要的是 Camera-to-World，即 P_world = R * P_cam + t
        try:
            c2w_matrix = np.linalg.inv(w2c_matrix)
        except np.linalg.LinAlgError:
            print(f"警告: 索引 {idx} ({image_name}) 的矩阵无法求逆，跳过。")
            continue

        # D. 拆解 c2w 矩阵
        # 旋转矩阵 R (保留 float64 以保证旋转精度)
        R_cam2world = c2w_matrix[:3, :3]
        # 平移向量 t (相机中心在世界坐标系的位置)
        t_cam2world = c2w_matrix[:3, 3].astype(np.float32)

        # --- 步骤 2: 保存为单独的 npz 文件 ---

        # 生成 npz 文件名：与图片 ID 对应
        # 例如: 0.jpg -> 0.npz
        save_name = f"{idx}.npz"
        save_path = os.path.join(target_dir, save_name)

        np.savez(save_path,
                 intrinsics=intrinsics,
                 R_cam2world=R_cam2world,
                 t_cam2world=t_cam2world)

        count += 1

    print("-" * 30)
    print(f"处理完成！")
    print(f"目标文件夹: {target_dir}")
    print(f"成功生成: {count} 个 .npz 文件")
    if skipped_not_found > 0:
        print(f"跳过(未找到对应图片): {skipped_not_found} 个 (这是正常的，因为脚本只处理当前文件夹下的图片)")

# --- 配置部分 ---
if __name__ == "__main__":
    # 指定图片所在的文件夹路径
    target_img_dir = "/root/autodl-tmp/images_undistort/001-020/1/1_neutral"

    # 指定 params.json 的路径
    # 请确保此路径指向你的 params.json 文件
    json_file_path = "/root/autodl-tmp/images_undistort/001-020/1/1_neutral/params.json"
    # 如果 json 不在那个目录下，请修改上面的路径

    # 执行处理
    process_and_convert_params(target_img_dir, json_file_path)