import os
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm

def check_facescape_data_strict(root_dir, save_path="error_log_strict.npy"):
    """
    更严谨的 Facescape 数据检查脚本。
    主要模拟第一次“光心对齐裁剪”，这是数据加载中最容易崩溃的步骤。
    """
    error_logs = []

    if not osp.exists(root_dir):
        print(f"Error: Root directory {root_dir} does not exist.")
        return

    scenes = sorted([d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))])
    print(f"Found {len(scenes)} scenes. Starting strict check...")

    for scene in tqdm(scenes):
        scene_path = osp.join(root_dir, scene)
        all_files = os.listdir(scene_path)
        ids = set([f.split('.')[0] for f in all_files if f.endswith('.npz')])

        for file_id in ids:
            jpg_path = osp.join(scene_path, f"{file_id}.jpg")
            exr_path = osp.join(scene_path, f"{file_id}.exr")
            npz_path = osp.join(scene_path, f"{file_id}.npz")

            # --- 1. 基础文件检查 ---
            paths_to_check = [jpg_path, exr_path, npz_path]
            is_empty = False
            for p in paths_to_check:
                if not osp.exists(p) or osp.getsize(p) == 0:
                    is_empty = True
                    break

            if is_empty:
                error_logs.append([scene, file_id, "File Missing/Empty", None, None, None, None, None, None])
                continue

            # --- 2. 模拟裁剪逻辑 (Crop 1) ---
            try:
                # 加载参数
                camera_params = np.load(npz_path, allow_pickle=True)
                intrinsics = np.float32(camera_params['intrinsics'])

                # 加载图片尺寸
                with Image.open(jpg_path) as img:
                    W, H = img.size

                # 计算光心
                cx, cy = intrinsics[:2, 2].round().astype(int)

                # --- 核心检查逻辑 ---
                # 代码原文: min_margin_x = min(cx, W - cx)
                min_margin_x = min(cx, W - cx)
                min_margin_y = min(cy, H - cy)

                # 计算边界
                l, t = cx - min_margin_x, cy - min_margin_y
                r, b = cx + min_margin_x, cy + min_margin_y

                # 错误类型 A: 边界逻辑错误 (导致 PIL 抛出 ValueError)
                # 这意味着光心 cx 在图像外部，或者刚好在边缘导致计算异常
                if r < l:
                    raise ValueError(f"Invalid X Crop: Right({r}) < Left({l}). CX={cx}, W={W}")
                if b < t:
                    raise ValueError(f"Invalid Y Crop: Bottom({b}) < Top({t}). CY={cy}, H={H}")

                # 错误类型 B: 裁剪区域过小或为零 (导致后续 Resize 失败)
                # 如果光心刚好在边缘，margin可能为0，导致裁剪出 0x0 的图片
                crop_w = r - l
                crop_h = b - t
                if crop_w <= 0 or crop_h <= 0:
                    raise ValueError(f"Zero Size Crop: {crop_w}x{crop_h}. Image will be empty.")

                # (可选) 错误类型 C: 裁剪区域极小，虽然不报错但数据无意义
                # 例如裁剪出来只有 10x10 像素
                if crop_w < 10 or crop_h < 10:
                    raise ValueError(f"Tiny Crop Warning: {crop_w}x{crop_h}. Likely bad intrinsics.")

                # 额外的越界检查
                if l < 0 or t < 0 or r > W or b > H:
                    raise ValueError(f"Crop Out of Bounds: [{l},{t},{r},{b}] in Img [{W}x{H}]")

            except Exception as e:
                # 捕获异常并记录
                # 尝试获取当前变量值，如果未定义则为 None
                curr_cx = locals().get('cx', None)
                curr_cy = locals().get('cy', None)
                curr_l = locals().get('l', None)
                curr_r = locals().get('r', None)
                curr_t = locals().get('t', None)
                curr_b = locals().get('b', None)

                # 统一转为 int 方便存储
                def safe_int(val): return int(val) if val is not None else None

                # 记录格式: [场景, id, 错误信息, cx, cy, l, r, t, b]
                error_msg = str(e)
                error_logs.append([
                    scene, file_id, error_msg,
                    safe_int(curr_cx), safe_int(curr_cy),
                    safe_int(curr_l), safe_int(curr_r), safe_int(curr_t), safe_int(curr_b)
                ])

    # 保存结果
    error_logs = np.array(error_logs, dtype=object)
    np.save(save_path, error_logs)

    print("-" * 30)
    print(f"Strict check completed.")
    print(f"Total problematic files: {len(error_logs)}")
    if len(error_logs) > 0:
        print("First 3 errors:")
        for err in error_logs[:3]:
            print(f"Scene: {err[0]}, ID: {err[1]}, Error: {err[2]}")

if __name__ == "__main__":
    ROOT_DIR = "/root/autodl-tmp/facescape_processed"
    check_facescape_data_strict(ROOT_DIR)