import json
import cv2
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- 配置 ---
SOURCE_ROOT = Path("/root/autodl-tmp/images")
TARGET_ROOT = Path("/root/autodl-tmp/images_undistort")
MAX_WORKERS = 25  # 根据你的 CPU 核心数调整，通常设为 CPU 核心数

def get_ids(json_data):

    unique_ids = set()
    for key in json_data.keys():
        if "_" in key:
            prefix = key.split('_')[0]
            if prefix.isdigit():  # 确保是数字ID
                unique_ids.add(prefix)

    # 对 ID 进行排序，方便按顺序处理
    sorted_ids = sorted(list(unique_ids), key=lambda x: int(x))
    return sorted_ids

def undistort_image(img, img_id, json_data):

    # 构建键名
    k_key = f"{img_id}_K"
    dist_key = f"{img_id}_distortion"
    width_key = f"{img_id}_width"
    height_key = f"{img_id}_height"

    # 获取参数
    K = np.array(json_data[k_key], dtype=np.float64)
    D = np.array(json_data[dist_key], dtype=np.float64)

    # 获取宽和高
    # 如果 JSON 中没有宽及高，尝试从图片中读取（作为兜底）
    if width_key in json_data and height_key in json_data:
        w = json_data[width_key]
        h = json_data[height_key]
    else:
        h, w = img.shape[:2]

    # --- 核心去畸变逻辑 ---

    # 计算新的相机矩阵
    # alpha=0: 裁剪图像，去除所有黑色边框（保留有效像素）
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))

    # 去畸变
    undistorted_img = cv2.undistort(img, K, D, None, new_K)

    # --- 更新 JSON 数据 ---

    # 更新内参矩阵 K -> 转换为 list 格式以符合 JSON 标准
    json_data[k_key] = new_K.tolist()

    # 更新畸变系数 -> 全部置为 0
    json_data[dist_key] = [0.0, 0.0, 0.0, 0.0, 0.0]

    return undistorted_img


def process_single_folder_task(src_json_path: Path):
    """
    单个文件夹的处理任务（在一个单独的进程中运行）
    """
    try:
        # 1. 路径计算
        # 计算相对路径，例如: "001-020/1/1_neutral/params.json"
        relative_path = src_json_path.relative_to(SOURCE_ROOT)

        # 对应的源文件夹和目标文件夹
        src_folder = src_json_path.parent
        dst_folder = (TARGET_ROOT / relative_path).parent

        # 创建目标文件夹
        dst_folder.mkdir(parents=True, exist_ok=True)

        # 2. 读取 JSON
        with open(src_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # calling user logic: 解析 ID
        img_ids = get_ids(json_data)

        # 3. 遍历并处理该文件夹下的所有指定图片
        for img_id in img_ids:
            # 拼凑文件名（假设是 .jpg，如果混合格式需另外判断）
            img_name = f"{img_id}.jpg"
            src_img_path = src_folder / img_name
            dst_img_path = dst_folder / img_name

            if not src_img_path.exists():
                # print(f"警告: 图片缺失 {src_img_path}")
                continue

            # 读取图片
            img = cv2.imread(str(src_img_path))
            if img is None:
                continue

            # calling user logic: 图像处理
            # 传入 img_id 和 json_data 是为了万一你的算法需要查表获取参数
            processed_img = undistort_image(img, img_id, json_data)

            # 保存处理后的图片
            cv2.imwrite(str(dst_img_path), processed_img)

        # 4. 保存 JSON
        target_json_path = dst_folder / "params.json"
        with open(target_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        return True  # 任务成功

    except Exception as e:
        return f"Error in {src_json_path}: {str(e)}"


def main():
    print("正在扫描所有 params.json 文件...")
    # 关键点：只找 params.json，以此为锚点锁定文件夹
    all_json_files = list(SOURCE_ROOT.rglob("params.json"))

    print(f"共发现 {len(all_json_files)} 个任务文件夹，准备开始多进程处理...")

    # 使用 ProcessPoolExecutor 进行多进程并行
    # 相比 ThreadPool，ProcessPool 能真正利用多核 CPU 处理图像计算
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        results = list(tqdm(
            executor.map(process_single_folder_task, all_json_files),
            total=len(all_json_files),
            unit="folder"
        ))

    # 简单的错误汇总
    errors = [res for res in results if res is not True]
    if errors:
        print(f"\n出现 {len(errors)} 个错误:")
        for err in errors[:5]:  # 只打印前5个错误
            print(err)
    else:
        print("\n所有任务全部完成！")


if __name__ == "__main__":
    main()