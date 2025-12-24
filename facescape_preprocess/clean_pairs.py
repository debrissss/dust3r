import numpy as np
import os
from tqdm import tqdm

def clean_pairs_file(pairs_file_path, error_log_path, output_path):
    """
    根据 error_log 移除 pairs 文件中包含坏图的配对。
    """
    # --- 1. 检查文件 ---
    if not os.path.exists(pairs_file_path):
        print(f"❌ 错误: 找不到配对文件 {pairs_file_path}")
        return

    if not os.path.exists(error_log_path):
        print(f"❌ 错误: 找不到错误日志 {error_log_path}")
        return

    print("正在加载数据...")
    try:
        # allow_pickle=True 是必须的，因为数据包含字符串对象
        pairs_data = np.load(pairs_file_path, allow_pickle=True)
        error_data = np.load(error_log_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 加载 Numpy 文件失败: {e}")
        return

    print(f"原始配对数量: {len(pairs_data)}")
    print(f"错误日志条目: {len(error_data)}")

    if len(error_data) == 0:
        print("✅ 错误日志为空，无需清洗。")
        return

    # --- 2. 构建黑名单集合 (Set) ---
    # 格式: (场景名, 图片ID)
    # 使用 Set 进行查找可以将时间复杂度从 O(N*M) 降低到 O(N)
    bad_keys = set()
    for row in error_data:
        # error_log 格式: [scene, id, error_msg, cx, cy, ...]
        scene = row[0]
        # 确保 ID 转为字符串，因为 pairs 文件里通常是字符串
        img_id = str(row[1])
        bad_keys.add((scene, img_id))

    print(f"待剔除的违规图片 (Scene, ID) 数量: {len(bad_keys)}")

    # --- 3. 过滤配对 ---
    valid_pairs = []
    removed_count = 0

    # pairs 格式: [scene, id1, id2, score]
    # 使用 tqdm 显示进度条
    for row in tqdm(pairs_data, desc="Cleaning Pairs"):
        scene = row[0]
        id1 = str(row[1])
        id2 = str(row[2])

        # 检查配对中的任意一张图是否在黑名单中
        if (scene, id1) in bad_keys or (scene, id2) in bad_keys:
            removed_count += 1
        else:
            valid_pairs.append(row)

    # --- 4. 保存结果 ---
    valid_pairs = np.array(valid_pairs, dtype=object)

    print("-" * 30)
    print(f"清洗完成 Summary:")
    print(f"原始数量 : {len(pairs_data)}")
    print(f"删除数量 : {removed_count}")
    print(f"剩余数量 : {len(valid_pairs)}")

    if len(valid_pairs) > 0:
        np.save(output_path, valid_pairs)
        print(f"✅ 已保存清洗后的文件至: {output_path}")
    else:
        print("⚠️ 警告: 所有配对都被删除了！请检查错误日志是否过于激进。")

if __name__ == "__main__":
    # --- 配置路径 ---
    # 原始配对文件路径
    PAIRS_PATH = "/root/autodl-tmp/facescape_processed/facescape_pairs.npy"

    # 上一步生成的错误日志 (请确保文件名一致)
    ERROR_LOG_PATH = "error_log_strict.npy"

    # 输出文件路径
    OUTPUT_PATH = "/root/autodl-tmp/facescape_processed/facescape_pairs.npy"

    clean_pairs_file(PAIRS_PATH, ERROR_LOG_PATH, OUTPUT_PATH)