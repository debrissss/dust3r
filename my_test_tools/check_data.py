import os
import numpy as np
import cv2
from tqdm import tqdm

# 配置路径
ROOT = '/root/autodl-tmp/facescape_processed'
PAIRS_FILE = 'facescape_pairs.npy'

def check_dataset():
    pairs_path = os.path.join(ROOT, PAIRS_FILE)
    if not os.path.exists(pairs_path):
        print(f"Error: {pairs_path} 不存在")
        return

    print(f"正在加载索引: {pairs_path}")
    pairs = np.load(pairs_path, allow_pickle=True)
    print(f"共加载 {len(pairs)} 对数据，开始抽样检查...")

    # 为了节省时间，你可以只检查前 5000 个，或者去掉切片 [:5000] 检查全部
    for i, p in enumerate(tqdm(pairs)):
        scene_name, id1, id2, angle = p
        scene_path = os.path.join(ROOT, scene_name)

        for view_id in [id1, id2]:
            impath = str(view_id) + ".jpg"
            img_full_path = os.path.join(scene_path, impath)

            # 1. 检查文件是否存在
            if not os.path.exists(img_full_path):
                print(f"\n[缺失] 第 {i} 条数据: 图片不存在 {img_full_path}")
                continue

            # 2. 检查图片是否能读取且尺寸正常
            try:
                img = cv2.imread(img_full_path)
                if img is None:
                    print(f"\n[损坏] 第 {i} 条数据: 图片无法读取 (None) {img_full_path}")
                elif img.shape[0] == 0 or img.shape[1] == 0:
                    print(f"\n[异常] 第 {i} 条数据: 图片尺寸为0 {img_full_path} shape={img.shape}")
            except Exception as e:
                print(f"\n[报错] 读取错误 {img_full_path}: {e}")

if __name__ == "__main__":
    check_dataset()