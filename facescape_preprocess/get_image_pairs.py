import os
import glob
import numpy as np
from itertools import combinations
from collections import defaultdict

# ================= 配置区域 =================

CONFIG = {
    # 根目录：包含各个场景文件夹的目录
    'ROOT_DIR': '/root/autodl-tmp/trainsets',

    # 结果保存路径
    'OUTPUT_NPY': '/root/autodl-tmp/trainsets/facescape_pairs.npy',

    # 角度分桶配置 (min, max, count)
    # 20-80度，每10度一个桶，每个桶取10组
    'ANGLE_BINS': [
        (20, 30, 10),
        (30, 40, 10),
        (40, 50, 10),
        (50, 60, 10),
        (60, 70, 10),
        (70, 80, 10)
    ]
}

# ===========================================

class CameraUtils:
    """相机工具类：适配 .npz 格式"""

    @staticmethod
    def load_cameras_from_dir(scene_path):
        """
        从场景目录下读取所有 .npz 文件
        返回: list of dict [{'id': '0', 'view_dir': ..., 'center': ...}, ...]
        """
        cameras = []
        # 查找所有 npz 文件
        npz_files = glob.glob(os.path.join(scene_path, "*.npz"))

        for npz_path in npz_files:
            try:
                # 获取文件名作为 ID (例如 0.npz -> 0)
                file_name = os.path.basename(npz_path)
                img_id = os.path.splitext(file_name)[0]

                data = np.load(npz_path)

                if 'R_cam2world' not in data or 't_cam2world' not in data:
                    print(f"[Warn] {file_name} 缺少 R/t 参数，跳过。")
                    continue

                R_c2w = data['R_cam2world'] # Shape (3, 3)
                t_c2w = data['t_cam2world'] # Shape (3,)

                # 1. 计算光轴方向 (View Direction)
                # 对于 Camera-to-World 矩阵，相机的 Z 轴在世界坐标系中的方向就是 R 的第三列
                view_dir = R_c2w[:, 2]
                # 归一化 (理论上旋转矩阵列向量已经是单位向量，但也防万一)
                norm = np.linalg.norm(view_dir)
                if norm > 0: view_dir /= norm

                # 2. 计算相机中心 (Center)
                # 对于 Camera-to-World，平移向量 t 就是相机中心在世界坐标系的位置
                center = t_c2w

                cameras.append({
                    'id': img_id,
                    'view_dir': view_dir,
                    'center': center
                })

            except Exception as e:
                print(f"[Error] 读取 {npz_path} 失败: {e}")

        # 按 ID 数字排序，方便后续处理
        cameras.sort(key=lambda x: int(x['id']) if x['id'].isdigit() else x['id'])
        return cameras

    @staticmethod
    def calculate_angle(cam1, cam2):
        """计算两个相机视线的夹角 (度)"""
        dot = np.clip(np.dot(cam1['view_dir'], cam2['view_dir']), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        return angle_deg

class PairSelector:
    """配对筛选器：实现分桶与最小复用策略"""

    @staticmethod
    def select_pairs_for_scene(cameras, config):
        """
        核心算法：
        1. 计算所有可能的 Pair
        2. 按角度分桶
        3. 在每个桶内，贪心选择复用次数最少的 Pair
        """
        if len(cameras) < 2:
            return []

        # 1. 计算所有候选 Pair
        bins_data = { (b_min, b_max): [] for b_min, b_max, _ in config['ANGLE_BINS'] }

        # 记录每张图片被使用的次数，用于贪心策略
        usage_count = defaultdict(int)

        # 遍历所有组合
        for i, j in combinations(range(len(cameras)), 2):
            cam1, cam2 = cameras[i], cameras[j]
            angle = CameraUtils.calculate_angle(cam1, cam2)

            # 找到对应的桶
            for b_min, b_max, _ in config['ANGLE_BINS']:
                if b_min <= angle < b_max:
                    bins_data[(b_min, b_max)].append({
                        'id1': cam1['id'],
                        'id2': cam2['id'],
                        'angle': angle
                    })
                    break

        final_pairs = []

        # 2. 逐桶筛选
        for (b_min, b_max, limit) in config['ANGLE_BINS']:
            candidates = bins_data[(b_min, b_max)]
            selected_in_bin = []

            # 如果候选数量少于限制，全选
            if len(candidates) <= limit:
                selected_in_bin = candidates
                # 更新使用计数
                for p in candidates:
                    usage_count[p['id1']] += 1
                    usage_count[p['id2']] += 1
            else:
                # --- 核心：最小复用贪心选择 ---
                # 循环 limit 次，每次选一个“当前代价最小”的 pair
                temp_candidates = candidates.copy()

                for _ in range(limit):
                    if not temp_candidates:
                        break

                    # 评分函数：(两张图已使用次数之和, 距离桶中心的角度偏差)
                    # 优先选使用少的，如果一样，选角度最正的
                    def cost_func(p):
                        usage_score = usage_count[p['id1']] + usage_count[p['id2']]
                        angle_score = abs(p['angle'] - (b_min + b_max)/2)
                        return (usage_score, angle_score)

                    # 排序
                    temp_candidates.sort(key=cost_func)

                    # 选最好的
                    best_pair = temp_candidates.pop(0)
                    selected_in_bin.append(best_pair)

                    # 更新计数
                    usage_count[best_pair['id1']] += 1
                    usage_count[best_pair['id2']] += 1

            final_pairs.extend(selected_in_bin)

        return final_pairs

def main():
    root_dir = CONFIG['ROOT_DIR']
    if not os.path.exists(root_dir):
        print(f"错误: 根目录 {root_dir} 不存在")
        return

    # 扫描根目录下的所有子文件夹（假设每个子文件夹是一个场景）
    # 如果根目录本身就是一个场景，请调整此处逻辑
    # 这里假设 /trainsets/ 下面有 206_13_lip_funneler 等多个场景文件夹
    scene_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # 如果没找到子文件夹，可能 root_dir 本身就是场景目录，尝试把它自己加进去
    if not scene_folders and glob.glob(os.path.join(root_dir, "*.npz")):
        scene_folders = ["."]
        print("根目录下直接发现 npz 文件，将根目录视为单个场景处理。")

    all_results = []
    total_scenes = 0

    print(f"开始扫描，根目录: {root_dir}")
    print(f"策略: 20-80度分桶，每桶10组，最小化图片复用")

    for scene_name in scene_folders:
        if scene_name == ".":
            scene_path = root_dir
            # 如果是当前目录，名字取文件夹名
            scene_display_name = os.path.basename(os.path.normpath(root_dir))
        else:
            scene_path = os.path.join(root_dir, scene_name)
            scene_display_name = scene_name

        # 1. 加载相机
        cameras = CameraUtils.load_cameras_from_dir(scene_path)
        if not cameras:
            continue

        # 2. 筛选配对
        pairs = PairSelector.select_pairs_for_scene(cameras, CONFIG)

        if not pairs:
            print(f"场景 {scene_display_name}: 未找到符合条件的配对")
            continue

        # 3. 格式化结果 [场景名, id1, id2, angle]
        for p in pairs:
            all_results.append([
                scene_display_name,
                p['id1'],
                p['id2'],
                p['angle']
            ])

        print(f"场景 {scene_display_name}: 找到 {len(pairs)} 组配对 (相机数: {len(cameras)})")
        total_scenes += 1

    # 4. 保存结果
    if all_results:
        out_path = CONFIG['OUTPUT_NPY']
        np.save(out_path, np.array(all_results, dtype=object))
        print("-" * 50)
        print(f"处理完成！共 {total_scenes} 个场景，{len(all_results)} 条数据。")
        print(f"结果已保存至: {out_path}")

        # 打印前5条预览
        print("预览前5条数据:")
        for row in all_results[:5]:
            print(row)
    else:
        print("未生成任何数据。")

if __name__ == "__main__":
    main()