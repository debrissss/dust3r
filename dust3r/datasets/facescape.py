# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed BlendedMVS
# dataset at https://github.com/YoYo000/BlendedMVS
# See datasets_preprocess/preprocess_blendedmvs.py
# --------------------------------------------------------
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class FaceScape (BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, *args, ROOT, split=None, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self._load_data(split)

    def _load_data(self, split):
        # 1. 加载你的 npy 文件
        # 注意：因为你的数据包含字符串，load 时必须加 allow_pickle=True
        pairs_path = osp.join(self.ROOT, 'facescape_pairs.npy')
        print(f"Loading pairs from {pairs_path}...")
        pairs = np.load(pairs_path, allow_pickle=True)

        # 2. 如果不区分 split，直接返回全量
        if split is None:
            self.pairs = pairs
        else:
            # 3. 实现 Train/Val 自动划分逻辑
            # 我们需要根据 "场景文件夹名称" (pairs[:, 0]) 来做确定性的划分
            filtered_pairs = []

            # 使用 zlib.adler32 进行字符串哈希，它比 python 自带的 hash() 更稳定（跨进程一致）
            import zlib

            for p in pairs:
                scene_name = p[0] # 获取场景名，例如 "154_8_mouth_left"

                # 将字符串转为整数 Hash 值
                scene_hash = zlib.adler32(scene_name.encode('utf-8'))

                # 取模运算，逻辑与 BlendedMVS 保持一致
                remainder = scene_hash % 10

                if split == 'train':
                    # 90% 的场景进入训练集 (余数 1-9)
                    if remainder > 0:
                        filtered_pairs.append(p)
                elif split == 'val':
                    # 10% 的场景进入验证集 (余数 0)
                    if remainder == 0:
                        filtered_pairs.append(p)

            self.pairs = filtered_pairs

        # 4. 更新场景列表 (用于统计)
        # 提取第0列（场景名），去重
        # 注意：如果 filtered_pairs 是 list，需要先转 numpy 或直接用 set
        self.scenes = np.unique([p[0] for p in self.pairs])
        print(f"[{split}] Loaded {len(self.pairs)} pairs from {len(self.scenes)} scenes.")

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        # 1. 解包数据 (对应你的数据结构)
        # [场景名, 图片1 ID, 图片2 ID, 角度差异]
        scene_name, id1, id2, angle = self.pairs[pair_idx]

        # 2. 拼接场景路径
        # 不需要 hex 转换，直接拼接字符串
        seq_path = osp.join(self.ROOT, scene_name)

        views = []

        for view_index in [id1, id2]:
            # 3. 获取图片文件名
            # 假设你的 ID "1" 对应的文件名就是 "1.jpg"
            # 如果文件名是 "00001.jpg"，你需要改成: impath = f"{int(view_index):05d}"
            impath = str(view_index)

            # 4. 读取文件 (jpg, exr, npz)
            # 这里的逻辑与原版保持一致，前提是你的目录下确实有这三个文件
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(seq_path, impath + ".exr"))
            camera_params = np.load(osp.join(seq_path, impath + ".npz"))

            # 5. 拼装相机矩阵 (保持原版逻辑)
            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params['R_cam2world']
            camera_pose[:3, 3] = camera_params['t_cam2world']

            # 6. 裁剪与缩放 (保持原版逻辑)
            # 这一步非常重要，它会处理图片缩放、内参调整和深度图缩放
            # image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #     image, depthmap, intrinsics, resolution, rng, info=(seq_path, impath))

            try:
                image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    image, depthmap, intrinsics, resolution, rng, info=(seq_path, impath)
                )
            except ValueError as e:
                # === 捕获异常并打印详细信息 ===
                print("\n" + "="*50)
                print(f"!!! CRITICAL ERROR IN DATASET PROCESSING !!!")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}")
                print("-" * 30)
                print(f"Problematic Scene: {scene_name}")
                print(f"Problematic View ID: {view_index}")
                print(f"Image Path: {full_img_path}")
                print("-" * 30)
                if hasattr(image, 'shape'):
                    print(f"Image Shape: {image.shape}")
                print(f"Target Resolution: {resolution}")
                print(f"Intrinsics:\n{intrinsics}")
                print("="*50 + "\n")
                raise e

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='FaceScape',  # 修改数据集标签
                label=scene_name,     # 用于 debug 显示
                instance=impath))

        return views


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = FaceScape(split='train', ROOT="data/facescape_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()