import os
import json
import time
import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr
import imageio.v2 as imageio
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. 核心渲染类 (保持不变)
# ==========================================
class DepthRenderer:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"[Init] Initializing nvdiffrast context on {device}...")
        try:
            self.glctx = dr.RasterizeCudaContext(device=device)
        except Exception as e:
            print(f"[Warning] CUDA init failed: {e}. Falling back to OpenGL.")
            self.glctx = dr.RasterizeGLContext(device=device)

    def load_mesh(self, ply_path):
        ply_path = str(ply_path)
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Mesh not found: {ply_path}")

        mesh = trimesh.load(ply_path, process=False)
        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).int()

        self.pos = verts.to(self.device).contiguous()
        self.tri = faces.to(self.device).contiguous()

    def render(self, K, Rt, H, W):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        n, f_plane = 1.0, 10000.0

        proj = torch.zeros((4, 4), device=self.device)
        proj[0, 0] = 2 * fx / W
        proj[0, 2] = - (2 * cx / W - 1)
        proj[1, 1] = - (2 * fy / H)
        proj[1, 2] = (1 - 2 * cy / H)
        proj[2, 2] = - (f_plane + n) / (f_plane - n)
        proj[2, 3] = - (2 * f_plane * n) / (f_plane - n)
        proj[3, 2] = -1.0

        cv2gl = torch.tensor([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0]
        ], device=self.device)

        Rt_tensor = torch.tensor(Rt, dtype=torch.float32, device=self.device)
        view_cv = torch.eye(4, device=self.device)
        view_cv[:3, :4] = Rt_tensor[:3, :4]

        view_gl = cv2gl @ view_cv
        mvp = proj @ view_gl

        pos_hom = torch.cat([self.pos, torch.ones((self.pos.shape[0], 1), device=self.device)], dim=1)
        pos_clip = pos_hom @ mvp.t()

        rast_out, _ = dr.rasterize(self.glctx, pos_clip[None, ...].contiguous(), self.tri, resolution=[H, W])

        w_clip = pos_clip[:, 3:4].contiguous()
        depth_buffer, _ = dr.interpolate(w_clip[None, ...], rast_out, self.tri)
        depth_map = depth_buffer[0, ..., 0]

        mask = rast_out[..., 3] > 0
        depth_map[~mask[0]] = 0.0

        return depth_map.detach().cpu().numpy()

# ==========================================
# 2. 异步保存任务
# ==========================================
def save_task(depth_map, save_path_npy, save_path_png):
    try:
        # 保存 .npy
        np.save(save_path_npy, depth_map)

        # 保存 .png
        if np.count_nonzero(depth_map) > 0:
            valid_vals = depth_map[depth_map > 0]
            d_min, d_max = valid_vals.min(), valid_vals.max()
            depth_vis = (depth_map - d_min) / (d_max - d_min + 1e-6)
            depth_vis[depth_map <= 0] = 0
            depth_vis = np.clip(depth_vis * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(save_path_png, depth_vis)
        else:
            h, w = depth_map.shape
            black_img = np.zeros((h, w), dtype=np.uint8)
            imageio.imwrite(save_path_png, black_img)
    except Exception as e:
        print(f"Save error: {e}")

# ==========================================
# 3. 业务逻辑 (已启用断点续传)
# ==========================================
IMAGES_ROOT = Path("/root/autodl-tmp/images_undistort")
SHAPES_ROOT = Path("/root/autodl-tmp/shapes")
DEPTH_ROOT  = Path("/root/autodl-tmp/depth")
IO_POOL_SIZE = 8

def process_single_scene(renderer, json_path: Path, io_executor):
    try:
        img_folder = json_path.parent
        relative_path = img_folder.relative_to(IMAGES_ROOT)
        ply_name = img_folder.name + ".ply"
        ply_path = SHAPES_ROOT / relative_path.parent / ply_name
        target_folder = DEPTH_ROOT / relative_path
        target_folder.mkdir(parents=True, exist_ok=True)

        # 检查 Mesh 是否存在，不存在直接跳过
        if not ply_path.exists():
            return False

        with open(json_path, 'r') as f:
            params = json.load(f)

        indices = sorted(list(set([
            int(k.split('_')[0]) for k in params.keys() if '_' in k and k.split('_')[0].isdigit()
        ])))

        # --- 【关键优化】预先过滤已完成的索引 ---
        # 如果对应的 .npy 文件已经存在，就视为已完成。
        # 这样可以避免加载 Mesh (如果整个文件夹都做完了)，极大加快重启速度。
        pending_indices = []
        for idx in indices:
            save_path_npy = target_folder / f"{idx}.npy"
            if not save_path_npy.exists():
                pending_indices.append(idx)

        # 如果这个文件夹里所有图片都处理完了，直接返回，连 Mesh 都不用加载了
        if not pending_indices:
            return True

        # ----------------------------------------
        # 只有确实有任务要做，才加载 Mesh 到 GPU
        renderer.load_mesh(ply_path)

        for idx in pending_indices:
            # 再次检查有效性标记
            if f"{idx}_valid" in params and not params[f"{idx}_valid"]:
                continue

            img_filename = f"{idx}.jpg"
            if not (img_folder / img_filename).exists():
                continue

            K = np.array(params[f"{idx}_K"])
            Rt = np.array(params[f"{idx}_Rt"])
            H = params[f"{idx}_height"]
            W = params[f"{idx}_width"]

            # 构造保存路径
            save_path_npy = str(target_folder / f"{idx}.npy")
            save_path_png = str(target_folder / f"{idx}.png")

            # 渲染
            depth_map = renderer.render(K, Rt, H, W)

            # 异步保存
            io_executor.submit(save_task, depth_map.copy(), save_path_npy, save_path_png)

        return True

    except Exception as e:
        print(f"\n[Error] {relative_path}: {e}")
        return False

def main():
    print("正在扫描任务...")
    all_json_files = list(IMAGES_ROOT.rglob("params.json"))
    print(f"共发现 {len(all_json_files)} 个场景。")

    renderer = DepthRenderer(device='cuda')

    print(f"启动 IO 线程池 (Workers={IO_POOL_SIZE})...")
    with ThreadPoolExecutor(max_workers=IO_POOL_SIZE) as io_executor:
        success_count = 0
        # tqdm 会显示总体进度
        for json_path in tqdm(all_json_files, unit="scene"):
            if process_single_scene(renderer, json_path, io_executor):
                success_count += 1

        print("所有渲染任务提交完成，正在等待剩余文件写入硬盘...")

    print(f"全部完成。成功处理场景: {success_count}/{len(all_json_files)}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()