import os
import json
import time
import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr
import imageio.v2 as imageio
import argparse

# ==========================================
# 核心渲染类
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
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Mesh not found: {ply_path}")

        print(f"[Mesh] Loading: {ply_path} ...")
        mesh = trimesh.load(ply_path, process=False)

        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).int()

        self.pos = verts.to(self.device).contiguous()
        self.tri = faces.to(self.device).contiguous()

        print(f"[Mesh] Loaded {self.pos.shape[0]} verts, {self.tri.shape[0]} faces.")

    def render(self, K, Rt, H, W):
        # 1. Projection Matrix (透视投影矩阵)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        n, f_plane = 1.0, 10000.0

        proj = torch.zeros((4, 4), device=self.device)

        # X 轴 (保持不变，标准 GL)
        proj[0, 0] = 2 * fx / W
        proj[0, 2] = - (2 * cx / W - 1)

        # 【最终修复点】 Y 轴 (完全反转)
        # 我们需要让图像在渲染时就"倒"过来（对应 ImageIO 的正向），同时保持 cy 位置正确。
        # 做法：将标准 GL 投影矩阵的第二行全部取反。
        # 1. 反转缩放:
        proj[1, 1] = - (2 * fy / H)
        # 2. 反转偏移 (这是之前导致向下偏移的原因！):
        # 标准是 -(2*cy/H - 1)，取反后变成 (2*cy/H - 1) ... 等等，标准其实是 (1 - 2*cy/H) 或类似
        # 让我们直接推导结论：
        # 我们要映射: v=0 (Top) -> NDC=-1 (Bottom/Row0)
        #             v=H (Bottom) -> NDC=+1 (Top/RowH)
        # 推导结果是: Offset = 1 - 2*cy/H
        proj[1, 2] = (1 - 2 * cy / H)

        # Z 轴 (保持不变)
        proj[2, 2] = - (f_plane + n) / (f_plane - n)
        proj[2, 3] = - (2 * f_plane * n) / (f_plane - n)
        proj[3, 2] = -1.0

        # 2. View Matrix (OpenCV -> OpenGL)
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

        # 3. Rasterization
        pos_hom = torch.cat([self.pos, torch.ones((self.pos.shape[0], 1), device=self.device)], dim=1)
        pos_clip = pos_hom @ mvp.t()

        rast_out, _ = dr.rasterize(self.glctx, pos_clip[None, ...].contiguous(), self.tri, resolution=[H, W])

        # 4. Interpolate Linear Depth (w_clip)
        w_clip = pos_clip[:, 3:4].contiguous()
        depth_buffer, _ = dr.interpolate(w_clip[None, ...], rast_out, self.tri)
        depth_map = depth_buffer[0, ..., 0]

        # Mask background
        mask = rast_out[..., 3] > 0
        depth_map[~mask[0]] = 0.0

        # 不需要 torch.flip，矩阵已经处理好了
        return depth_map.detach().cpu().numpy()

# ==========================================
# 数据处理与保存
# ==========================================
def process_dataset(root_dir, renderer):
    ply_path = os.path.join(root_dir, "2_smile.ply")
    params_path = os.path.join(root_dir, "params.json")
    output_dir = os.path.join(root_dir, "2_smile_depth")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(params_path):
        print(f"[Error] Not found: {params_path}")
        return

    renderer.load_mesh(ply_path)

    with open(params_path, 'r') as f:
        params = json.load(f)

    indices = sorted(list(set([int(k.split('_')[0]) for k in params.keys() if '_' in k])))
    print(f"[Task] Processing {len(indices)} images...")

    count = 0
    start_t = time.time()

    for idx in indices:
        if f"{idx}_valid" in params and not params[f"{idx}_valid"]:
            continue

        try:
            K = np.array(params[f"{idx}_K"])
            Rt = np.array(params[f"{idx}_Rt"])
            H = params[f"{idx}_height"]
            W = params[f"{idx}_width"]

            depth_map = renderer.render(K, Rt, H, W)

            # 只有当有内容时才保存
            if np.count_nonzero(depth_map) > 0:
                # 1. 保存高精度数据 (.npy) - 推荐后续算法使用
                np.save(os.path.join(output_dir, f"{idx}.npy"), depth_map)

                # 2. 保存预览图 (.png) - 仅供肉眼检查
                # 动态拉伸对比度，让人脸看清楚
                valid_vals = depth_map[depth_map > 0]
                d_min, d_max = valid_vals.min(), valid_vals.max()

                depth_vis = (depth_map - d_min) / (d_max - d_min + 1e-6)
                depth_vis[depth_map <= 0] = 0 # 背景归零
                depth_vis = np.clip(depth_vis * 255, 0, 255).astype(np.uint8)

                imageio.imwrite(os.path.join(output_dir, f"{idx}.png"), depth_vis)
                print(f" -> [{idx}] Saved. Depth Range: {d_min:.1f} - {d_max:.1f}")
            else:
                print(f" -> [{idx}] Empty (Black).")

            count += 1

        except Exception as e:
            print(f"[Error] {idx}: {e}")

    print(f"Done. Processed {count} images in {time.time()-start_t:.2f}s")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    if args.root:
        root = args.root
    else:
        root = "/root/dust3r/my_test_tools/facescape_test"

    if os.path.exists(root):
        renderer = DepthRenderer()
        process_dataset(root, renderer)
    else:
        print(f"Path not found: {root}")