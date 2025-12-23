import os

# ==========================================
# 【配置】环境变量与性能设置
# ==========================================
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# 防止 OpenCV 内部多线程与 Python 线程池冲突，强制单线程运行 OpenCV 函数，
# 依靠 Python 的 Process/ThreadPool 来实现并行。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
import datetime
import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr
import cv2
import gc
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

# 禁用 cv2 的多线程，避免在 ThreadPool 中发生死锁或资源争抢
cv2.setNumThreads(0)

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

        num_verts = verts.shape[0]
        if num_verts > 0 and faces.shape[0] > 0:
            valid_mask = (faces < num_verts).all(dim=1)
            if not valid_mask.all():
                faces = faces[valid_mask]

        self.pos = verts.to(self.device).contiguous()
        self.tri = faces.to(self.device).contiguous()

        del mesh
        del verts
        del faces

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
# 2. 核心优化：全能工作线程 (CPU 密集型任务)
# ==========================================
def process_and_save_task(
        depth_map_orig,     # 原始深度图 (numpy)
        src_img_path,       # 原图路径
        K_orig,             # 原始内参
        R_c2w, t_c2w,       # 外参
        jpg_path, exr_path, npz_path, # 保存路径
        semaphore
):
    """
    这个函数在后台线程运行。它负责所有慢速的 CPU 操作：
    读取图片 -> Resize图片(Lanczos) -> Resize深度图 -> 保存文件
    """
    try:
        # 1. 读取原图 (IO 操作)
        # 移到这里读取，避免阻塞主线程渲染
        img_original_cv = cv2.imread(str(src_img_path))
        if img_original_cv is None:
            return

        h_orig, w_orig = img_original_cv.shape[:2]

        # 2. 计算缩放 (CPU 计算)
        target_long_edge = 1024
        scale = target_long_edge / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)

        # 3. 图片缩放 (CPU 密集型 - Lanczos)
        # 这一步是之前的瓶颈，现在会在 20+ 个核心上并行执行
        img_pil = Image.fromarray(cv2.cvtColor(img_original_cv, cv2.COLOR_BGR2RGB))
        img_pil_resized = img_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # 4. 深度图缩放 (CPU 密集型)
        depth_map_resized = cv2.resize(depth_map_orig, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 5. 修改内参
        K_new = K_orig.copy()
        K_new[:2, :] *= scale

        # 6. 保存所有文件 (IO 操作)
        img_pil_resized.save(jpg_path, quality=95)

        cv2.imwrite(exr_path, depth_map_resized.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

        np.savez(npz_path,
                 intrinsics=K_new.astype(np.float32),
                 R_cam2world=R_c2w.astype(np.float64),
                 t_cam2world=t_c2w.astype(np.float32))

    except Exception as e:
        print(f"Error processing {src_img_path}: {e}")
    finally:
        # 显式释放大内存对象
        del depth_map_orig
        del img_original_cv
        del img_pil
        del img_pil_resized
        del depth_map_resized
        semaphore.release()

# ==========================================
# 3. 业务逻辑
# ==========================================
IMAGES_ROOT = Path("/root/autodl-tmp/images_undistort")
SHAPES_ROOT = Path("/root/autodl-tmp/shapes")
TRAINSETS_ROOT = Path("/root/autodl-tmp/trainsets")

# 【性能调优配置】
# CPU 核心数利用：设为 CPU 核心数 - 2 (留给主线程和 OS)
# 假设你有 25 个核心，这里开 22 个线程专门做 Resize 和 IO
IO_POOL_SIZE = 22

# 内存控制：
# 一张 4000x3000 的 float32 深度图约 48MB。
# 积压 500 张 = 24GB，加上读取的图片和中间变量，约占 40-50GB 内存。
# 对于 90GB 内存机器，设为 400-500 是安全的。
MAX_PENDING_TASKS = 400

def process_single_scene(renderer, json_path: Path, io_executor, semaphore):
    try:
        img_folder = json_path.parent
        relative_path = img_folder.relative_to(IMAGES_ROOT)

        ply_name = img_folder.name + ".ply"
        ply_path = SHAPES_ROOT / relative_path.parent / ply_name

        try:
            subject_id = int(img_folder.parent.name)
            scene_name = f"{subject_id:03d}_{img_folder.name}"
        except ValueError:
            scene_name = f"{img_folder.parent.name}_{img_folder.name}"

        scene_output_dir = TRAINSETS_ROOT / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        # 这里不返回，继续判断 ply 是否存在，如果不存在直接失败
        if not ply_path.exists():
            return False, scene_name

        with open(json_path, 'r') as f:
            params = json.load(f)

        indices = sorted(list(set([
            int(k.split('_')[0]) for k in params.keys() if '_' in k and k.split('_')[0].isdigit()
        ])))

        pending_indices = []
        for idx in indices:
            save_path_exr = scene_output_dir / f"{idx}.exr"
            if not save_path_exr.exists():
                pending_indices.append(idx)

        if not pending_indices:
            return True, scene_name

        try:
            renderer.load_mesh(ply_path)
        except Exception as e:
            print(f"[Skipped] Mesh load error {ply_path}: {e}")
            return False, scene_name

        # --- 极速循环 ---
        # 主线程只负责渲染，不等待 IO 和 Resize
        for idx in pending_indices:
            if f"{idx}_valid" in params and not params[f"{idx}_valid"]:
                continue

            img_filename = f"{idx}.jpg"
            src_img_path = img_folder / img_filename
            # 简单检查文件是否存在
            if not src_img_path.exists():
                continue

            # 【关键优化】直接从 params.json 获取宽高
            # 避免在主线程读取图片
            width_key = f"{idx}_width"
            height_key = f"{idx}_height"

            if width_key in params and height_key in params:
                w_orig = params[width_key]
                h_orig = params[height_key]
            else:
                # 兜底：万一 json 没写宽高，只能读图（会变慢，但为了稳健性）
                # 建议使用 imagesize 库只读头信息，这里为了不引入新库，暂用 imread
                temp_img = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)
                if temp_img is None: continue
                h_orig, w_orig = temp_img.shape[:2]
                del temp_img

            K_orig = np.array(params[f"{idx}_K"])
            Rt = np.array(params[f"{idx}_Rt"])

            # 1. GPU 渲染 (主线程，极快)
            depth_map_orig = renderer.render(K_orig, Rt, h_orig, w_orig)

            # 2. 准备参数
            T_w2c = np.eye(4)
            T_w2c[:3, :4] = Rt
            try:
                T_c2w = np.linalg.inv(T_w2c)
                R_c2w = T_c2w[:3, :3]
                t_c2w = T_c2w[:3, 3]
            except np.linalg.LinAlgError:
                print(f"Singular matrix for {scene_name} img {idx}")
                continue

            save_path_jpg = str(scene_output_dir / f"{idx}.jpg")
            save_path_exr = str(scene_output_dir / f"{idx}.exr")
            save_path_npz = str(scene_output_dir / f"{idx}.npz")

            # 3. 申请资源 (如果堆积太多任务，这里会暂停主线程)
            semaphore.acquire()

            # 4. 提交给工人线程 (包括读取原图、Resize、保存)
            # 注意：传入的是 depth_map_orig.copy()
            io_executor.submit(
                process_and_save_task,
                depth_map_orig.copy(), # 必须 copy，因为下一帧渲染会复用显存
                src_img_path,
                K_orig,
                R_c2w, t_c2w,
                save_path_jpg, save_path_exr, save_path_npz,
                semaphore
            )

        # 场景处理完，手动触发 GC
        gc.collect()

        return True, scene_name

    except Exception as e:
        print(f"\n[Error] {relative_path}: {e}")
        return False, "Unknown"

def main():
    print("正在扫描所有任务文件...")
    all_json_files = list(IMAGES_ROOT.rglob("params.json"))
    total_scenes = len(all_json_files)
    print(f"共发现 {total_scenes} 个场景。")

    renderer = DepthRenderer(device='cuda')
    task_semaphore = BoundedSemaphore(value=MAX_PENDING_TASKS)

    print(f"启动高性能处理池 (Workers={IO_POOL_SIZE}, Max Pending={MAX_PENDING_TASKS})...")
    print("-" * 80)

    start_time_global = time.time()

    with ThreadPoolExecutor(max_workers=IO_POOL_SIZE) as io_executor:
        success_count = 0

        for i, json_path in enumerate(all_json_files):
            # 处理单个场景
            status, scene_name = process_single_scene(renderer, json_path, io_executor, task_semaphore)

            if status:
                success_count += 1

            # --- 时间统计与日志 ---
            current_processed = i + 1
            elapsed_time = time.time() - start_time_global

            # 为了避免除以0
            if current_processed > 0:
                avg_time_per_scene = elapsed_time / current_processed
                remaining_scenes = total_scenes - current_processed
                eta_seconds = avg_time_per_scene * remaining_scenes

                elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

                # 打印紧凑的日志
                print(f"[{current_processed}/{total_scenes}] {scene_name} | 耗时: {elapsed_str} | ETA: {eta_str}")

        print("-" * 80)
        print("所有渲染任务已分发，正在等待后台线程完成最后的处理...")

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time_global)))
    print(f"全部完成。总耗时: {total_time_str}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()