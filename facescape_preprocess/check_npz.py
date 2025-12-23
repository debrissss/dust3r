import numpy as np
import os

def inspect_npz(file_path):
    print(f"\n{'='*40}")
    print(f"正在检查文件: {os.path.basename(file_path)}")
    print(f"{'='*40}")

    if not os.path.exists(file_path):
        print("错误: 文件不存在！")
        return

    try:
        data = np.load(file_path)

        # 1. 检查 Keys
        print(f"包含的 Keys: {data.files}")
        expected_keys = ['intrinsics', 'R_cam2world', 't_cam2world']
        missing_keys = [k for k in expected_keys if k not in data.files]
        if missing_keys:
            print(f"❌ 警告: 缺少关键 Keys: {missing_keys}")
        else:
            print(f"✅ Keys 完整。")

        # 2. 打印详细数值
        # --- Intrinsics ---
        K = data['intrinsics']
        print(f"\n[Intrinsics] (Shape: {K.shape}, Type: {K.dtype})")
        print(K)

        # --- Rotation ---
        R = data['R_cam2world']
        print(f"\n[R_cam2world] (Shape: {R.shape}, Type: {R.dtype})")
        print(R)

        # --- Translation ---
        t = data['t_cam2world']
        print(f"\n[t_cam2world] (Shape: {t.shape}, Type: {t.dtype})")
        print(t)

        print("-" * 40)

        # 3. 快速数值验证 (针对你的 Index 0 数据)
        # 如果是 0.npz，我们可以检查一下数值是否符合预期
        if "0.npz" in file_path:
            # 这里的预期值是基于你 params.json 中 index 0 计算出来的
            expected_t = np.array([-0.19445, -6.5372, 13.6883])
            diff = np.linalg.norm(t - expected_t)

            if diff < 1.0: # 允许一点浮点误差
                print("✅ 数值验证通过: t_cam2world 与预期计算结果一致。")
            else:
                print(f"⚠️ 数值警告: t_cam2world ({t}) 与预期值 ({expected_t}) 差异较大。")
                print("可能原因：坐标系转换逻辑未执行，或者取逆矩阵出错。")

    except Exception as e:
        print(f"读取出错: {e}")

if __name__ == "__main__":
    # 修改为你实际生成的 npz 路径
    target_file = "/root/autodl-tmp/trainsets/001_1_neutral/0.npz"
    # target_file = "/root/autodl-tmp/BlendedMVS_preprocessed/57f8d9bbe73f6760f10e916a/00000000.npz"
    inspect_npz(target_file)