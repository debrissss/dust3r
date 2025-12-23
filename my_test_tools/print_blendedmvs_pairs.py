import numpy as np

# 加载 npy 文件
path = '/root/autodl-tmp/blendedmvs_pairs.npy'
data = np.load(path, allow_pickle=True)

# 打印基本信息
print(f"数据总行数 (Pairs): {data.shape[0]}")
print(f"数据结构 (Dtype): {data.dtype.names}")

# 打印前 5 行样本数据
print("\n--- 前 5 条数据样本 ---")
for i in range(5):
    print(data[i])