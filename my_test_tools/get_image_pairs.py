import json
import numpy as np
from itertools import combinations

# 1. 读取 JSON 数据
params_path = '/root/dust3r/my_test_tools/facescape_test/params.json'
with open(params_path, 'r') as f:
    data = json.load(f)

# 2. 解析数据
parsed_cameras = {}
for key, value in data.items():
    if "_" in key:
        idx, param = key.split("_", 1)
        if idx not in parsed_cameras:
            parsed_cameras[idx] = {}
        parsed_cameras[idx][param] = value

# 3. 提取相机参数和光轴
cameras = []
for idx, params in parsed_cameras.items():
    if params.get('valid') is True:
        Rt = np.array(params['Rt'])
        # 提取旋转矩阵的第三行作为光轴方向 (View Direction)
        view_dir = Rt[2, :3]
        norm = np.linalg.norm(view_dir)
        if norm > 0:
            view_dir = view_dir / norm

        cameras.append({
            'id': idx,
            'filename': params.get('ori', f"Image_{idx}"),
            'view_dir': view_dir
        })

# 4. 设定阈值
ANGLE_THRESHOLD = 30.0

# 5. 计算并筛选
pairs = []
cnt = 0
for i, j in combinations(range(len(cameras)), 2):
    cnt = cnt + 1
    cam1 = cameras[i]
    cam2 = cameras[j]

    # 计算夹角
    dot_product = np.clip(np.dot(cam1['view_dir'], cam2['view_dir']), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_product))

    if angle_deg < ANGLE_THRESHOLD:
        pairs.append({
            'img1': cam1['filename'],
            'img2': cam2['filename'],
            'angle': angle_deg
        })
print(f"共计算 {cnt}次")
# 按相似度排序（角度越小越相似）
pairs.sort(key=lambda x: x['angle'])

# 6. 保存为 TXT 文件
output_file = 'image_pairs.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    # 写入文件头
    f.write(f"Total Pairs Found: {len(pairs)} (Threshold: {ANGLE_THRESHOLD} degrees)\n")
    f.write(f"{'Image 1':<20} {'Image 2':<20} {'Angle Difference'}\n")
    f.write("-" * 60 + "\n")

    # 写入数据
    for p in pairs:
        f.write(f"{p['img1']:<20} {p['img2']:<20} {p['angle']:.2f}\n")

print(f"成功！结果已保存至: {output_file}")