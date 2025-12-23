import os
import zipfile
import shutil

def process_shapes_space_saving():
    # ================= 配置区域 =================
    # 1. 工作根目录
    base_dir = "/root/autodl-tmp"

    # 2. 目标存放目录 (改为 shapes)
    target_base_dir = os.path.join(base_dir, "shapes")

    # 3. 文件前缀 (根据你的 ls 记录，这里是 shape 而不是 images)
    prefix = "fsmview_trainset_shape_"
    suffix = ".zip"
    # ===========================================

    # 如果 shapes 文件夹不存在，则创建
    if not os.path.exists(target_base_dir):
        os.makedirs(target_base_dir)
        print(f"已创建目标文件夹: {target_base_dir}")

    # 检查根目录
    if not os.path.exists(base_dir):
        print(f"错误: 找不到目录 {base_dir}")
        return

    files = os.listdir(base_dir)
    print(f"--- 开始处理 Shapes (模式: 解压成功即删除原包) ---")

    count = 0
    for filename in files:
        # 筛选条件
        if filename.startswith(prefix) and filename.endswith(suffix):
            count += 1

            # 获取文件的绝对路径
            full_zip_path = os.path.join(base_dir, filename)

            # 提取ID (例如把 "fsmview_trainset_shape_041-060.zip" 变成 "041-060")
            folder_id = filename[len(prefix):-4]

            # 最终存放路径: /root/autodl-tmp/shapes/041-060
            final_dest_path = os.path.join(target_base_dir, folder_id)

            # 检查是否已存在
            if os.path.exists(final_dest_path):
                print(f"[跳过] {folder_id} 已存在，为了安全不删除原压缩包。")
                continue

            print(f"正在处理: {filename}")

            # 临时解压目录
            temp_extract_dir = os.path.join(base_dir, f"temp_shape_{folder_id}")

            try:
                # 1. 解压
                with zipfile.ZipFile(full_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)

                # 2. 定位解压后的内部文件夹
                # 假设内部依然叫 "fsmview_trainset"，如果shape包内部名字不一样，脚本会报错提示
                extracted_inner_folder = os.path.join(temp_extract_dir, "fsmview_trainset")

                # 3. 移动并重命名
                if os.path.exists(extracted_inner_folder):
                    shutil.move(extracted_inner_folder, final_dest_path)
                    print(f"   [成功] 已解压并移动到: {final_dest_path}")

                    # === 核心逻辑：删除原压缩包 ===
                    os.remove(full_zip_path)
                    print(f"   [删除] 原文件 {filename} 已删除，释放空间。")
                    # ==========================

                else:
                    # 如果解压出来找不到 fsmview_trainset 文件夹，可能是压缩包内部结构变了
                    # 此时千万不能删原文件
                    print(f"   [警告] 解压成功，但未找到 'fsmview_trainset' 文件夹。")
                    print(f"   可能是压缩包内部结构不同，请手动检查: {temp_extract_dir}")
                    print(f"   原压缩包未删除。")
                    continue # 跳过清理步骤，保留现场

                # 4. 清理临时空文件夹
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)

            except zipfile.BadZipFile:
                print(f"   [错误] {filename} 文件损坏，跳过。")
            except Exception as e:
                print(f"   [错误] 处理 {filename} 时发生未知错误: {e}")

    if count == 0:
        print("未找到符合前缀 fsmview_trainset_shape_ 的压缩包。")
    else:
        print("--- 全部处理完毕 ---")

if __name__ == "__main__":
    process_shapes_space_saving()