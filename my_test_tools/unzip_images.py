import os
import zipfile
import shutil

def process_zips_absolute():
    # ================= 配置区域 =================
    # 1. 设置工作的绝对根目录
    base_dir = "/root/autodl-tmp"

    # 2. 设置解压的目标目录 (在根目录下的 images 文件夹)
    target_base_dir = os.path.join(base_dir, "images")
    # ===========================================

    # 如果目标 images 文件夹不存在，则创建
    if not os.path.exists(target_base_dir):
        os.makedirs(target_base_dir)
        print(f"创建目标文件夹: {target_base_dir}")

    # 获取指定目录下的所有文件
    if not os.path.exists(base_dir):
        print(f"错误: 找不到目录 {base_dir}")
        return

    files = os.listdir(base_dir)

    # 定义文件前缀和后缀
    prefix = "fsmview_trainset_images_"
    suffix = ".zip"

    print(f"--- 开始扫描目录: {base_dir} ---")

    count = 0
    for filename in files:
        # 筛选条件
        if filename.startswith(prefix) and filename.endswith(suffix):
            count += 1

            # 拼接压缩包的完整绝对路径
            full_zip_path = os.path.join(base_dir, filename)

            # 提取ID (文件名处理逻辑不变)
            folder_id = filename[len(prefix):-4]

            # 定义最终的目标路径 (例如 /root/autodl-tmp/images/341-359)
            final_dest_path = os.path.join(target_base_dir, folder_id)

            # 如果目标文件夹已存在，跳过
            if os.path.exists(final_dest_path):
                print(f"[跳过] {folder_id} 已存在于 images 中")
                continue

            print(f"正在处理: {filename} \n   -> 目标: {final_dest_path}")

            try:
                # 在 base_dir 下创建一个临时文件夹，确保在同一个磁盘分区，移动速度快
                temp_extract_dir = os.path.join(base_dir, f"temp_{folder_id}")

                # 解压
                with zipfile.ZipFile(full_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)

                # 解压出来的内部文件夹完整路径
                extracted_inner_folder = os.path.join(temp_extract_dir, "fsmview_trainset")

                # 检查并移动
                if os.path.exists(extracted_inner_folder):
                    # 移动并重命名
                    shutil.move(extracted_inner_folder, final_dest_path)
                    print(f"   Success! 完成。")
                else:
                    print(f"   Error: 解压后未发现 'fsmview_trainset' 文件夹。")

                # 清理临时目录
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)

            except zipfile.BadZipFile:
                print(f"   Error: {filename} 文件损坏。")
            except Exception as e:
                print(f"   Error: 未知错误: {e}")

    if count == 0:
        print("未找到符合条件的压缩包，请检查路径或文件名。")
    else:
        print("--- 全部处理完毕 ---")

if __name__ == "__main__":
    process_zips_absolute()