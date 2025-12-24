import os
import sys
import runpy

def main():
    # 1. 模拟 torchrun 设置分布式环境变量 (单机单卡模式)
    # 对应 --nproc_per_node=1
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",  # 任意空闲端口
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "CUDA_VISIBLE_DEVICES": "0" # 指定使用第一块 GPU
    })

    # 2. 模拟命令行参数
    # 对应 train.py @config_facescape.txt
    # 注意：sys.argv[0] 通常是脚本名，这里为了欺骗 argparse，我们填 train.py
    sys.argv = ["train.py", "@config_facescape.txt"]

    print(f">> Launcher: Starting training with config: {sys.argv[1]}")
    print(f">> Launcher: Environment set to Single-GPU mode (RANK=0, WORLD_SIZE=1)")

    # 3. 执行 train.py
    # runpy 会在当前进程中执行目标文件，就像你在命令行运行它一样
    # 这样 PyCharm 的调试器（Debugger）可以无缝断点调试 train.py 里的代码
    try:
        runpy.run_path("train.py", run_name="__main__")
    except Exception as e:
        print(f">> Launcher: Execution failed: {e}")
        raise e

if __name__ == "__main__":
    main()