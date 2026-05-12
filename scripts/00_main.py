#!/usr/bin/env python3
"""
主流程控制器：协调5个创新方案的数据生成
使用本地Qwen3-VL-30B-A3B-Instruct-FP8模型
vLLM服务在脚本内部控制启动和关闭
"""

import json
import os
import sys
import subprocess
import time
from pathlib import Path

# 加载配置
with open('/home/disk/q3s2/data_generation_pipeline/config/model_config.json') as f:
    MODEL_CONFIG = json.load(f)

with open('/home/disk/q3s2/data_generation_pipeline/config/paths_config.json') as f:
    PATHS_CONFIG = json.load(f)

# vLLM服务控制全局变量
VLLM_PROCESS = None
VLLM_LOG_FILE = None

# 导入各方案模块（使用 __import__ 支持数字开头模块名）
def import_modules():
    """动态导入各方案模块，使用 __import__ 支持数字开头模块名"""
    global generate_cross_modal, generate_counterfactual, generate_semantic_boundary, generate_red_blue, optimize_dataset

    try:
        # 确保脚本目录在路径中
        script_dir = str(Path(__file__).parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # 使用 __import__ 导入数字开头的模块
        # 方案2：跨模态因果解耦
        mod2 = __import__('01_cross_modal')
        generate_cross_modal = mod2.generate_cross_modal

        # 方案3：元反事实推理链
        mod3 = __import__('02_counterfactual')
        generate_counterfactual = mod3.generate_counterfactual

        # 方案1：语义拓扑边界搜索
        mod1 = __import__('03_semantic_boundary')
        generate_semantic_boundary = mod1.generate_semantic_boundary

        # 方案4：多智能体红蓝对抗
        mod4 = __import__('04_red_blue')
        generate_red_blue = mod4.generate_red_blue

        # 方案5：信息增益优化
        mod5 = __import__('05_optimize')
        optimize_dataset = mod5.optimize_dataset

        print("✓ 所有方案模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入模块失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_vllm():
    """启动vLLM服务"""
    global VLLM_PROCESS, VLLM_LOG_FILE

    model_path = MODEL_CONFIG['model_path']
    print(f"启动vLLM服务: {MODEL_CONFIG['model_name']}")
    print(f"模型路径: {model_path}")

    # ================= 新增：指定运行显卡 =================
    # 优先从配置文件读取，如果没有则默认使用 "0" 号显卡
    # 你可以修改配置文件，或者直接在这里写死，例如 target_gpus = "0,1"
    target_gpus = MODEL_CONFIG.get('cuda_visible_devices', '1')
    gpu_count = len(str(target_gpus).split(','))
    
    print(f"指定运行显卡 (CUDA_VISIBLE_DEVICES): {target_gpus}")
    print(f"使用的GPU数量 (Tensor Parallel Size): {gpu_count}")
    # ======================================================

    # 使用conda的vllm环境
    vllm_python = "/root/anaconda3/envs/vllm/bin/python"

    cmd = [
        vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--dtype", "auto",
        "--max-model-len", "8192",  # 核心修复1：限制上下文长度，防止KV Cache OOM
        "--port", "8000",  # 强制使用8000端口
        "--served-model-name", MODEL_CONFIG['model_name']  # 与配置及各模块中的model字段保持一致
    ]

    # 如果使用了多张显卡，必须告诉 vLLM 开启张量并行
    if gpu_count > 1:
        cmd.extend(["--tensor-parallel-size", str(gpu_count)])

    print(f"命令: {' '.join(cmd)}")

    # 核心修复2：注入离线环境变量，彻底切断Hugging Face的联网重试
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    
    # 将指定显卡的环境变量写入到子进程的环境中
    env["CUDA_VISIBLE_DEVICES"] = str(target_gpus)

    # 核心修复3：将日志写入文件而不是留给管道，防止缓冲区打满导致死锁卡死
    log_path = "vllm_startup.log"
    VLLM_LOG_FILE = open(log_path, "w")

    VLLM_PROCESS = subprocess.Popen(
        cmd,
        env=env,  # 这里的 env 现在包含了 CUDA_VISIBLE_DEVICES
        stdout=VLLM_LOG_FILE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    # 等待服务启动 (增加超时时间，30B模型加载和编译需要时间)
    print(f"等待vLLM服务启动... (详细加载日志请开新终端查看: tail -f {log_path})")
    for _ in range(120):  
        try:
            import urllib.request
            req = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
            if req.status == 200:
                print("✓ vLLM服务已启动并就绪")
                return True
        except:
            time.sleep(2) # 每2秒查一次

    print("✗ vLLM服务启动超时，请检查 vllm_startup.log 确定报错原因")
    return False

def stop_vllm():
    """停止vLLM服务"""
    global VLLM_PROCESS, VLLM_LOG_FILE
    if VLLM_PROCESS:
        print("清理并停止vLLM服务...")
        VLLM_PROCESS.terminate()
        try:
            VLLM_PROCESS.wait(timeout=10) # 给10秒钟优雅关闭
        except subprocess.TimeoutExpired:
            print("优雅关闭超时，强制终止vLLM服务...")
            VLLM_PROCESS.kill()
        print("✓ vLLM服务已安全停止，显存已释放")
        
    if VLLM_LOG_FILE and not VLLM_LOG_FILE.closed:
        VLLM_LOG_FILE.close()

def print_banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)

def main():
    # 导入模块
    if not import_modules():
        sys.exit(1)

    print_banner("数据生成流程启动")
    print(f"模型: {MODEL_CONFIG['model_name']}")
    print(f"激活参数: {MODEL_CONFIG.get('model_type', 'N/A')}")

    # 核心修复4：判断是谁启动了服务，并确保程序按顺序执行
    vllm_started_by_us = False
    if not check_vllm_service():
        print("检测到 vLLM API 未响应，准备自动拉起后端服务...")
        if not start_vllm():
            print("错误: vLLM服务拉起失败！流程终止。")
            sys.exit(1)
        vllm_started_by_us = True
    else:
        print("✓ 检测到 vLLM 服务已在后台独立运行，直接接入。")

    # 核心修复5：使用 try-finally 块，确保发生任何报错都会释放显存
    try:
        # 阶段1: 跨模态因果解耦（最优先）
        print_banner("阶段1: 跨模态因果解耦 (方案2)")
        cross_modal_path = PATHS_CONFIG['output']['cross_modal']
        if os.path.exists(cross_modal_path):
            print(f"✓ 检测到已存在: {cross_modal_path}，跳过生成")
            with open(cross_modal_path, 'r') as f:
                cross_modal_data = json.load(f)
        else:
            cross_modal_data = generate_cross_modal(
                output_path=cross_modal_path,
                num_samples=100
            )
        print(f"✓ 跨模态样本: {len(cross_modal_data)}")

        # 阶段2: 元反事实推理链
        print_banner("阶段2: 元反事实推理链 (方案3)")
        counterfactual_path = PATHS_CONFIG['output']['counterfactual']
        if os.path.exists(counterfactual_path):
            print(f"✓ 检测到已存在: {counterfactual_path}，跳过生成")
            with open(counterfactual_path, 'r') as f:
                counterfactual_data = json.load(f)
        else:
            counterfactual_data = generate_counterfactual(
                output_path=counterfactual_path,
                num_samples=100
            )
        print(f"✓ 反事实样本: {len(counterfactual_data)}")

        # 阶段3: 信息增益优化（过滤旧数据）
        print_banner("阶段3: 信息增益优化 (方案5)")
        optimized_path = PATHS_CONFIG['output']['optimized']
        if os.path.exists(optimized_path):
            print(f"✓ 检测到已存在: {optimized_path}，跳过生成")
            with open(optimized_path, 'r') as f:
                optimized_data = json.load(f)
        else:
            optimized_data = optimize_dataset(
                input_path=PATHS_CONFIG['input']['train_data'],
                output_path=optimized_path,
                target_size=100
            )
        print(f"✓ 优化后样本: {len(optimized_data)}")

        # 阶段4: 语义拓扑边界搜索
        print_banner("阶段4: 语义拓扑边界搜索 (方案1)")
        boundary_path = PATHS_CONFIG['output']['semantic_boundary']
        if os.path.exists(boundary_path):
            print(f"✓ 检测到已存在: {boundary_path}，跳过生成")
            with open(boundary_path, 'r') as f:
                boundary_data = json.load(f)
        else:
            boundary_data = generate_semantic_boundary(
                output_path=boundary_path,
                num_samples=100
            )
        print(f"✓ 边界样本: {len(boundary_data)}")

        # 阶段5: 多智能体红蓝对抗
        print_banner("阶段5: 多智能体红蓝对抗 (方案4)")
        red_blue_path = PATHS_CONFIG['output']['red_blue']
        if os.path.exists(red_blue_path):
            print(f"✓ 检测到已存在: {red_blue_path}，跳过生成")
            with open(red_blue_path, 'r') as f:
                red_blue_data = json.load(f)
        else:
            red_blue_data = generate_red_blue(
                output_path=red_blue_path,
                num_samples=100
            )
        print(f"✓ 对抗样本: {len(red_blue_data)}")

        # 合并所有数据
        print_banner("合并所有生成数据")
        all_data = []
        all_data.extend(cross_modal_data)
        all_data.extend(counterfactual_data)
        all_data.extend(optimized_data)
        all_data.extend(boundary_data)
        all_data.extend(red_blue_data)

        final_path = PATHS_CONFIG['output']['final_combined']
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 总计样本: {len(all_data)}")
        print(f"✓ 保存至: {final_path}")

        # 验证
        print_banner("验证生成数据")
        validate_dataset(final_path)

        print_banner("数据生成流程顺利完成!")
        print(f"下一步: 使用生成的数据微调模型")

    except Exception as e:
        print(f"\n❌ 数据生成流程遭遇异常中断: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 无论流程是成功跑完，还是因为代码报错中途崩溃，这行代码一定会执行
        if vllm_started_by_us:
            stop_vllm()

def check_vllm_service():
    """检查vLLM服务是否运行 (使用轻量级 /v1/models 端点)"""
    import urllib.request

    try:
        url = "http://localhost:8000/v1/models"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False

def validate_dataset(path):
    """验证生成的数据格式"""
    with open(path) as f:
        data = json.load(f)

    print(f"总样本数: {len(data)}")

    # 检查格式
    has_image = sum(1 for d in data if 'image' in d or 'images' in d)
    has_conv = sum(1 for d in data if 'conversations' in d)
    print(f"包含图像: {has_image}/{len(data)}")
    print(f"包含对话: {has_conv}/{len(data)}")

    # 检查子类分布
    from collections import Counter
    subcats = Counter()
    for d in data:
        if 'conversations' in d:
            subcats[d.get('subcategory', 'unknown')] += 1
        elif 'subcategory' in d:
            subcats[d['subcategory']] += 1

    print(f"\n子类分布 (Top 10):")
    for subcat, cnt in subcats.most_common(10):
        print(f"  {subcat}: {cnt}")

if __name__ == '__main__':
    main()