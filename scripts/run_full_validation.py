#!/usr/bin/env python3
"""
完整的数据验证和评估流程
一键执行数据质量验证和模型效果评估
"""

import os
import sys
import json
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

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*50)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/home/disk/q3s2/data_generation_pipeline/scripts')

        if result.stdout:
            print("输出:")
            print(result.stdout)

        if result.stderr:
            print("错误输出:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} 成功完成")
            return True
        else:
            print(f"❌ {description} 失败 (退出码: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def check_prerequisites():
    """检查前提条件"""
    print("检查前提条件...")

    # 检查输出文件是否存在
    output_dir = '/home/disk/q3s2/data_generation_pipeline/output'
    required_files = [
        'final_combined_dataset.json',
        'cross_modal_samples.json',
        'counterfactual_samples.json',
        'semantic_boundary_samples.json',
        'red_blue_samples.json',
        'optimized_samples.json'
    ]

    missing_files = []
    for filename in required_files:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)

    if missing_files:
        print(f"⚠️  缺少以下文件: {', '.join(missing_files)}")
        print("请先运行数据生成流程: python scripts/00_main.py")
        return False

    print("✅ 数据文件检查通过")
    return True

def run_data_validation():
    """运行数据质量验证"""
    return run_command(
        "python validate_dataset.py",
        "数据质量验证"
    )

def run_model_evaluation():
    """运行模型效果评估"""
    return run_command(
        "python evaluate_model.py",
        "模型效果评估"
    )

def generate_comprehensive_report():
    """生成综合报告"""
    print(f"\n{'='*50}")
    print("生成综合验证报告")
    print('='*50)

    output_dir = '/home/disk/q3s2/data_generation_pipeline/output'

    # 读取各个验证结果
    reports = {}

    # 数据验证报告
    validation_report_path = os.path.join(output_dir, 'validation_report.json')
    if os.path.exists(validation_report_path):
        with open(validation_report_path, 'r', encoding='utf-8') as f:
            reports['data_validation'] = json.load(f)

    # 模型评估报告
    eval_report_path = os.path.join(output_dir, 'model_evaluation_results.json')
    if os.path.exists(eval_report_path):
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            reports['model_evaluation'] = json.load(f)

    # 生成综合报告
    comprehensive_report = {
        'timestamp': '2026-05-12',
        'pipeline_version': '1.0',
        'reports': reports,
        'summary': {
            'data_quality_score': None,
            'model_performance_score': None,
            'overall_recommendations': []
        }
    }

    # 计算数据质量评分
    if 'data_validation' in reports:
        dv = reports['data_validation']
        quality_score = 0

        # 格式完整性评分
        completeness = dv['quality_metrics']['format_completeness']
        if completeness['conversation_fields'].split('/')[0] == completeness['conversation_fields'].split('/')[1]:
            quality_score += 2

        # 多样性评分
        diversity = dv['quality_metrics']['question_diversity']
        dup_rate = float(diversity['duplication_rate'].rstrip('%'))
        if dup_rate < 10:
            quality_score += 2
        elif dup_rate < 20:
            quality_score += 1

        # 类型分布评分
        type_dist = dv['quality_metrics']['type_distribution']
        if len(type_dist) >= 4:
            quality_score += 1

        comprehensive_report['summary']['data_quality_score'] = quality_score

    # 计算模型性能评分
    if 'model_evaluation' in reports:
        me = reports['model_evaluation']
        perf_score = me['summary']['average_score']
        comprehensive_report['summary']['model_performance_score'] = perf_score

    # 生成建议
    recommendations = []

    if comprehensive_report['summary']['data_quality_score'] is not None:
        score = comprehensive_report['summary']['data_quality_score']
        if score < 3:
            recommendations.append("数据质量需要改进，建议重新生成或优化数据")
        else:
            recommendations.append("数据质量良好，可以用于模型微调")

    if comprehensive_report['summary']['model_performance_score'] is not None:
        score = comprehensive_report['summary']['model_performance_score']
        if score < 0.5:
            recommendations.append("模型性能不佳，建议调整微调参数或增加训练数据")
        elif score < 0.8:
            recommendations.append("模型性能一般，可以尝试进一步优化")
        else:
            recommendations.append("模型性能优秀！")

    comprehensive_report['summary']['overall_recommendations'] = recommendations

    # 保存综合报告
    report_path = os.path.join(output_dir, 'comprehensive_validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)

    print("综合报告内容:")
    print(f"- 数据质量评分: {comprehensive_report['summary']['data_quality_score']}/5")
    if comprehensive_report['summary']['model_performance_score'] is not None:
        print(f"- 模型性能评分: {comprehensive_report['summary']['model_performance_score']:.2f}/1.0")
    else:
        print("- 模型性能评分: 未评估 (评估失败)")

    print("\n建议:")
    for rec in recommendations:
        print(f"- {rec}")

    print(f"\n详细报告已保存至: {report_path}")

    return True

def main():
    """主函数"""
    print("🚀 数据生成流程验证系统")
    print("=" * 60)

    # 检查前提条件
    if not check_prerequisites():
        print("\n❌ 前提条件不满足，请检查后重试")
        sys.exit(1)

    # vLLM服务控制变量
    vllm_started_by_us = False

    try:
        # 检查是否已有vLLM服务运行
        if not check_vllm_service():
            print("\n启动vLLM服务...")
            if not start_vllm():
                print("❌ vLLM服务启动失败")
                sys.exit(1)
            vllm_started_by_us = True
        else:
            print("✓ 检测到 vLLM 服务已在后台独立运行，直接接入。")

        success_count = 0
        total_steps = 3

        # 步骤1: 数据质量验证
        if run_data_validation():
            success_count += 1

        # 步骤2: 模型效果评估
        if run_model_evaluation():
            success_count += 1

        # 步骤3: 生成综合报告
        if generate_comprehensive_report():
            success_count += 1

        print(f"\n{'='*60}")
        print(f"验证流程完成: {success_count}/{total_steps} 个步骤成功")

        if success_count == total_steps:
            print("🎉 所有验证步骤均成功完成！")
            print("\n下一步建议:")
            print("1. 查看 output/ 目录下的验证报告")
            print("2. 如果数据质量和模型性能满意，可以进行大规模微调")
            print("3. 如果效果不佳，可以调整数据生成参数或重新生成数据")
        else:
            print("⚠️  部分验证步骤失败，请检查上述错误信息")

    except Exception as e:
        print(f"\n❌ 验证流程遭遇异常中断: {str(e)}")
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

if __name__ == '__main__':
    main()