import json
import os
import sys
from pathlib import Path

# 引入项目路径
sys.path.insert(0, str(Path(__file__).parent))
try:
    from 01_cross_modal import call_qwen3_vl, PAIR_PROMPTS
except ImportError:
    # 如果 01_cross_modal 无法直接导入，尝试更稳妥的方式
    import importlib.util
    spec = importlib.util.spec_from_file_location("cross_modal", "/home/disk/q3s2/data_generation_pipeline/scripts/01_cross_modal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    call_qwen3_vl = mod.call_qwen3_vl
    PAIR_PROMPTS = mod.PAIR_PROMPTS

def run_quick_test():
    # 加载刚提取的测试子集
    subset_path = '/home/disk/q3s2/data_generation_pipeline/test_input_subset.json'
    if not os.path.exists(subset_path):
        print(f"错误: 找不到测试子集文件 {subset_path}")
        return

    with open(subset_path, 'r') as f:
        subset = json.load(f)
    
    test_samples = subset[:5] # 只取前5个进行快速测试
    results = []
    
    print(f"开始快速测试... 目标: 生成 5 条跨模态解耦数据")
    
    for item in test_samples:
        img_path = item['images'][0]
        subcat = item['subcategory']
        print(f"正在处理图片: {os.path.basename(img_path)} (子类: {subcat})")
        
        try:
            # 使用“敏感图像 + 安全问题”方案进行测试
            response = call_qwen3_vl(img_path, PAIR_PROMPTS['sensitive_img_safe_q'])
            
            # 尝试解析 JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # 处理模型输出带有 Markdown 代码块的情况
                import re
                clean_response = re.search(r'\{.*\}', response, re.DOTALL)
                if clean_response:
                    result = json.loads(clean_response.group())
                else:
                    raise ValueError(f"无法从模型输出中解析JSON: {response[:100]}...")

            new_sample = {
                "original_id": item['id'],
                "new_question": result.get('question', 'N/A'),
                "ideal_answer": result.get('ideal_answer', 'N/A'),
                "reasoning": result.get('image_analysis', 'N/A')
            }
            results.append(new_sample)
            print(f"  ✓ 成功生成新问题: {new_sample['new_question'][:50]}...")
            
        except Exception as e:
            print(f"  ✗ 出错: {e}")

    # 保存测试结果
    output_dir = '/home/disk/q3s2/data_generation_pipeline/output'
    os.makedirs(output_dir, exist_ok=True)
    output_p = os.path.join(output_dir, 'quick_test_output.json')
    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试完成！结果已保存至: {output_p}")

if __name__ == "__main__":
    run_quick_test()
