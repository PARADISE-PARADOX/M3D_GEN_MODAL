#!/usr/bin/env python3
"""
方案1：语义拓扑边界搜索 (Semantic Topology Boundary Search)
使用Qwen3-VL找到模型"最不确定"的区域生成数据
解决：只会模板匹配，不会真正理解
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def load_config():
    with open('/home/disk/q3s2/data_generation_pipeline/config/model_config.json') as f:
        model_cfg = json.load(f)
    with open('/home/disk/q3s2/data_generation_pipeline/config/paths_config.json') as f:
        paths_cfg = json.load(f)
    return model_cfg, paths_cfg

def call_qwen3_vl(image_path, prompt):
    """调用本地Qwen3-VL模型 (使用 requests 替代 openai SDK)"""
    import base64
    import requests

    model_cfg, _ = load_config()

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model_cfg['model_name'],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        "temperature": 0.7,
        "max_tokens": 2048
    }

    resp = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

BOUNDARY_PROMPT = """
Look at this image carefully.

Step 1: Describe what sensitive content exists in this image.
Step 2: Generate a question that is AMBIGUOUS - not clearly malicious,
        but not clearly benign either. Something where the intent is unclear.
Step 3: Based on the image and question, explain:
        - What makes this query potentially problematic?
        - What makes it potentially acceptable?
        - Your final decision: should this be answered or refused?

Output format:
{
  "image_sensitivity": "...",
  "ambiguous_question": "...",
  "why_problematic": "...",
  "why_acceptable": "...",
  "decision": "answer/refuse",
  "reasoning": "..."
}
"""

def get_image_list(image_dir, max_per_subcat=30):
    """获取图像列表"""
    import os
    image_list = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(os.path.join(root, f))
    return image_list[:max_per_subcat * 20]

def generate_semantic_boundary(output_path=None, num_samples=100):
    """主生成函数"""
    model_cfg, paths_cfg = load_config()

    print("=== 语义拓扑边界搜索 ===")
    print(f"模型: {model_cfg['model_name']}")

    # 输出路径：优先使用参数，否则用配置
    if output_path is None:
        output_path = paths_cfg['output']['semantic_boundary']

    # 获取图像
    image_dir = paths_cfg['input']['image_base_train']
    images = get_image_list(image_dir)
    print(f"找到 {len(images)} 张图像")

    all_samples = []
    seen_patterns = set()

    for idx, img in enumerate(images[:num_samples]):
        print(f"\r进度: {idx+1}/{num_samples}", end='', flush=True)

        try:
            response = call_qwen3_vl(img, BOUNDARY_PROMPT)
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```', response, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
                
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    print(f"\n警告: 无法解析模型输出，原始输出前100字符: {response[:100].replace(chr(10), ' ')}")
                    continue

            # 检查多样性
            question = result.get('ambiguous_question', '')
            pattern = ' '.join(question.lower().split()[:5])

            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)

            # 构造训练样本
            sample = {
                "image": img,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": f"Reasoning: {result.get('reasoning', '')}\n\nFinal: " +
                                   ("I cannot assist with this request because " + result.get('why_problematic', '')
                                    if result.get('decision') == 'refuse'
                                    else result.get('why_acceptable', 'This appears to be...'))
                    }
                ],
                "metadata": {
                    "type": "boundary_sample",
                    "decision": result.get('decision', 'unknown')
                }
            }

            all_samples.append(sample)

        except Exception as e:
            print(f"\n错误 {img}: {e}")
            continue

    # 保存（output_path 已在函数入口处理）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 完成! 生成 {len(all_samples)} 个边界样本")
    print(f"保存至: {output_path}")

    return all_samples

if __name__ == '__main__':
    generate_semantic_boundary()
