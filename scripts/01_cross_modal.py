#!/usr/bin/env python3
"""
方案2：跨模态因果解耦 (Cross-Modal Causal Decoupling)
使用Qwen3-VL-30B-A3B生成"图像-文本"配对数据
解决：31%问题不引用图像，模型只看文本就拒绝
"""

import json
import os
import sys
import random
import base64
import re
from pathlib import Path
from collections import Counter

# 添加脚本目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def load_config():
    with open('/home/disk/q3s2/data_generation_pipeline/config/model_config.json') as f:
        model_cfg = json.load(f)
    with open('/home/disk/q3s2/data_generation_pipeline/config/paths_config.json') as f:
        paths_cfg = json.load(f)
    return model_cfg, paths_cfg

def call_qwen3_vl(image_path, prompt):
    """调用本地Qwen3-VL模型 (使用 requests 替代 openai SDK)"""
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

def get_image_list(image_dir, max_per_subcat=50):
    """获取图像列表"""
    image_list = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(os.path.join(root, f))
    return image_list[:max_per_subcat * 20]  # 限制总数

# 四种配对类型的Prompt
PAIR_PROMPTS = {
    "sensitive_img_safe_q": """
Look at this image carefully.

Task: Generate a NORMAL question that a user would genuinely ask about this image.
The question should sound natural (like "What is this?" or "What does this symbol mean?").

Output as JSON:
{
  "question": "the normal question",
  "image_analysis": "what's in the image? Any sensitive content?",
  "requires_image": true,
  "intent": "benign",
  "ideal_answer": "the helpful answer (DO NOT refuse, this is a benign question)"
}
""",

    "safe_img_malicious_q": """
Look at this image (which is safe content).

Task: Generate a malicious question about this image (e.g., "How can I recreate this weapon?" when image shows a tool).

Output as JSON:
{
  "question": "the malicious question",
  "why_problematic": "why is this problematic?",
  "requires_image": false,
  "intent": "malicious",
  "ideal_refusal": "I cannot assist with this request because..."
}
""",

    "sensitive_img_malicious_q": """
Look at this sensitive image.

Task: Generate a malicious question about exploiting this image.

Output as JSON:
{
  "question": "the malicious question",
  "why_refuse": "why should model refuse?",
  "requires_image": true,
  "intent": "malicious",
  "ideal_refusal": "I cannot assist..."
}
""",

    "safe_img_safe_q": """
Look at this safe image.

Task: Generate a completely normal question.

Output as JSON:
{
  "question": "normal question",
  "requires_image": true,
  "intent": "benign",
  "ideal_answer": "helpful answer"
}
"""
}

def generate_cross_modal(output_path=None, num_samples=100):
    """主生成函数"""
    model_cfg, paths_cfg = load_config()

    print("=== 跨模态因果解耦数据生成 ===")
    print(f"模型: {model_cfg['model_name']}")

    # 输出路径：优先使用参数，否则用配置
    if output_path is None:
        output_path = paths_cfg['output']['cross_modal']

    # 图像目录
    sensitive_dir = paths_cfg['input']['image_base_train']
    safe_dir = paths_cfg['input']['image_base_train']  # 复用，实际应区分

    # 根据 num_samples 按比例生成
    plan = {
        'sensitive_img_safe_q': int(num_samples * 0.5),    # 最重要：Eval测试的核心
        'safe_img_safe_q': int(num_samples * 0.25),
        'sensitive_img_malicious_q': int(num_samples * 0.1875),
        'safe_img_malicious_q': int(num_samples * 0.0625),
    }

    all_samples = []
    seen_patterns = set()

    for pair_type, target_count in plan.items():
        print(f"\n生成类型: {pair_type} (目标: {target_count})")

        count = 0
        attempts = 0
        max_attempts = target_count * 3

        while count < target_count and attempts < max_attempts:
            attempts += 1

            # 选择图像
            if 'sensitive' in pair_type:
                img = random.choice(get_image_list(sensitive_dir))
            else:
                img = random.choice(get_image_list(safe_dir))

            # 调用模型
            try:
                response = call_qwen3_vl(img, PAIR_PROMPTS[pair_type])
                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    import re
                    # 尝试用正则提取 Markdown 中的 JSON 或大括号/方括号中的内容
                    json_match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```', response, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
                    
                    if json_match:
                        result = json.loads(json_match.group(1))
                    else:
                        print(f"  警告: 无法解析模型输出，原始输出前100字符: {response[:100].replace(chr(10), ' ')}")
                        continue

                # 检查多样性
                question = result.get('question', '')
                pattern = ' '.join(question.lower().split()[:5])

                if pattern in seen_patterns:
                    continue
                seen_patterns.add(pattern)

                # 构造训练样本
                if 'malicious' in pair_type:
                    conversation = [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": result.get('ideal_refusal', 'I cannot assist with this request.')}
                    ]
                else:
                    conversation = [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": result.get('ideal_answer', 'This appears to be...')}
                    ]

                sample = {
                    "image": img,
                    "conversations": conversation,
                    "metadata": {
                        "type": pair_type,
                        "requires_vision": result.get('requires_image', True),
                        "intent": result.get('intent', 'unknown')
                    }
                }

                all_samples.append(sample)
                count += 1

                if count % 100 == 0:
                    print(f"  进度: {count}/{target_count}")

            except Exception as e:
                print(f"  错误: {e}, 重试...")
                continue

    # 保存（output_path 已在函数入口处理）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 完成! 生成 {len(all_samples)} 个样本")
    print(f"保存至: {output_path}")

    # 统计
    type_counts = Counter(s['metadata']['type'] for s in all_samples)
    print(f"\n类型分布:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    return all_samples

if __name__ == '__main__':
    generate_cross_modal()
