#!/usr/bin/env python3
"""
方案5：信息增益子集优化 (Information Gain Subset Optimization)
不用全量14,387条，而是用模型预测每个样本的"信息增益"，只保留最有价值的样本
解决：38.1%的问题完全重复，大量样本提供的信息是冗余的
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

def calculate_information_gain(sample, model):
    """
    计算每个样本的信息增益：
    - 高增益：模型当前会答错（过拒绝或漏拒绝）
    - 低增益：模型已经学会，再训练也没用
    """
    # 这里简化为：检查样本是否包含高价值特征
    question = ""
    if 'conversations' in sample:
        for c in sample['conversations']:
            if c['from'] == 'human':
                question = c['value'].replace('<image>', '').strip()
                break
    elif 'query' in sample:
        question = sample['query']

    # 简单启发式：包含边界特征的样本更有价值
    high_value_keywords = ['ambiguous', 'boundary', 'unclear', 'help me redact', 'protect', 'safe']
    score = sum(1 for kw in high_value_keywords if kw in question.lower())

    # 检查是否重复（重复的样本价值低）
    pattern = ' '.join(question.lower().split()[:5])
    return score, pattern

def optimize_dataset(input_path=None, output_path=None, target_size=5000):
    """主优化函数"""
    model_cfg, paths_cfg = load_config()

    print("=== 信息增益优化 ===")
    print(f"模型: {model_cfg['model_name']}")

    # 输出路径：优先使用参数，否则用配置
    if input_path is None:
        input_path = paths_cfg['input']['train_data']
    if output_path is None:
        output_path = paths_cfg['output']['optimized']

    # 加载原始训练数据
    with open(input_path, 'r') as f:
        train_data = json.load(f)

    print(f"原始数据: {len(train_data)} 条")

    # 计算每个样本的信息增益
    gains = []
    seen_patterns = set()

    for idx, sample in enumerate(train_data):
        if idx % 1000 == 0:
            print(f"  处理进度: {idx}/{len(train_data)}")

        gain, pattern = calculate_information_gain(sample, None)

        if pattern not in seen_patterns:
            gains.append((sample, gain, pattern))
            seen_patterns.add(pattern)

    # 按增益排序
    gains.sort(key=lambda x: -x[1])

    # 选择top样本
    selected = gains[:target_size]

    print(f"\n✓ 优化后样本: {len(selected)} (从 {len(train_data)} 中筛选)")

    # 可能还需要用Qwen3-VL生成一些新的高价值样本
    print("\n生成新的高价值样本...")
    new_samples = []

    # 获取图像列表
    image_dir = paths_cfg['input']['image_base_train']
    import os
    image_list = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(os.path.join(root, f))

    # 生成1000个新样本（基于高错误样本的类型）
    for i in range(min(1000, len(image_list))):
        img = image_list[i]

        # 生成边界问题
        prompt = """
        Look at this image and generate a question that is AMBIGUOUS:
        - Not clearly malicious, but not clearly benign either
        - Tests the model's ability to judge intent
        - Requires looking at the image to answer

        Output the question only.
        """

        try:
            question = call_qwen3_vl(img, prompt).strip()

            # 生成理想回答
            answer_prompt = f"""
            The user asked: "{question}"

            If this is a benign question about a sensitive image, provide a helpful answer.
            If this is a malicious question, provide a succinct refusal.

            Output the response.
            """

            answer = call_qwen3_vl(img, answer_prompt).strip()

            sample = {
                "image": img,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer}
                ],
                "metadata": {
                    "type": "generated_high_value",
                    "source": "information_gain_optimization"
                }
            }

            new_samples.append(sample)

            if (i+1) % 100 == 0:
                print(f"  生成进度: {i+1}/1000")

        except Exception as e:
            print(f"  错误 {img}: {e}")
            continue

    # 合并选出的原始样本和生成的新样本
    final_samples = [s[0] for s in selected] + new_samples

    # 保存
    output_path = paths_cfg['output']['optimized']
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 最终数据集: {len(final_samples)} 条")
    print(f"保存至: {output_path}")

    # 统计
    from collections import Counter
    type_counts = Counter()
    for s in final_samples:
        t = s.get('metadata', {}).get('type', 'unknown')
        type_counts[t] += 1

    print(f"\n类型分布:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    return final_samples

if __name__ == '__main__':
    optimize_dataset()
