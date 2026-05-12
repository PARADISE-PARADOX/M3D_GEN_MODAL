#!/usr/bin/env python3
"""
方案3：元反事实推理链 (Meta-Counterfactual Reasoning)
使用Qwen3-VL生成"如果图中X变了，还敏感吗？"的推理数据
解决：模型不知道"为什么拒绝"，只会模式匹配
"""

import json
import sys
import random
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

COUNTERFACTUAL_PROMPT = """
Look at this image: [image]

Now imagine COUNTERFACTUAL scenarios and reason about each:

Scenario 1 (Original):
  → What sensitive content exists in the original image?
  → Generate a natural question about this image.
  → Decision: ANSWER or REFUSE? Why?

Scenario 2 (Blur sensitive parts):
  Imagine blurring all PII/violence/private content in the image.
  → Same question as Scenario 1.
  → Is it still sensitive? Decision changes?

Scenario 3 (Change intent to protective):
  Same image, but ask: "Help me redact the sensitive parts."
  → Should refuse? No! This is protective intent.
  → Generate helpful answer about redaction.

Scenario 4 (Educational context):
  Same image, ask from educational perspective.
  → Decision: ANSWER (context matters!)

Output as JSON:
{
  "scenario1": {"question": "...", "sensitivity": "...", "decision": "ANSWER/REFUSE", "reasoning": "..."},
  "scenario2": {"question": "...", "sensitivity": "...", "decision": "ANSWER/REFUSE", "reasoning": "..."},
  "scenario3": {"question": "...", "sensitivity": "...", "decision": "ANSWER", "answer": "..."},
  "scenario4": {"question": "...", "sensitivity": "...", "decision": "ANSWER", "answer": "..."}
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

def generate_counterfactual(output_path=None, num_samples=100):
    """主生成函数"""
    model_cfg, paths_cfg = load_config()

    print("=== 元反事实推理链生成 ===")
    print(f"模型: {model_cfg['model_name']}")

    # 输出路径：优先使用参数，否则用配置
    if output_path is None:
        output_path = paths_cfg['output']['counterfactual']

    # 获取图像
    sensitive_dir = paths_cfg['input']['image_base_train']
    images = get_image_list(sensitive_dir)
    print(f"找到 {len(images)} 张图像")

    all_samples = []

    for idx, img in enumerate(images[:num_samples]):
        print(f"\r进度: {idx+1}/{num_samples}", end='', flush=True)

        try:
            response = call_qwen3_vl(img, COUNTERFACTUAL_PROMPT)
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

            # 构造多轮对话训练样本
            sample = {
                "image": img,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{result['scenario1']['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": f"Reasoning: {result['scenario1']['reasoning']}\n\nFinal: " +
                                   ("I cannot assist..." if result['scenario1']['decision'] == "REFUSE"
                                    else result['scenario1'].get('answer', 'This appears to be...'))
                    },
                    {
                        "from": "human",
                        "value": f"<image>\n{result['scenario3']['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": f"Reasoning: Intent changed to protective.\n\nFinal: {result['scenario3']['answer']}"
                    }
                ],
                "metadata": {
                    "type": "counterfactual_chain",
                    "num_scenarios": 4
                }
            }
            all_samples.append(sample)

        except Exception as e:
            print(f"\n错误 {img}: {e}")
            continue

    # 保存（output_path 已在函数入口处理）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 完成! 生成 {len(all_samples)} 个样本")
    print(f"保存至: {output_path}")

    return all_samples

if __name__ == '__main__':
    generate_counterfactual()
