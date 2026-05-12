# 创新数据生成流程

使用本地 **Qwen3-VL-30B-A3B-Instruct-FP8** 模型，实现5种创新方法生成高质量多模态安全训练数据。

## 目录结构

```
/home/disk/q3s2/data_generation_pipeline/
├── config/
│   ├── model_config.json      # 模型配置（路径、参数、性能）
│   └── paths_config.json     # 路径配置（输入、输出、脚本）
├── scripts/
│   ├── 00_main.py            # 主控制器（协调整个流程）
│   ├── 01_cross_modal.py     # 方案2：跨模态因果解耦
│   ├── 02_counterfactual.py # 方案3：元反事实推理链
│   ├── 03_semantic_boundary.py # 方案1：语义拓扑边界搜索
│   ├── 04_red_blue.py      # 方案4：多智能体红蓝对抗
│   └── 05_optimize.py      # 方案5：信息增益优化
├── output/                      # 生成的数据输出目录
│   ├── cross_modal_samples.json
│   ├── counterfactual_samples.json
│   ├── semantic_boundary_samples.json
│   ├── red_blue_samples.json
│   ├── optimized_samples.json
│   └── final_combined_dataset.json
└── README.md                   # 本文件
```

## 环境准备

### 1. 激活虚拟环境

```bash
source /home/disk/q3s2/.venv/bin/activate
```

### 2. 安装依赖

```bash
pip install openai json5 pathlib
```

### 3. 启动 vLLM 服务

```bash
# 启动 Qwen3-VL-30B-A3B-Instruct-FP8 服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --dtype fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# 验证服务是否启动
curl http://localhost:8000/v1/models
```

## 配置说明

### model_config.json

```json
{
    "model_name": "Qwen3-VL-30B-A3B-Instruct-FP8",
    "model_path": "/home/disk/q3s2/models/Qwen3-VL-30B-A3B-Instruct-FP8",
    "model_type": "qwen3-vl",
    "quantization": "fp8",
    "context_length": 131072,
    "performance": {
        "tokens_per_second": 130.1,
        "docvqa_score": 95.0,
        "ocrbench_score": 90.3
    }
}
```

### paths_config.json

```json
{
    "input": {
        "train_data": "/home/disk/q3s2/data_generation/Our_Dataset/final_dataset/20260309/sensitive_images/train_updated.json",
        "image_base_train": "/home/disk/q3s2/data_generation/Our_Dataset/images"
    },
    "output": {
        "pipeline_dir": "/home/disk/q3s2/data_generation_pipeline/output",
        "final_combined": "/home/disk/q3s2/data_generation_pipeline/output/final_combined_dataset.json"
    }
}
```

## 运行步骤

### 方法1：运行完整流程（推荐）

```bash
cd /home/disk/q3s2/data_generation_pipeline/scripts
python 00_main.py
```

这将按顺序执行：
1. **阶段1**：跨模态因果解耦（生成4000样本）
2. **阶段2**：元反事实推理链（生成2000样本）
3. **阶段3**：信息增益优化（筛选5000样本）
4. **阶段4**：语义拓扑边界搜索（生成2000样本）
5. **阶段5**：多智能体红蓝对抗（生成变数样本）
6. **合并**：所有数据合并为最终数据集

### 方法2：单独运行某个方案

```bash
# 只运行跨模态因果解耦
python /home/disk/q3s2/data_generation_pipeline/scripts/01_cross_modal.py

# 只运行元反事实推理链
python /home/disk/q3s2/data_generation_pipeline/scripts/02_counterfactual.py

# 只运行语义拓扑边界搜索
python /home/disk/q3s2/data_generation_pipeline/scripts/03_semantic_boundary.py

# 只运行多智能体红蓝对抗
python /home/disk/q3s2/data_generation_pipeline/scripts/04_red_blue.py

# 只运行信息增益优化
python /home/disk/q3s2/data_generation_pipeline/scripts/05_optimize.py
```

### 方法3：测试运行（10个样本）

修改任意脚本的 `main()` 函数：

```python
# 在 01_cross_modal.py 中
def generate_cross_modal():
    # ... 前面的代码 ...
    
    # 改为测试模式
    all_samples = []
    test_images = get_image_list(sensitive_dir)[:2]  # 只测试2张图
    
    for img in test_images:
        for pair_type in PAIR_PROMPTS:
            # ... 生成逻辑 ...
            if len(all_samples) >= 10:  # 只生成10个
                break
```

## 5种创新方案说明

| 方案 | 脚本 | 核心创新 | 解决什么问题 |
|------|--------|----------|----------|
| **方案1** | 03_semantic_boundary.py | 在语义空间找到模型"最不确定"的区域 | 只会模板匹配，不会真正理解 |
| **方案2** | 01_cross_modal.py | 构造"图像-文本"错配对，强迫分别理解 | 31%问题不引用图像，只看文本就拒绝 |
| **方案3** | 02_counterfactual.py | 生成"如果图变了，还敏感吗？"的推理链 | 不知道为什么拒绝，只会模式匹配 |
| **方案4** | 04_red_blue.py | 两个模型对抗，类似GAN | 对抗样本太单一，都是please+bypass |
| **方案5** | 05_optimize.py | 用模型预测每个样本的价值 | 14,387条中38.1%是重复的 |

### 推荐实施顺序

1. **第1周**：方案2（跨模态因果解耦）→ 解决"31%问题不引用图像"
2. **第2周**：方案3（元反事实推理）→ 加入推理链，解决"只会模式匹配"
3. **第3周**：方案5（信息增益优化）→ 过滤旧数据，去除重复
4. **迭代**：方案1+4（边界搜索+红蓝对抗）→ 持续提升

## 输出文件说明

| 文件 | 说明 | 预计大小 |
|------|------|----------|
| `cross_modal_samples.json` | 跨模态配对数据（4000条） | ~8MB |
| `counterfactual_samples.json` | 反事实推理链数据（2000条） | ~12MB |
| `semantic_boundary_samples.json` | 边界搜索数据（2000条） | ~6MB |
| `red_blue_samples.json` | 红蓝对抗数据（变数） | ~10MB |
| `optimized_samples.json` | 优化后的数据（5000条） | ~15MB |
| `final_combined_dataset.json` | **最终合并数据集** | ~50MB |

### 数据格式示例

```json
[
  {
    "image": "/home/disk/q3s2/data_generation/Our_Dataset/images/PCP/PII/PII972.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat type of document is this?"
      },
      {
        "from": "gpt",
        "value": "This is an ID document. It typically contains personal information..."
      }
    ],
    "metadata": {
      "type": "sensitive_img_safe_q",
      "requires_vision": true,
      "intent": "benign"
    }
  }
]
```

## 验证生成的数据

### 1. 检查格式

```bash
python -c "
import json
with open('/home/disk/q3s2/data_generation_pipeline/output/final_combined_dataset.json', 'r') as f:
    data = json.load(f)
print(f'总样本数: {len(data)}')
print(f'包含图像: {sum(1 for d in data if \"image\" in d or \"images\" in d)}')
print(f'包含对话: {sum(1 for d in data if \"conversations\" in d)}')
"
```

### 2. 检查子类分布

```bash
python -c "
import json
from collections import Counter
with open('/home/disk/q3s2/data_generation_pipeline/output/final_combined_dataset.json', 'r') as f:
    data = json.load(f)
subcats = Counter(d.get('metadata', {}).get('type', 'unknown') for d in data)
for t, c in subcats.most_common(10):
    print(f'{t}: {c}')
"
```

### 3. 检查多样性（重复率）

```bash
python -c "
import json
from collections import Counter
with open('/home/disk/q3s2/data_generation_pipeline/output/final_combined_dataset.json', 'r') as f:
    data = json.load(f)
questions = []
for d in data:
    if 'conversations' in d:
        for c in d['conversations']:
            if c['from'] == 'human':
                questions.append(c['value'].replace('<image>', '').strip())
dups = {k: v for k, v in Counter(questions).items() if v > 1}
print(f'总问题: {len(questions)}')
print(f'独特问题: {len(set(questions))}')
print(f'重复率: {len(dups)/len(questions)*100:.1f}%')
"
```

## 下一步：微调模型

```bash
# 使用生成的数据微调模型
python finetune.py \
    --train_data /home/disk/q3s2/data_generation_pipeline/output/final_combined_dataset.json \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --output_dir /home/disk/q3s2/finetuned_model/
```

## 常见问题

### Q: vLLM服务启动失败？

```bash
# 检查模型路径
ls /home/disk/q3s2/models/Qwen3-VL-30B-A3B-Instruct-FP8/

# 检查GPU显存
nvidia-smi

# 尝试降低显存占用
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

### Q: 生成速度太慢？

Qwen3-VL-30B-A3B 的推理速度是 **130.1 tokens/秒**，生成1000个样本约需：
- 每个样本平均：3-5秒（包含图像理解+生成）
- 1000个样本：约1-1.5小时

### Q: JSON解析失败？

```python
# 在调用模型后添加错误处理
try:
    response = call_qwen3_vl(img, prompt)
    result = json.loads(response)
except json.JSONDecodeError:
    # 尝试提取JSON部分
    import re
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
    else:
        print(f"无法解析: {response[:100]}")
        continue
```

## 参考文档

创新方案详细说明在：`/home/disk/q3s2/claude_code/`
- `00_overview.md` - 总览
- `01_semantic_topology.md` - 方案1详细说明
- `02_cross_modal_causal.md` - 方案2详细说明
- `03_meta_counterfactual.md` - 方案3详细说明
- `04_multi_agent_red_blue.md` - 方案4详细说明
- `05_information_gain.md` - 方案5详细说明

## 总结

这套流程解决了微调后拒绝率过高的核心问题：
1. **降低句式重复性**（从38.1% → <5%）
2. **增加视觉依赖样本**（100%问题必须引用图像）
3. **训练模型理解意图**（而非简单"文本模式→拒绝"）
4. **生成多样化数据**（5种创新方法，打破模板化）
# M3D_GEN_MODAL
