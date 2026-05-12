#!/usr/bin/env python3
"""
数据质量验证脚本
验证生成的数据集质量和效果
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

def load_dataset(file_path):
    """加载数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_basic_stats(data):
    """基础统计分析"""
    print("=== 基础统计 ===")

    total_samples = len(data)
    print(f"总样本数: {total_samples}")

    # 检查格式完整性
    has_image = sum(1 for d in data if 'image' in d)
    has_conversations = sum(1 for d in data if 'conversations' in d)
    has_metadata = sum(1 for d in data if 'metadata' in d)

    print(f"包含图像字段: {has_image}/{total_samples} ({has_image/total_samples*100:.1f}%)")
    print(f"包含对话字段: {has_conversations}/{total_samples} ({has_conversations/total_samples*100:.1f}%)")
    print(f"包含元数据字段: {has_metadata}/{total_samples} ({has_metadata/total_samples*100:.1f}%)")

    return total_samples, has_image, has_conversations, has_metadata

def analyze_type_distribution(data):
    """分析数据类型分布"""
    print("\n=== 数据类型分布 ===")

    type_counts = Counter()
    for sample in data:
        if 'metadata' in sample and 'type' in sample['metadata']:
            type_counts[sample['metadata']['type']] += 1
        else:
            type_counts['unknown'] += 1

    print("各类型样本数量:")
    for type_name, count in type_counts.most_common():
        print(f"  {type_name}: {count} ({count/len(data)*100:.1f}%)")

    return type_counts

def analyze_question_diversity(data):
    """分析问题多样性"""
    print("\n=== 问题多样性分析 ===")

    questions = []
    for sample in data:
        if 'conversations' in sample:
            for conv in sample['conversations']:
                if conv.get('from') == 'human':
                    question = conv['value'].replace('<image>', '').strip()
                    questions.append(question.lower())

    total_questions = len(questions)
    unique_questions = len(set(questions))

    print(f"总问题数: {total_questions}")
    print(f"独特问题数: {unique_questions}")
    print(f"重复率: {(total_questions-unique_questions)/total_questions*100:.1f}%")

    # 分析问题长度分布
    lengths = [len(q.split()) for q in questions]
    print(f"平均问题长度: {sum(lengths)/len(lengths):.1f} 词")
    print(f"最短问题: {min(lengths)} 词")
    print(f"最长问题: {max(lengths)} 词")

    # 检查是否包含图像引用
    image_references = sum(1 for q in questions if 'image' in q or 'picture' in q or 'photo' in q)
    print(f"明确引用图像的问题: {image_references}/{total_questions} ({image_references/total_questions*100:.1f}%)")

    return questions, unique_questions, total_questions

def analyze_intent_distribution(data):
    """分析意图分布"""
    print("\n=== 意图分布分析 ===")

    intent_counts = Counter()
    for sample in data:
        if 'metadata' in sample and 'intent' in sample['metadata']:
            intent_counts[sample['metadata']['intent']] += 1
        else:
            intent_counts['unknown'] += 1

    print("意图分布:")
    for intent, count in intent_counts.most_common():
        print(f"  {intent}: {count} ({count/len(data)*100:.1f}%)")

    return intent_counts

def analyze_vision_dependency(data):
    """分析视觉依赖程度"""
    print("\n=== 视觉依赖分析 ===")

    vision_required = 0
    text_only_solvable = 0

    for sample in data:
        if 'metadata' in sample:
            requires_vision = sample['metadata'].get('requires_vision', False)
            if requires_vision:
                vision_required += 1
            else:
                text_only_solvable += 1

    total = vision_required + text_only_solvable
    if total > 0:
        print(f"需要视觉理解: {vision_required}/{total} ({vision_required/total*100:.1f}%)")
        print(f"纯文本可解: {text_only_solvable}/{total} ({text_only_solvable/total*100:.1f}%)")

    return vision_required, text_only_solvable

def sample_manual_review(data, sample_size=10):
    """人工抽样检查"""
    print(f"\n=== 人工抽样检查 (随机抽取{sample_size}个样本) ===")

    import random
    samples = random.sample(data, min(sample_size, len(data)))

    for i, sample in enumerate(samples, 1):
        print(f"\n--- 样本 {i} ---")
        print(f"类型: {sample.get('metadata', {}).get('type', 'unknown')}")
        print(f"意图: {sample.get('metadata', {}).get('intent', 'unknown')}")

        if 'image' in sample:
            print(f"图像: {sample['image']}")

        if 'conversations' in sample:
            for conv in sample['conversations']:
                role = conv['from']
                content = conv['value'][:200] + "..." if len(conv['value']) > 200 else conv['value']
                print(f"{role}: {content}")

def generate_validation_report(data, output_path):
    """生成验证报告"""
    print(f"\n=== 生成验证报告 ===")

    report = {
        "summary": {
            "total_samples": len(data),
            "timestamp": "2026-05-12"
        },
        "quality_metrics": {},
        "recommendations": []
    }

    # 基础统计
    total, has_image, has_conv, has_meta = analyze_basic_stats(data)
    report["quality_metrics"]["format_completeness"] = {
        "image_fields": f"{has_image}/{total}",
        "conversation_fields": f"{has_conv}/{total}",
        "metadata_fields": f"{has_meta}/{total}"
    }

    # 类型分布
    type_dist = analyze_type_distribution(data)
    report["quality_metrics"]["type_distribution"] = dict(type_dist.most_common())

    # 问题多样性
    questions, unique_q, total_q = analyze_question_diversity(data)
    report["quality_metrics"]["question_diversity"] = {
        "total_questions": total_q,
        "unique_questions": unique_q,
        "duplication_rate": f"{(total_q-unique_q)/total_q*100:.1f}%"
    }

    # 意图分布
    intent_dist = analyze_intent_distribution(data)
    report["quality_metrics"]["intent_distribution"] = dict(intent_dist.most_common())

    # 视觉依赖
    vision_req, text_only = analyze_vision_dependency(data)
    report["quality_metrics"]["vision_dependency"] = {
        "vision_required": vision_req,
        "text_only_solvable": text_only
    }

    # 生成建议
    if (total_q - unique_q) / total_q > 0.1:  # 重复率超过10%
        report["recommendations"].append("建议增加问题多样性，当前重复率较高")

    if vision_req / (vision_req + text_only) < 0.5:  # 视觉依赖低于50%
        report["recommendations"].append("建议增加需要视觉理解的样本比例")

    if len(type_dist) < 4:  # 类型少于4种
        report["recommendations"].append("建议增加数据类型的多样性")

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"验证报告已保存至: {output_path}")

    return report

def main():
    """主函数"""
    # 检查输出目录
    output_dir = '/home/disk/q3s2/data_generation_pipeline/output'
    final_dataset = os.path.join(output_dir, 'final_combined_dataset.json')

    if not os.path.exists(final_dataset):
        print(f"错误: 找不到最终数据集文件 {final_dataset}")
        return

    # 加载数据
    print(f"加载数据集: {final_dataset}")
    data = load_dataset(final_dataset)

    # 执行各项分析
    analyze_basic_stats(data)
    analyze_type_distribution(data)
    analyze_question_diversity(data)
    analyze_intent_distribution(data)
    analyze_vision_dependency(data)
    sample_manual_review(data, sample_size=5)

    # 生成验证报告
    report_path = os.path.join(output_dir, 'validation_report.json')
    generate_validation_report(data, report_path)

    print("\n✅ 数据验证完成！")
    print("建议下一步: 使用此数据集进行模型微调测试")

if __name__ == '__main__':
    main()