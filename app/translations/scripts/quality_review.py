#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译质量审核与改进脚本
确保翻译符合"信达雅"标准，术语统一，表达无歧义
"""

import xml.etree.ElementTree as ET
import re
from collections import defaultdict

# 术语统一规范 - 基于TRANSLATION_STANDARDS.md
TERMINOLOGY_STANDARDS = {
    # 需要统一的术语对（错误 -> 正确）
    "云点": "点云",
    "格网": "网格",
    "法向量": "法线",
    "注册": "配准",
    "离群点": "异常值",
    "野值": "异常值",
    "噪音": "噪声",
    "边界框": "包围盒",
    "摄像机": "相机",
    "材料": "材质",
    "散列": "哈希",
    "标记": "标签",
    "重构": "重建",
    "执行": "应用",
    "对齐": "配准",  # 在Registration上下文中
}

# 需要检查一致性的术语（同一概念应该只用一种翻译）
CONSISTENCY_CHECK = {
    "Point Cloud": ["点云"],
    "Mesh": ["网格"],
    "Normal": ["法线"],
    "Scalar Field": ["标量场"],
    "Registration": ["配准"],
    "Bounding Box": ["包围盒"],
    "Filter": ["滤波"],
    "Segment": ["分割"],
    "Extract": ["提取"],
    "Transform": ["变换"],
    "Rotation": ["旋转"],
    "Translation": ["平移"],
    "Scale": ["缩放"],
}

# 需要改进表达的模式（直译 -> 信达雅）
EXPRESSION_IMPROVEMENTS = {
    # 改进冗长或不自然的表达
    "进行...操作": "...",  # 去除冗余
    "实施...": "执行...",
    "执行计算": "计算",
    "执行操作": "操作",
    
    # 改进被动语态为主动语态（符合中文习惯）
    "被选择的": "所选的",
    "被计算的": "已计算的",
    "将被": "将",
    
    # 改进欧化句式
    "它是": "这是",
    "这个是": "这是",
    "那个是": "那是",
}

# 歧义表达检查（可能产生歧义的词汇）
AMBIGUOUS_TERMS = {
    "处理": ["滤波", "计算", "操作"],  # "处理"太泛，应具体化
    "节点": ["顶点", "节点"],  # 需要区分几何顶点和数据结构节点
    "点": ["点", "顶点"],  # 需要区分Point和Vertex
    "面": ["面", "平面"],  # 需要区分Face和Plane
}


def check_terminology_consistency(text: str) -> list:
    """检查术语使用是否符合规范"""
    issues = []
    
    for wrong_term, correct_term in TERMINOLOGY_STANDARDS.items():
        if wrong_term in text:
            issues.append({
                'type': 'terminology',
                'severity': 'high',
                'message': f'使用了非标准术语 "{wrong_term}"，应改为 "{correct_term}"',
                'suggestion': text.replace(wrong_term, correct_term)
            })
    
    return issues


def check_expression_quality(text: str) -> list:
    """检查表达质量（信达雅）"""
    issues = []
    
    # 检查是否有直译痕迹
    for pattern, improvement in EXPRESSION_IMPROVEMENTS.items():
        if pattern in text:
            issues.append({
                'type': 'expression',
                'severity': 'medium',
                'message': f'表达可以改进："{pattern}" -> "{improvement}"',
                'suggestion': text.replace(pattern, improvement)
            })
    
    # 检查句子长度（中文句子不宜过长）
    if len(text) > 100 and '，' not in text[-50:]:
        issues.append({
            'type': 'readability',
            'severity': 'low',
            'message': '句子较长，建议适当断句',
            'suggestion': None
        })
    
    return issues


def check_ambiguity(text: str) -> list:
    """检查是否存在歧义"""
    issues = []
    
    for ambiguous, alternatives in AMBIGUOUS_TERMS.items():
        if ambiguous in text:
            issues.append({
                'type': 'ambiguity',
                'severity': 'medium',
                'message': f'"{ambiguous}"可能产生歧义，建议明确为：{", ".join(alternatives)}',
                'suggestion': None
            })
    
    return issues


def audit_translation_quality(ts_file: str) -> dict:
    """审核翻译文件质量"""
    
    tree = ET.parse(ts_file)
    root = tree.getroot()
    
    audit_results = {
        'total_messages': 0,
        'audited_messages': 0,
        'issues': [],
        'by_context': defaultdict(list),
        'by_type': defaultdict(int),
        'by_severity': defaultdict(int),
    }
    
    for context in root.findall('context'):
        context_name = context.find('name').text or "Unknown"
        
        for message in context.findall('message'):
            audit_results['total_messages'] += 1
            
            source = message.find('source')
            translation = message.find('translation')
            
            if source is None or translation is None:
                continue
            
            src_text = source.text or ""
            trans_text = translation.text or ""
            
            # 只审核已翻译的内容
            if not trans_text:
                continue
            
            audit_results['audited_messages'] += 1
            
            # 执行各项检查
            all_issues = []
            all_issues.extend(check_terminology_consistency(trans_text))
            all_issues.extend(check_expression_quality(trans_text))
            all_issues.extend(check_ambiguity(trans_text))
            
            if all_issues:
                for issue in all_issues:
                    issue['context'] = context_name
                    issue['source'] = src_text
                    issue['translation'] = trans_text
                    
                    audit_results['issues'].append(issue)
                    audit_results['by_context'][context_name].append(issue)
                    audit_results['by_type'][issue['type']] += 1
                    audit_results['by_severity'][issue['severity']] += 1
    
    return audit_results


def apply_improvements(ts_file: str, output_file: str) -> dict:
    """应用翻译改进"""
    
    tree = ET.parse(ts_file)
    root = tree.getroot()
    
    stats = {
        'total': 0,
        'improved': 0,
        'unchanged': 0,
        'improvements': []
    }
    
    for context in root.findall('context'):
        context_name = context.find('name').text or ""
        
        for message in context.findall('message'):
            stats['total'] += 1
            
            source = message.find('source')
            translation = message.find('translation')
            
            if source is None or translation is None:
                continue
            
            src_text = source.text or ""
            trans_text = translation.text or ""
            
            if not trans_text:
                stats['unchanged'] += 1
                continue
            
            # 应用术语统一改进
            improved_text = trans_text
            changed = False
            
            for wrong_term, correct_term in TERMINOLOGY_STANDARDS.items():
                if wrong_term in improved_text:
                    improved_text = improved_text.replace(wrong_term, correct_term)
                    changed = True
            
            # 应用表达改进
            for pattern, improvement in EXPRESSION_IMPROVEMENTS.items():
                if pattern in improved_text:
                    improved_text = improved_text.replace(pattern, improvement)
                    changed = True
            
            if changed:
                translation.text = improved_text
                stats['improved'] += 1
                stats['improvements'].append({
                    'context': context_name,
                    'source': src_text[:80],
                    'before': trans_text,
                    'after': improved_text
                })
            else:
                stats['unchanged'] += 1
    
    # 写入输出文件
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # 修复DOCTYPE
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '<!DOCTYPE TS>' not in content:
        content = content.replace(
            '<?xml version=\'1.0\' encoding=\'utf-8\'?>',
            '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>'
        )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return stats


def generate_audit_report(audit_results: dict, output_file: str):
    """生成审核报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 翻译质量审核报告\n\n")
        f.write(f"**审核日期**: 2026-01-14\n\n")
        
        # 总览
        f.write("## 审核总览\n\n")
        f.write(f"- 总消息数: {audit_results['total_messages']}\n")
        f.write(f"- 已审核: {audit_results['audited_messages']}\n")
        f.write(f"- 发现问题: {len(audit_results['issues'])}\n\n")
        
        # 按类型统计
        f.write("## 问题类型分布\n\n")
        f.write("| 类型 | 数量 |\n")
        f.write("|------|------|\n")
        for issue_type, count in sorted(audit_results['by_type'].items()):
            f.write(f"| {issue_type} | {count} |\n")
        f.write("\n")
        
        # 按严重程度统计
        f.write("## 问题严重程度\n\n")
        f.write("| 严重程度 | 数量 |\n")
        f.write("|----------|------|\n")
        for severity, count in sorted(audit_results['by_severity'].items()):
            f.write(f"| {severity} | {count} |\n")
        f.write("\n")
        
        # 按上下文统计（前10个）
        f.write("## 问题最多的组件（前10个）\n\n")
        f.write("| 组件 | 问题数 |\n")
        f.write("|------|--------|\n")
        sorted_contexts = sorted(
            audit_results['by_context'].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        for context, issues in sorted_contexts[:10]:
            f.write(f"| {context} | {len(issues)} |\n")
        f.write("\n")
        
        # 详细问题列表（高优先级）
        f.write("## 高优先级问题详情\n\n")
        high_priority = [i for i in audit_results['issues'] if i['severity'] == 'high']
        if high_priority:
            for idx, issue in enumerate(high_priority[:20], 1):
                f.write(f"### {idx}. {issue['context']}\n\n")
                f.write(f"**原文**: {issue['source'][:100]}\n\n")
                f.write(f"**当前翻译**: {issue['translation']}\n\n")
                f.write(f"**问题**: {issue['message']}\n\n")
                if issue['suggestion']:
                    f.write(f"**建议**: {issue['suggestion']}\n\n")
                f.write("---\n\n")
        else:
            f.write("✅ 未发现高优先级问题\n\n")
        
        # 建议
        f.write("## 改进建议\n\n")
        f.write("1. 优先修复高优先级问题\n")
        f.write("2. 统一术语使用\n")
        f.write("3. 改进表达流畅度\n")
        f.write("4. 消除歧义表达\n")
        f.write("5. 参考 TRANSLATION_STANDARDS.md 进行规范化\n")


def main():
    import os
    import sys
    
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ts_file = os.path.join(repo_root, "app", "translations", "ACloudViewer_zh.ts")
    
    print("=" * 80)
    print("翻译质量审核与改进")
    print("=" * 80)
    print()
    
    # 1. 审核现有翻译
    print("📊 正在审核翻译质量...")
    audit_results = audit_translation_quality(ts_file)
    
    print(f"\n审核完成:")
    print(f"  总消息数: {audit_results['total_messages']}")
    print(f"  已审核: {audit_results['audited_messages']}")
    print(f"  发现问题: {len(audit_results['issues'])}")
    
    if audit_results['issues']:
        print(f"\n问题分布:")
        for issue_type, count in audit_results['by_type'].items():
            print(f"  - {issue_type}: {count}")
        
        print(f"\n严重程度:")
        for severity, count in audit_results['by_severity'].items():
            print(f"  - {severity}: {count}")
    
    # 2. 生成审核报告
    report_file = os.path.join(repo_root, "app", "translations",
                               "QUALITY_AUDIT_REPORT.md")
    print(f"\n📝 生成审核报告: {report_file}")
    generate_audit_report(audit_results, report_file)
    
    # 3. 应用自动改进
    print(f"\n🔧 应用自动改进...")
    output_file = ts_file  # 直接改进原文件（已有备份）
    stats = apply_improvements(ts_file, output_file)
    
    print(f"\n改进统计:")
    print(f"  总计: {stats['total']}")
    print(f"  已改进: {stats['improved']}")
    print(f"  未变更: {stats['unchanged']}")
    
    if stats['improvements']:
        print(f"\n改进示例（前5个）:")
        for idx, imp in enumerate(stats['improvements'][:5], 1):
            print(f"\n  {idx}. [{imp['context']}]")
            print(f"     原文: {imp['source']}...")
            print(f"     改前: {imp['before']}")
            print(f"     改后: {imp['after']}")
    
    print("\n" + "=" * 80)
    print("✅ 审核与改进完成！")
    print("=" * 80)
    print(f"\n📄 查看详细报告: {report_file}")
    print(f"📘 参考翻译规范: app/translations/TRANSLATION_STANDARDS.md")


if __name__ == "__main__":
    main()
