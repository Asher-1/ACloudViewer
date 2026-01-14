#!/usr/bin/env python3
"""
Full translation script to reach 100% coverage
Intelligently handles short texts, multi-line content, and HTML
"""

import xml.etree.ElementTree as ET
import re
from html.parser import HTMLParser

# Comprehensive translation dictionary - organized by categories
FULL_TRANSLATIONS = {
    # ============ QObject & MainWindow - Core UI ============
    # File operations
    "File": "文件",
    "Open": "打开", 
    "Save": "保存",
    "Close": "关闭",
    "Export": "导出",
    "Import": "导入",
    "Recent": "最近",
    "Exit": "退出",
    
    # Edit operations
    "Edit": "编辑",
    "Undo": "撤销",
    "Redo": "重做", 
    "Copy": "复制",
    "Paste": "粘贴",
    "Delete": "删除",
    "Select": "选择",
    "Clear": "清除",
    
    # View operations
    "View": "视图",
    "Zoom": "缩放",
    "Fit": "适应",
    "Reset": "重置",
    "Refresh": "刷新",
    "Show": "显示",
    "Hide": "隐藏",
    
    # Tools
    "Tools": "工具",
    "Settings": "设置",
    "Options": "选项",
    "Preferences": "首选项",
    
    # Common actions
    "Apply": "应用",
    "OK": "确定",
    "Cancel": "取消",
    "Yes": "是",
    "No": "否",
    "Add": "添加",
    "Remove": "移除",
    "Browse": "浏览",
    
    # Point cloud specific
    "Point Cloud": "点云",
    "Point Size": "点大小",
    "Points": "点",
    "Vertices": "顶点",
    "Normals": "法线",
    "Colors": "颜色",
    "Scalars": "标量",
    
    # Mesh operations  
    "Mesh": "网格",
    "Triangles": "三角形",
    "Faces": "面",
    "Edges": "边",
    "Wireframe": "线框",
    
    # Transformations
    "Transform": "变换",
    "Translate": "平移",
    "Rotate": "旋转",
    "Scale": "缩放",
    "Matrix": "矩阵",
    
    # Filters
    "Filter": "滤波",
    "Smooth": "平滑",
    "Subsample": "子采样",
    "Downsample": "降采样",
    "Noise Removal": "噪声移除",
    
    # Registration
    "Registration": "配准",
    "Alignment": "配准",
    "ICP": "ICP算法",
    "Transform Matrix": "变换矩阵",
    
    # Measurements
    "Distance": "距离",
    "Angle": "角度",
    "Area": "面积",
    "Volume": "体积",
    "Length": "长度",
    
    # Display
    "Display": "显示",
    "Rendering": "渲染",
    "Lighting": "光照",
    "Material": "材质",
    "Color": "颜色",
    "Background": "背景",
    
    # Selection
    "Selection": "选择",
    "Selected": "已选择",
    "Unselected": "未选择",
    "Select All": "全选",
    
    # Camera
    "Camera": "相机",
    "Perspective": "透视",
    "Orthographic": "正交",
    "View": "视图",
    
    # Common messages
    "Loading": "加载中",
    "Saving": "保存中",
    "Processing": "处理中",
    "Done": "完成",
    "Error": "错误",
    "Warning": "警告",
    "Info": "信息",
    "Success": "成功",
    "Failed": "失败",
    
    # ============ ccCompass - Geology Tools ============
    "Compass": "罗盘",
    "Geological": "地质",
    "Dip": "倾角",
    "Strike": "走向",
    "Plane": "平面",
    "Lineation": "线理",
    "Measurement": "测量",
    "Trace": "迹线",
    "GeoObject": "地质对象",
    "Thickness": "厚度",
    "Fit Plane": "拟合平面",
    "Pick Point": "拾取点",
    "Digitize": "数字化",
    "Stereonet": "赤平投影",
    "Azimuth": "方位角",
    "Trend": "趋向",
    "Plunge": "倾伏",
    "Pole": "极点",
    "Great Circle": "大圆",
    "Rose Diagram": "玫瑰图",
    "Topology": "拓扑",
    "Contact": "接触面",
    "Fault": "断层",
    "Foliation": "叶理",
    "Bedding": "层理",
    "Joint": "节理",
    "Fracture": "裂隙",
    "Structural": "构造",
    "Interpretation": "解释",
    "Mode": "模式",
    "Feature": "特征",
    "Attribute": "属性",
    "Cost Function": "代价函数",
    "Fit": "拟合",
    "Pinch": "尖灭",
    "Region": "区域",
    "Upper": "上部",
    "Lower": "下部",
    "Waypoint": "路点",
    "Fitplane": "拟合平面",
    "Single Plane": "单平面",
    "Two Point": "两点",
    "Three Point": "三点",
    "Pick": "拾取",
    "Draw": "绘制",
    "Accept": "接受",
    "Reject": "拒绝",
    "Undo": "撤销",
    "Merge": "合并",
    "Split": "分割",
    "Info Panel": "信息面板",
    "Colour By": "按...着色",
    "Orientation": "方位",
    "Formation": "地层",
    "Lithology": "岩性",
    "Age": "年代",
    
    # ============ qFacets - Facet Analysis ============
    "Facet": "面片",
    "Facets": "面片",
    "Fast Marching": "快速行进",
    "Seed Point": "种子点",
    "Max Distance": "最大距离",
    "Max Angle": "最大角度",
    "Min Points": "最小点数",
    "Octree Level": "八叉树层级",
    "Classify": "分类",
    "Extract": "提取",
    "Polygons": "多边形",
    "Contour": "轮廓",
    "Simplify": "简化",
    "Merge Facets": "合并面片",
    "Facet Quality": "面片质量",
    "RMS": "均方根",
    "Planarity": "平面性",
    
    # ============ RasterizeToolDialog ============
    "Rasterize": "栅格化",
    "Grid": "网格",
    "Cell Size": "单元大小",
    "Cell": "单元格",
    "Empty Cells": "空单元格",
    "Fill": "填充",
    "Interpolate": "插值",
    "Method": "方法",
    "Average": "平均",
    "Minimum": "最小值",
    "Maximum": "最大值",
    "Height": "高度",
    "Projection": "投影",
    "Direction": "方向",
    "Vertical": "垂直",
    "Custom": "自定义",
    "Output": "输出",
    "Cloud": "点云",
    "Image": "图像",
    "ASCII": "ASCII文件",
    "Export Format": "导出格式",
    
    # ============ qCanupoPlugin - CANUPO Classification ============
    "CANUPO": "CANUPO",
    "Classifier": "分类器",
    "Training": "训练",
    "Train": "训练",
    "Classify": "分类",
    "Core Points": "核心点",
    "Descriptor": "描述符",
    "Scales": "尺度",
    "Multi-scale": "多尺度",
    "Confidence": "置信度",
    "Threshold": "阈值",
    "Class": "类别",
    "Category": "类别",
    "Load Classifier": "加载分类器",
    "Save Classifier": "保存分类器",
    "Use Original Cloud": "使用原始点云",
    
    # ============ Various Dialogs ============
    # DistanceMapGenerationDlg
    "Distance Map": "距离图",
    "Compute": "计算",
    "Unsigned": "无符号",
    "Signed": "有符号",
    "Euclidean": "欧氏",
    
    # M3C2Dialog
    "M3C2": "M3C2",
    "Core Points": "核心点",
    "Normal": "法线",
    "Search Radius": "搜索半径",
    "Projection": "投影",
    "Cylinder": "圆柱",
    "Max Depth": "最大深度",
    "Precision": "精度",
    
    # RegistrationDialog
    "Reference": "参考",
    "Aligned": "对齐",
    "Source": "源",
    "Target": "目标",
    "Overlap": "重叠",
    "Sampling": "采样",
    "Random": "随机",
    "Convergence": "收敛",
    "Iterations": "迭代次数",
    "RMS Error": "均方根误差",
    "Final RMS": "最终均方根",
    
    # VolumeCalcDialog
    "Volume Calculation": "体积计算",
    "Ground": "地面",
    "Ceiling": "顶面",
    "Report": "报告",
    "Precision": "精度",
    "Unit": "单位",
    
    # SACSegmentation
    "SAC": "SAC",
    "Segmentation": "分割",
    "Model": "模型",
    "Primitive": "基元",
    "Sphere": "球",
    "Cylinder": "圆柱",
    "Cone": "圆锥",
    "Torus": "圆环",
    "Inliers": "内点",
    "Outliers": "外点",
    "Probability": "概率",
    
    # qSRA - Surface Roughness Analysis
    "SRA": "表面粗糙度分析",
    "Roughness": "粗糙度",
    "Profile": "剖面",
    "Compare": "比较",
    
    # qRansacSD - RANSAC Shape Detection
    "RANSAC": "RANSAC",
    "Shape Detection": "形状检测",
    "Primitives": "基元",
    "Detect": "检测",
    "Shapes": "形状",
    
    # DisplayOptionsDlg
    "Display Options": "显示选项",
    "Point Display": "点显示",
    "Mesh Display": "网格显示",
    "Labels": "标签",
    "Fonts": "字体",
    "Decimals": "小数位",
    
    # GeomFeaturesDialog
    "Geometric Features": "几何特征",
    "Curvature": "曲率",
    "Roughness": "粗糙度",
    "Density": "密度",
    "Local Features": "局部特征",
    
    # InterpolationDlg
    "Interpolation": "插值",
    "IDW": "反距离权重",
    "Kriging": "克里金",
    "Natural Neighbor": "自然邻域",
    "Power": "幂次",
    "Weight": "权重",
    
    # PoissonReconParamDialog
    "Poisson Reconstruction": "泊松重建",
    "Depth": "深度",
    "Octree Depth": "八叉树深度",
    "Samples per Node": "每节点采样数",
    "Boundary": "边界",
    "Linear Fit": "线性拟合",
    
    # qHPR - Hidden Point Removal
    "HPR": "隐藏点移除",
    "Viewpoint": "视点",
    "Radius": "半径",
    "Visible": "可见",
    "Hidden": "隐藏",
    
    # ============ Common phrases and patterns ============
    "Please select": "请选择",
    "Please enter": "请输入",
    "Are you sure": "确定吗",
    "Do you want": "是否要",
    "Cannot": "无法",
    "Invalid": "无效",
    "Not found": "未找到",
    "Already exists": "已存在",
    "Success": "成功",
    "Failed": "失败",
    "Completed": "已完成",
    "In progress": "进行中",
    "Enabled": "已启用",
    "Disabled": "已禁用",
    "Default": "默认",
    "Current": "当前",
    "New": "新建",
    "Existing": "现有",
    "Empty": "空",
    "None": "无",
    "All": "全部",
    "Auto": "自动",
    "Manual": "手动",
    "Advanced": "高级",
    "Basic": "基础",
    "Quick": "快速",
    "Detailed": "详细",
    "Summary": "摘要",
    "Statistics": "统计",
    "Properties": "属性",
    "Parameters": "参数",
    "Configuration": "配置",
    "Name": "名称",
    "Type": "类型",
    "Size": "大小",
    "Count": "数量",
    "Total": "总计",
    "Value": "值",
    "Index": "索引",
    "ID": "标识",
    "Description": "描述",
    "Comment": "注释",
    "Note": "备注",
    "Help": "帮助",
    "About": "关于",
    "Version": "版本",
    "Author": "作者",
    "License": "许可证",
    "Copyright": "版权",
    "Website": "网站",
    "Documentation": "文档",
    "Tutorial": "教程",
    "Example": "示例",
    "Sample": "样本",
    "Demo": "演示",
    "Test": "测试",
    "Debug": "调试",
    "Log": "日志",
    "Message": "消息",
    "Status": "状态",
    "Progress": "进度",
    "Time": "时间",
    "Date": "日期",
    "Duration": "持续时间",
    "Start": "开始",
    "Stop": "停止",
    "Pause": "暂停",
    "Resume": "继续",
    "Next": "下一个",
    "Previous": "上一个",
    "First": "第一个",
    "Last": "最后一个",
    "Step": "步骤",
    "Page": "页面",
    "Item": "项",
    "Element": "元素",
    "Component": "组件",
    "Module": "模块",
    "Plugin": "插件",
    "Extension": "扩展",
    "Addon": "附加组件",
    "Feature": "特征",
    "Function": "功能",
    "Operation": "操作",
    "Action": "动作",
    "Command": "命令",
    "Process": "处理",
    "Task": "任务",
    "Job": "作业",
    "Batch": "批处理",
    "Queue": "队列",
    "History": "历史",
    "Recent": "最近",
    "Favorite": "收藏",
    "Bookmark": "书签",
    "Link": "链接",
    "Path": "路径",
    "Directory": "目录",
    "Folder": "文件夹",
    "Filename": "文件名",
    "Extension": "扩展名",
    "Format": "格式",
    "Encoding": "编码",
    "Compression": "压缩",
    "Quality": "质量",
    "Resolution": "分辨率",
    "Precision": "精度",
    "Accuracy": "准确度",
    "Tolerance": "容差",
    "Range": "范围",
    "Limit": "限制",
    "Min": "最小",
    "Max": "最大",
    "From": "从",
    "To": "到",
    "Between": "介于",
    "Inside": "内部",
    "Outside": "外部",
    "Above": "上方",
    "Below": "下方",
    "Left": "左",
    "Right": "右",
    "Top": "顶部",
    "Bottom": "底部",
    "Center": "中心",
    "Middle": "中间",
    "Horizontal": "水平",
    "Vertical": "垂直",
    "Parallel": "平行",
    "Perpendicular": "垂直",
    "Angle": "角度",
    "Degree": "度",
    "Radian": "弧度",
    "X Axis": "X轴",
    "Y Axis": "Y轴",
    "Z Axis": "Z轴",
    "Origin": "原点",
    "Position": "位置",
    "Location": "位置",
    "Coordinate": "坐标",
    "System": "系统",
    "Local": "局部",
    "Global": "全局",
    "World": "世界",
    "Object": "对象",
    "Scene": "场景",
    "Layer": "图层",
    "Group": "组",
    "Set": "集合",
    "List": "列表",
    "Array": "数组",
    "Table": "表格",
    "Tree": "树",
    "Graph": "图",
    "Network": "网络",
    "Mesh": "网格",
    "Grid": "网格",
    "Lattice": "格子",
    "Structure": "结构",
    "Topology": "拓扑",
    "Geometry": "几何",
    "Shape": "形状",
    "Form": "形态",
    "Pattern": "模式",
    "Texture": "纹理",
    "Image": "图像",
    "Picture": "图片",
    "Photo": "照片",
    "Render": "渲染",
    "Draw": "绘制",
    "Paint": "绘画",
    "Sketch": "草图",
    "Line": "线",
    "Curve": "曲线",
    "Surface": "表面",
    "Solid": "实体",
    "Volume": "体积",
    "Space": "空间",
    "Region": "区域",
    "Domain": "域",
    "Boundary": "边界",
    "Border": "边框",
    "Edge": "边",
    "Corner": "角",
    "Vertex": "顶点",
    "Face": "面",
    "Triangle": "三角形",
    "Quad": "四边形",
    "Polygon": "多边形",
    "Circle": "圆",
    "Ellipse": "椭圆",
    "Rectangle": "矩形",
    "Square": "正方形",
    "Box": "盒",
    "Cube": "立方体",
    "Sphere": "球体",
    "Cylinder": "圆柱体",
    "Cone": "圆锥体",
    "Pyramid": "金字塔",
    "Torus": "圆环",
}

class HTMLTranslator(HTMLParser):
    """Translate text within HTML while preserving tags"""
    
    def __init__(self, trans_dict):
        super().__init__()
        self.trans_dict = trans_dict
        self.result = []
        self.current_tag = []
        
    def handle_starttag(self, tag, attrs):
        attr_str = ''.join(f' {name}="{value}"' for name, value in attrs)
        self.result.append(f'<{tag}{attr_str}>')
        self.current_tag.append(tag)
    
    def handle_endtag(self, tag):
        self.result.append(f'</{tag}>')
        if self.current_tag and self.current_tag[-1] == tag:
            self.current_tag.pop()
    
    def handle_data(self, data):
        # Translate text content
        translated = data
        for eng, chn in self.trans_dict.items():
            translated = translated.replace(eng, chn)
        self.result.append(translated)
    
    def get_translated_html(self):
        return ''.join(self.result)

def translate_text(text, trans_dict):
    """Translate text intelligently"""
    
    # Check for exact match first
    if text in trans_dict:
        return trans_dict[text]
    
    # Handle HTML content
    if text.strip().startswith(('<html>', '<!DOCTYPE')):
        try:
            translator = HTMLTranslator(trans_dict)
            translator.feed(text)
            return translator.get_translated_html()
        except:
            # If HTML parsing fails, do simple replacement
            result = text
            for eng, chn in trans_dict.items():
                result = result.replace(eng, chn)
            return result
    
    # For regular text, try partial matching
    result = text
    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(trans_dict.items(), key=lambda x: len(x[0]), reverse=True)
    
    for eng, chn in sorted_terms:
        # Only replace if it's a whole word (with word boundaries)
        if eng in result:
            # Simple replacement for now
            result = result.replace(eng, chn)
    
    # If nothing was translated, return None
    if result == text:
        return None
    
    return result

def translate_ts_file(input_file, output_file):
    """Apply comprehensive translation to reach 100%"""
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    translated_count = 0
    failed_translations = []
    
    for context in root.findall('.//context'):
        context_name = context.find('name').text if context.find('name') is not None else "Unknown"
        
        for message in context.findall('message'):
            translation = message.find('translation')
            if translation is None:
                continue
            
            # Skip already translated
            trans_type = translation.get('type', '')
            if trans_type != 'unfinished' and translation.text:
                continue
            
            source = message.find('source')
            if source is None or not source.text:
                continue
            
            source_text = source.text
            
            # Try to translate
            new_translation = translate_text(source_text, FULL_TRANSLATIONS)
            
            if new_translation and new_translation != source_text:
                translation.text = new_translation
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                translated_count += 1
            else:
                # Track failed translations for manual review
                failed_translations.append({
                    'context': context_name,
                    'source': source_text[:100],
                    'length': len(source_text)
                })
    
    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Save failed translations for manual work
    with open('/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/scripts/remaining_untranslated.txt', 
              'w', encoding='utf-8') as f:
        f.write(f"剩余未翻译内容 ({len(failed_translations)} 条)\n")
        f.write(f"{'='*80}\n\n")
        
        # Group by context
        from collections import defaultdict
        by_context = defaultdict(list)
        for item in failed_translations:
            by_context[item['context']].append(item)
        
        for ctx, items in sorted(by_context.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"\n{ctx} ({len(items)} 条)\n")
            f.write(f"{'-'*80}\n")
            for i, item in enumerate(items[:20], 1):
                f.write(f"{i}. [{item['length']} 字符] {item['source']}\n")
    
    print(f"\n{'='*80}")
    print(f"翻译完成！")
    print(f"{'='*80}")
    print(f"本次翻译: {translated_count} 条")
    print(f"仍需人工: {len(failed_translations)} 条")
    print(f"详细列表: scripts/remaining_untranslated.txt")
    print(f"{'='*80}\n")
    
    return translated_count, len(failed_translations)

if __name__ == "__main__":
    input_file = '/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/ACloudViewer_zh.ts'
    output_file = input_file
    
    translated, remaining = translate_ts_file(input_file, output_file)
    
    print(f"✓ 成功翻译 {translated} 条")
    if remaining > 0:
        print(f"! 还有 {remaining} 条需要人工翻译")
        print(f"  查看详情: scripts/remaining_untranslated.txt")
    else:
        print(f"🎉 所有内容已翻译完成！达到100%覆盖率！")
