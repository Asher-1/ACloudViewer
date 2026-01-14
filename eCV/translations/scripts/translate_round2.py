#!/usr/bin/env python3
"""
Round 2 Translation - Focus on complete sentences, messages with parameters, and remaining content
"""

import xml.etree.ElementTree as ET
import re

# Round 2: Complete sentences, messages, and specific phrases
ROUND2_TRANSLATIONS = {
    # Progress and status messages
    "Preparing polar display...": "准备极坐标显示...",
    "Preparing colored DTM": "准备彩色DTM",
    "Please wait... reading in progress": "请稍候...正在读取",
    "Please wait... writing in progress": "请稍候...正在写入",
    "Please wait... saving in progress": "请稍候...正在保存",
    "Computing entities scales": "计算实体比例",
    "Triangulation in progress...": "三角化进行中...",
    "Computing strain estimates": "计算应变估计",
    "Calculating strain tensors...": "计算应变张量...",
    "Estimating P21 Intensity": "估计P21强度",
    "Calculating patch areas...": "计算面片区域...",
    
    # Error and warning messages with parameters
    "Missing parameter: filename after \"-%1\"": "缺少参数：\"-%1\" 后的文件名",
    "Missing parameter: value after \"-%1\"": "缺少参数：\"-%1\" 后的值",
    "Missing parameter: vertices count after \"-%1\"": "缺少参数：\"-%1\" 后的顶点数量",
    "Missing parameter: extension after '%1'": "缺少参数：'%1' 后的扩展名",
    "Missing parameter: precision value after '%1'": "缺少参数：'%1' 后的精度值",
    "Missing parameter: separator character after '%1'": "缺少参数：'%1' 后的分隔符",
    "Unhandled format specifier (%1)": "未处理的格式说明符 (%1)",
    "Couldn't find the plugin '%1'": "找不到插件 '%1'",
    "Could not compute octree for cloud '%1'": "无法为点云 '%1' 计算八叉树",
    "Third party library error: %1": "第三方库错误：%1",
    
    # Status messages with parameters
    "cloud %1/%2 (%3 points)": "点云 %1/%2（%3 个点）",
    "Approximate number of points: %1": "近似点数：%1",
    "Up to (%1 x %2 x %3) = %4 section(s)": "最多 (%1 x %2 x %3) = %4 个截面",
    
    # Special messages
    "Hum, it seems that ECV has crashed... Sorry about that :)": "嗯，看起来ECV崩溃了...对此感到抱歉 :)",
    "SmallWidgets Interface": "小部件界面",
    
    # File I/O messages
    "Can't save selected entity(ies) this way!": "无法以这种方式保存选中的实体！",
    "[I/O] The following selected entities won't be saved:": "[I/O] 以下选中的实体不会被保存：",
    "\t- %1s": "\t- %1秒",
    "Some entities were ingored! (see console)": "某些实体被忽略了！（见控制台）",
    
    # Transformation messages  
    "Entity '%1' has been translated: (%2,%3,%4) and rescaled of a factor %5 [original position will be restored after saving]": 
        "实体 '%1' 已被平移：(%2,%3,%4) 并按因子 %5 重新缩放 [保存后将恢复原始位置]",
    "Resutling coordinates will be too big (original precision may be lost!). Proceed anyway?":
        "结果坐标将太大（原始精度可能丢失！）。仍要继续吗？",
    "Point (%1 ; %2 ; %3) set as rotation center for interactive transformation":
        "点 (%1 ; %2 ; %3) 设置为交互变换的旋转中心",
    
    # Mesh operations messages
    "Full Screen 3D mode has not been implemented yet!": "全屏3D模式尚未实现！",
    "Only meshes with standard vertices are handled for now! Can't merge entity '%1'...":
        "目前只处理具有标准顶点的网格！无法合并实体 '%1'...",
    "Entity '%1' is neither a cloud nor a mesh, can't merge it!":
        "实体 '%1' 既不是点云也不是网格，无法合并！",
    "Can't mix point clouds and meshes!": "无法混合点云和网格！",
    "Couldn't allocate a new scalar field for storing the original cloud index! Try to free some memory ...":
        "无法分配新的标量场来存储原始点云索引！尝试释放一些内存...",
    "Fusion failed! (not enough memory?)": "融合失败！（内存不足？）",
    
    # Picking and interaction messages
    "Can't start the picking mechanism (another tool is already using it)":
        "无法启动拾取机制（另一个工具正在使用它）",
    "[Level] Point is too close from the others!": "[水平仪] 点距其他点太近！",
    "Use best fit plane (yes) or the current viewing direction (no)":
        "使用最佳拟合平面（是）还是当前查看方向（否）",
    
    # TBB message
    "[TBB] Using Intel's Threading Building Blocks %1.%2":
        "[TBB] 使用Intel的线程构建模块 %1.%2",
    
    # ccCompass specific
    "MCMC Stride (radians):": "MCMC步幅（弧度）：",
    "The minimum size of the normal-estimation window.": "法线估计窗口的最小尺寸。",
    "The maximum size of the normal-estimation window.": "法线估计窗口的最大尺寸。",
    "Standard deviation of the normal distribution used to calculate monte-carlo jumps during sampling. Larger values lead to more exploration (and longer runtimes).":
        "用于计算采样期间蒙特卡罗跳跃的正态分布的标准偏差。较大的值导致更多探索（和更长的运行时间）。",
    "The voxel size for computing strain. This should be large enough that most boxes contain SNEs.":
        "用于计算应变的体素大小。这应该足够大，以便大多数盒子包含SNE。",
    "Use SNE orientation estimates for outside the current cell if none are avaliable within it.":
        "如果当前单元格内没有SNE方向估计，则使用外部的估计。",
    "Build graphic strain ellipses and grid domains. Useful for validation.":
        "构建图形应变椭圆和网格域。对验证有用。",
    "Exaggerate the shape of strain ellipses for easier visualisation.":
        "夸大应变椭圆的形状以便于可视化。",
    "The search radius used to define the region to compute P21 within.":
        "用于定义计算P21区域的搜索半径。",
    "Only sample P21 on the each n'th point in the original outcrop model (decreases calculation time).":
        "仅在原始露头模型中的每第n个点上采样P21（减少计算时间）。",
    "CSV files (*.csv *.txt);XML (*.xml)": "CSV文件 (*.csv *.txt);XML (*.xml)",
    
    # Keyboard shortcuts (usually keep as-is, but add context)
    "Ctrl+O": "Ctrl+O",
    "Ctrl+S": "Ctrl+S",
    "Ctrl+Q": "Ctrl+Q",
    "Ctrl+P": "Ctrl+P",
    "Alt+B": "Alt+B",
    "Alt+C": "Alt+C",
    "Del": "Del",
    "5": "5",
    "4": "4",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    
    # RasterizeToolDialog
    "size of step of the grid generated (in the same units as the coordinates of the point cloud)":
        "生成网格的步长（与点云坐标相同的单位）",
    "Active layer (or 'scalar field')": "活动图层（或\"标量场\"）",
    "SF interpolation method": "标量场插值方法",
    "Use the nearest point of the input cloud in each cell instead of the cell center":
        "使用输入点云中每个单元格的最近点，而不是单元格中心",
    "Per-cell height computation method:\n - minimum = lowest point in the cell\n - average = mean height of all points in the cell (distance to the 'average plane')\n - maximum = highest point in the cell":
        "每单元高度计算方法：\n - 最小值 = 单元格中的最低点\n - 平均值 = 单元格中所有点的平均高度（到\"平均平面\"的距离）\n - 最大值 = 单元格中的最高点",
    "choose the value to fill the cells in which no point is projected : minimum value over the whole point cloud or NaN":
        "选择填充没有点投影的单元格的值：整个点云的最小值或NaN",
    "The contour plot is computed on the active layer": "轮廓图在活动图层上计算",
    "project contours on the altitude layer": "将轮廓投影到高程图层",
    "Hillshade is computed on the height layer": "山体阴影在高度图层上计算",
    "Zenith angle (in degrees) = 90 - altitude angle": "天顶角（度）= 90 - 高度角",
    " deg.": " 度",
    
    # qCanupoPlugin
    "Load a previously saved classifier file.": "加载先前保存的分类器文件。",
    "Save the current classifier to a file.": "将当前分类器保存到文件。",
    "Train a new classifier using the current core points.": "使用当前核心点训练新分类器。",
    "Classify the point cloud using the current classifier.": "使用当前分类器对点云进行分类。",
    "Clear all core point clouds.": "清除所有核心点云。",
    "Remove the selected core point cloud.": "移除选中的核心点云。",
    "Add a new class and its core points.": "添加新类别及其核心点。",
    "The confidence threshold for classification.": "分类的置信度阈值。",
    "Points with confidence below this threshold will be unclassified.": "置信度低于此阈值的点将不被分类。",
    "Use the original cloud for descriptor computation.": "使用原始点云进行描述符计算。",
    "Set this cloud for descriptor computation.": "设置此点云用于描述符计算。",
    "Multi-scale dimensionality descriptors.": "多尺度维度描述符。",
    "The scales at which to compute descriptors.": "计算描述符的尺度。",
    "Smaller scales capture fine details.": "较小的尺度捕获精细细节。",
    "Larger scales capture broader features.": "较大的尺度捕获更广泛的特征。",
    
    # DistanceMapGenerationDlg
    "Compute distance map": "计算距离图",
    "Distance computation type": "距离计算类型",
    "Signed distances (inside/outside)": "有符号距离（内部/外部）",
    "Unsigned distances": "无符号距离",
    "Euclidean distance": "欧氏距离",
    "Manhattan distance": "曼哈顿距离",
    "Chebyshev distance": "切比雪夫距离",
    
    # M3C2Dialog
    "Core points cloud": "核心点云",
    "Normal scale": "法线尺度",
    "Projection scale": "投影尺度",
    "Max depth": "最大深度",
    "Cylindrical projection": "圆柱投影",
    "Use cloud normals": "使用点云法线",
    "Compute normals": "计算法线",
    "Registration error": "配准误差",
    "Significance level": "显著性水平",
    
    # RegistrationDialog
    "Reference cloud": "参考点云",
    "Aligned cloud": "对齐点云",
    "Random sampling limit": "随机采样限制",
    "Final overlap": "最终重叠",
    "Use random sampling": "使用随机采样",
    "Adjust scale": "调整比例",
    "Number of iterations": "迭代次数",
    "Convergence criterion": "收敛准则",
    "Final RMS": "最终均方根",
    "Final transformation": "最终变换",
    
    # VolumeCalcDialog
    "Ground level": "地面高程",
    "Ceiling level": "顶面高程",
    "Constant": "常量",
    "From cloud": "来自点云",
    "From mesh": "来自网格",
    "Volume report": "体积报告",
    "Volume above": "上方体积",
    "Volume below": "下方体积",
    "Total volume": "总体积",
    "Surface area": "表面积",
    "Average height": "平均高度",
    
    # SACSegmentation  
    "Model type": "模型类型",
    "Sphere model": "球体模型",
    "Cylinder model": "圆柱模型",
    "Cone model": "圆锥模型",
    "Plane model": "平面模型",
    "Distance threshold": "距离阈值",
    "Max iterations": "最大迭代次数",
    "Probability": "概率",
    "Extract inliers": "提取内点",
    "Extract outliers": "提取外点",
    "Inlier count": "内点数量",
    "Model coefficients": "模型系数",
    
    # qSRA
    "Profile comparison": "剖面比较",
    "Reference profile": "参考剖面",
    "Compare profile": "比较剖面",
    "Roughness analysis": "粗糙度分析",
    "Profile length": "剖面长度",
    "Roughness index": "粗糙度指数",
    
    # qRansacSD
    "Detect primitives": "检测基元",
    "Primitive types": "基元类型",
    "Min support points": "最小支撑点数",
    "Sampling resolution": "采样分辨率",
    "Max normal deviation": "最大法线偏差",
    "Overlook probability": "忽略概率",
    "Detected shapes": "检测到的形状",
    
    # DisplayOptionsDlg
    "Point size": "点大小",
    "Line width": "线宽",
    "Default font": "默认字体",
    "Label font size": "标签字体大小",
    "Number precision": "数字精度",
    "Decimal places": "小数位数",
    "Background color": "背景颜色",
    "Text color": "文本颜色",
    "Point color": "点颜色",
    "Line color": "线颜色",
    
    # GeomFeaturesDialog
    "Compute roughness": "计算粗糙度",
    "Compute curvature": "计算曲率",
    "Compute density": "计算密度",
    "Kernel radius": "核半径",
    "Feature type": "特征类型",
    "Gaussian curvature": "高斯曲率",
    "Mean curvature": "平均曲率",
    "Normal change rate": "法线变化率",
    "Surface density": "表面密度",
    
    # InterpolationDlg
    "Interpolation method": "插值方法",
    "IDW (Inverse Distance Weighting)": "IDW（反距离加权）",
    "Kriging": "克里金",
    "Natural Neighbor": "自然邻域",
    "Power parameter": "幂次参数",
    "Search radius": "搜索半径",
    "Min neighbors": "最小邻域数",
    "Max neighbors": "最大邻域数",
    "Interpolate scalar field": "插值标量场",
    "Output grid": "输出网格",
    
    # PoissonReconParamDialog
    "Octree depth": "八叉树深度",
    "Solver divide": "求解器划分",
    "Samples per node": "每节点采样数",
    "Point weight": "点权重",
    "Trim threshold": "修剪阈值",
    "Linear fit": "线性拟合",
    "Density threshold": "密度阈值",
    "Boundary type": "边界类型",
    "Free boundary": "自由边界",
    "Dirichlet boundary": "狄利克雷边界",
    "Neumann boundary": "诺伊曼边界",
    
    # qHPR
    "Hidden point removal": "隐藏点移除",
    "Camera position": "相机位置",
    "Camera radius": "相机半径",
    "Remove hidden": "移除隐藏点",
    "Keep visible": "保留可见点",
    "Visible points": "可见点",
    "Hidden points": "隐藏点",
}

def translate_ts_file_round2(input_file, output_file):
    """Round 2 translation focusing on complete sentences and messages"""
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    translated_count = 0
    still_untranslated = []
    
    for context in root.findall('.//context'):
        context_name = context.find('name').text if context.find('name') is not None else "Unknown"
        
        for message in context.findall('message'):
            translation = message.find('translation')
            if translation is None:
                continue
            
            # Only process unfinished
            trans_type = translation.get('type', '')
            if trans_type != 'unfinished' and translation.text:
                continue
            
            source = message.find('source')
            if source is None or not source.text:
                continue
            
            source_text = source.text
            
            # Check exact match
            if source_text in ROUND2_TRANSLATIONS:
                translation.text = ROUND2_TRANSLATIONS[source_text]
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                translated_count += 1
            else:
                still_untranslated.append({
                    'context': context_name,
                    'source': source_text
                })
    
    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Save remaining for round 3
    with open('/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/scripts/remaining_round3.txt',
              'w', encoding='utf-8') as f:
        f.write(f"第3轮待翻译内容 ({len(still_untranslated)} 条)\n")
        f.write(f"{'='*80}\n\n")
        
        from collections import defaultdict
        by_context = defaultdict(list)
        for item in still_untranslated:
            by_context[item['context']].append(item['source'])
        
        for ctx, items in sorted(by_context.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"\n{ctx} ({len(items)} 条)\n")
            f.write(f"{'-'*80}\n")
            for i, source in enumerate(items[:30], 1):
                preview = source[:100].replace('\n', ' ')
                f.write(f"{i}. {preview}\n")
            if len(items) > 30:
                f.write(f"... 还有 {len(items)-30} 条\n")
    
    print(f"\n{'='*80}")
    print(f"第2轮翻译完成！")
    print(f"{'='*80}")
    print(f"本次翻译: {translated_count} 条")
    print(f"仍需翻译: {len(still_untranslated)} 条")
    print(f"详细列表: scripts/remaining_round3.txt")
    print(f"{'='*80}\n")
    
    return translated_count, len(still_untranslated)

if __name__ == "__main__":
    input_file = '/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/ACloudViewer_zh.ts'
    output_file = input_file
    
    translated, remaining = translate_ts_file_round2(input_file, output_file)
    
    print(f"✓ 第2轮成功翻译 {translated} 条")
    if remaining > 0:
        print(f"! 还有 {remaining} 条需要第3轮处理")
    else:
        print(f"🎉 100%覆盖率达成！")
