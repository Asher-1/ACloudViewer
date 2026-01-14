#!/usr/bin/env python3
"""
Round 3 Translation - Handle remaining 266 entries
Focus on error messages with parameters, console outputs, and technical descriptions
"""

import xml.etree.ElementTree as ET
import re

# Round 3: Complex error messages, console outputs, technical formulas
ROUND3_TRANSLATIONS = {
    # Complex error messages with parameters
    "Hum, it seems that ECV has crashed... Sorry about that :) ": "嗯，看起来ECV崩溃了...对此感到抱歉 :) ",
    "Missing parameter: number of lines after '%1'": "缺少参数：'%1' 后的行数",
    "Missing parameter: global shift vector or %1 after '%2'": "缺少参数：'%2' 后的全局偏移向量或 %1",
    "Missing parameter: global shift vector after '%1' (3 values expected)": "缺少参数：'%1' 后的全局偏移向量（需要3个值）",
    "Missing parameter: radius after \"-%1\"": "缺少参数：\"-%1\" 后的半径",
    "Missing parameter: resampling method after \"-%1\"": "缺少参数：\"-%1\" 后的重采样方法",
    "Missing parameter: number of points after \"-%1 RANDOM\"": "缺少参数：\"-%1 RANDOM\" 后的点数",
    "\tResult: %1 points": "\t结果：%1 个点",
    "Missing parameter: spatial step after \"-%1 SPATIAL\"": "缺少参数：\"-%1 SPATIAL\" 后的空间步长",
    "\tSpatial step: %1": "\t空间步长：%1",
    "Missing parameter: octree level after \"-%1 OCTREE\"": "缺少参数：\"-%1 OCTREE\" 后的八叉树层级",
    "\tOctree level: %1": "\t八叉树层级：%1",
    "OCTREE_LEVEL_%1_SUBSAMPLED": "八叉树层级_%1_降采样",
    "Missing parameter: octree level after \"-%1\"": "缺少参数：\"-%1\" 后的八叉树层级",
    "Missing parameter: minimum number of points per component after \"-%1 [octree level]\"": "缺少参数：\"-%1 [八叉树层级]\" 后每个组件的最小点数",
    "%1 component(s) were created": "创建了 %1 个组件",
    "Missing parameter: curvature type after \"-%1\"": "缺少参数：\"-%1\" 后的曲率类型",
    "\tKernel size: %1": "\t核大小：%1",
    "%1_CURVATURE_KERNEL_%2": "%1_曲率_核_%2",
    "Missing parameter: density type after \"-%1\" (KNN/SURFACE/VOLUME)": "缺少参数：\"-%1\" 后的密度类型（KNN/SURFACE/VOLUME）",
    "Missing parameter: sphere radius after \"-%1\"": "缺少参数：\"-%1\" 后的球半径",
    "Missing parameter: boolean (whether SF is euclidean or not) after \"-%1\"": "缺少参数：\"-%1\" 后的布尔值（标量场是否为欧氏）",
    "cmd.warning: cloud '%1' has no scalar field (it will be ignored)": "命令警告：点云 '%1' 没有标量场（将被忽略）",
    "cmd.warning: cloud '%1' has several scalar fields (the active one will be used by default, or the first one if none is active)": "命令警告：点云 '%1' 有多个标量场（默认使用活动的，如果没有活动的则使用第一个）",
    "Missing parameter: kernel size after \"-%1\"": "缺少参数：\"-%1\" 后的核大小",
    "Missing parameter: transformation file after \"-%1\"": "缺少参数：\"-%1\" 后的变换文件",
    "Missing parameter: color scale file after \"-%1\"": "缺少参数：\"-%1\" 后的色阶文件",
    "Missing parameter: boolean (whether to mix with existing colors or not) after \"-%1\"": "缺少参数：\"-%1\" 后的布尔值（是否与现有颜色混合）",
    "cmd.warning: cloud '%1' has no active scalar field (it will be ignored)": "命令警告：点云 '%1' 没有活动标量场（将被忽略）",
    "cmd.warning: cloud '%1' failed to convert SF to RGB": "命令警告：点云 '%1' 无法将标量场转换为RGB",
    
    # MainWindow error messages
    "Entity '%1' has been translated: (%2,%3,%4) and rescaled of a factor %5 [original position will be restored after saving]": 
        "实体 '%1' 已被平移：(%2,%3,%4) 并按因子 %5 重新缩放 [保存后将恢复原始位置]",
    "[Subdivide] An error occurred while trying to subdivide mesh '%1' (not enough memory?)": "[细分] 尝试细分网格 '%1' 时发生错误（内存不足？）",
    "[Subdivide] Works only on real meshes!": "[细分] 仅适用于真实网格！",
    "[changeLanguage] Change to English language": "[切换语言] 切换到英语",
    "[changeLanguage] Doesn't support Chinese temporarily": "[切换语言] 暂不支持中文",
    "An error occurred while cloning cloud %1": "克隆点云 %1 时发生错误",
    "An error occurred while cloning primitive %1": "克隆基元 %1 时发生错误",
    "An error occurred while cloning mesh %1": "克隆网格 %1 时发生错误",
    "An error occurred while cloning polyline %1": "克隆折线 %1 时发生错误",
    "An error occurred while cloning facet %1": "克隆面片 %1 时发生错误",
    "Entity '%1' can't be cloned (type not supported yet!)": "实体 '%1' 无法克隆（类型暂不支持！）",
    "This method is for test purpose only": "此方法仅用于测试",
    "Couldn't allocate a new scalar field for computing distances! Try to free some memory ...": "无法分配新的标量场来计算距离！尝试释放一些内存...",
    "This method is still under development: are you sure you want to use it? (a crash may likely happen)": "此方法仍在开发中：确定要使用它吗？（可能会崩溃）",
    "[Align] Resulting matrix:": "[对齐] 结果矩阵：",
    "[Register] ": "[配准] ",
    "[Register] Applied transformation matrix:": "[配准] 应用的变换矩阵：",
    "Theoretical overlap: %1%": "理论重叠：%1%",
    "This report has been output to Console (F8)": "此报告已输出到控制台（F8）",
    "Data mesh vertices are locked (they may be shared with other meshes): Do you wish to clone this mesh to apply transformation?": "数据网格顶点已锁定（它们可能与其他网格共享）：是否要克隆此网格以应用变换？",
    "Doesn't work on sub-meshes yet!": "尚不支持子网格！",
    "Drop shift information?": "放弃偏移信息？",
    "Spherical extraction test (%1)": "球形提取测试 (%1)",
    "Couldn't compute octree for cloud '%1'!": "无法为点云 '%1' 计算八叉树！",
    "[SNE_TEST] Mean extraction time = %1 ms (radius = %2, mean(neighbours) = %3)": "[SNE_TEST] 平均提取时间 = %1 毫秒（半径 = %2，平均邻域 = %3）",
    "[CNE_TEST] Mean extraction time = %1 ms (radius = %2, height = %3, mean(neighbours) = %4)": "[CNE_TEST] 平均提取时间 = %1 毫秒（半径 = %2，高度 = %3，平均邻域 = %4）",
    "Need at least two clouds!": "至少需要两个点云！",
    "%1 clouds and %2 positions": "%1 个点云和 %2 个位置",
    "Orthogonal dim (X=0 / Y=1 / Z=2)": "正交维度（X=0 / Y=1 / Z=2）",
    "%1 (%2 values) ": "%1（%2 个值）",
    
    # qFacets errors
    "closing facets dialog failed! [%1]": "关闭面片对话框失败！[%1]",
    "Internal error: invalid algorithm type!": "内部错误：无效的算法类型！",
    "Couldn't allocate a new scalar field for computing fusion labels! Try to free some memory ...": "无法分配新的标量场来计算融合标签！尝试释放一些内存...",
    " [Kd-tree][error < %1][angle < %2 deg.]": " [Kd树][误差 < %1][角度 < %2 度]",
    " [FM][level %2][error < %1]": " [FM][层级 %2][误差 < %1]",
    "An error occurred during the generation of facets!": "生成面片时发生错误！",
    "An error occurred during the fusion process!": "融合过程中发生错误！",
    " [facets]": " [面片]",
    "Couldn't find any facet in the current selection!": "在当前选择中找不到任何面片！",
    "An error occurred while classifying the facets! (not enough memory?)": "分类面片时发生错误！（内存不足？）",
    
    # DistanceMapGenerationDlg
    "Map angular step (horizontal)": "地图角度步长（水平）",
    "Map height step (vertical)": "地图高度步长（垂直）",
    "Map heights unit (for display only)": "地图高度单位（仅用于显示）",
    "m.": "米",
    "What to do when multiple values fall in the same grid cell?": "当多个值落入同一网格单元时该怎么办？",
    "What to do when a grid cell remains empty?": "当网格单元保持为空时该怎么办？",
    " m.": " 米",
    "Generatrix direction (in the 3D world)": "母线方向（在3D世界中）",
    "Mean radius (for map display, export as a cloud, etc. )": "平均半径（用于地图显示、导出为点云等）",
    
    # qCanupoPlugin errors
    "Internal error: failed to access core pointss?!": "内部错误：无法访问核心点！",
    ".core points (subsampled @ %1)": ".核心点（降采样于 %1）",
    "Can't save subsampled cloud (not enough memory)!": "无法保存降采样点云（内存不足）！",
    "[qCanupo] ": "[qCanupo] ",
    "Internal error: no core point source specified?!": "内部错误：未指定核心点源！",
    "At least one cloud (class #1 or #2) was not defined!": "至少一个点云（类别#1或#2）未定义！",
    "[qCanupo] Some descriptors couldn't be computed on cloud#1 (min scale may be too small)!": "[qCanupo] 某些描述符无法在点云#1上计算（最小尺度可能太小）！",
    "[qCanupo] Some descriptors couldn't be computed on cloud#2 (min scale may be too small)!": "[qCanupo] 某些描述符无法在点云#2上计算（最小尺度可能太小）！",
    "[qCanupo] Some descriptors couldn't be computed on evaluation cloud (min scale may be too small)!": "[qCanupo] 某些描述符无法在评估点云上计算（最小尺度可能太小）！",
    
    # GeomFeaturesDialog - Technical formulas
    "Number of neighbors / neighborhood area": "邻域数量 / 邻域面积",
    "Number of neighbors / neighborhood volume": "邻域数量 / 邻域体积",
    "Geometric features (based on local eigenvalues: (L1, L2, L3))": "几何特征（基于局部特征值：(L1, L2, L3)）",
    "(L1 * L2 * L3)^(1/3)": "(L1 * L2 * L3)^(1/3)",
    "-( L1*ln(L1) + L2*ln(L2) + L3*ln(L3) )": "-( L1*ln(L1) + L2*ln(L2) + L3*ln(L3) )",
    "(L1 - L3)/L1": "(L1 - L3)/L1",
    "(L2 - L3)/L1": "(L2 - L3)/L1",
    "(L1 - L2)/L1": "(L1 - L2)/L1",
    
    # RegistrationDialog - Long descriptions
    "the data cloud is the entity to align with the model cloud : it will be displaced (red cloud)": "数据点云是要与模型点云对齐的实体：它将被移动（红色点云）",
    "the model cloud is the reference : it won't move (yellow cloud)": "模型点云是参考：它不会移动（黄色点云）",
    "By choosing this criterion, you can control the computation time.": "通过选择此标准，您可以控制计算时间。",
    "By choosing this criterion, you can control the quality of the result.": "通过选择此标准，您可以控制结果的质量。",
    "Rough estimation of the final overlap ratio of the data cloud (the smaller, the better the initial registration should be)": "数据点云最终重叠比的粗略估计（越小，初始配准应该越好）",
    "Whether to adjust the scale of the 'data' entity": "是否调整'数据'实体的比例",
    "Chose this option to remove points that are likely to disturb the registration during the computation (that do not belong to any plane)": "选择此选项以移除可能在计算期间干扰配准的点（不属于任何平面的点）",
    
    # GreedyTriangulation
    "Greedy Triangulation from clouds": "贪婪三角化（从点云）",
    "[GreedyTriangulation::compute] generate new normals": "[贪婪三角化::计算] 生成新法线",
    "[GreedyTriangulation::compute] find normals and use the normals": "[贪婪三角化::计算] 查找并使用法线",
    "[greedy-triangulation-Reconstruction] %1 points, %2 face(s)": "[贪婪三角化-重建] %1 个点，%2 个面",
    "Greedy Triangulation does not returned any point. Try relaxing your parameters": "贪婪三角化未返回任何点。尝试放宽参数",
    
    # MatchScalesDialog
    "The scaling ratio will be deduced from the largest bounding-box dimension": "缩放比将从最大包围盒维度推导",
    "The scaling ratio will be deduced from the bounding-box volume": "缩放比将从包围盒体积推导",
    "The scaling ratio will be deduced from the principal cloud dimension (by PCA analysis)": "缩放比将从主点云维度推导（通过PCA分析）",
    "The scaling ratio will be deduced from automatic registration (with unconstrained scale). Should be used only with very similar entities!": "缩放比将从自动配准推导（不受限制的比例）。应仅用于非常相似的实体！",
    "Rough estimation of the final overlap ratio of the data cloud (the smaller, the better the initial registration should be)": "数据点云最终重叠比的粗略估计（越小，初始配准应该越好）",
    
    # PoissonReconParamDialog
    "The maximum depth of the tree that will be used for surface reconstruction": "用于表面重建的树的最大深度",
    "If this flag is enabled, the sampling density is output as a scalar field": "如果启用此标志，采样密度将作为标量场输出",
    
    # More MainWindow messages
    "Entity '%1' has its coordinate center still shifted": "实体 '%1' 的坐标中心仍有偏移",
    "Shift on loading: (%1;%2;%3)": "加载时的偏移：(%1;%2;%3)",
    "Shift on input: (%1;%2;%3)": "输入时的偏移：(%1;%2;%3)",
    "Entity is too big to be correctly displayed": "实体太大，无法正确显示",
    "To reduce display shifts and artifacts, you can apply a global shift": "为减少显示偏移和伪影，您可以应用全局偏移",
    "Global shift has been defined by user": "全局偏移已由用户定义",
    "Global shift has been defined automatically": "全局偏移已自动定义",
    
    # More QObject messages
    "Invalid parameter: %1": "无效参数：%1",
    "Invalid command: %1": "无效命令：%1",
    "Command '%1' requires at least %2 argument(s)": "命令 '%1' 需要至少 %2 个参数",
    "Command '%1' requires exactly %2 argument(s)": "命令 '%1' 需要确切 %2 个参数",
    "Unknown file extension: %1": "未知文件扩展名：%1",
    "File not found: %1": "文件未找到：%1",
    "Failed to load file: %1": "加载文件失败：%1",
    "Failed to save file: %1": "保存文件失败：%1",
    "Operation cancelled by user": "操作被用户取消",
    "Processing...": "处理中...",
    "Initializing...": "初始化中...",
    "Finalizing...": "完成中...",
}

def translate_ts_file_round3(input_file, output_file):
    """Round 3 translation - final push to 100%"""
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    translated_count = 0
    final_untranslated = []
    
    for context in root.findall('.//context'):
        context_name = context.find('name').text if context.find('name') is not None else "Unknown"
        
        for message in context.findall('message'):
            translation = message.find('translation')
            if translation is None:
                continue
            
            trans_type = translation.get('type', '')
            if trans_type != 'unfinished' and translation.text:
                continue
            
            source = message.find('source')
            if source is None or not source.text:
                continue
            
            source_text = source.text
            
            if source_text in ROUND3_TRANSLATIONS:
                translation.text = ROUND3_TRANSLATIONS[source_text]
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                translated_count += 1
            else:
                final_untranslated.append({
                    'context': context_name,
                    'source': source_text
                })
    
    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Save final untranslated
    if final_untranslated:
        with open('/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/scripts/final_untranslated.txt',
                  'w', encoding='utf-8') as f:
            f.write(f"最终待人工翻译内容 ({len(final_untranslated)} 条)\n")
            f.write(f"{'='*80}\n\n")
            
            from collections import defaultdict
            by_context = defaultdict(list)
            for item in final_untranslated:
                by_context[item['context']].append(item['source'])
            
            for ctx, items in sorted(by_context.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"\n{ctx} ({len(items)} 条)\n")
                f.write(f"{'-'*80}\n")
                for i, source in enumerate(items, 1):
                    f.write(f"{i}. {source}\n\n")
    
    print(f"\n{'='*80}")
    print(f"第3轮翻译完成！")
    print(f"{'='*80}")
    print(f"本次翻译: {translated_count} 条")
    print(f"最终剩余: {len(final_untranslated)} 条")
    if final_untranslated:
        print(f"详细列表: scripts/final_untranslated.txt")
    print(f"{'='*80}\n")
    
    return translated_count, len(final_untranslated)

if __name__ == "__main__":
    input_file = '/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/ACloudViewer_zh.ts'
    output_file = input_file
    
    translated, remaining = translate_ts_file_round3(input_file, output_file)
    
    print(f"✓ 第3轮成功翻译 {translated} 条")
    if remaining > 0:
        print(f"! 还有 {remaining} 条需要人工精细翻译（通常是非常特殊的技术内容或格式问题）")
        print(f"  建议使用Qt Linguist逐个处理")
    else:
        print(f"🎉🎉🎉 100%覆盖率达成！所有内容已翻译完成！")
