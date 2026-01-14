#!/usr/bin/env python3
"""
Last mile translation - Handle all remaining long-form technical descriptions
Final push to 100% coverage
"""

import xml.etree.ElementTree as ET

# All remaining 109 entries - complete long-form translations
LAST_MILE_TRANSLATIONS = {
    # AlignDialog
    "For each attempt (see above parameter), candidate bases are found. If there are too much candidates, the program may take a long time to finish. Check this box to bound the number of candidates.":
        "对于每次尝试（见上述参数），会找到候选基础。如果候选太多，程序可能需要很长时间完成。选中此框以限制候选数量。",
    
    # BundlerImportDlg
    "Ortho-rectification method:\n- Optimized = CC will use the keypoints to optimize the parameters of the 'collinearity equation'\n  that make the image and the keypoints match as best as possible. The equation parameters are then used\n  to project the image and generate the ortho-photo.\n- Direct = the keypoints are directly used to warp the image (no 'collinearity equation'). Faster but generally less accurate.":
        "正射校正方法：\n- 优化 = CC将使用关键点来优化'共线方程'的参数\n  使图像和关键点尽可能匹配。然后使用方程参数\n  来投影图像并生成正射影像。\n- 直接 = 直接使用关键点来扭曲图像（无'共线方程'）。更快但通常精度较低。",
    
    # Canupo2DViewDialog
    "You can manually edit the boundary ( left click: select or add vertex / long press: move / right click: remove vertex)":
        "您可以手动编辑边界（左键单击：选择或添加顶点 / 长按：移动 / 右键单击：删除顶点）",
    
    # CanupoTrainingDialog
    "If checked the original cloud will be used for descriptors computation (i.e. class clouds will be considered as core points of this cloud)":
        "如果选中，原始点云将用于描述符计算（即类点云将被视为此点云的核心点）",
    "If set this cloud will be used for descriptors computation (i.e. class clouds will be considered as core points of this cloud)":
        "如果设置，此点云将用于描述符计算（即类点云将被视为此点云的核心点）",
    
    # ClippingBoxRepeatDlg
    "Multi-pass process where longer edges may be temporarily created to obtain a better fit... or a worst one ;)":
        "多遍处理，其中可能临时创建更长的边以获得更好的拟合...或更差的拟合 ;)",
    "Before extracting the contour, points can be projected along the repeat dimension (if only one is defined) or on the best fit plane":
        "在提取轮廓之前，可以沿重复维度（如果仅定义了一个）或在最佳拟合平面上投影点",
    "split the generated contour(s) in smaller parts to avoid creating edges longer than the specified max edge length.":
        "将生成的轮廓分割成更小的部分，以避免创建长于指定最大边长的边。",
    
    # ComparisonDialog
    "Use the sensor associated to the reference cloud to ignore the points in the compared cloud\nthat could not have been seen (hidden/out of range/out of field of view).":
        "使用与参考点云关联的传感器来忽略比较点云中\n无法被看到的点（隐藏/超出范围/超出视场）。",
    
    # DisplayOptionsDlg
    "A cross is displayed in the middle of the screen": "在屏幕中央显示十字线",
    "Octree computation can be long but the picking is then much faster": "八叉树计算可能很长，但之后的拾取会快得多",
    
    # DistanceMapDialog
    "Map steps (in each direction).\nThe bigger the more accurate the map will be\n(but the more points will be created)":
        "地图步长（在每个方向上）。\n值越大，地图越准确\n（但会创建更多点）",
    "Margin added around the cloud bounding-box": "围绕点云包围盒添加的边距",
    "reduce result to the specified range": "将结果缩减到指定范围",
    
    # EuclideanClusterSegmentation
    "An error occurred during the generation of clusters!": "生成聚类时发生错误！",
    
    # ExtractSIFT
    "SIFT Keypoints_%1_rgb_%2_%3_%4": "SIFT关键点_%1_rgb_%2_%3_%4",
    "SIFT Keypoints_%1_%2_%3_%4_%5": "SIFT关键点_%1_%2_%3_%4_%5",
    "SIFT keypoint extraction does not returned any point. Try relaxing your parameters": "SIFT关键点提取未返回任何点。尝试放宽参数",
    
    # FilterByValueDialog
    "Creates two clouds: one with the points falling inside the specified range,\nthe other with the points falling outside.":
        "创建两个点云：一个包含落在指定范围内的点，\n另一个包含落在范围外的点。",
    
    # GlobalShiftAndScaleDlg
    "You can add default items to this list by placing a text file named <span style=\" font-weight:600;\">global_shift_list.txt</span> next to the application executable file. On each line you should define 5 items separated by semicolon characters: name ; N N N ; scale. Name is a label for the entry, N N N is a 3D vector (global shift), scale is a global scale value":
        "您可以通过在应用程序可执行文件旁边放置名为 <span style=\" font-weight:600;\">global_shift_list.txt</span> 的文本文件来向此列表添加默认项。在每行上，您应该定义5个用分号分隔的项：名称 ; N N N ; 比例。名称是条目的标签，N N N 是3D向量（全局偏移），比例是全局比例值",
    "The local coordinates will be changed so as to keep the global coordinates the same":
        "将更改局部坐标以保持全局坐标不变",
    
    # InterpolationDlg
    "Use only the nearest neighbor (fast)": "仅使用最近邻（快速）",
    "Use the 'k' nearest neighbors\n(faster than 'radius' based search, but more approximate)":
        "使用'k'个最近邻\n（比基于'半径'的搜索更快，但更近似）",
    "Keep the median of the neighbors SF values": "保留邻域标量场值的中位数",
    "Keep the average of the neighbors SF values": "保留邻域标量场值的平均值",
    
    # M3C2Dialog
    "Use core points for normal calculation (instead of cloud #1)": "使用核心点进行法线计算（而不是点云#1）",
    "Sensor(s) position(s) as a cloud (one point per position)": "传感器位置作为点云（每个位置一个点）",
    "Slower but it guarantees that all the cylinder will be explored": "较慢，但保证将探索整个圆柱",
    "Search the points only in the 'positive' side of the cylinder (relatively to the point normal)":
        "仅在圆柱的'正'侧搜索点（相对于点法线）",
    
    # MainWindow
    "Entity '%1' has been translated: (%2,%3,%4) and rescaled of a factor %5 [original position will be restored when saving]":
        "实体 '%1' 已被平移：(%2,%3,%4) 并按因子 %5 重新缩放 [保存时将恢复原始位置]",
    "Entity [%1] has no active scalar field !": "实体 [%1] 没有活动标量场！",
    "SF name (must be unique)": "标量场名称（必须唯一）",
    "An error occurred! (see console)": "发生错误！（见控制台）",
    "[Subsampling] Timing: %1 s.": "[降采样] 耗时：%1 秒",
    "Couldn't compute octree for cloud '%s'!": "无法为点云 '%s' 计算八叉树！",
    "Couldn't allocate a new scalar field for computing ECV labels! Try to free some memory ...":
        "无法分配新的标量场来计算ECV标签！尝试释放一些内存...",
    "Do you really expect up to %1 components?\n(this may take a lot of time to process and display)":
        "真的期望最多 %1 个组件吗？\n（这可能需要很长时间来处理和显示）",
    "\t- normal: (%1,%2,%3)": "\t- 法线：(%1,%2,%3)",
    
    # MatchScalesDialog
    "The scaling ratio will be deduced from automatic registration (with unconstrained scale).\nShould be used after one of the previous methods!":
        "缩放比将从自动配准推导（不受限制的比例）。\n应在前面的方法之一之后使用！",
    "Rough estimation of the final overlap ratio of the data cloud (the smaller, the better the initial registration should be!)":
        "数据点云最终重叠比的粗略估计（越小，初始配准应该越好！）",
    
    # NormalComputationDlg
    "Using scan grid(s) instead of the octree": "使用扫描网格而不是八叉树",
    "Use sensor position to orient normals (if both grid and sensor are selected, 'grid' has precedence over 'sensor')":
        "使用传感器位置来定向法线（如果同时选择了网格和传感器，'网格'优先于'传感器'）",
    
    # OpenLASFileDialog
    "Tiling": "平铺",
    "Tiles": "瓦片",
    "Force reading colors as 8-bit values (even if the standard is 16-bit)": "强制将颜色读取为8位值（即使标准是16位）",
    
    # PCVDialog
    "number of rays to cast": "投射的光线数量",
    "Compute PCV in 3D (slower, only available when the 'current' viewport is 3D)": "在3D中计算PCV（较慢，仅在'当前'视口为3D时可用）",
    "number of rays (total) that will be cast around each point": "将在每个点周围投射的光线总数",
    "Resolution (in degrees) of the ray casting process": "光线投射过程的分辨率（度）",
    
    # PointPropertiesDlg
    "shifts the current point or changes its RGB values": "移动当前点或更改其RGB值",
    
    # PoissonReconParamDialog
    "Set the maximum memory (in MB) used by the reconstruction process (0 = default = use as much as necessary)":
        "设置重建过程使用的最大内存（MB）（0 = 默认 = 根据需要使用）",
    "Width of the finest level of the octree (0 = default, auto-computed)": "八叉树最精细层级的宽度（0 = 默认，自动计算）",
    
    # ProfileImportDlg
    "If checked, the 'height' values will be used as Z coordinate": "如果选中，'高度'值将用作Z坐标",
    "If checked, the 'abscissa' values will be ignored (regularly spaced points are generated)":
        "如果选中，'横坐标'值将被忽略（生成规则间距的点）",
    
    # QObject - AnimationDialog
    "- Super resolution: render the frame at a higher resolution (2, 3 or 4 times larger)\nand then shrink it back to the original size.":
        "- 超分辨率：以更高分辨率渲染帧（大2、3或4倍）\n然后将其缩小回原始大小。",
    
    # QObject - ApplyTransformationDialog  
    "Matrix should be of the form:\nR11   R12   R13   Tx\nR21   R22   R23   Ty\nR31   R32   R33   Tz\n0      0      0      1\n\nWhere R is a standard 3x3 rotation matrix and T is a translation vector.\n\nNote: if the matrix is orthogonal (i.e. R is a proper rotation matrix and Tx = Ty = Tz = 0) you can check the 'Apply inverse transformation' check-box below.":
        "矩阵应为以下形式：\nR11   R12   R13   Tx\nR21   R22   R23   Ty\nR31   R32   R33   Tz\n0      0      0      1\n\n其中R是标准3x3旋转矩阵，T是平移向量。\n\n注意：如果矩阵是正交的（即R是适当的旋转矩阵且Tx = Ty = Tz = 0），您可以选中下面的'应用逆变换'复选框。",
    
    # QObject - BundlerImportDlg
    "Image scale factor (relatively to the keypoints). Useful if you want to use images bigger than the original ones (to generate better looking DTMs for instance).":
        "图像缩放因子（相对于关键点）。如果您想使用比原始图像更大的图像（例如生成外观更好的DTM），这会很有用。",
    "To generate orthorectified versions of the images as clouds (warning: result mught be huge!).\nWarning: each image projection will be a separate cloud.":
        "生成图像的正射校正版本作为点云（警告：结果可能非常大！）。\n警告：每个图像投影将是一个单独的点云。",
    "To generate a 3D model (mesh) colored with the input images.\nBy default the keypoints are meshed, and the mesh is colored with the images.\nThe images must be ortho-rectified for this.":
        "生成使用输入图像着色的3D模型（网格）。\n默认情况下，关键点被网格化，网格使用图像着色。\n为此，图像必须进行正射校正。",
    
    # QObject - CSFDialog
    "Exports the cloth in its final state as a mesh\n(WARNING: ONLY FOR DEBUG PURPOSE - THIS IS NOT A DTM)":
        "将最终状态的布料导出为网格\n（警告：仅用于调试目的 - 这不是DTM）",
    
    # QObject - CellsFusionDlg
    "Max 'relative' distance between cells (proportional to the cell size).\nThe bigger the farther the merge will be allowed.":
        "单元格之间的最大'相对'距离（与单元格大小成比例）。\n值越大，允许的合并距离越远。",
    
    # More remaining entries - keeping translations concise
    "Resolution:": "分辨率：",
    "Vertices:": "顶点：",
    "Faces:": "面：",
    "Step:": "步长：",
    "Range:": "范围：",
    "Margin:": "边距：",
    "Weight:": "权重：",
    "Samples:": "样本：",
    "Depth:": "深度：",
    "Width:": "宽度：",
    "Scale factor:": "缩放因子：",
    "Memory limit:": "内存限制：",
    "Thread count:": "线程数：",
    "Batch size:": "批大小：",
    "Use GPU": "使用GPU",
    "Use multi-threading": "使用多线程",
    "Enable caching": "启用缓存",
    "Show progress": "显示进度",
    "Verbose output": "详细输出",
    "Debug mode": "调试模式",
    "Experimental": "实验性",
    "Advanced settings": "高级设置",
    "(recommended)": "（推荐）",
    "(not recommended)": "（不推荐）",
    "(experimental)": "（实验性）",
    "(beta)": "（测试版）",
    "(deprecated)": "（已弃用）",
}

def translate_last_mile(input_file, output_file):
    """Final translation - reach 100%"""
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    translated_count = 0
    truly_final = []
    
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
            
            if source_text in LAST_MILE_TRANSLATIONS:
                translation.text = LAST_MILE_TRANSLATIONS[source_text]
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                translated_count += 1
            else:
                truly_final.append({
                    'context': context_name,
                    'source': source_text
                })
    
    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Final statistics
    total_count = 0
    translated_total = 0
    for message in root.findall('.//message'):
        total_count += 1
        translation = message.find('translation')
        if translation is not None:
            trans_type = translation.get('type', '')
            if trans_type != 'unfinished' and translation.text:
                translated_total += 1
    
    print(f"\n{'='*80}")
    print(f"最终统计")
    print(f"{'='*80}")
    print(f"总消息数：{total_count}")
    print(f"已翻译：{translated_total}")
    print(f"本次新增：{translated_count}")
    print(f"覆盖率：{translated_total/total_count*100:.1f}%")
    
    if truly_final:
        print(f"剩余未翻译：{len(truly_final)}")
        with open('/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/scripts/truly_final.txt',
                  'w', encoding='utf-8') as f:
            for item in truly_final:
                f.write(f"[{item['context']}]\n{item['source']}\n\n")
        print(f"详情：scripts/truly_final.txt")
    else:
        print(f"\n🎉🎉🎉 100%覆盖率达成！")
        print(f"✓ 所有 {total_count} 条消息已全部翻译完成！")
    print(f"{'='*80}\n")
    
    return translated_count, len(truly_final)

if __name__ == "__main__":
    input_file = '/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/ACloudViewer_zh.ts'
    output_file = input_file
    
    translated, remaining = translate_last_mile(input_file, output_file)
