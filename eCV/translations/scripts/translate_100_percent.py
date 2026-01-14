#!/usr/bin/env python3
"""
100% Translation - Handle the final 42 entries to reach complete coverage
"""

import xml.etree.ElementTree as ET

# The final 42 entries to reach 100%
FINAL_42 = {
    # RasterizeToolDialog
    "Per-cell height computation method:\n - minimum = lowest point in the cell\n - average = mean height of all points inside the cell\n - maximum = highest point in the cell":
        "每单元高度计算方法：\n - 最小值 = 单元格中的最低点\n - 平均值 = 单元格内所有点的平均高度\n - 最大值 = 单元格中的最高点",
    "choose the value to fill the cells in which no point is projected : minimum value over the whole point cloud or average value (over the whole cloud also). The cell is filled with the constant scalar value you have defined above.":
        "选择填充没有点投影的单元格的值：整个点云的最小值或平均值（也是整个点云）。单元格将用您在上面定义的常量标量值填充。",
    
    # RegistrationDialog
    "Chose this option to remove points that are likely to disturb the registration during the computation.":
        "选择此选项以移除可能在计算期间干扰配准的点。",
    
    # SACSegmentation
    "SACMODEL_REGISTRATION": "SACMODEL_REGISTRATION",
    "SACMODEL_REGISTRATION_2D": "SACMODEL_REGISTRATION_2D",
    "An error occurred during the generation of segments!": "生成分割段时发生错误！",
    
    # SaveLASFileDialog
    "Ensures optimal accuracy (up to 10^-7 absolute)": "确保最佳精度（绝对精度达10^-7）",
    "(0,0,0)": "(0,0,0)",
    
    # SaveSHPFileDlg
    "The height of each polyline (considered as constant!) will be saved as a field in the associated DBF file":
        "每条折线的高度（视为常量！）将作为字段保存在关联的DBF文件中",
    
    # ScaleDialog
    "Whether the cloud (center) should stay at the same place or not (i.e. coordinates are multiplied directly)":
        "点云（中心）是否应保持在同一位置（即直接乘以坐标）",
    
    # StatisticalTestDialog
    "false rejection probability": "误拒概率",
    "neighbors used to compute observed local dist.": "用于计算观察到的局部距离的邻域",
    
    # SubsamplingDialog
    "The more on the left, the less points will be kept": "越靠左，保留的点越少",
    
    # TemplateAlignmentDialog
    "Inp": "输入",
    
    # TracePolyLineDlg
    "Snap size": "捕捉大小",
    "C": "C",
    
    # VolumeCalcDialog
    "choose the value to fill the cells in which no point is projected : minimum value over the whole point cloud or average value (over the whole cloud also). The cell is filled with the constant scalar value you have defined above.":
        "选择填充没有点投影的单元格的值：整个点云的最小值或平均值（也是整个点云）。单元格将用您在上面定义的常量标量值填充。",
    "Per-cell height computation method:\n - minimum = lowest point in the cell\n - average = mean height of all points inside the cell\n - maximum = highest point in the cell":
        "每单元高度计算方法：\n - 最小值 = 单元格中的最低点\n - 平均值 = 单元格内所有点的平均高度\n - 最大值 = 单元格中的最高点",
    "Update the grid / display / measurements": "更新网格 / 显示 / 测量",
    "At least one of the cloud is sparse! You should fill the empty cells...": "至少有一个点云是稀疏的！您应该填充空单元格...",
    "Numerical precision (output measurements, etc.)": "数值精度（输出测量等）",
    
    # ccCompass
    "Standard deviation of the normal distribution used to calculate monte-carlo jumps during sampling. Larger numbers sample more widely but are slower to converge.":
        "用于计算采样期间蒙特卡罗跳跃的正态分布的标准偏差。较大的数字采样更广泛，但收敛更慢。",
    
    # compassDlg
    "Activate compass mode to make structural measurements": "激活罗盘模式进行构造测量",
    "Activate map mode to define geological features": "激活地图模式定义地质特征",
    "Change tool and visibility settings": "更改工具和可见性设置",
    "Ctrl+Z": "Ctrl+Z",
    
    # mapDlg
    "Digitise to interior of geo-object": "数字化到地质对象内部",
    
    # pointPairRegistrationDlg
    "Constrains the translation along particular axes (warning: experimental)": "约束沿特定轴的平移（警告：实验性）",
    "Convert list to new cloud (and close dialog)": "将列表转换为新点云（并关闭对话框）",
    
    # qAnimation
    "%1 At least 2 viewports must be selected.": "%1 至少必须选择2个视口。",
    
    # qHPR
    "Couldn't compute octree!": "无法计算八叉树！",
    
    # qPCV
    "An error occurred during entity '%1' illumination!": "实体 '%1' 照明时发生错误！",
    "Entity '%1' normals have been automatically disabled": "实体 '%1' 的法线已自动禁用",
    
    # qPoissonRecon
    "Reconstruction in progress ": "重建进行中 ",
    " [%1 thread(s)]": " [%1 个线程]",
    "Reconstruction failed!": "重建失败！",
    
    # qRansacSD
    "Computing normals (please wait)": "计算法线（请稍候）",
    "[qRansacSD] Apple-shaped torus are not handled by CLOUDVIEWER !": "[qRansacSD] CLOUDVIEWER不处理苹果形圆环！",
    "[qRansacSD] Input cloud has been automtically hidden!": "[qRansacSD] 输入点云已自动隐藏！",
    
    # qSRA
    "An error occurred while computing radial distances!": "计算径向距离时发生错误！",
    "You can compute the radial distances with the '%1' method": "您可以使用'%1'方法计算径向距离",
    
    # scalarFieldFromColorDlg
    "Composite = (R+G+B)/3": "复合 = (R+G+B)/3",
}

def translate_100_percent(input_file, output_file):
    """Final translation to achieve 100% coverage"""
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    translated_count = 0
    
    for context in root.findall('.//context'):
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
            
            if source_text in FINAL_42:
                translation.text = FINAL_42[source_text]
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                translated_count += 1
    
    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Final verification
    remaining = []
    total = 0
    translated_total = 0
    
    for message in root.findall('.//message'):
        total += 1
        translation = message.find('translation')
        if translation is not None:
            trans_type = translation.get('type', '')
            if trans_type == 'unfinished' or (not translation.text and trans_type != 'vanished'):
                source = message.find('source')
                if source is not None and source.text:
                    remaining.append(source.text[:100])
            else:
                translated_total += 1
    
    coverage = (translated_total / total * 100) if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"🎯 100%覆盖率冲刺结果")
    print(f"{'='*80}")
    print(f"总消息数：{total}")
    print(f"已翻译：{translated_total}")
    print(f"本次新增：{translated_count}")
    print(f"覆盖率：{coverage:.2f}%")
    print(f"未翻译：{len(remaining)}")
    print(f"{'='*80}\n")
    
    if len(remaining) == 0:
        print(f"🎉🎉🎉 100.00%覆盖率达成！")
        print(f"✓ 所有 {total} 条消息已全部翻译完成！")
        print(f"\n🏆 翻译工作圆满完成！")
    else:
        print(f"还有 {len(remaining)} 条未翻译")
        print(f"\n剩余内容：")
        for i, text in enumerate(remaining[:10], 1):
            print(f"{i}. {text}")
    
    return translated_count, len(remaining)

if __name__ == "__main__":
    input_file = '/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/ACloudViewer_zh.ts'
    output_file = input_file
    
    translated, remaining = translate_100_percent(input_file, output_file)
