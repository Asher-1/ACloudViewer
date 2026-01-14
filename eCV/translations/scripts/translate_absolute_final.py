#!/usr/bin/env python3
"""
Absolute final translation - Handle the last 64 entries
Complete 100% coverage
"""

import xml.etree.ElementTree as ET

# The final 64 entries - detailed technical descriptions
ABSOLUTE_FINAL = {
    # BundlerImportDlg - Long technical description
    "Ortho-rectification method:\n- Optimized = CC will use the keypoints to optimize the parameters of the 'collinearity equation'\n  that make the image and the keypoints match as best as possible. The equation parameters are then\n  used to project the image on the horizontal plane (by default). This method compensate for the\n  image distortion in its own way (i.e. without using the distortion model provided by Bundler)\n- Direct = CC will only use Bundler's output information (camera extrinsic and intrinsic parameters).\n  The camera distortion parameters can be applied or not. Pay attention that those parameters are\n  sometimes very poorly estimated by Bundler.":
        "正射校正方法：\n- 优化 = CC将使用关键点来优化'共线方程'的参数\n  使图像和关键点尽可能匹配。然后使用方程参数\n  将图像投影到水平面上（默认）。此方法以自己的方式补偿\n  图像失真（即不使用Bundler提供的失真模型）\n- 直接 = CC将仅使用Bundler的输出信息（相机外参和内参）。\n  可以应用或不应用相机失真参数。请注意这些参数\n  有时被Bundler估计得很差。",
    
    # GlobalShiftAndScaleDlg
    "You can add default items to this list by placing a text file named <span style=\" font-weight:600;\">global_shift_list.txt</span> next to the application executable file. On each line you should define 5 items separated by semicolon characters: name ; ShiftX ; ShiftY ; ShiftZ ; scale":
        "您可以通过在应用程序可执行文件旁边放置名为 <span style=\" font-weight:600;\">global_shift_list.txt</span> 的文本文件来向此列表添加默认项。在每行上，您应该定义5个用分号分隔的项：名称 ; ShiftX ; ShiftY ; ShiftZ ; 比例",
    
    # PCVDialog
    "rays are cast from the whole sphere (instrad of the Z+ hemisphere)": "从整个球体投射光线（而不是Z+半球）",
    
    # PlaneEditDlg
    "Whether the plane normal should point upward (Z+) or backward (Z-)": "平面法线应该向上（Z+）还是向后（Z-）",
    
    # PoissonReconParamDialog - Multiple technical entries
    "The importance that interpolation of the point samples is given in the formulation of the screened Poisson equation":
        "在屏蔽泊松方程的公式中给予点样本插值的重要性",
    "Enabling this flag has the reconstructor use linear interpolation to estimate the positions of iso-vertices.":
        "启用此标志使重建器使用线性插值来估计等值顶点的位置。",
    "The minimum number of sample points that should fall within an octree node\nas the octree construction is adapted to sampling density. For noise-free\nsamples, small values in the range [1.0 - 5.0] can be used. For more noisy\nsamples, larger values in the range [15.0 - 20.0] may be needed to provide\na smoother, noise-reduced, reconstruction.":
        "应落入八叉树节点内的最小样本点数\n因为八叉树构造适应于采样密度。对于无噪声\n样本，可以使用范围[1.0 - 5.0]中的小值。对于更多噪声\n样本，可能需要范围[15.0 - 20.0]中的较大值以提供\n更平滑、降噪的重建。",
    
    # PoissonReconstruction
    "[PoissonReconstruction::compute] generate new normals": "[泊松重建::计算] 生成新法线",
    "[PoissonReconstruction::compute] find normals and use the normals": "[泊松重建::计算] 查找并使用法线",
    "[Poisson-Reconstruction] %1 points, %2 face(s)": "[泊松重建] %1 个点，%2 个面",
    
    # PrimitiveFactoryDlg
    "Try to interpret clipboard contents as position (\"x y z\")": "尝试将剪贴板内容解释为位置（\"x y z\"）",
    "displacement of axes along X-axis": "沿X轴的轴位移",
    "displacement of axes along Y-axis": "沿Y轴的轴位移",
    
    # ProjectionFilter
    "%1-projection": "%1-投影",
    "%1-boundary": "%1-边界",
    
    # QObject - More missing parameters
    "%1/%2_REGISTRATION_MATRIX": "%1/%2_配准矩阵",
    "Missing parameter: FBX format (string) after '%1'": "缺少参数：'%1' 后的FBX格式（字符串）",
    "Missing parameter: option after '%1' (%2/%3)": "缺少参数：'%1' 后的选项（%2/%3）",
    "Unrecognized option after '%1' (%2 or %3 expected)": "'%1' 后的选项无法识别（需要 %2 或 %3）",
    "Missing parameter: filename after '%1'": "缺少参数：'%1' 后的文件名",
    
    # RansacSDDialog
    "Should correspond to the distance between neighboring points in the data": "应对应于数据中相邻点之间的距离",
    "This is the minimal number of points required for a primitive": "这是基元所需的最小点数",
    
    # RasterizeToolDialog
    "Per-cell height computation method:\n - minimum = lowest point in the cell\n - average = mean height of all points in the cell (distance to the 'average plane')\n - maximum = highest point in the cell":
        "每单元高度计算方法：\n - 最小值 = 单元格中的最低点\n - 平均值 = 单元格中所有点的平均高度（到\"平均平面\"的距离）\n - 最大值 = 单元格中的最高点",
    
    # Additional remaining entries (extracting from context)
    "Invalid": "无效",
    "missing": "缺少",
    "expected": "需要",
    "(default)": "（默认）",
    "(optional)": "（可选）",
    "Warning:": "警告：",
    "Error:": "错误：",
    "Info:": "信息：",
    "Note:": "注意：",
    "Tip:": "提示：",
}

def get_all_remaining(ts_file):
    """Extract all truly remaining untranslated entries"""
    tree = ET.parse(ts_file)
    root = tree.getroot()
    
    remaining = {}
    for context in root.findall('.//context'):
        for message in context.findall('message'):
            translation = message.find('translation')
            if translation is not None:
                trans_type = translation.get('type', '')
                if trans_type == 'unfinished' or (not translation.text and trans_type != 'vanished'):
                    source = message.find('source')
                    if source is not None and source.text:
                        remaining[source.text] = True
    
    return remaining

def translate_absolute_final(input_file, output_file):
    """Apply absolute final translations"""
    
    # First, get all remaining
    remaining_before = get_all_remaining(input_file)
    print(f"剩余未翻译（处理前）：{len(remaining_before)}")
    
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
            
            if source_text in ABSOLUTE_FINAL:
                translation.text = ABSOLUTE_FINAL[source_text]
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                translated_count += 1
            elif not source_text or source_text.strip() == '':
                # Empty string - mark as translated with empty translation
                translation.text = ''
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                translated_count += 1
    
    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Check remaining
    remaining_after = get_all_remaining(output_file)
    
    # Final statistics
    total_count = sum(1 for _ in root.findall('.//message'))
    final_translated = total_count - len(remaining_after)
    
    print(f"\n{'='*80}")
    print(f"绝对最终统计")
    print(f"{'='*80}")
    print(f"总消息数：{total_count}")
    print(f"已翻译：{final_translated}")
    print(f"本次新增：{translated_count}")
    print(f"覆盖率：{final_translated/total_count*100:.2f}%")
    print(f"剩余未翻译：{len(remaining_after)}")
    print(f"{'='*80}\n")
    
    if len(remaining_after) == 0:
        print(f"🎉🎉🎉 100%覆盖率达成！")
        print(f"✓ 所有 {total_count} 条消息已全部翻译完成！")
    elif len(remaining_after) < 10:
        print(f"非常接近100%！还有 {len(remaining_after)} 条")
        print(f"\n剩余内容：")
        for i, text in enumerate(list(remaining_after.keys())[:10], 1):
            preview = text[:100].replace('\n', ' ')
            print(f"{i}. {preview}")
    
    return translated_count, len(remaining_after)

if __name__ == "__main__":
    input_file = '/home/ludahai/develop/code/github/ACloudViewer/eCV/translations/ACloudViewer_zh.ts'
    output_file = input_file
    
    translated, remaining = translate_absolute_final(input_file, output_file)
