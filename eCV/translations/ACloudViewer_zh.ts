<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN">
<context>
    <name>AboutDialog</name>
    <message>
        <location filename="../ui_templates/aboutDlg.ui" line="17"/>
        <source>About CLOUDVIEWER </source>
        <translation>关于逸舟点云处理系统 </translation>
    </message>
    <message>
        <location filename="../ui_templates/aboutDlg.ui" line="47"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p align=&quot;center&quot;&gt;&lt;img src=&quot;:/Resources/images/icon/logo_128.png&quot;/&gt;&lt;br/&gt;&lt;span style=&quot; font-size:14pt; font-weight:600;&quot;&gt;ACloudViewer&lt;/span&gt;&lt;br/&gt;Version: %1&lt;/p&gt;&lt;p align=&quot;center&quot;&gt;&lt;a href=&quot;http://asher-1.github.io&quot;&gt;&lt;img src=&quot;:/Resources/images/donate.png&quot; width=&quot;200&quot;/&gt;&lt;/a&gt;&lt;/p&gt;&lt;p align=&quot;center&quot;&gt;&lt;a href=&quot;http://asher-1.github.io&quot;&gt;&lt;span style=&quot; text-decoration: underline; color:#0000ff;&quot;&gt;asher-1.github.io&lt;br/&gt;&lt;/span&gt;&lt;/a&gt;License: GNU GPL (General Public Licence)&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/aboutDlg.ui" line="87"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
</context>
<context>
    <name>AdjustZoomDialog</name>
    <message>
        <location filename="../ui_templates/adjustZoomDlg.ui" line="14"/>
        <source>Adjust zoom</source>
        <translation>调整缩放</translation>
    </message>
    <message>
        <location filename="../ui_templates/adjustZoomDlg.ui" line="20"/>
        <source>Window</source>
        <translation>窗口</translation>
    </message>
    <message>
        <location filename="../ui_templates/adjustZoomDlg.ui" line="49"/>
        <source>zoom</source>
        <translation>缩放比例</translation>
    </message>
    <message>
        <location filename="../ui_templates/adjustZoomDlg.ui" line="77"/>
        <source>pixel size</source>
        <translation>像素大小</translation>
    </message>
    <message>
        <location filename="../ui_templates/adjustZoomDlg.ui" line="86"/>
        <source> units</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/adjustZoomDlg.ui" line="115"/>
        <source> pixel(s)</source>
        <translation> 像素（s）</translation>
    </message>
</context>
<context>
    <name>AlignDialog</name>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="14"/>
        <source>Clouds alignment</source>
        <translation>点云配准</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="34"/>
        <source>Model and data</source>
        <translation>模型和数据</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="49"/>
        <source>Data:</source>
        <translation>数据:</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="56"/>
        <source>the data cloud is the entity to align with the model cloud: it will be displaced (green cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="77"/>
        <source>Model:</source>
        <translation>模型:</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="84"/>
        <source>the model cloud is the reference: it won&apos;t move (red cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="111"/>
        <location filename="../ui_templates/alignDlg.ui" line="114"/>
        <source>press once to exchange model and data clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="117"/>
        <source>swap</source>
        <translation>交换</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="144"/>
        <source>Alignment parameters</source>
        <translation>配准参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="152"/>
        <source>Number of trials:</source>
        <translation>试验数量:</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="159"/>
        <source>Number of 4 points bases tested to find the best rigid transform. Great values may lead to long computation time.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="182"/>
        <source>Overlap:</source>
        <translation>重叠:</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="189"/>
        <source>Rough estimation of the two clouds overlap rate (between 0 and 1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="215"/>
        <source>Delta:</source>
        <translation>增量:</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="222"/>
        <source>Estimation of the distance wished between the two clouds after registration.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="241"/>
        <location filename="../ui_templates/alignDlg.ui" line="244"/>
        <source>The computer will estimate the best delta parameter</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="247"/>
        <source>Estimate</source>
        <translation>估计</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="258"/>
        <source>For each attempt (see above parameter), candidate bases are found. If there are too much candidates, the program may take a long time to finish. Check this box to bound the number of candidates.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="261"/>
        <source>Limit max. number of candidates</source>
        <translation>限制最大候选量</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="268"/>
        <source>Maximal number of candidates allowed (check the left box to use this parameter)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="306"/>
        <source>Sampling</source>
        <translation>采样</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="314"/>
        <source>Method:</source>
        <translation>方法:</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="326"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="346"/>
        <location filename="../ui_templates/alignDlg.ui" line="466"/>
        <source>None</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="369"/>
        <location filename="../ui_templates/alignDlg.ui" line="489"/>
        <source>All</source>
        <translation>所有</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="381"/>
        <location filename="../ui_templates/alignDlg.ui" line="384"/>
        <source>Move to the left (none) to decrease the number of points  to keep in the data cloud.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="434"/>
        <location filename="../ui_templates/alignDlg.ui" line="554"/>
        <source>remaining points</source>
        <translation>剩余点</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="446"/>
        <source>Model</source>
        <translation>模型</translation>
    </message>
    <message>
        <location filename="../ui_templates/alignDlg.ui" line="501"/>
        <location filename="../ui_templates/alignDlg.ui" line="504"/>
        <source>Move to the left (none) to decrease the number of points  to keep in the model cloud.</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>AngleWidgetConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="14"/>
        <source>Form</source>
        <translation>形式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="22"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="35"/>
        <source>Point1</source>
        <translation>Point1</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="58"/>
        <source>Point2</source>
        <translation>卷帘窗</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="84"/>
        <source>2D Widget</source>
        <translation>2D Widgets</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="89"/>
        <source>3D Widget</source>
        <translation>3D Widget</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="100"/>
        <source>Type</source>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetconfig.ui" line="107"/>
        <source>Angle</source>
        <translation>角</translation>
    </message>
</context>
<context>
    <name>AngleWidgetWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/anglewidgetwindow.cpp" line="38"/>
        <source>Angle Widget</source>
        <translation>角小部件</translation>
    </message>
</context>
<context>
    <name>AnimationDialog</name>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="14"/>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="51"/>
        <source>Animation</source>
        <translation>动画</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="26"/>
        <source>Animation steps</source>
        <translation>动画的步骤</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="39"/>
        <source>Loop</source>
        <translation>循环</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="60"/>
        <source>Total duration</source>
        <translation>总持续时间</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="67"/>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="119"/>
        <source> sec.</source>
        <translation> 秒.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="86"/>
        <source>Current step</source>
        <translation>当前步骤</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="95"/>
        <source>Index</source>
        <translation>指数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="109"/>
        <source>Duration</source>
        <translation>持续时间</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="116"/>
        <source>Speed modifier for the current step</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="161"/>
        <source>Video output</source>
        <translation>视频输出</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="188"/>
        <source>Frame rate</source>
        <translation>帧率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="195"/>
        <source>Number of frames per second</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="198"/>
        <source> fps</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="227"/>
        <source>Bitrate</source>
        <translation>比特率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="234"/>
        <source>Bitrate (in kbit/s)
The higher the better the quality (but the bigger the file)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="238"/>
        <source> kbps</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="273"/>
        <source>- Super resolution: render the frame at a higher resolution (2, 3 or 4 times larger)
and then shrink it back down to size (this makes some noisy pixels drop off and
reduce the flicker that is often present in animations). Super resolution is only
applied on the output video (= not visible in Preview mode)
- Zoom: render the frame and the animation at a higher resolution (2, 3 or 4 times
larger). You may have to increase the points size beforehand.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="285"/>
        <source>super resolution</source>
        <translation>超分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="290"/>
        <source>zoom</source>
        <translation>变焦</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="298"/>
        <source>See combo-box tooltip</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="329"/>
        <source>Output file</source>
        <translation>输出文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="350"/>
        <source>Preview the animation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="353"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="360"/>
        <source>Creates the animation file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="363"/>
        <source>Render</source>
        <translation>渲染</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="370"/>
        <source>Export frames as individual images</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="373"/>
        <source>Export frames</source>
        <translation>导出帧</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/ui/animationDlg.ui" line="382"/>
        <source>Start preview from selected step</source>
        <translation>从选择位置开始预览</translation>
    </message>
</context>
<context>
    <name>ApplyTransformationDialog</name>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="14"/>
        <source>Apply transformation</source>
        <translation>应用转换</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="24"/>
        <source>Matrix 4x4</source>
        <translation>矩阵 4x4</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="32"/>
        <source>Enter 4x4 matrix values:</source>
        <translation>输入 4x4 矩阵值s:</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="52"/>
        <source>Load matrix from ascii file</source>
        <translatorcomment>从ascii文件读取矩阵</translatorcomment>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="55"/>
        <source>ASCII file</source>
        <translation>美国信息交换标准代码文件</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="62"/>
        <source>Paste clipboard contents</source>
        <translation>粘贴剪贴板内容</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="65"/>
        <source>clipboard</source>
        <translation>剪贴板</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="72"/>
        <source>Inits the matrix from dip/dip direction values
--&gt; assuming an initial position of (0,0)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="76"/>
        <source>dip / dip direction</source>
        <translation>倾向/倾斜 方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="83"/>
        <source>help</source>
        <translation>帮助</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="147"/>
        <source>Matrix should be of the form:
R11   R12   R13   Tx
R21   R22   R23   Ty
R31   R32   R33   Tz
0       0        0       1

Where R is a standard 3x3 rotation matrix and T is a translation vector.

Let P be a 3D point, the transformed point P&apos; will be such that: P&apos; = R.P + T.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="176"/>
        <source>Axis, Angle</source>
        <translation>轴、角</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="182"/>
        <source>Rotation axis</source>
        <translation>旋转轴</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="248"/>
        <source>Rotation angle (degrees)</source>
        <translation>旋转角（°）</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="263"/>
        <source> deg.</source>
        <translation> 角度.</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="295"/>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="447"/>
        <source>Translation</source>
        <translation>平移</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="375"/>
        <source>Euler angles</source>
        <translation>欧拉角</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="381"/>
        <source>Angles</source>
        <translation>角度</translation>
    </message>
    <message>
        <location filename="../ui_templates/applyTransformationDlg.ui" line="530"/>
        <source>Apply inverse transformation</source>
        <translation>应用反向变换</translation>
    </message>
</context>
<context>
    <name>AsciiOpenDialog</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="20"/>
        <source>Open Ascii File</source>
        <translation>打开文本文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="28"/>
        <source>Filename:</source>
        <translation>文件名:</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="50"/>
        <source>Here are the first lines of this file. Choose an attribute for each column (one cloud at a time):</source>
        <translation>以下是所选文件的前几行内容，请为每列选择对应属性（一次只能读取一个文件）：</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="60"/>
        <source>Header:</source>
        <translation>标题:</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="122"/>
        <source>Separator</source>
        <translation>分隔符</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="148"/>
        <source>(ASCII code:%i)</source>
        <translation>(ASCII代码:%i)</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="157"/>
        <source>ESP</source>
        <translation>空格符</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="164"/>
        <source>TAB</source>
        <translation>制表符</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="198"/>
        <source>Show labels in 2D (not recommended over 50).
Otherwise labels are shown in 3D.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="202"/>
        <source>Show labels in 2D</source>
        <translation>2D显示标签</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="215"/>
        <source>Skip lines</source>
        <translation>忽略行数</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="229"/>
        <source>+ comment/header lines skipped: 0</source>
        <translation>注释/首行忽略： 0</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="255"/>
        <source>extract scalar field names from first line</source>
        <translation>从首行提取标量字段名称</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="292"/>
        <source>Max number of points per cloud</source>
        <translation>单点云文件最大点数量</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="299"/>
        <source> Million</source>
        <translation> 百万</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="331"/>
        <source>Apply</source>
        <translation>应用</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="338"/>
        <source>Apply all</source>
        <translation>应用所有</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openAsciiFileDlg.ui" line="345"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>AsciiSaveDialog</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="14"/>
        <source>Save ASCII file</source>
        <translation>保存文本文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="22"/>
        <source>coordinates precision</source>
        <translation>坐标精度</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="39"/>
        <source>scalar precision</source>
        <translation>标量精度</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="56"/>
        <source>separator</source>
        <translation>分隔符</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="64"/>
        <source>space</source>
        <translation>空格符</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="69"/>
        <source>semicolon</source>
        <translation>;</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="74"/>
        <source>comma</source>
        <translation>,</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="79"/>
        <source>tabulation</source>
        <translation>制表符</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="87"/>
        <source>order</source>
        <translation>顺序</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="95"/>
        <source>[ASC] point, color, SF(s), normal</source>
        <translation>[ASC] 点、颜色、标量字段（s）、法线</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="100"/>
        <source>[PTS] point, SF(s), color, normal</source>
        <translation>[ASC] 点、标量字段（s）、颜色、法线</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="110"/>
        <source>Header</source>
        <translation>文件头</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="116"/>
        <source>columns title</source>
        <translation>列标题</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="123"/>
        <source>number of points (separate line)</source>
        <translation>点数量（分割线）</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="133"/>
        <source>Save RGB color components as floats values between 0 and 1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveAsciiFileDlg.ui" line="136"/>
        <source>Save colors as float values (0-1)</source>
        <translation>保存浮点型颜色值（0-1）</translation>
    </message>
</context>
<context>
    <name>AskThreeDoubleValuesDialog</name>
    <message>
        <location filename="../ui_templates/askThreeDoubleValuesDlg.ui" line="20"/>
        <source>Set Three Values</source>
        <translation>设置三个值</translation>
    </message>
    <message>
        <location filename="../ui_templates/askThreeDoubleValuesDlg.ui" line="28"/>
        <source>Value 1</source>
        <translation>值1</translation>
    </message>
    <message>
        <location filename="../ui_templates/askThreeDoubleValuesDlg.ui" line="54"/>
        <source>Value 2</source>
        <translation>值2</translation>
    </message>
    <message>
        <location filename="../ui_templates/askThreeDoubleValuesDlg.ui" line="80"/>
        <source>Value 3</source>
        <translation>值3</translation>
    </message>
</context>
<context>
    <name>AskTwoDoubleValuesDialog</name>
    <message>
        <location filename="../ui_templates/askTwoDoubleValuesDlg.ui" line="12"/>
        <source>Set Two Values</source>
        <translation>设置两个值</translation>
    </message>
    <message>
        <location filename="../ui_templates/askTwoDoubleValuesDlg.ui" line="20"/>
        <source>Value 1</source>
        <translation>值1</translation>
    </message>
    <message>
        <location filename="../ui_templates/askTwoDoubleValuesDlg.ui" line="46"/>
        <source>Value 2</source>
        <translation>值2</translation>
    </message>
</context>
<context>
    <name>BasePclModule</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="174"/>
        <source>Operation in progress</source>
        <translation>正在计算中</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="251"/>
        <source>No entity selected in tree.</source>
        <translation>在资源树中没有选中实体。</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="255"/>
        <source>Too many entities selected.</source>
        <translation>太多实体被选中。</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="259"/>
        <source>Wrong type of entity selected</source>
        <translation>所选实体类型错误</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="264"/>
        <source>Dialog was not correctly filled in</source>
        <translation>对话框参数值没有正确设置</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="269"/>
        <source>Errors while computing</source>
        <translation>计算出错</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="272"/>
        <source>Thread already in use!</source>
        <translation>线程正在被使用！</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/BasePclModule.cpp" line="277"/>
        <source>Undefined error in %1 module</source>
        <translation>未定义错误在 %1 模块</translation>
    </message>
</context>
<context>
    <name>BaseWidgetWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/basewidgetwindow.ui" line="14"/>
        <source>Form</source>
        <translation>形式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/basewidgetwindow.ui" line="30"/>
        <source>Config</source>
        <translation>配置</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/basewidgetwindow.ui" line="40"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
</context>
<context>
    <name>BoundingBoxEditorDialog</name>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="14"/>
        <source>Bounding Box Editor</source>
        <translation>边界框编辑器</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="32"/>
        <source>Min corner</source>
        <translation>最小边角</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="37"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="42"/>
        <source>Max corner</source>
        <translation>最大边角</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="57"/>
        <source>Width</source>
        <translation>宽度</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="211"/>
        <source>Warning, this box doesn&apos;t include the cloud bounding-box!</source>
        <translation>警告， 该箱体不包括点云外包围框！</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="227"/>
        <source>Orientation</source>
        <translation>方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="488"/>
        <source>automatically compute Z if checked</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="504"/>
        <source>automatically compute Y if checked</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="517"/>
        <source>automatically compute X if checked</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="548"/>
        <source>keep square</source>
        <translation>保持方形</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="568"/>
        <source>Default</source>
        <translation>默认</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="575"/>
        <source>Last</source>
        <translation>上一个</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="582"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../ui_templates/boundingBoxEditorDlg.ui" line="589"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>BundlerImportDlg</name>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="14"/>
        <source>Snavely&apos;s Bundler Import</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="26"/>
        <source>Information</source>
        <translation>信息</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="32"/>
        <source>File version:</source>
        <translation>文件版本:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="62"/>
        <source>keypoints:</source>
        <translation>要点:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="89"/>
        <source>Cameras:</source>
        <translation>相机:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="125"/>
        <source>Import images</source>
        <translation>导入图片</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="136"/>
        <source>Image list</source>
        <translation>图像列表</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="143"/>
        <source>List of the images corresponding to each camera</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="153"/>
        <source>Browse</source>
        <translation>浏览</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="180"/>
        <source>Image scale factor</source>
        <translation>图像尺度因子</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="187"/>
        <source>Image scale factor (relatively to the keypoints). Useful if you want to use images bigger than the ones you used to generate the Bundler .out file and the keypoints.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="190"/>
        <source>Image scale factor relatively to keypoints</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="205"/>
        <source>Orthorectification</source>
        <translation>Orthorectification</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="213"/>
        <source>To orthorectify images (as new images saved next to the original ones)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="216"/>
        <source>generate 2D orthophotos</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="226"/>
        <source>Ortho-rectification method:
- Optimized = CC will use the keypoints to optimize the parameters of the &apos;collinearity equation&apos;
  that make the image and the keypoints match as best as possible. The equation parameters are then
  used to project the image on the horizontal plane (by default). This method compensate for the
  image distortion in its own way (i.e. without using the distortion model provided by Bundler)
- Direct = CC will only use Bundler&apos;s output information (camera extrinsic and intrinsic parameters).
  The camera distortion parameters can be applied or not. Pay attention that those parameters are
  sometimes very poorly estimated by Bundler.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="237"/>
        <source>Optimized</source>
        <translation>优化</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="242"/>
        <source>Direct with undistortion</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="247"/>
        <source>Direct</source>
        <translation>直接</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="257"/>
        <source>To generate orthorectified versions of the images as clouds (warning: result mught be huge!).
Warning: the &apos;Optimized&apos; method is used by default.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="261"/>
        <source>generate 2D &quot;orthoclouds&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="270"/>
        <source>Vertical dimension:</source>
        <translation>竖直维度:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="277"/>
        <source>Keypoints vertical axis is X (1,0,0)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="287"/>
        <source>Keypoints vertical axis is Y (0,1,0)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="297"/>
        <source>Keypoints vertical axis is Z (0,0,1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="310"/>
        <source>Input a 4x4 transformation matrix that transforms the keypoint vertical axis into (0,0,1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="313"/>
        <source>Custom</source>
        <translation>自定义</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="322"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;1 0 0 0&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;0 1 0 0&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;0 0 1 0&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;0 0 0 1&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="340"/>
        <source>To generate a 3D model (mesh) colored with the input images.
By default the keypoints are meshed, and points are sampled on this first mesh.
The sampled points are then colored with the images and a final mesh is built on top of those points.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="345"/>
        <source>Colored model generation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="398"/>
        <source>vertices: </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="411"/>
        <source>Approximate number of vertices for the final mesh</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="447"/>
        <source>To use a cloud (or mesh) instead of the keypoints as base for the model generation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="450"/>
        <source>Use alternative keypoints</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="479"/>
        <source>...</source>
        <translation>...</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="496"/>
        <source>To keep images and their corresponding sensors in memory (i.e. as entities in the DB tree)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="499"/>
        <source>keep images (and sensors) loaded</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="522"/>
        <source>To undistort loaded images</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="525"/>
        <source>undistort images</source>
        <translation>andistort images</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/ui/openBundlerFileDlg.ui" line="550"/>
        <source>Import keypoints</source>
        <translation>进口要点</translation>
    </message>
</context>
<context>
    <name>CSFDialog</name>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="14"/>
        <source>Cloth Simulation Filter</source>
        <translation>布模拟滤波器</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="27"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;SimSun&apos;; font-size:9pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;img src=&quot;:/CC/plugin/qCSF/icon.png&quot; /&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt;&quot;&gt;	&lt;/span&gt;&lt;span style=&quot; font-size:11pt; font-weight:600;&quot;&gt;CSF Plugin Instruction&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;justify&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt; font-weight:600;&quot;&gt;Cloth Simulation Filter (CSF)&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt; is a tool to extract of ground points in discrete return LiDAR pointclouds. The detailed theory and algorithms could be found in the following paper:&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;justify&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;a name=&quot;OLE_LINK22&quot;&gt;&lt;/a&gt;&lt;span style=&quot; font-family:&apos;Arial,sans-serif&apos;; font-size:8pt; color:#000000;&quot;&gt;Z&lt;/span&gt;&lt;span style=&quot; font-family:&apos;Arial,sans-serif&apos;; font-size:8pt; color:#000000;&quot;&gt;hang W, Qi J, Wan P, Wang H, Xie D, Wang X, Yan G. An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation.&#xa0;&lt;/span&gt;&lt;span style=&quot; font-family:&apos;Arial,sans-serif&apos;; font-size:8pt; font-style:italic; color:#000000;&quot;&gt;Remote Sensing&lt;/span&gt;&lt;span style=&quot; font-family:&apos;Arial,sans-serif&apos;; font-size:8pt; color:#000000;&quot;&gt;. 2016; 8(6):501.&lt;/span&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt; &lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;justify&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; color:#000000; background-color:#cce8cf;&quot;&gt;And please cite the paper, If you use Cloth Simulation Filter (CSF) in your work.&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;justify&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; color:#000000; background-color:#cce8cf;&quot;&gt;You can download the paper from &lt;/span&gt;&lt;a href=&quot;https://www.researchgate.net/profile/Wuming_Zhang2)&quot;&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; text-decoration: underline; color:#0000ff; background-color:#cce8cf;&quot;&gt;https://www.researchgate.net/profile/Wuming_Zhang2 .&lt;/span&gt;&lt;/a&gt;&lt;/p&gt;
&lt;p align=&quot;justify&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; color:#000000; background-color:#cce8cf;&quot;&gt;You can also visit the homepage : &lt;/span&gt;&lt;a href=&quot;http://ramm.bnu.edu.cn/researchers/wumingzhang/english/default_contributions.htm&quot;&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; text-decoration: underline; color:#0000ff; background-color:#cce8cf;&quot;&gt;http://ramm.bnu.edu.cn/researchers/wumingzhang/english/default_contributions.htm&lt;/span&gt;&lt;/a&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; color:#000000; background-color:#cce8cf;&quot;&gt; for more information.&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;justify&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; color:#000000; background-color:#cce8cf;&quot;&gt;A mex version for programming in Matlab is at File Exchange of Mathworks website :  &lt;/span&gt;&lt;a href=&quot;http://www.mathworks.com/matlabcentral/fileexchange/58139-csf--ground-filtering-of-point-cloud-based-on-cloth-simulation&quot;&gt;&lt;span style=&quot; text-decoration: underline; color:#0000ff;&quot;&gt;http://www.mathworks.com/matlabcentral/fileexchange/58139-csf--ground-filtering-of-point-cloud-based-on-cloth-simulation&lt;/span&gt;&lt;/a&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;Microsoft YaHei UI,Tahoma&apos;; font-size:8pt; color:#000000; background-color:#cce8cf;&quot;&gt; Copyright &lt;/span&gt;&lt;span style=&quot; font-family:&apos;Arial,Helvetica,sans-serif&apos;; font-size:8.25pt; color:#333333; background-color:#e5eaee;&quot;&gt;©&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; background-color:#cce8cf;&quot;&gt;RAMM laboratory, School of Geography, Beijing Normal University&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; background-color:#cce8cf;&quot;&gt;(&lt;/span&gt;&lt;a href=&quot;http://ramm.bnu.edu.cn/&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; text-decoration: underline; color:#0000ff;&quot;&gt;http://ramm.bnu.edu.cn/&lt;/span&gt;&lt;/a&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; background-color:#cce8cf;&quot;&gt;)&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt;&quot;&gt;Wuming Zhang; Jianbo Qi; Peng Wan; Hongtao Wang&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt;&quot;&gt;contact us: &lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; background-color:#cce8cf;&quot;&gt;2009zwm@gmail.com; wpqjbzwm@126.com&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="62"/>
        <source>General parameter setting</source>
        <translation>通用参数设置</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="68"/>
        <source>Scenes</source>
        <translation>场景</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="93"/>
        <source>Steep slope</source>
        <translation>陡峭斜坡</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="119"/>
        <source>Relief</source>
        <translation>缓坡</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="145"/>
        <source>Flat</source>
        <translation>平坡</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="176"/>
        <source> Slope processing</source>
        <translation> 坡度处理</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="212"/>
        <source>Advanced parameter setting</source>
        <translation>高级参数设置</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="220"/>
        <source>Cloth resolution </source>
        <translation>布模型分辨率 </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="268"/>
        <source>Max iterations </source>
        <translation>最大迭代次数 </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="304"/>
        <source>Classification threshold</source>
        <translation>分类阈值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="343"/>
        <source>Exports the cloth in its final state as a mesh
(WARNING: ONLY FOR DEBUG PURPOSE - THIS IS NOT A DTM)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="347"/>
        <source>Export cloth mesh</source>
        <translation>导出布模型网格</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/ui/CSFDlg.ui" line="359"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;SimSun&apos;; font-size:9pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-weight:600;&quot;&gt;Advanced Parameter Instruction&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt; font-weight:600;&quot;&gt;1.&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt; Cloth resolution refers to the grid size (the unit is same as the unit of pointclouds) of cloth which is used to cover the terrain. The bigger cloth resolution you have set, the coarser DTM  you will get.&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt; font-weight:600;&quot;&gt;2.&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt; Max iterations refers to the maximum iteration times of terrain simulation. 500 is enough for most of scenes.&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt; font-weight:600;&quot;&gt;3.&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt; Classification threshold refers to a threshold (the unit is same as the unit of pointclouds) to classify the pointclouds into ground and non-ground parts based on the distances between points and the simulated terrain. 0.5 is adapted to most of scenes.&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>CSVMatrixOpenDlg</name>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="14"/>
        <source>Open CSV Matrix</source>
        <translation>开源CSV矩阵</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="20"/>
        <source>Grid</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="29"/>
        <source>Separator</source>
        <translation>分隔符</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="72"/>
        <source>X spacing</source>
        <translation>X间距</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="92"/>
        <source>Y spacing</source>
        <translation>Y 间距</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="115"/>
        <source>Invert row order</source>
        <translation>反转行序</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="122"/>
        <source>Load as mesh</source>
        <translation>加载为网格模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSVMatrixIO/ui/openCSVMatrixDlg.ui" line="131"/>
        <source>Use texture file</source>
        <translation>使用纹理文件</translation>
    </message>
</context>
<context>
    <name>Canupo2DViewDialog</name>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="14"/>
        <source>CANUPO training (result)</source>
        <translation>CANUPO 训练（结果）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="24"/>
        <source>You can manually edit the boundary ( left click: select or add vertex / long press: move / right click: remove vertex)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="50"/>
        <source>Legend</source>
        <translation>图例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="78"/>
        <source>Cloud1 name</source>
        <translation>点云1 名称</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="107"/>
        <source>Cloud2 name</source>
        <translation>点云2 名称</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="117"/>
        <source>Scales</source>
        <translation>尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="126"/>
        <source>In order to get a faster classifier, you can decrease the number of scales used (keeping only the smallest)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="140"/>
        <source>reset boundary</source>
        <translation>重置边界</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="150"/>
        <source>statistics</source>
        <translation>统计数据</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="157"/>
        <source>points size</source>
        <translation>点大小</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="189"/>
        <source>Save</source>
        <translation>保存训练模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo2DViewDialog.ui" line="196"/>
        <source>Done</source>
        <translation>完成</translation>
    </message>
</context>
<context>
    <name>CanupoClassifDialog</name>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="14"/>
        <source>CANUPO Classification</source>
        <translation>CANUPO 分类</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="24"/>
        <source>Classifier(s)</source>
        <translation>分类器(s)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="30"/>
        <source>file</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="42"/>
        <source>...</source>
        <translation>...</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="51"/>
        <source>info</source>
        <translation>信息</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="58"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#ff0000;&quot;&gt;No classifier loaded!&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#ff0000;&quot;&gt;无分类器导入!&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="72"/>
        <source>Core points are points on which the computation is actually performed (result is then propagated to the neighboring points).</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="75"/>
        <source>Core points</source>
        <translation>核心点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="81"/>
        <source>Warning, might be quite long on more than 100 000 points...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="84"/>
        <source>use selected cloud</source>
        <translation>使用选定点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="94"/>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="107"/>
        <source>Alternative core points cloud</source>
        <translation>可选核心点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="97"/>
        <source>use other cloud</source>
        <translation>使用其他点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="114"/>
        <source>Subsampled version of the selected cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="117"/>
        <source>subsample cloud</source>
        <translation>下采样点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="127"/>
        <source>Min. distance between points</source>
        <translation>最小点间距</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="140"/>
        <source>MSC files are generated by the original CANUPO tool (by N. Brodu)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="143"/>
        <source>from MSC file</source>
        <translation>从MSC文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="184"/>
        <source>Advanced</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="193"/>
        <source>Use confidence threshold for classification</source>
        <translation>分类中使用置信度阈值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="207"/>
        <source>threshold</source>
        <translation>阈值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="214"/>
        <source>Points having a confidence under this threshold won&apos;t be classified (or a SF will be used)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="248"/>
        <source>Try to classify points with a low confidence based on the local SF values</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="251"/>
        <source>use active SF to locally refine the classification</source>
        <translation>使用当前标量字段局部优化分类</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="261"/>
        <source>For test purpose!</source>
        <translation>仅用于测试!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="264"/>
        <source>generate one SF per scale with &apos;x-y&apos;</source>
        <translation>每个尺度生成一个标量字段（包含x-y）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="271"/>
        <source>generate one SF per scale with roughness</source>
        <translation>每个尺度生成一个标量字段（粗略）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoClassifDialog.ui" line="299"/>
        <source>Max thread count</source>
        <translation>最大线程数</translation>
    </message>
</context>
<context>
    <name>CanupoTrainingDialog</name>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="14"/>
        <source>CANUPO Training</source>
        <translation>CANUPO 训练</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="24"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="30"/>
        <source>Role</source>
        <translation>角色</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="37"/>
        <source>Cloud</source>
        <translation>点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="44"/>
        <source>Class label</source>
        <translation>类标签</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="51"/>
        <source>class #1</source>
        <translation>类型 #1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="65"/>
        <source>class #2</source>
        <translation>类型# 2</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="88"/>
        <source>Points belonging to class #1 </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="101"/>
        <source>Points belonging to class #2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="111"/>
        <source>Scales</source>
        <translation>尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="117"/>
        <source>ramp</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="142"/>
        <source>Mininum scale</source>
        <translation>最小比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="145"/>
        <source>Min = </source>
        <translation>最小值 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="167"/>
        <source>Step</source>
        <translation>步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="170"/>
        <source>Step = </source>
        <translation>步长 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="192"/>
        <source>Max scale</source>
        <translation>最大比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="195"/>
        <source>Max = </source>
        <translation>最大值 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="217"/>
        <source>Inp</source>
        <translation>列表</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="220"/>
        <source>list</source>
        <translation>列表</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="230"/>
        <source>Input scales as a list of values (separated by a space character)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="240"/>
        <source>Advanced</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="246"/>
        <source>Classification parameter</source>
        <translation>分类参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="254"/>
        <source>Dimensionality</source>
        <translation>维数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="262"/>
        <source>Max core points</source>
        <translation>最大核心点数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="269"/>
        <source>Maximum number of core points computed on each class</source>
        <translation>每类最大计算核心点数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="288"/>
        <source>Check this to add more points to the 2D classifier behavior representation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="291"/>
        <source>Show classifier behavior on </source>
        <translation>展示分类结果在 </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="307"/>
        <source>Additional points that will be added to the 2D classifier behavior representation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="314"/>
        <source>If checked the original cloud will be used for descriptors computation (i.e. class clouds will be considered as core points of this cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="317"/>
        <source>Use original cloud for descriptors</source>
        <translation>使用原始点云描述</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="333"/>
        <source>If set this cloud will be used for descriptors computation (i.e. class clouds will be considered as core points of this cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoTrainingDialog.ui" line="384"/>
        <source>Max thread count</source>
        <translation>最大线程数</translation>
    </message>
</context>
<context>
    <name>CellsFusionDlg</name>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="14"/>
        <source>Cell Fusion Parameters</source>
        <translation>单元融合参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="20"/>
        <source>Fusion algorithm</source>
        <translation>融合算法</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="30"/>
        <source>Kd-tree</source>
        <translation>Kd树</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="35"/>
        <source>Fast Marching</source>
        <translation>快速行进</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="46"/>
        <source>Kd-tree cells fusion parameters</source>
        <translation>KD树单元融合参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="55"/>
        <source>Max angle between cells (in degrees).
Kd-tree cells should be (roughly) planar.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="59"/>
        <source>Max angle</source>
        <translation>最大角</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="66"/>
        <source> deg.</source>
        <translation> 角度.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="82"/>
        <source>Max &apos;relative&apos; distance between cells (proportional to the cell size).
The bigger the farther the merged cells can be.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="86"/>
        <source>Max relative distance</source>
        <translation>最大相对距离</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="115"/>
        <source>FM cells fusion parameters</source>
        <translation>FM单元融合参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="123"/>
        <source>Octree level</source>
        <translation>八叉树层数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="130"/>
        <source>Octree Level (Fast Marching propagation process).</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="151"/>
        <source>use retro-projection error for propagation (slower)</source>
        <translation>前向传播使用逆投影误差（较慢）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="161"/>
        <source>Facets</source>
        <translation>面片</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="167"/>
        <source>Min points per facet</source>
        <translation>每个面片最少点数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="174"/>
        <source>Octree Level (for point cloud shape approx.)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="193"/>
        <source>Max edge length</source>
        <translation>最大边长度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="207"/>
        <source>Criterion for grouping several points in a single &apos;facet&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="214"/>
        <source>Max RMS</source>
        <translation>最大均方根</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="219"/>
        <source>Max distance @ 68%</source>
        <translation>最大距离 @ 68%</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="224"/>
        <source>Max distance @ 95%</source>
        <translation>最大距离 @ 95%</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="229"/>
        <source>Max distance @ 99%</source>
        <translation>最大距离 @ 99%</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="234"/>
        <source>Max distance</source>
        <translation>最大距离</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/cellsFusionDlg.ui" line="267"/>
        <source>Warning: cloud has no normals!
Output facet normals may be randomly oriented
(e.g. colors and classification may be jeopardized)</source>
        <translation>警告：点云如果没有法线！
输出面片法线可能随机指定方向
（例如：颜色和分类可能会受影响）</translation>
    </message>
</context>
<context>
    <name>ClassificationParamsDlg</name>
    <message>
        <location filename="../../plugins/core/qFacets/ui/classificationParamsDlg.ui" line="14"/>
        <source>Classification</source>
        <translation>分类</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/classificationParamsDlg.ui" line="54"/>
        <source>angular step</source>
        <translation>步进角</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/classificationParamsDlg.ui" line="61"/>
        <source>max distance</source>
        <translation>最大距离</translation>
    </message>
</context>
<context>
    <name>ClassifyDisclaimerDialog</name>
    <message>
        <location filename="../../plugins/core/qCanupo/classifyDisclaimerDlg.ui" line="14"/>
        <source>qCANUPO (disclaimer)</source>
        <translation>qCANUPO (disclaimer)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/classifyDisclaimerDlg.ui" line="48"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-weight:600; color:#1f497d;&quot;&gt;Multi-scale dimensionality classification (CANUPO)&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-style:italic; color:#1f497d;&quot;&gt;Brodu and Lague, 3D Terrestrial LiDAR data classification of complex natural scenes using a multi-scale dimensionality criterion, ISPRS j. of Photogram.&#xa0;Rem. Sens., 2012&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-style:italic; color:#1f497d; background-color:#ffffff;&quot;&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d; background-color:#ffffff;&quot;&gt;Funded by Université Européenne de Bretagne, Centre National de la Recherche Scientifique and EEC Marie-Curie actions&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d; background-color:#ffffff;&quot;&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d;&quot;&gt;Enjoy!&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>ClipConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/clipconfig.ui" line="20"/>
        <source>Form</source>
        <translation>形式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/clipconfig.ui" line="47"/>
        <source>Origin</source>
        <translation>原点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/clipconfig.ui" line="60"/>
        <source>Normal</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/clipconfig.ui" line="73"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/clipconfig.ui" line="87"/>
        <source>Open ...</source>
        <translation>打开……</translation>
    </message>
</context>
<context>
    <name>ClipWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/clipwindow.cpp" line="27"/>
        <source>Clip</source>
        <translation>剪辑</translation>
    </message>
</context>
<context>
    <name>ClippingBoxDlg</name>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="20"/>
        <source>Clipping Box</source>
        <translation>剪切盒</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="28"/>
        <source>Show/hide bounding box</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="48"/>
        <source>Show/hide interactors</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="84"/>
        <source>Restore the last clipping box used with this cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="101"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="104"/>
        <source>Reset</source>
        <translation>重置</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="121"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="124"/>
        <source>Close</source>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="143"/>
        <source>Box thickness</source>
        <translation>箱体厚度</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="215"/>
        <source>advanced</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="242"/>
        <source>Contour</source>
        <translation>轮廓</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="266"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="269"/>
        <source>Extracts the contour as a polyline (concave hull)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="286"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="289"/>
        <source>Removes last extracted contour</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="303"/>
        <source>Slices</source>
        <translation>片</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="327"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="330"/>
        <source>Export selection as a new cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="341"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="344"/>
        <source>Export multiple slices by repeating the process along one or several dimensions (+ contour extraction)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="413"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="416"/>
        <source>Set &apos;left&apos; view</source>
        <translation>设置“左”的观点</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="427"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="430"/>
        <source>Set &apos;right&apos; view</source>
        <translation>设置“正确”的观点</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="441"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="444"/>
        <source>Set &apos;front&apos; view</source>
        <translation>设置“前”的观点</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="455"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="458"/>
        <source>Set &apos;back&apos; view</source>
        <translation>设置“返回”的观点</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="469"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="472"/>
        <source>Set &apos;down&apos; view</source>
        <translation>设置“下降”的观点</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="483"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="486"/>
        <source>Set &apos;up&apos; view</source>
        <translation>设置”的观点</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="519"/>
        <source>Shift box</source>
        <translation>转变的箱体</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="562"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="584"/>
        <source>Shift box along its X dimension</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="606"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="628"/>
        <source>Shift box along its Y dimension</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="650"/>
        <location filename="../ui_templates/clippingBoxDlg.ui" line="672"/>
        <source>Shift box along its Z dimension</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>ClippingBoxRepeatDlg</name>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="14"/>
        <source>Repeat</source>
        <translation>重复</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="20"/>
        <source>The segmentation process will be repeated along the following dimensions (+/-)</source>
        <translation>分割进程将会沿着下列维度进行 （+/-）</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="23"/>
        <source>Repeat dimensions</source>
        <translation>重复维度</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="68"/>
        <source>Check that option if you wish to extract contour(s) form each slice</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="71"/>
        <source>Extract contour(s)</source>
        <translation>提取轮廓(s)</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="85"/>
        <source>Max edge length</source>
        <translation>最大边长度</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="92"/>
        <source>Max edge length (if 0, generates a unique and closed contour = convex hull)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="107"/>
        <source>Multi-pass process where longer edges may be temporarily created to obtain a better fit... or a worst one ;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="110"/>
        <source>multi-pass</source>
        <translation>多通道</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="117"/>
        <source>Before extracting the contour, points can be projected along the repeat dimension (if only one is defined) or on the best fit plane</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="120"/>
        <source>project slice(s) points on their best fit plane</source>
        <translation>在最佳拟合平面投影截面点</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="127"/>
        <source>split the generated contour(s) in smaller parts to avoid creating edges longer than the specified max edge length.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="130"/>
        <source>split contour(s) on longer edges</source>
        <translation>在较长边上分割轮廓（s）</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="137"/>
        <source>Display a dialog with step-by-step execution of the algorithm (debug mode - very slow)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="140"/>
        <source>visual debug mode</source>
        <translation>可视化调试模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="150"/>
        <source>Other options</source>
        <translation>其他选项</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="158"/>
        <source>Gap</source>
        <translation>间距</translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="165"/>
        <source>Gap between the slices</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="180"/>
        <source>If checked, a random color will be assigned to each slice (warning: will overwrite any existing color!)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/clippingBoxRepeatDlg.ui" line="183"/>
        <source>random colors per slice (will overwrite existing color!)</source>
        <translation>随机化颜色每个切片（此操作会覆盖已有颜色！）</translation>
    </message>
</context>
<context>
    <name>ColorGradientDialog</name>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="14"/>
        <source>Gradient color</source>
        <translation>渐变颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="20"/>
        <source>Color ramp</source>
        <translation>颜色渐变</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="26"/>
        <source>Default</source>
        <translation>默认</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="36"/>
        <source>Custom</source>
        <translation>自定义</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="61"/>
        <source>First color</source>
        <translation>第一颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="88"/>
        <source>Second color</source>
        <translation>第二颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="105"/>
        <source>Banding</source>
        <translation>条带型</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="136"/>
        <source>Period</source>
        <translation>周期</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorGradientDlg.ui" line="180"/>
        <source>direction</source>
        <translation>方向</translation>
    </message>
</context>
<context>
    <name>ColorLevelsDialog</name>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="14"/>
        <source>Change Color Levels</source>
        <translation>改变颜色的层级</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="35"/>
        <source>Channel(s)</source>
        <translation>通道 (s)</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="43"/>
        <source>RGB</source>
        <translation>RGB</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="48"/>
        <source>Red</source>
        <translation>红色</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="53"/>
        <source>Green</source>
        <translation>绿色</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="58"/>
        <source>Blue</source>
        <translation>蓝色</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="96"/>
        <source>Input levels</source>
        <translation>输入层</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorLevelsDlg.ui" line="172"/>
        <source>Output levels</source>
        <translation>输出层</translation>
    </message>
</context>
<context>
    <name>ColorScaleEditorDlg</name>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="14"/>
        <source>Color Scale Editor</source>
        <translation>颜色刻度编辑器</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="48"/>
        <source>Current</source>
        <translation>当前的</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="62"/>
        <source>Export color scale to a file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="73"/>
        <source>Import color scale from a file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="142"/>
        <source>Mode</source>
        <translation>模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="156"/>
        <source>relative</source>
        <translation>相对</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="161"/>
        <source>absolute</source>
        <translation>绝对</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="169"/>
        <source>Rename</source>
        <translation>重命名</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="176"/>
        <source>Save</source>
        <translation>保存</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="183"/>
        <source>Delete</source>
        <translation>删除</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="193"/>
        <source>Copy</source>
        <translation>复制</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="200"/>
        <source>New</source>
        <translation>新建</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="225"/>
        <source>(this ramp is locked - copy it before editing it)</source>
        <translation>（该色标已被锁 - 编辑前请复制一份）</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="263"/>
        <source>Selected slider</source>
        <translation>所选滑块</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="293"/>
        <source>Color</source>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="308"/>
        <source>Value</source>
        <translation>值</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="354"/>
        <source>Custom labels (one value per line)</source>
        <translation>自定义 标签（每行一个值）</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="366"/>
        <source>(auto)</source>
        <translation>（自动)</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="391"/>
        <source>Apply</source>
        <translation>应用</translation>
    </message>
    <message>
        <location filename="../ui_templates/colorScaleEditorDlg.ui" line="398"/>
        <source>Close</source>
        <translation>关闭</translation>
    </message>
</context>
<context>
    <name>ComparisonDialog</name>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="14"/>
        <source>Distance computation</source>
        <translation>距离计算</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="26"/>
        <source>Compared</source>
        <translation>比较目标</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="44"/>
        <source>Reference</source>
        <translation>参考目标</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="60"/>
        <source>Precise results</source>
        <translation>精确结果</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="70"/>
        <source>General parameters</source>
        <translation>通用参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="78"/>
        <source>Level of subdivision used for computing the distances</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="81"/>
        <source>Octree level</source>
        <translation>八叉树层数</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="91"/>
        <location filename="../ui_templates/comparisonDlg.ui" line="104"/>
        <source>Acceleration: distances above this limit won&apos;t be computed accurately</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="94"/>
        <source>max. distance</source>
        <translation>最大距离</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="120"/>
        <location filename="../ui_templates/comparisonDlg.ui" line="123"/>
        <source>compute signed distances (slower)</source>
        <translation>计算无符号距离（较慢）</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="126"/>
        <source>signed distances</source>
        <translation>无符号距离</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="136"/>
        <source>flip normals</source>
        <translation>翻转法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="145"/>
        <location filename="../ui_templates/comparisonDlg.ui" line="148"/>
        <source>Generate 3 supplementary scalar fields with distances along each dimension</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="151"/>
        <source>split X,Y and Z components</source>
        <translation>分离X、Y和Z组件</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="158"/>
        <source>Use the sensor associated to the reference cloud to ignore the points in the compared cloud
that could not have been seen (hidden/out of range/out of field of view).</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="162"/>
        <source>use reference sensor to filter hidden points</source>
        <translation>使用参考传感器滤波隐藏点</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="171"/>
        <source>multi-threaded</source>
        <translation>多线程</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="215"/>
        <source>max thread count</source>
        <translation>最大线程数</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="222"/>
        <source>Maximum number of threads/cores to be used
(CC or your computer might not respond for a while if you use all available cores)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="252"/>
        <source>Local modeling</source>
        <translation>局部建模</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="260"/>
        <source>Local model</source>
        <translation>局部模型</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="294"/>
        <source>Points (kNN)</source>
        <translation>点 (k最近邻)</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="318"/>
        <source>Radius (Sphere)</source>
        <translation>半径(球)</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="342"/>
        <source>faster but more ... approximate</source>
        <translation>快速但粗略</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="345"/>
        <source>use the same model for nearby points</source>
        <translation>紧邻点使用相同模型</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="369"/>
        <source>Approx. results</source>
        <translation>估计结果</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="405"/>
        <source>Warning: approximate results are only provided
to help you set the general parameters</source>
        <translation>警告：预估结果仅仅用于帮助您设置通用参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="479"/>
        <source>Compute</source>
        <translation>计算</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="509"/>
        <source>Ok</source>
        <translation>好吧</translation>
    </message>
    <message>
        <location filename="../ui_templates/comparisonDlg.ui" line="516"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>ComponentType</name>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="837"/>
        <source>Array</source>
        <translation>数组</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="840"/>
        <source>Name</source>
        <translation>的名字</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="840"/>
        <source>undefined</source>
        <translation>未定义的</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="843"/>
        <source>Elements</source>
        <translation>元素</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="846"/>
        <source>Capacity</source>
        <translation>能力</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="849"/>
        <source>Memory</source>
        <translation>内存</translation>
    </message>
</context>
<context>
    <name>ComputeOctreeDialog</name>
    <message>
        <location filename="../ui_templates/computeOctreeDlg.ui" line="14"/>
        <source>Compute Octree</source>
        <translation>计算八叉树</translation>
    </message>
    <message>
        <location filename="../ui_templates/computeOctreeDlg.ui" line="20"/>
        <source>Max subdivision level: ??</source>
        <translation>最大细分层数： ？？</translation>
    </message>
    <message>
        <location filename="../ui_templates/computeOctreeDlg.ui" line="27"/>
        <source>Octree box</source>
        <translation>八叉树单元</translation>
    </message>
    <message>
        <location filename="../ui_templates/computeOctreeDlg.ui" line="33"/>
        <source>Default</source>
        <translation>默认</translation>
    </message>
    <message>
        <location filename="../ui_templates/computeOctreeDlg.ui" line="43"/>
        <source>Cell size at max level</source>
        <translation>最大层单元尺寸</translation>
    </message>
    <message>
        <location filename="../ui_templates/computeOctreeDlg.ui" line="63"/>
        <source>Custom bounding box</source>
        <translation>自定义的边界框</translation>
    </message>
    <message>
        <location filename="../ui_templates/computeOctreeDlg.ui" line="73"/>
        <source>Edit</source>
        <translation>编辑</translation>
    </message>
</context>
<context>
    <name>ContourExtractorDlg</name>
    <message>
        <location filename="../ui_templates/contourExtractorDlg.ui" line="14"/>
        <source>Contour Extractor Visual Debug</source>
        <translation>轮廓抽取可视化调试</translation>
    </message>
    <message>
        <location filename="../ui_templates/contourExtractorDlg.ui" line="44"/>
        <source>Message</source>
        <translation>消息</translation>
    </message>
    <message>
        <location filename="../ui_templates/contourExtractorDlg.ui" line="57"/>
        <source>no message</source>
        <translation>没有消息</translation>
    </message>
    <message>
        <location filename="../ui_templates/contourExtractorDlg.ui" line="64"/>
        <source>auto</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../ui_templates/contourExtractorDlg.ui" line="71"/>
        <source>Next</source>
        <translation>下一个</translation>
    </message>
    <message>
        <location filename="../ui_templates/contourExtractorDlg.ui" line="78"/>
        <source>Skip</source>
        <translation>跳过</translation>
    </message>
</context>
<context>
    <name>ContourWidgetConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/contourwidgetconfig.ui" line="14"/>
        <source>Form</source>
        <translation>表格</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/contourwidgetconfig.ui" line="22"/>
        <source>Show Selected Nodes</source>
        <translation>显示选中节点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/contourwidgetconfig.ui" line="29"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
</context>
<context>
    <name>ContourWidgetWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/contourwidgetwindow.cpp" line="59"/>
        <source>Contour Widget</source>
        <translation>等高线小部件</translation>
    </message>
</context>
<context>
    <name>ContourWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.ui" line="14"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.cpp" line="20"/>
        <source>Contour</source>
        <translation>轮廓</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.ui" line="32"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.ui" line="40"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.cpp" line="96"/>
        <source>Import Data</source>
        <translation>导入数据</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.ui" line="57"/>
        <source>Random</source>
        <translation>随机</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.ui" line="64"/>
        <source>Rows</source>
        <translation>行</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/contourwindow.ui" line="84"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
</context>
<context>
    <name>ConvexConcaveHullDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/ConvexConcaveHullDlg.ui" line="20"/>
        <source>ConvexConcaveHull</source>
        <translation>凸包/凹包重建</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/ConvexConcaveHullDlg.ui" line="32"/>
        <source>Convex Concave Parameters</source>
        <translation>凸包/凹包参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/ConvexConcaveHullDlg.ui" line="76"/>
        <source>Alpha(0 ? Convex : Concave)</source>
        <translation>Alpha(0 ？ 凸包重建 ：凹包重建)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/ConvexConcaveHullDlg.ui" line="83"/>
        <source>dimension</source>
        <translation>维数</translation>
    </message>
</context>
<context>
    <name>ConvexConcaveHullReconstruction</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="42"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="43"/>
        <source>ConvexConcaveHull Reconstruction</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="44"/>
        <source>ConvexConcaveHull Reconstruction from clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="144"/>
        <source>[Concave-Reconstruction] %1 points, %2 face(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="145"/>
        <source>Concave Reconstruction</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="149"/>
        <source>[Convex-Reconstruction] %1 points, %2 face(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="150"/>
        <source>Convex Reconstruction</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="174"/>
        <source>Selected entity does not have any suitable scalar field or RGB.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="176"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/ConvexConcaveHullReconstruction.cpp" line="178"/>
        <source>Convex Concave Hull Reconstruction does not returned any point. Try relaxing your parameters</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>CorrespondenceMatching</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="41"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="42"/>
        <source>Correspondence Matching</source>
        <translation>对应点匹配</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="43"/>
        <source>Correspondence Matching from clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="120"/>
        <source>Invalid scale parameters!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="134"/>
        <source>At least one cloud (model #1 or #2) was not defined!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="230"/>
        <source>Scene total points: %1; Selected Keypoints: %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="239"/>
        <source>Model %1 total points: %2; Selected Keypoints: %3</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="287"/>
        <source>Correspondences found for model %1 : %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="350"/>
        <source>No instances found in Model %1 </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="354"/>
        <source>Model %1 instances found number : %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="373"/>
        <source>--- ICP Start ---------</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="381"/>
        <source>Model %1 Instances %2 Aligned!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="385"/>
        <source>Model %1 Instances %2 Not Aligned!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="390"/>
        <source>--- ICP End ---------</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="399"/>
        <source>--- Hypotheses Verification Start ---------</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="411"/>
        <source>Model %1 Instances %2 is GOOD!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="415"/>
        <source>Model %1 Instances %2 is bad and will be discarded!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="418"/>
        <source>--- Hypotheses Verification End ---------</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="520"/>
        <source>Cannot match anything by this model !</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="439"/>
        <source>correspondence-grouping-cluster(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="473"/>
        <source>%1-correspondence-%2-%3</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="522"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/CorrespondenceMatching.cpp" line="524"/>
        <source>Correspondence Matching does not returned any point. Try relaxing your parameters</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>CorrespondenceMatchingDialog</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="14"/>
        <source>Correspondence Matching</source>
        <translation>对应点匹配</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="24"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="30"/>
        <source>Role</source>
        <translation>角色</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="37"/>
        <source>Cloud</source>
        <translation>云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="44"/>
        <source>model #1</source>
        <translation>模型 #1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="57"/>
        <source>Points belonging to class #1 </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="64"/>
        <source>Model 1</source>
        <translation>模型1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="74"/>
        <source>model #2</source>
        <translation>模型 #2</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="90"/>
        <source>Points belonging to class #2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="97"/>
        <source>Model 2</source>
        <translation>模型2</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="104"/>
        <source>scene</source>
        <translation>场景模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="120"/>
        <source>Additional points that will be added to the 2D classifier behavior representation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="130"/>
        <source>Scales</source>
        <translation>尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="136"/>
        <source>ramp</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="161"/>
        <source>Mininum scale</source>
        <translation>最少购买比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="164"/>
        <source>Min = </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="186"/>
        <source>Step</source>
        <translation>一步</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="189"/>
        <source>Step = </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="211"/>
        <source>Max scale</source>
        <translation>最大比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="214"/>
        <source>Max = </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="236"/>
        <source>Inp</source>
        <translation>可使</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="239"/>
        <source>list</source>
        <translation>列表</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="249"/>
        <source>Input scales as a list of values (separated by a space character)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="259"/>
        <source>Advanced</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="304"/>
        <source>Normal K Search</source>
        <translation>法线K近邻搜索数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="311"/>
        <source>Scene Search Radius</source>
        <translation>场景搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="337"/>
        <source>Leaf Size = </source>
        <translation>叶子尺寸 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="359"/>
        <source>SHOT Descriptor Radius</source>
        <translation>SHOT描述子半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="369"/>
        <source>Voxel Grid</source>
        <translation>体素网格滤波</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="379"/>
        <source>Geometric Consistency Grouping</source>
        <translation>几何一致性分组</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="385"/>
        <source>apply GC</source>
        <translation>应用几何一致性算法</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="395"/>
        <source>Consensus Set Resolution</source>
        <translation>一致性分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="418"/>
        <source>Minimum Cluster Size</source>
        <translation>最小聚类大小</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="447"/>
        <source>Hough 3D Grouping</source>
        <translation>霍夫3D分组</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="456"/>
        <source>Hough Bin Size</source>
        <translation>霍夫划分尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="466"/>
        <source>Hough Threshold</source>
        <translation>霍夫阈值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="476"/>
        <source>LRF support radius</source>
        <translation>局部参考坐标系支撑面半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="540"/>
        <source>apply Hough</source>
        <translation>应用霍夫3D算法</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="550"/>
        <source>Model Search Radius</source>
        <translation>模型搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.ui" line="573"/>
        <source>Max thread count</source>
        <translation>最大线程数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.cpp" line="73"/>
        <source>You need at least 1 loaded clouds to perform matching</source>
        <translation>您至少需要一个点云模型执行匹配</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/CorrespondenceMatchingDialog.cpp" line="340"/>
        <source>unnamed</source>
        <translation>未命名</translation>
    </message>
</context>
<context>
    <name>CutConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="20"/>
        <source>Form</source>
        <translation>分割</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="28"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="320"/>
        <source>Sphere</source>
        <translation>球</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="36"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="43"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="95"/>
        <source>Cut Type</source>
        <translation>分割体类型</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="103"/>
        <source>Transparent</source>
        <translation>透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="108"/>
        <source>Points</source>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="113"/>
        <source>Opaque</source>
        <translation>不透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="118"/>
        <source>Wireframe</source>
        <translation>线框</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="126"/>
        <source>File:</source>
        <translation>文件:</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="133"/>
        <source>Display Effect</source>
        <translation>显示效果</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="140"/>
        <source>Open ...</source>
        <translation>打开……</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="153"/>
        <source>Data Table</source>
        <translation>数据表</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="161"/>
        <source>Rows</source>
        <translation>行</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="178"/>
        <source>Random</source>
        <translation>随机</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="197"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="315"/>
        <source>Plane</source>
        <translation>平面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="205"/>
        <source>Origin</source>
        <translation>原点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="242"/>
        <source>Normal</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="279"/>
        <source>Show Plane</source>
        <translation>显示平面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="289"/>
        <source>Show Contour Lines</source>
        <translation>显示轮廓线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="301"/>
        <source>Gradient</source>
        <translation>梯度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/cutconfig.ui" line="325"/>
        <source>Box</source>
        <translation>箱体</translation>
    </message>
</context>
<context>
    <name>DecimateConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="14"/>
        <source>Form</source>
        <translation>批量删除</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="20"/>
        <source>Decimate</source>
        <translation>批量滤除</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="28"/>
        <source>Feature angle</source>
        <translation>特征角度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="35"/>
        <source>Output Points Precision</source>
        <translation>输出点云数值精度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="42"/>
        <source>Target reduction</source>
        <translation>目标删减</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="49"/>
        <source>Split angle</source>
        <translation>分离角</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="56"/>
        <source>Inflection Point Ratio</source>
        <translation>拐点曲率</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="63"/>
        <source>Degress</source>
        <translation>角度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="70"/>
        <source>Preserve topology</source>
        <translation>保持拓扑结构</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="77"/>
        <source>Splitting</source>
        <translation>分离</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="101"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="108"/>
        <source>Open ...</source>
        <translation>打开……</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="115"/>
        <source>Presplit mesh</source>
        <translation>预分离 网格</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="139"/>
        <source>Accumulate error</source>
        <translation>累积误差</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimateconfig.ui" line="146"/>
        <source>Boundary vertex deletion</source>
        <translation>边界顶点删除</translation>
    </message>
</context>
<context>
    <name>DecimateWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/decimatewindow.cpp" line="19"/>
        <source>Decimate</source>
        <translation>批量滤除</translation>
    </message>
</context>
<context>
    <name>Dialog</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.ui" line="14"/>
        <source>QVTK Tutorial: Constructing surface from points</source>
        <translation>基于点云重建曲面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.ui" line="35"/>
        <source>Settings</source>
        <translation>设置</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.ui" line="46"/>
        <source>Random</source>
        <translation>随机</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.ui" line="53"/>
        <source>Load File</source>
        <translation>加载文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.ui" line="77"/>
        <source>Rows</source>
        <translation>行</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.ui" line="87"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.cpp" line="33"/>
        <source>Construct Surface</source>
        <translation>构建表面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/dialog.cpp" line="46"/>
        <source>Import Data</source>
        <translation>导入数据</translation>
    </message>
</context>
<context>
    <name>DipDirTransformationDialog</name>
    <message>
        <location filename="../ui_templates/dipDirTransformationDlg.ui" line="14"/>
        <source>Dip / dip dir. transformation</source>
        <translation>倾斜/倾斜方向 转换</translation>
    </message>
    <message>
        <location filename="../ui_templates/dipDirTransformationDlg.ui" line="22"/>
        <source>Dip</source>
        <translation>倾斜</translation>
    </message>
    <message>
        <location filename="../ui_templates/dipDirTransformationDlg.ui" line="29"/>
        <location filename="../ui_templates/dipDirTransformationDlg.ui" line="49"/>
        <source> deg.</source>
        <translation> 角度.</translation>
    </message>
    <message>
        <location filename="../ui_templates/dipDirTransformationDlg.ui" line="42"/>
        <source>Dip direction</source>
        <translation>倾斜方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/dipDirTransformationDlg.ui" line="64"/>
        <source>rotate about selection center</source>
        <translation>绕选择中心旋转</translation>
    </message>
</context>
<context>
    <name>DisclaimerDialog</name>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/disclaimerDlg.ui" line="14"/>
        <source>qM3C2 (disclaimer)</source>
        <translation>qM3C2(免责声明)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/disclaimerDlg.ui" line="48"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-weight:600; color:#1f497d;&quot;&gt;Point cloud comparison with M3C2&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-style:italic; color:#1f497d;&quot;&gt;Lague et al., Accurate 3D comparison of complex topography with terrestrial laser scanner, ISPRS j. of Photogram.&#xa0;Rem. Sens., 2013&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-style:italic; color:#1f497d; background-color:#ffffff;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2,serif&apos;; font-size:8pt; color:#aa007f; background-color:#ffffff;&quot;&gt;This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;arial,sans-serif&apos;; font-size:10pt; font-style:italic; color:#222222; background-color:#ffffff;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d; background-color:#ffffff;&quot;&gt;Funded by Université Européenne de Bretagne, Centre National de la Recherche Scientifique and EEC Marie-Curie actions&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/disclaimerDlg.ui" line="20"/>
        <source>qFacets (disclaimer)</source>
        <translation>qFacets(免责声明)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/disclaimerDlg.ui" line="58"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:10pt;&quot;&gt;This plugin development was funded by Thomas Dewez – BRGM&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published bythe Free Software Foundation; version 2 or later of the License.&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#0000ff;&quot;&gt;copyright BRGM&lt;/span&gt;&lt;span style=&quot; font-family:&apos;sans-serif&apos;; font-size:8pt; color:#0000ff; background-color:#ffffff;&quot;&gt;©&lt;/span&gt;&lt;span style=&quot; font-size:8pt; color:#0000ff;&quot;&gt; 2013&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;a href=&quot;http://www.brgm.eu/&quot;&gt;&lt;span style=&quot; font-size:8pt; text-decoration: underline; color:#0000ff;&quot;&gt;http://www.brgm.eu&lt;/span&gt;&lt;/a&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>DisplayOptionsDlg</name>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="20"/>
        <source>Display options</source>
        <translation>显示选项</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="49"/>
        <source>Colors and Materials</source>
        <translation>颜色和材质</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="79"/>
        <source>Light</source>
        <translation>光线</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="94"/>
        <source>Ambient</source>
        <translation>环境光</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="112"/>
        <source>Specular</source>
        <translation>镜面光</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="130"/>
        <source>Diffuse</source>
        <translation>漫反射光</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="139"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Whether to use a double-sided light model or not&lt;/p&gt;&lt;p&gt;(if disabled, triangles will appear black when looked from behind)&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="142"/>
        <source>double-sided</source>
        <translation>双面的</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="168"/>
        <source>Default Materials</source>
        <translation>默认材质</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="183"/>
        <source>Mesh Front</source>
        <translation>网格前景</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="201"/>
        <source>Mesh Back</source>
        <translation>网格后景</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="219"/>
        <source>Mesh specular</source>
        <translation>网格镜面</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="237"/>
        <source>Points</source>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="249"/>
        <source>Colors</source>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="267"/>
        <source>Background</source>
        <translation>背景色</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="288"/>
        <source>Bounding-box</source>
        <translation>外包围框</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="306"/>
        <source>Text</source>
        <translation>文本</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="349"/>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="352"/>
        <source>Gradient goes from &apos;background&apos; color to inverse of &apos;Points&apos; color</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="355"/>
        <source>display gradient background</source>
        <translation>显示渐变背景色</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="381"/>
        <source>Color scale (scalar field)</source>
        <translation>颜色标量（标量字段）</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="387"/>
        <source>Show histogram next to color ramp</source>
        <translation>把直方图显示在色标傍边</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="397"/>
        <source>Enable shader for faster display (ATI cards: use at your own risk ;)</source>
        <translation>允许使用shader（用于快速显示）</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="406"/>
        <source>Ramp width</source>
        <translation>色带宽度</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="413"/>
        <source> pixels</source>
        <translation> 像素</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="458"/>
        <source>Labels</source>
        <translation>标签</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="485"/>
        <source>Label opacity</source>
        <translation>标签不透明度</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="505"/>
        <source>Label font size</source>
        <translation>标签字体大小</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="525"/>
        <source>background color</source>
        <translation>背景颜色</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="542"/>
        <source>Labels marker size</source>
        <translation>标签标记大小</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="559"/>
        <source>marker color</source>
        <translation>标记颜色</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="593"/>
        <source>Other options</source>
        <translation>其他选项</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="601"/>
        <source>Default font size</source>
        <translation>默认字体大小</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="638"/>
        <source>Displayed numbers precision</source>
        <translation>数值显示精度</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="669"/>
        <source>Perspective zooming speed</source>
        <translation>透视缩放速度</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="676"/>
        <source>Zoom/walk speed in perspective mode (default = 1.0)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="712"/>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="715"/>
        <source>Automatically decimate big point clouds when moved</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="718"/>
        <source>Decimate clouds over</source>
        <translation>批量滤除点云在</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="728"/>
        <source>Minimum number of points to activate decimation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="753"/>
        <source>points when moved</source>
        <translation>点移动时</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="777"/>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="780"/>
        <source>Automatically decimate big meshes when moved</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="783"/>
        <source>Decimate meshes over</source>
        <translation>批量滤除网格在</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="793"/>
        <source>Minimum number of triangles to activate decimation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="821"/>
        <source>triangles when moved</source>
        <translation>三角面片时移动</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="843"/>
        <source>Draw rounded points (slower)</source>
        <translation>绘制光滑点云（较慢）</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="850"/>
        <source>Try to load clouds on GPU for faster display</source>
        <translation>尝试在GPU上加载点云（快速显示）</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="860"/>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="863"/>
        <source>A cross is displayed in the middle of the screen</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="866"/>
        <source>Show middle screen cross</source>
        <translation>显示中等大小的cross</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="876"/>
        <source>Automatically display normals at loading time (when available)</source>
        <translation>读取时自动显示法线（当法线可用时）</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="883"/>
        <source>Use native load / save dialogs</source>
        <translation>使用本地 读取/保存 对话框</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="895"/>
        <source>Auto-compute octree for picking</source>
        <translation>拾取时自动计算八叉树</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="902"/>
        <source>Octree computation can be long but the picking is then much faster</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="909"/>
        <source>Always</source>
        <translation>总是</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="914"/>
        <source>Ask each time</source>
        <translation>每次询问</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="919"/>
        <source>Never</source>
        <translation>从不</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="974"/>
        <source>Ok</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="981"/>
        <source>Apply</source>
        <translation>应用</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="988"/>
        <source>Reset</source>
        <translation>重置</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/displayOptionsDlg.ui" line="995"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>DistanceMapDialog</name>
    <message>
        <location filename="../ui_templates/distanceMapDlg.ui" line="14"/>
        <source>Distance Map</source>
        <translation>距离地图</translation>
    </message>
    <message>
        <location filename="../ui_templates/distanceMapDlg.ui" line="22"/>
        <source>Steps</source>
        <translation>步长</translation>
    </message>
    <message>
        <location filename="../ui_templates/distanceMapDlg.ui" line="29"/>
        <source>Map steps (in each direction).
The bigger the more accurate the map will be
(but the more points will be created)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/distanceMapDlg.ui" line="63"/>
        <source>Outer margin</source>
        <translation>外边界</translation>
    </message>
    <message>
        <location filename="../ui_templates/distanceMapDlg.ui" line="70"/>
        <source>Margin added around the cloud bounding-box</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/distanceMapDlg.ui" line="87"/>
        <source>reduce result to the specified range</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/distanceMapDlg.ui" line="90"/>
        <source>Range</source>
        <translation>范围</translation>
    </message>
</context>
<context>
    <name>DistanceMapGenerationDlg</name>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="14"/>
        <source>2D distance map (Surface of Revolution)</source>
        <translation>2D距离地图（旋转曲面）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="53"/>
        <source>Projection</source>
        <translation>投影</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="63"/>
        <source>Cylindrical</source>
        <translation>圆柱</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="68"/>
        <source>Conical (Lambert)</source>
        <translation>锥形(Lambert)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="91"/>
        <source>Spanning ratio</source>
        <translation>生成率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="136"/>
        <source>Map</source>
        <translation>地图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="148"/>
        <source>Resolution</source>
        <translation>分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="175"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="517"/>
        <source>angle (X)</source>
        <translation>角度(X)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="188"/>
        <source>Map angular step (horizontal)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="191"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="232"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="350"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="777"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="806"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="879"/>
        <source>step = </source>
        <translation>步长 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="216"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="530"/>
        <source>height (Y)</source>
        <translation>高度(Y)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="229"/>
        <source>Map height step (vertical)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="260"/>
        <source>size</source>
        <translation>尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="280"/>
        <source>Map angles unit</source>
        <translation>映射角度单位</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="287"/>
        <source>deg</source>
        <translation>角度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="292"/>
        <source>rad</source>
        <translation>弧度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="297"/>
        <source>grad</source>
        <translation>梯度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="317"/>
        <source>Map heights unit (for display only)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="320"/>
        <source>m.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="330"/>
        <source>Counterclockwise unrolling</source>
        <translation>逆时针展开</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="333"/>
        <source>CCW</source>
        <translation>公约</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="340"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="557"/>
        <source>latitude</source>
        <translation>纬度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="347"/>
        <source>Map latitude step</source>
        <translation>地图纬度步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="378"/>
        <source>Limits</source>
        <translation>限制</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="408"/>
        <source>Minimum map angle</source>
        <translation>最小地图角度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="433"/>
        <source>Minimum map height (relative to the generatrix origin)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="464"/>
        <source>Maximum map angle</source>
        <translation>最大地图角</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="489"/>
        <source>Maximum map height (relative to the generatrix origin)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="537"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="547"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="570"/>
        <source>Minimum map latitude (relative to the generatrix origin - always positive - in degrees)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="573"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="604"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="780"/>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="882"/>
        <source> grad</source>
        <translation> 梯度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="601"/>
        <source>Maximum map latitude (relative to the generatrix origin - always positive - in degrees)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="629"/>
        <source>Filling</source>
        <translation>填充</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="650"/>
        <source>strategy</source>
        <translation>策略</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="657"/>
        <source>What to do when multiple values fall in the same grid cell?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="664"/>
        <source>minimum value</source>
        <translation>最小值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="669"/>
        <source>average value</source>
        <translation>平均值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="674"/>
        <source>maximum value</source>
        <translation>最大值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="682"/>
        <source>empty cells</source>
        <translation>空单元</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="689"/>
        <source>What to do when a grid cell remains empty?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="693"/>
        <source>leave empty</source>
        <translation>置空</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="698"/>
        <source>fill with zero</source>
        <translation>填充零</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="703"/>
        <source>interpolate</source>
        <translation>插值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="734"/>
        <source>Display</source>
        <translation>显示</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="746"/>
        <source>Overlay grid</source>
        <translation>重叠网格</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="770"/>
        <source>Angle (X)</source>
        <translation>角度(X)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="799"/>
        <source>Height (Y)</source>
        <translation>高度(Y)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="809"/>
        <source> m.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="831"/>
        <source>Show X labels</source>
        <translation>显示X标签</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="843"/>
        <source>Grid color</source>
        <translation>网格颜色</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="862"/>
        <source>Show Y labels</source>
        <translation>显示Y标签</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="872"/>
        <source>Latitude</source>
        <translation>纬度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="904"/>
        <source>2D symbols</source>
        <translation>2D 符号</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="927"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Load a set of symbols / labels from a text file.&lt;br/&gt;On each line: &apos;Label X Y Z&apos; (expressed relatively to the profile origin)&lt;br/&gt;&lt;span style=&quot; font-weight:600;&quot;&gt;(warning: the height values - along the revolution axis - must be expressed relative to the profile origin)&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="930"/>
        <source>Load</source>
        <translation>加载</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="940"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="951"/>
        <source>Symbol size</source>
        <translation>符号尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="987"/>
        <source>Symbol color</source>
        <translation>符号颜色</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1006"/>
        <source>Color ramp</source>
        <translation>颜色渐变</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1023"/>
        <source>Steps</source>
        <translation>步骤</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1059"/>
        <source>Display color scale</source>
        <translation>显示颜色标尺</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1083"/>
        <source>Font size</source>
        <translation>字体大小</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1125"/>
        <source>Displayed numbers precision (digits)</source>
        <translation>数值显示精度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1128"/>
        <source>Precision</source>
        <translation>精度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1167"/>
        <source>Generatrix</source>
        <translation>母线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1173"/>
        <source>Axis</source>
        <translation>轴</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1180"/>
        <source>Generatrix direction (in the 3D world)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1205"/>
        <source>Origin (3D)</source>
        <translation>原点 (3D)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1260"/>
        <source>Base radius</source>
        <translation>基半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1267"/>
        <source>Mean radius (for map display, export as a cloud, etc. )</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1284"/>
        <source>Measures</source>
        <translation>测量</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1290"/>
        <source>Surface and volume (approximate)</source>
        <translation>曲面和体积（粗略）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1307"/>
        <source>Update</source>
        <translation>更新</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1321"/>
        <source>Export map</source>
        <translation>导出地图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1332"/>
        <source>ASCII grid</source>
        <translation>ASCII网格</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1339"/>
        <source>Image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1346"/>
        <source>DXF</source>
        <translation>DXF</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1370"/>
        <source>Cloud</source>
        <translation>点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationDlg.ui" line="1377"/>
        <source>Mesh</source>
        <translation>网格</translation>
    </message>
</context>
<context>
    <name>DistanceWidgetConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetconfig.ui" line="14"/>
        <source>Form</source>
        <translation>距离测量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetconfig.ui" line="25"/>
        <source>Point2</source>
        <translation>点2</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetconfig.ui" line="48"/>
        <source>Point1</source>
        <translation>点1</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetconfig.ui" line="67"/>
        <source>Type</source>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetconfig.ui" line="74"/>
        <source>Distance</source>
        <translation>距离</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetconfig.ui" line="82"/>
        <source>2D Widget</source>
        <translation>2D 窗口</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetconfig.ui" line="87"/>
        <source>3D Widget</source>
        <translation>3D 窗口</translation>
    </message>
</context>
<context>
    <name>DistanceWidgetWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/distancewidgetwindow.cpp" line="40"/>
        <source>Distance Widget</source>
        <translation>距离小部件</translation>
    </message>
</context>
<context>
    <name>DxfProfilesExportDlg</name>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="14"/>
        <source>Export profiles to DXF</source>
        <translation>导出轮廓为DXF</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="20"/>
        <source>Vertical profiles</source>
        <translation>垂直轮廓</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="31"/>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="126"/>
        <source>File name</source>
        <translation>文件名称</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="52"/>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="147"/>
        <source>Title</source>
        <translation>标题</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="59"/>
        <source>VERTICAL PROFILE: DEVIATIONS</source>
        <translation>垂直轮廓： 偏差</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="66"/>
        <source>(+ profile angle)</source>
        <translation>(+轮廓角)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="77"/>
        <source>Angular steps</source>
        <translation>角度步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="115"/>
        <source>Horizontal profiles</source>
        <translation>水平轮廓</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="154"/>
        <source>HORIZONTAL PROFILE: DEVIATIONS</source>
        <translation>水平轮廓：偏差</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="161"/>
        <source>(+ profile height)</source>
        <translation>(+轮廓高度)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="172"/>
        <source>Height steps</source>
        <translation>高度步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="210"/>
        <source>Deviation</source>
        <translation>偏差</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="218"/>
        <source>Values scaling (for labels only)</source>
        <translation>值缩放比例（仅标签）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="241"/>
        <source>Units</source>
        <translation>单位</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="278"/>
        <source>Precision</source>
        <translation>精度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="312"/>
        <source>Drawing magnification</source>
        <translation>绘制倍率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="350"/>
        <source>Legend</source>
        <translation>图例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="356"/>
        <source>Theoretical profile name</source>
        <translation>理论轮廓名称</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="363"/>
        <source>Theoretical</source>
        <translation>理论值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="370"/>
        <source>Measured profile(s) name</source>
        <translation>实测轮廓名称</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/dxfProfilesExportDlg.ui" line="377"/>
        <source>Real</source>
        <translation>真实值</translation>
    </message>
</context>
<context>
    <name>EuclideanClusterDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/EuclideanClusterDlg.ui" line="20"/>
        <source>EuclideanCluster Segmentation</source>
        <translation>欧式聚类分割</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/EuclideanClusterDlg.ui" line="30"/>
        <source>Euclidean Cluster Segmentation Parameters</source>
        <translation>欧式聚类分割参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/EuclideanClusterDlg.ui" line="45"/>
        <source>Voxel Grid [Leaf Size]</source>
        <translation>使用体素滤波 [叶子尺寸]</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/EuclideanClusterDlg.ui" line="116"/>
        <source>Maximum Cluster Size</source>
        <translation>最大簇尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/EuclideanClusterDlg.ui" line="123"/>
        <source>Minimum Cluster Size</source>
        <translation>最小簇尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/EuclideanClusterDlg.ui" line="130"/>
        <source>Cluster Tolerance</source>
        <translation>聚类容忍度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/EuclideanClusterDlg.ui" line="137"/>
        <source>Random Cluster Color</source>
        <translation>随机簇颜色</translation>
    </message>
</context>
<context>
    <name>EuclideanClusterSegmentation</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="42"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="43"/>
        <source>EuclideanCluster Segmentation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="44"/>
        <source>EuclideanCluster Segmentation from clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="153"/>
        <source>-Tolerance(%1)-ClusterSize(%2-%3)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="158"/>
        <source>[EuclideanClusterSegmentation] %1 cluster(s) where created from cloud &apos;%2&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="164"/>
        <source>Error(s) occurred during the generation of clusters! Result may be incomplete</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="194"/>
        <source>Selected entity does not have any suitable scalar field or RGB.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="196"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="198"/>
        <source>EuclideanCluster Segmentation could not get any cluster or the clusters are more than 300 for the given dataset. Try relaxing your parameters</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/EuclideanClusterSegmentation.cpp" line="201"/>
        <source>An error occurred during the generation of clusters!</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>ExportCoordToSFDlg</name>
    <message>
        <location filename="../ui_templates/exportCoordToSFDlg.ui" line="14"/>
        <source>Export coordinates to SF</source>
        <translation>导出坐标为标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/exportCoordToSFDlg.ui" line="29"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../ui_templates/exportCoordToSFDlg.ui" line="71"/>
        <source>Warning, already existing SF(s) with same name will be overwritten</source>
        <translation>警告， 已存在同名标量字段将会被覆盖</translation>
    </message>
</context>
<context>
    <name>ExtractSIFT</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="40"/>
        <source>Extract SIFT</source>
        <translation>提取筛选</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="41"/>
        <source>Extract SIFT Keypoints</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="42"/>
        <source>Extract SIFT keypoints for clouds with intensity/RGB or any scalar field</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="242"/>
        <source>SIFT Keypoints_%1_rgb_%2_%3_%4</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="244"/>
        <source>SIFT Keypoints_%1_%2_%3_%4_%5</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="272"/>
        <source>Selected entity does not have any suitable scalar field or RGB. Intensity scalar field or RGB are needed for computing SIFT</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="274"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ExtractSIFT.cpp" line="276"/>
        <source>SIFT keypoint extraction does not returned any point. Try relaxing your parameters</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>FacetsExportDlg</name>
    <message>
        <location filename="../../plugins/core/qFacets/ui/facetsExportDlg.ui" line="14"/>
        <source>Export facets</source>
        <translation>导出面片</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/facetsExportDlg.ui" line="20"/>
        <source>Destination</source>
        <translation>导出路径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/facetsExportDlg.ui" line="39"/>
        <source>Vertical orientation (only for polygons)</source>
        <translation>垂直方向（仅多边形）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/facetsExportDlg.ui" line="47"/>
        <source>Native</source>
        <translation>原始</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/facetsExportDlg.ui" line="54"/>
        <source>Mean normal</source>
        <translation>中点法向</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/facetsExportDlg.ui" line="64"/>
        <source>Custom</source>
        <translation>自定义</translation>
    </message>
</context>
<context>
    <name>FilterByValueDialog</name>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="14"/>
        <source>Filter by value</source>
        <translation>值滤波器</translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="22"/>
        <source>Range</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="29"/>
        <source>Min range value</source>
        <translation>最小值范围</translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="61"/>
        <source>Max range value</source>
        <translation>最大范围值</translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="81"/>
        <source>Exports the points falling inside the specified range.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="84"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="91"/>
        <source>Creates two clouds: one with the points falling inside the specified range,
the other with the points falling outside.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="95"/>
        <source>Split</source>
        <translation>分离</translation>
    </message>
    <message>
        <location filename="../ui_templates/filterByValueDlg.ui" line="102"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>FilterWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/filterwindow.cpp" line="428"/>
        <source>Open File</source>
        <translation>打开的文件</translation>
    </message>
</context>
<context>
    <name>GeneralFilterWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/generalfilterwindow.ui" line="14"/>
        <source>Form</source>
        <translation>分割窗口</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/generalfilterwindow.ui" line="36"/>
        <source>Config</source>
        <translation>配置</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/generalfilterwindow.ui" line="67"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
</context>
<context>
    <name>GeomFeaturesDialog</name>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="14"/>
        <source>Geometric features</source>
        <translation>几何特征</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="20"/>
        <source>Local neighborhood radius</source>
        <translation>局部邻域半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="42"/>
        <source>radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="85"/>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="91"/>
        <source>Roughness</source>
        <translation>粗糙度</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="101"/>
        <source>Curvature</source>
        <translation>曲率</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="107"/>
        <source>Mean curvature (unsigned)</source>
        <translation>平均曲率（无符号）</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="110"/>
        <source>Mean</source>
        <translation>均值</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="117"/>
        <source>Gaussian curvature (unsigned)</source>
        <translation>高斯曲率（无符号）</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="120"/>
        <source>Gaussian</source>
        <translation>高斯</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="127"/>
        <source>&apos;Speed&apos; of orientation change</source>
        <translation>方向改变速率</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="130"/>
        <source>Normal change rate</source>
        <translation>法线变化率</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="140"/>
        <source>Density</source>
        <translation>密度</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="146"/>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="149"/>
        <source>Number of neighbors</source>
        <translation>近邻数</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="156"/>
        <source>Number of neighbors / neighborhood area</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="159"/>
        <source>Surface density</source>
        <translation>表面密度</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="166"/>
        <source>Number of neighbors / neighborhood volume</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="169"/>
        <source>Volume density</source>
        <translation>体积密度</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="196"/>
        <source>Geometric features (based on local eigenvalues: (L1, L2, L3))</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="199"/>
        <source>Feature</source>
        <translation>特征</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="205"/>
        <source>L1 + L2 + L3</source>
        <translation>L1 + L2 + L3</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="208"/>
        <source>Sum of eigenvalues</source>
        <translation>总特征值</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="215"/>
        <source>(L1 * L2 * L3)^(1/3)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="218"/>
        <source>Ominvariance</source>
        <translation>协方差（OmiVariance）</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="225"/>
        <source>-( L1*ln(L1) + L2*ln(L2) + L3*ln(L3) )</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="228"/>
        <source>Eigenentropy</source>
        <translation>特征熵</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="235"/>
        <source>(L1 - L3)/L1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="238"/>
        <source>Anisotropy</source>
        <translation>各向异性</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="245"/>
        <source>(L2 - L3)/L1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="248"/>
        <source>Planarity</source>
        <translation>平面性</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="255"/>
        <source>(L1 - L2)/L1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="258"/>
        <source>Linearity</source>
        <translation>线性</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="265"/>
        <source>L1 / (L1 + L2 + L3)</source>
        <translation>L1 / (L1 + L2 + L3)</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="268"/>
        <source>PCA1</source>
        <translation>PCA1</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="275"/>
        <source>L2 / (L1 + L2 + L3)</source>
        <translation>L2/(L1 + L2 + L3)</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="278"/>
        <source>PCA2</source>
        <translation>PCA2</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="285"/>
        <source>L3 / (L1 + L2 + L3)</source>
        <translation>L3/(L1 + L2 + L3)</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="288"/>
        <source>Surface variation</source>
        <translation>表面变化</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="295"/>
        <source>L3 / L1</source>
        <translation>L3/L1</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="298"/>
        <source>Sphericity</source>
        <translation>球形</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="305"/>
        <source>1 - |Z.N|</source>
        <translation>1 - Z.N</translation>
    </message>
    <message>
        <location filename="../ui_templates/geomFeaturesDlg.ui" line="308"/>
        <source>Verticality</source>
        <translation>垂直度</translation>
    </message>
</context>
<context>
    <name>GlobalShiftAndScaleAboutDlg</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleAboutDlg.ui" line="14"/>
        <source>Issue with big coordinates</source>
        <translation>大坐标值问题</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleAboutDlg.ui" line="20"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:10pt; font-weight:600;&quot;&gt;Why CLOUDVIEWER  bugs me about &amp;quot;too big coordinates&amp;quot;?&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;To reduce memory consumption of big clouds, CLOUDVIEWER  stores their points coordinates on 32 bits (&lt;/span&gt;&lt;a href=&quot;http://en.wikipedia.org/wiki/Single-precision_floating-point_format&quot;&gt;&lt;span style=&quot; font-size:9pt; text-decoration: underline; color:#0000ff;&quot;&gt;single-precision floating-point format&lt;/span&gt;&lt;/a&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;). In effect this roughly corresponds to a &lt;/span&gt;&lt;span style=&quot; font-size:9pt; font-weight:600;&quot;&gt;relative&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt; precision of 10&lt;/span&gt;&lt;span style=&quot; font-size:9pt; vertical-align:super;&quot;&gt;-7&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;.&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;This is generally totally acceptable for an object of a few meters wide (in which case the precision will be around a few tenths of microns). However if the coordinates are of the order of 10&lt;/span&gt;&lt;span style=&quot; font-size:9pt; vertical-align:super;&quot;&gt;5&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt; or 10&lt;/span&gt;&lt;span style=&quot; font-size:9pt; vertical-align:super;&quot;&gt;6&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt; meters and you still expect a precision around a few microns this won&apos;t do (at all). Importing such coordinates in 32 bits format will result in a precision of several centimeters or worse!&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt; font-style:italic;&quot;&gt;You&apos;ll probably also observe very strange things in 3D as OpenGL doesn&apos;t like those big coordinates either.&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt; font-style:italic;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:10pt; font-weight:600;&quot;&gt;What can I do?&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt; font-weight:600;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;This &amp;quot;big coordinates&amp;quot; issue typically arises when an object of a few meters wide is expressed in a global geographic coordinate system. This happens also for other units (&lt;/span&gt;&lt;span style=&quot; font-size:9pt; font-style:italic;&quot;&gt;we used meters here as an example&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;).&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;As the absolute position of clouds is generally not used during a comparison process (and most of the other processings) the best solution to this &amp;quot;big coordinates&amp;quot; issue is to temporarily shift the data to a local coordinate system.  &lt;/span&gt;&lt;span style=&quot; font-size:9pt; font-weight:600;&quot;&gt;The inverse shift will be applied to the data at export time so that no information is lost.&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;Another example: your cloud still represents a several meters wide object but its coordinates are expressed in microns (&lt;/span&gt;&lt;span style=&quot; font-size:9pt; font-style:italic;&quot;&gt;once again meters and microns are used as an example&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;). In this case you can apply a scaling factor so as to work in a more &amp;quot;standard&amp;quot; local coordinate system (&lt;/span&gt;&lt;span style=&quot; font-size:9pt; font-style:italic;&quot;&gt;e.g. in centimeters or meters&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;). In effect this is equivalent to changing the cloud units (temporarily). The inverse scaling factor will be applied at export time.&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:10pt; font-weight:600;&quot;&gt;Shift &amp;amp; Scale values&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;By default CLOUDVIEWER  tries to guess the shift vector itself. But you can of course input your own version (especially if you work with several clouds and you want to shift them all in the same local coordinate system).&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;Once you input a shift vector (and/or a scale factor) you&apos;ll be able to use it again while importing other clouds (&lt;/span&gt;&lt;span style=&quot; font-size:9pt; font-style:italic;&quot;&gt;it will correspond to the &amp;quot;Last input&amp;quot; entry of the combo-box above the shift fields&lt;/span&gt;&lt;span style=&quot; font-size:9pt;&quot;&gt;).&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:9pt; color:#0055ff;&quot;&gt;However this information will only be stored during the active session of CLOUDVIEWER  (it will be lost once you close the program). To keep the information persistent, you can edit the &lt;/span&gt;&lt;span style=&quot; font-size:9pt; font-style:italic; color:#0055ff;&quot;&gt;global_shift_list_template.txt&lt;/span&gt;&lt;span style=&quot; font-size:9pt; color:#0055ff;&quot;&gt; file next to CLOUDVIEWER &apos;s executable and follow the instructions inside. This is a good way to store persistent shift/scale information sets (kind of &amp;quot;bookmarks&amp;quot;).&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>GlobalShiftAndScaleDlg</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="20"/>
        <source>Global shift/scale</source>
        <translation>全局偏移/比例</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="76"/>
        <source>Coordinates are too big (original precision may be lost)!</source>
        <translation>坐标值太大（原始精度可能丢失）！</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="86"/>
        <source>More information about this issue</source>
        <translation>有关此问题的更多信息</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="132"/>
        <source>Do you wish to translate/rescale the entity?</source>
        <translation>您希望平移或者缩放实体吗？</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="168"/>
        <source>shift/scale information is stored and used to restore the original coordinates at export time</source>
        <translation>偏移/缩放比例信息将会保存 并在导出时恢复原始坐标</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="213"/>
        <source>This version corresponds to the input (or output) file</source>
        <translation>这是原始导入文件版本坐标</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="237"/>
        <source>Point in original
coordinate system (on disk)</source>
        <translation>原始点坐标系统（硬盘）</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="315"/>
        <source>diagonal = 3213132123.3215</source>
        <translation>对角线= 3213132123.3215</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="423"/>
        <source>Shift</source>
        <translation>偏移</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="450"/>
        <source>Scale</source>
        <translation>缩放比例</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="563"/>
        <source>You can add default items to this list by placing a text file named &lt;span style=&quot; font-weight:600;&quot;&gt;global_shift_list.txt&lt;/span&gt; next to the application executable file. On each line you should define 5 items separated by semicolon characters: name ; ShiftX ; ShiftY ; ShiftZ ; scale</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="605"/>
        <source>This version is the one CLOUDVIEWER  will work with. Mind the digits!</source>
        <translation>此为逸舟点云处理系统使用坐标版本 注意数值！</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="629"/>
        <source>Point in local
coordinate system</source>
        <translation>转换后点坐标系统</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="707"/>
        <source>diagonal = 321313</source>
        <translation>对角线= 321313</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="730"/>
        <source>Warning: previously used shift and/or scale don&apos;t seem adapted to this entity</source>
        <translation>警告：之前使用的偏移或者缩放比例好像不适用于当前实体</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="774"/>
        <source>Preserve global shift on save</source>
        <translation>保存时保持全局偏移</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="784"/>
        <source>The local coordinates will be changed so as to keep the global coordinates the same</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/globalShiftAndScaleDlg.ui" line="787"/>
        <source>Keep original position fixed</source>
        <translation>保持原点位置固定</translation>
    </message>
</context>
<context>
    <name>GlobalShiftSettingsDialog</name>
    <message>
        <location filename="../ui_templates/globalShiftSettingsDlg.ui" line="14"/>
        <source>Global Shift settings</source>
        <translation>全局偏移设置</translation>
    </message>
    <message>
        <location filename="../ui_templates/globalShiftSettingsDlg.ui" line="26"/>
        <source>The Global Shift &amp; Scale mechanism aims at reducing the loss of precision when working with big coordinates.

As CLOUDVIEWER  and most graphic cards work with 32 bits floating point values, their resolution and the computation precision is limited. The bigger the numbers the less resolute they are.

Here below you can set the limits above which the &apos;Global Shift &amp; Scale&apos; mechanism will be triggered.

Note: the diagonal is not tested at loading time.</source>
        <translation>全局移位和缩放机制旨在减少使用大坐标时的精度损失。

由于CLOUDVIEWER和大多数图形卡都使用32位浮点值，因此它们的分辨率和计算精度受到限制。 数字越大，确定性就越差。

在下面，您可以设置限制，在该限制之上将触发“全局移位和缩放”机制。

注意：对角线在加载时未经过测试。</translation>
    </message>
    <message>
        <location filename="../ui_templates/globalShiftSettingsDlg.ui" line="42"/>
        <source>Global Shift &amp; Scale triggering limits:</source>
        <translation>全局偏移和缩放触发限制：</translation>
    </message>
    <message>
        <location filename="../ui_templates/globalShiftSettingsDlg.ui" line="51"/>
        <source>Max absolute coordinate</source>
        <translation>最大绝对坐标</translation>
    </message>
    <message>
        <location filename="../ui_templates/globalShiftSettingsDlg.ui" line="58"/>
        <source>CLOUDVIEWER  will suggest to apply a Global Shift to the loaded entities if their coordinates are above this limit</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/globalShiftSettingsDlg.ui" line="80"/>
        <source>Max entity diagonal</source>
        <translation>最大实体对角</translation>
    </message>
    <message>
        <location filename="../ui_templates/globalShiftSettingsDlg.ui" line="87"/>
        <source>CLOUDVIEWER  will suggest to apply a Global Scale to the loaded entities if their bounding-box diagonal is above this limit</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>GlyphConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="14"/>
        <source>Form</source>
        <translation>形式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="22"/>
        <source>File:</source>
        <translation>文件:</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="30"/>
        <source>Arrow</source>
        <translation>箭头</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="35"/>
        <source>Cone</source>
        <translation>锥</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="40"/>
        <source>Line</source>
        <translation>行</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="45"/>
        <source>Cylinder</source>
        <translation>圆柱体</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="50"/>
        <source>Sphere</source>
        <translation>球体</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="55"/>
        <source>Point</source>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="63"/>
        <source>Shape</source>
        <translation>形状</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="70"/>
        <source>Size</source>
        <translation>大小</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="77"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="198"/>
        <source>Color</source>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="91"/>
        <source>Open ...</source>
        <translation>打开……</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="102"/>
        <source>Transparent</source>
        <translation>透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="107"/>
        <source>Opaque</source>
        <translation>不透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="112"/>
        <source>Wireframe</source>
        <translation>线框</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="120"/>
        <source>Display Effect</source>
        <translation>显示效果</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="133"/>
        <source>Data Table</source>
        <translation>数据表</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="141"/>
        <source>Rows</source>
        <translation>行</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="158"/>
        <source>Random</source>
        <translation>随机</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="177"/>
        <source>Label</source>
        <translation>标签</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="191"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="256"/>
        <source>PushButton</source>
        <translation>按钮</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="205"/>
        <source>Mode</source>
        <translation>模式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="213"/>
        <source>Ids</source>
        <translation>id</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="218"/>
        <source>Scalars</source>
        <translation>标量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="223"/>
        <source>Vectors</source>
        <translation>向量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="228"/>
        <source>Normals</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="233"/>
        <source>TCoords</source>
        <translation>TCoords</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="238"/>
        <source>Tensors</source>
        <translation>张量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphconfig.ui" line="243"/>
        <source>Field Data</source>
        <translation>字段数据</translation>
    </message>
</context>
<context>
    <name>GlyphWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/glyphwindow.cpp" line="35"/>
        <source>Glyph</source>
        <translation>字形</translation>
    </message>
</context>
<context>
    <name>GraphicalFilteringWindowDlg</name>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="20"/>
        <source>Filter</source>
        <translation>过滤器</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="41"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="44"/>
        <source>mode</source>
        <translation>模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="58"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="61"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="64"/>
        <source>segment single</source>
        <translation>段单</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="75"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="78"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="81"/>
        <source>segment Mult</source>
        <translation>Much segment</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="92"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="95"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="98"/>
        <source>extract Contour</source>
        <translation>提取轮廓</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="118"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="121"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="124"/>
        <source>remove Last Contour</source>
        <translation>删除最后一个轮廓</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="135"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="138"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="141"/>
        <source>resetButton</source>
        <translation>resetButton</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="152"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="155"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="158"/>
        <source>restoreToolButton</source>
        <translation>restoreToolButton</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="169"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="175"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="172"/>
        <source>cancelButton</source>
        <translation>cancelButton</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="190"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="193"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="196"/>
        <source>Segmentation Extraction</source>
        <translation>分割抽取</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="205"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="208"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="211"/>
        <location filename="../ui_templates/graphicalFilteringWindowDlg.ui" line="214"/>
        <source>Polyline Extraction</source>
        <translation>多段线提取</translation>
    </message>
</context>
<context>
    <name>GraphicalRenderSurfaceWindowDlg</name>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="26"/>
        <source>Surface</source>
        <translation>曲面</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="38"/>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="41"/>
        <source>Export multiple slices by repeating the process along one or several dimensions (+ contour extraction)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="44"/>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="73"/>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="119"/>
        <source>...</source>
        <translation>...</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="67"/>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="70"/>
        <source>Export selection as a new cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="93"/>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="96"/>
        <source>Reset</source>
        <translation>重置</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="99"/>
        <source>raz</source>
        <translation>Raz</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="113"/>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="116"/>
        <source>Restore the last clipping box used with this cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="136"/>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="139"/>
        <source>Close</source>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalRenderSurfaceWindowDlg.ui" line="142"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>GraphicalSegmentationDlg</name>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="20"/>
        <source>Segmentation</source>
        <translation>分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="41"/>
        <source>Pause segmentation</source>
        <translation>暂停分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="44"/>
        <source>Pause segmentation  (allow rotation/panning of 3D view)</source>
        <translation>暂停分割（允许旋转缩放移动3D视图）</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="47"/>
        <source>pause</source>
        <translation>暂停</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="64"/>
        <source>Load / save segmentation polyline</source>
        <translation>读取/保存分割多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="67"/>
        <source>load/save segmentation polyline</source>
        <translation>读取/保存分割多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="81"/>
        <source>Polyline selection mode</source>
        <translation>多边形选择模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="84"/>
        <source>polyline selection</source>
        <translation>多边形选择</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="98"/>
        <source>Segment In</source>
        <translation>分割选择区域</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="101"/>
        <source>Segment (keep points inside)</source>
        <translation>分割（保存内部点）</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="104"/>
        <source>in</source>
        <translation>分割选择区域内部</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="115"/>
        <source>Segment Out</source>
        <translation>分割选择区域外</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="118"/>
        <source>Segment (keep points outside)</source>
        <translation>分割（保持外部点）</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="121"/>
        <source>out</source>
        <translation>分割选择区域外部</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="135"/>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="138"/>
        <source>Clear segmentation</source>
        <translation>清除分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="141"/>
        <source>raz</source>
        <translation>Raz</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="152"/>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="155"/>
        <source>Confirm segmentation</source>
        <translation>确认分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="158"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="169"/>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="172"/>
        <source>Confirm and delete hidden points</source>
        <translation>确认并删除隐藏点</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="183"/>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="189"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="186"/>
        <source>Cancel segentation</source>
        <translation>取消分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="204"/>
        <source>Rectangular selection</source>
        <translation>矩形选择</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="207"/>
        <source>Activates rectangular selection</source>
        <translation>矩形选择模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="216"/>
        <source>Polygonal selection</source>
        <translation>多边形选择</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="219"/>
        <source>Activaites polyline selection</source>
        <translation>多边形选择模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="228"/>
        <source>Use existing polyline</source>
        <translation>使用存在的多边形</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="231"/>
        <source>Import polyline from DB for segmentation</source>
        <translation>从资源树中导入多段线用于分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="240"/>
        <source>Export segmentation polyline</source>
        <translation>导出分割多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalSegmentationDlg.ui" line="243"/>
        <source>Export segmentation polyline as new entity</source>
        <translation>导出分割多段线为新实体</translation>
    </message>
</context>
<context>
    <name>GraphicalTransformationDlg</name>
    <message>
        <location filename="../ui_templates/graphicalTransformationDlg.ui" line="20"/>
        <source>Graphical Transformation</source>
        <translation>模型空间变换</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalTransformationDlg.ui" line="81"/>
        <source>Pause segmentation</source>
        <translation>暂停分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalTransformationDlg.ui" line="84"/>
        <source>Pause transformation (allow rotation/panning of 3D view)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalTransformationDlg.ui" line="87"/>
        <source>pause</source>
        <translation>暂停</translation>
    </message>
    <message>
        <location filename="../ui_templates/graphicalTransformationDlg.ui" line="166"/>
        <source>Rotation</source>
        <translation>旋转</translation>
    </message>
</context>
<context>
    <name>GreedyTriangulation</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="40"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="41"/>
        <source>Greedy Triangulation</source>
        <translation>贪婪的三角</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="42"/>
        <source>Greedy Triangulation from clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="151"/>
        <source>[GreedyTriangulation::compute] generate new normals</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="157"/>
        <source>[GreedyTriangulation::compute] find normals and use the normals</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="186"/>
        <source>[greedy-triangulation-Reconstruction] %1 points, %2 face(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="188"/>
        <source>greedy-triangulation searchRadius[%1]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="209"/>
        <source>Selected entity does not have any suitable scalar field or RGB.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="211"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/GreedyTriangulation.cpp" line="213"/>
        <source>Greedy Triangulation does not returned any point. Try relaxing your parameters</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>GreedyTriangulationDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="20"/>
        <source>Greedy Triangulation</source>
        <translation>贪婪三角化</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="32"/>
        <source>Normal Estimation Parameters</source>
        <translation>法线评估参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="47"/>
        <source>Normal Search Radius</source>
        <translation>法线搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="76"/>
        <source>Use Knn Search</source>
        <translation>使用K近邻搜索</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="102"/>
        <source>Normal Consistency</source>
        <translation>法线一致性</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="118"/>
        <source>Greedy Triangulation Parameters</source>
        <translation>贪婪三角化参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="130"/>
        <source>Maximum Surface Angle</source>
        <translation>最大曲面角</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="150"/>
        <source>Minimum Angle</source>
        <translation>最小角</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="170"/>
        <source>Maximum Nearest Neighbors</source>
        <translation>最大最近邻数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="190"/>
        <source>Maximum Angle</source>
        <translation>最大角</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="210"/>
        <source>Triangulation Search Radius</source>
        <translation>三角化搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/GreedyTriangulationDlg.ui" line="249"/>
        <source>Weighting Factor</source>
        <translation>权重因子</translation>
    </message>
</context>
<context>
    <name>HPRDialog</name>
    <message>
        <location filename="../../plugins/core/qHPR/ui/hprDlg.ui" line="14"/>
        <source>HPR</source>
        <translation>HPR</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/ui/hprDlg.ui" line="22"/>
        <source>Level</source>
        <translation>八叉树层数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/ui/hprDlg.ui" line="29"/>
        <source>Octree Level (for point cloud shape approx.)</source>
        <translation>八叉树层数（点云形状估计）</translation>
    </message>
</context>
<context>
    <name>HistogramDialog</name>
    <message>
        <location filename="../ui_templates/histogramDlg.ui" line="14"/>
        <source>Histogram</source>
        <translation>直方图</translation>
    </message>
    <message>
        <location filename="../ui_templates/histogramDlg.ui" line="60"/>
        <source>Export histogram to a CSV file (can be imported into Excel ;)</source>
        <translation>导出直方图为CSV文件（可以导入Excel软件）</translation>
    </message>
    <message>
        <location filename="../ui_templates/histogramDlg.ui" line="71"/>
        <source>Export histogram to an image file</source>
        <translation>导出直方图为图像文件</translation>
    </message>
</context>
<context>
    <name>ImportDBFFieldDlg</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/importDBFFieldDlg.ui" line="14"/>
        <source>Choose altitude field</source>
        <translation>选择高程域</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/importDBFFieldDlg.ui" line="20"/>
        <source>Do you wish to use one of the DBF field as altitude?</source>
        <translation>您希望使用DBF域之一作为高程吗？</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/importDBFFieldDlg.ui" line="32"/>
        <source>Values scaling</source>
        <translation>值缩放比例</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/importDBFFieldDlg.ui" line="72"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/importDBFFieldDlg.ui" line="79"/>
        <source>Ignore</source>
        <translation>忽略</translation>
    </message>
</context>
<context>
    <name>InterpolationDlg</name>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="14"/>
        <source>Interpolation</source>
        <translation>插值</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="20"/>
        <source>Neighbors extraction</source>
        <translation>近邻提取</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="26"/>
        <source>Radius of the sphere inside which neighbors will be extracted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="39"/>
        <source>Extracts the neighbors inside a sphere</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="42"/>
        <source>Radius (Sphere)</source>
        <translation>半径(球)</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="52"/>
        <source>Use only the nearest neighbor (fast)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="55"/>
        <source>Nearest neighbor</source>
        <translation>近邻</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="62"/>
        <source>Use the &apos;k&apos; nearest neighbors
(faster than &apos;radius&apos; based search, but more approximate)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="66"/>
        <source>Nearest neighbors</source>
        <translation>最近邻</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="76"/>
        <source>Number of neighbors to extract</source>
        <translation>近邻提取数</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="95"/>
        <source>Interpolation algorithm</source>
        <translation>插值算法</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="101"/>
        <source>Keep the median of the neighbors SF values</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="104"/>
        <source>Median</source>
        <translation>中位数</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="114"/>
        <source>Keep the average of the neighbors SF values</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="117"/>
        <source>Average</source>
        <translation>平均</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="124"/>
        <source>Compute a weighted average of the neighbors SF values
(the weights will follow a Normal distribution)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="128"/>
        <source>Normal distribution</source>
        <translation>正态分布</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="163"/>
        <source>sigma</source>
        <translation>σ</translation>
    </message>
    <message>
        <location filename="../ui_templates/interpolationDlg.ui" line="170"/>
        <source>Kernel of the Normal distribution</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>IsosurfaceConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="20"/>
        <source>Form</source>
        <translation>等值曲面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="28"/>
        <source>Min Scalar</source>
        <translation>My Scalar</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="35"/>
        <source>Scalars</source>
        <translation>标量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="42"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="62"/>
        <source>Max Scalar</source>
        <translation>最大标量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="69"/>
        <source>Number of Contours</source>
        <translation>轮廓数量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="89"/>
        <source>Open...</source>
        <translation>打开……</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="114"/>
        <source>Transparent</source>
        <translation>透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="119"/>
        <source>Opaque</source>
        <translation>不透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="124"/>
        <source>Wireframe</source>
        <translation>线框</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="132"/>
        <source>Display Effect</source>
        <translation>显示效果</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfaceconfig.ui" line="142"/>
        <source>Gradient</source>
        <translation>渐变色</translation>
    </message>
</context>
<context>
    <name>IsosurfaceWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/isosurfacewindow.cpp" line="38"/>
        <source>Isosurface</source>
        <translation>等值面</translation>
    </message>
</context>
<context>
    <name>ItemSelectionDlg</name>
    <message>
        <location filename="../ui_templates/itemSelectionDlg.ui" line="14"/>
        <source>Selection</source>
        <translation>选择</translation>
    </message>
    <message>
        <location filename="../ui_templates/itemSelectionDlg.ui" line="20"/>
        <source>Please select one %1:</source>
        <translation>请选择一个 %1:</translation>
    </message>
</context>
<context>
    <name>LabelingDialog</name>
    <message>
        <location filename="../ui_templates/labelingDlg.ui" line="20"/>
        <source>Connected Components</source>
        <translation>连接组件</translation>
    </message>
    <message>
        <location filename="../ui_templates/labelingDlg.ui" line="28"/>
        <source>Grid subdivision level: the greater, the finest</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/labelingDlg.ui" line="31"/>
        <source>Octree Level</source>
        <translation>八叉树层数</translation>
    </message>
    <message>
        <location filename="../ui_templates/labelingDlg.ui" line="57"/>
        <source>Minimum number of points per component</source>
        <translation>每个部分最小数量</translation>
    </message>
    <message>
        <location filename="../ui_templates/labelingDlg.ui" line="60"/>
        <source>Min. points per component</source>
        <translation>每个部分最少点数</translation>
    </message>
    <message>
        <location filename="../ui_templates/labelingDlg.ui" line="91"/>
        <source>random colors (warning: overwrites existing ones)</source>
        <translation>随机颜色（警告：覆盖已有颜色）</translation>
    </message>
</context>
<context>
    <name>M3C2Dialog</name>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="14"/>
        <source>M3C2 distance</source>
        <translation>M3C2距离</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="28"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="787"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="910"/>
        <source>Cloud #1</source>
        <translation>点云 #1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="41"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="777"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="915"/>
        <source>Cloud #2</source>
        <translation>点云 #2</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="70"/>
        <source>Main parameters</source>
        <translation>主要参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="76"/>
        <source>Scales</source>
        <translation>尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="82"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="130"/>
        <source>diameter = </source>
        <translation>直径 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="107"/>
        <source>Projection</source>
        <translation>投影</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="120"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="302"/>
        <source>Normals</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="149"/>
        <source>max depth = </source>
        <translation>最大深度 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="174"/>
        <source>Core points</source>
        <translation>核心点</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="180"/>
        <source>use cloud #1</source>
        <translation>使用点云 # 1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="187"/>
        <source>use other cloud</source>
        <translation>使用其他点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="197"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="553"/>
        <source>Alternative core points cloud</source>
        <translation>可选核心点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="204"/>
        <source>subsample cloud #1</source>
        <translation>下采样点云 # 1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="214"/>
        <source>Min. distance between points</source>
        <translation>最小点间距</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="239"/>
        <source>Registration error</source>
        <translation>配准误差</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="249"/>
        <source>Registration error (RMS - to be input by the user)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="275"/>
        <source>Tries to guess some parameters automatically</source>
        <translation>尝试自动化猜测一些参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="278"/>
        <source>Guess params</source>
        <translation>评估参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="323"/>
        <source>Calculation mode</source>
        <translation>计算模式</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="331"/>
        <source>Default fixed scale calculation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="334"/>
        <source>Default</source>
        <translation>默认</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="344"/>
        <source>Automatically use the scale at which the cloud is the more &apos;flat&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="347"/>
        <source>Multi-scale</source>
        <translation>多尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="354"/>
        <source>Make the resulting normals purely Vertical</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="357"/>
        <source>Vertical</source>
        <translation>垂直的</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="364"/>
        <source>Make the resulting normals purely Horizontal</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="367"/>
        <source>Horizontal</source>
        <translation>水平的</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="382"/>
        <source>Mininum scale</source>
        <translation>最少购买比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="385"/>
        <source>Min = </source>
        <translation>最小值 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="404"/>
        <source>Step</source>
        <translation>步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="407"/>
        <source>Step = </source>
        <translation>步长 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="426"/>
        <source>Max scale</source>
        <translation>最大比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="429"/>
        <source>Max = </source>
        <translation>最大值 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="451"/>
        <source>Use core points for normal calculation (instead of cloud #1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="454"/>
        <source>Use core points for normal calculation</source>
        <translation>基于核心点云法线计算</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="464"/>
        <source>Orientation</source>
        <translation>方向</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="470"/>
        <source>Preferred orientation</source>
        <translation>优选方向</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="517"/>
        <source>+Barycenter</source>
        <translation>+重心</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="522"/>
        <source>- Barycenter</source>
        <translation>-重心</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="540"/>
        <source>Sensor(s) position(s) as a cloud (one point per position)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="543"/>
        <source>Use sensor position(s)</source>
        <translation>使用传感器位置</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="580"/>
        <source>Advanced</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="586"/>
        <source>Slower but it guarantees that all the cylinder will be explored</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="589"/>
        <source>Do not use multiple pass for depth</source>
        <translation>深度不使用多通道</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="596"/>
        <source>Search the points only in the &apos;positive&apos; side of the cylinder (relatively to the point normal)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="599"/>
        <source>Only search points in the positive half-space (relatively to the normal)</source>
        <translation>仅在正半空间搜索点（相对于法线）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="606"/>
        <source>Use median and interquatile range (instead of mean and std. dev.)</source>
        <translation>使用中值范围（不是均值和标准差）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="628"/>
        <source>Specify minimum number of points for statistics computation</source>
        <translation>指定统计计算最少点数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="688"/>
        <source>Max thread count</source>
        <translation>最大线程数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="732"/>
        <source>Precision maps</source>
        <translation>精度地图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="744"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:10pt; font-weight:600;&quot;&gt;3D uncertainty-based topographic change detection with SfM photogrammetry: precision maps for ground control and directly georeferenced surveys&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;Mike R. James, Stuart Robson and Mark W. Smith (&lt;/span&gt;&lt;a href=&quot;http://onlinelibrary.wiley.com/doi/10.1002/esp.4125/abstract&quot;&gt;&lt;span style=&quot; font-size:8pt; text-decoration: underline; color:#0000ff;&quot;&gt;DOI: 10.1002/esp.4125&lt;/span&gt;&lt;/a&gt;&lt;span style=&quot; font-size:8pt;&quot;&gt;)&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="765"/>
        <source>Use precision information in scalar fields instead of roughness-based uncertainty estimates</source>
        <translation>在标量字段使用精度信息而不是基于粗略不定估计</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="797"/>
        <source>Sigma(y)</source>
        <translation>Sigma (y)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="813"/>
        <source>Sigma(x)</source>
        <translation>Sigma (x)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="826"/>
        <source>Sigma(z)</source>
        <translation>Sigma (z)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="836"/>
        <source>Scale</source>
        <translation>比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="843"/>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="862"/>
        <source>From SF units to cloud units</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="885"/>
        <source>Output</source>
        <translation>输出</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="893"/>
        <source>Project core points on</source>
        <translation>投影核心点集在</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="920"/>
        <source>Keep original positions</source>
        <translation>保持原点位置</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="928"/>
        <source>use original cloud</source>
        <translation>使用原始点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="950"/>
        <source>Adds two scalar fields (std_cloud#1 and std_cloud#2)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="953"/>
        <source>Export standard deviation information</source>
        <translation>导出标准差信息</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="960"/>
        <source>Adds one scalar field (point count per core point)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="963"/>
        <source>Export point density at projection scale</source>
        <translation>导出投影范围内导出点密度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="989"/>
        <source>Load parameters from file</source>
        <translation>从文件读取参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/ui/qM3C2Dialog.ui" line="1000"/>
        <source>Save parameters to file</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>MLSDialog</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="14"/>
        <source>Moving Least Squares Smoothing and Reconstruction</source>
        <translation>移动最小二乘平滑和重建</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="30"/>
        <source>Search Radius</source>
        <translation>搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="53"/>
        <source>Compute Normals</source>
        <translation>计算法线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="66"/>
        <source>Use Polynomial 
(instead of tangent)</source>
        <translation>使用多项式（替代切线）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="77"/>
        <source>Polynomial Order</source>
        <translation>多项式阶数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="94"/>
        <source>Squared Gaussian 
Parameter</source>
        <translation>高斯平方参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="118"/>
        <source>Upsampling Method</source>
        <translation>上采样方法</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="138"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.cpp" line="47"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.cpp" line="56"/>
        <source>Sample Local Plane</source>
        <translation>采样局部平面</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="147"/>
        <source>Upsampling Radius</source>
        <translation>上采样半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="176"/>
        <source>Upsampling Step Size</source>
        <translation>上采样步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="205"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.cpp" line="48"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.cpp" line="60"/>
        <source>Random Uniform Density</source>
        <translation>随机均匀密度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="214"/>
        <source>Step Point Density</source>
        <translation>步点密度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="237"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.cpp" line="49"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.cpp" line="64"/>
        <source>Voxel Grid Dilation</source>
        <translation>体素网格膨胀</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="246"/>
        <source>Dilation Voxel Size</source>
        <translation>膨胀体素大小</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.ui" line="272"/>
        <source>Dilation Iterations</source>
        <translation>膨胀迭代次数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/MLSDialog.cpp" line="46"/>
        <source>None</source>
        <translation>无</translation>
    </message>
</context>
<context>
    <name>MLSSmoothingUpsampling</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/MLSSmoothingUpsampling.cpp" line="41"/>
        <source>MLS smoothing</source>
        <translation>MLS平滑</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/MLSSmoothingUpsampling.cpp" line="42"/>
        <source>Smooth using MLS, optionally upsample</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/MLSSmoothingUpsampling.cpp" line="43"/>
        <source>Smooth the cloud using Moving Least Sqares algorithm, estimate normals and optionally upsample</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>MainViewerClass</name>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="35"/>
        <source>ACloudViewer</source>
        <translation>逸舟点云处理系统</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="80"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="90"/>
        <location filename="../ui_templates/MainWindow.ui" line="754"/>
        <source>About</source>
        <translation>关于</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="99"/>
        <source>Option</source>
        <translation>选项</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="103"/>
        <source>Theme</source>
        <translation>主题</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="129"/>
        <source>Language</source>
        <translation>语言</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="142"/>
        <source>Display</source>
        <translation>显示</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="146"/>
        <source>Angle view</source>
        <translation>角度视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="163"/>
        <source>Toolbars</source>
        <translation>工具栏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="180"/>
        <source>Tools</source>
        <translation>工具</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="184"/>
        <source>Segmentation</source>
        <translation>分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="196"/>
        <source>Sand box (research)</source>
        <translation>砂箱(研究)</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="217"/>
        <source>Volume</source>
        <translation>体积</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="223"/>
        <source>Distances</source>
        <translation>距离</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="231"/>
        <location filename="../ui_templates/MainWindow.ui" line="1747"/>
        <source>Fit</source>
        <translation>拟合</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="240"/>
        <source>Batch export</source>
        <translation>批量导出</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="247"/>
        <source>Registration</source>
        <translation>配准</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="272"/>
        <location filename="../ui_templates/MainWindow.ui" line="1758"/>
        <source>Edit</source>
        <translation>编辑</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="276"/>
        <location filename="../ui_templates/MainWindow.ui" line="279"/>
        <source>Color Edit Menu</source>
        <translation>颜色编辑菜单</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="282"/>
        <source>Color</source>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="303"/>
        <location filename="../ui_templates/MainWindow.ui" line="1448"/>
        <source>Scalar fields</source>
        <translation>标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="329"/>
        <source>Octree</source>
        <translation>八叉树</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="337"/>
        <source>Normals</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="341"/>
        <source>Orient normals</source>
        <translation>方向法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="348"/>
        <source>Convert to</source>
        <translation>转换为</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="362"/>
        <source>Mesh</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="366"/>
        <source>Scalar Field</source>
        <translation>标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="388"/>
        <source>Polyline</source>
        <translation>多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="395"/>
        <location filename="../ui_templates/MainWindow.ui" line="2063"/>
        <source>Plane</source>
        <translation>平面</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="449"/>
        <source>DB Tree</source>
        <translation>资源树</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="517"/>
        <source>Properties</source>
        <translation>属性</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="578"/>
        <location filename="../ui_templates/MainWindow.ui" line="1011"/>
        <source>Console</source>
        <translation>控制台</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="612"/>
        <location filename="../ui_templates/MainWindow.ui" line="615"/>
        <source>Viewing tools</source>
        <translation>视图工具栏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="638"/>
        <location filename="../ui_templates/MainWindow.ui" line="641"/>
        <source>Main tools</source>
        <translation>主工具栏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="675"/>
        <location filename="../ui_templates/MainWindow.ui" line="678"/>
        <source>Scalar field tools</source>
        <translation>标量字段工具栏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="700"/>
        <source>Open</source>
        <translation>打开</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="703"/>
        <source>open a exsting file</source>
        <translation>打开已存在文件</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="706"/>
        <source>Ctrl+O</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="715"/>
        <source>Save</source>
        <translation>保存</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="718"/>
        <source>save the file</source>
        <translation>保存文件</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="721"/>
        <source>Ctrl+S</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="730"/>
        <source>Primitive factory</source>
        <translation>基础模型库</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="733"/>
        <source>generate a cube point cloud</source>
        <translation>生成一个立方点云体</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="742"/>
        <source>Help</source>
        <translation>帮助</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="745"/>
        <source>show help information</source>
        <translation>显示帮助信息</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="757"/>
        <source>show some information of the software</source>
        <translation>显示系统信息</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="766"/>
        <source>Exit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="769"/>
        <source>Exit Application</source>
        <translation>退出应用程序</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="772"/>
        <source>Ctrl+Q</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="781"/>
        <source>Set background color</source>
        <translation>设置背景颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="784"/>
        <location filename="../ui_templates/MainWindow.ui" line="787"/>
        <source>Set Background color</source>
        <translation>设置背景颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="790"/>
        <source>Alt+B</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="802"/>
        <source>Front View</source>
        <translation>前视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="805"/>
        <location filename="../ui_templates/MainWindow.ui" line="808"/>
        <source>Set front view</source>
        <translation>设置前视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="811"/>
        <source>5</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="823"/>
        <location filename="../ui_templates/MainWindow.ui" line="826"/>
        <source>Left Side View</source>
        <translation>左视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="829"/>
        <location filename="../ui_templates/MainWindow.ui" line="832"/>
        <source>Set left side view</source>
        <translation>设置左侧视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="835"/>
        <source>4</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="847"/>
        <source>Top view</source>
        <translation>俯视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="850"/>
        <location filename="../ui_templates/MainWindow.ui" line="853"/>
        <source>Set top view</source>
        <translation>设置俯视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="856"/>
        <source>8</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="868"/>
        <location filename="../ui_templates/MainWindow.ui" line="871"/>
        <source>Clear All</source>
        <translation>清空资源树</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="880"/>
        <source>Point picking</source>
        <translation>点拾取</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="883"/>
        <location filename="../ui_templates/MainWindow.ui" line="886"/>
        <source>Point picking (point information, distance between 2 points, angles between 3 points, etc.)</source>
        <translation>点拾取（单点信息，两点距离，三点角度面积等）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="895"/>
        <source>Point list picking</source>
        <translation>点拾取列表</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="898"/>
        <source>Pick several points (and export them to ASCII file, a new cloud, etc.)</source>
        <translation>拾取几个点（并导出到ASCII文件或者生成新的点云等）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="903"/>
        <source>Blue</source>
        <translation>蓝色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="908"/>
        <source>LightBlue</source>
        <translation>浅蓝色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="913"/>
        <source>English</source>
        <translation>英语</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="918"/>
        <source>Chinese</source>
        <translation>中文</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="926"/>
        <source>Enable Qt warnings in Console</source>
        <translation>控制台显示QT警告信息</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="931"/>
        <source>DarkBlue</source>
        <translation>深蓝色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="936"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="941"/>
        <source>LightBlack</source>
        <translation>浅黑色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="946"/>
        <source>FlatBlack</source>
        <translation>纯黑色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="951"/>
        <source>DarkBlack</source>
        <translation>深黑色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="956"/>
        <source>PsBlack</source>
        <translation>PS黑色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="961"/>
        <source>Silver</source>
        <translation>银色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="973"/>
        <source>&amp;Full screen</source>
        <translation>&amp;全屏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="976"/>
        <source>Full screen</source>
        <translation>全屏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="979"/>
        <location filename="../ui_templates/MainWindow.ui" line="982"/>
        <source>Switch to full screen</source>
        <translation>切换全屏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="985"/>
        <source>F9</source>
        <translation>F9</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="993"/>
        <source>Reset all GUI element positions</source>
        <translation>重置界面布局</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="996"/>
        <source>Reset all GUI element positions (after restart)</source>
        <translation>重置界面布局（重启生效）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1014"/>
        <source>F8</source>
        <translation>F8</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1029"/>
        <source>Full screen (3D view)</source>
        <translation>全屏（3D视图）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1032"/>
        <source>Exclusive full screen (3D view)</source>
        <translation>全屏（3D 视图）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1035"/>
        <source>F11</source>
        <translation>F11</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1047"/>
        <location filename="../ui_templates/MainWindow.ui" line="2146"/>
        <source>Delete</source>
        <translation>删除</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1050"/>
        <source>Del</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1062"/>
        <source>Set unique color</source>
        <translation>设置单一色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1065"/>
        <location filename="../ui_templates/MainWindow.ui" line="1068"/>
        <source>Set a unique color</source>
        <translation>设置单一的颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1071"/>
        <source>Alt+C</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1083"/>
        <source>Colorize</source>
        <translation>彩色化</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1086"/>
        <location filename="../ui_templates/MainWindow.ui" line="1089"/>
        <source>Colorize entity (lightness values are unchanged)</source>
        <translation>上色实体(保持灯光值不变)</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1097"/>
        <source>Levels</source>
        <translation>层</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1102"/>
        <source>Height Ramp</source>
        <translation>高度斜坡</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1105"/>
        <location filename="../ui_templates/MainWindow.ui" line="1108"/>
        <source>Apply a color ramp along X, Y or Z</source>
        <translation>沿着 X, Y or Z轴应用色带</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1117"/>
        <source>Convert to grey scale</source>
        <translation>转成灰色域</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1120"/>
        <location filename="../ui_templates/MainWindow.ui" line="1123"/>
        <source>Convert RGB colors to grey scale colors</source>
        <translation>RGB颜色域转灰度域</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1132"/>
        <location filename="../ui_templates/MainWindow.ui" line="1135"/>
        <source>Convert to Scalar field</source>
        <translation>转到标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1140"/>
        <location filename="../ui_templates/MainWindow.ui" line="2216"/>
        <source>Interpolate from another entity</source>
        <translation>从另一实体插值</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1143"/>
        <location filename="../ui_templates/MainWindow.ui" line="1146"/>
        <source>Interpolate colors from another entity (cloud or mesh) - color is taken from the nearest neighbor</source>
        <translation>从另一实体（点云或者网格）插值颜色-基于最近邻颜色插值</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1151"/>
        <location filename="../ui_templates/MainWindow.ui" line="1154"/>
        <source>Enhance with intensities</source>
        <translation>增强强度</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1163"/>
        <location filename="../ui_templates/MainWindow.ui" line="1166"/>
        <location filename="../ui_templates/MainWindow.ui" line="1169"/>
        <location filename="../ui_templates/MainWindow.ui" line="1172"/>
        <source>Clear colors</source>
        <translation>清除颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1181"/>
        <source>Zoom &amp; Center</source>
        <translation>缩放 &amp; 居中</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1184"/>
        <source>ZoomCenter</source>
        <translation>缩放中心</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1187"/>
        <location filename="../ui_templates/MainWindow.ui" line="1190"/>
        <source>Zoom and center on selected entities (Z)</source>
        <translation>居中缩放实体（Z）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1202"/>
        <location filename="../ui_templates/MainWindow.ui" line="1205"/>
        <source>Global Zoom</source>
        <translation>全局缩放</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1214"/>
        <location filename="../ui_templates/MainWindow.ui" line="1217"/>
        <source>Display settings</source>
        <translation>显示设置</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1226"/>
        <source>Back View</source>
        <translation>后视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1229"/>
        <location filename="../ui_templates/MainWindow.ui" line="1232"/>
        <source>Set back view</source>
        <translation>设置后视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1235"/>
        <source>0</source>
        <translation>0</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1247"/>
        <source>Right Side View</source>
        <translation>右侧视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1250"/>
        <location filename="../ui_templates/MainWindow.ui" line="1253"/>
        <source>Set right side view</source>
        <translation>设置右侧视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1256"/>
        <source>6</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1268"/>
        <source>Bottom View</source>
        <translation>底视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1271"/>
        <location filename="../ui_templates/MainWindow.ui" line="1274"/>
        <source>Set bottom view</source>
        <translation>设置底视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1277"/>
        <source>2</source>
        <translation>2</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1289"/>
        <source>Iso 1</source>
        <translation>Iso 1</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1292"/>
        <location filename="../ui_templates/MainWindow.ui" line="1295"/>
        <source>Set view to &apos;front&apos; isometric</source>
        <translation>设置等距前视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1298"/>
        <source>7</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1310"/>
        <source>Iso 2</source>
        <translation>Iso 2</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1313"/>
        <location filename="../ui_templates/MainWindow.ui" line="1316"/>
        <source>Set view to &apos;back&apos; isometric</source>
        <translation>设置等距后视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1319"/>
        <source>9</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1331"/>
        <source>Clone</source>
        <translation>克隆</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1334"/>
        <location filename="../ui_templates/MainWindow.ui" line="1337"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Clone the selected entities&lt;/p&gt;&lt;p&gt;&lt;span style=&quot; font-style:italic;&quot;&gt;(yes Claire ... these are Nyan sheep!)&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>复制实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1346"/>
        <source>Merge</source>
        <translation>融合</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1349"/>
        <location filename="../ui_templates/MainWindow.ui" line="1352"/>
        <source>Merge multiple clouds</source>
        <translation>多点云合并</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1364"/>
        <location filename="../ui_templates/MainWindow.ui" line="1367"/>
        <source>Cross Section</source>
        <translation>矩形框裁剪</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1376"/>
        <location filename="../ui_templates/MainWindow.ui" line="1379"/>
        <source>Segment</source>
        <translation>分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1388"/>
        <location filename="../ui_templates/MainWindow.ui" line="1527"/>
        <source>Compute</source>
        <translation>计算</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1391"/>
        <location filename="../ui_templates/MainWindow.ui" line="1394"/>
        <source>Compute unsigned normals (least squares approx.)</source>
        <translation>计算无符号法线（最小二乘法）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1403"/>
        <source>Clear</source>
        <translation>清空</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1406"/>
        <location filename="../ui_templates/MainWindow.ui" line="1409"/>
        <source>Delete normals</source>
        <translation>删除法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1414"/>
        <source>Invert</source>
        <translation>反转</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1417"/>
        <location filename="../ui_templates/MainWindow.ui" line="1420"/>
        <source>Invert normals</source>
        <translation>反转法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1431"/>
        <source>Main</source>
        <translation>主</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1434"/>
        <location filename="../ui_templates/MainWindow.ui" line="1437"/>
        <source>Show/hide main toolbar</source>
        <translation>显示/隐藏主工具栏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1451"/>
        <location filename="../ui_templates/MainWindow.ui" line="1454"/>
        <source>Show/hide scalar fields toolbar</source>
        <translation>显示/隐藏标量工具栏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1465"/>
        <source>View</source>
        <translation>视图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1468"/>
        <location filename="../ui_templates/MainWindow.ui" line="1471"/>
        <source>Show/hide view toolbar</source>
        <translation>显示/隐藏视图工具栏</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1476"/>
        <source>About Plugins...</source>
        <translation>关于插件…</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1485"/>
        <source>Color Scales Manager</source>
        <translation>颜色标量管理</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1494"/>
        <source>Filter Section</source>
        <translation>过滤部分</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1497"/>
        <location filename="../ui_templates/MainWindow.ui" line="1500"/>
        <source>Filter Window Section</source>
        <translation>滤波窗口</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1509"/>
        <source>Trace Polyline</source>
        <translation>循迹多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1512"/>
        <location filename="../ui_templates/MainWindow.ui" line="1515"/>
        <source>Trace a polyline by point picking</source>
        <translation>基于点拾取循迹多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1518"/>
        <source>Ctrl+P</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1530"/>
        <location filename="../ui_templates/MainWindow.ui" line="1533"/>
        <source>Compute octree</source>
        <translation>计算八叉树</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1538"/>
        <source>Resample</source>
        <translation>重新取样</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1541"/>
        <location filename="../ui_templates/MainWindow.ui" line="1544"/>
        <source>Resample entity with octree</source>
        <translation>基于八叉树重新采样实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1549"/>
        <location filename="../ui_templates/MainWindow.ui" line="1552"/>
        <source>With Minimum Spanning Tree</source>
        <translation>基于最小生成树</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1557"/>
        <location filename="../ui_templates/MainWindow.ui" line="1560"/>
        <source>With Fast Marching</source>
        <translation>基于快速行进模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1565"/>
        <location filename="../ui_templates/MainWindow.ui" line="1568"/>
        <source>HSV colors</source>
        <translation>HSV颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1573"/>
        <location filename="../ui_templates/MainWindow.ui" line="1576"/>
        <source>Dip/Dip direction SFs</source>
        <translation>倾斜方向标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1581"/>
        <source>Surface</source>
        <translation>曲面</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1584"/>
        <location filename="../ui_templates/MainWindow.ui" line="1587"/>
        <source>Surface Rendering</source>
        <translation>曲面渲染</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1592"/>
        <source>Delaunay 2.5D (XY plane)</source>
        <translation>德劳内2.5D（XY平面）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1595"/>
        <location filename="../ui_templates/MainWindow.ui" line="1598"/>
        <source>Compute &quot;2D1/2&quot; mesh by projecting points on the XY plane</source>
        <translation>基于XY平面投影点计算&quot;2D1/2&quot; 网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1603"/>
        <source>Delaunay 2.5D (best fitting plane)</source>
        <translation>德劳内2.5D（最佳拟合平面）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1606"/>
        <location filename="../ui_templates/MainWindow.ui" line="1609"/>
        <source>Compute &quot;2D1/2&quot; mesh by projecting points on the (least squares) best fitting plane</source>
        <translation>基于最佳拟合平面（最小二乘）投影点计算&quot;2D1/2&quot; 网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1614"/>
        <location filename="../ui_templates/MainWindow.ui" line="1617"/>
        <source>Surface between 2 polylines</source>
        <translation>基于二多段线曲面重建</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1622"/>
        <source>Mesh scan grids</source>
        <translation>网格扫描网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1625"/>
        <location filename="../ui_templates/MainWindow.ui" line="1628"/>
        <source>Mesh scan grids (structured point clouds)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1637"/>
        <location filename="../ui_templates/MainWindow.ui" line="1640"/>
        <source>Convert texture/material to RGB</source>
        <translation>纹理/材质转RGB</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1649"/>
        <source>Sample Points</source>
        <translation>采样点</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1652"/>
        <location filename="../ui_templates/MainWindow.ui" line="1655"/>
        <source>Sample points on a mesh</source>
        <translation>基于网格模型采样点</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1660"/>
        <location filename="../ui_templates/MainWindow.ui" line="1663"/>
        <source>Smooth (Laplacian)</source>
        <translation>平滑(拉普拉斯算子)</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1668"/>
        <location filename="../ui_templates/MainWindow.ui" line="1671"/>
        <source>Subdivide</source>
        <translation>细分网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1676"/>
        <source>Measure surface</source>
        <translation>测量表面</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1679"/>
        <location filename="../ui_templates/MainWindow.ui" line="1682"/>
        <source>Measure mesh surface</source>
        <translation>测量网格表面</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1687"/>
        <location filename="../ui_templates/MainWindow.ui" line="1690"/>
        <source>Measure volume</source>
        <translation>测量体积</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1695"/>
        <source>Flag vertices by type</source>
        <translation>基于类型标记顶点</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1698"/>
        <location filename="../ui_templates/MainWindow.ui" line="1701"/>
        <source>Flag vertices by type: normal (0), border (1), non-manifold (2)</source>
        <translation>基于类型标记顶点：法线（0），外包框（1），非重叠（2）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1706"/>
        <source>Smooth</source>
        <translation>平滑</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1709"/>
        <location filename="../ui_templates/MainWindow.ui" line="1712"/>
        <source>Smooth mesh scalar field</source>
        <translation>平滑网格标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1717"/>
        <source>Enhance</source>
        <translation>增强</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1720"/>
        <location filename="../ui_templates/MainWindow.ui" line="1723"/>
        <source>Enhance Scalar Field</source>
        <translation>提高标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1728"/>
        <location filename="../ui_templates/MainWindow.ui" line="1731"/>
        <source>Sample points</source>
        <translation>采样点</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1736"/>
        <source>Create</source>
        <translation>创建</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1739"/>
        <location filename="../ui_templates/MainWindow.ui" line="1742"/>
        <source>Create a plane</source>
        <translation>创建平面</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1750"/>
        <location filename="../ui_templates/MainWindow.ui" line="1753"/>
        <location filename="../ui_templates/MainWindow.ui" line="2066"/>
        <location filename="../ui_templates/MainWindow.ui" line="2069"/>
        <source>Fit a plane on a set of point</source>
        <translation>基于点集拟合平面</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1761"/>
        <location filename="../ui_templates/MainWindow.ui" line="1764"/>
        <source>Edit the plane parameters</source>
        <translation>编辑平面参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1769"/>
        <location filename="../ui_templates/MainWindow.ui" line="1772"/>
        <source>Compute Kd-tree</source>
        <translation>计算KD树</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1777"/>
        <source>Bounding box P.C.A. fit</source>
        <translation>基于PCA拟合外包围框</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1780"/>
        <location filename="../ui_templates/MainWindow.ui" line="1783"/>
        <source>Makes BB fit principal components (rotates entity!)</source>
        <translation>基于主成分方向调整外包围框（旋转实体！）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1788"/>
        <location filename="../ui_templates/MainWindow.ui" line="1791"/>
        <source>Distance map to best-fit 3D quadric</source>
        <translation>最佳拟合3D二次曲面距离地图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1796"/>
        <location filename="../ui_templates/MainWindow.ui" line="1799"/>
        <source>Distance map</source>
        <translation>距离地图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1804"/>
        <source>Auto align clouds</source>
        <translation>自动配准点云</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1807"/>
        <location filename="../ui_templates/MainWindow.ui" line="1810"/>
        <source>Tries to automatically register (roughly) two points clouds</source>
        <translation>尝试自动配准（粗略）两点云</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1815"/>
        <source>SNE test</source>
        <translation>SNE test</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1818"/>
        <location filename="../ui_templates/MainWindow.ui" line="1821"/>
        <source>Spherical Neighbourhood Extraction test</source>
        <translation>球型邻域提取测试</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1826"/>
        <source>CNE test</source>
        <translation>CNE测试</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1829"/>
        <location filename="../ui_templates/MainWindow.ui" line="1832"/>
        <location filename="../ui_templates/MainWindow.ui" line="1835"/>
        <source>Cylindrical Neighbourhood Extraction test</source>
        <translation>柱形邻域提取测试</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1840"/>
        <location filename="../ui_templates/MainWindow.ui" line="1843"/>
        <source>Find biggest inner rectangle (2D)</source>
        <translation>寻找最大内部矩形框（2D）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1848"/>
        <location filename="../ui_templates/MainWindow.ui" line="1851"/>
        <source>Create cloud from selected entities centers</source>
        <translation>从所选实体中心创建点云</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1856"/>
        <source>Compute best registration RMS matrix</source>
        <translation>计算最佳配准均方根矩阵</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1859"/>
        <location filename="../ui_templates/MainWindow.ui" line="1862"/>
        <source>Computes the best registration between all couples among multiple entities and save the resulting RMS in a matrix (CSV) file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1867"/>
        <source>Enable Visual Debug Traces</source>
        <translation>开启可视化跟踪调试</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1870"/>
        <location filename="../ui_templates/MainWindow.ui" line="1873"/>
        <source>Enables visual debug traces (active 3D view)</source>
        <translation>开启可视化跟踪调试（激活3D视图）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1882"/>
        <location filename="../ui_templates/MainWindow.ui" line="1885"/>
        <source>Show histogram</source>
        <translation>显示柱状图</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1894"/>
        <source>Compute stat. params</source>
        <translation>计算饱和度.参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1897"/>
        <location filename="../ui_templates/MainWindow.ui" line="1900"/>
        <source>Fits a statistical model on the active scalar field</source>
        <translation>基于当前标量字段拟合统计模型</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1909"/>
        <source>Convert to RGB</source>
        <translation>转到RGB</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1912"/>
        <location filename="../ui_templates/MainWindow.ui" line="1915"/>
        <source>Convert current scalar field to RGB colors</source>
        <translation>转换当前变量域值到RGB颜色值</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1920"/>
        <location filename="../ui_templates/MainWindow.ui" line="1923"/>
        <source>Convert to random RGB</source>
        <translation>转成随机RGB颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1928"/>
        <location filename="../ui_templates/MainWindow.ui" line="1931"/>
        <source>RenameSF</source>
        <translation>修改标量名称</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1936"/>
        <location filename="../ui_templates/MainWindow.ui" line="1939"/>
        <source>Edit global shift and scale</source>
        <translation>编辑全局偏移和比例值</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1948"/>
        <source>Subsample</source>
        <translation>下采样</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1951"/>
        <location filename="../ui_templates/MainWindow.ui" line="1954"/>
        <source>Subsample a point cloud</source>
        <translation>下采样点云</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1963"/>
        <source>Gradient</source>
        <translation>梯度</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1966"/>
        <location filename="../ui_templates/MainWindow.ui" line="1969"/>
        <source>SFGradient</source>
        <translation>标量梯度</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1974"/>
        <location filename="../ui_templates/MainWindow.ui" line="1977"/>
        <source>Compute 2.5D volume</source>
        <translation>计算2.5 d体积</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1986"/>
        <source>Cloud/Cloud Dist.</source>
        <translation>点云/点云距离.</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="1989"/>
        <location filename="../ui_templates/MainWindow.ui" line="1992"/>
        <source>Compute cloud/cloud distance</source>
        <translation>计算点云间距离</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2001"/>
        <source>Cloud/Mesh Dist</source>
        <translation>点云/网格距离</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2004"/>
        <location filename="../ui_templates/MainWindow.ui" line="2007"/>
        <source>Compute cloud/mesh distance</source>
        <translation>计算网格模型与点云间距离</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2012"/>
        <source>Closest Point Set</source>
        <translation>最近点集</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2015"/>
        <location filename="../ui_templates/MainWindow.ui" line="2018"/>
        <source>Compute closest point set</source>
        <translation>计算最近点集</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2027"/>
        <source>Label Connected Comp.</source>
        <translation>标注联通区域.</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2030"/>
        <location filename="../ui_templates/MainWindow.ui" line="2033"/>
        <source>Label connected components</source>
        <translation>标注联通区域</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2038"/>
        <source>K-Means</source>
        <translation>K均值</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2041"/>
        <location filename="../ui_templates/MainWindow.ui" line="2044"/>
        <source>classify point (K-Means applied on a scalar field)</source>
        <translation>分类点（基于标量字段应用K均值）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2052"/>
        <source>Front propagation</source>
        <translation>前向传播</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2055"/>
        <location filename="../ui_templates/MainWindow.ui" line="2058"/>
        <source>Classify points by propagating a front on a scalar field</source>
        <translation>分类点-在标量字段上前向传播</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2074"/>
        <source>Sphere</source>
        <translation>球体</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2077"/>
        <location filename="../ui_templates/MainWindow.ui" line="2080"/>
        <source>Fits a sphere on the selected cloud</source>
        <translation>基于所选点云拟合球体</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2085"/>
        <location filename="../ui_templates/MainWindow.ui" line="2088"/>
        <source>2D polygon (facet)</source>
        <translation>2D多边形(面片)</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2093"/>
        <location filename="../ui_templates/MainWindow.ui" line="2096"/>
        <source>2.5D quadric</source>
        <translation>2.5D二次曲面</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2101"/>
        <source>Export cloud info</source>
        <translation>导出点云信息</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2104"/>
        <location filename="../ui_templates/MainWindow.ui" line="2107"/>
        <source>Export cloud info to a CSV file (name, size, barycenter, scalar fields info, etc.)</source>
        <translation>导出点云信息到CSV文件（名称、大小、重心、标量信息等）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2112"/>
        <source>Export plane info</source>
        <translation>导出平面信息</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2115"/>
        <location filename="../ui_templates/MainWindow.ui" line="2118"/>
        <source>Export plane info to a CSV file (name, width, height, center, normal, dip and dip direction, etc.)</source>
        <translation>导出面信息到CSV文件（名称、宽度、高度、中心、法线、倾斜角度等）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2123"/>
        <source>Compute geometric features</source>
        <translation>计算几何特征</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2126"/>
        <location filename="../ui_templates/MainWindow.ui" line="2129"/>
        <source>Compute geometric features (density, curvature, roughness, etc.)</source>
        <translation>计算几何特征（密度，曲率，粗糙度等）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2134"/>
        <location filename="../ui_templates/MainWindow.ui" line="2137"/>
        <source>Remove duplicate points</source>
        <translation>移除重复点</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2149"/>
        <source>Delete Scalar Field</source>
        <translation>删除标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2152"/>
        <location filename="../ui_templates/MainWindow.ui" line="2155"/>
        <source>Delete current scalar field</source>
        <translation>删除当前变量域</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2160"/>
        <source>Delete all (!)</source>
        <translation>删除所有(!)</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2163"/>
        <location filename="../ui_templates/MainWindow.ui" line="2166"/>
        <source>Delete all scalar fields</source>
        <translation>删除所有标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2175"/>
        <location filename="../ui_templates/MainWindow.ui" line="2178"/>
        <source>Add constant SF</source>
        <translation>添加定值标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2183"/>
        <source>Add point indexes as SF</source>
        <translation>添加点索引作为标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2186"/>
        <location filename="../ui_templates/MainWindow.ui" line="2189"/>
        <source>Adds a scalar field with ordered integers for each point in the cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2194"/>
        <source>Export coordinate(s) to SF(s)</source>
        <translation>坐标值转到标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2197"/>
        <location filename="../ui_templates/MainWindow.ui" line="2200"/>
        <source>Export X, Y and/or Z coordinates to scalar field(s)</source>
        <translation>导出XYZ坐标值到标量字段(s)</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2205"/>
        <source>Set SF as coordinate(s)</source>
        <translation>设置标量字段为坐标(s)</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2208"/>
        <location filename="../ui_templates/MainWindow.ui" line="2211"/>
        <source>Set SF as coordinate(s) (X, Y or Z)</source>
        <translation>设置标量字段为坐标(s) （X,Y或者Z）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2219"/>
        <location filename="../ui_templates/MainWindow.ui" line="2222"/>
        <source>Interpolate scalar-field(s) from another cloud or mesh</source>
        <translation>从另一点云或者网格模型插值标量值</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2231"/>
        <source>Arithmetic</source>
        <translation>算术</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2234"/>
        <source>SF arithmetic</source>
        <translation>标量字段算术</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2237"/>
        <location filename="../ui_templates/MainWindow.ui" line="2240"/>
        <source>Add, subtract, multiply or divide two scalar fields</source>
        <translation>两标量字段之间做加减乘除运算</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2249"/>
        <source>Gaussian filter</source>
        <translation>高斯滤波器</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2252"/>
        <location filename="../ui_templates/MainWindow.ui" line="2255"/>
        <source>Compute gaussian filter</source>
        <translation>计算高斯滤波</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2264"/>
        <source>Bilateral filter</source>
        <translation>双边滤波器</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2267"/>
        <location filename="../ui_templates/MainWindow.ui" line="2270"/>
        <source>Compute bilateral filter</source>
        <translation>计算双边滤波</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2279"/>
        <source>Filter By Value</source>
        <translation>基于值滤波</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2282"/>
        <location filename="../ui_templates/MainWindow.ui" line="2285"/>
        <source>Filter points by value</source>
        <translation>基于值滤除点</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2290"/>
        <source>Contour plot (polylines) to mesh</source>
        <translation>基于多段线轮廓生成网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2293"/>
        <location filename="../ui_templates/MainWindow.ui" line="2296"/>
        <source>Contour plot (set of polylines) to a 2.5D mesh</source>
        <translation>基于多段线轮廓生成2.5D网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2301"/>
        <source>Unroll</source>
        <translation>展开实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2304"/>
        <location filename="../ui_templates/MainWindow.ui" line="2307"/>
        <source>Unroll entity on a cylinder or a cone</source>
        <translation>在圆柱或者类圆锥模型上展开实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2312"/>
        <source>Match bounding-box centers</source>
        <translation>匹配外包围框中心</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2315"/>
        <location filename="../ui_templates/MainWindow.ui" line="2318"/>
        <source>Synchronize selected entities bbox centers</source>
        <translation>同步所选实体外包围框中心</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2327"/>
        <location filename="../ui_templates/MainWindow.ui" line="2330"/>
        <source>Match scales</source>
        <translation>匹配尺度</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2339"/>
        <source>Align (point pairs picking)</source>
        <translation>配准（基于拾取对应点）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2342"/>
        <location filename="../ui_templates/MainWindow.ui" line="2345"/>
        <source>Aligns two clouds by picking (at least 4) equivalent point pairs</source>
        <translation>基于拾取至少四个对应点对配准点云</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2354"/>
        <source>Fine registration (ICP)</source>
        <translation>精配准（ICP）</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2357"/>
        <location filename="../ui_templates/MainWindow.ui" line="2360"/>
        <source>Finely registers already (roughly) aligned entities (clouds or meshes)</source>
        <translation>基于粗配准实体（点云或者网格）进行精配准</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2365"/>
        <source>Apply transformation</source>
        <translation>应用转换</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2368"/>
        <location filename="../ui_templates/MainWindow.ui" line="2371"/>
        <source>Apply rotation and/or translation</source>
        <translation>应用旋转、平移或旋转平移</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2380"/>
        <location filename="../ui_templates/MainWindow.ui" line="2383"/>
        <source>Translate/Rotate</source>
        <translation>平移/旋转</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2392"/>
        <source>Multiply/Scale</source>
        <translation>按比例缩放</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2395"/>
        <location filename="../ui_templates/MainWindow.ui" line="2398"/>
        <source>Multiply coordinates (separately)</source>
        <translation>坐标分别相乘</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2403"/>
        <source>Global Shift settings</source>
        <translation>全局偏移值设置</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2406"/>
        <location filename="../ui_templates/MainWindow.ui" line="2409"/>
        <source>Set Global Shift &amp; Scale mechanism parameters</source>
        <translation>设置全局偏移和比例机制参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2414"/>
        <source>Gray</source>
        <translation>灰色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2419"/>
        <source>LightGray</source>
        <translation>浅灰色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2424"/>
        <source>DarkGray</source>
        <translation>深灰色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2429"/>
        <source>FlatWhite</source>
        <translation>纯白色</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2434"/>
        <source>BF</source>
        <translation>BF</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2439"/>
        <source>Test</source>
        <translation>测试</translation>
    </message>
    <message>
        <location filename="../ui_templates/MainWindow.ui" line="2444"/>
        <source>Default</source>
        <translation>默认</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../MainWindow.cpp" line="213"/>
        <source>Ready</source>
        <translation>完成</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="218"/>
        <source>[TBB] Using Intel&apos;s Threading Building Blocks %1.%2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="223"/>
        <source>[ACloudViewer Software start], Welcome to use ACloudViewer</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="670"/>
        <source>[about] Asher, Welcome to corporation ACloudViewer: http://asher-1.github.io !</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="896"/>
        <source>Open file(s)</source>
        <translation>Open file (s)</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="977"/>
        <source>%1 file(s) loaded</source>
        <translation>%1 file (s) loaded</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1013"/>
        <source>Entity &apos;%1&apos; has been translated: (%2,%3,%4) and rescaled of a factor %5 [original position will be restored when saving]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1061"/>
        <source>[MainWindow::addToDB] Internal error: no associated db?!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1082"/>
        <source>clouds</source>
        <translation>云</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1083"/>
        <source>meshes</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1084"/>
        <source>polylines</source>
        <translation>折线</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1085"/>
        <source>other</source>
        <translation>其他</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1086"/>
        <source>serializable</source>
        <translation>可序列化的</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1138"/>
        <source>Can&apos;t save selected entity(ies) this way!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1266"/>
        <source>[I/O] First entity&apos;s name would make an invalid filename! Can&apos;t use it...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1267"/>
        <source>project</source>
        <translation>项目</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1275"/>
        <source>Save file</source>
        <translation>保存文件</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1290"/>
        <source>[I/O] The following selected entities won&apos;t be saved:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1293"/>
        <source>	- %1s</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1322"/>
        <source>[I/O] None of the selected entities can be saved this way...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1437"/>
        <location filename="../MainWindow.cpp" line="6500"/>
        <source>Original</source>
        <translation>原始</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1442"/>
        <source>Suggested</source>
        <translation>建议</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1471"/>
        <source>[ApplyTransformation] Process cancelled by user</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1487"/>
        <source>[ApplyTransformation] Cloud &apos;%1&apos; global shift/scale information has been updated: shift = (%2,%3,%4) / scale = %5</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1507"/>
        <source>[ApplyTransformation] Applied transformation matrix:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1509"/>
        <location filename="../MainWindow.cpp" line="4820"/>
        <location filename="../MainWindow.cpp" line="4939"/>
        <location filename="../MainWindow.cpp" line="5035"/>
        <source>Hint: copy it (CTRL+C) and apply it - or its inverse - on any entity with the &apos;Edit &gt; Apply transformation&apos; tool</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1550"/>
        <source>[Apply scale] Entity &apos;%1&apos; can&apos;t be scaled this way</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1594"/>
        <source>Big coordinates</source>
        <translation>大的坐标</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1595"/>
        <source>Resutling coordinates will be too big (original precision may be lost!). Proceed anyway?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1626"/>
        <source>[Apply scale] No eligible entities (point clouds or meshes) were selected!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1715"/>
        <source>No entity eligible for manual transformation! (see console)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1720"/>
        <source>Some entities were ingored! (see console)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1734"/>
        <location filename="../MainWindow.cpp" line="6612"/>
        <location filename="../MainWindow.cpp" line="6671"/>
        <location filename="../MainWindow.cpp" line="6725"/>
        <source>Unexpected error!</source>
        <translation>意想不到的错误!</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1775"/>
        <source>Close all</source>
        <translation>关闭所有</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="1776"/>
        <source>Are you sure you want to remove all loaded entities?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2023"/>
        <source>TempGroup</source>
        <translation>TempGroup</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2200"/>
        <source>Restart</source>
        <translation>重新启动</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2200"/>
        <source>To finish the process, you&apos;ll have to close and restart ACloudViewer</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2223"/>
        <source>Full Screen 3D mode has not been implemented yet!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2258"/>
        <location filename="../MainWindow.cpp" line="7986"/>
        <source>Select one and only one entity!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2265"/>
        <source>Wrong type of entity</source>
        <translation>实体类型错误</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2271"/>
        <source>Points must be visible!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2519"/>
        <source>Only meshes with standard vertices are handled for now! Can&apos;t merge entity &apos;%1&apos;...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2524"/>
        <source>Entity &apos;%1&apos; is neither a cloud nor a mesh, can&apos;t merge it!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2530"/>
        <location filename="../MainWindow.cpp" line="3670"/>
        <location filename="../MainWindow.cpp" line="3847"/>
        <location filename="../MainWindow.cpp" line="4565"/>
        <location filename="../MainWindow.cpp" line="4588"/>
        <location filename="../MainWindow.cpp" line="4601"/>
        <location filename="../MainWindow.cpp" line="4609"/>
        <location filename="../MainWindow.cpp" line="4700"/>
        <location filename="../MainWindow.cpp" line="4878"/>
        <location filename="../MainWindow.cpp" line="5351"/>
        <location filename="../MainWindow.cpp" line="5379"/>
        <location filename="../MainWindow.cpp" line="5459"/>
        <location filename="../MainWindow.cpp" line="5518"/>
        <location filename="../MainWindow.cpp" line="5570"/>
        <location filename="../MainWindow.cpp" line="5614"/>
        <location filename="../MainWindow.cpp" line="6382"/>
        <source>Not enough memory!</source>
        <translation>没有足够的内存!</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2536"/>
        <source>Select only clouds or meshes!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2541"/>
        <source>Can&apos;t mix point clouds and meshes!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2576"/>
        <source>Original cloud index</source>
        <translation>原来云指数</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2576"/>
        <source>Do you want to generate a scalar field with the original cloud index?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2585"/>
        <source>Couldn&apos;t allocate a new scalar field for storing the original cloud index! Try to free some memory ...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2628"/>
        <location filename="../MainWindow.cpp" line="2681"/>
        <source>Fusion failed! (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2787"/>
        <source>Selected entities have no valid bounding-box!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2842"/>
        <source>Can&apos;t start the picking mechanism (another tool is already using it)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2926"/>
        <source>[Level] Point is too close from the others!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="2936"/>
        <source>P#%1</source>
        <translation>P#%1</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3002"/>
        <source>Point (%1 ; %2 ; %3) set as rotation center for interactive transformation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3077"/>
        <source>Quit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3078"/>
        <source>Are you sure you want to quit?</source>
        <translation>确定退出吗？</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3190"/>
        <source>Selected one and only one point cloud or mesh!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3195"/>
        <source>Compute Kd-tree</source>
        <translation>计算kd tree</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3195"/>
        <source>Max error per leaf cell:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3230"/>
        <source>An error occurred!</source>
        <translation>一个错误发生!</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3248"/>
        <source>Resample with octree</source>
        <translation>重新取样和八叉树</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3248"/>
        <source>Points (approx.)</source>
        <translation>点(约)</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3280"/>
        <source>Could not compute octree for cloud &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3319"/>
        <source>[ResampleWithOctree] Errors occurred during the process! Result may be incomplete!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3341"/>
        <location filename="../MainWindow.cpp" line="3477"/>
        <source>Triangulate</source>
        <translation>由三角形组成的</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3341"/>
        <location filename="../MainWindow.cpp" line="7776"/>
        <source>Max edge length (0 = no limit)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3370"/>
        <source>Keep old normals?</source>
        <translation>保持旧的法线?</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3371"/>
        <source>Cloud(s) already have normals. Do you want to update them (yes) or keep the old ones (no)?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3378"/>
        <source>Triangulation</source>
        <translation>三角测量</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3379"/>
        <source>Triangulation in progress...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3416"/>
        <source>Error(s) occurred! See the Console messages</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3432"/>
        <source>Select 2 and only 2 polylines</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3442"/>
        <source>Projection method</source>
        <translation>投影法</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3443"/>
        <source>Use best fit plane (yes) or the current viewing direction (no)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3461"/>
        <source>[Mesh two polylines] Failed to compute normals!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3466"/>
        <source>Failed to create mesh (see Console)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3477"/>
        <source>Min triangle angle (in degrees)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3584"/>
        <source>[doActionSamplePointsOnMesh] Errors occurred during the process! Result may be incomplete!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3595"/>
        <source>Points Sampling on polyline</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3635"/>
        <source>[doActionSamplePointsOnPolyline] Errors occurred during the process! Result may be incomplete!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3677"/>
        <source>Select a group of polylines or multiple polylines (contour plot)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3682"/>
        <source>Projection dimension</source>
        <translation>投影尺寸</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3683"/>
        <source>Contour plot to mesh</source>
        <translation>等高线网格图</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3716"/>
        <source>Not enough segments!</source>
        <translation>不够的部分!</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3731"/>
        <location filename="../MainWindow.cpp" line="3780"/>
        <location filename="../MainWindow.cpp" line="8098"/>
        <source>Not enough memory</source>
        <translation>没有足够的内存</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3771"/>
        <source>Third party library error: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3776"/>
        <source>vertices</source>
        <translation>顶点</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3836"/>
        <source>[Contour plot to mesh] Failed to compute normals!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3875"/>
        <source>Subdivide mesh</source>
        <translation>细分网格</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3875"/>
        <source>Max area per triangle:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3897"/>
        <source>[Subdivide] An error occurred while trying to subdivide mesh &apos;%1&apos; (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3902"/>
        <source>%1.subdivided(S&lt;%2)</source>
        <translation>%1. Subdivided (S &lt; %2)</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3910"/>
        <source>[Subdivide] Failed to subdivide mesh &apos;%1&apos; (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3915"/>
        <source>[Subdivide] Works only on real meshes!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3929"/>
        <location filename="../MainWindow.cpp" line="3932"/>
        <source>Smooth mesh</source>
        <translation>光滑的网</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3929"/>
        <source>Iterations:</source>
        <translation>迭代:</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3932"/>
        <source>Smoothing factor:</source>
        <translation>平滑系数:</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3955"/>
        <source>Failed to apply Laplacian smoothing to mesh &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="3986"/>
        <location filename="../MainWindow.cpp" line="4015"/>
        <source>Not enough memory to flag the vertices of mesh &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4009"/>
        <source>[Mesh Quality] Mesh &apos;%1&apos; edges: %2 total (normal: %3 / on hole borders: %4 / non-manifold: %5)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4033"/>
        <source>[Mesh Quality] SF flags: %1 (NORMAL) / %2 (BORDER) / (%3) NON-MANIFOLD</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4038"/>
        <source>Error(s) occurred! Check the console...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4055"/>
        <source>[Mesh Volume] Mesh &apos;%1&apos;: V=%2 (cube units)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4063"/>
        <source>[Mesh Volume] The above volume might be invalid (mesh has holes)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4067"/>
        <source>[Mesh Volume] The above volume might be invalid (mesh has non-manifold edges)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4072"/>
        <source>[Mesh Volume] The above volume might be invalid (not enough memory to check if the mesh is closed)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4095"/>
        <source>[Mesh Surface] Mesh &apos;%1&apos;: S=%2 (square units)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4098"/>
        <source>[Mesh Surface] Average triangle surface: %1 (square units)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4139"/>
        <source>[ACloudViewer help] http://asher-1.github.io !</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4162"/>
        <source>[changeLanguage] Change to English language</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4166"/>
        <source>[changeLanguage] Doesn&apos;t support Chinese temporarily</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4192"/>
        <source>[Global Shift] Max abs. coord = %1 / max abs. diag = %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4210"/>
        <source>No cloud in database!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4335"/>
        <source>An error occurred while cloning cloud %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4343"/>
        <source>An error occurred while cloning primitive %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4351"/>
        <source>An error occurred while cloning mesh %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4360"/>
        <source>An error occurred while cloning polyline %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4369"/>
        <source>An error occurred while cloning facet %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4374"/>
        <source>Entity &apos;%1&apos; can&apos;t be cloned (type not supported yet!)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4402"/>
        <source>Clear console</source>
        <translation>清空控制台</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4403"/>
        <source>Enable console</source>
        <translation>启用控制台</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4404"/>
        <source>Disable console</source>
        <translation>禁用控制台</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4449"/>
        <source>This method is for test purpose only</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4450"/>
        <source>Cloud(s) are going to be rotated while still displayed in their previous position! Proceed?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4596"/>
        <source>.distance_grid(%1)</source>
        <translation>. distance_grid (%1)</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4637"/>
        <source>[DistanceMap] Cloud &apos;%1&apos;: no point falls inside the specified range</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4659"/>
        <source>Distance to best fit quadric (3D)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4659"/>
        <source>Steps (per dim.)</source>
        <translation>步骤(每暗淡。)</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4688"/>
        <source>Failed to get the center of gravity of cloud &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4709"/>
        <location filename="../MainWindow.cpp" line="7643"/>
        <source>Couldn&apos;t allocate a new scalar field for computing distances! Try to free some memory ...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4749"/>
        <source>Distance map to 3D quadric</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4755"/>
        <source>Failed to compute 3D quadric on cloud &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4766"/>
        <source>Work in progress</source>
        <translation>进行中的工作</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4767"/>
        <source>This method is still under development: are you sure you want to use it? (a crash may likely happen)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4773"/>
        <location filename="../MainWindow.cpp" line="4780"/>
        <location filename="../MainWindow.cpp" line="7265"/>
        <location filename="../MainWindow.cpp" line="7610"/>
        <location filename="../MainWindow.cpp" line="7617"/>
        <source>Select 2 point clouds!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4818"/>
        <source>[Align] Resulting matrix:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4827"/>
        <location filename="../MainWindow.cpp" line="5146"/>
        <source>.registered</source>
        <translation>.registered</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4840"/>
        <source>[Align] Registration failed!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4937"/>
        <source>[Synchronize] Transformation matrix (%1 --&gt; %2):</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="4965"/>
        <source>Select 2 point clouds or meshes!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5020"/>
        <source>Final RMS: %1 (computed on %2 points)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5021"/>
        <location filename="../MainWindow.cpp" line="5041"/>
        <source>[Register] </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5029"/>
        <source>Transformation matrix</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5033"/>
        <source>[Register] Applied transformation matrix:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5040"/>
        <source>Scale: %1 (already integrated in above matrix!)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5046"/>
        <source>[Register] Scale: fixed (1.0)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5047"/>
        <source>Scale: fixed (1.0)</source>
        <translation>比例:固定(1.0)</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5052"/>
        <source>Theoretical overlap: %1%</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5053"/>
        <source>[Register] %1</source>
        <translation>[Register] %1</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5057"/>
        <source>This report has been output to Console (F8)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5077"/>
        <source>Registration</source>
        <translation>登记</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5078"/>
        <source>Data mesh vertices are locked (they may be shared with other meshes): Do you wish to clone this mesh to apply transformation?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5091"/>
        <source>Doesn&apos;t work on sub-meshes yet!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5103"/>
        <source>Failed to clone &apos;data&apos; mesh! (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5132"/>
        <source>[ICP] Aligned entity global shift has been updated to match the reference: (%1,%2,%3) [x%4]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5136"/>
        <source>Drop shift information?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5136"/>
        <source>Aligned entity is shifted but reference cloud is not: drop global shift information?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5140"/>
        <source>[ICP] Aligned entity global shift has been reset to match the reference!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5153"/>
        <source>Register info</source>
        <translation>注册信息</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5164"/>
        <source>Select one or two entities (point cloud or mesh)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5175"/>
        <source>Select point clouds or meshes only!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5182"/>
        <source>Aligned</source>
        <translation>对齐</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5183"/>
        <location filename="../MainWindow.cpp" line="7270"/>
        <location filename="../MainWindow.cpp" line="7335"/>
        <location filename="../MainWindow.cpp" line="7622"/>
        <source>Reference</source>
        <translation>参考</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5202"/>
        <source>[PointPairRegistration] Failed to create dedicated 3D view!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5254"/>
        <source>Invalid kernel size!</source>
        <translation>无效的内核大小!</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5259"/>
        <source>SNE test</source>
        <translation>SNE test</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5259"/>
        <source>Radius:</source>
        <translation>半径:</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5264"/>
        <source>Spherical extraction test (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5283"/>
        <source>Failed to create scalar field on cloud &apos;%1&apos; (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5295"/>
        <source>Couldn&apos;t compute octree for cloud &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5324"/>
        <source>[SNE_TEST] Mean extraction time = %1 ms (radius = %2, mean(neighbours) = %3)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5339"/>
        <location filename="../MainWindow.cpp" line="5343"/>
        <source>CNE Test</source>
        <translation>CNE测试</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5339"/>
        <source>radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5343"/>
        <source>height</source>
        <translation>高度</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5347"/>
        <source>cube</source>
        <translation>多维数据集</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5432"/>
        <source>[CNE_TEST] Mean extraction time = %1 ms (radius = %2, height = %3, mean(neighbours) = %4)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5437"/>
        <source>Failed to compute octree!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5456"/>
        <source>centers</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5488"/>
        <source>No cloud in selection?!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5525"/>
        <source>Need at least two clouds!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5577"/>
        <source>Testing all possible positions</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5578"/>
        <source>%1 clouds and %2 positions</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5639"/>
        <source>An error occurred while performing ICP!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5675"/>
        <source>Best case #%1 / #%2 - RMS = %3</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5679"/>
        <source>[doActionComputeBestICPRmsMatrix] Comparison #%1 / #%2: min RMS = %3 (phi = %4 / theta = %5 deg.)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5684"/>
        <source>[doActionComputeBestICPRmsMatrix] Comparison #%1 / #%2: INVALID</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5707"/>
        <location filename="../MainWindow.cpp" line="7411"/>
        <location filename="../MainWindow.cpp" line="7513"/>
        <source>Select output file</source>
        <translation>选择输出文件</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5744"/>
        <source>[doActionComputeBestICPRmsMatrix] Job done</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5748"/>
        <source>Failed to save output file?!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5762"/>
        <source>Select one point cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5767"/>
        <source>Dimension</source>
        <translation>维</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5767"/>
        <source>Orthogonal dim (X=0 / Y=1 / Z=2)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5800"/>
        <source>Histogram [%1]</source>
        <translation>Histogram [%1]</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5811"/>
        <source>%1 (%2 values) </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5814"/>
        <source>Count</source>
        <translation>数</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="5886"/>
        <source>Entity [%1] has no active scalar field !</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6006"/>
        <location filename="../MainWindow.cpp" line="6280"/>
        <source>Previously selected entities (sources) have been hidden!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6057"/>
        <source>Select only one cloud or one mesh!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6075"/>
        <source>Constant</source>
        <translation>常数</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6079"/>
        <source>Constant #%1</source>
        <translation>常数 #%1</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6084"/>
        <source>New SF name</source>
        <translation>新的科幻小说名</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6084"/>
        <source>SF name (must be unique)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6089"/>
        <source>Invalid name</source>
        <translation>无效的名称</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6094"/>
        <source>Name already exists!</source>
        <translation>名称已经存在!</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6098"/>
        <source>Add constant value</source>
        <translation>添加恒定值</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6098"/>
        <source>value</source>
        <translation>价值</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6107"/>
        <source>An error occurred! (see console)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6123"/>
        <source>New scalar field added to %1 (constant value: %2)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6186"/>
        <source>Remove duplicate points</source>
        <translation>移除重复点</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6186"/>
        <source>Min distance between points:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6214"/>
        <source>Couldn&apos;t create temporary scalar field! Not enough memory?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6244"/>
        <source>Cloud &apos;%1&apos; has no duplicate points</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6248"/>
        <source>Cloud &apos;%1&apos; has %2 duplicate point(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6256"/>
        <source>%1.clean</source>
        <translation>%1 清除</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6272"/>
        <source>An error occurred! (Not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6318"/>
        <location filename="../MainWindow.cpp" line="7504"/>
        <source>Select at least one point cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6336"/>
        <source>Subsampling</source>
        <translation>二次抽样</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6349"/>
        <source>[Subsampling] Failed to subsample cloud &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6376"/>
        <source>[Subsampling] Not enough memory: colors, normals or scalar fields may be missing!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6387"/>
        <source>[Subsampling] Timing: %1 s.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6391"/>
        <source>Errors occurred (see console)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6518"/>
        <source>[Global Shift/Scale] New shift: (%1, %2, %3)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6519"/>
        <source>[Global Shift/Scale] New scale: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6549"/>
        <source>[Global Shift/Scale] To preserve its original position, the entity &apos;%1&apos; has been translated of (%2,%3,%4) and rescaled of a factor %5</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6757"/>
        <source>No segmentable entity in active window!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="6822"/>
        <source>[Segmentation] Label %1 depends on cloud %2 and will be removed</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7012"/>
        <source>[CreateComponentsClouds] Not enough memory to sort components by size!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7075"/>
        <source>[createComponentsClouds] Failed to create component #%1! (not enough memory)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7094"/>
        <source>[createComponentsClouds] %1 component(s) were created from cloud &apos;%2&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7101"/>
        <source>[createComponentsClouds] Original cloud has been automatically hidden</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7155"/>
        <source>Couldn&apos;t compute octree for cloud &apos;%s&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7168"/>
        <source>Couldn&apos;t allocate a new scalar field for computing ECV labels! Try to free some memory ...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7200"/>
        <source>Many components</source>
        <translation>许多组件</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7201"/>
        <source>Do you really expect up to %1 components?
(this may take a lot of time to process and display)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7221"/>
        <location filename="../MainWindow.cpp" line="7226"/>
        <source>[doActionLabelConnectedComponents] Something went wrong while extracting CCs from cloud %1...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7246"/>
        <location filename="../MainWindow.cpp" line="7251"/>
        <source>Not yet implemented! Sorry ...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7269"/>
        <location filename="../MainWindow.cpp" line="7334"/>
        <location filename="../MainWindow.cpp" line="7621"/>
        <source>Compared</source>
        <translation>相比</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7293"/>
        <source>Select 2 entities!</source>
        <translation>选择2实体!</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7315"/>
        <source>Select at least one mesh!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7320"/>
        <source>Select one mesh and one cloud or two meshes!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7402"/>
        <source>No plane in selection</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7426"/>
        <location filename="../MainWindow.cpp" line="7527"/>
        <source>Failed to open file for writing! (check file permissions)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7475"/>
        <source>[I/O] File &apos;%1&apos; successfully saved (%2 plane(s))</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7602"/>
        <source>[I/O] File &apos;%1&apos; successfully saved (%2 cloud(s))</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7632"/>
        <source>Compared cloud must be a real point cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7678"/>
        <source>Select one or two point clouds!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7687"/>
        <location filename="../MainWindow.cpp" line="7702"/>
        <source>Select point clouds only!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7740"/>
        <source>[Fit sphere] Failed to fit a sphere on cloud &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7744"/>
        <source>[Fit sphere] Cloud &apos;%1&apos;: center (%2,%3,%4) - radius = %5 [RMS = %6]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7754"/>
        <source>Sphere r=%1 [rms %2]</source>
        <translation>球体r=%1 [rms %2]</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7776"/>
        <source>Fit facet</source>
        <translation>健康方面</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7856"/>
        <source>[Orientation] Entity &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7857"/>
        <source>	- plane fitting RMS: %f</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7862"/>
        <source>	- normal: (%1,%2,%3)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7876"/>
        <source>[Orientation] A matrix that would make this plane horizontal (normal towards Z+) is:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7878"/>
        <source>[Orientation] You can copy this matrix values (CTRL+C) and paste them in the &apos;Apply transformation tool&apos; dialog</source>
        <translation>[Orientation] 你可以复制当前矩阵值 (CTRL+C) 并粘贴到 &apos;应用转换工具&apos; 对话框</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7898"/>
        <source>Failed to fit a plane/facet on entity &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7928"/>
        <source>Quadric (%1)</source>
        <translation>Quadric (%1)</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7932"/>
        <source>[doActionFitQuadric] Quadric local coordinate system:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7934"/>
        <source>[doActionFitQuadric] Quadric equation (in local coordinate system): </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7935"/>
        <source>[doActionFitQuadric] RMS: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7969"/>
        <source>Failed to compute quadric on cloud &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="7977"/>
        <source>Error(s) occurred: see console</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="8003"/>
        <source>Method can&apos;t be applied on locked vertices or virtual point clouds!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="8039"/>
        <source>Error</source>
        <translation>错误</translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="8039"/>
        <source>Invalid angular range</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../MainWindow.cpp" line="8071"/>
        <source>[Unroll] Original cloud has been automatically hidden</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>MatchScalesDialog</name>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="14"/>
        <source>Match scales</source>
        <translation>匹配尺度</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="20"/>
        <source>Choose the reference entity (its scale won&apos;t change)</source>
        <translation>选择参考实体（其缩放尺度不会改变）</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="23"/>
        <source>Reference entity</source>
        <translation>参考实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="35"/>
        <source>Matching criterion</source>
        <translation>匹配准则</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="41"/>
        <source>The scaling ratio will be deduced from the largest bounding-box dimension</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="44"/>
        <source>max bounding-box dimension</source>
        <translation>最大外包围框尺寸</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="51"/>
        <source>The scaling ratio will be deduced from the bounding-box volume</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="54"/>
        <source>bounding-box volume</source>
        <translation>外包围框体积</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="61"/>
        <source>The scaling ratio will be deduced from the principal cloud dimension (by PCA analysis)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="64"/>
        <source>principal dimension (PCA)</source>
        <translation>主维度（PCA）</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="74"/>
        <source>The scaling ratio will be deduced from automatic registration (with unconstrained scale).
Should be used after one of the previous methods!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="78"/>
        <source>ICP (only the scale will be applied)</source>
        <translation>ICP（仅缩放比例会为应用）</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="91"/>
        <source>Parameters for ICP registration</source>
        <translation>ICP配准参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="94"/>
        <source>ICP parameters</source>
        <translation>ICP参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="100"/>
        <source>RMS difference</source>
        <translation>RMS的区别</translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="113"/>
        <source>Set the minimum RMS improvement between 2 consecutive iterations (below which the registration process will stop).</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="129"/>
        <location filename="../ui_templates/matchScalesDlg.ui" line="154"/>
        <source>Rough estimation of the final overlap ratio of the data cloud (the smaller, the better the initial registration should be!)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/matchScalesDlg.ui" line="157"/>
        <source>Final overlap</source>
        <translation>最终重叠率</translation>
    </message>
</context>
<context>
    <name>MatrixDisplayDlg</name>
    <message>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="23"/>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="45"/>
        <source>Matrix</source>
        <translation>矩阵</translation>
    </message>
    <message>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="85"/>
        <source>Axis/Angle</source>
        <translation>轴/角</translation>
    </message>
    <message>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="100"/>
        <source>Axis</source>
        <translation>轴</translation>
    </message>
    <message>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="120"/>
        <source>Angle</source>
        <translation>角度</translation>
    </message>
    <message>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="140"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="161"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../ui_templates/matrixDisplayDlg.ui" line="177"/>
        <source>Clipboard</source>
        <translation>剪贴板</translation>
    </message>
</context>
<context>
    <name>NoiseFilterDialog</name>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="14"/>
        <source>Filter noise</source>
        <translation>过滤噪音</translation>
    </message>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="20"/>
        <source>Neighbors</source>
        <translation>近邻</translation>
    </message>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="26"/>
        <source>Points (kNN)</source>
        <translation>点 (kNN)</translation>
    </message>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="33"/>
        <source>Radius (Sphere)</source>
        <translation>半径(球)</translation>
    </message>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="72"/>
        <source>Max error</source>
        <translation>最大误差</translation>
    </message>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="78"/>
        <source>Relative</source>
        <translation>相对的</translation>
    </message>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="88"/>
        <source>Absolute</source>
        <translation>绝对的</translation>
    </message>
    <message>
        <location filename="../ui_templates/noiseFilterDlg.ui" line="121"/>
        <source>Remove isolated points</source>
        <translation>移除孤立点</translation>
    </message>
</context>
<context>
    <name>NormalComputationDlg</name>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="14"/>
        <source>Compute normals</source>
        <translation>计算法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="20"/>
        <source>Surface approximation</source>
        <translation>曲面估计</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="29"/>
        <source>Local surface model</source>
        <translation>局部曲面模型</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="36"/>
        <source>Local surface estimation model</source>
        <translation>局部曲面评估模型</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="43"/>
        <source>Plane</source>
        <translation>平面</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="48"/>
        <source>Quadric</source>
        <translation>二次曲面</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="53"/>
        <source>Triangulation</source>
        <translation>三角化</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="70"/>
        <source>Neighbors</source>
        <translation>近邻</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="81"/>
        <source>Using scan grid(s) instead of the octree</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="84"/>
        <source>use scan grid(s) whenever possible</source>
        <translation>尽可能使用扫描网格（非八叉树）</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="125"/>
        <source>min triangulation angle</source>
        <translation>最小三角形角度</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="128"/>
        <source>min angle</source>
        <translation>最小角</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="138"/>
        <source>Min angle of local triangles (in degrees)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="188"/>
        <source>Octree</source>
        <translation>八叉树</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="223"/>
        <location filename="../ui_templates/normalComputationDlg.ui" line="233"/>
        <source>Radius of the sphere in which the neighbors will be extracted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="226"/>
        <source>radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="255"/>
        <source>Auto</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="276"/>
        <source>Orientation</source>
        <translation>方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="288"/>
        <source>Use scan grid(s) (robust method)</source>
        <translation>使用扫描网格（鲁棒方法）</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="291"/>
        <source>Use scan grid(s)  whenever possible</source>
        <translation>尽可能使用扫描网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="301"/>
        <source>Use sensor position to orient normals (if both grid and sensor are selected, &apos;grid&apos; has precedence over &apos;sensor&apos;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="304"/>
        <source>Use sensor(s) whenever possible</source>
        <translation>尽可能使用传感器</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="313"/>
        <source>To give a hint on how to orient normals</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="316"/>
        <source>Use preferred orientation</source>
        <translation>使用首选方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="376"/>
        <source>+ Barycenter</source>
        <translation>+ Barycenter</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="381"/>
        <source>- Barycenter</source>
        <translation>——重心</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="396"/>
        <source>Use previous normal</source>
        <translation>使用前一法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="408"/>
        <source>Generic 3D orientation algorithm</source>
        <translation>通用3D方向算法</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="411"/>
        <source>Use Minimum Spanning Tree</source>
        <translation>使用最小生成树</translation>
    </message>
    <message>
        <location filename="../ui_templates/normalComputationDlg.ui" line="437"/>
        <source>Number of neighbors used to build the tree</source>
        <translation>用于创建树的近邻数量</translation>
    </message>
</context>
<context>
    <name>NormalEstimation</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/NormalEstimation.cpp" line="39"/>
        <source>Estimate Normals</source>
        <translation>估计法线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/NormalEstimation.cpp" line="40"/>
        <source>Estimate Normals and Curvature</source>
        <translation>估计法线和曲率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/NormalEstimation.cpp" line="41"/>
        <source>Estimate Normals and Curvature for the selected entity</source>
        <translation>为所选实体估计法线和曲率</translation>
    </message>
</context>
<context>
    <name>NormalEstimationDialog</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/NormalEstimationDlg.ui" line="20"/>
        <source>Normal Estimation</source>
        <translation>法线估计</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/NormalEstimationDlg.ui" line="32"/>
        <source>Search Radius</source>
        <translation>搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/NormalEstimationDlg.ui" line="62"/>
        <source>Use Knn Search</source>
        <translation>使用资讯搜索</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/NormalEstimationDlg.ui" line="81"/>
        <source>Overwrite Curvature</source>
        <translation>覆盖曲率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/NormalEstimationDlg.ui" line="93"/>
        <source>Search Surface</source>
        <translation>搜索表面</translation>
    </message>
</context>
<context>
    <name>OpenLASFileDialog</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="14"/>
        <source>Open LAS File</source>
        <translation>打开LAS文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="24"/>
        <source>Standard fields</source>
        <translation>标准字段</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="93"/>
        <source>Classification</source>
        <translation>分类</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="103"/>
        <source>decompose</source>
        <translation>分解</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="115"/>
        <source>Value</source>
        <translation>值</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="125"/>
        <source>Synthetic flag</source>
        <translation>合成标志</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="135"/>
        <source>Key-point</source>
        <translation>关键点</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="145"/>
        <source>Withheld</source>
        <translation>保留</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="158"/>
        <source>Time</source>
        <translation>时间</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="168"/>
        <source>Point source ID</source>
        <translation>点源ID</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="178"/>
        <source>Number of returns</source>
        <translation>返回量</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="188"/>
        <source>Return number</source>
        <translation>返回数</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="198"/>
        <source>Scan direction flag</source>
        <translation>扫描方向标志</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="208"/>
        <source>Edge of flight line</source>
        <translation>航线边缘</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="218"/>
        <source>Scan angle rank</source>
        <translation>扫描角排序</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="228"/>
        <source>User data</source>
        <translation>用户数据</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="240"/>
        <source>Intensity</source>
        <translation>强度</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="251"/>
        <source>Extended fields</source>
        <translation>扩展字段</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="257"/>
        <source>Load additional field(s)</source>
        <translation>加载额外字段</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="280"/>
        <source>Tiling</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="286"/>
        <source>Tile input file</source>
        <translation>tile输入文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="300"/>
        <source>Dimension</source>
        <translation>维度</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="332"/>
        <source>Tiles</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="366"/>
        <source>x</source>
        <translation>x</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="409"/>
        <source>Output path</source>
        <translation>输出路径</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="421"/>
        <source>...</source>
        <translation>...</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="433"/>
        <source>Warning: the cloud won&apos;t be loaded in memory.
It will be saved as multiple tiles on the disk.</source>
        <translation>警告：点云不会被读入内存
在磁盘上将会被保存为多tiles。</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="461"/>
        <source>Info</source>
        <translation>信息</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="467"/>
        <source>Points</source>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="481"/>
        <source>Bounding-box</source>
        <translation>外包围框</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="499"/>
        <source>Ignore fields with default values only</source>
        <translation>忽略仅有默认值字段</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="509"/>
        <source>Force reading colors as 8-bit values (even if the standard is 16-bit)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="512"/>
        <source>Force 8-bit colors</source>
        <translation>强制8位颜色</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="566"/>
        <source>Apply</source>
        <translation>应用</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="573"/>
        <source>Apply all</source>
        <translation>应用所有</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openLASFileDlg.ui" line="580"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>PCVDialog</name>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="14"/>
        <source>ShadeVis</source>
        <translation>ShadeVis</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="30"/>
        <source>Light rays</source>
        <translation>光射线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="36"/>
        <source>Samples rays on a sphere</source>
        <translation>在球体上采样射线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="49"/>
        <source>Count</source>
        <translation>数量</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="56"/>
        <source>number of rays to cast</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="88"/>
        <source>rays are cast from the whole sphere (instrad of the Z+ hemisphere)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="91"/>
        <source>Only northern hemisphere (+Z)</source>
        <translation>仅北半球（+Z）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="104"/>
        <source>Use cloud normals as light rays</source>
        <translation>使用点云法线作为光射线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="117"/>
        <source>cloud</source>
        <translation>点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="142"/>
        <source>Render context resolution</source>
        <translation>渲染环境分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="149"/>
        <source>rendering buffer resolution</source>
        <translation>渲染缓冲分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="185"/>
        <source>Accelerates computation if the mesh is closed (no holes)</source>
        <translation>如果网格模型是封闭的（没有孔洞）则进行加速计算</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/pcvDlg.ui" line="188"/>
        <source>closed mesh</source>
        <translation>封闭网格</translation>
    </message>
</context>
<context>
    <name>PclGridProjection</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="40"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="41"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="167"/>
        <source>Grid Projection</source>
        <translation>网格投影</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="42"/>
        <source>Grid Projection from clouds</source>
        <translation>基于点云进行网格投影</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="135"/>
        <source>[PclGridProjection::compute] generate new normals</source>
        <translation>[PclGridProjection::compute] 计算新的法线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="141"/>
        <source>[PclGridProjection::compute] find normals and use the normals</source>
        <translation>[PclGridProjection::compute] 寻找法线并使用法线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="165"/>
        <source>[GridProjection-Reconstruction] %1 points, %2 face(s)</source>
        <translation>[GridProjection-Reconstruction] %1 点, %2 面片</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="188"/>
        <source>Selected entity does not have any suitable scalar field or RGB.</source>
        <translation>所选实体没有合适的标量字段或者RGB.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="190"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation>参数错误：一个或者多个参数不能被接受</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PclGridProjection.cpp" line="192"/>
        <source>Pcl Grid Projection does not returned any point. Try relaxing your parameters</source>
        <translation>该模块没有返回任何点，尝试调整参数</translation>
    </message>
</context>
<context>
    <name>PclGridProjectionDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="20"/>
        <source>Grid Projection</source>
        <translation>网格投影</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="32"/>
        <source>Normal Estimation Parameters</source>
        <translation>法线评估参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="47"/>
        <source>Use Knn Search</source>
        <translation>使用K近邻搜索</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="89"/>
        <source>Normal Search Radius</source>
        <translation>法线搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="115"/>
        <source>Grid Projection Parameters</source>
        <translation>网格投影参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="127"/>
        <source>Resolution</source>
        <translation>分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="153"/>
        <source>Padding Size</source>
        <translation>填充大小</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PclGridProjectionDlg.ui" line="173"/>
        <source>Maximum BinarySearch Level</source>
        <translation>最大二分搜索层数</translation>
    </message>
</context>
<context>
    <name>PickOneElementDialog</name>
    <message>
        <location filename="../../common/ui_templates/pickOneElementDlg.ui" line="32"/>
        <source>Choose parameter</source>
        <translation>选择参数</translation>
    </message>
    <message>
        <location filename="../../common/ui_templates/pickOneElementDlg.ui" line="49"/>
        <source>Elements</source>
        <translation>元素</translation>
    </message>
</context>
<context>
    <name>PlaneEditDlg</name>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="14"/>
        <source>Plane properties</source>
        <translation>平面属性</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="24"/>
        <source>Dip / dip direction</source>
        <translation>倾向/倾斜方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="30"/>
        <source>dip</source>
        <translation>倾斜</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="53"/>
        <source>dip direction</source>
        <translation>倾斜方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="76"/>
        <source>Whether the plane normal should point upward (Z+) or backward (Z-)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="79"/>
        <source>upward</source>
        <translation>向上</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="90"/>
        <source>Normal</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="148"/>
        <source>Dimensions</source>
        <translation>尺寸</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="157"/>
        <source>width</source>
        <translation>宽度</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="164"/>
        <source>Plane width</source>
        <translation>面宽度</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="183"/>
        <source>height</source>
        <translation>高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="190"/>
        <source>Plane height</source>
        <translation>平面的高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="212"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../ui_templates/planeEditDlg.ui" line="237"/>
        <source>Pick the plane center (click again to cancel)</source>
        <translation>拾取平面中心（再次单击取消）</translation>
    </message>
</context>
<context>
    <name>PlyOpenDlg</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="14"/>
        <source>Ply File Open</source>
        <translation>打开Ply文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="22"/>
        <source>Type</source>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="43"/>
        <source>Elements</source>
        <translation>元素</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="69"/>
        <source>Properties</source>
        <translation>属性</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="95"/>
        <source>Textures</source>
        <translation>纹理</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="131"/>
        <source>Point X</source>
        <translation>点 X</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="141"/>
        <source>Point Y</source>
        <translation>点 Y</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="151"/>
        <source>Point Z</source>
        <translation>点 Z</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="161"/>
        <source>Red</source>
        <translation>红色</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="171"/>
        <source>Green</source>
        <translation>绿色</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="181"/>
        <source>Blue</source>
        <translation>蓝色</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="191"/>
        <source>Intensity</source>
        <translation>强度</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="231"/>
        <source>Faces</source>
        <translation>面片</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="241"/>
        <source>Texture coordinates</source>
        <translation>纹理坐标</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="254"/>
        <source>Scalar</source>
        <translation>标量</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="261"/>
        <source>Texture index</source>
        <translation>纹理索引</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="273"/>
        <source>Add Scalar field</source>
        <translation>添加标量字段</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="327"/>
        <source>Apply</source>
        <translation>应用</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="334"/>
        <source>Apply all</source>
        <translation>应用所有</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/openPlyFileDlg.ui" line="341"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>PointListPickingDlg</name>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="20"/>
        <source>Point list picking</source>
        <translation>点拾取列表</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="31"/>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="34"/>
        <source>Remove last entry</source>
        <translation>删除最后一个条目</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="37"/>
        <source>remove last</source>
        <translation>移除最后一行</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="48"/>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="51"/>
        <source>export to ASCII file</source>
        <translation>导出为ASCII文件</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="68"/>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="71"/>
        <source>Convert list to new cloud (and close dialog)</source>
        <translation>转换拾取点列表到新的点云（并关闭对话框）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="74"/>
        <source>to cloud</source>
        <translation>到点云</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="85"/>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="88"/>
        <source>Close dialog (list will be lost)</source>
        <translation>关闭对话框（列表将会被丢弃）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="91"/>
        <source>stop</source>
        <translation>停止</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="115"/>
        <source>count</source>
        <translation>数量</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="168"/>
        <source>Index</source>
        <translation>索引</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="193"/>
        <source>marker size</source>
        <translation>标记尺寸</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="226"/>
        <source>start index</source>
        <translation>开始索引</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="262"/>
        <source>Show global coordinates (instead of shifted ones)</source>
        <translation>显示全局坐标（非偏移后的）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointListPickingDlg.ui" line="265"/>
        <source>show global coordinates</source>
        <translation>显示全局坐标</translation>
    </message>
</context>
<context>
    <name>PointPropertiesDlg</name>
    <message>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="14"/>
        <source>Points Properties</source>
        <translation>点属性</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="32"/>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="35"/>
        <source>Select one point and display its information</source>
        <comment>Display selected point properties</comment>
        <translation>选择一个点并显示其信息</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="49"/>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="52"/>
        <source>Select 2 points and display segment information (length, etc.)</source>
        <comment>Compute point to point distance</comment>
        <translation>选择两个点并显示段信息 (长度等.)</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="69"/>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="72"/>
        <source>Select 3 points and display corresponding triangle information</source>
        <translation>选择三个点并显示对应三角形信息</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="86"/>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="89"/>
        <source>Define a rectangular 2D label</source>
        <translation>定义一个2D矩形标签</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="92"/>
        <source>2D zone</source>
        <translation>2D区域</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="103"/>
        <location filename="../ui_templates/pointPropertiesDlg.ui" line="106"/>
        <source>Save current label (added to cloud children)</source>
        <translation>保存当前标签（添加到当前点云下一级）</translation>
    </message>
</context>
<context>
    <name>PointsSamplingDialog</name>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="14"/>
        <source>Points Sampling on mesh</source>
        <translation>从网格采样点</translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="32"/>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="55"/>
        <source>Total number of sampled points (approx.)</source>
        <translation>总采样点数（粗略）</translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="35"/>
        <source>Points Number</source>
        <translation>点数量</translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="45"/>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="77"/>
        <source>Density: pts/square unit</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="48"/>
        <source>Density</source>
        <translation>密度</translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="111"/>
        <source>generate normals</source>
        <translation>生成法线</translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="123"/>
        <source>get colors</source>
        <translation>选取颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="130"/>
        <source>from RGB</source>
        <translation>从RGB</translation>
    </message>
    <message>
        <location filename="../ui_templates/ptsSamplingDlg.ui" line="140"/>
        <source>or from material/texture if available</source>
        <translation>或者从可用材质/纹理</translation>
    </message>
</context>
<context>
    <name>PoissonReconParamDialog</name>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="14"/>
        <source>Poisson Surface Reconstruction</source>
        <translation>泊松曲面重建</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="23"/>
        <source>Octree depth</source>
        <translation>八叉树深度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="33"/>
        <source>The maximum depth of the tree that will be used for surface reconstruction</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="49"/>
        <source>Resolution</source>
        <translation>分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="78"/>
        <source>interpolate cloud colors</source>
        <translation>插值点云颜色</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="92"/>
        <source>Density</source>
        <translation>密度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="98"/>
        <source>If this flag is enabled, the sampling density is output as a scalar field</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="101"/>
        <source>output density as SF</source>
        <translation>输出密度作为标量字段</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="111"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;SimSun&apos;; font-size:9pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;Density is useful to reduce the extents of the output mesh to fit as much as possible the input point cloud.&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;On the output mesh:&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;- Change the SF &apos;&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt; font-style:italic;&quot;&gt;min displayed&apos;&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;&apos; value (in the mesh properties) until the visible part meets your expectations&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;- Eventually export this mesh as a new one with &apos;&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt; font-style:italic;&quot;&gt;Edit &amp;gt; Scalar fields &amp;gt; Filter by Value&lt;/span&gt;&lt;span style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8pt;&quot;&gt;&apos; (you can delete the &apos;density&apos; scalar field afterwards) &lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="127"/>
        <source>Advanced</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="133"/>
        <source>boundary</source>
        <translation>边界</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="144"/>
        <source>Free</source>
        <translation>无限制的</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="149"/>
        <source>Dirichlet</source>
        <translation>狄利克雷</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="154"/>
        <source>Neumann</source>
        <translation>诺埃曼</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="162"/>
        <source>The importance that interpolation of the point samples is given in the formulation of the screened Poisson equation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="165"/>
        <source>point weight</source>
        <translation>点权重</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="172"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Interpolation weight (twice the b-spline degree by default)&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="185"/>
        <source>Enabling this flag has the reconstructor use linear interpolation to estimate the positions of iso-vertices.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="188"/>
        <source>Linear fit</source>
        <translation>线性适合</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="198"/>
        <source>The minimum number of sample points that should fall within an octree node
as the octree construction is adapted to sampling density. For noise-free
samples, small values in the range [1.0 - 5.0] can be used. For more noisy
samples, larger values in the range [15.0 - 20.0] may be needed to provide
a smoother, noise-reduced, reconstruction.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="205"/>
        <source>samples per node</source>
        <translation>每个节点采样量</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/poissonReconParamDlg.ui" line="212"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Minimum number of sample points that should fall within an octree node as the octree construction is adapted to sampling density.&lt;/p&gt;&lt;p&gt;For noise-free samples, small values in the range [1.0 - 5.0] can be used.&lt;/p&gt;&lt;p&gt;&lt;span style=&quot; font-weight:600; color:#ff0000;&quot;&gt;For more noisy samples&lt;/span&gt;, larger values in the range [15.0 - 20.0] may be needed to provide a smoother, noise-reduced, reconstruction.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>PoissonReconstruction</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="40"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="41"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="188"/>
        <source>Poisson Reconstruction</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="42"/>
        <source>Poisson Reconstruction from clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="150"/>
        <source>[PoissonReconstruction::compute] generate new normals</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="156"/>
        <source>[PoissonReconstruction::compute] find normals and use the normals</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="186"/>
        <source>[Poisson-Reconstruction] %1 points, %2 face(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="209"/>
        <source>Selected entity does not have any suitable scalar field or RGB.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="211"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/PoissonReconstruction.cpp" line="213"/>
        <source>Poisson Reconstruction does not returned any point. Try relaxing your parameters</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>PoissonReconstructionDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="20"/>
        <source>Poisson Reconstruction</source>
        <translation>泊松重建</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="32"/>
        <source>Normal Estimation Parameters</source>
        <translation>法线评估参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="47"/>
        <source>Normal Search Radius</source>
        <translation>法线搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="76"/>
        <source>Use Knn Search</source>
        <translation>使用K近邻搜索</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="115"/>
        <source>Poisson Reconstruction Parameters</source>
        <translation>泊松重建参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="140"/>
        <source>SolverDivide</source>
        <translation>求解器划分</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="147"/>
        <source>Scale</source>
        <translation>比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="173"/>
        <source>SamplesPerNode</source>
        <translation>每节点采样数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="205"/>
        <source>Confidence</source>
        <translation>置信度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="221"/>
        <source>Manifold</source>
        <translation>重叠</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="231"/>
        <source>Degree</source>
        <translation>角度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="251"/>
        <source>Tree Depth</source>
        <translation>树深度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="284"/>
        <source>IsoDivide</source>
        <translation>等值划分</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Surfaces/dialogs/PoissonReconstructionDlg.ui" line="297"/>
        <source>OutputPolygons</source>
        <translation>输出多边形</translation>
    </message>
</context>
<context>
    <name>PrimitiveFactoryDlg</name>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="14"/>
        <source>Primitive factory</source>
        <translation>基础模型库</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="34"/>
        <source>Plane</source>
        <translation>平面</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="121"/>
        <source>Box</source>
        <translation>箱体</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="231"/>
        <source>Sphere</source>
        <translation>球体</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="239"/>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="402"/>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="830"/>
        <source>radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="290"/>
        <source>Position</source>
        <translation>位置</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="364"/>
        <source>Set position to origin</source>
        <translation>把位置设置到原点</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="367"/>
        <source>clear</source>
        <translation>清空</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="374"/>
        <source>Try to interpret clipboard contents as position (&quot;x y z&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="377"/>
        <source>clipboard</source>
        <translation>剪贴板</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="394"/>
        <source>Cylinder</source>
        <translation>圆柱体</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="425"/>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="537"/>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="853"/>
        <source>height</source>
        <translation>高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="481"/>
        <source>Cone</source>
        <translation>锥体</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="491"/>
        <source>top radius</source>
        <translation>顶部半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="514"/>
        <source>bottom radius</source>
        <translation>底部半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="562"/>
        <source>Snout mode</source>
        <translation>鼻子模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="574"/>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="577"/>
        <source>displacement of axes along X-axis</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="580"/>
        <source>x offset</source>
        <translation>x偏移</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="603"/>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="606"/>
        <source>displacement of axes along Y-axis</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="609"/>
        <source>y offset</source>
        <translation>y偏移</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="668"/>
        <source>Torus</source>
        <translation>类圆环体</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="678"/>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="761"/>
        <source>inside radius</source>
        <translation>内半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="701"/>
        <source>outside radius</source>
        <translation>外半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="724"/>
        <source>angle (degrees)</source>
        <translation>角度(°)</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="749"/>
        <source>Rectangular section</source>
        <translation>矩形截面</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="820"/>
        <source>Dish</source>
        <translation>碗体</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="878"/>
        <source>Ellipsoid mode</source>
        <translation>椭球体模式</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="890"/>
        <source>radius 2</source>
        <translation>半径2</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="950"/>
        <source>Initial precision</source>
        <translation>初始精度</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="986"/>
        <source>Create</source>
        <translation>创建</translation>
    </message>
    <message>
        <location filename="../ui_templates/primitiveFactoryDlg.ui" line="993"/>
        <source>Close</source>
        <translation>关闭</translation>
    </message>
</context>
<context>
    <name>ProbeConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="14"/>
        <source>Form</source>
        <translation>Prob窗口</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="22"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="92"/>
        <source>Sphere</source>
        <translation>球体</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="43"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="50"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="65"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="79"/>
        <source>Source</source>
        <translation>过滤类型</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="87"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="117"/>
        <source>Line</source>
        <translation>线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="97"/>
        <source>Box</source>
        <translation>箱体</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="102"/>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="162"/>
        <source>Implicit Plane</source>
        <translation>隐式平面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="110"/>
        <source>Open ...</source>
        <translation>打开……</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="131"/>
        <source>Point1</source>
        <translation>点1</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="138"/>
        <source>Point2</source>
        <translation>点2</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="173"/>
        <source>Normal</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="192"/>
        <source>Origin</source>
        <translation>原点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probeconfig.ui" line="207"/>
        <source>Plot</source>
        <translation>绘制</translation>
    </message>
</context>
<context>
    <name>ProbeWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/probewindow.cpp" line="50"/>
        <source>Probe</source>
        <translation>探针</translation>
    </message>
</context>
<context>
    <name>ProfileImportDlg</name>
    <message>
        <location filename="../../plugins/core/qSRA/profileImportDlg.ui" line="14"/>
        <source>Import profile</source>
        <translation>导入配置文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/profileImportDlg.ui" line="20"/>
        <source>Profile file</source>
        <translation>概要文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/profileImportDlg.ui" line="43"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;Here is an example of profile file:&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; font-weight:600; font-style:italic; color:#787878;&quot;&gt;(don&apos;t insert blank lines, don&apos;t change the columns names)&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt; font-weight:600; color:#787878;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;Xc	Yc	Zc	&lt;/span&gt;&lt;span style=&quot; font-size:8pt; color:#55aaff;&quot;&gt;(profile origin)&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;4667.000	10057.000	171.000	&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;R		H	&lt;/span&gt;&lt;span style=&quot; font-size:8pt; color:#55aaff;&quot;&gt;(radius and height of profile vertices)&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;59.3235190427553	28.685&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;58.8177164625621	30.142&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;58.32550519856	31.594&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;57.8404034801208	33.044&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; color:#787878;&quot;&gt;...&lt;/span&gt;&lt;/p&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt; color:#787878;&quot;&gt;&lt;br /&gt;&lt;/p&gt;
&lt;p style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;span style=&quot; font-size:8pt; font-style:italic; color:#ff0000;&quot;&gt;Note: accurate position of the profile origin on the rotation axis is only necessary for conical projection&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/profileImportDlg.ui" line="71"/>
        <source>profile axis</source>
        <translation>轮廓轴</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/profileImportDlg.ui" line="100"/>
        <source>height values are absolute (i.e. not relative to profile origin)</source>
        <translation>绝对高度值（例如：非相对轮廓原点）</translation>
    </message>
</context>
<context>
    <name>ProjectionFilter</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="38"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="39"/>
        <source>Projection Filter</source>
        <translation>投影滤波器</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="40"/>
        <source>Projection Filter for the selected entity</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="146"/>
        <source>%1-projection</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="194"/>
        <source>%1-boundary</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="238"/>
        <source>Selected entity does not have any suitable scalar field or RGB. Intensity scalar field or RGB are needed for computing SIFT</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="240"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/ProjectionFilter.cpp" line="242"/>
        <source>Projection extraction does not returned any point. Try relaxing your parameters</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>ProjectionFilterDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="20"/>
        <source>Projection Filter</source>
        <translation>投影滤波器</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="47"/>
        <source>Projection</source>
        <translation>平面投影</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="55"/>
        <source>ax + by + cz + d = 0</source>
        <translation>ax + by + cz + d = 0</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="65"/>
        <source>a</source>
        <translation>a</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="91"/>
        <source>b</source>
        <translation>b</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="114"/>
        <source>c</source>
        <translation>c</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="137"/>
        <source>d</source>
        <translation>d</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="163"/>
        <source>Boundary</source>
        <translation>边界提取</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="174"/>
        <source>Voxel Grid [Leaf Size]</source>
        <translation>使用体素滤波[叶子尺寸]</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="244"/>
        <source>Use Knn Search</source>
        <translation>使用K近邻搜索</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="257"/>
        <source>Normal Search Radius</source>
        <translation>法线搜索半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/ProjectionFilterDlg.ui" line="264"/>
        <source>Boundary Angle Threshold（degree）</source>
        <translation>边界角度阈值</translation>
    </message>
</context>
<context>
    <name>QObject</name>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationTool.cpp" line="317"/>
        <source>Cloud to profile radial distance</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/distanceMapGenerationTool.cpp" line="318"/>
        <source>Polyline: %1 vertices
Cloud: %2 points</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/src/qM3C2Process.cpp" line="800"/>
        <source>M3C2 Distances Computation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/src/qM3C2Process.cpp" line="801"/>
        <source>Core points: %1</source>
        <translation>核心点集: %1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qM3C2/src/qM3C2Commands.h" line="72"/>
        <source>_M3C2</source>
        <translation>_M3C2</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qLAS_FWF/Filter/LASFWFFilter.cpp" line="394"/>
        <source>Export LAS file</source>
        <translation>拉斯维加斯导出文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qLAS_FWF/Filter/LASFWFFilter.cpp" line="395"/>
        <location filename="../../plugins/core/qLAS_FWF/Filter/LASFWFFilter.cpp" line="653"/>
        <source>Points: %1</source>
        <translation>Points: %1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qLAS_FWF/Filter/LASFWFFilter.cpp" line="645"/>
        <source>Reading %1 points</source>
        <translation>读取 %1 点</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qLAS_FWF/Filter/LASFWFFilter.cpp" line="652"/>
        <source>Import LAS file</source>
        <translation>拉斯维加斯文件导入</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qLAS_FWF/Filter/LASFWFFilter.cpp" line="1085"/>
        <source>No valid point in file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/src/stereogramDlg.cpp" line="124"/>
        <source>Stereogram</source>
        <translation>立体图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/src/stereogramDlg.cpp" line="125"/>
        <source>Preparing polar display...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoProcess.cpp" line="532"/>
        <source>Remaining points to classify: %1
Source points: %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoProcess.cpp" line="533"/>
        <source>Classification</source>
        <translation>分类</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoProcess.cpp" line="901"/>
        <source>Core points: %1
Source points: %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupoProcess.cpp" line="902"/>
        <source>Labelling</source>
        <translation>标签</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/SoiFilter.cpp" line="87"/>
        <source>Open SOI file</source>
        <translation>打开SOI文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/SoiFilter.cpp" line="88"/>
        <source>%1 scans / %2 points</source>
        <translation>%1扫描/ %2点</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/PVFilter.cpp" line="91"/>
        <source>Save PV file</source>
        <translation>保存PV文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/PVFilter.cpp" line="92"/>
        <location filename="../../plugins/core/qAdditionalIO/src/PVFilter.cpp" line="158"/>
        <location filename="../../plugins/core/qAdditionalIO/src/PNFilter.cpp" line="89"/>
        <location filename="../../plugins/core/qAdditionalIO/src/PNFilter.cpp" line="161"/>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="261"/>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1020"/>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1090"/>
        <location filename="../../libs/eCV_db/ecvPointCloud.cpp" line="4067"/>
        <source>Points: %L1</source>
        <translation>Points:% L1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/PVFilter.cpp" line="157"/>
        <source>Open PV file</source>
        <translation>打开PV文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/PNFilter.cpp" line="88"/>
        <source>Save PN file</source>
        <translation>保存PN文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/PNFilter.cpp" line="160"/>
        <source>Open PN file</source>
        <translation>打开PN文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerFilter.cpp" line="227"/>
        <source>Open Bundler file</source>
        <translation>打开打包机文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerFilter.cpp" line="228"/>
        <source>Cameras: %1
Points: %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerFilter.cpp" line="623"/>
        <source>Open &amp; process images</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerFilter.cpp" line="624"/>
        <source>Images: %1</source>
        <translation>图像: %1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerFilter.cpp" line="643"/>
        <source>Preparing colored DTM</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerCommand.cpp" line="37"/>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerCommand.cpp" line="59"/>
        <location filename="../ecvCommandLineCommands.h" line="363"/>
        <source>Missing parameter: filename after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerCommand.cpp" line="41"/>
        <source>Importing Bundler file: &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerCommand.cpp" line="68"/>
        <location filename="../ecvCommandLineCommands.h" line="2823"/>
        <location filename="../ecvCommandLineCommands.h" line="2835"/>
        <source>Missing parameter: value after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerCommand.cpp" line="72"/>
        <location filename="../ecvCommandLineCommands.h" line="2827"/>
        <location filename="../ecvCommandLineCommands.h" line="2839"/>
        <source>Invalid parameter: value after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerCommand.cpp" line="87"/>
        <source>Missing parameter: vertices count after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAdditionalIO/src/BundlerCommand.cpp" line="91"/>
        <source>Invalid parameter: vertices count after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/SimpleBinFilter.cpp" line="188"/>
        <location filename="../../libs/eCV_io/SimpleBinFilter.cpp" line="477"/>
        <source>Simple BIN file</source>
        <translation>简单的本文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/SimpleBinFilter.cpp" line="189"/>
        <source>Saving %1 points / %2 scalar field(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/SimpleBinFilter.cpp" line="478"/>
        <source>Loading %1 points / %2 scalar field(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ShpFilter.cpp" line="1918"/>
        <source>Load SHP file</source>
        <translation>负载SHP文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ShpFilter.cpp" line="1919"/>
        <source>File size: %1</source>
        <translation>文件大小: %1</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="114"/>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="206"/>
        <location filename="../../libs/eCV_io/ObjFilter.cpp" line="100"/>
        <source>Saving mesh [%1]</source>
        <translation>保存网格 [%1]</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="115"/>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="207"/>
        <source>Number of facets: %1</source>
        <translation>刻面数量:%1</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="619"/>
        <source>(ASCII) STL file</source>
        <translation>(ASCII) STL文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="620"/>
        <location filename="../../libs/eCV_io/PlyFilter.cpp" line="1659"/>
        <location filename="../../libs/eCV_io/ObjFilter.cpp" line="467"/>
        <source>Loading in progress...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="946"/>
        <source>Loading binary STL file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/STLFilter.cpp" line="947"/>
        <source>Loading %1 faces</source>
        <translation>加载 %1 面片</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/PlyFilter.cpp" line="1660"/>
        <source>PLY file</source>
        <translation>的文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/PTXFilter.cpp" line="114"/>
        <source>Loading PTX file</source>
        <translation>PTX加载文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ObjFilter.cpp" line="101"/>
        <source>Triangles: %1</source>
        <translation>三角形: %1</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ObjFilter.cpp" line="466"/>
        <source>OBJ file</source>
        <translation>OBJ file</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="260"/>
        <source>Save LAS file</source>
        <translation>拉斯维加斯文件保存</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1019"/>
        <source>Open LAS file</source>
        <translation>打开LAS文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1066"/>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1107"/>
        <source>LAS file</source>
        <translation>LAS file</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1067"/>
        <source>Please wait... reading in progress</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1089"/>
        <source>Tiling points</source>
        <translation>瓷砖点</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/LASFilter.cpp" line="1108"/>
        <source>Please wait... writing in progress</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/BinFilter.cpp" line="160"/>
        <location filename="../../libs/eCV_io/BinFilter.cpp" line="373"/>
        <source>BIN file</source>
        <translation>BIN file</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/BinFilter.cpp" line="161"/>
        <source>Please wait... saving in progress</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/BinFilter.cpp" line="374"/>
        <source>Loading: %1</source>
        <translation>加载: %1</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/BinFilter.cpp" line="1079"/>
        <source>Open Bin file (old style)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/BinFilter.cpp" line="1105"/>
        <source>cloud %1/%2 (%3 points)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/AsciiFilter.cpp" line="205"/>
        <source>Saving cloud [%1]</source>
        <translation>保存点云 [%1]</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/AsciiFilter.cpp" line="206"/>
        <source>Number of points: %1</source>
        <translation>点数:%1</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/AsciiFilter.cpp" line="767"/>
        <source>Open ASCII file [%1]</source>
        <translation>打开ASCII文件[%1]</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/AsciiFilter.cpp" line="768"/>
        <location filename="../../libs/eCV_io/AsciiFilter.cpp" line="875"/>
        <source>Approximate number of points: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvRasterGrid.cpp" line="217"/>
        <source>Grid generation</source>
        <translation>网格生成</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvRasterGrid.cpp" line="218"/>
        <source>Points: %L1
Cells: %L2 x %L3</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvPolyline.cpp" line="593"/>
        <source>sampled</source>
        <translation>采样</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvPointCloud.cpp" line="3850"/>
        <source>Normals computation</source>
        <translation>法线计算</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvPointCloud.cpp" line="3878"/>
        <source>Grid: %1 x %2</source>
        <translation>Grid: %1 x %2</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvPointCloud.cpp" line="4066"/>
        <source>Orienting normals</source>
        <translation>定向法线</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvPointCloud.cpp" line="4328"/>
        <source>FWF amplitude</source>
        <translation>FWF振幅</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvPointCloud.cpp" line="4329"/>
        <source>Determining min and max FWF values
Points: </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvMinimumSpanningTreeForNormsDirection.cpp" line="199"/>
        <source>Orient normals (MST)</source>
        <translation>东方法线(MST)</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvMinimumSpanningTreeForNormsDirection.cpp" line="201"/>
        <source>Compute Minimum spanning tree
Points: %1
Edges: %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvMinimumSpanningTreeForNormsDirection.cpp" line="203"/>
        <source>Compute Minimum spanning tree
Points: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvMesh.cpp" line="558"/>
        <source>Laplacian smooth</source>
        <translation>拉普拉斯算子的光滑</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_db/ecvMesh.cpp" line="559"/>
        <source>Iterations: %1
Vertices: %2
Faces: %3</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/widgetsSurfaceInterface.cpp" line="13"/>
        <source>Surface Interface</source>
        <translation>表面的界面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/smallWidgets/smallWidgetsInterface.cpp" line="36"/>
        <source>SmallWidgets Interface</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/widgetsFiltersInterface.cpp" line="19"/>
        <source>Filters Interface</source>
        <translation>过滤器接口</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="122"/>
        <source>Fifty shades of grey</source>
        <translation>五十度灰</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="123"/>
        <source>Dusk</source>
        <translation>黄昏</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="124"/>
        <source>Miami Dolphins</source>
        <translation>迈阿密海豚</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="125"/>
        <source>Superman</source>
        <translation>超人</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="126"/>
        <source>Timber</source>
        <translation>木材</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="127"/>
        <source>Lizard</source>
        <translation>蜥蜴</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="128"/>
        <source>Ukraine</source>
        <translation>乌克兰</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="129"/>
        <source>Starfall</source>
        <translation>Starcase</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="130"/>
        <source>Kyoto</source>
        <translation>《京都议定书》</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="131"/>
        <source>Miaka</source>
        <translation>Miaka</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="132"/>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="140"/>
        <source>Calm Darya</source>
        <translation>平静的河</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="133"/>
        <source>Mantle</source>
        <translation>地幔</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="134"/>
        <source>Opa</source>
        <translation>Grandpa</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="135"/>
        <source>Horizon</source>
        <translation>地平线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="136"/>
        <source>Rose Water</source>
        <translation>玫瑰水</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="137"/>
        <source>Harmonic Energy</source>
        <translation>谐波能量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="138"/>
        <source>Purple Paradise</source>
        <translation>紫色的天堂</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="139"/>
        <source>Aqua Marine</source>
        <translation>Aqua海洋</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="141"/>
        <source>Bora Bora</source>
        <translation>Bora Bora</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="142"/>
        <source>Winter</source>
        <translation>冬天</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="143"/>
        <source>Bright Vault</source>
        <translation>明亮的地下室</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="144"/>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="209"/>
        <source>Sunset</source>
        <translation>日落</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="145"/>
        <source>Sherbert</source>
        <translation>果汁牛奶冻</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="146"/>
        <source>Deep Sea Space</source>
        <translation>深海的空间</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="147"/>
        <source>Transfile</source>
        <translation>Transfile</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="148"/>
        <source>Ali</source>
        <translation>Ali</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="149"/>
        <source>Alihossein</source>
        <translation>Alihossein</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="150"/>
        <source>Christmas</source>
        <translation>圣诞节</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="151"/>
        <source>Minnesota Vikings</source>
        <translation>明尼苏达维京人</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="152"/>
        <source>Pizelex</source>
        <translation>Pizelex</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="153"/>
        <source>Netflix</source>
        <translation>网飞公司</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="154"/>
        <source>Green and Blue</source>
        <translation>绿色和蓝色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="155"/>
        <source>Fresh Turboscent</source>
        <translation>新鲜Turboscent</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="156"/>
        <source>Purple Bliss</source>
        <translation>紫色的幸福</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="157"/>
        <source>Martini</source>
        <translation>马提尼</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="158"/>
        <source>Shore</source>
        <translation>海岸</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="159"/>
        <source>Earthly</source>
        <translation>世俗的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="160"/>
        <source>Titanium</source>
        <translation>钛</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="161"/>
        <source>Sun on the Horizon</source>
        <translation>地平线上的太阳</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="162"/>
        <source>Grapefruit Sunset</source>
        <translation>葡萄柚日落</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="163"/>
        <source>Politics</source>
        <translation>政治</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="164"/>
        <source>Sweet Morning</source>
        <translation>甜蜜的早晨</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="165"/>
        <source>Forest</source>
        <translation>森林</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="166"/>
        <source>Back to the Future</source>
        <translation>回到未来</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="167"/>
        <source>Dark Knight</source>
        <translation>黑暗骑士</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="168"/>
        <source>Green to dark</source>
        <translation>绿色,黑色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="169"/>
        <source>Virgin America</source>
        <translation>维珍美国航空公司</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="170"/>
        <source>Turquoise flow</source>
        <translation>蓝绿色的流</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="171"/>
        <source>Portrait</source>
        <translation>肖像</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="172"/>
        <source>Flickr</source>
        <translation>Flickr</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="173"/>
        <source>Predawn</source>
        <translation>黎明前的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="174"/>
        <source>Crazy Orange I</source>
        <translation>我疯狂的橙色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="175"/>
        <source>ServQuick</source>
        <translation>ServQuick</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="176"/>
        <source>Cheer Up Emo Kid</source>
        <translation>振作起来，情绪小子</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="177"/>
        <source>Man of Steel</source>
        <translation>钢的人</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="178"/>
        <source>Moor</source>
        <translation>沼泽</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="179"/>
        <source>Forever Lost</source>
        <translation>永远的失去了</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="180"/>
        <source>Dracula</source>
        <translation>吸血鬼</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="181"/>
        <source>Moss</source>
        <translation>莫斯</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="182"/>
        <source>Lemon Twist</source>
        <translation>柠檬扭曲</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="183"/>
        <source>Frozen</source>
        <translation>冻</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="184"/>
        <source>Emerald Water</source>
        <translation>翡翠水</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="185"/>
        <source>Mirage</source>
        <translation>海市蜃楼</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="186"/>
        <source>Shroom Haze</source>
        <translation>金色烟雾</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="187"/>
        <source>Venice Blue</source>
        <translation>威尼斯蓝</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="190"/>
        <source>Alpine</source>
        <translation>高山</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="191"/>
        <source>Lake</source>
        <translation>湖</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="192"/>
        <source>Army</source>
        <translation>军队</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="193"/>
        <source>Mint</source>
        <translation>Mint</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="194"/>
        <source>Atlantic</source>
        <translation>大西洋</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="195"/>
        <source>Neon</source>
        <translation>霓虹灯</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="196"/>
        <source>Aurora</source>
        <translation>极光</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="197"/>
        <source>Pearl</source>
        <translation>珍珠</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="198"/>
        <source>Avocado</source>
        <translation>鳄梨</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="199"/>
        <source>Plum</source>
        <translation>李子</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="200"/>
        <source>Beach</source>
        <translation>海滩</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="201"/>
        <source>Rose</source>
        <translation>玫瑰</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="202"/>
        <source>Candy</source>
        <translation>糖果</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="203"/>
        <source>Solar</source>
        <translation>太阳能</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="204"/>
        <source>CMYK</source>
        <translation>CMYK</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="205"/>
        <source>South West</source>
        <translation>南西</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="206"/>
        <source>Deep Sea</source>
        <translation>深海</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="207"/>
        <source>Starry Night</source>
        <translation>繁星闪烁的夜晚,</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="208"/>
        <source>Fall</source>
        <translation>秋天</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="210"/>
        <source>Fruit Punch</source>
        <translation>果汁</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="211"/>
        <source>Thermometer</source>
        <translation>温度计</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="212"/>
        <source>Island</source>
        <translation>岛</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/gradientcombobox.cpp" line="213"/>
        <source>Watermelon</source>
        <translation>西瓜</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="12"/>
        <source>Clip</source>
        <translation>剪辑</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="16"/>
        <source>Cut</source>
        <translation>减少</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="20"/>
        <source>Slice</source>
        <translation>片</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="24"/>
        <source>Isosurface</source>
        <translation>等值面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="28"/>
        <source>Threshold</source>
        <translation>阈值</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="32"/>
        <source>Streamline</source>
        <translation>简化</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="36"/>
        <source>Smooth</source>
        <translation>光滑的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/datafilter.cpp" line="40"/>
        <source>Decimate</source>
        <translation>批量滤除</translation>
    </message>
    <message>
        <location filename="../main.cpp" line="80"/>
        <location filename="../main.cpp" line="224"/>
        <source>[Global Shift] Max abs. coord = %1 / max abs. diag = %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../main.cpp" line="117"/>
        <source>Translation</source>
        <translation>翻译</translation>
    </message>
    <message>
        <location filename="../main.cpp" line="117"/>
        <source>Failed to load language file &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../main.cpp" line="132"/>
        <location filename="../main.cpp" line="163"/>
        <source>Error</source>
        <translation>错误</translation>
    </message>
    <message>
        <location filename="../main.cpp" line="133"/>
        <source>This application needs OpenGL 2.1 at least to run!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../main.cpp" line="164"/>
        <source>Failed to initialize the main application window?!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../main.cpp" line="256"/>
        <source>Failed to start the plugin &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../main.cpp" line="264"/>
        <source>Couldn&apos;t find the plugin &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../main.cpp" line="295"/>
        <location filename="../main.cpp" line="299"/>
        <source>ECV crashed!</source>
        <translation>ECV crashed!</translation>
    </message>
    <message>
        <location filename="../main.cpp" line="295"/>
        <source>Hum, it seems that ECV has crashed... Sorry about that :)
</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../main.cpp" line="299"/>
        <source>Hum, it seems that ECV has crashed... Sorry about that :)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvVolumeCalcTool.cpp" line="625"/>
        <source>Volume computation</source>
        <translation>体积计算</translation>
    </message>
    <message>
        <location filename="../ecvVolumeCalcTool.cpp" line="626"/>
        <source>Cells: %1 x %2</source>
        <translation>Cells: %1 x %2</translation>
    </message>
    <message>
        <location filename="../ecvLibAlgorithms.cpp" line="606"/>
        <source>Computing entities scales</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvLibAlgorithms.cpp" line="607"/>
        <source>Entities: %1</source>
        <translation>实体: %1</translation>
    </message>
    <message>
        <location filename="../ecvLibAlgorithms.cpp" line="735"/>
        <source>Rescaling entities</source>
        <translation>重新调节实体</translation>
    </message>
    <message>
        <location filename="../ecvFilterWindowTool.cpp" line="711"/>
        <location filename="../ecvClippingBoxTool.cpp" line="675"/>
        <source>Section extraction</source>
        <translation>部分提取</translation>
    </message>
    <message>
        <location filename="../ecvFilterWindowTool.cpp" line="712"/>
        <location filename="../ecvClippingBoxTool.cpp" line="676"/>
        <source>Section(s): %L1</source>
        <translation>Section (s):% L1</translation>
    </message>
    <message>
        <location filename="../ecvFilterWindowTool.cpp" line="829"/>
        <location filename="../ecvClippingBoxTool.cpp" line="799"/>
        <source>Up to (%1 x %2 x %3) = %4 section(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvFilterWindowTool.cpp" line="914"/>
        <location filename="../ecvClippingBoxTool.cpp" line="881"/>
        <source>Contour(s): %L1</source>
        <translation>Contour (s):% L1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="166"/>
        <source>Unhandled format specifier (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="190"/>
        <source>Output export format (clouds) set to: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="215"/>
        <location filename="../ecvCommandLineCommands.h" line="340"/>
        <source>Missing parameter: extension after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="218"/>
        <source>New output extension for clouds: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="226"/>
        <source>Missing parameter: precision value after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="230"/>
        <source>Invalid value for precision! (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="233"/>
        <location filename="../ecvCommandLineCommands.h" line="252"/>
        <location filename="../ecvCommandLineCommands.h" line="281"/>
        <location filename="../ecvCommandLineCommands.h" line="296"/>
        <source>Argument &apos;%1&apos; is only applicable to ASCII format!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="249"/>
        <source>Missing parameter: separator character after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="266"/>
        <source>Invalid separator! (&apos;%1&apos;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="327"/>
        <source>Output export format (meshes) set to: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="343"/>
        <source>New output extension for meshes: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="377"/>
        <source>Missing parameter: number of lines after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="384"/>
        <source>Invalid parameter: number of lines after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="387"/>
        <source>Will skip %1 lines</source>
        <translation>将跳过%1行</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="396"/>
        <source>Missing parameter: global shift vector or %1 after &apos;%2&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="413"/>
        <source>Missing parameter: global shift vector after &apos;%1&apos; (3 values expected)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="421"/>
        <source>Invalid parameter: X coordinate of the global shift vector after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="424"/>
        <source>Invalid parameter: Y coordinate of the global shift vector after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="427"/>
        <source>Invalid parameter: Z coordinate of the global shift vector after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="464"/>
        <source>No entity loaded (be sure to open at least one file with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="516"/>
        <source>No point cloud to normal calculation (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="521"/>
        <source>Missing parameter: radius after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="528"/>
        <source>Invalid radius</source>
        <translation>无效的半径</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="531"/>
        <source>	Radius: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="568"/>
        <source>Invalid parameter: unknown orientation &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="572"/>
        <source>Missing orientation</source>
        <translation>丢失方向</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="588"/>
        <source>Invalid parameter: unknown model &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="592"/>
        <source>Missing model</source>
        <translation>缺失的模型</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="609"/>
        <source>cloud-&gt;hasNormals: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="616"/>
        <source>.OctreeNormal</source>
        <translation>.OctreeNormal</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="641"/>
        <source>No point cloud to resample (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="646"/>
        <source>Missing parameter: resampling method after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="650"/>
        <source>	Method: </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="655"/>
        <source>Missing parameter: number of points after &quot;-%1 RANDOM&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="664"/>
        <source>	Output points: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="669"/>
        <location filename="../ecvCommandLineCommands.h" line="726"/>
        <location filename="../ecvCommandLineCommands.h" line="792"/>
        <location filename="../ecvCommandLineCommands.h" line="901"/>
        <location filename="../ecvCommandLineCommands.h" line="3238"/>
        <source>	Processing cloud #%1 (%2)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="676"/>
        <location filename="../ecvCommandLineCommands.h" line="734"/>
        <location filename="../ecvCommandLineCommands.h" line="802"/>
        <source>	Result: %1 points</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="685"/>
        <location filename="../ecvCommandLineCommands.h" line="743"/>
        <location filename="../ecvCommandLineCommands.h" line="811"/>
        <source>.subsampled</source>
        <translation>.subsampled</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="699"/>
        <location filename="../ecvCommandLineCommands.h" line="757"/>
        <location filename="../ecvCommandLineCommands.h" line="825"/>
        <source>_SUBSAMPLED</source>
        <translation>_SUBSAMPLED</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="713"/>
        <source>Missing parameter: spatial step after &quot;-%1 SPATIAL&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="721"/>
        <source>	Spatial step: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="771"/>
        <source>Missing parameter: octree level after &quot;-%1 OCTREE&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="780"/>
        <location filename="../ecvCommandLineCommands.h" line="873"/>
        <source>	Octree level: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="815"/>
        <source>OCTREE_LEVEL_%1_SUBSAMPLED</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="859"/>
        <source>No point cloud loaded (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="865"/>
        <source>Missing parameter: octree level after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="878"/>
        <source>Missing parameter: minimum number of points per component after &quot;-%1 [octree level]&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="885"/>
        <source>	Min number of points per component: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="958"/>
        <source>_COMPONENT_%1</source>
        <translation>_COMPONENT _ %1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="972"/>
        <source>Failed to create component #%1! (not enough memory)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="988"/>
        <source>%1 component(s) were created</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1017"/>
        <source>Missing parameter: curvature type after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1035"/>
        <source>Invalid curvature type after &quot;-%1&quot;. Got &apos;%2&apos; instead of MEAN or GAUSS.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1045"/>
        <source>Failed to read a numerical parameter: kernel size (after curvature type). Got &apos;%1&apos; instead.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1046"/>
        <location filename="../ecvCommandLineCommands.h" line="1271"/>
        <source>	Kernel size: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1049"/>
        <source>No point cloud on which to compute curvature! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1060"/>
        <source>%1_CURVATURE_KERNEL_%2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1070"/>
        <location filename="../ecvCommandLineCommands.h" line="1120"/>
        <location filename="../ecvCommandLineCommands.h" line="1166"/>
        <source>Missing parameter: density type after &quot;-%1&quot; (KNN/SURFACE/VOLUME)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1088"/>
        <source>Invalid parameter: density type is expected after &quot;-%1&quot; (KNN/SURFACE/VOLUME)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1102"/>
        <source>No point cloud on which to compute approx. density! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1147"/>
        <source>Missing parameter: sphere radius after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1153"/>
        <source>Failed to read a numerical parameter: sphere radius (after &quot;-%1&quot;). Got &apos;%2&apos; instead.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1154"/>
        <source>	Sphere radius: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1174"/>
        <source>No point cloud on which to compute density! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1202"/>
        <source>Missing parameter: boolean (whether SF is euclidean or not) after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1212"/>
        <location filename="../ecvCommandLineCommands.h" line="1429"/>
        <source>Invalid boolean value after &quot;-%1&quot;. Got &apos;%2&apos; instead of TRUE or FALSE.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1216"/>
        <source>No point cloud on which to compute SF gradient! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1227"/>
        <location filename="../ecvCommandLineCommands.h" line="1443"/>
        <source>cmd.warning: cloud &apos;%1&apos; has no scalar field (it will be ignored)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1232"/>
        <source>cmd.warning: cloud &apos;%1&apos; has several scalar fields (the active one will be used by default, or the first one if none is active)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1264"/>
        <source>Missing parameter: kernel size after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1270"/>
        <source>Failed to read a numerical parameter: kernel size (after &quot;-%1&quot;). Got &apos;%2&apos; instead.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1274"/>
        <source>No point cloud on which to compute roughness! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1285"/>
        <source>ROUGHNESS_KERNEL_%2</source>
        <translation>ROUGHNESS_KERNEL_%2</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1302"/>
        <source>Missing parameter: transformation file after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1307"/>
        <source>Failed to read transformation matrix file &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1309"/>
        <source>Transformation:
</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1312"/>
        <source>No entity on which to apply the transformation! (be sure to open one with &quot;-%1 [filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1351"/>
        <source>No loaded entity! (be sure to open one with &quot;-%1 [filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1382"/>
        <source>Missing parameter: color scale file after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1389"/>
        <source>Failed to read color scale file &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1392"/>
        <source>No point cloud on which to change the SF color scale! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1419"/>
        <source>Missing parameter: boolean (whether to mix with existing colors or not) after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1433"/>
        <source>No point cloud on which to convert SF to RGB! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1447"/>
        <source>cmd.warning: cloud &apos;%1&apos; has no active scalar field (it will be ignored)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1461"/>
        <source>cmd.warning: cloud &apos;%1&apos; failed to convert SF to RGB</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1500"/>
        <source>Missing parameter: min value after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1520"/>
        <source>Failed to read a numerical parameter: min value (after &quot;-%1&quot;). Got &apos;%2&apos; instead.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1529"/>
        <source>Missing parameter: max value after &quot;-%1&quot; {min}</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1549"/>
        <source>Failed to read a numerical parameter: max value (after min value). Got &apos;%1&apos; instead.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1553"/>
        <source>	Interval: [%1 - %2]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1556"/>
        <source>No point cloud on which to filter SF! (be sure to open one or generate one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1605"/>
        <source>		Cloud &apos;%1&apos; --&gt; %2/%3 points remaining</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1611"/>
        <source>_FILTERED_[%1_%2]</source>
        <translation>_FILTERED _ [%1 _%2]</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1661"/>
        <source>Missing argument: filename after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1683"/>
        <source>Mesh &apos;%1&apos;</source>
        <translation>Mesh &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1686"/>
        <source> (#%2)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1689"/>
        <source>V = %2</source>
        <translation>V = %2</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1734"/>
        <source>Can&apos;t merge mesh &apos;%1&apos; (unhandled type)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1764"/>
        <location filename="../ecvCommandLineCommands.h" line="1817"/>
        <source>_MERGED</source>
        <translation>_MERGED</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1835"/>
        <source>Missing parameter: scalar field index after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1841"/>
        <source>Failed to read a numerical parameter: S.F. index (after &quot;-%1&quot;). Got &apos;%2&apos; instead.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1842"/>
        <source>Set active S.F. index: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1845"/>
        <source>No point cloud loaded! (be sure to open one with &quot;-%1 [cloud filename]&quot; before &quot;-%2&quot;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1854"/>
        <source>Cloud &apos;%1&apos; has less scalar fields than the index to select!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="1963"/>
        <source>Entity &apos;%1&apos; has been translated: (%2,%3,%4)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2012"/>
        <location filename="../ecvCommandLineCommands.h" line="2125"/>
        <location filename="../ecvCommandLineCommands.h" line="2196"/>
        <location filename="../ecvCommandLineCommands.h" line="3104"/>
        <source>No cloud available. Be sure to open one first!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2023"/>
        <source>Plane successfully fitted: rms = %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2040"/>
        <source>%1/%2_BEST_FIT_PLANE_INFO</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2042"/>
        <location filename="../ecvCommandLineCommands.h" line="3750"/>
        <source>_%1</source>
        <translation>_%1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2043"/>
        <location filename="../ecvCommandLineCommands.h" line="3751"/>
        <source>.txt</source>
        <translation>.txt</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2048"/>
        <source>Filename: %1</source>
        <translation>Filename: %1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2049"/>
        <source>Fitting RMS: %1</source>
        <translation>拟合均方根: %1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2056"/>
        <source>Normal: (%1,%2,%3)</source>
        <translation>Normal: (%1, %2, %3)</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2087"/>
        <source>Cloud &apos;%1&apos; has been transformed with the above matrix</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2088"/>
        <source>_HORIZ</source>
        <translation>_HORIZ</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2099"/>
        <source>Failed to compute best fit plane for cloud &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2116"/>
        <source>Missing parameter: number of neighbors after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2122"/>
        <location filename="../ecvCommandLineCommands.h" line="2186"/>
        <source>Invalid parameter: number of neighbors (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2147"/>
        <source>_NORMS_REORIENTED</source>
        <translation>_NORMS_REORIENTED</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2157"/>
        <source>Failed to orient the normals of cloud &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2180"/>
        <source>Missing parameter: number of neighbors mode after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2189"/>
        <source>Missing parameter: sigma multiplier after number of neighbors (SOR)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2193"/>
        <source>Invalid parameter: sigma multiplier (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2222"/>
        <source>.clean</source>
        <translation>. Clean</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2236"/>
        <source>_SOR</source>
        <translation>_SOR</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2242"/>
        <source>Not enough memory to create a clean version of cloud &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2251"/>
        <source>Failed to apply SOR filter on cloud &apos;%1&apos;! (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2275"/>
        <location filename="../ecvCommandLineCommands.h" line="2347"/>
        <location filename="../ecvCommandLineCommands.h" line="2779"/>
        <source>No mesh available. Be sure to open one first!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2291"/>
        <source>.vertices</source>
        <translation>.vertices</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2326"/>
        <source>Missing parameter: sampling mode after &quot;-%1&quot; (POINTS/DENSITY)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2337"/>
        <source>Invalid parameter: unknown sampling mode &quot;%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2340"/>
        <source>Missing parameter: value after sampling mode</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2344"/>
        <source>Invalid parameter: value after sampling mode</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2362"/>
        <source>Cloud sampling failed!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2366"/>
        <source>Sampled cloud created: %1 points</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2367"/>
        <source>_SAMPLED_POINTS</source>
        <translation>_SAMPLED_POINTS</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2397"/>
        <source>Missing parameter: box extents after &quot;-%1&quot; (Xmin:Ymin:Zmin:Xmax:Ymax:Zmax)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2399"/>
        <source>No point cloud or mesh available. Be sure to open or generate one first!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2407"/>
        <source>Invalid parameter: box extents (expected format is &apos;Xmin:Ymin:Zmin:Xmax:Ymax:Zmax&apos;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2416"/>
        <source>Invalid parameter: box extents (component #%1 is not a valid number)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2496"/>
        <source>Missing parameter after &quot;-%1&quot; (DIMENSION)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2498"/>
        <location filename="../ecvCommandLineCommands.h" line="2549"/>
        <location filename="../ecvCommandLineCommands.h" line="2769"/>
        <source>No point cloud available. Be sure to open or generate one first!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2511"/>
        <location filename="../ecvCommandLineCommands.h" line="2700"/>
        <source>Invalid parameter: dimension after &quot;-%1&quot; (expected: X, Y or Z)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2520"/>
        <source>_%1_TO_SF</source>
        <translation>_%1_TO_SF</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2530"/>
        <source>Failed to export coord. %1 to SF on cloud &apos;%2&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2547"/>
        <source>Missing parameter(s) after &quot;-%1&quot; (ORTHO_DIM N X1 Y1 X2 Y2 ... XN YN)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2566"/>
        <source>Invalid parameter: orthogonal dimension after &quot;-%1&quot; (expected: X, Y or Z)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2576"/>
        <source>Invalid parameter: number of vertices for the 2D polyline after &quot;-%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2591"/>
        <source>Missing parameter(s): vertex #%1 data and following</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2599"/>
        <source>Invalid parameter: X-coordinate of vertex #%1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2603"/>
        <source>Invalid parameter: Y-coordinate of vertex #%1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2644"/>
        <source>.cropped</source>
        <translation>.cropped</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2655"/>
        <source>Not enough memory to crop cloud &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2662"/>
        <source>No point of cloud &apos;%1&apos; falls inside the input box!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2667"/>
        <source>Crop process failed! (not enough memory)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2684"/>
        <source>Missing parameter(s) after &quot;-%1&quot; (DIM FREQUENCY)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2686"/>
        <source>No entity available. Be sure to open or generate one first!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2710"/>
        <source>Invalid parameter: frequency after &quot;-%1 DIM&quot; (in Hz, integer value)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2724"/>
        <location filename="../ecvCommandLineCommands.h" line="2748"/>
        <source>COLOR_BANDING_%1_%2</source>
        <translation>COLOR_BANDING_ %1 _ %2</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2743"/>
        <source>Vertices of mesh &apos;%1&apos; are locked (they may be shared by multiple entities for instance). Can&apos;t apply the current command on them.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2787"/>
        <source>Only one point cloud available. Be sure to open or generate a second one before performing C2C distance!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2866"/>
        <source>Invalid parameter: unknown model type &quot;%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2870"/>
        <source>Missing parameter: model type after &quot;-%1&quot; (LS/TRI/HF)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2881"/>
        <source>Invalid parameter: unknown neighborhood type &quot;%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2885"/>
        <source>Missing parameter: expected neighborhood type after model type (KNN/SPHERE)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2894"/>
        <source>Invalid parameter: neighborhood size</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2898"/>
        <source>Missing parameter: expected neighborhood size after neighborhood type (neighbor count/sphere radius)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2907"/>
        <location filename="../ecvCommandLineCommands.h" line="3618"/>
        <source>Missing parameter: max thread count after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2912"/>
        <location filename="../ecvCommandLineCommands.h" line="3623"/>
        <source>Invalid thread count! (after %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="2984"/>
        <source>_MAX_DIST_%1</source>
        <translation>_MAX_DIST_ %1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3023"/>
        <source>Missing parameter: distribution type after &quot;-%1&quot; (GAUSS/WEIBULL)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3030"/>
        <source>Missing parameter: mean value after &quot;GAUSS&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3034"/>
        <source>Invalid parameter: mean value after &quot;GAUSS&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3037"/>
        <source>Missing parameter: sigma value after &quot;GAUSS&quot; {mu}</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3041"/>
        <source>Invalid parameter: sigma value after &quot;GAUSS&quot; {mu}</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3051"/>
        <source>Missing parameter: a value after &quot;WEIBULL&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3055"/>
        <source>Invalid parameter: a value after &quot;WEIBULL&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3058"/>
        <source>Missing parameter: b value after &quot;WEIBULL&quot; {a}</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3062"/>
        <source>Invalid parameter: b value after &quot;WEIBULL&quot; {a}</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3065"/>
        <source>Missing parameter: shift value after &quot;WEIBULL&quot; {a} {b}</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3069"/>
        <source>Invalid parameter: shift value after &quot;WEIBULL&quot; {a} {b}</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3077"/>
        <source>Invalid parameter: unknown distribution &quot;%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3085"/>
        <source>Missing parameter: p-value after distribution</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3089"/>
        <source>Invalid parameter: p-value after distribution</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3096"/>
        <source>Missing parameter: neighbors after p-value</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3100"/>
        <source>Invalid parameter: neighbors after p-value</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3144"/>
        <source>Couldn&apos;t compute octree for cloud &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3151"/>
        <source>[Chi2 Test] %1 test result = %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3167"/>
        <source>_STAT_TEST_%1</source>
        <translation>_STAT_TEST_ %1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3219"/>
        <source>Missing parameter: max edge length value after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3223"/>
        <source>Invalid value for max edge length! (after %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3224"/>
        <source>Max edge length: %1</source>
        <translation>最大边缘长度:%1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3232"/>
        <source>Axis aligned: %1</source>
        <translation>轴对齐:%1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3249"/>
        <source>	Resulting mesh: #%1 faces, %2 vertices</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3293"/>
        <source>Missing parameter(s): SF index and/or operation after &apos;%1&apos; (2 values expected)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3305"/>
        <location filename="../ecvCommandLineCommands.h" line="3394"/>
        <location filename="../ecvCommandLineCommands.h" line="3589"/>
        <location filename="../ecvCommandLineCommands.h" line="3609"/>
        <source>Invalid SF index! (after %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3314"/>
        <location filename="../ecvCommandLineCommands.h" line="3404"/>
        <source>Unknown operation! (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3318"/>
        <location filename="../ecvCommandLineCommands.h" line="3408"/>
        <source>Operation %1 can&apos;t be applied with %2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3330"/>
        <location filename="../ecvCommandLineCommands.h" line="3437"/>
        <source>Failed top apply operation on cloud &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3353"/>
        <location filename="../ecvCommandLineCommands.h" line="3460"/>
        <source>Failed top apply operation on mesh &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3380"/>
        <source>Missing parameter(s): SF index and/or operation and/or scalar value after &apos;%1&apos; (3 values expected)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3419"/>
        <source>Invalid scalar value! (after %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3528"/>
        <source>Missing parameter: min error difference after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3532"/>
        <source>Invalid value for min. error difference! (after %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3540"/>
        <source>Missing parameter: number of iterations after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3545"/>
        <source>Invalid number of iterations! (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3553"/>
        <source>Missing parameter: overlap percentage after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3558"/>
        <source>Invalid overlap value! (%1 --&gt; should be between 10 and 100)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3566"/>
        <source>Missing parameter: random sampling limit value after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3570"/>
        <source>Invalid random sampling limit! (after %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3578"/>
        <location filename="../ecvCommandLineCommands.h" line="3598"/>
        <source>Missing parameter: SF index after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3644"/>
        <source>Invalid parameter: unknown rotation filter &quot;%1&quot;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3648"/>
        <source>Missing parameter: rotation filter after &quot;-%1&quot; (XYZ/X/Y/Z/NONE)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3690"/>
        <source>Invalid SF index for data entity! (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3696"/>
        <source>[ICP] SF #%1 (data entity) will be used as weights</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3706"/>
        <source>Invalid SF index for model entity! (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3712"/>
        <source>[ICP] SF #%1 (model entity) will be used as weights</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3742"/>
        <source>Entity &apos;%1&apos; has been registered</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3743"/>
        <source>RMS: %1</source>
        <translation>均方根: %1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3744"/>
        <source>Number of points used for final step: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3748"/>
        <source>%1/%2_REGISTRATION_MATRIX</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3759"/>
        <source>_REGISTERED</source>
        <translation>_REGISTERED</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3783"/>
        <source>Missing parameter: FBX format (string) after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3786"/>
        <source>FBX format: %1</source>
        <translation>FBX Format: %1</translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3803"/>
        <source>Missing parameter: format (ASCII, BINARY_LE, or BINARY_BE) after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="3818"/>
        <source>Invalid PLY format! (&apos;%1&apos;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="4020"/>
        <source>Missing parameter: option after &apos;%1&apos; (%2/%3)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="4035"/>
        <source>Unrecognized option after &apos;%1&apos; (%2 or %3 expected)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecvCommandLineCommands.h" line="4049"/>
        <source>Missing parameter: filename after &apos;%1&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ecv2.5DimEditor.cpp" line="132"/>
        <source>invalid grid box</source>
        <translation>无效的网格框</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="73"/>
        <source>None</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="76"/>
        <source>RGB</source>
        <translation>RGB</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="77"/>
        <source>Scalar field</source>
        <translation>标量字段</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="80"/>
        <source>Default</source>
        <translation>默认的</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="81"/>
        <source>Default Width</source>
        <translation>默认的宽度</translation>
    </message>
    <message>
        <location filename="../../common/CommonSettings.h" line="51"/>
        <source>ACloudViewer</source>
        <translation>ACloudViewer</translation>
    </message>
</context>
<context>
    <name>QUIWidget</name>
    <message>
        <location filename="../ecvUIManager.cpp" line="1318"/>
        <source>restore(&amp;R)</source>
        <translation>恢复(&amp;R)</translation>
    </message>
    <message>
        <location filename="../ecvUIManager.cpp" line="1319"/>
        <source>exit(&amp;Q)</source>
        <translation>退出(&amp;Q)</translation>
    </message>
</context>
<context>
    <name>QuadricWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/quadricwindow.ui" line="14"/>
        <source>Form</source>
        <translation>二次曲面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/quadricwindow.ui" line="30"/>
        <source>Config</source>
        <translation>配置</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/quadricwindow.ui" line="55"/>
        <source>Formula</source>
        <translation>公式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/quadricwindow.ui" line="80"/>
        <source>Data Table</source>
        <translation>数据表</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/quadricwindow.ui" line="103"/>
        <source>Rows</source>
        <translation>行</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/quadricwindow.ui" line="131"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
</context>
<context>
    <name>RansacSDDialog</name>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="14"/>
        <source>Ransac Shape Detection</source>
        <translation>随机采样一致形状检测</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="24"/>
        <source>Primitives</source>
        <translation>基础模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="30"/>
        <source>Plane</source>
        <translation>平面</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="40"/>
        <source>Sphere</source>
        <translation>球体</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="50"/>
        <source>Cylinder</source>
        <translation>圆柱体</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="60"/>
        <source>Cone</source>
        <translation>类锥体</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="70"/>
        <source>Torus</source>
        <translation>类圆环体</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="83"/>
        <source>Advanced parameters</source>
        <translation>高级参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="91"/>
        <source>max distance to primitive</source>
        <translation>距基础模型最大距离</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="98"/>
        <source>Maximum distance of samples to the ideal shape</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="124"/>
        <source>sampling resolution</source>
        <translation>采样分辨率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="131"/>
        <source>Should correspond to the distance between neighboring points in the data</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="157"/>
        <source>max normal deviation</source>
        <translation>最大法线偏差</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="164"/>
        <source>Maximum deviation from the ideal shape normal vector (in degrees)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="190"/>
        <source>overlooking probability</source>
        <translation>总概率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="197"/>
        <source>Probability that no better candidate was overlooked during sampling (the lower the better!)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="223"/>
        <source>Min support points per primitive</source>
        <translation>每个基础模型的最少支持点数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="230"/>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="233"/>
        <location filename="../../plugins/core/qRANSAC_SD/ransacSDDlg.ui" line="236"/>
        <source>This is the minimal number of points required for a primitive</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>RasterExportOptionsDialog</name>
    <message>
        <location filename="../ui_templates/rasterExportOptionsDlg.ui" line="14"/>
        <source>Raster export options</source>
        <translation>光栅导出选项</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterExportOptionsDlg.ui" line="35"/>
        <source>Raster dimensions:</source>
        <translation>光栅尺寸:</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterExportOptionsDlg.ui" line="64"/>
        <source>Export RGB colors</source>
        <translation>导出RGB颜色</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterExportOptionsDlg.ui" line="71"/>
        <source>Export heights</source>
        <translation>导出的高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterExportOptionsDlg.ui" line="81"/>
        <source>Export active layer</source>
        <translation>导出活跃层</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterExportOptionsDlg.ui" line="88"/>
        <source>Export all scalar fields</source>
        <translation>导出多有标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterExportOptionsDlg.ui" line="95"/>
        <source>Export density (population per cell)</source>
        <translation>导出密度（每单元量）</translation>
    </message>
</context>
<context>
    <name>RasterizeToolDialog</name>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="14"/>
        <source>Rasterize</source>
        <translation>栅格化</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="46"/>
        <source>Grid</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="55"/>
        <source>size</source>
        <translation>大小</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="62"/>
        <source>step</source>
        <translation>一步</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="78"/>
        <source>size of step of the grid generated (in the same units as the coordinates of the point cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="100"/>
        <source>Edit grid</source>
        <translation>编辑网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="109"/>
        <source>Active layer (or &apos;scalar field&apos;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="112"/>
        <source>active layer</source>
        <translation>活性层</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="122"/>
        <source>range</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="139"/>
        <source>Projection</source>
        <translation>投影</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="145"/>
        <source>SF interpolation method</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="152"/>
        <source>minimum value</source>
        <translation>最小值</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="157"/>
        <source>average value</source>
        <translation>平均值</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="162"/>
        <source>maximum value</source>
        <translation>最大值</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="170"/>
        <source>cell height</source>
        <translation>细胞高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="177"/>
        <source>direction</source>
        <translation>方向</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="190"/>
        <source>Use the nearest point of the input cloud in each cell instead of the cell center</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="193"/>
        <source>resample input cloud</source>
        <translation>重新取样输入云</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="225"/>
        <source>Per-cell height computation method:
 - minimum = lowest point in the cell
 - average = mean height of all points inside the cell
 - maximum = highest point in the cell</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="235"/>
        <source>minimum</source>
        <translation>最低</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="240"/>
        <source>average</source>
        <translation>平均</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="245"/>
        <source>maximum</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="259"/>
        <source>Interpolate scalar field(s)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="262"/>
        <source>interpolate SF(s)</source>
        <translation>插入科幻(s)</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="278"/>
        <source>Warning: the original point&apos;s height will be
replaced by the cell&apos;s average height!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="292"/>
        <source>Empty cells</source>
        <translation>空的细胞</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="304"/>
        <source>Fill with</source>
        <translation>充满</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="317"/>
        <source>choose the value to fill the cells in which no point is projected : minimum value over the whole point cloud or average value (over the whole cloud also)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="321"/>
        <source>leave empty</source>
        <translation>离开空</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="326"/>
        <source>minimum height</source>
        <translation>最小的高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="331"/>
        <location filename="../ui_templates/rasterizeDlg.ui" line="526"/>
        <source>average height</source>
        <translation>平均身高</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="336"/>
        <source>maximum height</source>
        <translation>最大高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="341"/>
        <source>user specified value</source>
        <translation>用户指定的值</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="346"/>
        <source>interpolate</source>
        <translation>插入</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="363"/>
        <source>Custom value for empty cells</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="394"/>
        <source>Update grid</source>
        <translation>更新网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="427"/>
        <location filename="../ui_templates/rasterizeDlg.ui" line="800"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="444"/>
        <source>Export grid as a point cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="447"/>
        <source>Cloud</source>
        <translation>云</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="454"/>
        <source>Export grid as a mesh</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="457"/>
        <source>Mesh</source>
        <translation>网</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="466"/>
        <source>Export per-cell statistics as SF(s):</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="503"/>
        <source>Adds a scalar field with the grid density (= number of points inside each cell)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="506"/>
        <source>population</source>
        <translation>人口</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="513"/>
        <source>Adds a scalar field with the min. height of the points inside the cell</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="516"/>
        <source>min height</source>
        <translation>最小的高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="523"/>
        <source>Adds a scalar field with the average height of the points inside the cell</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="533"/>
        <source>Adds a scalar field with the max. height of the points inside the cell</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="536"/>
        <source>max height</source>
        <translation>最大高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="543"/>
        <location filename="../ui_templates/rasterizeDlg.ui" line="553"/>
        <source>Adds a scalar field with the standard deviation of the heights of the points inside the cell</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="546"/>
        <source>height std. dev.</source>
        <translation>std. dev高度。</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="556"/>
        <source>height range</source>
        <translation>高度范围</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="588"/>
        <source>Export grid as a raster (geotiff)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="591"/>
        <source>Raster</source>
        <translation>光栅</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="598"/>
        <source>Export grid as an image</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="601"/>
        <source>Image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="608"/>
        <source>Export grid as a matrix (text file)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="611"/>
        <source>Matrix</source>
        <translation>矩阵</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="621"/>
        <source>Contour plot</source>
        <translation>等高线图</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="644"/>
        <source>The contour plot is computed on the active layer</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="653"/>
        <source>Start value</source>
        <translation>开始值</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="673"/>
        <source>Step</source>
        <translation>一步</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="693"/>
        <source>Min. vertex count</source>
        <translation>分钟顶点数</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="700"/>
        <source>Min vertex count per iso-line (to ignore the smallest ones)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="713"/>
        <source>Line width</source>
        <translation>线宽</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="720"/>
        <source>Default contour lines width</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="737"/>
        <source>colorize</source>
        <translation>彩色化</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="744"/>
        <source>ignore borders</source>
        <translation>忽略边界</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="772"/>
        <source>project contours on the altitude layer</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="784"/>
        <source>Remove all contour lines</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="787"/>
        <source>Clear</source>
        <translation>清空</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="797"/>
        <source>Export contour lines to the DB tree</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="807"/>
        <location filename="../ui_templates/rasterizeDlg.ui" line="910"/>
        <source>Generate</source>
        <translation>生成</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="817"/>
        <source>Hillshade</source>
        <translation>Hillshade</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="834"/>
        <source>Hillshade is computed on the height layer</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="843"/>
        <source>Sun zenith</source>
        <translation>太阳天顶</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="850"/>
        <source>Zenith angle (in degrees) = 90 - altitude angle</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="853"/>
        <location filename="../ui_templates/rasterizeDlg.ui" line="879"/>
        <source> deg.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="869"/>
        <source>Sun azimuth</source>
        <translation>太阳方位</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="876"/>
        <source>Azimuth angle (in degrees)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="918"/>
        <location filename="../ui_templates/rasterizeDlg.ui" line="927"/>
        <source>Volume</source>
        <translation>体积</translation>
    </message>
    <message>
        <location filename="../ui_templates/rasterizeDlg.ui" line="941"/>
        <source>Non empty cells</source>
        <translation>非空单元格</translation>
    </message>
</context>
<context>
    <name>RegionGrowingSegmentation</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/RegionGrowingSegmentation.cpp" line="46"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/RegionGrowingSegmentation.cpp" line="47"/>
        <source>Region Growing Segmentation</source>
        <translation>区域生长分割</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/RegionGrowingSegmentation.cpp" line="48"/>
        <source>Region Growing Segmentation from clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/RegionGrowingSegmentation.cpp" line="221"/>
        <source>Selected entity does not have any suitable scalar field or RGB</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/RegionGrowingSegmentation.cpp" line="223"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/RegionGrowingSegmentation.cpp" line="225"/>
        <source>Region Growing Segmentation does not returned any point. Try relaxing your parameters</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>RegionGrowingSegmentationDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="20"/>
        <source>Region Growing Segmentation</source>
        <translation>区域生长分割</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="43"/>
        <source>Basic</source>
        <translation>基本的</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="57"/>
        <source>1.0</source>
        <translation>1.0</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="64"/>
        <source>K Search</source>
        <translation>K近邻搜索数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="71"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="244"/>
        <source>Min Cluster Size</source>
        <translation>最小簇尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="84"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="150"/>
        <source>50</source>
        <translation>50</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="91"/>
        <source>Max Cluster Size</source>
        <translation>最大簇尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="104"/>
        <source>1000000</source>
        <translation>1000000</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="117"/>
        <source>30</source>
        <translation>30</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="124"/>
        <source>Smoothness Theta</source>
        <translation>平滑θ</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="137"/>
        <source>0.052359878</source>
        <translation>0.052359878</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="160"/>
        <source>Neighbour Num</source>
        <translation>近邻数量</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="167"/>
        <source>Curvature</source>
        <translation>曲率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="190"/>
        <source>Color-Based</source>
        <translation>基于颜色的</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="204"/>
        <source>100</source>
        <translation>100</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="217"/>
        <source>6</source>
        <translation>6</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="224"/>
        <source>Region Color Diff</source>
        <translation>区域颜色差异</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="237"/>
        <source>5</source>
        <translation>5</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="257"/>
        <source>10</source>
        <translation>10</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="264"/>
        <source>Neighbours Dist</source>
        <translation>近邻距离</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/RegionGrowingSegmentationDlg.ui" line="271"/>
        <source>Point Color Diff</source>
        <translation>点颜色差异</translation>
    </message>
</context>
<context>
    <name>RegistrationDialog</name>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="14"/>
        <source>Clouds registration</source>
        <translation>点云配准</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="26"/>
        <source>Role assignation</source>
        <translation>角色分配</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="32"/>
        <source>&apos;data&apos; entity</source>
        <translation>“数据”的实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="35"/>
        <source>aligned</source>
        <translation>待配准实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="42"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;entity to align (will be displaced)&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="45"/>
        <source>the data cloud is the entity to align with the model cloud : it will be displaced (red cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="69"/>
        <source>&apos;model&apos; entity</source>
        <translation>“模型”的实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="72"/>
        <source>reference</source>
        <translation>参考实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="79"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;reference entity (won&apos;t move)&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="82"/>
        <source>the model cloud is the reference : it won&apos;t move (yellow cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="92"/>
        <location filename="../ui_templates/registrationDlg.ui" line="95"/>
        <source>press once to exchange model and data clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="98"/>
        <source>swap</source>
        <translation>交换</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="115"/>
        <source>Parameters</source>
        <translation>参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="123"/>
        <location filename="../ui_templates/registrationDlg.ui" line="126"/>
        <source>By choosing this criterion, you can control the computation time.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="129"/>
        <source>Number of iterations</source>
        <translation>迭代次数</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="139"/>
        <source>Set the maximal number of step for the algorithm regsitration computation .</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="158"/>
        <source>By choosing this criterion, you can control the quality of the result.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="161"/>
        <source>RMS difference</source>
        <translation>均方根误差</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="177"/>
        <source>Set the minimum RMS improvement between 2 consecutive iterations (below which the registration process will stop).</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="193"/>
        <location filename="../ui_templates/registrationDlg.ui" line="206"/>
        <source>Rough estimation of the final overlap ratio of the data cloud (the smaller, the better the initial registration should be!)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="199"/>
        <source>Final overlap</source>
        <translation>最终重叠</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="249"/>
        <source>Whether to adjust the scale of the &apos;data&apos; entity</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="252"/>
        <source>adjust scale</source>
        <translation>调整比例</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="282"/>
        <source>max thread count</source>
        <translation>最大线程数</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="289"/>
        <source>Maximum number of threads/cores to be used
(CC or your computer might not respond for a while if you use all available cores)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="319"/>
        <source>Research</source>
        <translation>研究</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="327"/>
        <source>Random sampling limit</source>
        <translation>随机采样限制</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="334"/>
        <source>Above this limit, clouds are randomly resampled at each iteration</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="359"/>
        <source>Rotation</source>
        <translation>旋转</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="408"/>
        <source>Translation</source>
        <translation>平移</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="485"/>
        <location filename="../ui_templates/registrationDlg.ui" line="488"/>
        <source>Chose this option to remove points that are likely to disturb the registration during the computation.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="491"/>
        <source>Enable farthest points removal</source>
        <translation>允许删除最远点</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="498"/>
        <location filename="../ui_templates/registrationDlg.ui" line="508"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Use the displayed scalar field as weights (the bigger its associated scalar value/weight is, the more influence the point will have).&lt;/p&gt;&lt;p&gt;Note that only absolute distances are considered (i.e. minimal weight is 0).&lt;/p&gt;&lt;p&gt;Weights are automatically normalized.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="501"/>
        <source>Data: use displayed S.F. as weights</source>
        <translation>数据：使用显示标量字段最为权重</translation>
    </message>
    <message>
        <location filename="../ui_templates/registrationDlg.ui" line="511"/>
        <source>Model: use displayed S.F. as weights (only for clouds)</source>
        <translation>模型：使用显示的标量字段作为权重（仅仅点云）</translation>
    </message>
</context>
<context>
    <name>RenderSurface</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="14"/>
        <source>Form</source>
        <translation>曲面渲染窗口</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="28"/>
        <source>Config</source>
        <translation>配置</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="54"/>
        <source>Cube Axes</source>
        <translation>立方体轴</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="68"/>
        <source>Show Z Axis</source>
        <translation>显示Z轴</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="78"/>
        <source>Fly Mode</source>
        <translation>飞行模式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="85"/>
        <source>Color</source>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="106"/>
        <source>Number of Labels</source>
        <translation>标签数量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="114"/>
        <source>Outer Edge</source>
        <translation>外缘</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="119"/>
        <source>Closest Triad</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="124"/>
        <source>None</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="132"/>
        <source>Show X Axis</source>
        <translation>显示X轴</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="142"/>
        <source>Show Y Axis</source>
        <translation>显示Y轴</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="152"/>
        <source>X Label</source>
        <translation>X标签</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="159"/>
        <source>Y Label</source>
        <translation>And Label</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="166"/>
        <source>Z Label</source>
        <translation>With Label</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="178"/>
        <source>Contour</source>
        <translation>轮廓</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="192"/>
        <source>Show Labels</source>
        <translation>显示标签</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="199"/>
        <source>Lines Color</source>
        <translation>线的颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="220"/>
        <source>Labels Color</source>
        <translation>标签的颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="227"/>
        <source>Number of Contours</source>
        <translation>轮廓数量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="237"/>
        <source>Show Contour Lines</source>
        <translation>显示轮廓线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="249"/>
        <source>Outline</source>
        <translation>大纲</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="270"/>
        <source>Outline Color</source>
        <translation>轮廓的颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="300"/>
        <source>Opacity</source>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="307"/>
        <source>Generate Faces</source>
        <translation>生成面片</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="314"/>
        <source>Line Width</source>
        <translation>线宽</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="329"/>
        <source>Scalar Bar</source>
        <translation>标量条</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="343"/>
        <source>Show Color Bar</source>
        <translation>显示颜色栏</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="353"/>
        <source>Number of Colors</source>
        <translation>数量的颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="360"/>
        <source>Show Tick Labels</source>
        <translation>显示标记标签</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="374"/>
        <source>Show Frame</source>
        <translation>显示帧</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="381"/>
        <source>Orientation</source>
        <translation>方向</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="389"/>
        <source>Horizontal</source>
        <translation>水平</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="394"/>
        <source>Vertical</source>
        <translation>垂直</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="402"/>
        <source>Frame Color</source>
        <translation>框架颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="409"/>
        <source>Frame Width</source>
        <translation>帧宽度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="431"/>
        <source>General</source>
        <translation>一般</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="442"/>
        <source>Gradient</source>
        <translation>梯度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="456"/>
        <source>Sources</source>
        <translation>来源</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="463"/>
        <source>Bg Color:</source>
        <translation>背景颜色:</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="470"/>
        <source>Edge Color</source>
        <translation>边缘的颜色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="477"/>
        <source>View Fit</source>
        <translation>适应视图</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="484"/>
        <source>PushButton</source>
        <translation>按钮</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="491"/>
        <source>Show Edge</source>
        <translation>显示边</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="498"/>
        <source>Show Orientation Marker</source>
        <translation>显示方向标记</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="513"/>
        <source>Table</source>
        <translation>表格</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="521"/>
        <source>Rows</source>
        <translation>行</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="551"/>
        <source>Random</source>
        <translation>随机</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="558"/>
        <source>Load File</source>
        <translation>加载文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="598"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.ui" line="612"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.cpp" line="63"/>
        <source>Render Surface</source>
        <translation>使表面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.cpp" line="820"/>
        <source>random</source>
        <translation>随机</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.cpp" line="834"/>
        <source>Load Data File</source>
        <translation>加载数据文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/surface/rendersurface.cpp" line="838"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
</context>
<context>
    <name>RenderToFileDialog</name>
    <message>
        <location filename="../ui_templates/renderToFileDialog.ui" line="14"/>
        <source>Render to file</source>
        <translation>渲染文件</translation>
    </message>
    <message>
        <location filename="../ui_templates/renderToFileDialog.ui" line="22"/>
        <source>File Name</source>
        <translation>文件名称</translation>
    </message>
    <message>
        <location filename="../ui_templates/renderToFileDialog.ui" line="43"/>
        <source>Zoom</source>
        <translation>缩放比</translation>
    </message>
    <message>
        <location filename="../ui_templates/renderToFileDialog.ui" line="60"/>
        <source>Result:</source>
        <translation>结果:</translation>
    </message>
    <message>
        <location filename="../ui_templates/renderToFileDialog.ui" line="89"/>
        <source>Don&apos;t scale features (e.g. points size, lines thickness, text, etc.)</source>
        <translation>不缩放特征（例如：点大小，线粗，文本等）</translation>
    </message>
    <message>
        <location filename="../ui_templates/renderToFileDialog.ui" line="96"/>
        <source>Render overlay items (scale, trihedron, etc.)</source>
        <translation>渲染重叠项（缩放比例，三面体等）</translation>
    </message>
</context>
<context>
    <name>RoleChoiceDialog</name>
    <message>
        <location filename="../ui_templates/roleChoiceDlg.ui" line="14"/>
        <source>Choose role</source>
        <translation>选择对象</translation>
    </message>
    <message>
        <location filename="../ui_templates/roleChoiceDlg.ui" line="29"/>
        <source>Compared</source>
        <translation>比较</translation>
    </message>
    <message>
        <location filename="../ui_templates/roleChoiceDlg.ui" line="50"/>
        <source>Reference</source>
        <translation>参考</translation>
    </message>
    <message>
        <location filename="../ui_templates/roleChoiceDlg.ui" line="94"/>
        <source>Swap</source>
        <translation>交换</translation>
    </message>
</context>
<context>
    <name>SACSegmentation</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="42"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="43"/>
        <source>SAC Segmentation</source>
        <translation>随机采样一致性分割</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="44"/>
        <source>SAC Segmentation from clouds</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="89"/>
        <source>SAC_RANSAC</source>
        <translation>随机采样一致</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="90"/>
        <source>SAC_LMEDS</source>
        <translation>最小中位数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="91"/>
        <source>SAC_MSAC</source>
        <translation>移动评估</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="92"/>
        <source>SAC_RRANSAC</source>
        <translation>随机的随机采样一致</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="93"/>
        <source>SAC_RMSAC</source>
        <translation>随机移动评估</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="94"/>
        <source>SAC_MLESAC</source>
        <translation>最大似然估计</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="95"/>
        <source>SAC_PROSAC</source>
        <translation>SAC_PROSAC</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="100"/>
        <source>SACMODEL_PLANE</source>
        <translation>平面模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="101"/>
        <source>SACMODEL_LINE</source>
        <translation>线模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="102"/>
        <source>SACMODEL_CIRCLE2D</source>
        <translation>2D圆模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="103"/>
        <source>SACMODEL_CIRCLE3D</source>
        <translation>3D圆模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="104"/>
        <source>SACMODEL_SPHERE</source>
        <translation>球体模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="105"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="106"/>
        <source>SACMODEL_CYLINDER</source>
        <translation>圆柱体模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="107"/>
        <source>SACMODEL_CONE</source>
        <translation>类圆锥体模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="108"/>
        <source>SACMODEL_TORUS</source>
        <translation>类圆环体模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="109"/>
        <source>SACMODEL_PARALLEL_LINE</source>
        <translation>单平行线模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="110"/>
        <source>SACMODEL_PERPENDICULAR_PLANE</source>
        <translation>垂直面模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="111"/>
        <source>SACMODEL_PARALLEL_LINES</source>
        <translation>多平行线模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="112"/>
        <source>SACMODEL_NORMAL_PLANE</source>
        <translation>法线平面模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="113"/>
        <source>SACMODEL_NORMAL_SPHERE</source>
        <translation>法线球体模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="114"/>
        <source>SACMODEL_REGISTRATION</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="115"/>
        <source>SACMODEL_REGISTRATION_2D</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="116"/>
        <source>SACMODEL_PARALLEL_PLANE</source>
        <translation>平行平面模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="117"/>
        <source>SACMODEL_NORMAL_PARALLEL_PLANE</source>
        <translation>法线平行面模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="118"/>
        <source>SACMODEL_STICK</source>
        <translation>条状模型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="207"/>
        <source>-MaxRemainingRatio[%1]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="212"/>
        <source>-SingleSegmentation[Distance Threshold %1]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="217"/>
        <source>[SACSegmentation::compute] %1 extracted segment(s) where created from cloud &apos;%2&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="223"/>
        <source>Error(s) occurred during the generation of segments! Result may be incomplete</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="265"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="283"/>
        <source>PCLUtils::GetSACSegmentation failed</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="273"/>
        <source>SAC Segmentation could not estimate a planar model from the remaining cloud.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="306"/>
        <source>must select one of &apos;Export Extraction&apos; and &apos;exportRemaining&apos; or both</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="308"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="310"/>
        <source>SAC Segmentation could not estimate a planar model for the given dataset. Try relaxing your parameters</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/SACSegmentation.cpp" line="312"/>
        <source>An error occurred during the generation of segments!</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>SACSegmentationDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="20"/>
        <source>SAC Segmentation</source>
        <translation>随机采样一致性分割</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="30"/>
        <source>SAC Segmentation Parameters</source>
        <translation>随机采样一致分割参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="42"/>
        <source>Model Type</source>
        <translation>模型类型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="52"/>
        <source>Minimum Radius</source>
        <translation>最小半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="78"/>
        <source>Method Type</source>
        <translation>方法类型</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="88"/>
        <source>Maximum Radius</source>
        <translation>最大半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="114"/>
        <source>Distance Threshold</source>
        <translation>距离阈值</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="140"/>
        <source>Maximum Iterations</source>
        <translation>最大迭代数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="179"/>
        <source>Voxel Grid [Leaf Size]</source>
        <translation>体素滤波 [叶子尺寸]</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="208"/>
        <source>Probability</source>
        <translation>概率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="217"/>
        <source>Export Parameters</source>
        <translation>导出参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="229"/>
        <source>Export Extraction</source>
        <translation>导出抽取部分</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="239"/>
        <source>Export Remaining</source>
        <translation>导出剩余部分</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Segmentations/dialogs/SACSegmentationDlg.ui" line="252"/>
        <source>Recursive Extraction [ratio]</source>
        <translation>循环抽取剩余比例</translation>
    </message>
</context>
<context>
    <name>SFArithmeticsDlg</name>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="14"/>
        <source>Scalar fields arithmetics</source>
        <translation>标量字段计算</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="28"/>
        <source>SF 1</source>
        <translation>标量字段 1</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="51"/>
        <source>operation</source>
        <translation>操作</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="65"/>
        <source>plus</source>
        <translation>+</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="70"/>
        <source>minus</source>
        <translation>-</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="75"/>
        <source>multiply</source>
        <translation>*</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="80"/>
        <source>divided by</source>
        <translation>/</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="85"/>
        <source>square root</source>
        <translation>平方根</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="90"/>
        <source>power of 2</source>
        <translation>2的幂</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="95"/>
        <source>power of 3</source>
        <translation>3的幂</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="145"/>
        <source>integer part</source>
        <translation>整数部分</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="150"/>
        <source>inverse (1/x)</source>
        <translation>逆(1 / x)</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="164"/>
        <source>SF 2</source>
        <translation>标量字段 2</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="202"/>
        <source>Update the first scalar field directly (instead of creating a new SF)</source>
        <translation>直接更新第一标量字段（非创建一个新的标量字段）</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfArithmeticsDlg.ui" line="205"/>
        <source>Update SF1 directly</source>
        <translation>直接更新标量字段1</translation>
    </message>
</context>
<context>
    <name>SFEditDlg</name>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="17"/>
        <source>SF Values</source>
        <translation>标量字段值</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="42"/>
        <source>Display ranges</source>
        <translation>显示范围</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="102"/>
        <source>displayed</source>
        <translation>显示</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="155"/>
        <source>saturation</source>
        <translation>饱和</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="184"/>
        <source>Parameters</source>
        <translation>参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="190"/>
        <source>hidden otherwise...</source>
        <translation>隐藏,否则……</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="193"/>
        <source>show NaN/out of range values in grey</source>
        <translation>显示非法值/超出灰度范围值</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="203"/>
        <source>always show 0 in color scale</source>
        <translation>在色标里总显示0</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="210"/>
        <source>symmetrical color scale</source>
        <translation>对称色标</translation>
    </message>
    <message>
        <location filename="../ui_templates/sfEditDlg.ui" line="217"/>
        <source>log scale</source>
        <translation>对数尺度</translation>
    </message>
</context>
<context>
    <name>SIFTExtractDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/SIFTExtractDlg.ui" line="20"/>
        <source>SIFT Extraction</source>
        <translation>SIFT特征点抽取</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/SIFTExtractDlg.ui" line="32"/>
        <source>Intensity Field</source>
        <translation>强度字段</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/SIFTExtractDlg.ui" line="42"/>
        <source>Scales per Octave</source>
        <translation>每八度的尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/SIFTExtractDlg.ui" line="59"/>
        <source>Minimum Scale</source>
        <translation>最小比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/SIFTExtractDlg.ui" line="82"/>
        <source>Number of Octaves</source>
        <translation>八度数量</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/SIFTExtractDlg.ui" line="102"/>
        <source>Minimum Contrast</source>
        <translation>最小对比度</translation>
    </message>
</context>
<context>
    <name>SaveLASFileDialog</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="14"/>
        <source>LAS/LAZ scale</source>
        <translation>LAS / LAZ 尺度</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="22"/>
        <source>Choose the output LAS/LAZ scale/resolution:</source>
        <translation>选择输出LAS/LAZ 比例和分辨率：</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="46"/>
        <source>Highest resolution</source>
        <translation>最高分辨率</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="80"/>
        <source>Ensures optimal accuracy (up to 10^-7 absolute)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="83"/>
        <source>will decrease LAZ compression efficiency</source>
        <translation>LAZ压缩效率将会降低</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="95"/>
        <source>Original resolution</source>
        <translation>原始分辨率</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="108"/>
        <source>(0,0,0)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="135"/>
        <source>might not preserve data accuracy
(especially if you have transformed the original data)</source>
        <translation>可能不会保留数据准确度
尤其您已经空间转换了原始数据</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="148"/>
        <source>Custom resolution</source>
        <translation>自定义分辨率</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="197"/>
        <source>bigger scale = best LAZ compression = lower resolution(*)</source>
        <translation>最大比例 = 最佳LAZ压缩 = 较低分辨率（*）</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="215"/>
        <source>(*) careful, if chosen too low coordinates will loose accuracy</source>
        <translation>（*）注意，选择太小的点云坐标精度会受损</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveLASFileDlg.ui" line="225"/>
        <source>Save additional field(s)</source>
        <translation>保存额外字段</translation>
    </message>
</context>
<context>
    <name>SaveSHPFileDlg</name>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveSHPFileDlg.ui" line="14"/>
        <source>Save SHP file</source>
        <translation>保存SHP文件</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveSHPFileDlg.ui" line="20"/>
        <source>3D polylines</source>
        <translation>3D多段线</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveSHPFileDlg.ui" line="28"/>
        <source>Vertical dimension</source>
        <translation>垂直维度</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveSHPFileDlg.ui" line="62"/>
        <source>Save the 3D polylines as 2D ones (make sure to set the right &apos;vertical dimension&apos;)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveSHPFileDlg.ui" line="65"/>
        <source>save as 2D polylines</source>
        <translation>保存为2D折线</translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveSHPFileDlg.ui" line="72"/>
        <source>The height of each polyline (considered as constant!) will be saved as a field in the associated DBF file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../libs/eCV_io/ui_templates/saveSHPFileDlg.ui" line="75"/>
        <source>save (constant) height of polylines as a DBF field</source>
        <translation>保存（常量）多段线高度作为DBF字段</translation>
    </message>
</context>
<context>
    <name>ScaleDialog</name>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="14"/>
        <source>Scale / multiply</source>
        <translation>缩放比例/相乘</translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="22"/>
        <source>Scale(x)</source>
        <translation>缩放比例(x)</translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="55"/>
        <source>Scale(y)</source>
        <translation>缩放比例(y)</translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="88"/>
        <source>Scale(z)</source>
        <translation>缩放比例(z)</translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="110"/>
        <location filename="../ui_templates/scaleDlg.ui" line="113"/>
        <source>Same scale for all dimensions</source>
        <translation>所有维度应用相同比例</translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="120"/>
        <source>Whether the cloud (center) should stay at the same place or not (i.e. coordinates are multiplied directly)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="123"/>
        <source>Keep entity in place</source>
        <translation>保持实体在原地</translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="130"/>
        <source>To apply the same scale(s) to the &apos;Global Shift&apos; as well</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/scaleDlg.ui" line="133"/>
        <source>Rescale Global shift</source>
        <translation>重新调节全局偏移</translation>
    </message>
</context>
<context>
    <name>SelectChildrenDialog</name>
    <message>
        <location filename="../ui_templates/selectChildrenDlg.ui" line="14"/>
        <source>Select children by type and/or name</source>
        <translation>根据类型或者名称选择子对象</translation>
    </message>
    <message>
        <location filename="../ui_templates/selectChildrenDlg.ui" line="20"/>
        <source>Select children...</source>
        <translation>选择子对象…</translation>
    </message>
    <message>
        <location filename="../ui_templates/selectChildrenDlg.ui" line="29"/>
        <source>of type</source>
        <translation>类型的</translation>
    </message>
    <message>
        <location filename="../ui_templates/selectChildrenDlg.ui" line="39"/>
        <source>with name</source>
        <translation>使用名称</translation>
    </message>
    <message>
        <location filename="../ui_templates/selectChildrenDlg.ui" line="64"/>
        <source>regex</source>
        <translation>正则表达式</translation>
    </message>
    <message>
        <location filename="../ui_templates/selectChildrenDlg.ui" line="99"/>
        <source>strict</source>
        <translation>严格的</translation>
    </message>
</context>
<context>
    <name>SliceWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/slicewindow.cpp" line="30"/>
        <source>Slice</source>
        <translation>片</translation>
    </message>
</context>
<context>
    <name>SmoothConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="14"/>
        <source>Form</source>
        <translation>平滑实体</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="22"/>
        <source>Number of Iterations</source>
        <translation>迭代次数</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="46"/>
        <source>Convergence</source>
        <translation>收敛</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="60"/>
        <source>Feature Angle</source>
        <translation>特征角度</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="67"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="74"/>
        <source>Open ..</source>
        <translation>打开 ..</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="81"/>
        <source>Edge Angle</source>
        <translation>边缘角</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="95"/>
        <source>Boundary Smoothing</source>
        <translation>边界平滑</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothconfig.ui" line="102"/>
        <source>Feature Edge Smoothing</source>
        <translation>特征边缘平滑</translation>
    </message>
</context>
<context>
    <name>SmoothWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/smoothwindow.cpp" line="18"/>
        <source>Smooth</source>
        <translation>光滑的</translation>
    </message>
</context>
<context>
    <name>StatisticalOutliersRemover</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/StatisticalOutliersRemover.cpp" line="37"/>
        <source>Statistical Outlier Removal</source>
        <translation>基于统计的离群点移除</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/StatisticalOutliersRemover.cpp" line="38"/>
        <source>Filter outlier data based on point neighborhood statistics</source>
        <translation>基于点云近邻统计数据过滤外部数据点</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/StatisticalOutliersRemover.cpp" line="39"/>
        <source>Filter the points that are farther of their neighbors than the average (plus a number of times the standard deviation)</source>
        <translation>过滤比平均距离更远的点（加上标准偏差的几倍）</translation>
    </message>
</context>
<context>
    <name>StatisticalOutliersRemoverDlg</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/StatisticalOutliersRemoverDlg.ui" line="14"/>
        <source>Statistical Outliers Removal</source>
        <translation>统计离群点剔除</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/StatisticalOutliersRemoverDlg.ui" line="26"/>
        <source>Number of points to use for 
mean distance estimation</source>
        <translation>平均距离评估使用点数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/StatisticalOutliersRemoverDlg.ui" line="51"/>
        <source>Standard deviation
 multiplier threshold (nSigma)</source>
        <translation>标准差
相乘阈值（nSigma）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Filters/dialogs/StatisticalOutliersRemoverDlg.ui" line="76"/>
        <source>(max distance = average distance + nSigma * std. dev.)</source>
        <translation>(最大距离 = 平均距离 + nSigma * 标准差.)</translation>
    </message>
</context>
<context>
    <name>StatisticalTestDialog</name>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="14"/>
        <source>Dialog</source>
        <translation>对话框</translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="20"/>
        <source>Noise model</source>
        <translation>噪声模型</translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="26"/>
        <source>param1</source>
        <translation>参数1</translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="68"/>
        <source>param2</source>
        <translation>参数2</translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="135"/>
        <source>param3</source>
        <translation>参数3</translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="160"/>
        <source>false rejection probability</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="163"/>
        <source>p(Chi2)</source>
        <translation>p(Chi2)</translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="206"/>
        <source>neighbors used to compute observed local dist.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/statisticalTestDlg.ui" line="209"/>
        <source>Neighbors</source>
        <translation>近邻数</translation>
    </message>
</context>
<context>
    <name>StereogramDialog</name>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="14"/>
        <source>Stereogram</source>
        <translation>立体图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="50"/>
        <source>dip direction: 0°</source>
        <translation>倾斜方向:0°</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="202"/>
        <source>[Mean] dip direction: 0° - dip 0°</source>
        <translation>平均倾斜方向：  0° - dip 0°</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="224"/>
        <source>Stereogram parameters</source>
        <translation>立体图参数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="256"/>
        <source>Display options</source>
        <translation>显示选项</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="271"/>
        <source>Density color scale</source>
        <translation>密度色标</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="286"/>
        <source>Steps</source>
        <translation>步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="318"/>
        <source>Other</source>
        <translation>其他</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="335"/>
        <source>Ticks frequency</source>
        <translation>Ticks频率</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="371"/>
        <source>Show families color (on the stereogram outer edge)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="374"/>
        <source>Show families color</source>
        <translation>显示全范围颜色</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="387"/>
        <source>Interactive filter</source>
        <translation>交互过滤</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="405"/>
        <source>Filter facets by orientation</source>
        <translation>按方向过滤面片</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="420"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="427"/>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="474"/>
        <source>dip</source>
        <translation>倾斜</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="434"/>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="454"/>
        <source> deg.</source>
        <translation> 角度.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="447"/>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="497"/>
        <source>dip dir.</source>
        <translation>倾斜方向.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="467"/>
        <source>Span</source>
        <translation>跨度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramDlg.ui" line="565"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
</context>
<context>
    <name>StereogramParamsDlg</name>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramParamsDlg.ui" line="14"/>
        <source>Stereogram</source>
        <translation>立体图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramParamsDlg.ui" line="22"/>
        <source>main sectors step</source>
        <translation>主段步长</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramParamsDlg.ui" line="29"/>
        <location filename="../../plugins/core/qFacets/ui/stereogramParamsDlg.ui" line="55"/>
        <source> deg.</source>
        <translation> 角度.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/ui/stereogramParamsDlg.ui" line="48"/>
        <source>resolution</source>
        <translation>分辨率</translation>
    </message>
</context>
<context>
    <name>StreamlineConfig</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="14"/>
        <source>Form</source>
        <translation>线型处理</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="22"/>
        <source>Config Ruled Surface</source>
        <translation>配置规则曲面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="30"/>
        <source>Close Surface</source>
        <translation>近表面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="37"/>
        <source>Distance Factor</source>
        <translation>距离因子</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="53"/>
        <source>Offset</source>
        <translation>偏移量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="60"/>
        <source>Pass Lines</source>
        <translation>通过线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="67"/>
        <source>Ruled Mode</source>
        <translation>统治模式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="74"/>
        <source>Resolution</source>
        <translation>决议</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="82"/>
        <source>Resample</source>
        <translation>重采样</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="87"/>
        <source>Point Walk</source>
        <translation>点走</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="98"/>
        <source>OnRatio</source>
        <translation>OnRatio</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="105"/>
        <source>Orient Loops</source>
        <translation>方向循环</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="117"/>
        <source>Display Effect</source>
        <translation>显示效果</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="124"/>
        <source>Open ...</source>
        <translation>打开……</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="132"/>
        <source>FORWARD</source>
        <translation>向前的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="137"/>
        <source>BACKWARD</source>
        <translation>向后的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="142"/>
        <source>BOTH</source>
        <translation>全部</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="150"/>
        <source>Integration Direction</source>
        <translation>积分方向</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="157"/>
        <source>Integrator Type</source>
        <translation>积分器类型</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="165"/>
        <source>RUNGE_KUTTA2</source>
        <translation>RUNGE_KUTTA2</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="170"/>
        <source>RUNGE_KUTTA4</source>
        <translation>RUNGE_KUTTA4</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="175"/>
        <source>RUNGE_KUTTA45</source>
        <translation>RUNGE_KUTTA45</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="180"/>
        <source>NONE</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="185"/>
        <source>UNKNOWN</source>
        <translation>未知的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="200"/>
        <source>Config Line</source>
        <translation>配置线</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="222"/>
        <source>Point1</source>
        <translation>点1</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="236"/>
        <source>Point2</source>
        <translation>点2</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="270"/>
        <source>None</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="275"/>
        <source>Tube</source>
        <translation>管</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="280"/>
        <source>Surface</source>
        <translation>表面</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="288"/>
        <source>Display Mode</source>
        <translation>显示模式</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="296"/>
        <source>Transparent</source>
        <translation>透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="301"/>
        <source>Opaque</source>
        <translation>不透明的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="306"/>
        <source>Wireframe</source>
        <translation>线框</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="314"/>
        <source>Config Point</source>
        <translation>配置点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="322"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="353"/>
        <source>Number of Points</source>
        <translation>点数量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="360"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="379"/>
        <source>File:</source>
        <translation>文件:</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="386"/>
        <source>Config Tube</source>
        <translation>配置管道</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="400"/>
        <source>Number of Sides</source>
        <translation>边数量</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="407"/>
        <source>Radius Factor</source>
        <translation>半径因子</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="414"/>
        <source>Capping</source>
        <translation>限制</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="426"/>
        <source>Source</source>
        <translation>源</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="434"/>
        <source>Point</source>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlineconfig.ui" line="439"/>
        <source>Line</source>
        <translation>行</translation>
    </message>
</context>
<context>
    <name>StreamlineWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/streamlinewindow.cpp" line="31"/>
        <source>Streamline</source>
        <translation>简化</translation>
    </message>
</context>
<context>
    <name>SubsamplingDialog</name>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="14"/>
        <source>Cloud sub sampling</source>
        <translation>点云下采样</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="20"/>
        <source>Sampling parameters</source>
        <translation>采样参数</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="40"/>
        <source>method</source>
        <translation>方法</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="54"/>
        <source>none</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="77"/>
        <source>all</source>
        <translation>所有</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="89"/>
        <source>The more on the left, the less points will be kept</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="113"/>
        <source>value</source>
        <translation>值</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="166"/>
        <source>To modulate the sampling distance with a scalar field value</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="169"/>
        <source>Use active SF</source>
        <translation>使用当前标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="184"/>
        <source>SF value</source>
        <translation>标量字段值</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="191"/>
        <source>Spacing value</source>
        <translation>间距值</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="198"/>
        <source>min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../ui_templates/subsamplingDlg.ui" line="234"/>
        <source>max</source>
        <translation>最大</translation>
    </message>
</context>
<context>
    <name>TemplateAlignment</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="40"/>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="41"/>
        <source>Template Alignment</source>
        <translation>基于模板匹配识别</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="42"/>
        <source>Template Alignment from clouds</source>
        <translation>基于点云的模板匹配</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="126"/>
        <source>Invalid scale parameters!</source>
        <translation>非法比例参数！</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="140"/>
        <source>At least one cloud (class #1 or #2) was not defined!</source>
        <translation>至少需要一个点云模型定义！</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="229"/>
        <source>(values less than 0.00002 are good) Best fitness score: %1</source>
        <translation>(小于 0.00002 即可) 最佳拟合系数: %1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="292"/>
        <source>[ApplyTransformation] Applied transformation matrix:</source>
        <translation>[ApplyTransformation] 应用转换矩阵:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="294"/>
        <source>Hint: copy it (CTRL+C) and apply it - or its inverse - on any entity with the &apos;Edit &gt; Apply transformation&apos; tool</source>
        <translation>提示：复制(CTRL+C) 并应用 - 或者反转 - 在 &apos;编辑 &gt; 应用转换&apos; 工具</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="305"/>
        <source>Selected entity does not have any suitable scalar field or RGB.</source>
        <translation>所选点云没有任何适用的标量字段或者RGB.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="307"/>
        <source>Wrong Parameters. One or more parameters cannot be accepted</source>
        <translation>参数错误：一个或者多个参数不被接受</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/TemplateAlignment.cpp" line="309"/>
        <source>Template Alignment does not returned any point. Try relaxing your parameters</source>
        <translation>模板匹配模块没有返回任何点，可以尝试调整参数后重试</translation>
    </message>
</context>
<context>
    <name>TemplateAlignmentDialog</name>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="14"/>
        <source>Template Alignment</source>
        <translation>点云模板匹配</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="24"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="30"/>
        <source>Role</source>
        <translation>角色</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="37"/>
        <source>Cloud</source>
        <translation>点云</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="44"/>
        <source>template #1</source>
        <translation>模板 #1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="57"/>
        <source>Points belonging to class #1 </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="64"/>
        <source>Template 1</source>
        <translation>模板1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="74"/>
        <source>template #2</source>
        <translation>模板 #2</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="90"/>
        <source>Points belonging to class #2</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="97"/>
        <source>Template 2</source>
        <translation>模板2</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="104"/>
        <source>target</source>
        <translation>目标</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="120"/>
        <source>Additional points that will be added to the 2D classifier behavior representation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="130"/>
        <source>Scales</source>
        <translation>尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="136"/>
        <source>ramp</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="161"/>
        <source>Mininum scale</source>
        <translation>最少尺度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="164"/>
        <source>Min = </source>
        <translation>最小值 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="186"/>
        <source>Step</source>
        <translation>一步</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="189"/>
        <source>Step = </source>
        <translation>步长 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="211"/>
        <source>Max scale</source>
        <translation>最大比例</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="214"/>
        <source>Max = </source>
        <translation>最大值 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="236"/>
        <source>Inp</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="239"/>
        <source>list</source>
        <translation>列表</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="249"/>
        <source>Input scales as a list of values (separated by a space character)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="259"/>
        <source>Advanced</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="281"/>
        <source>Maximum Iterations</source>
        <translation>最大迭代数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="288"/>
        <source>Minimum Sample Distance</source>
        <translation>最小采样距离</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="311"/>
        <source>Feature Radius</source>
        <translation>特征半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="321"/>
        <source>Voxel Grid</source>
        <translation>体素网格滤波</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="331"/>
        <source>Maximum Correspondence Distance</source>
        <translation>最大对应点距离</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="370"/>
        <source>Leaf Size = </source>
        <translation>叶子尺寸 = </translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="405"/>
        <source>Maximum number of core points computed on each class</source>
        <translation>每类最大计算核心点数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="424"/>
        <source>Normal Radius</source>
        <translation>法线半径</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.ui" line="431"/>
        <source>Max thread count</source>
        <translation>最大线程数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.cpp" line="72"/>
        <source>You need at least 1 loaded clouds to perform alignment</source>
        <translation>你需要至少选择一个加载的点云实现模板匹配</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/PclAlgorithms/Recognitions/dialogs/TemplateAlignmentDialog.cpp" line="298"/>
        <source>unnamed</source>
        <translation>未命名的</translation>
    </message>
</context>
<context>
    <name>ThresholdWindow</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkWidgets/filters/thresholdwindow.cpp" line="18"/>
        <source>Threshold</source>
        <translation>阈值</translation>
    </message>
</context>
<context>
    <name>TracePolyLineDlg</name>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="20"/>
        <source>Segmentation</source>
        <translation>分割</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="41"/>
        <source>Width</source>
        <translation>宽度</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="80"/>
        <source>Snap size</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="110"/>
        <source>Oversample</source>
        <translation>上采样</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="143"/>
        <source>Export current polyline to the main DB</source>
        <translation>导出当前多段线到主资源树</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="157"/>
        <source>Reset current polyline</source>
        <translation>重置当前多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="171"/>
        <source>Continue the current polyline edition</source>
        <translation>继续编辑当前多段线</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="178"/>
        <source>C</source>
        <extracomment>Shortcut for continue</extracomment>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="194"/>
        <source>Confirm polyline creation and exit</source>
        <translation>确认创建多段线并退出</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="197"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="214"/>
        <location filename="../ui_templates/tracePolylineDlg.ui" line="217"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
</context>
<context>
    <name>TrainDisclaimerDialog</name>
    <message>
        <location filename="../../plugins/core/qCanupo/trainDisclaimerDlg.ui" line="14"/>
        <source>qCANUPO (disclaimer)</source>
        <translation>qCANUPO (免责声明)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/trainDisclaimerDlg.ui" line="48"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;MS Shell Dlg 2&apos;; font-size:8.25pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-weight:600; color:#1f497d;&quot;&gt;Classifier training based on multi-scale dimensionality (CANUPO)&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; font-style:italic; color:#1f497d;&quot;&gt;Brodu and Lague, 3D Terrestrial LiDAR data classification of complex natural scenes using a multi-scale dimensionality criterion, ISPRS j. of Photogram.&#xa0;Rem. Sens., 2012&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d; background-color:#ffffff;&quot;&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d; background-color:#ffffff;&quot;&gt;Funded by Université Européenne de Bretagne, Centre National de la Recherche Scientifique and EEC Marie-Curie actions&lt;/span&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d; background-color:#ffffff;&quot;&gt;&lt;/p&gt;
&lt;p align=&quot;center&quot; style=&quot; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; background-color:#ffffff;&quot;&gt;&lt;span style=&quot; font-family:&apos;Calibri,sans-serif&apos;; font-size:10pt; color:#1f497d;&quot;&gt;Enjoy!&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>UnrollDialog</name>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="14"/>
        <source>Unroll</source>
        <translation>展开</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="22"/>
        <source>Type</source>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="30"/>
        <source>Cylinder</source>
        <translation>圆柱体</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="35"/>
        <source>Cone</source>
        <translation>锥体</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="40"/>
        <source>Straightened cone</source>
        <translation>直锥</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="61"/>
        <source>Deviation from the theoretical shape (i.e. cone or cylinder)</source>
        <translation>理论形状偏差（例如：类圆锥体或者圆柱体）</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="64"/>
        <source>Export deviation scalar field</source>
        <translation>导出偏差标量字段</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="73"/>
        <source>Shape</source>
        <translation>形状</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="81"/>
        <source>Axis</source>
        <translation>轴</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="150"/>
        <location filename="../ui_templates/unrollDlg.ui" line="160"/>
        <source>Cone angle (0-180°)</source>
        <translation>锥角 (0-180 °)</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="153"/>
        <source>Half angle</source>
        <translation>半角</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="166"/>
        <location filename="../ui_templates/unrollDlg.ui" line="450"/>
        <location filename="../ui_templates/unrollDlg.ui" line="492"/>
        <source> deg</source>
        <translation> 角度</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="213"/>
        <location filename="../ui_templates/unrollDlg.ui" line="226"/>
        <source>Cylinder (or cone base) radius</source>
        <translation>圆柱体（或者类圆锥体底部）半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="216"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="250"/>
        <source>Axis position</source>
        <translation>轴位置</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="402"/>
        <source>Auto (gravity center)</source>
        <translation>自动（重心）</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="431"/>
        <source>Unroll range (can do multiple turns)</source>
        <translation>展开范围（可以进行多轮）</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="434"/>
        <source>Unroll range</source>
        <translation>展开范围</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="440"/>
        <source>Start angle</source>
        <translation>开始角度</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="447"/>
        <source>Starting angle (can be negative)</source>
        <translation>开始角度（可以为负值）</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="482"/>
        <source>Stop angle</source>
        <translation>停止角度</translation>
    </message>
    <message>
        <location filename="../ui_templates/unrollDlg.ui" line="489"/>
        <source>Stop angle (can be negative)</source>
        <translation>停止角度（可以为负值）</translation>
    </message>
</context>
<context>
    <name>VolumeCalcDialog</name>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="20"/>
        <source>Volume calculation</source>
        <translation>体积计算</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="53"/>
        <source>Ground / Before</source>
        <translation>地面/之前</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="62"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="197"/>
        <source>Source</source>
        <translation>源</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="72"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="214"/>
        <source>choose the value to fill the cells in which no point is projected : minimum value over the whole point cloud or average value (over the whole cloud also)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="76"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="218"/>
        <source>leave empty</source>
        <translation>离开空</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="81"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="223"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="412"/>
        <source>minimum height</source>
        <translation>最小的高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="86"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="228"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="417"/>
        <source>average height</source>
        <translation>平均身高</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="91"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="233"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="422"/>
        <source>maximum height</source>
        <translation>最大高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="96"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="238"/>
        <source>user specified value</source>
        <translation>用户指定的值</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="101"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="243"/>
        <source>interpolate</source>
        <translation>插入</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="109"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="207"/>
        <source>Empty cells</source>
        <translation>空的细胞</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="125"/>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="260"/>
        <source>Custom value for empty cells</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="168"/>
        <source>Swap</source>
        <translation>交换</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="171"/>
        <source>swap</source>
        <translation>交换</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="191"/>
        <source>Ceil / After</source>
        <translation>后装天花板/</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="282"/>
        <source>Grid</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="303"/>
        <source>step</source>
        <translation>一步</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="315"/>
        <source>size of step of the grid generated (in the same units as the coordinates of the point cloud)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="337"/>
        <source>Edit grid</source>
        <translation>编辑网格</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="346"/>
        <source>size</source>
        <translation>大小</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="353"/>
        <source>Grid size corresponding to the current step / boundaries</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="363"/>
        <source>projection dir.</source>
        <translation>投影dir。</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="370"/>
        <source>Projection direction (X, Y or Z)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="395"/>
        <source>cell height</source>
        <translation>细胞高度</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="402"/>
        <source>Per-cell height computation method:
 - minimum = lowest point in the cell
 - average = mean height of all points inside the cell
 - maximum = highest point in the cell</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="433"/>
        <source>Update the grid / display / measurements</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="439"/>
        <source>Update</source>
        <translation>更新</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="449"/>
        <source>Results</source>
        <translation>结果</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="458"/>
        <source>At least one of the cloud is sparse!
You should fill the empty cells...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="479"/>
        <source>Copy to clipboard</source>
        <translation>复制到剪贴板</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="489"/>
        <source>Export the grid as a point cloud
(warning, the points heights will be the difference of altitude!)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="493"/>
        <source>Export grid as a cloud</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="515"/>
        <source>Num. precision</source>
        <translation>Num.精度</translation>
    </message>
    <message>
        <location filename="../ui_templates/volumeCalcDlg.ui" line="522"/>
        <source>Numerical precision (output measurements, etc.)</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>VtkUtils::VtkPlotWidget</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/vtkplotwidget.cpp" line="38"/>
        <source>ChartXY</source>
        <translation>ChartXY</translation>
    </message>
</context>
<context>
    <name>Widgets::ColorComboBox</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="212"/>
        <source>black</source>
        <translation>黑色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="213"/>
        <source>red</source>
        <translation>红色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="214"/>
        <source>green</source>
        <translation>绿色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="215"/>
        <source>blue</source>
        <translation>蓝色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="216"/>
        <source>cyan</source>
        <translation>青色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="217"/>
        <source>magenta</source>
        <translation>品红色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="218"/>
        <source>yellow</source>
        <translation>黄色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="219"/>
        <source>dark yellow</source>
        <translation>深黄色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="220"/>
        <source>navy</source>
        <translation>海军</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="221"/>
        <source>purple</source>
        <translation>紫色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="222"/>
        <source>wine</source>
        <translation>酒</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="223"/>
        <source>olive</source>
        <translation>橄榄</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="224"/>
        <source>dark cyan</source>
        <translation>暗青色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="225"/>
        <source>royal</source>
        <translation>皇家</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="226"/>
        <source>orange</source>
        <translation>橙色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="227"/>
        <source>violet</source>
        <translation>紫罗兰色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="228"/>
        <source>pink</source>
        <translation>粉红色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="229"/>
        <source>white</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="230"/>
        <source>light gray</source>
        <translation>浅灰色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="231"/>
        <source>gray</source>
        <translation>灰色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="232"/>
        <source>light yellow</source>
        <translation>淡黄色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="233"/>
        <source>light cyan</source>
        <translation>青色光</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="234"/>
        <source>light magenta</source>
        <translation>红色光</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/colorcombobox.cpp" line="235"/>
        <source>dark gray</source>
        <translation>深灰色的</translation>
    </message>
</context>
<context>
    <name>Widgets::ColorPickerPopup</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="903"/>
        <source>Custom</source>
        <translation>自定义</translation>
    </message>
</context>
<context>
    <name>Widgets::FontPushButton</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/fontpushbutton.cpp" line="37"/>
        <source>Font</source>
        <translation>字体</translation>
    </message>
</context>
<context>
    <name>Widgets::QtColorPicker</name>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="284"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="410"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="513"/>
        <source>Black</source>
        <translation>黑色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="411"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="514"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="412"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="515"/>
        <source>Red</source>
        <translation>红色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="413"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="516"/>
        <source>Dark red</source>
        <translation>深红色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="414"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="517"/>
        <source>Green</source>
        <translation>绿色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="415"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="518"/>
        <source>Dark green</source>
        <translation>深绿色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="416"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="519"/>
        <source>Blue</source>
        <translation>蓝色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="417"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="520"/>
        <source>Dark blue</source>
        <translation>深蓝色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="418"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="521"/>
        <source>Cyan</source>
        <translation>青色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="419"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="522"/>
        <source>Dark cyan</source>
        <translation>暗青色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="420"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="523"/>
        <source>Magenta</source>
        <translation>品红色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="421"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="524"/>
        <source>Dark magenta</source>
        <translation>黑红色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="422"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="525"/>
        <source>Yellow</source>
        <translation>黄色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="423"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="526"/>
        <source>Dark yellow</source>
        <translation>深黄色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="424"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="527"/>
        <source>Gray</source>
        <translation>灰色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="425"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="528"/>
        <source>Dark gray</source>
        <translation>深灰色的</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="426"/>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="529"/>
        <source>Light gray</source>
        <translation>浅灰色</translation>
    </message>
    <message>
        <location filename="../../libs/PCLEngine/VtkUtils/qtcolorpicker.cpp" line="444"/>
        <source>Custom</source>
        <translation>自定义</translation>
    </message>
</context>
<context>
    <name>ccClippingBoxTool</name>
    <message>
        <location filename="../ecvClippingBoxTool.cpp" line="605"/>
        <source>Preparing extraction</source>
        <translation>准备提取</translation>
    </message>
    <message>
        <location filename="../ecvClippingBoxTool.cpp" line="619"/>
        <source>Cloud &apos;%1</source>
        <translation>点云 &apos; %1</translation>
    </message>
    <message>
        <location filename="../ecvClippingBoxTool.cpp" line="620"/>
        <source>Points: %L1</source>
        <translation>点:% L1</translation>
    </message>
</context>
<context>
    <name>ccComparisonDlg</name>
    <message>
        <location filename="../ecvComparisonDlg.cpp" line="539"/>
        <source>Determining optimal octree level</source>
        <translation>确定最佳八叉树层数</translation>
    </message>
    <message>
        <location filename="../ecvComparisonDlg.cpp" line="540"/>
        <source>Testing %1 levels...</source>
        <translation>测试 第 %1 层……</translation>
    </message>
</context>
<context>
    <name>ccCompass</name>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="114"/>
        <source>No Selection</source>
        <translation>没有选择</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="213"/>
        <source>[ccCompass] Could not find valid 3D window.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="301"/>
        <source>Compass</source>
        <translation>指南针</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="302"/>
        <source>Converting Compass types...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="483"/>
        <source>Error: ccCompass could not find the ACloudViewer window. Abort!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="567"/>
        <source>[ccCompass] Could not retrieve valid picking hub. Measurement aborted.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="575"/>
        <source>Another tool is already using the picking mechanism. Stop it first</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="607"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="618"/>
        <source>[ccCompass] Error: Please select a GeoObject to digitize to.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="628"/>
        <source>[ccCompass] Warning: Could not retrieve valid mapping region for the active GeoObject.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="647"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="656"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="674"/>
        <source>measurements</source>
        <translation>测量</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="727"/>
        <source>[Item picking] Shit&apos;s fubar (Picked point is not in pickable entities DB?)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="784"/>
        <source>ccCompassType</source>
        <translation>ccCompassType</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1028"/>
        <source>[Compass] Select several GeoObjects to merge.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1066"/>
        <source>[Compass] Merged selected GeoObjects to </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1074"/>
        <source>[Compass] fitPlane</source>
        <translation>(指南针)fitPlane</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1124"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1167"/>
        <source>[Compass] Not enough 3D information to generate sensible fit plane.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1338"/>
        <source>Minimum trace size (points):</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1340"/>
        <source>Maximum trace size (points):</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1342"/>
        <source>Wishart Degrees of Freedom:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1344"/>
        <source>Likelihood power:</source>
        <translation>可能性能量：</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1346"/>
        <source>Calculate thickness:</source>
        <translation>计算厚度:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1347"/>
        <source>Calculate thickness</source>
        <translation>计算厚度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1348"/>
        <source>Distance cutoff (m):</source>
        <translation>距离截止(米):</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1350"/>
        <source>Samples:</source>
        <translation>样品:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1352"/>
        <source>MCMC Stride (radians):</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1356"/>
        <source>The minimum size of the normal-estimation window.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1357"/>
        <source>The maximum size of the normal-estimation window.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1358"/>
        <source>Sets the degrees of freedom parameter for the Wishart distribution. Due to non-independent data/errors in traces, this should be low (~10). Higher give more confident results - use with care!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1359"/>
        <source>The furthest distance to search for points on the opposite surface of a GeoObject during thickness calculations.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1360"/>
        <source>Sample n orientation estimates at each point in each trace to quantify uncertainty.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1361"/>
        <source>Fudge factor to change the balance between the prior and likelihood functions. Advanced use only - see docs for details.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1362"/>
        <source>Standard deviation of the normal distribution used to calculate monte-carlo jumps during sampling. Larger numbers sample more widely but are slower to run.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1411"/>
        <source>[ccCompass] Error - provided maxsize is less than minsize? Get your shit together...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1417"/>
        <source>[ccCompass] Estimating structure normals. This may take a while...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1433"/>
        <source>Estimating Structure Normals</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1434"/>
        <source>Gathering data...</source>
        <translation>收集数据……</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1511"/>
        <source>[ccCompass] No GeoObjects or Traces could be found to estimate structure normals for. Please select some!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1519"/>
        <source>Processing %1 of %2 datasets: Calculating fit planes...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1574"/>
        <source>[ccCompass] Warning: Region %1 contains less than minsize points. Region ignored.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1588"/>
        <source>[ccCompass] Warning: Could not compute eigensystem for region %1. Region ignored.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1680"/>
        <source>[ccCompass] Warning: Cannot compensate for outcrop-surface bias as point cloud has no normals. Structure normal estimates may be misleading or incorrect.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1841"/>
        <source>[ccCompass] Warning: Region %1 contains no valid points (PinchNodes break the trace into small segments?). Region ignored.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="1938"/>
        <source>Processing %1 of %2 datasets: Sampling points...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2027"/>
        <source>[ccCompass] Warning - MCMC sampler failed so sampling will be incomplete.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2069"/>
        <source>Processing %1 of %2 datasets: Estimating thickness...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2209"/>
        <source>[ccCompass] Structure normal estimation complete.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2255"/>
        <source>[ccCompass] Error - no traces or SNEs found to compute estimate strain with.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2278"/>
        <source>Voxel Size:</source>
        <translation>立体像素大小:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2280"/>
        <source>Use external SNE:</source>
        <translation>使用外部新力:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2281"/>
        <source>Build graphics:</source>
        <translation>构建图形:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2282"/>
        <source>Shape exaggeration factor:</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2286"/>
        <source>The voxel size for computing strain. This should be large enough that most boxes contain SNEs.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2287"/>
        <source>Use SNE orientation estimates for outside the current cell if none are avaliable within it.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2288"/>
        <source>Build graphic strain ellipses and grid domains. Useful for validation.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2289"/>
        <source>Exaggerate the shape of strain ellipses for easier visualisation.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2321"/>
        <source>Computing strain estimates</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2342"/>
        <source>Gathering GeoObjects...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2373"/>
        <source>[ccCompass] Error: cell %1 is outside of mesh bounds (with total size = %2 [%3,%4,%5]).</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2396"/>
        <source>Blocks</source>
        <translation>块</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2404"/>
        <source>Calculating strain tensors...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2421"/>
        <source>DataInCell</source>
        <translation>DataInCell</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2574"/>
        <source>Strain</source>
        <translation>应变</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2603"/>
        <source>Ellipses</source>
        <translation>椭圆</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2604"/>
        <source>Grid</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2782"/>
        <source>[ccCompass] Error - no polylines or traces found to compute P21.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2801"/>
        <source>[ccCompass] Error - cannot calculate P21 intensity for structures digitised from different point clouds.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2830"/>
        <source>Search Radius:</source>
        <translation>搜索半径:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2832"/>
        <source>Subsample:</source>
        <translation>子样品:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2835"/>
        <source>The search radius used to define the region to compute P21 within.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2836"/>
        <source>Only sample P21 on the each n&apos;th point in the original outcrop model (decreases calculation time).</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2858"/>
        <source>[ccCompass] Estimating P21 Intensity using a search radius of of %1.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2869"/>
        <source>P21 Intensity</source>
        <translation>P21强度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2897"/>
        <source>Estimating P21 Intensity</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2898"/>
        <source>Sampling structures...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="2930"/>
        <source>Calculating patch areas...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3020"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3076"/>
        <source>ConvertedLines</source>
        <translation>ConvertedLines</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3066"/>
        <source>[Compass] No polylines or traces converted - none found.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3116"/>
        <source>[Compass] No objects selected.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3182"/>
        <source>[Compass] Warning: No GeoObject could be found that matches %1.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3351"/>
        <source>New GeoObject</source>
        <translation>New GeoObject</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3351"/>
        <source>GeoObject Name:</source>
        <translation>GeoObject名称:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3362"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3371"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3387"/>
        <source>interpretation</source>
        <translation>解释</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3438"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3446"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3576"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3584"/>
        <source>Please select a point cloud containing your field data (this can be loaded from a text file)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3457"/>
        <source>Dip Field:</source>
        <translation>下降:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3458"/>
        <source>Dip-Direction Field:</source>
        <translation>倾向:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3459"/>
        <source>Plane Size</source>
        <translation>平面尺寸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3502"/>
        <source>Error: Dip and Dip-Direction scalar fields must be different!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3595"/>
        <source>Trend Field:</source>
        <translation>趋势:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3596"/>
        <source>Plunge Field:</source>
        <translation>跳水:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3597"/>
        <source>Display Length</source>
        <translation>显示长度</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3635"/>
        <source>Error: Trend and plunge scalar fields must be different!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3657"/>
        <source>verts</source>
        <translation>Verts</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3678"/>
        <source>SVG Output file</source>
        <translation>SVG输出文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3678"/>
        <source>SVG files (*.svg)</source>
        <translation>SVG文件(* .)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3753"/>
        <source>[ccCompass] Successfully saved %1 polylines to .svg file.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3760"/>
        <source>[ccCompass] Could not write polylines to .svg - no polylines found!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3829"/>
        <source>Output file</source>
        <translation>输出文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3829"/>
        <source>CSV files (*.csv *.txt);XML (*.xml)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3912"/>
        <source>[ccCompass] Successfully exported plane data.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3917"/>
        <source>[ccCompass] No plane data found.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3923"/>
        <source>[ccCompass] Successfully exported trace data.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3928"/>
        <source>[ccCompass] No trace data found.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3934"/>
        <source>[ccCompass] Successfully exported lineation data.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3939"/>
        <source>[ccCompass] No lineation data found.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3945"/>
        <source>[ccCompass] Successfully exported thickness data.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3950"/>
        <source>[ccCompass] No thickness data found.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="3958"/>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="4216"/>
        <source>[ccCompass] Could not open output files... ensure CC has write access to this location.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ccCompass.cpp" line="4210"/>
        <source>[ccCompass] Successfully exported data-tree to xml.</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>ccDBRoot</name>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="255"/>
        <source>Expand branch</source>
        <translation>展开分支</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="256"/>
        <source>Collapse branch</source>
        <translation>折叠分支</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="257"/>
        <source>Information (recursive)</source>
        <translation>信息（循环）</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="258"/>
        <source>Sort children by type</source>
        <translation>按类型排列</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="259"/>
        <source>Sort children by name (A-Z)</source>
        <translation>按名称排列（A-Z）</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="260"/>
        <source>Sort children by name (Z-A)</source>
        <translation>按名称排列 (Z-A)</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="261"/>
        <source>Select children by type and/or name</source>
        <translation>根据类型或者名称选择</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="262"/>
        <source>Delete</source>
        <translation>删除</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="263"/>
        <source>Toggle</source>
        <translation>开关</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="264"/>
        <source>Toggle visibility</source>
        <translation>开关可见性</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="265"/>
        <source>Toggle color</source>
        <translation>开关颜色</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="266"/>
        <source>Toggle normals</source>
        <translation>开关法线</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="267"/>
        <source>Toggle materials/textures</source>
        <translation>开关材质/纹理</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="268"/>
        <source>Toggle SF</source>
        <translation>开关 标量字段</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="269"/>
        <source>Toggle 3D name</source>
        <translation>开关3D名称</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="270"/>
        <source>Add empty group</source>
        <translation>添加空组</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="271"/>
        <source>Align camera</source>
        <translation>调整相机</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="272"/>
        <source>Align camera (reverse)</source>
        <translation>反向对齐相机</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="273"/>
        <source>Bubble-view</source>
        <translation>弹窗</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="274"/>
        <location filename="../db_tree/ecvDBRoot.cpp" line="2131"/>
        <source>Edit scalar value</source>
        <translation>编辑标量值</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="465"/>
        <source>[ccDBRoot::removeElements] Internal error: object &apos;%1&apos; has no parent</source>
        <translation>[ccDBRoot::removeElements] 内部错误: 对象 &apos;%1&apos; 没有父组件</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="506"/>
        <source>[ccDBRoot::removeElement] Internal error: object has no parent</source>
        <translation>[ccDBRoot::removeElement] 内部错误: 对象 没有父组件</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="554"/>
        <source>Object &apos;%1&apos; can&apos;t be deleted this way (locked)</source>
        <translation>对象 &apos;%1&apos; 不能被删除 (被锁)</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="580"/>
        <source>Vertices can&apos;t be deleted without their parent mesh</source>
        <translation>顶点不能被单独删除（需要连同网格一起删除）</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="971"/>
        <source>[Selection] Labels and other entities can&apos;t be mixed (release the CTRL key to start a new selection)</source>
        <translation>[Selection] 标签和其他实体 不能混合选择 (请释放 CTRL 键 开始新的拾取)</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1012"/>
        <source>[ccDBRoot::selectEntities] Not enough memory</source>
        <translation>[ccDBRoot::selectEntities] 内存不足</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1366"/>
        <source>Vertices can&apos;t leave their parent mesh</source>
        <translation>顶点不能移出其父网格</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1377"/>
        <source>Sub-meshes can&apos;t leave their mesh group</source>
        <translation>子网格不能移出其网格组</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1385"/>
        <source>Meshes can&apos;t leave their associated cloud (vertices set)</source>
        <translation>网格不能移出关联点云（顶点集）</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1394"/>
        <source>This kind of entity can&apos;t leave their parent</source>
        <translation>此类型实体不能移出其父元素</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1618"/>
        <source>[Align camera] Corresponding view matrix:</source>
        <translation>[Align camera] 对应视角矩阵:</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1619"/>
        <source>[Orientation] You can copy this matrix values (CTRL+C) and paste them in the &apos;Apply transformation tool&apos; dialog</source>
        <translation>[Orientation] 你可以复制当前矩阵值 (CTRL+C) 并粘贴到 &apos;应用转换工具&apos; 对话框</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1661"/>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1732"/>
        <source>Not engough memory</source>
        <translation>内存不足</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1743"/>
        <source>Point(s):		%L1</source>
        <translation>点 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1744"/>
        <source>Triangle(s):		%L1</source>
        <translation>三角形 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1748"/>
        <source>Color(s):		%L1</source>
        <translation>颜色 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1750"/>
        <source>Normal(s):		%L1</source>
        <translation>法线 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1752"/>
        <source>Scalar field(s):		%L1</source>
        <translation>标量字段(s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1754"/>
        <source>Material(s):		%L1</source>
        <translation>材质 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1757"/>
        <source>Cloud(s):		%L1</source>
        <translation>点云:		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1758"/>
        <source>Mesh(es):		%L1</source>
        <translation>网格 (es):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1760"/>
        <source>Octree(s):		%L1</source>
        <translation>八叉树 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1762"/>
        <source>Image(s):		%L1</source>
        <translation>图像 (s)：		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1764"/>
        <source>Label(s):		%L1</source>
        <translation>标签 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1766"/>
        <source>Sensor(s):		%L1</source>
        <translation>传感器 (s):		%1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1770"/>
        <source>Information</source>
        <translation>信息</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1860"/>
        <source>Point cloud</source>
        <translation>点云</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1861"/>
        <source>Poly-line</source>
        <translation>多段线</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1862"/>
        <source>Mesh</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1863"/>
        <source>  Sub-mesh</source>
        <translation>  子网格</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1864"/>
        <source>  Primitive</source>
        <translation>  基础模型</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1865"/>
        <source>    Plane</source>
        <translation>    平面</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1866"/>
        <source>    Sphere</source>
        <translation>    球体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1867"/>
        <source>    Torus</source>
        <translation>    类圆环体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1868"/>
        <source>    Cylinder</source>
        <translation>    圆柱体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1869"/>
        <source>    Cone</source>
        <translation>    类圆锥体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1870"/>
        <source>    Box</source>
        <translation>    箱体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1871"/>
        <source>    Dish</source>
        <translation>    碗体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1872"/>
        <source>    Extrusion</source>
        <translation>    铸件</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1873"/>
        <source>Sensor</source>
        <translation>传感器</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1874"/>
        <source>  GBL/TLS sensor</source>
        <translation>  GBL/TLS 传感器</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1875"/>
        <source>  Camera sensor</source>
        <translation>  相机传感器</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1876"/>
        <source>Image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1877"/>
        <source>Facet</source>
        <translation>面片</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1878"/>
        <source>Label</source>
        <translation>标签</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1879"/>
        <source>Area label</source>
        <translation>区域标签</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1880"/>
        <source>Octree</source>
        <translation>八叉树</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1881"/>
        <source>Kd-tree</source>
        <translation>KD树</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1882"/>
        <source>Viewport</source>
        <translation>视窗</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1883"/>
        <source>Custom Types</source>
        <translation>自定义类型</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="1987"/>
        <source>[selectChildrenByTypeAndName] Not enough memory</source>
        <translation>[selectChildrenByTypeAndName] 内存不足</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="2059"/>
        <source>Group</source>
        <translation>组</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvDBRoot.cpp" line="2124"/>
        <source>[editLabelScalarValue] No active scalar field</source>
        <translation>[editLabelScalarValue] 没有激活的标量字段</translation>
    </message>
</context>
<context>
    <name>ccItemSelectionDlg</name>
    <message>
        <location filename="../ecvItemSelectionDlg.cpp" line="36"/>
        <source>Please select one or several %1:
(press CTRL+A to select all)</source>
        <translation>请选择一个或多个 %1:
(请 CTRL+A 选择全部)</translation>
    </message>
    <message>
        <location filename="../ecvItemSelectionDlg.cpp" line="41"/>
        <source>Please select one %1</source>
        <translation>请选择一个 %1</translation>
    </message>
    <message>
        <location filename="../ecvItemSelectionDlg.cpp" line="98"/>
        <source>entity</source>
        <translation>实体</translation>
    </message>
    <message>
        <location filename="../ecvItemSelectionDlg.cpp" line="124"/>
        <source>entities</source>
        <translation>实体</translation>
    </message>
</context>
<context>
    <name>ccPluginInfoDlg</name>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="14"/>
        <source>Dialog</source>
        <translation>对话框</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="96"/>
        <source>Description</source>
        <translation>描述</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="109"/>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="182"/>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="256"/>
        <source>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:&apos;.SF NS Text&apos;; font-size:13pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;&quot;&gt;&lt;br /&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="119"/>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="263"/>
        <source>(none listed)</source>
        <translation>(没有列出)</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="136"/>
        <source>Plugins</source>
        <translation>插件</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="149"/>
        <source>Maintainers</source>
        <translation>维护人员</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="189"/>
        <source>(CLOUDVIEWER  Team)</source>
        <translation>(逸舟点云处理团队)</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="230"/>
        <source>Authors</source>
        <translation>作者</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="246"/>
        <source>References</source>
        <translation>参考</translation>
    </message>
    <message>
        <location filename="../pluginManager/ui/ecvPluginInfoDlg.ui" line="278"/>
        <source>CLOUDVIEWER  looks for plugins in the following directories:</source>
        <translation>CLOUDVIEWER  从下列路径搜索插件:</translation>
    </message>
    <message>
        <location filename="../pluginManager/ecvPluginInfoDlg.cpp" line="78"/>
        <source>About Plugins</source>
        <translation>关于插件</translation>
    </message>
    <message>
        <location filename="../pluginManager/ecvPluginInfoDlg.cpp" line="176"/>
        <source>(No plugin selected)</source>
        <translation>(没有插件选择)</translation>
    </message>
    <message>
        <location filename="../pluginManager/ecvPluginInfoDlg.cpp" line="213"/>
        <source>PCL Algorithm</source>
        <translation>PCL 算法</translation>
    </message>
    <message>
        <location filename="../pluginManager/ecvPluginInfoDlg.cpp" line="224"/>
        <source>I/O</source>
        <translation>输入/输出</translation>
    </message>
</context>
<context>
    <name>ccPluginManager</name>
    <message>
        <location filename="../../common/ecvPluginManager.cpp" line="195"/>
        <source>[Plugin] Searching: %1</source>
        <translation>[插件] 搜索：%1</translation>
    </message>
    <message>
        <location filename="../../common/ecvPluginManager.cpp" line="215"/>
        <source>	%1 does not seem to be a valid plugin	(%2)</source>
        <translation>	%1 好像不是一个有效的插件	(%2)</translation>
    </message>
    <message>
        <location filename="../../common/ecvPluginManager.cpp" line="226"/>
        <source>	%1 does not seem to be a valid plugin or it is not supported by this version</source>
        <translation>	%1 好像不是有效插件 或者当前版本暂时不支持该插件</translation>
    </message>
    <message>
        <location filename="../../common/ecvPluginManager.cpp" line="237"/>
        <source>	Plugin %1 has a blank name</source>
        <translation>	插件 %1 名称为空</translation>
    </message>
    <message>
        <location filename="../../common/ecvPluginManager.cpp" line="261"/>
        <source>	%1 overridden</source>
        <translation>	%1 overridden</translation>
    </message>
    <message>
        <location filename="../../common/ecvPluginManager.cpp" line="270"/>
        <source>	Plugin found: %1 (%2)</source>
        <translation>	发现插件: %1 (%2)</translation>
    </message>
</context>
<context>
    <name>ccPluginUIManager</name>
    <message>
        <location filename="../pluginManager/ecvPluginUIManager.cpp" line="331"/>
        <location filename="../pluginManager/ecvPluginUIManager.cpp" line="342"/>
        <location filename="../pluginManager/ecvPluginUIManager.cpp" line="451"/>
        <source>Plugins</source>
        <translation>插件</translation>
    </message>
    <message>
        <location filename="../pluginManager/ecvPluginUIManager.cpp" line="335"/>
        <source>PCL Algorithm</source>
        <translation>PCL算法</translation>
    </message>
    <message>
        <location filename="../pluginManager/ecvPluginUIManager.cpp" line="343"/>
        <location filename="../pluginManager/ecvPluginUIManager.cpp" line="457"/>
        <source>PCL ALgorithms</source>
        <translation>PCL算法</translation>
    </message>
</context>
<context>
    <name>ccPropertiesTreeDelegate</name>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="192"/>
        <source>Property</source>
        <translation>属性</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="193"/>
        <source>State/Value</source>
        <translation>状态/值</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="265"/>
        <source>Transformation history</source>
        <translation>空间转换记录</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="270"/>
        <source>Display transformation</source>
        <translation>显示窗口转换</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="345"/>
        <source>Meta data</source>
        <translation>元数据</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="372"/>
        <source>ECV Object</source>
        <translation>数据对象</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="375"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="761"/>
        <source>Name</source>
        <translation>名称</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="379"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="529"/>
        <source>Visible</source>
        <translation>可视的</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="383"/>
        <source>Normals</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="386"/>
        <source>Show name (in 3D)</source>
        <translation>显示3D名称</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="390"/>
        <source>Colors</source>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="412"/>
        <source>Local box dimensions</source>
        <translation>外包围框尺寸</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="412"/>
        <source>Box dimensions</source>
        <translation>外包围框尺寸</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="417"/>
        <source>Box center</source>
        <translation>外包框中心</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="423"/>
        <source>Info</source>
        <translation>信息</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="423"/>
        <source>Object ID: %1 - Children: %2</source>
        <translation>对象 ID: %1 - 子对象: %2</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="427"/>
        <source>Current Display</source>
        <translation>当前显示窗口</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="436"/>
        <source>Global shift</source>
        <translation>全局偏移量</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="439"/>
        <source>Global scale</source>
        <translation>全局比例</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="446"/>
        <source>Cloud</source>
        <translation>点云</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="449"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="997"/>
        <source>Points</source>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="455"/>
        <source>Point size</source>
        <translation>点大小</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="470"/>
        <source>Scan grids</source>
        <translation>扫描网格</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="472"/>
        <source>Scan grid</source>
        <translation>扫描网格</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="478"/>
        <source>Scan #%1</source>
        <translation>扫描 # %1</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="478"/>
        <source>%1 x %2 (%3 points)</source>
        <translation>%1 x %2 (%3 点)</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="485"/>
        <source>Waveform</source>
        <translation>波形</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="486"/>
        <source>Waves</source>
        <translation>波</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="487"/>
        <source>Descriptors</source>
        <translation>描述子</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="490"/>
        <source>Data size</source>
        <translation>数据大小</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="508"/>
        <source>Scalar Fields</source>
        <translation>标量字段</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="508"/>
        <source>Scalar Field</source>
        <translation>标量字段</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="511"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="778"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="808"/>
        <source>Count</source>
        <translation>数量</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="514"/>
        <source>Active</source>
        <translation>当前的</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="520"/>
        <source>Color Scale</source>
        <translation>色标</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="523"/>
        <source>Current</source>
        <translation>当前的</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="526"/>
        <source>Steps</source>
        <translation>步数</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="531"/>
        <source>SF display params</source>
        <translation>标量字段显示参数</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="543"/>
        <source>Primitive</source>
        <translation>基础模型</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="546"/>
        <source>Type</source>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="551"/>
        <source>Drawing precision</source>
        <translation>绘制精度</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="556"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="563"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="560"/>
        <source>Height</source>
        <translation>高度</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="567"/>
        <source>Bottom radius</source>
        <translation>底部半径</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="568"/>
        <source>Top radius</source>
        <translation>顶部半径</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="582"/>
        <source>Facet</source>
        <translation>面片</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="588"/>
        <source>Surface</source>
        <translation>曲面</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="591"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="714"/>
        <source>RMS</source>
        <translation>均方根</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="594"/>
        <source>Center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="598"/>
        <source>Show contour</source>
        <translation>显示轮廓</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="602"/>
        <source>Show polygon</source>
        <translation>显示多边形</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="609"/>
        <source>Normal</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="614"/>
        <source>Dip / Dip dir.</source>
        <translation>倾斜 / 倾斜方向.</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="617"/>
        <source>Show normal vector</source>
        <translation>显示法向量</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="626"/>
        <source>Sub-mesh</source>
        <translation>子网格</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="626"/>
        <source>Mesh</source>
        <translation>网格</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="629"/>
        <source>Faces</source>
        <translation>面</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="633"/>
        <source>Materials/textures</source>
        <translation>材质/纹理</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="636"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="1667"/>
        <source>Wireframe</source>
        <translation>线框</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="639"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="1649"/>
        <source>Pointsframe</source>
        <translation>点框</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="643"/>
        <source>Stippling</source>
        <translation>画点画</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="655"/>
        <source>Polyline</source>
        <translation>多段线</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="658"/>
        <source>Vertices</source>
        <translation>顶点</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="661"/>
        <source>Length</source>
        <translation>长度</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="664"/>
        <source>Line width</source>
        <translation>线宽</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="674"/>
        <source>Octree</source>
        <translation>八叉树</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="677"/>
        <source>Display mode</source>
        <translation>显示模式</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="680"/>
        <source>Display level</source>
        <translation>显示层级</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="682"/>
        <source>Current level</source>
        <translation>当前层</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="690"/>
        <source>Cell size</source>
        <translation>单元大小</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="694"/>
        <source>Cell count</source>
        <translation>单元数量</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="697"/>
        <source>Filled volume</source>
        <translation>填充体积</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="704"/>
        <source>Kd-tree</source>
        <translation>Kd树</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="707"/>
        <source>Max Error</source>
        <translation>最大误差</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="717"/>
        <source>Max dist @ 68%</source>
        <translation>最大距离 @ 68%</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="720"/>
        <source>Max dist @ 95%</source>
        <translation>最大距离 @ 95%</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="723"/>
        <source>Max dist @ 99%</source>
        <translation>最大距离 @ 99%</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="726"/>
        <source>Max distance</source>
        <translation>最大距离</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="730"/>
        <source>unknown</source>
        <translation>未知的</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="733"/>
        <source>Error measure</source>
        <translation>误差测量</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="741"/>
        <source>Label</source>
        <translation>标签</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="745"/>
        <source>Body</source>
        <translation>主体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="748"/>
        <source>Show 2D label</source>
        <translation>显示2D标签</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="751"/>
        <source>Show legend(s)</source>
        <translation>显示图例</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="758"/>
        <source>Viewport</source>
        <translation>视窗</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="761"/>
        <source>undefined</source>
        <translation>未定义的</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="764"/>
        <source>Apply viewport</source>
        <translation>应用视口</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="767"/>
        <source>Update viewport</source>
        <translation>更新视口</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="775"/>
        <source>Trans. buffer</source>
        <translation>转换 缓冲</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="781"/>
        <source>Show path</source>
        <translation>显示路径</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="784"/>
        <source>Show trihedrons</source>
        <translation>显示三面体</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="787"/>
        <source>Scale</source>
        <translation>比例</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="795"/>
        <source>Drawing scale</source>
        <translation>绘制比例</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="798"/>
        <source>Apply Viewport</source>
        <translation>应用视口</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="801"/>
        <source>Position/Orientation</source>
        <translation>位置/方向</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="805"/>
        <source>Associated positions</source>
        <translation>关联位置</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="815"/>
        <source>Indexes</source>
        <translation>索引</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="818"/>
        <source>Active index</source>
        <translation>当前索引</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="826"/>
        <source>Array</source>
        <translation>数组</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="830"/>
        <source>Shared</source>
        <translation>共享</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="922"/>
        <source>None</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="996"/>
        <source>Wire</source>
        <translation>线框</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="998"/>
        <source>Plain cubes</source>
        <translation>纯矩形框</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="1114"/>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="1124"/>
        <source>Apply</source>
        <translation>应用</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="1133"/>
        <source>Update</source>
        <translation>更新</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="1844"/>
        <source>Internal error: color scale doesn&apos;t seem to exist anymore!</source>
        <translation>内部错误：颜色域丢失或者不存在！</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="2038"/>
        <source>[ApplySensorViewport] Viewport applied</source>
        <translation>[ApplySensorViewport] 应用视口</translation>
    </message>
    <message>
        <location filename="../db_tree/ecvPropertiesTreeDelegate.cpp" line="2065"/>
        <source>Viewport &apos;%1&apos; has been updated</source>
        <translation>视口 &apos;%1&apos; 已更新</translation>
    </message>
</context>
<context>
    <name>ccRasterizeTool</name>
    <message>
        <location filename="../ecvRasterizeTool.cpp" line="1911"/>
        <source>Contour plot</source>
        <translation>轮廓绘制</translation>
    </message>
    <message>
        <location filename="../ecvRasterizeTool.cpp" line="1912"/>
        <source>Levels: %1
Cells: %2 x %3</source>
        <translation>层: %1
单元: %2 x %3</translation>
    </message>
</context>
<context>
    <name>ccRenderToFileDlg</name>
    <message>
        <location filename="../ecvRenderToFileDlg.cpp" line="107"/>
        <source>Save Image</source>
        <translation>保存图像</translation>
    </message>
</context>
<context>
    <name>ccTranslationManager</name>
    <message>
        <location filename="../../common/ecvTranslationManager.cpp" line="75"/>
        <source>No Translation (English)</source>
        <translation>无翻译（默认语言英语）</translation>
    </message>
    <message>
        <location filename="../../common/ecvTranslationManager.cpp" line="179"/>
        <source>Language Change</source>
        <translation>修改语言</translation>
    </message>
    <message>
        <location filename="../../common/ecvTranslationManager.cpp" line="180"/>
        <source>Language change will take effect when ACloudViewer is restarted</source>
        <translation>语言在ACloudViewer重启后生效</translation>
    </message>
</context>
<context>
    <name>commandLineDlg</name>
    <message>
        <location filename="../ui_templates/commandLineDlg.ui" line="14"/>
        <source>CLOUDVIEWER  - command line mode</source>
        <translation>CLOUDVIEWER  - 命令行模式</translation>
    </message>
</context>
<context>
    <name>compassDlg</name>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="20"/>
        <source>Compass</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="38"/>
        <source>Mode:</source>
        <translation>模式:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="45"/>
        <source>Activate compass mode to make structural measurements</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="68"/>
        <source>Activate map mode to define geological features</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="92"/>
        <source>Tool:</source>
        <translation>工具:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="105"/>
        <source>Picking Tool. Use this to select GeoObjects or measurements.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="122"/>
        <source>Plane Tool: Measure surface orientations</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="148"/>
        <source>Trace Tool: Measure orientation from structure trace</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="168"/>
        <source>Lineation Tool: Measure distances and directions</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="185"/>
        <source>Other Tools</source>
        <translation>其他工具</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="212"/>
        <source>Change tool and visibility settings</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="235"/>
        <source>Undo last action</source>
        <translation>撤消最后的动作</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="242"/>
        <source>Ctrl+Z</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="249"/>
        <source>Export interpretation and measurements</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="252"/>
        <source>Save current label (added to cloud children)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="263"/>
        <source>Show readme and help information</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="274"/>
        <source>Accept latest changes</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="281"/>
        <source>Return</source>
        <translation>返回</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="288"/>
        <source>Exit ccCompass plugin. Thanks for visiting :)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/compassDlg.ui" line="295"/>
        <source>Esc</source>
        <translation>Esc</translation>
    </message>
</context>
<context>
    <name>ecvFilterWindowTool</name>
    <message>
        <location filename="../ecvFilterWindowTool.cpp" line="641"/>
        <source>Preparing extraction</source>
        <translation>准备提取</translation>
    </message>
    <message>
        <location filename="../ecvFilterWindowTool.cpp" line="655"/>
        <source>Cloud &apos;%1</source>
        <translation>Cloud &apos; %1</translation>
    </message>
    <message>
        <location filename="../ecvFilterWindowTool.cpp" line="656"/>
        <source>Points: %L1</source>
        <translation>Points:% L1</translation>
    </message>
</context>
<context>
    <name>ecvRecentFiles</name>
    <message>
        <location filename="../ecvRecentFiles.cpp" line="40"/>
        <source>Open Recent...</source>
        <translation>打开最近……</translation>
    </message>
    <message>
        <location filename="../ecvRecentFiles.cpp" line="42"/>
        <source>Clear Menu</source>
        <translation>清空历史记录</translation>
    </message>
</context>
<context>
    <name>mapDlg</name>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="20"/>
        <source>Map</source>
        <translation>地图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="38"/>
        <source>GeoObjects:</source>
        <translation>几何对象:</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="48"/>
        <source>Create new GeoObject</source>
        <translation>创建新的GeoObject</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="86"/>
        <source>No Selection</source>
        <translation>没有选择</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="176"/>
        <source>Digitise to lower-contact of GeoObject</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="179"/>
        <source>Lower</source>
        <translation>较低的</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="259"/>
        <source>Digitise to upper-contact of GeoObject</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="262"/>
        <source>Upper</source>
        <translation>向上</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="284"/>
        <source>Digitise to interior of geo-object</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCompass/ui/mapDlg.ui" line="287"/>
        <source>Interior</source>
        <translation>室内</translation>
    </message>
</context>
<context>
    <name>pointPairRegistrationDlg</name>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="14"/>
        <source>Point list picking</source>
        <translation>点拾取列表</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="25"/>
        <source>show &apos;to align&apos; cloud</source>
        <translation>显示待配准实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="64"/>
        <source>Pick spheres instead of single points (for clouds only)</source>
        <translation>拾取球体而不是单点（仅点云）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="81"/>
        <source>search radius (or the spheres radius if you know it)</source>
        <translation>搜索半径（或者已知球体半径）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="106"/>
        <source>Max RMS (as a percentage of the radius)</source>
        <translation>最大均方根（半径百分比）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="184"/>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="294"/>
        <source>Error</source>
        <translation>误差</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="202"/>
        <source>show &apos;reference&apos; cloud</source>
        <translation>显示参考实体</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="309"/>
        <source>adjust scale</source>
        <translation>调整缩放比例</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="335"/>
        <source>Rotation</source>
        <translation>旋转</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="351"/>
        <source>Constrains the rotation around a single axis (warning: experimental)</source>
        <translation>限制在单轴旋转（警告：实验性质的）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="403"/>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="422"/>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="441"/>
        <source>Constrains the translation along particular axes (warning: experimental)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="458"/>
        <source>auto update zoom</source>
        <translation>自动更新视图缩放</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="490"/>
        <source>align</source>
        <translation>对齐</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="500"/>
        <source>reset</source>
        <translation>重置</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="510"/>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="513"/>
        <source>Convert list to new cloud (and close dialog)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="516"/>
        <source>to cloud</source>
        <translation>到点云</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="527"/>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="530"/>
        <source>Close dialog (list will be lost)</source>
        <translation>关闭对话框（列表数据会丢失）</translation>
    </message>
    <message>
        <location filename="../ui_templates/pointPairRegistrationDlg.ui" line="533"/>
        <source>stop</source>
        <translation>停止</translation>
    </message>
</context>
<context>
    <name>qAnimation</name>
    <message>
        <location filename="../../plugins/core/qAnimation/qAnimation.cpp" line="72"/>
        <source>%1
At least 2 viewports must be selected.</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/qAnimation.cpp" line="104"/>
        <source>No active 3D view!</source>
        <translation>没有活动3D视图!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/qAnimation.cpp" line="112"/>
        <source>[qAnimation] Selected viewports: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/qAnimation.cpp" line="118"/>
        <source>Failed to initialize the plugin dialog (not enough memory?)</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>qAnimationDlg</name>
    <message>
        <location filename="../../plugins/core/qAnimation/src/qAnimationDlg.cpp" line="279"/>
        <source>Output animation file</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qAnimation/src/qAnimationDlg.cpp" line="283"/>
        <source>Open Directory</source>
        <translation>开的目录</translation>
    </message>
</context>
<context>
    <name>qCSF</name>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="98"/>
        <source>Select only one cloud!</source>
        <translation>只能选择一个点云！</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="108"/>
        <source>Select a real point cloud!</source>
        <translation>请选择有效点云！</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="124"/>
        <source>Not enough memory!</source>
        <translation>内存不足!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="182"/>
        <source>Computing....</source>
        <translation>计算....</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="205"/>
        <source>Process failed</source>
        <translation>处理失败</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="209"/>
        <source>[CSF] %1% of points classified as ground points</source>
        <translation>[CSF] %1% 的点作为地面点</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="210"/>
        <source>[CSF] Timing: %1 s.</source>
        <translation>[CSF] 计时: %1 s.</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="228"/>
        <source>Failed to extract the ground subset (not enough memory)</source>
        <translation>抽取地面子集失败（内存不足）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="248"/>
        <source>Failed to extract the off-ground subset (not enough memory)</source>
        <translation>抽取离地点子集失败（内存不足）</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="268"/>
        <source>ground points</source>
        <translation>地面点</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCSF/qCSF.cpp" line="275"/>
        <source>off-ground points</source>
        <translation>离地点</translation>
    </message>
</context>
<context>
    <name>qCanupoPlugin</name>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="82"/>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="83"/>
        <source>Train classifier</source>
        <translation>训练分类器</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="91"/>
        <source>Classify</source>
        <translation>分类</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="92"/>
        <source>Classify cloud</source>
        <translation>云进行分类</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="117"/>
        <source>Select one and only one point cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="162"/>
        <source>Internal error: failed to access core pointss?!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="183"/>
        <source>Failed to compute sub-sampled core points!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="193"/>
        <source>.core points (subsampled @ %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="200"/>
        <source>Can&apos;t save subsampled cloud (not enough memory)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="210"/>
        <source>MSC core points</source>
        <translation>MSC核心分</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="227"/>
        <source>[qCanupo] </source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="240"/>
        <source>Internal error: no core point source specified?!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="292"/>
        <source>Invalid scale parameters!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="305"/>
        <source>At least one cloud (class #1 or #2) was not defined!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="321"/>
        <source>Internal error: unhandled descriptor ID (%1)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="332"/>
        <source>To compute this type of descriptor, all clouds must have an active scalar field!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="362"/>
        <source>Failed to compute sub-sampled version of evaluation points!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="380"/>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="391"/>
        <source>Failed to compute sub-sampled version of cloud #1!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="413"/>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="439"/>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="466"/>
        <source>Failed to compute core points descriptors: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="418"/>
        <source>[qCanupo] Some descriptors couldn&apos;t be computed on cloud#1 (min scale may be too small)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="444"/>
        <source>[qCanupo] Some descriptors couldn&apos;t be computed on cloud#2 (min scale may be too small)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="472"/>
        <source>[qCanupo] Some descriptors couldn&apos;t be computed on evaluation cloud (min scale may be too small)!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qCanupo/qCanupo.cpp" line="495"/>
        <source>[qCanupo] Classifier training failed...</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>qFacets</name>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="99"/>
        <source>closing facets dialog failed! [%1]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="109"/>
        <source>Extract facets (Kd-tree)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="110"/>
        <source>Detect planar facets by fusing Kd-tree cells</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="118"/>
        <source>Extract facets (Fast Marching)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="119"/>
        <source>Detect planar facets with Fast Marching</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="127"/>
        <source>Export facets (SHP)</source>
        <translation>导出方面(SHP)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="128"/>
        <source>Exports one or several facets to a shapefile</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="136"/>
        <source>Export facets info (CSV)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="137"/>
        <source>Exports various information on a set of facets (ASCII CSV file)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="145"/>
        <source>Classify facets by orientation</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="146"/>
        <source>Classifies facets based on their orienation (dip &amp; dip direction)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="156"/>
        <source>Show stereogram</source>
        <translation>显示立体图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="157"/>
        <source>Computes and displays a stereogram (+ interactive filtering)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="214"/>
        <source>Select one and only one point cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="220"/>
        <source>Internal error: invalid algorithm type!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="292"/>
        <source>Couldn&apos;t allocate a new scalar field for computing fusion labels! Try to free some memory ...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="314"/>
        <source>[qFacets] Kd-tree construction timing: %1 s</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="328"/>
        <source>Failed to build Kd-tree! (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="355"/>
        <source>Failed to extract fused components! (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="374"/>
        <source> [Kd-tree][error &lt; %1][angle &lt; %2 deg.]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="377"/>
        <source> [FM][level %2][error &lt; %1]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="385"/>
        <source>[qFacets] %1 facet(s) where created from cloud &apos;%2&apos;</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="390"/>
        <source>Error(s) occurred during the generation of facets! Result may be incomplete</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="405"/>
        <source>An error occurred during the generation of facets!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="411"/>
        <source>No facet remains! Check the parameters (min size, etc.)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="419"/>
        <source>An error occurred during the fusion process!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="449"/>
        <source> [facets]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="460"/>
        <source>Facets creation</source>
        <translation>方面创造</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="461"/>
        <source>Components: %1</source>
        <translation>Components: %1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="489"/>
        <source>facet %1 (rms=%2)</source>
        <translation>面片 %1 (rms = %2)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="707"/>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="1072"/>
        <source>Couldn&apos;t find any facet in the current selection!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="733"/>
        <source>File already exists!</source>
        <translation>文件已经存在!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="734"/>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="1101"/>
        <source>File already exists! Are you sure you want to overwrite it?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="740"/>
        <source>index</source>
        <translation>指数</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="741"/>
        <source>surface</source>
        <translation>表面</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="742"/>
        <source>rms</source>
        <translation>rms</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="743"/>
        <source>dip_dir</source>
        <translation>dip_dir</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="744"/>
        <source>dip</source>
        <translation>浸</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="745"/>
        <source>family_ind</source>
        <translation>family_ind</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="746"/>
        <source>subfam_ind</source>
        <translation>subfam_ind</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="747"/>
        <source>normal</source>
        <translation>法线</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="748"/>
        <source>center</source>
        <translation>中心</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="749"/>
        <source>horiz_ext</source>
        <translation>horiz_ext</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="750"/>
        <source>vert_ext</source>
        <translation>vert_ext</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="751"/>
        <source>surf_ext</source>
        <translation>surf_ext</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="772"/>
        <source>Not enough memory!</source>
        <translation>没有足够的内存!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="777"/>
        <source>facets</source>
        <translation>方面</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="886"/>
        <source>Failed to change the orientation of polyline &apos;%1&apos;! (not enough memory)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="944"/>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="1164"/>
        <source>[qFacets] File &apos;%1&apos; successfully saved</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="948"/>
        <source>[qFacets] Failed to save file &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="970"/>
        <source>Select a group of facets or a point cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="1006"/>
        <source>Select a group of facets!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="1046"/>
        <source>An error occurred while classifying the facets! (not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="1100"/>
        <source>Overwrite</source>
        <translation>覆盖</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qFacets/qFacets.cpp" line="1111"/>
        <source>Failed to open file for writing! Check available space and access rights</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>qHPR</name>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="222"/>
        <source>Select only one cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="230"/>
        <source>No active window!</source>
        <translation>没有活动窗口!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="239"/>
        <source>Perspective mode only!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="268"/>
        <source>Couldn&apos;t compute octree!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="298"/>
        <source>Error while simplifying point cloud with octree!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="305"/>
        <source>[HPR] Cells: %1 - Time: %2 s</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="337"/>
        <source>Couldn&apos;t fetch the list of octree cell indexes! (Not enough memory?)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="359"/>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="397"/>
        <source>Not enough memory!</source>
        <translation>没有足够的内存!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="369"/>
        <source>[HPR] Visible points: %1</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="373"/>
        <source>No points were removed!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="384"/>
        <source>.visible_points</source>
        <translation>.visible_points</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qHPR/qHPR.cpp" line="388"/>
        <source>Viewport</source>
        <translation>视窗</translation>
    </message>
</context>
<context>
    <name>qM3C2Plugin</name>
    <message>
        <location filename="../../plugins/core/qM3C2/qM3C2.cpp" line="77"/>
        <source>Select two point clouds!</source>
        <translation>请选择两个点云！</translation>
    </message>
</context>
<context>
    <name>qPCL</name>
    <message>
        <location filename="../../plugins/core/qPCL/qPCL.cpp" line="91"/>
        <source>Filters</source>
        <translation>点云滤波</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/qPCL.cpp" line="101"/>
        <source>Surfaces</source>
        <translation>曲面重建</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/qPCL.cpp" line="110"/>
        <source>Segmentations</source>
        <translation>点云分割</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCL/qPCL.cpp" line="118"/>
        <source>Recognitions</source>
        <translation>点云识别</translation>
    </message>
</context>
<context>
    <name>qPCV</name>
    <message>
        <location filename="../../plugins/core/qPCV/qPCV.cpp" line="280"/>
        <source>An error occurred during entity &apos;%1&apos; illumination!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/qPCV.cpp" line="292"/>
        <source>Entity &apos;%1&apos; normals have been automatically disabled</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPCV/qPCV.cpp" line="309"/>
        <source>Process has been cancelled by the user</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>qPoissonRecon</name>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="296"/>
        <source>Select only one cloud!</source>
        <translation>仅能选择一个点云！</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="306"/>
        <source>Select a cloud!</source>
        <translation>选择一个点云!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="314"/>
        <source>Cloud must have normals!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="396"/>
        <source>vertices</source>
        <translation>顶点</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="404"/>
        <source>[PoissonRecon] Job started (level %1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="407"/>
        <source>Initialization</source>
        <translation>初始化</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="408"/>
        <source>Poisson Reconstruction</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="412"/>
        <source>Reconstruction in progress
</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="414"/>
        <source>level: %1</source>
        <translation>层: %1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="416"/>
        <source>resolution: %1</source>
        <translation>分辨率: %1</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="417"/>
        <source> [%1 thread(s)]</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="466"/>
        <source>Reconstruction failed!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="472"/>
        <source>[PoissonRecon] Job finished (%1 triangles, %2 vertices)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qPoissonRecon/qPoissonRecon.cpp" line="475"/>
        <source>Mesh[%1] (level %2)</source>
        <translation>网格[%1] (层 %2)</translation>
    </message>
</context>
<context>
    <name>qRansacSD</name>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="123"/>
        <source>Select only one cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="131"/>
        <source>Select a real point cloud!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="154"/>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="403"/>
        <source>Not enough memory!</source>
        <translation>没有足够的内存!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="223"/>
        <source>No primitive type selected!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="248"/>
        <source>Computing normals (please wait)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="250"/>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="309"/>
        <source>Ransac Shape Detection</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="273"/>
        <source>Not enough memory to compute normals!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="308"/>
        <source>Operation in progress (please wait)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="373"/>
        <source>Segmentation failed...</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="389"/>
        <source>Inconsistent result!</source>
        <translation>不一致的结果!</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="486"/>
        <source>Sphere (r=%1)</source>
        <translation>球体 (r = %1)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="511"/>
        <source>Cylinder (r=%1/h=%2)</source>
        <translation>圆柱体 (r = %1 / h = %2)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="554"/>
        <source>Cone (alpha=%1/h=%2)</source>
        <translation>类圆锥体 (alpha = %1/h = %2)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="593"/>
        <source>[qRansacSD] Apple-shaped torus are not handled by CLOUDVIEWER !</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="603"/>
        <source>Torus (r=%1/R=%2)</source>
        <translation>类圆环体 (r = %1/R = %2)</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="635"/>
        <source>Ransac Detected Shapes (%1)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qRANSAC_SD/qRANSAC_SD.cpp" line="650"/>
        <source>[qRansacSD] Input cloud has been automtically hidden!</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>qSRA</name>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="64"/>
        <source>Load profile</source>
        <translation>加载配置文件</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="65"/>
        <source>Loads the 2D profile of a Surface of Revolution (from a dedicated ASCII file)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="73"/>
        <source>Cloud-SurfRev radial distance</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="74"/>
        <source>Computes the radial distances between a cloud and a Surface of Revolution (polyline/profile, cone or cylinder)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="82"/>
        <source>2D distance map</source>
        <translation>2 d距离地图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="83"/>
        <source>Creates the 2D deviation map (radial distances) from a Surface or Revolution (unroll)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="188"/>
        <source>Failed to load file &apos;%1&apos;!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="241"/>
        <source>[qSRA] File &apos;%1&apos; successfully loaded</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="377"/>
        <source>Generate map</source>
        <translation>生成地图</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="378"/>
        <source>Do you want to generate a 2D deviation map?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="390"/>
        <source>Select exactly one cloud and one Surface of Revolution (polyline/profile, cone or cylinder)</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="425"/>
        <source>An error occurred while computing radial distances!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="505"/>
        <source>Distance field</source>
        <translation>距离场</translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="506"/>
        <source>Cloud has no &apos;%1&apos; field. Do you want to use the active scalar field instead?</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="516"/>
        <source>Cloud has no no &apos;%1&apos; field and no active scalar field!</source>
        <translation></translation>
    </message>
    <message>
        <location filename="../../plugins/core/qSRA/qSRA.cpp" line="521"/>
        <source>You can compute the radial distances with the &apos;%1&apos; method</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>scalarFieldFromColorDlg</name>
    <message>
        <location filename="../ui_templates/scalarFieldFromColorDlg.ui" line="14"/>
        <source>SF from RGB</source>
        <translation>从RGB科幻</translation>
    </message>
    <message>
        <location filename="../ui_templates/scalarFieldFromColorDlg.ui" line="22"/>
        <source>R Channel</source>
        <translation>R通道</translation>
    </message>
    <message>
        <location filename="../ui_templates/scalarFieldFromColorDlg.ui" line="32"/>
        <source>G Channel</source>
        <translation>G通道</translation>
    </message>
    <message>
        <location filename="../ui_templates/scalarFieldFromColorDlg.ui" line="42"/>
        <source>B Channel</source>
        <translation>B通道</translation>
    </message>
    <message>
        <location filename="../ui_templates/scalarFieldFromColorDlg.ui" line="52"/>
        <source>Composite = (R+G+B)/3</source>
        <translation></translation>
    </message>
</context>
<context>
    <name>sensorComputeDistancesDlg</name>
    <message>
        <location filename="../ui_templates/sensorComputeDistancesDlg.ui" line="14"/>
        <source>Sensor range computation</source>
        <translation>传感器范围计算</translation>
    </message>
    <message>
        <location filename="../ui_templates/sensorComputeDistancesDlg.ui" line="20"/>
        <source>Squared distances</source>
        <translation>距离平方</translation>
    </message>
</context>
</TS>
