<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN" sourcelanguage="en">
<context>
    <name>SpirulaDialog</name>
    <message>
        <source>Spirula Studio</source>
        <translation>Spirula 工作室</translation>
    </message>
    <message>
        <source>&lt;b&gt;Spirula Studio&lt;/b&gt; — 3D Gaussian Splatting trainer (official upstream GUI)</source>
        <translation>&lt;b&gt;Spirula 工作室&lt;/b&gt; — 3D 高斯泼溅训练器(官方上游 GUI)</translation>
    </message>
    <message>
        <source>Launch Spirula Studio GUI</source>
        <translation>启动 Spirula 工作室图形界面</translation>
    </message>
    <message>
        <source>Start the official Spirula Studio application (training, editing, rendering, meshing)</source>
        <translation>启动官方 Spirula 工作室应用(训练、编辑、渲染、网格化)</translation>
    </message>
    <message>
        <source>Import Result to DB Tree…</source>
        <translation>导入结果到数据库树…</translation>
    </message>
    <message>
        <source>Add a trained splat.ply / mesh.ply (or a run directory) to the ACloudViewer DB tree</source>
        <translation>将训练得到的 splat.ply / mesh.ply(或运行目录)加入 ACloudViewer 数据库树</translation>
    </message>
    <message>
        <source>Run Headless Training…</source>
        <translation>运行无界面训练…</translation>
    </message>
    <message>
        <source>Drive `spirula train` headlessly with live output and cancel support</source>
        <translation>以无界面方式驱动 spirula train,支持实时输出与取消</translation>
    </message>
    <message>
        <source>Standalone GPLv3 executable (zero patches; sync = SPIRULA_TAG bump + smoke tests).</source>
        <translation>独立 GPLv3 可执行程序(零补丁;同步 = 升级 SPIRULA_TAG + 冒烟测试)。</translation>
    </message>
    <message>
        <source>Executable: &lt;code&gt;%1&lt;/code&gt;</source>
        <translation>可执行文件:&lt;code&gt;%1&lt;/code&gt;</translation>
    </message>
    <message>
        <source>&lt;span style=&quot;color:#c00&quot;&gt;Spirula Studio executable not found.&lt;/span&gt; Build it with -DBUILD_SPIRULA=ON or place 'spirula' on the PATH.</source>
        <translation>&lt;span style=&quot;color:#c00&quot;&gt;未找到 Spirula 工作室可执行文件。&lt;/span&gt;请使用 -DBUILD_SPIRULA=ON 构建它,或将 spirula 加入 PATH。</translation>
    </message>
</context>
<context>
    <name>SpirulaTrainDialog</name>
    <message>
        <source>Spirula Studio — Headless Training</source>
        <translation>Spirula 工作室 — 无界面训练</translation>
    </message>
    <message>
        <source>Dataset folder (COLMAP model, Nerfstudio transforms.json, or plain images for the built-in SfM):</source>
        <translation>数据集目录(COLMAP 模型、Nerfstudio transforms.json,或供内置 SfM 使用的普通图像):</translation>
    </message>
    <message>
        <source>Run name:</source>
        <translation>运行名称:</translation>
    </message>
    <message>
        <source>Device:</source>
        <translation>设备:</translation>
    </message>
    <message>
        <source> spirula --device selector: auto, an index, a name substring, or uuid:hex</source>
        <translation> spirula --device 选择器:auto、序号、名称子串或 uuid:hex</translation>
    </message>
    <message>
        <source>Extra arguments (optional, passed through verbatim):</source>
        <translation>额外参数(可选,原样透传):</translation>
    </message>
    <message>
        <source>Start Training</source>
        <translation>开始训练</translation>
    </message>
    <message>
        <source>Cancel Training</source>
        <translation>取消训练</translation>
    </message>
    <message>
        <source>Training finished: %1</source>
        <translation>训练完成:%1</translation>
    </message>
    <message>
        <source>Training finished. Import the result into the DB tree now?</source>
        <translation>训练完成。是否立即将结果导入数据库树?</translation>
    </message>
    <message>
        <source>Training crashed — see the log above and &lt;config&gt;/crash.log.</source>
        <translation>训练崩溃 — 请查看上方日志与 &lt;config&gt;/crash.log。</translation>
    </message>
    <message>
        <source>Training exited with code %1.</source>
        <translation>训练以退出码 %1 结束。</translation>
    </message>
    <message>
        <source>Spirula Studio executable not found. Build it with -DBUILD_SPIRULA=ON.</source>
        <translation>未找到 Spirula 工作室可执行文件。请使用 -DBUILD_SPIRULA=ON 构建。</translation>
    </message>
    <message>
        <source>Dataset folder does not exist:
%1</source>
        <translation>数据集目录不存在:
%1</translation>
    </message>
</context>
<context>
    <name>qSpirula</name>
    <message>
        <source>Spirula Studio</source>
        <translation>Spirula 工作室</translation>
    </message>
    <message>
        <source>Spirula Studio — launch the official 3DGS trainer GUI, run headless training, and import results into the DB tree</source>
        <translation>Spirula 工作室 — 启动官方 3DGS 训练器界面、运行无界面训练,并将结果导入数据库树</translation>
    </message>
    <message>
        <source>Import Spirula result</source>
        <translation>导入 Spirula 结果</translation>
    </message>
    <message>
        <source>Importing Spirula result…</source>
        <translation>正在导入 Spirula 结果…</translation>
    </message>
    <message>
        <source>Import failed for '%1'.</source>
        <translation>导入 '%1' 失败。</translation>
    </message>
    <message>
        <source>No splat.ply found in '%1' (run directories and step-*.ckpt are resolved by their splat.ply).</source>
        <translation>在 '%1' 中未找到 splat.ply(运行目录与 step-*.ckpt 通过其 splat.ply 解析)。</translation>
    </message>
</context>
</TS>
