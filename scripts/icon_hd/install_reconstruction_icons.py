"""Create a provenance-checked 128px reconstruction icon set."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from .logo.icon import fit
from .logo.reconstructiontools import BUILDERS, UPSTREAM

ROOT=Path(__file__).resolve().parents[2]
MEDIA=ROOT/'app/reconstruction/media'
REVIEW=Path.home()/'ACloudViewer-icon-reviews/reconstruction-media'

def dimensions_128(data):
    text=data.decode('utf-8')
    match=re.search(r'<svg\b[^>]*>',text)
    if not match:raise RuntimeError('No svg root')
    tag=match.group()
    for attr in ('width','height'):
        pattern=rf'\b{attr}="[^"]*"'
        tag=re.sub(pattern,f'{attr}="128"',tag,count=1) if re.search(pattern,tag) else tag[:-1]+f' {attr}="128">'
    return (text[:match.start()]+tag+text[match.end():]).encode()

def run(paraview_root=Path('/home/ludahai/develop/code/github/ParaView')):
    if len(BUILDERS)!=29 or len(UPSTREAM)!=4:raise RuntimeError('Expected 29 authored paths and 4 source SVGs')
    REVIEW.mkdir(parents=True,exist_ok=True);originals=REVIEW/'originals';originals.mkdir(exist_ok=True);pvarchive=REVIEW/'paraview-originals';pvarchive.mkdir(exist_ok=True)
    snapshots={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'app/Resources/images').rglob('*') if p.is_file() and MEDIA not in p.parents}
    protect=REVIEW/'protected-other-images.json'
    if protect.exists():
        for relative,digest in json.loads(protect.read_text()).items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Out-of-scope image changed: {relative}')
    else:protect.write_text(json.dumps(snapshots,indent=2)+'\n')
    protected_media = REVIEW/'protected-media-sha256.json'
    protected_names=['LICENSE','PARAVIEW_NOTICE.txt','reconstruction_32.ico','reconstruction_64.ico','reconstruction_128.ico']
    protected_now={name:hashlib.sha256((MEDIA/name).read_bytes()).hexdigest() for name in protected_names}
    if protected_media.exists():
        if json.loads(protected_media.read_text())!=protected_now:
            raise RuntimeError('Reconstruction media license or application ICO changed outside this task')
    else:
        protected_media.write_text(json.dumps(protected_now,indent=2)+'\n')
    plans=[]
    for name,builder in BUILDERS.items():
        target=MEDIA/name;backup=originals/name;current=target.read_bytes()
        if backup.exists():
            if current!=backup.read_bytes():
                expected=dimensions_128(fit(builder()).encode())
                if current!=expected:raise RuntimeError(f'Unreviewed media edit: {name}')
        else:backup.write_bytes(current)
        final=dimensions_128(fit(builder()).encode());plans.append((name,target,backup,final,'redrawn-functional-svg',None))
    for target_name,source_name in UPSTREAM.items():
        target=MEDIA/target_name;source=paraview_root/'Qt/Components/Resources/Icons'/source_name;backup=originals/target_name
        if not source.is_file():raise RuntimeError(f'Missing matched ParaView source {source_name}')
        current=target.read_bytes()
        final=dimensions_128(source.read_bytes())
        if backup.exists():
            if current!=backup.read_bytes() and current!=final:raise RuntimeError(f'Unreviewed media edit: {target_name}')
        else:backup.write_bytes(current)
        pvcopy=pvarchive/source_name
        if pvcopy.exists() and pvcopy.read_bytes()!=source.read_bytes():raise RuntimeError(f'ParaView source snapshot changed: {source_name}')
        if not pvcopy.exists():pvcopy.write_bytes(source.read_bytes())
        plans.append((target_name,target,backup,final,'paraView-svg-root-size-only',str(pvcopy)))
    names={n for n,_,_,_,_,_ in plans}
    if names!={p.name for p in MEDIA.glob('*.svg')}:raise RuntimeError('Media SVG inventory changed; refresh the audited mapping before install')
    qrcs=[ROOT/'app/reconstruction/resources.qrc',ROOT/'libs/Reconstruction/src/ui/resources.qrc']
    for qrc in qrcs:
        tree=ET.parse(qrc).getroot()
        for name,_,_,_,_,_ in plans:
            found=[(group,node) for group in tree.findall('qresource') for node in group.findall('file') if Path((node.text or '').strip()).name==name]
            if len(found)!=1:raise RuntimeError(f'{qrc}: expected exactly one registration of {name}, found {len(found)}')
            nodepath=(qrc.parent/(found[0][1].text or '').strip()).resolve()
            if nodepath!=(MEDIA/name).resolve():raise RuntimeError(f'QRC source mismatch: {qrc} {name} {nodepath}')
    for name,target,backup,data,kind,source in plans:
        ET.fromstring(data)
        target.write_bytes(data)
    description = {
      'automatic-reconstruction.svg':'多相机自动重建流程','bundle-adjustment.svg':'多相机位姿与稀疏点联合BA','database-management.svg':'影像数据库与特征匹配管理',
      'dense-reconstruction.svg':'多视图稠密重建点面','export-all.svg':'导出全部模型','export-as-text.svg':'按文本格式导出模型（原图文字未确认）','export-as.svg':'按模型格式另存','export.svg':'导出选定模型',
      'feature-extraction.svg':'图像关键点特征提取','feature-matching.svg':'跨图特征对应匹配','grab-image.svg':'从视图截取静态图','grab-movie.svg':'录制渲染视图视频',
      'import-from.svg':'从选定文件导入（文件格式见滤镜选择）','import.svg':'从COLMAP项目目录导入','log.svg':'查看日志','match-matrix.svg':'图像配对匹配矩阵',
      'project-edit.svg':'编辑项目配置','project-new.svg':'新建项目','project-open.svg':'打开INI项目配置','project-save-as.svg':'另存INI项目配置','project-save.svg':'保存INI项目配置',
      'reconstruction-normalize.svg':'归一化重建尺度和中心','reconstruction-options.svg':'重建参数选项（原图文字未确认）',
      'reconstruction-pause.svg':'暂停自动重建','reconstruction-reset.svg':'重置重建并可能覆写输出（含确认）','reconstruction-start.svg':'开始/继续自动重建',
      'reconstruction-stats.svg':'统计重建模型信息','reconstruction-step.svg':'递进处理下一张影像','render-disabled.svg':'当前已禁用渲染：按钮可重新启用','render-enabled.svg':'当前已启用渲染：按钮可关闭',
      'render-options.svg':'视图渲染参数（原图文字未确认）','render-reset-view.svg':'重置模型视图相机','undistort.svg':'对相机影像执行去畸变'}
    evidence={
      'automatic-reconstruction.svg':('ReconstructionWidget.cpp:277-281','ReconstructionWidget::AutomaticReconstruction'),
      'bundle-adjustment.svg':('ReconstructionWidget.cpp:325-331','ReconstructionWidget::BundleAdjustment'),
      'database-management.svg':('ReconstructionWidget.cpp:266-272','ReconstructionWidget::DatabaseManagement'),
      'dense-reconstruction.svg':('ReconstructionWidget.cpp:333-337','ReconstructionWidget::DenseReconstruction'),
      'export-all.svg':('ReconstructionWidget.cpp:226-230','ReconstructionWidget::ExportAll'),
      'export-as-text.svg':('ReconstructionWidget.cpp:238-242','ReconstructionWidget::ExportAsText'),
      'export-as.svg':('ReconstructionWidget.cpp:232-236','ReconstructionWidget::ExportAs'),
      'export.svg':('ReconstructionWidget.cpp:220-224','ReconstructionWidget::Export'),
      'feature-extraction.svg':('ReconstructionWidget.cpp:252-258','ReconstructionWidget::FeatureExtraction'),
      'feature-matching.svg':('ReconstructionWidget.cpp:259-265','ReconstructionWidget::FeatureMatching'),
      'grab-image.svg':('ReconstructionWidget.cpp:383-387','ReconstructionWidget::GrabImage'),
      'grab-movie.svg':('ReconstructionWidget.cpp:388-392','ModelViewerWidget::GrabMovie'),
      'import-from.svg':('ReconstructionWidget.cpp:214-219','ReconstructionWidget::ImportFrom'),
      'import.svg':('ReconstructionWidget.cpp:208-213','ReconstructionWidget::Import'),
      'log.svg':('ReconstructionWidget.cpp:378-382','ReconstructionWidget::showLog'),
      'match-matrix.svg':('ReconstructionWidget.cpp:373-377','ReconstructionWidget::MatchMatrix'),
      'project-edit.svg':('ReconstructionWidget.cpp:191-195','ReconstructionWidget::ProjectEdit'),
      'project-new.svg':('ReconstructionWidget.cpp:179-184','ReconstructionWidget::ProjectNew'),
      'project-open.svg':('ReconstructionWidget.cpp:185-190','ReconstructionWidget::ProjectOpen'),
      'project-save-as.svg':('ReconstructionWidget.cpp:202-207','ReconstructionWidget::ProjectSaveAs'),
      'project-save.svg':('ReconstructionWidget.cpp:196-201','ReconstructionWidget::ProjectSave'),
      'reconstruction-normalize.svg':('ReconstructionWidget.cpp:311-317','ReconstructionWidget::ReconstructionNormalize'),
      'reconstruction-options.svg':('ReconstructionWidget.cpp:318-324','ReconstructionWidget::ReconstructionOptions'),
      'reconstruction-pause.svg':('ReconstructionWidget.cpp:297-304','ReconstructionWidget::ReconstructionPause'),
      'reconstruction-reset.svg':('ReconstructionWidget.cpp:305-310','ReconstructionWidget::ReconstructionOverwrite / reset confirmation'),
      'reconstruction-start.svg':('ReconstructionWidget.cpp:283-289, 1090-1164','ReconstructionStart; resume text state'),
      'reconstruction-stats.svg':('ReconstructionWidget.cpp:367-372','ReconstructionWidget::ReconstructionStats'),
      'reconstruction-step.svg':('ReconstructionWidget.cpp:290-296, 1117-1135','Reconstruct next image (incremental)'),
      'render-disabled.svg':('ReconstructionWidget.cpp:1418-1420','render toggle currently disabled; action reenables'),
      'render-enabled.svg':('ReconstructionWidget.cpp:343-347, 1424-1426','render toggle initially enabled; action disables'),
      'render-options.svg':('ReconstructionWidget.cpp:353-357','ReconstructionWidget::RenderOptions'),
      'render-reset-view.svg':('ReconstructionWidget.cpp:348-352','ModelViewerWidget::ResetView'),
      'undistort.svg':('ReconstructionWidget.cpp:393-398, 1315+','ReconstructionWidget::UndistortImages'),
    }
    rows=[]
    for index,(name,target,backup,data,kind,source) in enumerate(sorted(plans),1):
        root=ET.fromstring(data)
        rows.append({'id':index,'name':name,'action':description[name],'before':str(backup),'after':str(target),
                     'source_kind':kind,'paraView_source':source,'code_ui_evidence':evidence[name],
                      'before_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),
                     'after_sha256':hashlib.sha256(data).hexdigest(),'intrinsic':[128,128],
                     'viewBox':root.attrib.get('viewBox'),'source_lettering':'No live text node; embedded-raster glyph content not fully OCR-certified.'})
    ico=[]
    for name in ['reconstruction_32.ico','reconstruction_64.ico','reconstruction_128.ico']:
        p=MEDIA/name;ico.append({'name':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'Windows standalone reconstruction application identity; unchanged'})
    report={'count':len(rows),'authored_vector':len(BUILDERS),'exact_ParaView_geometry':len(UPSTREAM),'QRCs':[str(p.relative_to(ROOT)) for p in qrcs],
            'runtime_URLs_unchanged':True,'icons':rows,'ICO_preserved':ico,
            'preserved_Fugue_license_sha256':hashlib.sha256((MEDIA/'LICENSE').read_bytes()).hexdigest(),
            'protected_media_hashes':protected_now,
            'license_files':['app/reconstruction/media/LICENSE','app/reconstruction/media/PARAVIEW_NOTICE.txt'],
            'ParaView_commit':subprocess.check_output(['git','-C',str(paraview_root),'rev-parse','HEAD']).decode().strip(),
            'protected_other_image_count':len(snapshots)}
    (REVIEW/'migration-ledger.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(f'PASS: {len(rows)} SVGs installed at 128x128; {len(BUILDERS)} functional vectors + {len(UPSTREAM)} ParaView originals; two QRC bundles; ICO unchanged')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paraview-root',type=Path,default=Path('/home/ludahai/develop/code/github/ParaView'));run(parser.parse_args().paraview_root)
