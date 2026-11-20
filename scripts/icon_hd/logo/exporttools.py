"""Five export/save resources, retaining each source's distinct pictogram."""
from . import paraview_style as P


def surface_cloud():
    defs = P.linear('cloud', [(0, P.GREY_LT), (.45, P.BLUE_PALE), (1, P.BLUE_MID)])
    body = P.fill('M 4.2 18.4 C 1 18.5 .5 14.5 3.2 12.9 '
                  'C 1.8 9.5 4.8 6.9 8 7.8 C 7.8 3 14 1.8 16.7 5.2 '
                  'C 20.3 4.4 23 7.3 21.8 10.5 C 24.6 12 24 16.1 21.5 17.2 '
                  'C 20.8 20.8 15.4 21.4 13 18.6 Z', 'url(#cloud)')
    return P.svg('<title>Export surface selection as a cloud</title>\n' + body, defs=defs)


def export_selection():
    defs = P.linear('arrow', [(0, P.YELLOW), (1, P.TAN)])
    body = [P.stroke('M 9.5 3 H 5 Q 2 3 2 6 V 19 Q 2 22 5 22 '
                     'H 18 Q 21 22 21 19 V 16', sw=1.3),
            P.fill('M 7 18 C 5 10 10 7 16 7 V 3 L 23 9.5 L 16 16 '
                   'V 12 C 11 12 8 13 7 18 Z', 'url(#arrow)')]
    return P.svg('<title>Export selection — outgoing curved arrow</title>\n' +
                 '\n'.join(body), defs=defs)


def save_annotations():
    defs = '\n'.join([
        P.linear('disk', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)]),
        P.linear('arrow', [(0, P.GREEN_LT), (1, P.GREEN)]),
    ])
    body = [P.fill('M 2 2 H 19 L 22 5 V 22 H 2 Z', 'url(#disk)'),
            P.rect(5, 2, 10, 5, P.GREY_LT, r=.2),
            P.rect(7, 2, 2.2, 4, P.GREY_DK, r=.1),
            P.rect(4.5, 15, 14.5, 7, '#FAFAF6', r=.3),
            P.rect(4.5, 20, 14.5, 2, P.TAN, r=0),
            P.fill('M 15 8 H 20 V 14 H 23 L 17.5 21 L 11 14 H 15 Z',
                   'url(#arrow)', sw=.7)]
    return P.svg('<title>Save annotations — floppy and green down arrow</title>\n' +
                 '\n'.join(body), defs=defs)


def _download(green=False):
    light, dark = (P.GREEN_LT, P.GREEN) if green else (P.BLUE_PALE, P.BLUE_STRONG)
    defs = P.linear('export', [(0, light), (1, dark)])
    body = [P.fill('M 8 2 H 16 V 10.5 H 21 L 12 20.5 L 3 10.5 H 8 Z',
                   'url(#export)'),
            P.rect(3, 20.5, 18, 2, dark, r=.3)]
    title = 'Export cloud or measurement result' if green else 'Export plane information'
    return P.svg(f'<title>{title} — down arrow and tray</title>\n' +
                 '\n'.join(body), defs=defs)


def export_plane():
    return _download(False)


def export_cloud():
    return _download(True)


ICONS = {
    'cloud.svg': ('surface/surfaceExportCloud.svg', surface_cloud, '曲面选择导出为点云'),
    'exportIcon.png': ('annotation/annotationExportSelection.svg', export_selection, '导出分割选区'),
    'small_save.png': ('annotation/annotationSave.svg', save_annotations, '保存标注'),
    'export.svg': ('main/toolMainExportPlaneInfo.svg', export_plane, '导出平面信息'),
    'exportCloud.png': ('small/ioExportCloud.svg', export_cloud, '导出点云 / 结果'),
}
