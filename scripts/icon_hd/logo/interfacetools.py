"""Interface icons restyled without changing their source identifiers.

Keep the source power button, a/文 board, three theme tiles, question-mark book,
folder-plus, magnifier, bucket, artist palette, cleaning brush and wrench. The
registry records actual consumers rather than guessing actions from filenames.
"""
import math

from . import paraview_style as P

PAPER = '#FAFAF6'


def _button(body, title):
    defs = P.radial('button', [(0, PAPER), (0.7, P.GREY_LT), (1, P.GREY)],
                    cx=0.4, cy=0.3, r=0.9)
    return P.svg(f'<title>{title}</title>\n' +
                 P.circle(12, 12, 10.3, 'url(#button)') + '\n' + body,
                 defs=defs)


def close_window():
    body = P.stroke('M 7.6 7.2 A 6.5 6.5 0 1 0 16.4 7.2',
                    colour=P.BROWN, sw=1.9)
    body += P.stroke('M 12 4.7 V 11.2', colour=P.BROWN, sw=2.1)
    return _button(body, 'Close window — source power-button symbol')


def maximize_window():
    # ParaView fullscreen idiom (pqFullscreen): pale-blue panel + four yellow
    # corner arrows pointing OUT = expand the window to full. Pairs with
    # minimize_window's collapse-to-bar; both share panel/outline/accent family.
    body = [P.rect(3, 3, 18, 18, P.BLUE_PALE, r=0.4)]
    for d in ['M 2 8 V 2 H 8 L 6.1 4 L 9 6.9 L 6.9 9 L 4 6.1 Z',
              'M 16 2 H 22 V 8 L 20 6.1 L 17.1 9 L 15 6.9 L 17.9 4 Z',
              'M 2 16 V 22 H 8 L 6.1 20 L 9 17.1 L 6.9 15 L 4 17.9 Z',
              'M 16 22 H 22 V 16 L 20 17.9 L 17.1 15 L 15 17.1 L 17.9 20 Z']:
        body.append(P.fill(d, P.YELLOW, sw=0.55))
    return P.svg('<title>Maximize window — expand to full</title>\n' + ''.join(body))


def minimize_window():
    # Same panel family as maximize: a yellow down-arrow collapsing into a
    # bottom bar reads as "reduce the window to a bar", not a delete medallion.
    body = [P.rect(3, 3, 18, 18, P.BLUE_PALE, r=0.4),
            P.fill('M 10.6 5.6 H 13.4 V 11.6 H 15.8 L 12 15.6 L 8.2 11.6 H 10.6 Z',
                   P.YELLOW, sw=0.55),
            P.fill('M 5 17 H 19 V 19.4 H 5 Z', P.YELLOW, sw=0.55)]
    return P.svg('<title>Minimize window — collapse to a bar</title>\n' + ''.join(body))


def theme():
    defs = '\n'.join([
        P.linear('warm', [(0, P.YELLOW), (1, P.TAN)]),
        P.linear('blue', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)]),
        P.linear('green', [(0, P.GREEN_LT), (1, P.GREEN)]),
    ])
    body = [P.rect(10, 2, 12, 12, 'url(#warm)', r=1),
            P.rect(6, 6, 12, 12, 'url(#blue)', r=1),
            P.rect(2, 10, 12, 12, 'url(#green)', r=1)]
    return P.svg('<title>Interface theme — three overlapping tiles</title>\n' +
                 '\n'.join(body), defs=defs)


def language():
    defs = P.linear('frame', [(0, P.TAN), (1, P.TAN_DK)])
    body = [P.rect(1.5, 2.5, 21, 19, 'url(#frame)', r=0.8),
            P.rect(3.2, 4.2, 17.6, 15.6, P.GREEN_DK, r=0.25),
            # Both glyphs are source content. Paths avoid a host-font dependency.
            P.stroke('M 5 10.2 C 5.6 8.5 9.8 8.6 9.8 11 V 16 '
                     'M 9.7 12 C 4.5 11.6 4.3 16.9 8.2 15.8 L 9.7 14.5',
                     colour=PAPER, sw=1.25),
            P.stroke('M 15.2 7.7 L 15.9 8.8 M 12.2 10 H 19.6 '
                     'M 13.7 10.3 C 14.5 13.5 16.6 15.6 19.6 16.5 '
                     'M 18.2 10.3 C 17.4 13.2 15.5 15.6 12.4 16.5',
                     colour=PAPER, sw=1.15)]
    return P.svg('<title>Language — a and 文</title>\n' + '\n'.join(body), defs=defs)


def full_screen():
    defs = P.linear('screen', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)])
    body = [P.fill('M 10 16 H 14 V 19.5 L 18 21 V 22 H 6 V 21 '
                   'L 10 19.5 Z', P.GREY_MID),
            P.rect(2, 2, 20, 15, P.GREY_LT, r=0.8),
            P.rect(3.8, 3.8, 16.4, 11.4, 'url(#screen)', r=0.3),
            P.stroke('M 5 7 V 5 H 7.5 M 16.5 5 H 19 V 7 '
                     'M 5 12 V 14 H 7.5 M 16.5 14 H 19 V 12',
                     colour=P.YELLOW, sw=0.95)]
    return P.svg('<title>Exclusive full screen for the 3D view</title>\n' +
                 '\n'.join(body), defs=defs)


def about():
    body = P.circle(12.3, 6.8, 1.2, P.BLUE_STRONG, outline=None)
    body += P.fill('M 10.1 10 H 13.6 L 12.3 16.7 H 14.3 V 18 H 8.8 '
                   'V 16.7 H 10.3 L 11.3 11.4 H 10.1 Z',
                   P.BLUE_STRONG, outline=None)
    return _button(body, 'About — information symbol')


def help_book():
    defs = P.linear('cover', [(0, P.BLUE_MID), (1, P.BLUE_STRONG)])
    body = [P.fill('M 2 9.8 L 12 17 V 21 L 2 13.8 Z', P.BLUE_MID),
            P.fill('M 12 17 L 22 9.8 V 13.8 L 12 21 Z', P.GREY_LT),
            P.fill('M 2 9.8 L 12 2.5 L 22 9.8 L 12 17 Z', 'url(#cover)'),
            P.stroke('M 9.1 7.6 C 9.1 4.5 15.3 4.5 15.4 7.5 '
                     'C 15.5 9.4 12.2 9.5 12 11.8', colour=PAPER, sw=1.4),
            P.circle(12, 13.9, 0.9, PAPER, outline=None)]
    return P.svg('<title>Help — question-mark book</title>\n' +
                 '\n'.join(body), defs=defs)


def import_file():
    # Shared by importing scalar fields and annotation class sets.
    defs = '\n'.join([
        P.linear('folder', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)]),
        P.linear('add', [(0, P.GREEN_LT), (1, P.GREEN)]),
    ])
    body = [P.fill('M 4 2 H 15 L 19 6 V 20 H 4 Z', P.GREY_LT),
            P.fill('M 15 2 V 6 H 19 Z', PAPER),
            P.fill('M 2 6 H 8 L 10 8 H 18.5 V 21 H 2 Z', 'url(#folder)'),
            P.circle(17.3, 17.1, 5.2, 'url(#add)'),
            P.fill('M 16.3 13.6 H 18.3 V 16.1 H 20.8 V 18.1 '
                   'H 18.3 V 20.6 H 16.3 V 18.1 H 13.8 V 16.1 H 16.3 Z',
                   PAPER, outline=None)]
    return P.svg('<title>Import data from file — folder plus</title>\n' +
                 '\n'.join(body), defs=defs)


def zoom():
    # Registered but currently unused: no claim that this means global zoom.
    defs = P.radial('lens', [(0, PAPER), (0.45, P.BLUE_PALE), (1, P.BLUE_MID)],
                    cx=0.35, cy=0.25, r=0.9)
    body = [P.fill('M 9.5 12.8 L 12 15.1 L 5.2 22.4 '
                   'Q 4.3 23.2 3.3 22.1 L 2.7 21.5 Q 2 20.8 2.9 19.8 Z',
                   P.GREY_DK),
            P.circle(14.6, 8.9, 6.9, 'url(#lens)'),
            P.stroke('M 10.3 7.7 C 10.9 5 13.4 4.1 15.9 4.6',
                     colour=PAPER, sw=1.0)]
    return P.svg('<title>View magnifier</title>\n' + '\n'.join(body), defs=defs)


def background_color():
    defs = P.linear('bucket', [(0, P.GREY_LT), (0.4, PAPER), (1, P.TAN)])
    body = [P.fill('M 2 9 C 0 7 2 5 4 6 L 7 8 Z', P.GREEN),
            P.fill('M 18 5 C 22 2 24 6 21 9 Z', P.BLUE_STRONG),
            P.fill('M 2 21 C 3 18 10 18 11 21 C 10 23 3 23 2 21 Z', P.BROWN),
            P.fill('M 18 16 C 21 17 23 20 21 22 '
                   'C 18 23 16 20 18 16 Z', P.YELLOW),
            '<g transform="rotate(-35 12 12)">',
            P.stroke('M 7.5 9 C 4.5 2.5 18 1 16.5 9', sw=0.85),
            P.fill('M 7 7 H 17 L 16 18 C 15 21 9 21 8 18 Z', 'url(#bucket)'),
            f'<ellipse cx="12" cy="7" rx="5" ry="2" '
            f'fill="{P.YELLOW}" stroke="{P.INK}"/>',
            '</g>']
    return P.svg('<title>Set background color — paint bucket</title>\n' +
                 '\n'.join(body), defs=defs)


def object_color():
    defs = P.linear('palette', [(0, P.TAN), (1, P.YELLOW)])
    contour = ('M 12 3 C 20 3 23 10 20 16 C 19 18 15 17 15 14 '
               'C 15 12 12 12 11 14 C 10 16 14 19 11 21 '
               'C 3 23 0.5 14 2 8 C 3 4 7 3 12 3 Z')
    body = [P.fill(contour, 'url(#palette)'),
            P.circle(5.2, 8.3, 1.5, P.GREEN),
            P.circle(9, 6.4, 1.4, P.BLUE_STRONG),
            P.circle(13.3, 6.6, 1.4, P.BROWN),
            P.circle(17.3, 9, 1.3, P.BLUE_MID),
            P.fill('M 1.5 1.5 L 3 1 L 11.7 15 L 10.1 16 Z', P.BROWN, sw=0.65),
            P.fill('M 9.4 13.5 L 11 12.6 L 13 16 L 11.2 17 Z', P.GREY_LT, sw=0.6),
            P.fill('M 11.2 16.6 L 13 15.7 C 16.2 18 14.9 21 11.2 21 '
                   'C 12.3 19.3 10.5 18 11.2 16.6 Z', P.BROWN, sw=0.65)]
    return P.svg('<title>Set a unique entity color — palette and brush</title>\n' +
                 '\n'.join(body), defs=defs)


def clear_colors():
    defs = P.linear('bristles', [(0, P.TAN), (1, P.YELLOW)])
    body = [P.rect(2, 3, 19, 18, P.BLUE_PALE, r=1),
            P.rect(4, 5, 6, 5, P.GREEN, r=0.3),
            P.rect(11, 5, 7, 5, P.BLUE_MID, r=0.3),
            P.rect(4, 11, 6, 5, P.YELLOW, r=0.3),
            P.rect(11, 11, 7, 5, P.GREY_LT, r=0.3),
            P.fill('M 20 2 L 22 3.5 L 12.2 16 L 10.6 14.5 Z', P.BROWN, sw=0.65),
            P.fill('M 10.5 13.8 L 13 15.7 L 10.6 18.6 L 8.2 16.8 Z', P.GREY_LT),
            P.fill('M 8.6 16 L 11.4 18 L 8.8 22 L 1.8 18.8 '
                   'L 5.7 14.8 Z', 'url(#bristles)'),
            P.stroke('M 5 17 L 3.4 19.3 M 7 18 L 5.4 20.3 '
                     'M 9 19 L 7.5 21.2', colour=P.TAN_DK, sw=0.6)]
    return P.svg('<title>Clear entity colors — color swatches and broom</title>\n' +
                 '\n'.join(body), defs=defs)


def color_menu():
    colors = [P.YELLOW, P.BROWN, P.BLUE_MID, P.BLUE_STRONG, P.GREEN, P.GREEN_LT]
    body = ['<title>Color menu — color wheel</title>']
    for i, color in enumerate(colors):
        # Six angular sectors make a wheel, not a collection of loose swatches.
        a, b = (math.radians(-90 + j * 60) for j in (i, i + 1))
        x1, y1 = 12 + 10 * math.cos(a), 12 + 10 * math.sin(a)
        x2, y2 = 12 + 10 * math.cos(b), 12 + 10 * math.sin(b)
        body.append(P.fill(f'M 12 12 L {x1:g} {y1:g} '
                           f'A 10 10 0 0 1 {x2:g} {y2:g} Z', color, outline=None))
    body += [P.circle(12, 12, 10, 'none'), P.circle(12, 12, 4.5, P.GREY_DK)]
    return P.svg('\n'.join(body))


def display_settings():
    defs = P.linear('panel', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)])
    body = [P.fill('M 3 2 L 18 5 V 21 L 3 18 Z', 'url(#panel)'),
            # Open jaw at the upper left, one solid diagonal shaft.
            P.fill('M 5.3 4.5 C 2.2 3.2 0.4 6.5 2 9.2 L 15 22 '
                   'C 16.7 23.5 19.5 21 17.8 19.3 L 6.9 8.5 '
                   'C 7.6 6.2 6.9 5.1 5.3 4.5 '
                   'L 5.5 7.3 L 3.8 8.2 L 2.6 6.7 L 3.6 4.7 Z', P.GREY_LT),
            P.circle(16.3, 20.4, 0.7, P.BLUE_MID, outline=None)]
    return P.svg('<title>Display settings — panel and wrench</title>\n' +
                 '\n'.join(body), defs=defs)


def rpc_interface():
    """RPC-01: local client/server request and reply; plugin-owned artwork.

    JsonRPCPlugin::getActions/triggered toggles the WebSocket JSON-RPC server.
    A workstation and server with opposing arrows communicate that contract;
    neither a document-save glyph nor a cloud implies the actual operation.
    """
    from .iconlabels import label_path

    defs = '\n'.join([
        P.linear('rpcScreen', [(0, P.BLUE_PALE), (1, P.BLUE_MID)]),
        P.linear('rpcServer', [(0, '#FAFAF6'), (1, P.GREY)]),
        P.linear('rpcRequest', [(0, P.GREEN_LT), (1, P.GREEN)]),
    ])
    body = [
        '<title>RPC-01 — JSON-RPC WebSocket client/server request and reply</title>',
        # Left: external automation client, a recognisable workstation.
        P.fill('M 4.5 12.5 H 6.6 V 15.4 H 8.6 V 16.5 H 2.5 V 15.4 H 4.5 Z',
               P.GREY_MID,
               sw=0.65),
        P.rect(1.2, 4.2, 8.3, 9.0, 'url(#rpcServer)', r=0.75),
        P.rect(2.5, 5.6, 5.7, 5.8, 'url(#rpcScreen)', r=0.25),
        # Right: application RPC endpoint, neutral case + two service slots.
        P.fill('M 16.2 3.1 L 18.2 1.8 L 22.8 3.0 V 14.6 L 21 16.0 H 16.2 Z',
               P.GREY_MID,
               sw=0.7),
        P.rect(16.2, 3.1, 4.8, 12.9, 'url(#rpcServer)', r=0.45),
        P.stroke('M 17.4 5.7 H 19.8 M 17.4 8.2 H 19.8',
                 colour=P.GREY_DK,
                 sw=0.8),
        P.circle(18.6, 13.5, 0.8, P.GREEN),
        # Request to the service, response back to the client. Filled arrows
        # remain distinct at toolbar scale and use the shared ParaView palette.
        P.fill('M 9.2 5.0 H 12.4 V 3.4 L 16.1 6.5 L 12.4 9.6 V 8.0 H 9.2 Z',
               'url(#rpcRequest)',
               sw=0.6),
        P.fill(
            'M 15.5 10.2 H 12.4 V 8.9 L 8.9 11.8 L 12.4 14.7 V 13.3 H 15.5 Z',
            P.GREEN,
            sw=0.6),
        # Outlined label: no runtime font dependency, deliberately integrated
        # into the base rather than a separate floating text tile.
        P.rect(1.5, 17.0, 21.0, 6.2, P.BLUE_PALE, r=0.7),
        label_path('RPC', x=3.2, y=22.1, width=17.6, height=4.25),
    ]
    return P.svg('\n'.join(body),
                 defs=defs).replace('width="24" height="24"',
                                    'width="128" height="128"', 1)


ICONS = {
    'close.png': ('window/windowClose.svg', close_window, '关闭窗口 · 电源符号'),
    'max.png': ('window/windowMaximize.svg', maximize_window, '最大化窗口'),
    'mini.svg': ('window/windowMinimize.svg', minimize_window, '最小化窗口'),
    'theme.png': ('main/toolMainTheme.svg', theme, '界面主题'),
    'language.png': ('main/toolMainLanguage.svg', language, '界面语言 · a / 文'),
    'FullScreen.png': ('main/toolMainExclusiveFullScreen.svg', full_screen, '3D 视图独占全屏'),
    'about.png': ('main/toolMainAbout.svg', about, '关于 · 信息符号'),
    'help.png': ('main/toolMainHelp.svg', help_book, '帮助 · 问号书本'),
    'add.png': ('main/toolMainImportFromFile.svg', import_file, '导入文件数据'),
    'zoom.png': ('view/viewZoom.svg', zoom, '视图放大镜 · 预留资源'),
    'ecvBackgroundColor.png': ('main/toolMainSetBackgroundColor.svg', background_color, '背景颜色 · 油漆桶'),
    'ecvObjectColor.png': ('main/toolMainSetUniqueColor.svg', object_color, '统一颜色 · 调色板'),
    'ecvClearColor.png': ('main/toolMainClearColors.svg', clear_colors, '清除实体颜色'),
    'ecvColorMenu.png': ('main/toolMainColorMenu.svg', color_menu, '颜色菜单 · 色轮'),
    'ecvDisplaySettings.png': ('main/toolMainDisplaySettings.svg', display_settings, '显示设置 · 扳手'),
}
