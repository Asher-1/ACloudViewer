"""Utility icons preserving the shipped book, cross, exit, merge and minus."""
from . import paraview_style as P
from .icon import fit


def console():
    # Paper is neutral, not glass-blue. The ochre cover and binding use the
    # measured ParaView clipboard palette; geometry stays unchanged.
    defs = '\n'.join([
        P.linear('paper', [(0, '#FAFAF6'), (1, P.GREY_LT)]),
        P.linear('cover', [(0, P.TAN), (1, P.TAN_DK)]),
    ])
    body = [P.fill('M 2 19 L 12 22 L 22 19 V 21 L 12 23 L 2 21 Z', 'url(#cover)'),
            P.fill('M 2.5 5 L 11.8 6.5 V 20 L 2.5 18.4 Z', 'url(#paper)'),
            P.fill('M 11.8 6.5 L 21.5 3 V 18.4 L 11.8 20 Z', 'url(#cover)'),
            P.fill('M 12 6.4 L 20.5 1.8 V 16.5 L 12 19.8 Z', 'url(#paper)'),
            P.stroke('M 11.8 6.5 V 20', colour=P.TAN_DK, sw=1.1)]
    return P.svg('<title>Console — open book</title>\n'+'\n'.join(body), defs=defs)


def delete_entity():
    # Cross geometry stays recognisable; use the warm tan/yellow sample palette.
    defs = P.linear('cross', [(0, P.YELLOW), (1, P.BROWN)])
    return P.svg('<title>Delete selected entities</title>\n'+P.fill(
        'M 4 2 L 12 10 L 20 2 L 22 4 L 14 12 L 22 20 '
        'L 20 22 L 12 14 L 4 22 L 2 20 L 10 12 L 2 4 Z', 'url(#cross)'), defs=defs)


def exit_view():
    # This resource exits the full-screen view. Keep the running-exit pictogram.
    defs = P.linear('exit', [(0, P.GREEN_LT), (1, P.GREEN_DK)])
    body = [P.rect(1.5, 1.5, 21, 21, 'url(#exit)', r=0.6),
            P.stroke('M 5 20 V 4 H 17 V 9', colour='#FAFAF6', sw=1.25),
            P.circle(12.6, 6.3, 1.6, '#FAFAF6', outline=None),
            P.fill('M 11 8.3 L 14 8.7 L 15.9 11 H 19 V 13 H 15 '
                   'L 13.3 11.5 L 12.5 15 L 15.4 17.1 L 16.8 21 H 14.4 '
                   'L 13.3 18.5 L 10.4 17 L 8.6 21 H 6.3 L 9.2 14.5 '
                   'L 10 10.8 L 8.3 12.2 H 6 V 10.3 H 7.5 Z', '#FAFAF6', outline=None)]
    return P.svg('<title>Exit full-screen view</title>\n'+'\n'.join(body), defs=defs)


def merge_entities():
    defs = P.linear('entity', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)])
    body = [P.fill('M 1.5 3 H 10 V 6.8 H 5.5 V 17.2 H 10 V 21 H 1.5 Z', 'url(#entity)'),
            P.rect(18, 7.2, 4.5, 9.6, 'url(#entity)', r=0.3),
            P.fill('M 7 12 L 12 7.5 V 10 H 17 V 14 H 12 V 16.5 Z', P.YELLOW, sw=0.75)]
    return P.svg('<title>Merge entities into one</title>\n'+'\n'.join(body), defs=defs)


def minus():
    defs = P.linear('minus', [(0, P.GREEN_LT), (0.5, P.GREEN), (1, P.GREEN_DK)])
    return P.svg('<title>Trim selection — minus</title>\n'+P.rect(1.5, 9.6, 21, 4.8, 'url(#minus)', r=0.5), defs=defs)


ICONS = {
    'console': ('toolMainConsole', console),
    'delete': ('toolMainDeleteEntity', delete_entity),
    'exit': ('toolMainExit', exit_view),
    'merge': ('toolMainMergeEntities', merge_entities),
    'minus': ('toolMainMinus', minus),
}


def built(key):
    return fit(ICONS[key][1]())
