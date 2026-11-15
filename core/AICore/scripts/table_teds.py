#!/usr/bin/env python3
"""TEDS / TEDS-S table similarity (PubTabNet metric), standard library only.

Reproduces ibm-aur-nlp/PubTabNet ``src/metric.py`` (Apache-2.0):

* the HTML is parsed, ``body/table`` is the root, every element is a node;
* a ``td`` node carries (colspan, rowspan, content) where content is the
  token list ``['<b>', 'x', '</b>', ...]`` of its text and child markup,
  and its children are not tree nodes;
* insert/delete cost 1; rename cost 1 when tag/colspan/rowspan differ,
  else the normalised Levenshtein distance of the two contents (0 when
  both are empty or ``structure_only``);
* ``TEDS = 1 - distance / max(|nodes_pred|, |nodes_true|)`` where the node
  counts are the descendant elements of the table (``.//*``).

The upstream implementation computes the tree edit distance with APTED;
this one uses Zhang-Shasha. Both are exact optimal edit-distance algorithms
under the same cost model, so the scores are identical (the GT matrix
cross-checks this against the upstream package when it is importable).

    table_teds.py pred.html true.html [--structure-only]
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser

# Void elements never get children (lxml/libxml2 HTML semantics).
_VOID = {"br", "hr", "img", "input", "meta", "link", "col", "area", "base",
         "wbr", "source", "embed", "param", "track"}


class _Element:
    __slots__ = ("tag", "attrs", "children", "text", "tail")

    def __init__(self, tag: str, attrs: dict[str, str]):
        self.tag = tag
        self.attrs = attrs
        self.children: list[_Element] = []
        self.text = ""
        self.tail = ""


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Element("#root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        element = _Element(tag, {k: (v or "") for k, v in attrs})
        self.stack[-1].children.append(element)
        if tag not in _VOID:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_Element(tag, {k: (v or "") for k, v in attrs}))

    def handle_endtag(self, tag):
        # Pop to the matching open element; a stray end tag is ignored.
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data):
        parent = self.stack[-1]
        if parent.children:
            parent.children[-1].tail += data
        else:
            parent.text += data


def _find_table(html: str) -> _Element | None:
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()

    def search(node: _Element):
        for child in node.children:
            if child.tag == "table":
                return child
            found = search(child)
            if found is not None:
                return found
        return None
    return search(builder.root)


def _cell_tokens(td: _Element) -> list[str]:
    tokens: list[str] = []

    def walk(node: _Element, is_cell: bool):
        tokens.append(f"<{node.tag}>")
        tokens.extend(node.text)
        for child in node.children:
            walk(child, False)
        tokens.append(f"</{node.tag}>")
        if not is_cell:
            tokens.extend(node.tail)
    walk(td, True)
    return tokens[1:-1]


def _span(td: _Element, name: str) -> int:
    try:
        return int(td.attrs.get(name, "1"))
    except ValueError:
        return 1


class _Node:
    __slots__ = ("tag", "colspan", "rowspan", "content", "children")

    def __init__(self, tag, colspan=None, rowspan=None, content=None):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children: list[_Node] = []


def _convert(element: _Element, structure_only: bool) -> _Node:
    if element.tag == "td":
        return _Node("td", _span(element, "colspan"), _span(element, "rowspan"),
                     [] if structure_only else _cell_tokens(element))
    node = _Node(element.tag)
    node.children = [_convert(child, structure_only) for child in element.children]
    return node


def _count_descendants(element: _Element) -> int:
    return sum(1 + _count_descendants(child) for child in element.children)


def levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _rename_cost(n1: _Node, n2: _Node) -> float:
    if n1.tag != n2.tag or n1.colspan != n2.colspan or n1.rowspan != n2.rowspan:
        return 1.0
    if n1.tag == "td" and (n1.content or n2.content):
        return levenshtein(n1.content, n2.content) / max(len(n1.content), len(n2.content))
    return 0.0


def _postorder(root: _Node):
    """Post-order nodes, leftmost-leaf index per node and the keyroots."""
    nodes: list[_Node] = []
    leftmost: list[int] = []

    def walk(node: _Node) -> int:
        first = None
        for child in node.children:
            lm = walk(child)
            if first is None:
                first = lm
        index = len(nodes)
        nodes.append(node)
        leftmost.append(index if first is None else first)
        return leftmost[index]
    walk(root)
    seen: dict[int, int] = {}
    for index, lm in enumerate(leftmost):
        seen[lm] = index  # the highest node sharing a leftmost leaf
    return nodes, leftmost, sorted(seen.values())


def tree_edit_distance(a: _Node, b: _Node) -> float:
    """Zhang-Shasha. O(|a| |b| min(depth, leaves)^2) time, O(|a| |b|)
    memory: a few hundred nodes per table is well under a second."""
    na, la, ka = _postorder(a)
    nb, lb, kb = _postorder(b)
    treedist = [[0.0] * len(nb) for _ in na]
    rename: dict[tuple[int, int], float] = {}
    for i in ka:
        for j in kb:
            li, lj = la[i], lb[j]
            rows, cols = i - li + 2, j - lj + 2
            fd = [[0.0] * cols for _ in range(rows)]
            for x in range(1, rows):
                fd[x][0] = fd[x - 1][0] + 1.0
            for y in range(1, cols):
                fd[0][y] = fd[0][y - 1] + 1.0
            for x in range(1, rows):
                ix = li + x - 1
                for y in range(1, cols):
                    jy = lj + y - 1
                    if la[ix] == li and lb[jy] == lj:
                        key = (ix, jy)
                        cost = rename.get(key)
                        if cost is None:
                            cost = rename[key] = _rename_cost(na[ix], nb[jy])
                        fd[x][y] = min(fd[x - 1][y] + 1.0, fd[x][y - 1] + 1.0,
                                       fd[x - 1][y - 1] + cost)
                        treedist[ix][jy] = fd[x][y]
                    else:
                        px = la[ix] - li
                        py = lb[jy] - lj
                        fd[x][y] = min(fd[x - 1][y] + 1.0, fd[x][y - 1] + 1.0,
                                       fd[px][py] + treedist[ix][jy])
    return treedist[len(na) - 1][len(nb) - 1]


def teds(pred: str, true: str, structure_only: bool = False) -> float:
    if not pred or not true:
        return 0.0
    pred_table = _find_table(pred)
    true_table = _find_table(true)
    if pred_table is None or true_table is None:
        return 0.0
    n_nodes = max(_count_descendants(pred_table), _count_descendants(true_table))
    if n_nodes == 0:
        return 1.0
    distance = tree_edit_distance(_convert(pred_table, structure_only),
                                  _convert(true_table, structure_only))
    return 1.0 - distance / n_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pred")
    parser.add_argument("true")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    with open(args.pred, encoding="utf-8") as p, open(args.true, encoding="utf-8") as t:
        print(f"{teds(p.read(), t.read(), args.structure_only):.6f}")


if __name__ == "__main__":
    main()
