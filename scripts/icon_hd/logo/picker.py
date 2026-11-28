"""Build an interactive HTML picker for the fifty icon styles.

Why a web page and not a PNG
----------------------------
Three earlier rounds were reviewed from rendered PNG sheets, and each time the
sizes that mattered were baked in: an 18 px tile is four times smaller than a
thumbnail and cannot show what a 1.4-unit stroke looks like, while a 96 px tile
hides that the same stroke disappears in the toolbar.  A page embeds the SVGs
themselves, so the reviewer can change size, background and comparison without
waiting for another render.

The page carries no dependencies and no build step: open it in a browser.

Usage
-----
    python3 -m scripts.icon_hd.logo.picker
    python3 -m scripts.icon_hd.logo.picker --out /tmp/picker.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

if str(ROOT / "scripts/icon_hd") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts/icon_hd"))


# Groups follow the library's technique frequencies, so the order of the page is
# the order of convention: what the category most often does comes first.
GROUPS = [
    ("contain", "包围式 Containment", "31.6%", "库中最常见：图形被一个几何外形框住"),
    ("knockout", "挖空 Knockout", "19.8%", "负空间：标记本身是挖出来的洞"),
    ("line", "单线 Monoline", "17.4%", "统一线宽的描边，无填充"),
    ("colour", "色块 Colour", "14.2%", "实色分区，靠色块划分结构"),
    ("hybrid", "混合 Hybrid", "—", "描边定形 + 实色块点明操作对象"),
    ("iso", "等轴 Isometric", "6.3%", "2.5D 立体，贴合点云/网格类软件"),
]


def _group_of(style: dict) -> str:
    n = style["name"]
    if "contain" in n or "badge" in n:
        return "contain"
    if "knockout" in n:
        return "knockout"
    if style["geometry"] == "iso":
        return "iso"
    if style["construction"] == "hybrid":
        return "hybrid"
    if style["construction"] == "line":
        return "line"
    return "colour"


CSS = """
:root{
  --bg:#0F1419; --panel:#161C23; --panel2:#1C242D; --line:#2A343E;
  --ink:#E6EDF3; --dim:#8B98A5; --accent:#1E9BD7; --good:#3FB950; --bad:#F85149;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC",sans-serif}
header{position:sticky;top:0;z-index:20;background:rgba(15,20,25,.94);
  backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:14px 22px}
h1{margin:0 0 3px;font-size:17px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12.5px;margin:0}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
.chip{background:var(--panel2);border:1px solid var(--line);color:var(--ink);
  border-radius:999px;padding:5px 13px;font-size:12.5px;cursor:pointer;
  transition:.13s;user-select:none}
.chip:hover{border-color:var(--accent)}
.chip.on{background:var(--accent);border-color:var(--accent);color:#04141F;font-weight:600}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{background:var(--panel2);border:0;color:var(--dim);padding:5px 12px;
  font-size:12.5px;cursor:pointer;border-right:1px solid var(--line)}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--accent);color:#04141F;font-weight:600}
.lbl{color:var(--dim);font-size:12px;margin-right:2px}
main{padding:22px 22px 90px}
section{margin-bottom:34px}
h2{font-size:15px;margin:0 0 3px;display:flex;align-items:baseline;gap:9px}
h2 .pct{color:var(--accent);font-size:12px;font-weight:600}
h2 .desc{color:var(--dim);font-size:12px;font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(268px,1fr));
  gap:12px;margin-top:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
  padding:13px 14px 12px;cursor:pointer;transition:.14s;position:relative}
.card:hover{border-color:#3D4E5C;transform:translateY(-1px)}
.card.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.card.bad{opacity:.45}
.tools{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 10px;
  align-items:center;justify-items:center}
.tools svg{display:block}
.bigwrap{display:grid;place-items:center;width:100%}
.name{font-size:13px;font-weight:600;margin-top:9px}
.en{font:10.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:#5F6E7C;margin-top:1px}
.meta{display:flex;gap:9px;align-items:center;margin-top:7px;font-size:11px}
.ink{color:var(--good)}
.ink.warn{color:#D29922}
.ink.bad{color:var(--bad)}
.num{color:#5F6E7C;font:10.5px ui-monospace,monospace}
.tag{font-size:10px;padding:1.5px 7px;border-radius:6px;background:#243039;
  color:#8B98A5;font-weight:600}
.sel-mark{position:absolute;top:11px;right:12px;width:17px;height:17px;
  border-radius:50%;border:1.5px solid #38454F;display:grid;place-items:center;
  font-size:11px;color:transparent}
.card.sel .sel-mark{background:var(--accent);border-color:var(--accent);color:#04141F}
footer{position:fixed;bottom:0;left:0;right:0;background:rgba(22,28,35,.97);
  backdrop-filter:blur(12px);border-top:1px solid var(--line);padding:11px 22px;
  display:flex;gap:14px;align-items:center;font-size:13px;z-index:30}
footer .picked{color:var(--dim);flex:1}
footer b{color:var(--accent)}
button.act{background:var(--accent);border:0;color:#04141F;font-weight:650;
  padding:7px 16px;border-radius:8px;cursor:pointer;font-size:13px}
button.act.ghost{background:transparent;border:1px solid var(--line);color:var(--dim)}
.light .card{background:#fff;border-color:#E2E8EE}
/* 深色模式下把图标主色提亮，否则 INK 与背景对比度只有 1.62:1 */
body:not(.light) .tools svg{filter:brightness(2.1) saturate(.9)}
.light{background:#F7F9FB;color:#111}
.light header,.light footer{background:rgba(255,255,255,.96);border-color:#E2E8EE}
.light .en,.light .num{color:#8B98A5}
.light .chip,.light .seg button{background:#fff;border-color:#E2E8EE;color:#222}
.light .chip.on,.light .seg button.on{background:var(--accent);color:#fff}
"""

JS = """
const state = {size:48, group:'all', okOnly:false, picked:new Set()};
function render(){
  document.querySelectorAll('.card').forEach(c=>{
    const g=c.dataset.group, bad=c.dataset.bad==='1';
    let show = (state.group==='all'||state.group===g) && (!state.okOnly||!bad);
    c.style.display = show?'':'none';
    c.classList.toggle('sel', state.picked.has(c.dataset.name));
  });
  document.querySelectorAll('section').forEach(s=>{
    const any=[...s.querySelectorAll('.card')].some(c=>c.style.display!=='none');
    s.style.display=any?'':'none';
  });
  document.querySelectorAll('.tools svg').forEach(svg=>{
    svg.style.width = state.size+'px'; svg.style.height = state.size+'px';
  });
  document.getElementById('cnt').textContent = state.picked.size;
  document.getElementById('list').textContent =
    state.picked.size ? [...state.picked].join(', ') : '未选择';
}
function pick(el){
  const n=el.dataset.name;
  if(state.picked.has(n)) state.picked.delete(n); else state.picked.add(n);
  render();
}
function setGroup(g,btn){
  state.group=g;
  document.querySelectorAll('[data-grp]').forEach(b=>b.classList.toggle('on',b===btn));
  render();
}
function setSize(s,btn){
  state.size=s;
  document.querySelectorAll('[data-sz]').forEach(b=>b.classList.toggle('on',b===btn));
  render();
}
function toggleOk(btn){
  state.okOnly=!state.okOnly; btn.classList.toggle('on',state.okOnly); render();
}
function toggleTheme(btn){
  document.body.classList.toggle('light'); render();
}
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.card').forEach(c=>c.addEventListener('click',()=>pick(c)));
  const all=document.querySelector('[data-grp="all"]'); if(all) all.classList.add('on');
  const d=document.querySelector('[data-sz="48"]'); if(d) d.classList.add('on');
  render();
});
"""


def build() -> str:
    from logo import styles50 as S

    import cairosvg
    import io

    from PIL import Image
    import numpy as np

    def ink(name_style: dict) -> tuple[float, float]:
        vals = []
        for t in S.TOOL_ORDER:
            svg = S.render(name_style, t)
            a = np.asarray(Image.open(io.BytesIO(
                cairosvg.svg2png(bytestring=svg.encode(), output_width=18,
                                 output_height=18))).convert("RGBA"))
            vals.append(float((a[..., 3] > 80).mean()))
        return float(np.median(vals)), float(max(vals))

    buckets: dict[str, list] = {k: [] for k, *_ in GROUPS}
    for st in S.STYLES:
        buckets[_group_of(st)].append(st)

    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>图标风格筛选 · 50 组</title>',
        f'<style>{CSS}</style></head><body class="light">',
        '<header>',
        '<h1>图标风格筛选 <span style="color:#5F6E7C;font-weight:400">· 50 组</span></h1>',
        '<p class="sub">每张卡片 = 同一种风格下的 6 个不同工具（加号、复制、柱状、取点、平滑、裁剪）。'
        '只有撑得住这 6 个的，才算一套风格，不是一张好看的图。</p>',
        '<div class="bar">',
        '<span class="lbl">分组</span>',
        '<span class="chip" data-grp="all" onclick="setGroup(\'all\',this)">全部 50</span>',
    ]
    for key, cn, pct, _ in GROUPS:
        n = len(buckets[key])
        if n:
            parts.append(f'<span class="chip" data-grp="{key}" onclick="setGroup(\'{key}\',this)">'
                         f'{cn.split()[0]} {n}</span>')
    parts += [
        '</div><div class="bar">',
        '<span class="lbl">尺寸</span><div class="seg">',
        *[f'<button data-sz="{s}" onclick="setSize({s},this)">{s}px</button>'
          for s in (18, 24, 48, 96)],
        '</div>',
        '<span class="chip" onclick="toggleOk(this)">只看 18px 可用</span>',
        '<span class="chip" onclick="toggleTheme(this)">浅色背景</span>',
        '</div></header><main>',
    ]

    for key, cn, pct, desc in GROUPS:
        items = buckets[key]
        if not items:
            continue
        parts.append(f'<section><h2>{html.escape(cn)} '
                     f'<span class="pct">{pct}</span>'
                     f'<span class="desc">{html.escape(desc)}</span></h2>')
        parts.append('<div class="grid">')
        for st in items:
            med, mx = ink(st)
            bad = mx > 0.55
            idx = S.STYLES.index(st) + 1
            tools = "".join(
                f'<span class="bigwrap">{S.render(st, t)}</span>'
                for t in S.TOOL_ORDER
            )
            cls = "card bad" if bad else "card"
            parts.append(
                f'<div class="{cls}" data-name="{html.escape(st["name"])}" '
                f'data-group="{key}" data-bad="{1 if bad else 0}">'
                f'<div class="sel-mark">✓</div>'
                f'<div class="tools">{tools}</div>'
                f'<div class="name">{html.escape(st["cn"])}</div>'
                f'<div class="en">{html.escape(st["name"])}</div>'
                f'<div class="meta">'
                f'<span class="num">#{idx}</span>'
                f'<span class="ink {"bad" if bad else ("warn" if mx > 0.45 else "")}">'
                f'18px 墨量 {mx:.2f}</span>'
                f'{"" if not bad else "<span class=tag>小尺寸过密</span>"}'
                f'</div></div>'
            )
        parts.append("</div></section>")

    parts += [
        '</main>',
        '<footer><span class="picked">已选 <b id="cnt">0</b> 组 · <span id="list">未选择</span></span>',
        '<button class="act ghost" onclick="state.picked.clear();render()">清空</button>',
        '</footer>',
        f'<script>{JS}</script>',
        '</body></html>',
    ]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the interactive style picker.")
    ap.add_argument("--out", default=str(HERE / "data/picker.html"))
    a = ap.parse_args(argv)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    page = build()
    out.write_text(page)
    print(f"wrote {out}  ({len(page):,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
