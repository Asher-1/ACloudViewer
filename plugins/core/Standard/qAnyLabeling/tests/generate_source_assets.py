"""Regenerate the plugin's source-owned UI defaults and panel stylesheets.

The upstream YAML and the upstream style generators are the only inputs, so
the committed resources stay byte-comparable with compare_config_upstream.py
instead of being hand-edited. Only the object names and the image prefix are
rewritten for this plugin's widgets.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import yaml

PLUGIN_REPLACEMENTS = (
    (":/images/images/", ":/CC/plugin/qAnyLabeling/images/"),
    ("#sidebarPanel", "#AnnotationSidebarSection"),
    ("#FileList", "#AnnotationFiles"),
)


def plugin_style(text):
    for source, target in PLUGIN_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    args = parser.parse_args()
    config = args.upstream / "anylabeling/configs/xanylabeling_config.yaml"
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    header = (
        f"// UI defaults from X-AnyLabeling-GGML {revision}\n"
        "// anylabeling/configs/xanylabeling_config.yaml; GPL-3.0-only, see"
        " UPSTREAM-LICENSE.\n"
        f"// Source SHA-256: {hashlib.sha256(config.read_bytes()).hexdigest()}\n")
    body = json.dumps(yaml.safe_load(config.read_text()), indent=2)
    args.defaults.write_text(header + 'R"json(\n' + body + '\n)json"\n')

    sys.path.insert(0, str(args.upstream.resolve()))
    spec = importlib.util.spec_from_file_location(
        "upstream_theme", args.upstream / "anylabeling/views/labeling/utils/theme.py")
    theme = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theme)
    from anylabeling.views.labeling.utils import theme as panel_theme, style
    panels = {}
    for mode in ("dark", "light"):
        panel_theme._active_mode = mode
        panel_theme._active_theme = getattr(panel_theme, mode.upper())
        panels[mode] = {
            "dock": plugin_style(style.get_dock_style()),
            "panel": plugin_style(style.get_panel_style()),
            "description": plugin_style(style.get_plain_text_edit_style()),
            "checkbox": plugin_style(style.get_checkbox_indicator_style()),
        }
    args.panels.write_text(json.dumps(panels, indent=2) + "\n")
    print(f"regenerated {args.defaults} and {args.panels} from {revision}")


if __name__ == "__main__":
    raise SystemExit(main())
