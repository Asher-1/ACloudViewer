"""Compare the real plugin's cold-start UI defaults with the upstream source.

The reference is generated at runtime. No user QSettings or cached models are
used. Explicitly report keys with no implemented control instead of implying
that a copy of the default data establishes complete workflow parity.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_suffix(".probe.json")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    subprocess.run([str(args.probe.resolve()), str(args.plugin.resolve()),
                    str(raw_output.resolve())], env=env, check=True)
    actual = json.loads(raw_output.read_text())
    source = args.upstream / "anylabeling/configs/xanylabeling_config.yaml"
    expected = yaml.safe_load(source.read_text())
    errors = []
    if actual["defaults"] != expected:
        errors.append("UI configuration data differs from upstream YAML")
    # QAction stores platform-independent key sequences (e.g. Ctrl++).
    from PyQt6.QtGui import QKeySequence
    checked = []
    missing = []
    for key, value in expected["shortcuts"].items():
        if key not in actual["shortcuts"]:
            missing.append(key)
            continue
        sequences = value if isinstance(value, list) else [value] if value else []
        sequences = [QKeySequence(s).toString(QKeySequence.SequenceFormat.PortableText)
                     for s in sequences]
        if actual["shortcuts"][key] != sequences:
            errors.append(f"shortcut {key}: {actual['shortcuts'][key]} != {sequences}")
        checked.append(key)
    bool_keys = {key: key for key in [
        "auto_save", "store_data", "keep_prev_brightness", "keep_prev_contrast",
        "keep_prev_scale", "show_texts", "show_attributes", "show_labels",
        "show_masks", "show_linking"]}
    bool_keys.update({"toggle_visibility_shapes": "show_shapes",
                      "toggle_auto_use_last_label": "auto_use_last_label",
                      "use_system_clipboard": "system_clipboard"})
    for key, source_key in bool_keys.items():
        if key not in actual["checks"] or actual["checks"][key] != expected[source_key]:
            errors.append(f"checkable action {key} differs or is absent")
    dock_keys = {"Description": "description_dock", "Files": "file_dock",
                 "Labels": "label_dock", "Objects": "shape_dock", "ImageFlags": "flag_dock"}
    for name, source_key in dock_keys.items():
        wanted = dict(expected[source_key])
        if name == "ImageFlags" and not expected["flags"]:
            wanted["show"] = False  # actual upstream cold-start label_widget rule
        if actual["docks"].get(name) != wanted:
            errors.append(f"dock {name}: {actual['docks'].get(name)} != {wanted}")
    model_rows = []
    configs = args.upstream / "anylabeling/configs/auto_labeling"
    for family, values in actual["models"].items():
        source_family = family.replace("-world", "-worldv2")
        path = configs / (source_family.replace("-", "_") + ".yaml")
        if not path.is_file():
            continue
        config = yaml.safe_load(path.read_text())
        for field, source_key in [("confidence", "conf_threshold"), ("iou", "iou_threshold")]:
            if source_key in config and field in values:
                if abs(float(config[source_key]) - values[field]) > 1e-6:
                    errors.append(f"{family} {field}: {values[field]} != {config[source_key]}")
                model_rows.append(f"{family}/{field}")
    if missing:
        errors.append(f"unregistered source shortcuts: {missing}")
    # Execute the actual upstream palette and stylesheet generators. This
    # checks the window's live properties, independently of the copied QSS.
    theme_source = args.upstream / "anylabeling/views/labeling/utils/theme.py"
    spec = importlib.util.spec_from_file_location("upstream_theme", theme_source)
    theme = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theme)
    theme._active_mode, theme._active_theme = "dark", theme.DARK
    # Qt's stylesheet engine resolves palette roles on each widget. Compare
    # that effective palette with the source application's actual setup.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QMainWindow
    qt_app = QApplication.instance() or QApplication([])
    qt_app.setPalette(theme.get_dark_palette())
    qt_app.setStyleSheet(theme.get_app_stylesheet())
    reference_window = QMainWindow()
    reference_window.show()
    qt_app.processEvents()
    palette = reference_window.palette()
    from PyQt6.QtGui import QPalette
    theme_checks = []
    qt_palette_differences = []
    for key, color in actual["dark_palette"].items():
        group, role = key.split("/")
        wanted = palette.color(getattr(QPalette.ColorGroup, group),
                               getattr(QPalette.ColorRole, role)).name()
        if role == "PlaceholderText" and actual["qt_version"].startswith("5."):
            # Qt 5 retains the explicit placeholder palette; Qt 6 replaces
            # the window's role during stylesheet polish. The input QSS and
            # the source's declared placeholder color remain identical.
            wanted = theme.get_dark_palette().color(
                getattr(QPalette.ColorGroup, group), QPalette.ColorRole.PlaceholderText).name()
            qt_palette_differences.append(key)
        if color != wanted:
            errors.append(f"dark palette {key}: {color} != {wanted}")
        theme_checks.append(key)
    if len(theme_checks) != 48:
        errors.append("Incomplete dark palette coverage")
    qss = theme.get_app_stylesheet().replace(
        ":/images/images/", ":/CC/plugin/qAnyLabeling/images/")
    if hashlib.sha256(qss.encode()).hexdigest() != actual["dark_stylesheet_sha256"]:
        errors.append("Dark stylesheet differs from upstream generator")
    if not actual["light_stylesheet_empty"]:
        errors.append("Light theme must restore native styling")
    classifier_source = args.upstream / "anylabeling/views/labeling/classifier/utils.py"
    classifier_function = next(
        node for node in ast.parse(classifier_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "create_ai_prompt_template")
    namespace = {"List": list}
    exec(compile(ast.Module(body=[classifier_function], type_ignores=[]),
                 str(classifier_source), "exec"), namespace)
    prompt_checks = actual["classifier_prompts"]
    for row in prompt_checks:
        wanted = namespace["create_ai_prompt_template"](row["labels"], row["single_label"])
        if row["prompt"] != wanted:
            errors.append(f"classifier prompt differs: {row['labels']}/{row['single_label']}")
    sys.path.insert(0, str(args.upstream.resolve()))
    from anylabeling.views.labeling.utils import theme as panel_theme, style
    panel_checks = []
    for mode in ("dark", "light"):
        panel_theme._active_mode = mode
        panel_theme._active_theme = getattr(panel_theme, mode.upper())
        actual_panels = actual["panel_styles"][mode]
        for key, function in (("dock", style.get_dock_style),
                              ("panel", style.get_panel_style),
                              ("description", style.get_plain_text_edit_style),
                              ("checkbox", style.get_checkbox_indicator_style)):
            expected_style = function().replace(
                ":/images/images/", ":/CC/plugin/qAnyLabeling/images/").replace(
                "#sidebarPanel", "#AnnotationSidebarSection").replace(
                "#FileList", "#AnnotationFiles")
            if actual_panels[key] != expected_style:
                errors.append(f"{mode}/{key} differs from source widget style")
            panel_checks.append(f"{mode}/{key}")
        if actual_panels["canvas_background"] != panel_theme.get_theme()["background"]:
            errors.append(f"{mode} canvas background differs from source")
    report = {
        "classifier_prompt_checks": len(prompt_checks),
        "panel_style_checks": panel_checks,
        "canvas_backgrounds": {mode: actual["panel_styles"][mode]["canvas_background"]
                               for mode in ("dark", "light")},
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shortcut_checks": checked, "missing_shortcut_controls": missing,
        "checkable_action_checks": list(bool_keys), "dock_checks": list(dock_keys),
        "exact_model_config_checks": model_rows,
        "theme_source_sha256": hashlib.sha256(theme_source.read_bytes()).hexdigest(),
        "dark_palette_checks": theme_checks,
        "qt_version": actual["qt_version"],
        "qt5_placeholder_roles_checked_against_declared_source_palette": qt_palette_differences,
        "dark_stylesheet_sha256": actual["dark_stylesheet_sha256"],
        "light_stylesheet_empty": actual["light_stylesheet_empty"],
        "complete_window_visual_parity": False,
        "scope": "defaults, shortcuts, dark window palette, light/dark panel QSS, canvas pixels and classifier prompts; not full screenshot parity",
        "errors": errors, "verdict": "PASS" if not errors else "FAIL",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
