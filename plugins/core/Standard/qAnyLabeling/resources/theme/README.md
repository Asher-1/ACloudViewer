# Upstream window theme

`dark.qss` is generated from `get_app_stylesheet()` in X-AnyLabeling-GGML
`745f6c355f94cddb06f6965b3a4118f82033e935`,
`anylabeling/views/labeling/utils/theme.py` (GPL-3.0-only).
Only the Qt resource prefix is adapted to this plugin. Source SHA-256:
`49b21421d72d1ab15a7916c0ae968e1731aeb790d8ea33c4d780f841b3ef18b1`.

The stylesheet is compiled into the plugin with Qt RCC. No external theme
folder is needed at installation or runtime. See `UPSTREAM-LICENSE`.

`panels.json` contains both light and dark results of upstream `style.py`
`get_dock_style`, `get_panel_style`, `get_plain_text_edit_style` and
`get_checkbox_indicator_style`. Only resource prefixes and the two local
widget object-name selectors are adapted. Source SHA-256:
`10ad46b295a160322fa69412d60f14fa1de81510b9daf800e288d70bd395a947`.
The configuration comparator executes all eight actual source generators
and compares the styles on the live plugin widgets. It also samples the
canvas pixels in both themes; this does not establish full screenshot parity.
