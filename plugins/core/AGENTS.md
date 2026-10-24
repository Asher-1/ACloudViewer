# plugins/core — Scoped Guide

Feature plugins live in `Standard/`, I/O plugins in `IO/`. This file is the
scoped navigation for the largest hot zone in the repository (2,500+ files);
see the root [AGENTS.md](../../AGENTS.md) for the global map, build options, and
hard rules.

## Plugin layout (Standard)

- One folder per plugin: `q<Name>/` with `CMakeLists.txt`, `info.json`, `.qrc`, `README.md`
- CMake target is `<QNAME>_PLUGIN`, registered via `AddPlugin(NAME ...)` in
  [Plugins.cmake](../cmake/Plugins.cmake); enable with `-DPLUGIN_STANDARD_Q<NAME>=ON`
- Base class `ccStdPluginInterface` lives in [CVPluginAPI](../../libs/CVPluginAPI/);
  the stub loader in `libs/CVPluginStub/`
- AI plugins (`qDA3`, `qFreeSplatter`, `qYOLO`, …) require `AICore_ENABLED=ON`
  and link `libAICore`; ggml model changes must follow the patch flow in
  [ggml rules](../../.agents/rules/acloudviewer-ggml-aicore.mdc)

## Scoped rules (read when touching this tree)

- Plugin development: [acloudviewer-plugin-dev.mdc](../../.agents/rules/acloudviewer-plugin-dev.mdc)
- AI / ggml / AICore: [acloudviewer-ggml-aicore.mdc](../../.agents/rules/acloudviewer-ggml-aicore.mdc)
- Agent / JSON-RPC integration: [acloudviewer-agent-dev.mdc](../../.agents/rules/acloudviewer-agent-dev.mdc)
- CI debugging: [acloudviewer-ci-debugging.mdc](../../.agents/rules/acloudviewer-ci-debugging.mdc)

## Hot-zone conventions

- Naming: plugin folders `q` + PascalCase; new entity classes `cc` + PascalCase
- Dialog UI lives in `src/` (e.g. `qFreeSplatter/src/FreeSplatterDialog.cpp`);
  keep worker logic separate and testable — hot plugins have no UI tests today,
  add focused tests for any algorithm core
- UI performance: lightweight VTK preview + debounced `renderScene()` during
  slider drag; full sync (`ensureRepresentation`, `changeEntityProperties`) on release
- Folder recursion: use `obj->isGroup()` (`HIERARCHY_OBJECT`), not
  `getChildrenNumber() > 0`
- Multi-view: `ecvDisplayTools` is per-view (`ecvViewContext&`); redraw targets
  a specific view, not a global display
