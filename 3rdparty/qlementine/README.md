# Qlementine: app-private Qt 5 / Qt 6 source port

ACloudViewer pins upstream **v1.5.2**, commit
`7825161fce66ac5a49bc0ffe39ab13df5ff8ff4a`, from
[oclero/qlementine](https://github.com/oclero/qlementine/tree/v1.5.2).
This is a maintained source backport, **not an upstream-supported Qt 5 release**.

## Ownership and build

The integration follows the repository's libpng-style ExternalProject flow in
`qlementine.cmake`: official upstream URL, pinned version/SHA-256, the shared
third-party download cache, and an explicit CMake `PATCH_COMMAND`. No upstream
source archive is stored in this directory. A reviewed Git patch is applied
with `git apply --no-index` into extracted build sources; never edit those
sources. The adjacent manifest records original and patched hashes for every
changed file. Every reused build validates the complete patched state, not a
prior done stamp. A pin/patch/manifest change selects fresh ExternalProject
sources. A first build needs the official download or a pre-populated verified
shared cache; subsequent cached builds need no network.

The embedding application's central `cmake/CMakeExternalLibs.cmake` selects and
finds Qt once through `USE_QT6`. This recipe consumes its `QT_PREFIX` package
family and unified `Qt::` targets; there is no second outer version branch or
outer `find_package`. Only the independent ExternalProject CMake process performs
its own `find_package` for Qt, constrained to the exact parent-selected package
paths forwarded below. The same public application appearance API and saved theme identifiers are used for
both majors; this is not binary compatibility between Qt 5 and Qt 6. The library
requires Qt Core, Widgets and Svg, C++17.

**Supported Qt range: every release from Qt 5.12 through Qt 6 (all 6.x
minors).** Qt 5.12 is the source floor; no Qt installation with backported
APIs or enumerators is required. Version-dependent APIs resolve through
version guards in the compatibility patch and the application's unified
`core/include/QtCompat.h`, so stock SDKs across the whole range compile the
same pinned source. Parent toolchains, deployment targets
and bounded build concurrency remain authoritative. Qt autogen is capped at four
jobs. ACloudViewer links one private static copy into CVAppCommon and does not
install the standalone style plugin into the platform-style search path.

The library still builds its original widgets, theme editor, animations and
optional standalone style plugin. The latter was also compiled in the isolated
Qt5 probe; it is deliberately excluded from app deployment, not removed to make
the port compile.

## Compatibility changes

- Versioned event signatures, QVariant type queries, icon DPR overloads, menu
  shortcut fields, line-edit metrics and tab-index access.
- One local painter-state RAII adapter for Qt versions before the upstream
  QPainterStateGuard class (Qt6.9). No drawing paths are disabled.
- Pixel/logical-size and screen APIs compatible with the project's Qt5.12
  source floor; **actual local build/runtime validation uses Qt5.15.2**.
- Direct includes and a target-private exclusion of an embedding application's
  opt-in QStringBuilder macro, avoiding changed deferred-expression semantics.
- Proper unpolish restoration in the existing widget registry: record only
  properties actually changed by the framework; restore only while their values
  still equal the framework's applied values. Application changes made during
  Custom mode survive exit, including dimensions, focus and layout margins.
  Restore combo delegates and message-document margins symmetrically as well.
- No forced combo AdjustToContents or slider-orientation changes: styles must
  not override application layout policy or business orientation.
- Safe animation-manager iteration/teardown, idempotent resource/font setup,
  app-state restoration and asynchronous menu callback ownership.
- Application custom delegates without Q_OBJECT remain custom; Qt metadata alone
  cannot distinguish such subclasses from their Qt base.
- Cross-version API shims for `QTabBar::isTabVisible` (Qt5.15+) and
  `QStyle::SE_PushButtonBevel` (Qt6.7+) are owned by the application's unified
  `core/include/QtCompat.h` (`qtCompatTabBarIsTabVisible`,
  `qtCompatPushButtonBevelSubElement`), not by this port; they only use
  QtCore/QtWidgets headers every consumer already links. The recipe forwards
  `QLEMENTINE_QTCOMPAT_INCLUDE_DIR` and folds the header's SHA-256 into the
  ExternalProject source identity; the private `QtCompat.hpp` bridges to it.
  The `QSvgRenderer::setAspectRatioMode` (Qt6.7+) shim stays in that private
  header because Qt Svg is a port-only dependency and the unified header must
  not impose it on its other consumers. The pre-6.7 bevel value stays out of
  switch labels (`-Wswitch`).
  Compilation is additionally verified against stock Qt5.14.2, so the port no
  longer requires a Qt installation with backported enumerators.

Compatibility and lifecycle corrections are isolated in the pinned library,
except the cross-version API shims above, which the unified application
compatibility header owns for every Qt consumer.
There is no second application ThemeManager, no global old-QSS overlay and no
per-task or per-plugin rendering backend.

## Application integration

Only explicit **Custom** mode installs Qlementine. System default, Fusion and
other genuine platform styles do not install it. The existing application owner
continues to handle preview, commit, cancellation and startup migration.

Legacy theme names are preserved as identifiers. Their colors are now explicit
JSON token presets sharing Qlementine geometry and state handling; old CSS is
not parsed or layered over the backend at runtime. Existing optional legacy UI
color readers may still retain their original resources. Approved DB Tree local
rules, ParaView functional artwork, plugin-local resources and AnyLabeling files
are not rewritten or auto-tinted.

## Validation and limits

```bash
cmake --build build_app --target ACloudViewer -j4
python3 scripts/icon_hd/verify_appearance_modes.py --build build_app --out /tmp/appearance-owner
python3 scripts/icon_hd/verify_qt_components.py --build build_app --out /tmp/qlementine-gallery
```

The first probe exercises the real Display Options dialog, isolated persistent
settings and separate-process restore. The second exercises real-owner widget
windows, popup/menu/tab/dock input, fractional DPI, disabled rendering, animation
switches and Native/Custom round trips. Keep all Qt diagnostics; offscreen
limitations must not be misreported as zero warnings.

Local evidence is Linux Qt5.15.2. A full application build (GUI, plugins,
pybind) plus the focused UI/shortcut ctest rows also pass on stock Linux
Qt5.14.2, exercising the pre-5.15 and pre-6.7 shim branches. Qt5.12, Qt6,
Windows/macOS and real multi-monitor
transitions still require their own SDK/runtime matrix. Do not turn the shared
source path into an untested promise of regression-free or binary-seamless
upgrades. Visual approval of the application-specific density and colors remains
separate from compilation and state/lifetime correctness.

## Controlled upstream upgrades

Update the default version and official archive digest in `qlementine.cmake`,
review the compatibility patch against that exact upstream, and regenerate the
full before/after file manifest. The runner rejects a mismatched version, patch
identity, partial application or drift. Do not merely change the tag and assume
an old patch still means the same thing; do not track floating HEAD. Test a fresh
ExternalProject, verified reuse, intentional corruption/conflicts, consumer
linking and installed license/dependency closure. Release and Debug archive
locations are distinct and use the parent's MSVC CRT/configuration.

The install subtree is private to the dependency. Application packaging installs
its license notices but does not export dependency headers or a global style
plugin as an application API. Native modes do not depend on a Qt plugin search
path to work.

## Licensing

Upstream code is MIT; original notices are retained in the immutable source and
patch context. Bundled Inter/Inter Display and Roboto Mono font binaries remain
unmodified and use SIL Open Font License 1.1. Complete notices are supplied in
this directory and installed into the application's shared license directory.
Font notices were obtained from the [Inter project](https://github.com/rsms/inter/blob/master/LICENSE.txt)
and [Google Fonts Roboto Mono](https://github.com/google/fonts/blob/main/ofl/robotomono/OFL.txt).
ACloudViewer's overall GPL license is not replaced by the dependency's MIT license.
