# Unified shortcut system

## Status and re-verification

This is the implementation contract for the shortcut refactor, not a claim that
all consumers have already migrated. The isolated Qt 5.15 audit probe was rerun
against the starting worktree on 2026-11-23. Its output again showed:

```text
prefixMatches=1 detectorConflicts=0
secondaryConflicts=0
resetFirst=Ctrl+K
resetReopened=Ctrl+8
actionKey=X standaloneKey=S
vtkChordRegistered=1 vtkFirstStrokeFound=0
```

The starting-worktree review also confirmed a second hardcoded VTK default-key
table and an interactor-style raw-key fallback to `handleShortcut()`. That
combination explained why removing the Qt lookup table alone could not make old
keys silent after remapping. The implementation and production VTK regression
checks below supersede this baseline finding; it is retained as the reason for
using one dispatch owner.

## Implemented scope and verification (2026-11-24)

The implemented owners now share the registry and transaction described below:
main-window commands, selection render/chart proxies, per-frame view actions,
tabs, VTK operations, log commands, Python-view Run, spreadsheet Copy and
exported plugin actions. Hover Copy uses the existing transient modal owner in
its actual render-widget context, and releases it when tooltip content is no
longer valid. Ctrl/Shift mouse modifiers remain gestures. Data-generated Python
script/plugin menus explicitly opt out of persistent command identity.

The following checks passed on Ubuntu 20.04, GCC 9.4 and Qt 5.15.2:

| Gate | Evidence |
| --- | --- |
| Historical standalone QtTest fixture | Before GoogleTest migration: 13 cases plus init/cleanup passed; this is not the current suite count |
| Production application | `cmake --build build_app --target ACloudViewer -j6` succeeded |
| Modified exported-action plugins | QPCL, QFACETS, QSRA, QCANUPO, Q3DMASC, QCOLORIMETRIC_SEGMENTER, QPYTHONRUNTIME and QSIBR targets built and linked |
| Production VTK integration probe | 11 checks passed against the actual Qt/VTK libraries, including operation execution and removal of raw default-key fallback |
| Production render-widget key events under Xvfb | Additional four checks passed: actual view owns focus, old key is silent, chord and Meta execute the real projection operation |
| Existing plugin gates | `test_cvpluginapi_ui_helpers`, `test_qpythonruntime_plugin_loading`, `test_qpythonruntime_readline_isolation`: 3/3 passed |

The graphical probe must initialize a real `vtkGLView` and its display-tools
owner before showing the widget. Showing an isolated raw QVTK widget without a
display owner violates its resize-event contract; the integration fixture uses
the production creation path instead of changing unrelated production code.

The shortcut regression cases additionally cover rejected import transactions, batch key swaps,
invalid file types/versions/sequences, legacy alias collisions, failed settings
writes, unopened plugin profiles retained until registration, inactive command
metadata, actual Python editor Run, native text Copy, per-view focus and
transient tool release. The default-key audit reads the actual Designer file and
VTK catalog, plus runtime selection/layout defaults.

No claim is made that every plugin's internal editor gestures are centrally
configurable. Plugin primary/exported commands are managed; anonymous external
multi-command plugins must supply stable IDs, and plugin-internal controls may
retain their own local behavior. End-to-end MainWindow render/chart selection
with loaded scene data, every composite-view arrangement, arbitrary keyboard
layouts, Qt 6 and macOS/Windows GUI acceptance remain unverified. The production
VTK operation and independent Qt routing checks do not replace that matrix.

Reproduce the production VTK gate after a GUI build with unit tests enabled:

```bash
cmake --build build_app --target test_shortcuts test_vtk_shortcuts -j4
ctest --test-dir build_app -R '^test_(shortcuts|vtk_shortcuts)$' --output-on-failure
# Linux only: registered when xvfb-run is available at configure time.
ctest --test-dir build_app -R '^test_vtk_shortcuts_graphics$' --no-tests=error --output-on-failure
```

## Lifecycle and settings follow-up (2026-11-25)

Temporary shortcuts retain the tool owner's requested activation separately from
its effective Qt listener. Explicit activation takes priority; releasing or
removing that owner restores the most recently requested overlapping owner,
never a tool that was explicitly stopped. Window merges, splits, context changes,
visibility and availability changes reconcile ownership without a settings write.
Decorators observe suspension without cancelling the owner's request. Disabled
persistent QActions retain their configured keys but do not block live owners.

The settings table separates configured bindings from the union of effective
bindings across loaded instances; that union is not a claim about every instance
or the focused window. Factory defaults remain visible in tooltips. Conflict and
save failures keep the editor open with its complete draft. Tooltip/status hints
append an explicit current shortcut line rather than replacing key-shaped text
inside descriptions; clearing a binding displays Unbound.

The Python plugin editor no longer intercepts ambiguous shortcuts application-wide.
Its editor actions use widget-with-children scope and normal Qt dispatch.
qAnyLabeling reuses central sequence validity and symmetric prefix checks, including
alternative bindings. Its settings save checks QSettings synchronization before
changing live actions, and startup validates the complete saved batch before
applying any override. Invalid startup batches retain factory bindings and leave
the stored configuration untouched. Annotation-workspace ownership narrows host
shortcut context rather than changing the host action's semantic enabled state.
The plugin still owns its settings namespace and primary-binding editor; this is
not a migration of its entire command catalog into the central settings table.

## Default shortcut reference for new users

This section records **factory declarations**, not your saved configuration.
Open **Shortcuts** to inspect the currently effective registered bindings. A blank
factory binding means that the button still works with the mouse; it does not
mean the command is unavailable. `Ctrl+K, Ctrl+P` means successive strokes;
`Ctrl+Shift+P` means one chord. A menu label's `&` marks an Alt mnemonic and is
not a registered command shortcut. Pure Ctrl/Shift while selecting are mouse
modifiers, not editable command bindings.

Literal key strings below use Qt PortableText. On macOS, Qt may display Control
and Meta using platform symbols; **Qt StandardKey** entries are resolved by Qt
for the active platform/key scheme and must not be read as identical literal
keys on every OS. Main-window fullscreen additionally has an explicit macOS
source override: `Ctrl+Meta+F` replaces Designer's `F9`.

### Main-window commands with factory keys

The current [MainWindow.ui](../../app/ui_templates/MainWindow.ui) declares 240 actions:
35 with explicit keys and 205 without. Runtime commands and overrides follow in
separate tables so the Designer inventory does not hide them.

| Command ID | Button / menu label | Factory binding | Source |
| --- | --- | --- | --- |
| actionOpen | Open | Ctrl+O | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1086) |
| actionSave | Save | Ctrl+S | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1110) |
| actionSaveProject | Save project | Ctrl+Shift+S | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1128) |
| helpAction | Help | F1 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1161) |
| actionExit | Exit | Ctrl+Q | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1194) |
| actionSetBackgroundColor | Set background color | Ctrl+B | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1215) |
| actionSetViewFront | Front View | Alt+5 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1236) |
| actionSetViewLeft | Left Side View | Alt+4 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1257) |
| actionSetViewTop | Top view | Alt+8 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1281) |
| actionFullScreen | Full screen | F9 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1411) |
| actionResetRegisteredSettings | Reset registered settings | Ctrl+Shift+R | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1456) |
| actionConsole | Console | F8 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1527) |
| actionExclusiveFullScreen | Full screen (3D view) | Shift+F11 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1548) |
| actionDelete | Delete | Del | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1569) |
| actionSetUniqueColor | Set unique color | Ctrl+Shift+C | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1584) |
| actionZoomToBox | Zoom to Box | Alt+Z | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1759) |
| actionZoomAndCenter | Zoom  Center | Z | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1783) |
| actionDisplayOptions | Display settings | F12 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1819) |
| actionSetViewBack | Back View | Alt+0 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1837) |
| actionSetViewRight | Right Side View | Alt+6 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1858) |
| actionSetViewBottom | Bottom View | Alt+2 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1879) |
| actionSetViewIso1 | Iso 1 | Alt+7 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1900) |
| actionSetViewIso2 | Iso 2 | Alt+9 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1921) |
| actionTracePolyline | Trace Polyline | Ctrl+Shift+P | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2177) |
| actionApplyTransformation | Apply transformation | Ctrl+T | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3085) |
| actionPerspectiveProjection | Perspective  Projection | F3 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3343) |
| actionOrthogonalProjection | Orthogonal  Projection | F4 | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3364) |
| actionNew3DView | New 3D View | Ctrl+Shift+N | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3610) |
| actionSaveViewportAsObject | Save viewport as object | Ctrl+Alt+V | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3922) |
| actionPasteCloudFromClipboard | Paste from clipboard | Ctrl+Shift+V | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3955) |
| actionLockRotationAxis | Lock rotation about an axis | Alt+L | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3985) |
| actionToggleActiveSFColorScale | Toggle color scale | Shift+C | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L4006) |
| actionShowActiveSFPrevious | Show previous SF | Shift+Up | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L4026) |
| actionShowActiveSFNext | Show next SF | Shift+Down | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L4058) |
| actionAnimation | Animation | Shift+A | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L4078) |

<details>
<summary>All 205 main-window actions without a Designer binding</summary>

These are explicitly **unbound in Designer**, not proof that every runtime owner
leaves them unbound or makes them configurable. Toolbar visibility actions,
menus and dynamic data entries may be excluded from central settings.

| Command ID | Button / menu label | Factory binding | Source |
| --- | --- | --- | --- |
| actionPrimitiveFactory | Primitive factory | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1149) |
| actionAbout | About | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1182) |
| actionClearAll | Clear All | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1302) |
| actionPointPicking | Point picking | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1314) |
| actionPointListPicking | Point list picking | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1329) |
| BlueThemeAction | Blue | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1341) |
| LightBlueThemeAction | LightBlue | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1346) |
| englishAction | English | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1351) |
| chineseAction | Chinese | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1356) |
| actionEnableQtWarnings | Enable Qt warnings in Console | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1361) |
| DarkBlueThemeAction | DarkBlue | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1376) |
| BlackThemeAction | Black | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1381) |
| LightBlackThemeAction | LightBlack | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1386) |
| FlatBlackThemeAction | FlatBlack | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1391) |
| DarkBlackThemeAction | DarkBlack | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1396) |
| PsBlackThemeAction | PsBlack | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1401) |
| SilverThemeAction | Silver | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1406) |
| actionResetGUIElementsPos | reset all GUI element positions | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1441) |
| actionSaveCustomLayout | Save Current Layout | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1477) |
| actionRestoreDefaultLayout | Restore Default Layout | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1488) |
| actionRestoreCustomLayout | Restore Custom Layout | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1499) |
| actionRestoreWindowOnStartup | Restore Default Window on Startup | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1510) |
| actionColorize | Colorize | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1605) |
| actionChangeColorLevels | Levels | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1620) |
| actionSetColorGradient | Height Ramp | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1628) |
| actionRGBToGreyScale | Convert to grey scale | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1643) |
| actionRGBGaussianFilter | Gaussian filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1658) |
| actionRGBBilateralFilter | Bilateral filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1673) |
| actionRGBMeanFilter | Mean filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1688) |
| actionRGBMedianFilter | Median filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1699) |
| actionScalarFieldFromColor | Convert to Scalar field | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1710) |
| actionInterpolateColors | Interpolate from another entity | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1722) |
| actionEnhanceRGBWithIntensities | Enhance with intensities | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1733) |
| actionClearColor | Clear colors | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1741) |
| actionGlobalZoom | Global Zoom | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1807) |
| actionClone | Clone | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1942) |
| actionMerge | Merge | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1957) |
| actionClipFilter | Clip | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1972) |
| actionSegment | Segment | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1990) |
| actionComputeNormals | Compute | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2002) |
| actionClearNormals | Clear | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2017) |
| actionInvertNormals | Invert | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2032) |
| actionDisplayMainTools | Main | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2043) |
| actionDisplayScalarFieldsTools | Scalar fields | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2060) |
| actionDisplayViewTools | View | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2077) |
| actionDisplaySelectionTools | Selection | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2094) |
| actionAboutPlugins | About Plugins... | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2111) |
| actionOpenColorScalesManager | Color Scales Manager | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2123) |
| actionDistanceWidget | Distance Widget | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2132) |
| actionProtractorWidget | Protractor Widget | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2147) |
| actionContourWidget | Contour Widget | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2162) |
| actionComputeOctree | Compute | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2198) |
| actionResampleWithOctree | Resample | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2213) |
| actionOrientNormalsMST | With Minimum Spanning Tree | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2224) |
| actionOrientNormalsFM | With Fast Marching | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2232) |
| actionConvertNormalToHSV | HSV colors | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2240) |
| actionConvertNormalToDipDir | Dip/Dip direction SFs | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2248) |
| actionSurface | Surface | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2256) |
| actionComputeMeshAA | Delaunay 2.5D (XY plane) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2271) |
| actionComputeMeshLS | Delaunay 2.5D (best fitting plane) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2282) |
| actionMeshTwoPolylines | Surface between 2 polylines | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2293) |
| actionMeshScanGrids | Mesh scan grids | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2301) |
| actionConvertTextureToColor | Convert texture/material to RGB | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2312) |
| actionSamplePointsOnMesh | Sample Points | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2324) |
| actionSmoothMeshLaplacian | Smooth (Laplacian) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2339) |
| actionSubdivideMesh | Subdivide | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2351) |
| actionMeasureMeshSurface | Measure surface | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2363) |
| actionMeasureMeshVolume | Measure volume | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2374) |
| actionFlagMeshVertices | Flag vertices by type | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2382) |
| actionSmoothMeshSF | Smooth | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2393) |
| actionEnhanceMeshSF | Enhance | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2404) |
| actionSamplePointsOnPolyline | Sample points | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2415) |
| actionSmoothPolyline | Smooth | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2423) |
| actionCreatePlane | Create | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2428) |
| actionFitPlaneProxy | Fit | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2439) |
| actionEditPlane | Edit | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2450) |
| actionComputeKdTree | Compute Kd-tree | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2461) |
| actionComputeBestFitBB | Bounding box P.C.A. fit | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2469) |
| actionDistanceToBestFitQuadric3D | Distance map to best-fit 3D quadric | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2480) |
| actionDistanceMap | Distance map | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2488) |
| actionAlign | Auto align clouds | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2496) |
| actionSNETest | SNE test | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2507) |
| actionCNETest | CNE test | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2518) |
| actionFindBiggestInnerRectangle | Find biggest inner rectangle (2D) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2532) |
| actionCreateCloudFromEntCenters | Create cloud from selected entities centers | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2540) |
| actionComputeBestICPRmsMatrix | Compute best registration RMS matrix | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2548) |
| actionEnableVisualDebugTraces | Enable Visual Debug Traces | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2559) |
| actionShowHistogram | Show histogram | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2570) |
| actionComputeStatParams | Compute stat. params | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2582) |
| actionConvertToRGB | Convert to RGB | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2597) |
| actionConvertToRandomRGB | Convert to random RGB | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2612) |
| actionRenameSF | RenameSF | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2620) |
| actionEditGlobalShiftAndScale | Edit global shift and scale | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2628) |
| actionSubsample | Subsample | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2640) |
| actionSFGradient | Gradient | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2655) |
| actionCompute2HalfDimVolume | Compute 2.5D volume | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2670) |
| actionCloudCloudDist | Cloud/Cloud Dist. | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2678) |
| actionCloudMeshDist | Cloud/Mesh Dist | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2693) |
| actionCPS | Closest Point Set | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2708) |
| actionLabelConnectedComponents | Label Connected Comp. | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2719) |
| actionKMeans | K-Means | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2734) |
| actionFrontPropagation | Front propagation | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2745) |
| actionFitPlane | Plane | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2759) |
| actionFitSphere | Sphere | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2774) |
| actionFitFacet | 2D polygon (facet) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2789) |
| actionFitQuadric | 2.5D quadric | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2801) |
| actionExportCloudInfo | Export cloud info | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2809) |
| actionExportPlaneInfo | Export plane info | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2824) |
| actionComputeGeometricFeature | Compute geometric features | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2839) |
| actionRemoveDuplicatePoints | Remove duplicate points | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2850) |
| actionDeleteScalarField | Delete | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2858) |
| actionDeleteAllSF | Delete all (!) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2876) |
| actionAddConstantSF | Add constant SF | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2887) |
| actionAddIdField | Add point indexes as SF | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2899) |
| actionExportCoordToSF | Export coordinate(s) to SF(s) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2910) |
| actionSetSFAsCoord | Set SF as coordinate(s) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2921) |
| actionInterpolateSFs | Interpolate from another entity | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2932) |
| actionScalarFieldArithmetic | Arithmetic | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2943) |
| actionGaussianFilter | Gaussian filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2961) |
| actionBilateralFilter | Bilateral filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2976) |
| actionFilterByValue | Filter By Value | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L2991) |
| actionConvertPolylinesToMesh | Contour plot (polylines) to mesh | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3006) |
| actionUnroll | Unroll | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3017) |
| actionMatchBBCenters | Match bounding-box centers | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3032) |
| actionMatchScales | Match scales | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3043) |
| actionPointPairsAlign | Align (point pairs picking) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3055) |
| actionRegister | Fine registration (ICP) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3070) |
| actionTranslateRotate | Translate/Rotate | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3102) |
| actionApplyScale | Multiply/Scale | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3114) |
| actionGlobalShiftSettings | Global Shift settings | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3129) |
| actionShortcutSettings | Shortcut Settings... | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3141) |
| actionLogManager | Log Manager... | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3156) |
| GrayThemeAction | Gray | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3171) |
| LightGrayThemeAction | LightGray | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3176) |
| DarkGrayThemeAction | DarkGray | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3181) |
| FlatWhiteThemeAction | FlatWhite | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3186) |
| BFThemeAction | BF | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3191) |
| TestThemeAction | Test | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3196) |
| MaterialDarkThemeAction | Material Dark | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3201) |
| MaterialLightThemeAction | Material Light | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3206) |
| NordThemeAction | Nord | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3211) |
| DraculaThemeAction | Dracula | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3216) |
| FluentThemeAction | Fluent Design | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3221) |
| MacOSThemeAction | macOS Big Sur | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3226) |
| OneDarkThemeAction | One Dark | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3231) |
| CatppuccinThemeAction | Catppuccin | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3236) |
| TokyoNightThemeAction | Tokyo Night | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3241) |
| GruvboxThemeAction | Gruvbox | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3246) |
| DfaultThemeAction | Default | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3251) |
| actionSORFilter | SOR filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3256) |
| actionNoiseFilter | Noise filter | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3271) |
| actionToggleOrientationMarker | Toggle Orientation Marker | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3286) |
| actionToggleCameraOrientationWidget | Camera Orientation Widget | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3304) |
| actionSemanticAnnotation | Semantic Annotation | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3325) |
| actionBoxAnnotation | Box Annotation | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3334) |
| actionSliceFilter | Slice | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3385) |
| actionDecimateFilter | Decimate | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3400) |
| actionIsoSurfaceFilter | IsoSurface | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3415) |
| actionThresholdFilter | Threshold | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3430) |
| actionSmoothFilter | Smooth | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3445) |
| actionProbeFilter | Probe | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3460) |
| ParaviewThemeAction | Paraview | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3475) |
| actionStreamlineFilter | Streamline | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3480) |
| actionGlyphFilter | Glyph | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3495) |
| actionModifySensor | Edit | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3510) |
| actionCreateGBLSensor | Create | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3519) |
| actionShowDepthBuffer | Create/show depth buffer | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3524) |
| actionExportDepthBuffer | Export depth buffer | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3532) |
| actionComputePointsVisibility | Compute points visibility (with depth buffer) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3537) |
| actionCreateCameraSensor | Create | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3542) |
| actionProjectUncertainty | Project Uncertainty | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3547) |
| actionCheckPointsInsideFrustum | Compute points visibility (with octree) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3552) |
| actionViewFromSensor | View from sensor | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3557) |
| actionComputeDistancesFromSensor | Compute Ranges | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3569) |
| actionComputeScatteringAngles | Compute Scattering Angles | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3580) |
| actionShowWaveDialog | 2D Waveform viewer | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3591) |
| actionCompressFWFData | Compress FWF data | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3599) |
| actionDisplayFilterTools | actionDisplayFilterTools | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3628) |
| actionDisplayAnnotationTools | actionDisplayAnnotationTools | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3639) |
| actionDisplayMeasurementsTools | Measurements | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3650) |
| actionSemanticSegmentation | Semantic Segmentation | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3667) |
| actionAutoPickPivot | Auto-pick rotation center | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3672) |
| actionResetPivot | Reset Rotation Center | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3690) |
| actionShowPivot | Show Rotation Center | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3702) |
| actionImportSFFromFile | Import SF from file | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3717) |
| actionBSplineFittingOnCloud | BSpline Fitting On Cloud | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3729) |
| actionPoissonReconstruction | Poisson Reconstruction | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3737) |
| actionVoxelSampling | Voxel Sampling | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3749) |
| actionConvexHull | Convex Hull | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3764) |
| actionDBScanCluster | DBScan Cluster | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3776) |
| actionPlaneSegmentation | Plane Segmentation | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3787) |
| actionBBCenterToOrigin | Move bounding-box center to origin | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3795) |
| actionBBMinCornerToOrigin | Move bounding-box min corner to origin | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3806) |
| actionBBMaxCornerToOrigin | Move bounding-box max corner to origin | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3817) |
| actionFlipPlane | Flip | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3828) |
| actionComparePlanes | Compare | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3839) |
| actionPromoteCircleToCylinder | Promote to cylinder | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3850) |
| actionFlipMeshTriangles | Flip triangles | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3865) |
| actionCloudPrimitiveDist | Cloud/Primitive Dist | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3880) |
| actionExportNormalToSF | Export normals to SF(s) | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3895) |
| actionFilterByLabel | Filter By Label | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3906) |
| actionColorFromScalarField | From Scalar fields | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3917) |
| actionCreateSinglePointCloud | Create single point cloud | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3940) |
| actionCheckForUpdates | Check for Updates | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L3976) |
| actionFitCircle | Circle | Unbound in Designer | [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L4046) |

</details>

### Runtime application, layout and selection commands

Undo/Redo use `QKeySequence::Undo` / `Redo` (commonly Ctrl+Z /
Ctrl+Shift+Z on the verified Linux setup; platform alternatives vary). They are
factory standard-key declarations, not a promise of one cross-platform string.

| Command ID | Command | Factory binding | Scope / status | Source |
| --- | --- | --- | --- | --- |
| actionUndo | Undo | Qt StandardKey Undo | Owning window; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionRedo | Redo | Qt StandardKey Redo | Owning window; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionToggleViewDecorations | Toggle View Decorations | Ctrl+D | Owning window; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionLayoutFullScreen | Full Screen (Layout) | F11 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionActiveViewFullScreen | Full Screen (Active View) | Ctrl+F11 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionQuickCreateView.SpreadSheet | SpreadSheet View | Ctrl+Shift+1 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionQuickCreateView.LineChart | Line Chart View | Ctrl+Shift+2 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionQuickCreateView.BarChart | Bar Chart View | Ctrl+Shift+3 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionQuickCreateView.Histogram | Histogram View | Ctrl+Shift+4 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionQuickCreateView.OrthoSlice | Orthographic Slice View | Ctrl+Shift+5 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionQuickCreateView.PythonView | Python View | Ctrl+Shift+6 | Application; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |
| actionPreviousTab | Previous Tab | Ctrl+PgUp | Owning window; Registered command | [ecvTabbedMultiViewWidget.cpp](../../app/ecvTabbedMultiViewWidget.cpp) |
| actionNextTab | Next Tab | Ctrl+PgDown | Owning window; Registered command | [ecvTabbedMultiViewWidget.cpp](../../app/ecvTabbedMultiViewWidget.cpp) |
| actionNewTab | New Tab | Ctrl+Shift+T | Owning window; Registered command | [ecvTabbedMultiViewWidget.cpp](../../app/ecvTabbedMultiViewWidget.cpp) |
| actionCloseTab | Close Tab | Ctrl+W | Owning window; Registered command | [ecvTabbedMultiViewWidget.cpp](../../app/ecvTabbedMultiViewWidget.cpp) |
| actionSelectSurfaceCells | Select Cells On | S | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionSelectSurfacePoints | Select Points On | D | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionSelectFrustumCells | Select Cells Through | F | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionSelectFrustumPoints | Select Points Through | G | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionSelectBlocks | Select Block | B | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionSelectFrustumBlocks | Select Blocks Through | N | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionGrowSelection | Grow Selection | + | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionShrinkSelection | Shrink Selection | - | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionSelectPolygonCells | Select Cells With Polygon | Unbound | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionSelectPolygonPoints | Select Points With Polygon | Unbound | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionInteractiveSelectCells | Interactive Select Cells On | Unbound | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionInteractiveSelectPoints | Interactive Select Points On | Unbound | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionHoverCells | Hover Cells On | Unbound | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionHoverPoints | Hover Points On | Unbound | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionClearSelection | Clear Selection | Unbound | Active-view route in owning window; Registered canonical command; chart S/D proxies use rectangle/polygon | [cvSelectionToolController.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvSelectionToolController.cpp) |
| actionCancelSelectionTool | Cancel Selection Tool | Esc | Owning window; fullscreen/tool precedence applies; Registered command | [MainWindow.cpp](../../app/MainWindow.cpp) |

Selection **Add / Subtract / Toggle** buttons choose persistent selection
modes. Holding **Ctrl / Shift / Ctrl+Shift** while selecting is a separate mouse
modifier contract; changing a command key does not redefine that gesture.
Clear Selection is deliberately not labelled Esc: Esc cancels the selection
mode. Layout fullscreen and legacy exclusive fullscreen are distinct commands.

Runtime-created commands with no explicit factory key are listed too. All
rows below come from [MainWindow.cpp](../../app/MainWindow.cpp), and central
registration is available when their owning feature is built. Buttons/proxies
do not acquire a second independent binding simply because they appear in more
views.

| Command ID | Command | Factory binding | Scope |
| --- | --- | --- | --- |
| view/actionCameraUndo | Camera Undo | Unbound | View frame and descendants |
| view/actionCameraRedo | Camera Redo | Unbound | View frame and descendants |
| view/actionCaptureView | Capture Screenshot | Unbound | View frame and descendants |
| view/view3DToggleAction | 3D View / 3D projection | Unbound | View frame and descendants |
| view/actionEditViewCamera | Adjust Camera | Unbound | View frame and descendants |
| view/actionResetViewCamera | Reset Camera | Unbound | View frame and descendants |
| view/actionZoomViewToData | Zoom to Data / Fit | Unbound | View frame and descendants |
| actionLinkAllCameras | Link All Cameras | Unbound | Owning window |
| actionAddCameraLink | Add Camera Link | Unbound | Owning window |
| actionManageCameraLinks | Manage Camera Links | Unbound | Owning window |
| actionEqualizeHorizontal | Equalize Horizontally | Unbound | Owning window |
| actionEqualizeVertical | Equalize Vertically | Unbound | Owning window |
| actionEqualizeBoth | Equalize Both | Unbound | Owning window |
| actionCustomPreviewSize | Custom Preview Size | Unbound | Owning window |
| actionLockViewSize300 | Lock View Size 300x300 | Unbound | Owning window |
| actionLockCustomViewSize | Lock Custom View Size | Unbound | Owning window |
| actionUnlockViewSize | Unlock View Size | Unbound | Owning window |
| actionShowFpsCounter | Show FPS Counter | Unbound | Owning window |
| actionPreview1280x720 | Preview 1280x720 | Unbound | Owning window |
| actionPreview1280x800 | Preview 1280x800 | Unbound | Owning window |
| actionPreview1280x1024 | Preview 1280x1024 | Unbound | Owning window |
| actionPreview1600x900 | Preview 1600x900 | Unbound | Owning window |
| actionPreview1920x1080 | Preview 1920x1080 | Unbound | Owning window |
| actionPreview3840x2160 | Preview 3840x2160 | Unbound | Owning window |

### Render-view VTK operations

All 22 catalog entries are configurable widget-local commands. The render
widget must own focus. These operate on the VTK view, and are not aliases for
the similarly named whole-application commands.

| Command ID | Operation | Factory binding | Source |
| --- | --- | --- | --- |
| vtk_screenshot | VTK: Screenshot | Ctrl+Alt+J | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L38) |
| vtk_camera_info | VTK: Camera Info | Ctrl+Alt+C | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L41) |
| vtk_toggle_grid | VTK: Toggle Grid | Ctrl+Alt+G | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L44) |
| vtk_toggle_lut | VTK: Toggle LUT | Ctrl+Alt+K | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L47) |
| vtk_toggle_proj | VTK: Toggle Parallel Proj | Ctrl+Alt+O | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L50) |
| vtk_fullscreen | VTK: Toggle Fullscreen | Ctrl+Alt+F | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L53) |
| vtk_stereo | VTK: Toggle Stereo | Ctrl+Alt+S | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L56) |
| vtk_zoom_in | VTK: Zoom In | Ctrl+Alt++ | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L59) |
| vtk_zoom_out | VTK: Zoom Out | Ctrl+Alt+- | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L62) |
| vtk_points_mode | VTK: Points Mode | Ctrl+Shift+D | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L67) |
| vtk_wireframe_mode | VTK: Wireframe Mode | Ctrl+Shift+W | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L70) |
| vtk_surface_mode | VTK: Surface Mode | Ctrl+Shift+F | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L73) |
| vtk_ptsize_inc | VTK: Increase Point Size | Ctrl+Shift++ | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L76) |
| vtk_ptsize_dec | VTK: Decrease Point Size | Ctrl+Shift+- | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L79) |
| vtk_save_camera | VTK: Save Camera | Ctrl+Alt+M | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L84) |
| vtk_restore_camera | VTK: Restore Camera | Ctrl+R | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L87) |
| vtk_view_front | VTK: Front View (+Z) | Ctrl+Alt+1 | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L91) |
| vtk_view_back | VTK: Back View (-Z) | Ctrl+Alt+2 | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L94) |
| vtk_view_left | VTK: Left View (-X) | Ctrl+Alt+3 | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L97) |
| vtk_view_right | VTK: Right View (+X) | Ctrl+Alt+4 | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L100) |
| vtk_view_top | VTK: Top View (+Y) | Ctrl+Alt+5 | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L103) |
| vtk_view_bottom | VTK: Bottom View (-Y) | Ctrl+Alt+6 | [VtkShortcutRegistry.h](../../libs/VtkEngine/VTKExtensions/Widgets/VtkShortcutRegistry.h#L106) |

### Local application windows

The following bindings are registered in the shared settings system while their
owners are available; they retain local widget/window scope. StandardKey values
below are Qt enums, with common Linux display examples only.

| Command ID | Command | Factory key / Qt StandardKey | Scope | Source |
| --- | --- | --- | --- | --- |
| log.find | Find | Find (usually Ctrl+F) | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.findNext | Find Next | FindNext (usually F3) | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.findPrevious | Find Previous | FindPrevious (usually Shift+F3) | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.refresh | Refresh | Refresh (usually F5) | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.close | Close | Close (usually Ctrl+W) | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.openFolder | Open Folder | Unbound | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.export | Export Log | Unbound | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.delete | Delete Logs | Unbound | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| log.cleanOld | Clean Old Logs | Unbound | Log viewer and children; configurable | [ecvLogManagerDlg.cpp](../../app/ecvLogManagerDlg.cpp) |
| python.run | Run Python Script | Ctrl+Return | Python view and children; configurable | [ecvPythonView.cpp](../../app/ecvPythonView.cpp) |
| spreadsheet.copy | Copy Table Selection | Copy (usually Ctrl+C) | Table and children; configurable | [ecvSpreadSheetView.cpp](../../app/ecvSpreadSheetView.cpp) |

### Tool-local gestures and reserved interactions

These rows are not a second list of configurable global commands. Their owner
must be active. Normal text editing, dialog Enter/Esc and widget arrow keys
remain Qt control interactions, including when a button has an Alt mnemonic.

| Active owner | Gesture | Meaning / boundary | Source |
| --- | --- | --- | --- |
| Segmentation | Space; Esc; Return; Delete; Tab; I; O; C; E | Pause; cancel; apply; apply and delete; rectangular/polygon mode; segment in; segment out; classify; export | [ecvGraphicalSegmentationTool.cpp](../../app/ecvGraphicalSegmentationTool.cpp) |
| Annotations | I; U; N; T; R; S; H; L; Space | Intersect; union; new; trim; reset; save; show/hide; label selection; pause | [ecvAnnotationsTool.cpp](../../app/ecvAnnotationsTool.cpp) |
| Graphical transformation | Space; Esc; Return | Pause; cancel; accept | [ecvGraphicalTransformationTool.cpp](../../app/ecvGraphicalTransformationTool.cpp) |
| Trace polyline | Esc; Return; C | Cancel; apply; continue button (C declared in Designer) | [ecvTracePolylineTool.cpp](../../app/ecvTracePolylineTool.cpp) |
| Measurement overlay | Esc | Cancel measurement overlay | [ecvMeasurementTool.cpp](../../app/ecvMeasurementTool.cpp) |
| Distance picking | P / Ctrl+P; 1 / Ctrl+1; 2 / Ctrl+2; N | Pick arbitrary point; first endpoint; second endpoint; surface-normal option; Ctrl variants snap to point | [cvDistanceTool.cpp](../../libs/VtkEngine/Tools/MeasurementTools/cvDistanceTool.cpp) |
| Angle picking | 1 / Ctrl+1; C / Ctrl+C; 2 / Ctrl+2 | Pick first endpoint; centre; second endpoint; Ctrl variants snap to point | [cvProtractorTool.cpp](../../libs/VtkEngine/Tools/MeasurementTools/cvProtractorTool.cpp) |
| Hover tooltip | Qt StandardKey Copy | Copy visible tooltip content; temporary view-local ownership | [cvRenderViewSelectionReaction.cpp](../../libs/VtkEngine/Tools/SelectionTools/cvRenderViewSelectionReaction.cpp) |
| Orthographic slice | Arrow keys; PageUp / PageDown | Contextual slice navigation; see owning event handler | [vtkOrthoSliceViewWidget.cpp](../../libs/VtkEngine/VTKExtensions/Views/vtkOrthoSliceViewWidget.cpp) |

Distance/angle picking now declares WidgetWithChildrenShortcut on its actual
render-widget owner; it no longer requests ApplicationShortcut. This source-level
scope correction is distinct from end-to-end clipboard and measurement-tool GUI
acceptance, which is not established for every arrangement.

### qAnyLabeling annotation-window defaults

This is a **plugin-local** configuration table from
[AnnotationDefaults.inc](../../plugins/core/Standard/qAnyLabeling/include/AnnotationDefaults.inc#L193),
read through [AnnotationConfig.h](../../plugins/core/Standard/qAnyLabeling/include/AnnotationConfig.h#L47).
It is not the application's shared shortcut registry. The annotation window
uses its own settings and runtime assignments in
[AnnotationWindow.cpp](../../plugins/core/Standard/qAnyLabeling/src/AnnotationWindow.cpp); these
may override or repurpose a config key by mode. Config keys are the stable IDs
in this table; labels below are descriptive expansions, not inferred QAction
registration identities. A repeated key can be intentional across exclusive
editing modes (for example Undo and Undo Last Point); this table alone does not
prove those contexts are conflict-free. Dynamic digit shortcuts 0–9 select
labels according to the plugin's current mapping, not fixed application commands.

| Plugin config key | Descriptive action | Factory declaration | Scope / configurability |
| --- | --- | --- | --- |
| close | Close | Ctrl+W | Annotation window / mode; plugin-local configuration |
| open | Open | Ctrl+I | Annotation window / mode; plugin-local configuration |
| open_video | Open Video | Ctrl+O | Annotation window / mode; plugin-local configuration |
| open_dir | Open Dir | Ctrl+U | Annotation window / mode; plugin-local configuration |
| open_chatbot | Open Chatbot | Ctrl+1 | Annotation window / mode; plugin-local configuration |
| open_vqa | Open Vqa | Ctrl+2 | Annotation window / mode; plugin-local configuration |
| open_image_classifier | Open Image Classifier | Ctrl+3 | Annotation window / mode; plugin-local configuration |
| open_paddleocr | Open Paddleocr | Ctrl+4 | Annotation window / mode; plugin-local configuration |
| open_video_classifier | Open Video Classifier | Ctrl+5 | Annotation window / mode; plugin-local configuration |
| open_pointcloud | Open Pointcloud | Ctrl+6 | Annotation window / mode; plugin-local configuration |
| quit | Quit | Ctrl+Q | Annotation window / mode; plugin-local configuration |
| save | Save | Ctrl+S | Annotation window / mode; plugin-local configuration |
| save_as | Save As | Ctrl+Shift+S | Annotation window / mode; plugin-local configuration |
| save_to | Save To | Unbound | Annotation window / mode; plugin-local configuration |
| delete_file | Delete File | Ctrl+Delete | Annotation window / mode; plugin-local configuration |
| delete_image_file | Delete Image File | Ctrl+Shift+Delete | Annotation window / mode; plugin-local configuration |
| open_next | Open Next | D | Annotation window / mode; plugin-local configuration |
| open_prev | Open Prev | A | Annotation window / mode; plugin-local configuration |
| open_next_unchecked | Open Next Unchecked | Ctrl+Shift+D | Annotation window / mode; plugin-local configuration |
| open_prev_unchecked | Open Prev Unchecked | Ctrl+Shift+A | Annotation window / mode; plugin-local configuration |
| toggle_annotation_checked | Toggle Annotation Checked | Ctrl+Alt+K | Annotation window / mode; plugin-local configuration |
| clear_annotation_checked | Clear Annotation Checked | Ctrl+Alt+U | Annotation window / mode; plugin-local configuration |
| fail_annotation_checked | Fail Annotation Checked | Ctrl+Alt+J | Annotation window / mode; plugin-local configuration |
| zoom_in | Zoom In | Ctrl++ | Annotation window / mode; plugin-local configuration |
| zoom_out | Zoom Out | Ctrl+- | Annotation window / mode; plugin-local configuration |
| zoom_to_original | Zoom To Original | Ctrl+= | Annotation window / mode; plugin-local configuration |
| fit_window | Fit Window | Ctrl+F | Annotation window / mode; plugin-local configuration |
| fit_width | Fit Width | Ctrl+Shift+F | Annotation window / mode; plugin-local configuration |
| show_navigator | Show Navigator | F9 | Annotation window / mode; plugin-local configuration |
| toggle_image_tags | Toggle Image Tags | Ctrl+Shift+T | Annotation window / mode; plugin-local configuration |
| create_polygon | Create Polygon | P | Annotation window / mode; plugin-local configuration |
| create_brush_polygon | Create Brush Polygon | Ctrl+N | Annotation window / mode; plugin-local configuration |
| create_magic_wand | Create Magic Wand | Shift+W | Annotation window / mode; plugin-local configuration |
| create_rectangle | Create Rectangle | R | Annotation window / mode; plugin-local configuration |
| create_cuboid | Create Cuboid | Ctrl+R | Annotation window / mode; plugin-local configuration |
| create_rotation | Create Rotation | O | Annotation window / mode; plugin-local configuration |
| create_quadrilateral | Create Quadrilateral | T | Annotation window / mode; plugin-local configuration |
| create_circle | Create Circle | Shift+C | Annotation window / mode; plugin-local configuration |
| create_line | Create Line | Shift+L | Annotation window / mode; plugin-local configuration |
| create_point | Create Point | Shift+P | Annotation window / mode; plugin-local configuration |
| create_linestrip | Create Linestrip | Shift+S | Annotation window / mode; plugin-local configuration |
| edit_polygon | Edit Polygon | Ctrl+J | Annotation window / mode; plugin-local configuration |
| edit_brush_mode | Edit Brush Mode | Shift+B | Annotation window / mode; plugin-local configuration |
| delete_polygon | Delete Polygon | Delete | Annotation window / mode; plugin-local configuration |
| duplicate_polygon | Duplicate Polygon | Ctrl+D | Annotation window / mode; plugin-local configuration |
| copy_polygon | Copy Polygon | Ctrl+C | Annotation window / mode; plugin-local configuration |
| paste_polygon | Paste Polygon | Ctrl+V | Annotation window / mode; plugin-local configuration |
| undo | Undo | Ctrl+Z | Annotation window / mode; plugin-local configuration |
| undo_last_point | Undo Last Point | Ctrl+Z | Annotation window / mode; plugin-local configuration |
| add_point_to_edge | Add Point To Edge | Ctrl+Shift+P | Annotation window / mode; plugin-local configuration |
| edit_label | Edit Label | Ctrl+E | Annotation window / mode; plugin-local configuration |
| edit_digit_shortcut | Edit Digit Shortcut | Alt+D | Annotation window / mode; plugin-local configuration |
| edit_group_id | Edit Group Id | Alt+G | Annotation window / mode; plugin-local configuration |
| edit_labels | Edit Labels | Alt+L | Annotation window / mode; plugin-local configuration |
| edit_shapes | Edit Shapes | Alt+S | Annotation window / mode; plugin-local configuration |
| open_settings | Open Settings | Ctrl+0 | Annotation window / mode; plugin-local configuration |
| toggle_keep_prev_mode | Toggle Keep Prev Mode | Ctrl+P | Annotation window / mode; plugin-local configuration |
| remove_selected_point | Remove Selected Point | Backspace | Annotation window / mode; plugin-local configuration |
| group_selected_shapes | Group Selected Shapes | G | Annotation window / mode; plugin-local configuration |
| ungroup_selected_shapes | Ungroup Selected Shapes | U | Annotation window / mode; plugin-local configuration |
| hide_selected_polygons | Hide Selected Polygons | S | Annotation window / mode; plugin-local configuration |
| show_hidden_polygons | Show Hidden Polygons | W | Annotation window / mode; plugin-local configuration |
| show_overview | Show Overview | Ctrl+G | Annotation window / mode; plugin-local configuration |
| show_masks | Show Masks | Ctrl+M | Annotation window / mode; plugin-local configuration |
| show_texts | Show Texts | Ctrl+T | Annotation window / mode; plugin-local configuration |
| show_labels | Show Labels | Ctrl+L | Annotation window / mode; plugin-local configuration |
| show_linking | Show Linking | Ctrl+K | Annotation window / mode; plugin-local configuration |
| show_attributes | Show Attributes | Ctrl+Shift+L | Annotation window / mode; plugin-local configuration |
| union_selected_shapes | Union Selected Shapes | Ctrl+Shift+M | Annotation window / mode; plugin-local configuration |
| toggle_auto_use_last_label | Toggle Auto Use Last Label | Ctrl+Y | Annotation window / mode; plugin-local configuration |
| toggle_auto_use_last_gid | Toggle Auto Use Last Gid | Ctrl+Shift+G | Annotation window / mode; plugin-local configuration |
| toggle_visibility_shapes | Toggle Visibility Shapes | Ctrl+H | Annotation window / mode; plugin-local configuration |
| toggle_compare_view | Toggle Compare View | Ctrl+Alt+C | Annotation window / mode; plugin-local configuration |
| auto_label | Auto Label | Ctrl+A | Annotation window / mode; plugin-local configuration |
| auto_run | Auto Run | Ctrl+B | Annotation window / mode; plugin-local configuration |
| loop_thru_labels | Loop Thru Labels | Ctrl+Shift+N | Annotation window / mode; plugin-local configuration |
| loop_select_labels | Loop Select Labels | Ctrl+Shift+C | Annotation window / mode; plugin-local configuration |
| auto_labeling_add_point | Auto Labeling Add Point | Q | Annotation window / mode; plugin-local configuration |
| auto_labeling_remove_point | Auto Labeling Remove Point | E | Annotation window / mode; plugin-local configuration |
| auto_labeling_run | Auto Labeling Run | I | Annotation window / mode; plugin-local configuration |
| auto_labeling_clear | Auto Labeling Clear | B | Annotation window / mode; plugin-local configuration |
| auto_labeling_finish_object | Auto Labeling Finish Object | F | Annotation window / mode; plugin-local configuration |

The plugin additionally declares runtime controls that are not one-to-one
entries in this JSON (for example Redo is explicitly unbound). Other plugin
internal windows, including video/point-cloud/OCR editors and PythonRuntime,
have their own local actions. They are not claimed to be fully enumerated or
centrally rebindable here. Exported plugin menu commands are a separate surface.

### Additional plugin Designer shortcuts

These are the remaining explicitly bound first-party Designer controls outside
the main window and Trace Polyline. They belong to the named plugin window;
they are not evidence of central application shortcut registration. Runtime
code can replace them, so the table below distinguishes Designer declarations
from central registration. Controls without a Designer key are not listed.

| Source | Object ID | Command / control | Designer binding | Central status |
| --- | --- | --- | --- | --- |
| [compassDlg.ui](../../plugins/core/Standard/qCompass/ui/compassDlg.ui) | undoButton | Undo | Ctrl+Z | Plugin-local button; central status unknown |
| [compassDlg.ui](../../plugins/core/Standard/qCompass/ui/compassDlg.ui) | acceptButton | Accept | Return | Plugin-local button; central status unknown |
| [compassDlg.ui](../../plugins/core/Standard/qCompass/ui/compassDlg.ui) | closeButton | Close | Esc | Plugin-local button; central status unknown |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionNew | New | Ctrl+N | Plugin-local editor |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionOpen | Open | Ctrl+O | Plugin-local editor |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionSave | Save | Ctrl+S | Plugin-local editor |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionSaveAs | Save As | Ctrl+Shift+S | Plugin-local editor |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionClose | Close | Alt+Q | Plugin-local editor |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionRun | Run | F5 | Plugin-local editor |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionComment | Comment | Ctrl+K, Ctrl+C | Plugin-local editor chord |
| [PythonEditor.ui](../../plugins/core/Standard/qPythonRuntime/src/Ui/PythonEditor.ui) | actionUncomment | Uncomment | Ctrl+K, Ctrl+U | Plugin-local editor chord |

### Designer inventory coverage boundary

Designer declarations are source evidence, not proof of runtime command
registration or effective shortcut scope. The source tables above identify
reviewed runtime-created commands; they are not a runtime widget census.
An unbound button's normal click, dialog default-button behavior or mnemonic is
not silently counted as a factory command key.

## Re-review corrections and remaining acceptance boundary (2026-11-24)

A fresh source review found the following residual paths after the first
refactor. The current implementation corrects them at their existing owners;
these source changes must not be confused with complete desktop acceptance.

| Reviewed path | Current source state | Evidence / remaining verification |
| --- | --- | --- |
| F12 Display Options registration | Main-window collection no longer excludes actions by the `actionDisplay` name prefix. The real toolbar/data-menu exclusions remain separate, so `actionDisplayOptions` can enter the normal command collection. | [MainWindow.cpp](../../app/MainWindow.cpp#L15665), [MainWindow.ui](../../app/ui_templates/MainWindow.ui#L1819). Full MainWindow settings-dialog interaction is not inferred from the Designer default table. |
| Legacy exclusive fullscreen and chord exit | The detached host is associated with the original fullscreen QAction; its keys are matched by Qt. The filter reserves only unmodified Esc. The on-view exit hint is built from the current configured keys instead of fixed F11 text. | [MainWindow.cpp](../../app/MainWindow.cpp#L1504), [exclusive host association](../../app/MainWindow.cpp#L6627). The independent real-view Qt/VTK chord checks pass, but do not establish every MainWindow exclusive-fullscreen entry/exit arrangement. |
| Measurement picking scope | `cvPointPickingHelper` creates and re-associates its modal shortcut with the actual render widget using WidgetWithChildrenShortcut, rather than ApplicationShortcut. | [cvPointPickingHelper.cpp](../../libs/VtkEngine/Tools/PickingTools/cvPointPickingHelper.cpp#L65). Cross-window interaction for every distance/angle tool remains a GUI acceptance item. |
| Python editor input | The event filter consumes only explicit Ctrl+Space completion; ordinary key events reach `ecvPythonCodeEditor` through Qt dispatch. | [ecvPythonView.cpp](../../app/ecvPythonView.cpp#L771). The GoogleTest suite includes `PythonViewPreservesNormalEditing` on the actual `ecvPythonView`, not only the bare editor. |
| Modal sibling arbitration | Arbitration compares both exact and prefix-related sequences and disables another listener only when the two shortcut contexts overlap. Disjoint view listeners can remain enabled. | [ecvKeySequences.cpp](../../libs/CV_db/src/Shortcuts/ecvKeySequences.cpp#L739). The GoogleTest suite includes `ModalContextsAndPrefixes`. |

The current root CTest run passed all three shortcut entries:
`test_shortcuts`, `test_vtk_shortcuts`, and `test_vtk_shortcuts_graphics` (3/3,
zero failures). The first uses GoogleTest with real Qt control/event delivery;
the graphics entry uses an initialized production render-view owner under Xvfb.
This evidence covers those registered checks, not loaded-scene MainWindow
render/chart selection, all composite/fullscreen arrangements, arbitrary
keyboard layouts or all plugin-internal editors. Post-registration widget
re-association and context changes outside the reviewed owners still require
explicit tests. Qt StandardKey expansion on other operating systems remains
platform-dependent.

## Ownership and dispatch

Use the existing `ecvKeySequences` owner in CV_db for configurable command
metadata and configuration transactions. Keep its existing modal-shortcut
arbitration separate: mutually exclusive transient tool listeners are not
independent application commands.

A command has a stable, nontranslated ID, category, immutable factory defaults,
current list of bindings, and zero or more live QAction instances. Multiple
instances are allowed only in disjoint contexts, for example the same VTK
operation in different render widgets. Registering another instance must not
recapture a customized key as its default. QAction lifetime is tracked with
QPointer; destroying a view removes its executable instance without losing the
command's saved configuration. Plugin IDs must be namespaced by their owner;
translated action text must never become a new persistent identity. Plugin
registration uses the existing plugin manager's IID namespace. A sole exported
primary action can use the stable `primary` slot; unnamed multi-action exports
are diagnosed and need explicit objectNames from their producer, rather than
unstable translated labels or list indices. Set context and widget associations
before registration, and change registered keys through the registry: direct
QAction key mutations are restored to the authoritative effective binding.
A newly loaded command whose default or saved key conflicts is kept editable but
unbound with a warning, never introduced as an ambiguous active shortcut.

Qt remains the only configurable-command key dispatcher. Do not introduce a
second global event filter, a custom chord state machine, or a per-frame registry
scan. Qt handles multistroke matching, platform modifiers, focus and shortcut
activation. Registry work occurs at registration and configuration time.

### Selection

`cvSelectionToolController` owns command semantics and default metadata.
`MainWindow` owns dispatch to the active render/chart view. Reuse the existing
per-view activation and reaction chain rather than bypassing it with direct
controller calls. A toolbar mirror is a presentation/trigger proxy, not a second
shortcut owner or a second persistent command. Selection gets its own category.

Keep clear-selection and cancel-tool as separate semantic commands. Escape is
cancel-tool, not a misleading label for clear-selection. Reuse the controller's
`handleEscapeKey()` cleanup. Fullscreen and active modal tools retain their
explicit Escape precedence. Add the missing through-block selection binding only
when its executable command is registered. Remove fixed key suffixes from action
names; render actual current bindings in tooltips.

### VTK

Keep `VtkShortcutRegistry` as the operation catalog (stable IDs, defaults and VTK
operation parameters), not as a second settings-backed dispatcher. Each
`QVTKWidgetCustom` owns widget-local QActions associated with the actual render
widget. Their triggered handlers call the existing
`vtkCustomInteractorStyle::handleShortcut()` operation implementation.

Delete the settings-backed string dispatch cache, the hardcoded
`isVtkViewerShortcut()` interception table and the duplicate raw-key invocation
for these managed operations. Preserve unrelated interactor/manipulator key
behavior and account explicitly for non-Qt consumers before deleting a fallback.
Qt then supplies multistroke and Meta matching without a parallel implementation.

### Tabs, fullscreen and dynamic actions

The tab widget owns its navigation QActions and exposes them through the same
registry. Remove the main dialog's manually maintained list of tab reservations.
Fullscreen toggles use their actual command bindings in both entry and exit
paths; Escape remains a documented contextual cancellation gesture, not another
secret configurable F11 binding.

Dynamic views register at creation and unregister on destruction. Non-selection
frame-toolbar commands are widget-with-children instances under a shared `view/`
ID, so changing one command updates every view while focus selects the executable
instance. Chart selection buttons are proxies for the same selection commands,
not extra configurable bindings. Chart cancellation exits the selection mode
without clearing selected data; clear-selection is a separate command.

The dialog reads the live registry, not a startup-only action inventory. It only
rebuilds rows while visible or when opened, not once per command while a hidden
settings dialog observes a new view. Main-menu and
plugin actions must be registered with stable identities by their existing
owners; generated recent-file entries and anonymous presentation actions are
not persistent commands.

## Context and conflicts

Compare every binding of every command, including secondary keys. Two sequences
conflict when either matches the other's prefix, not only on exact equality.
Index candidates by first stroke so unrelated keys are never compared. This is
configuration-time work; key-event processing stays in Qt.

Check potential overlapping contexts, not only currently enabled/visible state:

- Application shortcuts overlap all executable instances.
- Window shortcuts overlap instances in the same owning window.
- Widget shortcuts overlap the same widget.
- Widget-with-children shortcuts additionally overlap descendants.
- Menu popup ownership belongs to the owning window.
- Disjoint widget/window contexts can reuse keys.
- Unassociated QActions have no executable Qt shortcut; setting a parent alone
  must not be mistaken for `widget->addAction()`.

Same-command instances are not cross-command conflicts, but overlapping
instances of the same command must not be registered as independent dispatchers.
Transient modal tools temporarily remove overlapping effective QAction bindings
only in their own context. The configured map and enabled state are preserved;
releasing or destroying the tool restores the latest configuration, including
changes made while the tool owned a key. Overlay tools claim unmodified gestures
at ShortcutOverride and do not accidentally consume modified global commands.
These keys need explicit precedence, rather than being flattened into permanent
global conflicts. Ctrl/Shift mouse modifiers,
text editing, dialog Enter/Escape and local arrow navigation are gestures, not
claims that every such key is a globally configurable command.

## Configuration transaction

Editing, import, selected/all defaults and startup restore use one pipeline:

1. Construct the complete candidate binding map without changing live actions.
2. Validate IDs, format, all sequences, prefix collisions and overlapping scope.
3. Reject the entire candidate with actionable command IDs on failure.
4. Commit one versioned settings value, then update all live instances without
   yielding to the event loop; emit one configuration notification.

The settings commit record uses PortableText arrays and an explicit version.
Empty arrays mean deliberately unbound. Defaults come from registration before
restoration, never from dialog construction. Old single-QKeySequence settings
are read for migration, including the existing VTK group; new writes use stable
IDs. Retain settings for temporarily unloaded plugins. Malformed or conflicting
stored settings must be reported and must not partially replace live bindings.

The UI uses the same validator for conflict highlighting and submission; there
must not be a weaker import/reset/startup path. All bindings are visible, not
only QAction::shortcut(). Configuration failures must preserve both the previous
live configuration and the user's stored data wherever the persistence API can
report failure; do not describe QSettings as a multi-file database transaction.

## Root tests module and regression gates

The repository root [tests module](../../tests/README.md) owns first-party C++
test registration, build closures, capability labels and CI execution. Unit
assertions use GoogleTest. Qt remains the production control/event system under
test; QtTest discovery, assertions and QTEST_MAIN are not required by this
shortcut suite. CTest remains the cross-platform scheduler and result protocol.

With `BUILD_GUI=ON` and `BUILD_UNIT_TESTS=ON`, the application contributes its
shortcut targets through `cloudviewer_add_test`. They link the parent build's
production CV_db/Core libraries and selected Qt. The test executable compiles
application-private dialog/editor/view sources because those are not exported
as a production library. There is no private registry copy or generated
compiler-command parsing. The VTK integration target is conditional on the
configured VTK backend; its real graphics entry is registered on Linux when
Xvfb is available.

```bash
cmake -S . -B build_app -DBUILD_GUI=ON -DBUILD_UNIT_TESTS=ON # plus normal platform flags
# Repository CI selection: build its dependencies, then execute cpp-ci tests.
cmake --build build_app --target check-cpp -j6
# Explicit graphics group; absent capabilities must not become an empty success.
cmake --build build_app --target check-cpp-graphics -j6
# Narrow reproduction while developing this subsystem:
cmake --build build_app --target test_shortcuts test_vtk_shortcuts -j6
ctest --test-dir build_app -R '^test_(shortcuts|vtk_shortcuts|vtk_shortcuts_graphics)$' \
  --no-tests=error --output-on-failure
```

The root module derives both selected build dependencies and CTest labels from
one registration. Its generated `<build>/tests/catalog-<config>.tsv` records
name, owner, targets and labels. `cpp-ci` excludes external model/GPU/graphics/
network/end-to-end requirements; `cpp-graphics` is an explicit separate group.
The root [Run.cmake](../../tests/Run.cmake) is the common script entry point for
platform CI callers. Do not reintroduce shortcut-specific test-name lists in
CI/util scripts or the removed local VTK compiler/linker probe.

The latest reviewed shortcut result is 3/3 registered CTest entries passed on
Linux Qt 5.15.2. Earlier standalone QtTest counts are historical evidence for
the prior fixture, not the current GoogleTest suite size. Windows, macOS and
hosted GitHub Actions were not executed by this documentation update. Tests use
isolated temporary settings, and ordinary control tests use the offscreen Qt
platform; the separate Xvfb entry intentionally exercises xcb/OpenGL delivery.

## Acceptance gates

Use small real-Qt regression probes for the uncovered high-risk behavior, then
build the affected owners and run relevant existing CTest tests. Required cases:

- Exact, prefix and secondary-binding conflicts; two different chords with a
  common nonterminal prefix remain legal.
- Disjoint widgets/windows versus overlapping WidgetWithChildren/Window/
  Application contexts, including menu ownership.
- Batch key swaps succeed; rejected imports/default resets/restores apply nothing.
- A remap activates only the new key; old Qt and VTK fallback paths are silent.
- Multi-stroke, Meta, disabled actions and focused text inputs behave correctly.
- Reopening the dialog does not change defaults; settings roundtrip preserves all
  bindings and deliberately unbound commands.
- New/recreated/closed views consume the current binding without stale pointers
  or duplicate window-level activation.
- Render/chart selection routing, composite subviews, tool cancellation,
  fullscreen precedence and plugin-local scopes retain their behavior.

Linux Qt 5.15 evidence is not Qt 6, macOS or arbitrary keyboard-layout evidence.
Full GUI and cross-platform gaps must be stated explicitly when handing back.
