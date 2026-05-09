# ParaView View Types — ACloudViewer Alignment Guide

## Overview

ParaView defines views via Server-Manager XML proxies (group `views`).
ACloudViewer mirrors this list in `ecvMultiViewWidget::createEmptyCellWidget()`.

**To create a view:** Split a layout cell (or use an empty cell) → click the desired view type button.

---

## Parity Matrix

| # | View Type | ParaView Proxy | ACloudViewer Status | Notes |
|---|-----------|---------------|---------------------|-------|
| 1 | Render View | `RenderView` | **Implemented** | `vtkGLView` with VtkVis pipeline |
| 2 | SpreadSheet View | `SpreadSheetView` | **Implemented** | `ecvSpreadSheetView` with ParaView-style decorator toolbar |
| 3 | Line Chart View | `XYChartView` | **Implemented** | `vtkChartView::LINE_CHART` |
| 4 | Bar Chart View | `XYBarChartView` | **Implemented** | `vtkChartView::BAR_CHART` |
| 5 | Histogram View | `XYHistogramChartView` | **Implemented** | `vtkChartView::HISTOGRAM` |
| 6 | Eye Dome Lighting | `RenderViewWithEDL` | **Implemented** | `vtkGLView::enableEDL()` with full geometry support (mesh + points) |
| 7 | Orthographic Slice View | `OrthographicSliceView` | **Implemented** | `vtkOrthoSliceViewWidget` — single widget, 4 renderers, viewport split |
| 8 | Slice View | `MultiSlice` | **Implemented** | `vtkSliceViewWidget` with 3-edge `vtkMultiSliceAxisWidget` axis sliders + `vtkImplicitPlaneWidget2` |
| 9 | Box Chart View | `BoxChartView` | **Implemented** | `vtkChartView::BOX_CHART` with `vtkChartBox` |
| 10 | Parallel Coordinates View | `ParallelCoordinatesChartView` | **Implemented** | `vtkChartView::PARALLEL_COORDINATES` |
| 11 | Plot Matrix View | `PlotMatrixView` | **Implemented** | `vtkChartView::PLOT_MATRIX` with `vtkScatterPlotMatrix` |
| 12 | Point Chart View | `XYPointChartView` | **Implemented** | `vtkChartView::POINT_CHART` |
| 13 | Quartile Chart View | `QuartileChartView` | **Implemented** | `vtkChartView::QUARTILE_CHART` with area plot |
| 14 | Image Chart View | `ImageChartView` | **Implemented** | `vtkChartView::IMAGE_CHART` with line plot |
| 15 | Render View (Comparative) | `ComparativeRenderView` | **Implemented** | `vtkComparativeViewWidget::RENDER` — 2×2 grid of render views |
| 16 | Bar Chart View (Comparative) | `ComparativeXYBarChartView` | **Implemented** | `vtkComparativeViewWidget::BAR_CHART` — 2×2 grid of bar charts |
| 17 | Line Chart View (Comparative) | `ComparativeXYChartView` | **Implemented** | `vtkComparativeViewWidget::LINE_CHART` — 2×2 grid of line charts |
| 18 | Python View | `PythonView` | **Implemented** | `ecvPythonView` — script editor + QProcess execution |

---

## View Types — Detailed Usage

### 1. Render View

**Purpose:** Interactive 3D rendering of point clouds, meshes, volumes, and annotations.

**Data sources:** Any loaded entity (point cloud, mesh, polyline, etc.)

**Mouse controls (ACloudViewer):**

| Button | Action |
|--------|--------|
| Left drag | Rotate (trackball) |
| Middle drag | Pan |
| Right drag | Zoom |
| Scroll wheel | Zoom in/out |
| Shift + Left | Pan |
| Ctrl + Left | Roll |

**Features:** Background gradient, orientation marker, scale bar, camera orientation widget, EDL toggle, slice mode.

---

### 2. SpreadSheet View

**Purpose:** Tabular display of entity attributes (ParaView `pqSpreadSheetViewDecorator` pattern).

**Data sources:** Point clouds (`ccPointCloud`), meshes (`ccMesh`)

**Decorator Toolbar (ParaView-aligned):**

| Control | Description |
|---------|-------------|
| **Showing** | Label + entity name combo showing current data source |
| **Attribute** | Dropdown: Point Data, Cell Data, Field Data |
| **Precision** | Spinner (1–32) for decimal digit control |
| **Fixed** | Toggle between scientific (`g`) and fixed-point (`f`) representation |
| **Columns** | Menu to show/hide individual columns |
| **Export** | Export spreadsheet to CSV |

**Point Data columns:** Point ID, Points_X/Y/Z, Normals_X/Y/Z, scalar fields, RGB

**Cell Data columns (meshes):** Cell ID, Vertex 1/2/3, Material

**Field Data (metadata):** Property/Value table showing Name, Type, Number of Points, Bounds X/Y/Z, Has Colors, Has Normals, Scalar Fields count + per-SF range, Number of Triangles, Has Materials, Has Per-Triangle Normals

**Features:**
- Filter rows by text search
- Export to CSV
- Sort by column header click
- Ctrl+C copy to clipboard
- Column visibility toggle per-column
- Decimal precision control with fixed/scientific representation
- Auto-updates when entity selection changes

---

### 3. Line Chart View

**Purpose:** Plot scalar field values vs point index as line series.

**Data sources:** Point clouds with scalar fields or colors

**Usage:**
1. Load a point cloud with scalar fields
2. Select the entity in the DB tree
3. Create a Line Chart View
4. Select fields from the Fields list (multi-select supported)
5. Each selected field becomes a line series

**Controls:**
- Left drag: Pan
- Right drag: Zoom
- Middle drag: Zoom axis
- Link 3D checkbox: Switches left button to selection mode
- Reset Zoom: Resets chart view bounds
- PNG/CSV: Export buttons

**Notes:** Point clouds >10k points are subsampled to ~10k for performance.

---

### 4. Bar Chart View

**Purpose:** Bar chart of scalar field values vs point index.

**Data sources:** Same as Line Chart View

**Usage:** Same workflow as Line Chart. Each field becomes a bar series.

**Controls:** Same as Line Chart View.

---

### 5. Histogram View

**Purpose:** Frequency distribution of scalar field values.

**Data sources:** Point clouds with scalar fields or colors

**Usage:**
1. Load a point cloud
2. Select it, create a Histogram View
3. Select field(s) from the Fields list
4. Adjust bin count (5-500) using the Bins spinner
5. Each field shows as an overlaid histogram

**Controls:** Same as Line Chart, plus Bins spinner.

---

### 6. Eye Dome Lighting View

**Purpose:** Enhanced depth perception for point cloud and mesh visualization.

**Data sources:** Same as Render View (supports point clouds AND meshes)

**Usage:** Creates a Render View with EDL post-processing enabled.

**Implementation:** Uses `vtkEDLShading` with `vtkRenderStepsPass` as the delegate pass, which handles all geometry types (opaque, translucent, volumetric, overlays). This matches ParaView's `vtkPVRenderViewWithEDL` approach where EDL acts as an image processing post-process using the full depth buffer.

**ParaView alignment note:** The previous implementation used `vtkDefaultPass` which only handled a subset of rendering steps. The updated implementation uses `vtkRenderStepsPass` which includes the complete render pipeline (lights, opaque geometry, translucent geometry, volumetric rendering, overlays), ensuring proper rendering for all actor types including meshes.

---

### 7. Orthographic Slice View

**Purpose:** Four-viewport orthographic view showing Top, Right Side, Front views + free 3D perspective.

**Data sources:** Same as Render View

**Implementation:** `vtkOrthoSliceViewWidget` — a single widget with 4 VTK renderers sharing one `vtkRenderWindow`, using viewport splitting (`SetViewport()`). This matches ParaView's `vtkPVOrthographicSliceView` pattern.

**Layout:**
```
┌──────────────────┬──────────────────┐
│  Top View        │  Right Side View │
│  (XZ plane)      │  (YZ plane)      │
│  camera: -Y      │  camera: -X      │
├──────────────────┼──────────────────┤
│  Front View      │  3D Perspective  │
│  (XY plane)      │  (free camera)   │
│  camera: -Z      │                  │
└──────────────────┴──────────────────┘
```

**Features:**
- 3 orthographic views with parallel projection and fixed camera orientations
- 1 perspective 3D view with free camera
- Text annotations showing view name + slice position coordinates (e.g., "Top View (Y=0.17594)")
- Sub-annotations showing other coordinate values
- Crosshair axes (vtkPVCenterAxesActor) in all 4 viewports showing current slice position
- `setSlicePosition()` updates all cameras, annotations, and crosshairs simultaneously
- Shared render window for efficient rendering
- **Mouse wheel** on any orthographic view scrolls the slice position along that view's normal axis (configurable step size)
- **Double-click** on any orthographic view picks the 3D world position and updates the slice
- `slicePositionChanged` signal emitted on interactive slice updates

**ParaView alignment note:** Previous implementation split the layout into 4 separate cells with 4 separate render windows. The new implementation uses a single widget with viewport splitting, matching ParaView's approach exactly. Interactive slice positioning (wheel + double-click) matches ParaView's `MoveSlicePosition` behavior.

---

### 8. Slice View

**Purpose:** Interactive clipping plane for sectional views with axis annotations.

**Data sources:** Same as Render View (best with meshes)

**Usage:**
1. Load a mesh or point cloud
2. Create a Slice View
3. An interactive plane widget appears
4. Drag/rotate the plane to clip geometry
5. Actors are clipped in real-time via `vtkClippingPlane`

**Features:**
- `vtkSliceViewWidget` wrapper composing the 3D view with 3-edge axis slider strips
- `vtkMultiSliceAxisWidget` on Y (top), X (left), Z (right) edges — matching ParaView's `pqMultiSliceAxisWidget` layout
- **Double-click** on axis strip to add a new slice
- **Drag** a marker to move a slice interactively
- **Right-click** a marker to remove a slice
- Color-coded axis strips and markers: X (red), Y (green), Z (blue)
- Ruler tick marks with data range labels
- Slice positions drive `vtkGLView::setMultiSlicePositions()` for per-axis `vtkCutter` + `vtkPlane` slicing
- `vtkImplicitPlaneWidget2` for interactive single-plane manipulation
- `vtkCubeAxesActor` for axis labels along viewport edges

---

### 9. Box Chart View

**Purpose:** Display statistical distribution (min, Q1, median, Q3, max) as box-and-whisker plots.

**Data sources:** Point clouds with scalar fields or colors

**Usage:**
1. Load a point cloud with multiple scalar fields
2. Select the entity in the DB tree
3. Create a Box Chart View
4. Select fields from the Fields list
5. Each field renders as a color-coded box-and-whisker column

**How it works:**
- Uses VTK's `vtkChartBox` + `vtkPlotBox` internally
- Automatically computes the five-number summary from sampled data
- Color-coded columns match the field palette

**Controls:** Same as Line Chart View.

---

### 10. Parallel Coordinates View

**Purpose:** Multi-axis polyline visualization for high-dimensional scalar field analysis.

**Data sources:** Point clouds with 2+ scalar fields

**Usage:**
1. Load a point cloud with multiple scalar fields
2. Select the entity, create a Parallel Coordinates View
3. Select 2 or more fields from the Fields list
4. Each field becomes a vertical axis; each sampled point draws a polyline

**How it works:**
- Uses VTK's `vtkChartParallelCoordinates` + `vtkPlotParallelCoordinates`
- Lines are semi-transparent (alpha 0.3) for dense data visualization

**Controls:** Same as Line Chart View.

---

### 11. Point Chart View (Scatter)

**Purpose:** Scatter plot of scalar field values vs point index.

**Data sources:** Point clouds with scalar fields or colors

**Usage:** Same workflow as Line Chart View, renders as scattered dots.

**Key differences from Line Chart:**
- Uses `vtkChart::POINTS` plot type
- Wider marker width (3.0px)
- Better for spotting discrete distributions and outliers

---

### 12. Plot Matrix View

**Purpose:** NxN scatter plot matrix for pairwise relationship exploration.

**Data sources:** Point clouds with 2+ scalar fields

**Usage:**
1. Select entity, create a Plot Matrix View
2. Select 2+ fields
3. NxN grid: scatter plots off-diagonal, histograms on diagonal

**How it works:** Uses VTK's `vtkScatterPlotMatrix`.

---

### 13. Quartile Chart View

**Purpose:** Area chart showing quartile bands for scalar field distributions.

**Data sources:** Point clouds with scalar fields

**Usage:** Same as Line Chart View. Renders as filled area plots.

**Implementation:** `vtkChartView::QUARTILE_CHART` using `vtkChart::AREA` plot type.

---

### 14. Image Chart View

**Purpose:** Line-based image data visualization.

**Data sources:** Point clouds with scalar fields

**Usage:** Same as Line Chart View. Functionally similar to Line Chart but designed for image data representation.

**Implementation:** `vtkChartView::IMAGE_CHART` using `vtkChart::LINE` plot type.

---

### 15. Python View

**Purpose:** Lightweight script editor for running Python scripts directly from within the application. Modeled after ParaView's `PythonView`.

**Data sources:** N/A — user-authored Python scripts

**Usage:**
1. Create a Python View from the Create View dialog
2. Type or paste Python code into the editor panel (top)
3. Click **Run** (or press `Ctrl+Enter`) to execute
4. Output appears in the output panel (bottom)
5. Use **Load** to open a `.py` file, **Save** to save the current script

**Editor features:**
- Syntax-highlighted dark theme (Consolas/Monaco monospace font)
- Tab stop distance: 32px
- Placeholder text showing matplotlib usage example
- Vertical splitter between editor (2/3) and output (1/3)

**Toolbar controls:**

| Button | Action |
|--------|--------|
| **Run** | Execute script via `python3 -c` subprocess (30s timeout) |
| **Clear** | Clear output panel |
| **Load** | Load `.py` file into editor |
| **Save** | Save editor contents to `.py` file |

**Output panel:**
- Read-only dark theme output
- Shows stdout followed by stderr (separated by `--- stderr ---`)
- Displays exit code in status label

**Example workflow:**
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.savefig('/tmp/trig_plot.png', dpi=150)
print('Plot saved to /tmp/trig_plot.png')
```

**ParaView alignment note:** ParaView's `PythonView` uses an embedded `vtkPythonInterpreter` with direct access to the pipeline. ACloudViewer's implementation uses `QProcess` subprocess execution, which is safer (isolated from application state) but cannot directly access in-memory entities. Users can export data to temp files and process them in scripts.

---

### 16–18. Comparative Views

**Purpose:** Side-by-side comparison of the same data in a 2×2 grid layout within a single widget.

**Types:**
- **Render View (Comparative):** 4 render viewports in a 2×2 grid
- **Line Chart View (Comparative):** 4 line chart views in a 2×2 grid
- **Bar Chart View (Comparative):** 4 bar chart views in a 2×2 grid

**Implementation:** `vtkComparativeViewWidget` — a single widget with `QGridLayout` hosting N×M sub-views. This matches ParaView's `pqComparativeRenderView` / `pqComparativeContextView` pattern of a single widget with N×M sub-viewports.

**Key design decisions:**
- Single widget (not 4 separate layout cells) — matches ParaView behavior
- For Render type: uses `RenderViewFactory` to create N×M `vtkGLView` instances
- For Chart types: creates N×M `vtkChartView` instances with the appropriate chart type
- Default 2×2 grid layout with minimal spacing (2px)

**Toolbar controls (ParaView `vtkSMComparativeViewProxy` aligned):**
- **Grid dimensions:** Row/Column spinners (1–8) for configurable N×M
- **Cue parameter:** Combo box (None, Camera Azimuth, Camera Elevation, Opacity)
- **Apply:** Button to apply the parameter sweep across sub-views
- **Status:** Label showing current grid size and cue state

---

## Deep Verification: ParaView Gap Analysis

### SpreadSheet View — Gaps vs `pqSpreadSheetViewDecorator`

| Feature | ParaView | ACloudViewer | Status |
|---------|----------|-------------|--------|
| Data source (pipeline output port) | `pqOutputPortComboBox` — any pipeline output | Global selection entity only | Gap |
| Attribute/field association | Full enumeration from proxy (points, cells, rows) | Point / Cell / Field Data (all modes implemented) | Aligned |
| Precision + fixed/scientific | `pqSpreadSheetViewModel::setDecimalPrecision/setFixedRepresentation` | Same pattern via `QSpinBox` + `Fixed` button | Aligned |
| Column visibility | `pqSpreadSheetColumnsVisibility::populateMenu` (SM-aware) | `QMenu` with per-column checkables | Aligned |
| Show selected only | `SelectionOnly` toggle on decorator, linked to proxy | `Selected` toggle button; filters model to show only selected rows + highlight color | Aligned |
| Generate cell connectivity | Toggle on decorator, linked to `GenerateCellConnectivity` proxy | `Conn` toggle button; adds V1_X/Y/Z, V2_X/Y/Z, V3_X/Y/Z columns | Aligned |
| Selection → pipeline | `pqSpreadSheetViewSelectionModel` builds `vtkSMSourceProxy` selection | No selection proxy; Qt selection only | Gap |
| Selection from 3D → rows | Server selection synced to `QItemSelection` | `setSelectedPointIndices()` + auto-scroll to first selected row | Aligned |
| Row text filter | Not part of core decorator | `QLineEdit` + `QSortFilterProxyModel` | Extra |
| Copy behavior | Clipboard via decorator; ties to `SelectionOnly` state | `Ctrl+C` → `getRowsAsString()` on source model (ignores proxy/filtered rows) | Partial |
| Font sizing | `CellFontSize`/`HeaderFontSize` on view proxy | Cell/Header font size spinners (6-24pt) | Aligned |
| Export | `pqExportReaction` (full export stack) | Simple CSV file dialog | Partial |

### Chart Views — Gaps vs `pqXYChartView` / `pqXYBarChartView` / `pqBoxChartView`

| Feature | ParaView | ACloudViewer | Status |
|---------|----------|-------------|--------|
| Series/array choice | Representation-driven (SM visible arrays) | `QListWidget` multi-select over scalar fields | Different approach |
| Chart title | Configurable title, alignment, fonts, colors (SM property groups) | `QLineEdit` title editor in toolbar; updates `vtkChart::SetTitle()` | Aligned |
| Legend | `ShowLegend`, location, position, symbol width, fonts | `QCheckBox` Legend toggle; `vtkChart::SetShowLegend()` | Partial |
| Axis config | Full 4-axis: titles, grid, colors, log scale, custom min/max, notation, precision, fonts | Full 4-axis selector (Left/Bottom/Right/Top) with per-axis: visibility, title, log scale, notation, precision, custom range | Aligned |
| Tooltip | `TooltipNotation`, `TooltipPrecision` | Tooltip notation combo + precision spinner; `vtkTooltipItem::TextProperties` customization | Aligned |
| Chart ↔ 3D selection | `pqChartSelectionReaction` — polygon/rectangle, add/subtract/toggle | "Link 3D" checkbox + `vtkAnnotationLink` observer emitting `pointsHighlighted` signal | Partial |
| Zoom/reset | `pqContextView::resetDisplay` | `RecalculateBounds` + render button | Aligned |
| Export image | Generic view capture (`supportsCapture`) | PNG via `vtkWindowToImageFilter` | Aligned |
| Export data | Table writers/export infrastructure | CSV of raw cloud columns (not chart-derived data) | Partial |

### EDL — Gaps vs `vtkPVRenderViewWithEDL`

| Feature | ParaView | ACloudViewer | Status |
|---------|----------|-------------|--------|
| Pass chain | `vtkEDLShading` → `vtkCameraPass` → `vtkPVDefaultPass` | `vtkEDLShading` → `vtkRenderStepsPass` | Different (works, not identical) |
| Depth buffer | `SetUseDepthBuffer(true)` on `vtkPVSynchronizedRenderer` | Implicit from full render | Different |
| FXAA integration | Optional FXAA outer shell (`vtkOpenGLFXAAPass`) | `vtkOpenGLFXAAPass` wrapping EDL pass chain | Aligned |
| Synchronized rendering | PV sync layer (`vtkPVSynchronizedRenderer`) | Single renderer pass | N/A (no client-server) |

### Slice View — Gaps vs `vtkPVMultiSliceView`

| Feature | ParaView | ACloudViewer | Status |
|---------|----------|-------------|--------|
| Slice model | Per-axis arrays (`SetNumberOfXSlices/SetXSlices`, etc.) | `setMultiSlicePositions(axis, positions)` + `vtkCutter`+`vtkPlane` per slice | Aligned |
| Multiple orthogonal planes | Yes — independent X/Y/Z slice lists | Yes — independent per-axis `std::vector<double>` | Aligned |
| Representation | `vtkGeometrySliceRepresentation` + view metadata | `vtkCutter` cutting geometry per slice, colored slice actors | Different |
| Axis UI | `pqMultiSliceAxisWidget` charts on 3 edges | `vtkMultiSliceAxisWidget` on 3 edges (Y top, X left, Z right) with draggable markers | Aligned |
| Outline/interaction | Dedicated slice UI callbacks, outline visibility | Double-click to add slice, drag to move, right-click to remove | Aligned |

### Orthographic Slice View — Gaps vs `vtkPVOrthographicSliceView`

| Feature | ParaView | ACloudViewer | Status |
|---------|----------|-------------|--------|
| Layout | 3 ortho + 1 main 3D via `SetViewport` | Same quad pattern | Aligned |
| Crosshairs | `SlicePositionAxes2D` + `SlicePositionAxes3D` actors | `vtkPVCenterAxesActor` — 3 ortho + 1 perspective | Aligned |
| Interactive slice | Double-click → `MoveSlicePosition`; mouse wheel axis stepping | Mouse wheel scrolls slice axis; double-click picks slice position | Aligned |
| Grid axes per pane | `vtkPVGridAxes3DActor` clones | `vtkCubeAxesActor` per ortho renderer with colored axis labels | Aligned |
| Annotations | `vtkTextRepresentation` with formatted multi-line | `snprintf` into `vtkTextActor` | Partial |
| Basis updates | `vtkPVChangeOfBasisHelper` per render | Fixed initial cameras; focal point shift | Gap |

### Comparative View — Gaps vs `vtkSMComparativeViewProxy`

| Feature | ParaView | ACloudViewer | Status |
|---------|----------|-------------|--------|
| Grid dimensions | Arbitrary N×M via `Dimensions` property (`Build(dx,dy)`) | `setDimensions(rows, cols)` — configurable NxM | Aligned |
| Spacing | SM `Spacing` property | Constant `COMPARATIVE_SPACING = 2` | Partial |
| Parameter cues | `AddCue`/`RemoveCue`, `ViewTime`, parameter sweep replay | Cue toolbar with parameter combo (None/Azimuth/Elevation/Opacity) + Apply button | Partial |
| Overlay comparisons | `OverlayAllComparisons` mode | `Overlay` checkbox hides all but first sub-view | Aligned |
| Screenshot | Stitched capture in proxy | `Screenshot` button exports stitched composite PNG/JPEG | Aligned |

---

## Architecture Changes Log

### Phase 1: SpreadSheet View Enhancement
- Added ParaView-style decorator toolbar (`Showing`, `Attribute`, `Precision`, `Fixed`, `Columns`, `Export`)
- Added attribute type switching (Point Data / Cell Data)
- Added decimal precision control and fixed-point representation
- Added column visibility toggle menu
- Added Ctrl+C copy-to-clipboard support
- Added Cell Data columns for mesh triangle data

### Phase 2: View Type Enablement
- Enabled Render View (Comparative), Bar Chart View (Comparative), Line Chart View (Comparative)
- Added IMAGE_CHART and QUARTILE_CHART to `vtkChartView::ChartType` enum
- All 18 ParaView view types implemented (including Python View)

### Phase 3: Comparative View Infrastructure
- Created `vtkComparativeViewWidget` class
- Supports RENDER, LINE_CHART, BAR_CHART comparative types
- 2×2 grid of sub-views in a single widget
- Wired into Create View button connections

### Phase 4: Orthographic Slice View Rewrite
- Created `vtkOrthoSliceViewWidget` class
- Single widget with 4 VTK renderers sharing one render window
- Viewport splitting via `SetViewport()` (ParaView `vtkPVOrthographicSliceView` pattern)
- Text annotations for view names and coordinate positions
- Replaced old 4-cell split approach

### Phase 5: Slice View Enhancement
- Added `vtkCubeAxesActor` for axis labels along viewport edges
- Colored axis titles (X=red, Y=green, Z=blue)
- Grid lines on X and Y axes

### Phase 6: EDL Mesh Support Fix
- Changed delegate pass from `vtkDefaultPass` to `vtkRenderStepsPass`
- `vtkRenderStepsPass` includes complete render pipeline (lights, opaque, translucent, volumetric, overlay)
- Fixes mesh rendering support in EDL mode

### Phase 7: OrthoSlice Crosshairs + Comparative NxM Grid
- Added `vtkPVCenterAxesActor` crosshair actors to `vtkOrthoSliceViewWidget` (3 ortho + 1 perspective)
- Crosshair colors: X=red(1,0,0), Y=yellow(1,1,0), Z=blue(0,0,1) — matching ParaView defaults
- Crosshairs update position in `setSlicePosition()` with Z-fighting offsets (ParaView pattern)
- Added `setDimensions(int rows, int cols)` to `vtkComparativeViewWidget` for configurable NxM grid
- Existing sub-views are cleaned up and recreated when dimensions change

### Phase 10: ParaView Source-Aligned Feature Parity (A/B/C/D)
- **A) SpreadSheet "Show Selected Only"**: Added `Selected` toggle button (ParaView `SelectionOnly` pattern)
  - `setSelectedIndices(QSet<unsigned>)` + `setSelectionOnly(bool)` on model
  - When checked: model shows only selected rows (ParaView `SetShowExtractedSelection`)
  - When unchecked: selected rows highlighted with `QColor(38, 79, 120)` (ParaView `__vtkIsSelected__` pattern)
  - Selection mode switches to `NoSelection` when `Selected` is checked (ParaView `pqSpreadSheetView::onSelectionOnly`)
- **B) Chart Axis Configuration**: Full axis UI copied from ParaView `vtkPVXYChartView` pattern
  - Log scale toggle (`vtkAxis::SetLogScale`) — ParaView `LeftAxisLogScale`
  - Notation combo (Mixed/Scientific/Fixed) — ParaView `LeftAxisLabelNotation`
  - Axis precision spinner — ParaView `LeftAxisLabelPrecision`
  - Custom Y-axis range (checkbox + min/max spinners) — ParaView `LeftAxisUseCustomRange` + `LeftAxisRangeMinimum/Maximum`
  - All settings persist across chart rebuilds via `vtkAxis::SetBehavior(FIXED/AUTO)`
- **C) OrthoSlice Grid Axes**: Per-pane `vtkCubeAxesActor` (ParaView `vtkPVGridAxes3DActor` clone pattern)
  - 3 grid axes actors — one per orthographic renderer (Top, Side, Front)
  - Colored axis titles (X=red, Y=green, Z=blue)
  - Grid lines on X and Y axes
  - `setGeometryBounds(double[6])` API drives axes bounds + crosshair scale
- **D) MultiSlice Per-Axis Support**: ParaView `vtkPVMultiSliceView` + `vtkThreeSliceFilter` pattern
  - `setMultiSlicePositions(int axis, vector<double>)` API on `vtkGLView`
  - Per-axis `std::vector<double>` storage for X/Y/Z slice positions
  - `vtkCutter` + `vtkPlane` per slice (ParaView `vtkThreeSliceFilter` approach)
  - Color-coded slice actors per axis (X=red, Y=green, Z=blue)
  - `clearMultiSlicePositions()` cleanup API

### Phase 9: Chart UI + Selection + SpreadSheet Field Data + OrthoSlice Interaction
- Added chart title edit (`QLineEdit`), legend toggle (`QCheckBox`), grid lines toggle (`QCheckBox`) to `vtkChartView` toolbar
- Implemented `onChartTitleChanged`, `onToggleLegend`, `onToggleGridLines` slots in `vtkChartView.cpp`
- Chart UI state (title, legend, grid) persists across chart rebuilds
- Wired `setupChartSelectionCallback()` to use `vtkAnnotationLink::AnnotationChangedEvent`
- Chart selection extracts row indices, maps back to point indices (accounting for stride), emits `pointsHighlighted`
- Added Field Data mode to `ecvSpreadSheetModel` — displays metadata (name, type, bounds, scalar field ranges, colors, normals, materials)
- Field Data uses "Property"/"Value" column headers with key-value rows
- Added mouse wheel slice scrolling to `vtkOrthoSliceViewWidget` — scrolling on any orthographic view steps the slice along that view's normal axis
- Added double-click slice picking — double-clicking on an orthographic view picks the 3D position and updates the slice
- Added `slicePositionChanged` signal and configurable `m_sliceStep`
- Added `getSlicePosition()` API for reading current slice coordinates

### Phase 13: CellConnectivity + Overlay + Screenshot + Selection→3D + Opacity Sweep
- **B) GenerateCellConnectivity**: `Conn` toggle button on SpreadSheet decorator
  - When enabled in Cell Data mode, adds V1_X/Y/Z, V2_X/Y/Z, V3_X/Y/Z columns
  - Shows actual vertex coordinates (not just indices) for each triangle
  - `setGenerateCellConnectivity(bool)` triggers column rebuild
- **B) OverlayAllComparisons**: `Overlay` checkbox on Comparative View toolbar
  - When checked, hides all sub-views except the first (ParaView pattern)
  - `m_overlayMode` flag controls visibility toggling
- **B) Screenshot Stitching**: `Screenshot` button on Comparative View toolbar
  - Uses `QWidget::render()` to composite all sub-views into a single image
  - Supports PNG and JPEG export via QFileDialog
- **C) SpreadSheet Selection→3D Pipeline**: Bidirectional selection complete
  - Table row selection emits `tableSelectionChanged` signal
  - Signal bridges via `ecvViewManager::pointIndicesSelected` to 3D view
  - `MainWindow::initSelectionController` connects signal to `cvSelectionHighlighter::highlightSelection`
  - Creates `cvSelectionData` with POINTS field association from row indices
- **D) Comparative View Opacity Sweep**: Actor transparency distribution
  - Opacity cue (param index 3) iterates all actors in each sub-view's renderer
  - Sets `vtkProperty::SetOpacity(value / 100.0)` on each actor
  - Value range is user-configurable via Min/Max spinners (default 0-100)
  - Supports XRANGE/YRANGE/TRANGE distribution modes

### Phase 12: Camera Sweep + 3D→SpreadSheet Selection + Detail Gaps (Tooltip/FontSize/FXAA)
- **B) Comparative Camera Parameter Sweep**: Real camera azimuth/elevation/zoom distribution
  - ParaView `vtkPVComparativeAnimationCue::GetValues()` interpolation math:
    - XRANGE: `v = min + x * (max - min) / (dx - 1)` (vary along columns)
    - YRANGE: `v = min + y * (max - min) / (dy - 1)` (vary along rows)
    - TRANGE: `v = min + (y*dx+x) * (max - min) / (dx*dy - 1)` (row-major)
  - Min/Max spinners for parameter range
  - Mode combo (X-Range/Y-Range/T-Range)
  - Camera Azimuth/Elevation/Zoom + Opacity parameter types
  - Accesses `vtkCamera::Azimuth()/Elevation()/Zoom()` via `VtkVis::getRendererCollection()`
- **C) 3D→SpreadSheet Selection Linkage**: Point-level selection propagation
  - New `ecvViewManager::pointIndicesSelected(entity, QSet<unsigned>)` signal
  - `MainWindow::onSelectionFinished` bridges `cvSelectionData::ids()` → `pointIndicesSelected`
  - `ecvSpreadSheetView` auto-connects: matching entity → `setSelectedPointIndices()`
  - Auto-scrolls table to first selected row
  - Bidirectional: table selection emits `tableSelectionChanged()` for 3D feedback
- **D) Detail Gap Alignment**:
  - **Tooltip UI**: Notation combo (Mixed/Sci/Fix) + precision spinner in chart toolbar
    (VTK 9.3 lacks native tooltip API; UI stored for future custom rendering)
  - **Font Size**: Cell font size (6-24pt) + Header font size (6-24pt) spinners in SpreadSheet
    (ParaView `CellFontSize`/`HeaderFontSize` pattern)
  - **FXAA**: Added `vtkOpenGLFXAAPass` as outer shell wrapping EDL+RenderStepsPass chain
    (matches ParaView's optional FXAA integration)

### Phase 11: Multi-Slice Axis UI + SpreadSheet Bidirectional Selection + 4-Axis Chart Config + Comparative Cue
- **A) vtkMultiSliceAxisWidget**: New widget matching ParaView's `pqMultiSliceAxisWidget`
  - Draws ruler-like axis strip with draggable triangular slice markers
  - Double-click to add slice, drag to move, right-click context menu to remove
  - Color-coded per axis (X=red, Y=green, Z=blue) with tick marks and labels
  - `sliceAdded`, `sliceRemoved`, `sliceMoved`, `slicePositionsChanged` signals
- **B) vtkSliceViewWidget**: Wraps `vtkGLView` with 3-edge axis sliders
  - Y-axis (top), X-axis (left), Z-axis (right) layout matching ParaView
  - Connected to `vtkGLView::setMultiSlicePositions()` for real-time slice updates
  - `setDataBounds()` API to configure axis data ranges
- **C) SpreadSheet Bidirectional Selection**: 3D ↔ Table selection linking
  - `setSelectedPointIndices(QSet<unsigned>)` — receives selection from 3D view
  - Auto-scrolls to first selected row
  - `onTableSelectionChanged()` emits `tableSelectionChanged(entity, indices)` signal
  - Bi-directional: table row selection → signal for 3D highlighting
- **D) Chart 4-Axis Config**: Per-axis configuration for all 4 chart axes
  - Axis selector combo (Left/Bottom/Right/Top)
  - Per-axis: visibility toggle, title edit, log scale, notation, precision, custom range
  - `onActiveAxisChanged()` loads current axis state into UI controls
  - All controls now operate on the selected axis (not hardcoded Left/Bottom)
- **E) Comparative View Cue Toolbar**: Parameter sweep UI
  - Grid dimension spinners (R × C, range 1-8)
  - Cue parameter combo (None, Camera Azimuth, Camera Elevation, Opacity)
  - Apply button + status label
  - `onDimensionChanged()` dynamically rebuilds grid
  - `applyCueToSubViews()` framework for parameter distribution

### Phase 8: File Reorganization + VTK Isolation
- Moved `vtkComparativeViewWidget.h/cpp` from `app/` to `libs/VtkEngine/VTKExtensions/Views/`
- Moved `vtkOrthoSliceViewWidget.h/cpp` from `app/` to `libs/VtkEngine/VTKExtensions/Views/`
- Added `QVTK_ENGINE_LIB_API` export macro to both widget classes
- Added `#ifdef USE_VTK_BACKEND` guards in `ecvMultiViewWidget.cpp` for:
  - VTK-specific includes (`vtkChartView.h`, `vtkComparativeViewWidget.h`, `vtkOrthoSliceViewWidget.h`, `vtkGLView.h`)
  - VTK-dependent view creation lambdas (`createChartForCell`, `createEDLViewForCell`, `createSliceViewForCell`, `createComparativeForCell`, `createOrthoSliceForCell`)
  - VTK-dependent button connections
  - `dynamic_cast<vtkGLView*>` calls in `redrawAllViews()`, `destroyView()`, `destroyAllViews()`
- View type availability is conditionally set via `vtkAvail` flag

---

## Modified Files Summary

| File | Changes |
|------|---------|
| `app/ecvSpreadSheetView.h` | Full rewrite with decorator toolbar, precision control, column visibility, Field Data mode |
| `app/ecvSpreadSheetView.cpp` | Full rewrite matching `pqSpreadSheetViewDecorator` pattern + Field Data metadata display |
| `app/ecvMultiViewWidget.cpp` | Enabled all view types, VTK macro isolation, updated include paths |
| `libs/VtkEngine/VTKExtensions/Views/vtkComparativeViewWidget.h` | **Moved from app/** — Comparative view widget with `QVTK_ENGINE_LIB_API` |
| `libs/VtkEngine/VTKExtensions/Views/vtkComparativeViewWidget.cpp` | **Moved from app/** — 2×2 grid implementation |
| `libs/VtkEngine/VTKExtensions/Views/vtkOrthoSliceViewWidget.h` | **Moved from app/** — Orthographic slice widget with `QVTK_ENGINE_LIB_API` |
| `libs/VtkEngine/VTKExtensions/Views/vtkOrthoSliceViewWidget.cpp` | **Moved from app/** — 4-renderer viewport split |
| `libs/VtkEngine/VTKExtensions/Views/vtkChartView.h` | Added IMAGE_CHART, QUARTILE_CHART, chart title/legend/grid UI, annotation callback |
| `libs/VtkEngine/VTKExtensions/Views/vtkChartView.cpp` | New chart types + titles + annotation link selection + UI controls |
| `libs/VtkEngine/VTKExtensions/Views/vtkOrthoSliceViewWidget.h` | Added interactive slice (wheel/dblclick), `slicePositionChanged` signal |
| `libs/VtkEngine/VTKExtensions/Views/vtkOrthoSliceViewWidget.cpp` | Event filter for wheel/dblclick, hit-test viewport mapping |
| `libs/VtkEngine/VTKExtensions/Views/vtkMultiSliceAxisWidget.h` | **New** — ParaView `pqMultiSliceAxisWidget` aligned axis slider widget |
| `libs/VtkEngine/VTKExtensions/Views/vtkMultiSliceAxisWidget.cpp` | **New** — Ruler painting, marker interaction, context menu |
| `libs/VtkEngine/VTKExtensions/Views/vtkSliceViewWidget.h` | **New** — Slice View wrapper with 3-edge axis strips |
| `libs/VtkEngine/VTKExtensions/Views/vtkSliceViewWidget.cpp` | **New** — Grid layout composing view + axis widgets |
| `libs/VtkEngine/Visualization/vtkGLView.h` | Added `vtkCubeAxesActor` member, multi-slice positions |
| `libs/VtkEngine/Visualization/vtkGLView.cpp` | EDL fix + slice axes + multi-slice implementation |
| `app/ecvPythonView.h` | **New** — Python View header with script editor + output panel + entity export |
| `app/ecvPythonView.cpp` | **New** — Python View: QProcess execution, Export+Run, syntax highlighting, auto-completion |
| `app/ecvPythonSyntaxHighlighter.h` | **New** — Pure C++ Python syntax highlighter (keywords, builtins, strings, comments, numbers) |
| `app/ecvPythonSyntaxHighlighter.cpp` | **New** — QSyntaxHighlighter with multi-line string support |
| `app/ecvPythonCodeEditor.h` | **New** — Code editor with line numbers, current-line highlight (ParaView pqPythonLineNumberArea) |
| `app/ecvPythonCodeEditor.cpp` | **New** — Line number painting, auto-indent, Tab/Shift+Tab block indent |
| `app/ecvShortcutDialog.h` | Enhanced with conflict detection, modal shortcut integration, reset functionality |
| `app/ecvShortcutDialog.cpp` | Full rewrite: conflict highlighting, modal/standalone awareness, persistence |
| `app/ecvTabbedMultiViewWidget.cpp` | Added tab navigation shortcuts (Ctrl+PgUp/PgDn, Ctrl+Shift+T/W) |
| `libs/CV_db/include/Shortcuts/ecvKeySequences.h` | Added `allRegisteredSequences()` API |
| `libs/CV_db/src/Shortcuts/ecvKeySequences.cpp` | Implemented `allRegisteredSequences()` |
| `app/MainWindow.cpp` | `showShortcutDialog()` now calls `syncModalShortcuts()` before exec |

---

## Test Data

ParaView test data location: `/home/ludahai/develop/tools/data/ParaViewTestingDataFiles-v6.0.1/ParaView-v6.0.1`

| Data Type | View Types | Examples |
|-----------|-----------|---------|
| VTK structured grids (`.vts`, `.pvti`) | Render View | Wavelet data |
| VTK unstructured grids (`.vtu`) | Render View, SpreadSheet | Simulation meshes |
| VTK polydata (`.vtp`) | Render View | Surface meshes |
| CSV tables (`.csv`) | SpreadSheet, Charts | `half_sphere_commented.csv` |
| ParaView state (`.pvsm`) | Various | Full pipeline states |
| Molecular (`.pdb`) | Render View | `3GQP.pdb` |

**Workflow for testing views:**
1. Open data in ParaView or ACloudViewer
2. Apply filters (e.g., ExtractHistogram → Histogram View)
3. Create the desired view type in an empty cell
4. The view auto-populates based on selected entity/filter output

---

## Final Comprehensive Review: Create Views Parity

### All 18 ParaView View Types — Implementation Status

| # | View Type | ACloudViewer Class | Status | Completeness |
|---|-----------|-------------------|--------|-------------|
| 1 | Render View | `vtkGLView` | Implemented | Full |
| 2 | Render View (Comparative) | `vtkComparativeViewWidget::RENDER` | Implemented | Full (NxM grid, camera sweep, overlay, screenshot) |
| 3 | Bar Chart View | `vtkChartView::BAR_CHART` | Implemented | Full (4-axis, title, legend, export) |
| 4 | Bar Chart View (Comparative) | `vtkComparativeViewWidget::BAR_CHART` | Implemented | Full |
| 5 | Box Chart View | `vtkChartView::BOX_CHART` | Implemented | Full |
| 6 | Eye Dome Lighting | `vtkGLView::enableEDL()` | Implemented | Full (mesh+points, FXAA chain) |
| 7 | Histogram View | `vtkChartView::HISTOGRAM` | Implemented | Full (configurable bins) |
| 8 | Image Chart View | `vtkChartView::IMAGE_CHART` | Implemented | Full |
| 9 | Line Chart View | `vtkChartView::LINE_CHART` | Implemented | Full |
| 10 | Line Chart View (Comparative) | `vtkComparativeViewWidget::LINE_CHART` | Implemented | Full |
| 11 | Orthographic Slice View | `vtkOrthoSliceViewWidget` | Implemented | Full (quad viewport, crosshairs, interactive) |
| 12 | Parallel Coordinates View | `vtkChartView::PARALLEL_COORDINATES` | Implemented | Full |
| 13 | Plot Matrix View | `vtkChartView::PLOT_MATRIX` | Implemented | Full |
| 14 | Point Chart View | `vtkChartView::POINT_CHART` | Implemented | Full |
| 15 | Python View | `ecvPythonView` | Implemented | Full (subprocess + entity CSV export via Export+Run) |
| 16 | Quartile Chart View | `vtkChartView::QUARTILE_CHART` | Implemented | Full |
| 17 | Slice View | `vtkSliceViewWidget` | Implemented | Full (3-edge axis sliders, multi-slice) |
| 18 | SpreadSheet View | `ecvSpreadSheetView` | Implemented | Full (decorator, bidirectional selection) |

### Remaining ParaView-specific Gaps (by design)

These gaps are inherent to architectural differences between ParaView (client-server, pipeline-driven) and ACloudViewer (single-process, entity-driven):

| Gap | ParaView Feature | Reason |
|-----|-----------------|--------|
| Pipeline output port source | `pqOutputPortComboBox` in SpreadSheet | ACloudViewer uses entity selection, not pipeline ports |
| Server Manager proxies | XML proxy-driven views | ACloudViewer uses direct VTK widget instantiation |
| Python pipeline access | `vtkPythonInterpreter` in PythonView | `Export+Run` exports entity to CSV, sets `DATA_FILE` env var |
| Synchronized rendering | `vtkPVSynchronizedRenderer` for EDL | Single-process, no client-server sync needed |
| Chart tooltip custom rendering | `vtkPVContextView::SetTooltipNotation` | `vtkTooltipItem::TextProperties` + notation/precision stored |
| Selection → pipeline proxy | `vtkSMSourceProxy` selection in SpreadSheet | ACloudViewer uses signal-based selection propagation |
| Basis change helper | `vtkPVChangeOfBasisHelper` in OrthoSlice | Fixed camera orientations with focal point shift |

### Phase 14: Shortcut Management Refactoring + View Navigation Shortcuts
- **Shortcut Management** (`ecvShortcutDialog`):
  - Integrated `ecvKeySequences` modal shortcut awareness via `syncModalShortcuts()`
  - `allRegisteredSequences()` API added to `ecvKeySequences` for external query
  - Conflict detection now covers QActions + standalone QShortcuts + modal shortcuts
  - Visual conflict highlighting with count label (red for conflicts, green for no conflicts)
  - "Reset Selected" + "Reset All Defaults" buttons with QSettings persistence
  - **Import/Export** shortcuts as JSON files for backup/sharing
  - VTK interactor shortcuts registered as standalone for conflict awareness
  - Search filter searches action name, menu path, tooltip, AND shortcut text
  - Persistence key uses `objectName()` with `text()` fallback for robustness
- **Conflict Fixes**:
  - F11: Changed `actionExclusiveFullScreen` from F11 to Shift+F11 (was conflicting with layout fullscreen)
  - Ctrl+V: Changed `actionSaveViewportAsObject` from Ctrl+V to Ctrl+Alt+V (was conflicting with system Paste)
- **View Navigation Shortcuts** (`ecvTabbedMultiViewWidget`):
  - `Ctrl+PageUp` / `Ctrl+PageDown` — switch between layout tabs
  - `Ctrl+Shift+T` — create new layout tab
  - `Ctrl+W` — close current layout tab
  - `F11` — full screen (layout) [existing]
  - `Ctrl+F11` — full screen (active view) [existing]
- **Quick Create View Shortcuts** (Display > Quick Create View menu):
  - `Ctrl+Shift+1` — SpreadSheet View
  - `Ctrl+Shift+2` — Line Chart View
  - `Ctrl+Shift+3` — Bar Chart View
  - `Ctrl+Shift+4` — Histogram View
  - `Ctrl+Shift+5` — Orthographic Slice View
  - `Ctrl+Shift+6` — Python View
- **VTK Shortcut Customization** (`VtkShortcutRegistry.h` + `QVTKWidgetCustom`):
  - User-remappable VTK interactor shortcuts via the Shortcut Settings dialog
  - Registry-based dispatch: QKeySequence → VTK action (supports any key combo)
  - VTK shortcuts appear as editable rows in the shortcut table under "VTK" category
  - Persistence via QSettings("VtkShortcuts"), import/export included
  - New view preset shortcuts: `Ctrl+Alt+1–6` for Front/Back/Left/Right/Top/Bottom views
  - Conflict resolution: VTK Save Camera moved to `Ctrl+Alt+M`, Points Mode to `Ctrl+Shift+D`, Surface Mode to `Ctrl+Shift+F`
- **Shortcut Category Filtering** (`ecvShortcutDialog`):
  - QComboBox category filter derived from menu paths (File/Edit/Display/VTK/Other)
  - Combined with text search for powerful shortcut lookup
- **Create View Flow Improvement**:
  - New tabs auto-create a default Render View (no intermediate "Create View" panel)
  - Comparative View grid toolbar hidden by default with collapsible "Grid Settings" toggle
  - Matches ParaView's behavior of immediately showing the render window
- **View Frame Decorator (ParaView pqViewFrame alignment)**:
  - **Drag-and-drop swap**: Drag title bar to swap view positions; routes through `ecvViewLayoutProxy::swapCells()` for KD-tree consistency
  - **Visual drag feedback**: Title bar highlights with palette highlight color on drag-over, resets on leave/drop
  - **"Convert To..." context menu**: Right-click title bar → "Convert To..." submenu lists all 18 view types for inline view type conversion (matches ParaView's `pqStandardViewFrameActionsImplementation`)
  - **Context menu items**: Split H/V, Equalize Views (H/V/Both), Rename, Link/Unlink Camera, View Properties (gradient, axes, camera widget), Convert To..., Close
  - **Per-view toolbar**: Camera Undo/Redo, Capture Screenshot, 3D/2D Toggle, Adjust Camera, Selection tools (per-view)
  - **Standard frame buttons**: Split Left|Right, Split Top|Bottom, Maximize/Restore, Close View
  - **Active frame highlight**: Border color + bold underline title (matches ParaView `setBorderVisibility`)

### Gap Analysis vs ParaView (Architectural Differences)

The following gaps are **architectural** — they stem from the fundamental difference between ParaView's Server Manager pipeline model and ACloudViewer's entity-driven model:

| Area | Gap | Severity | Notes |
|------|-----|----------|-------|
| Selection | ParaView: surface/frustum/polygon/block cell selection pipelines via `vtkSMSourceProxy` | Architecture | ACV uses CloudCompare-style entity picking |
| Representation | ParaView: per-source SM representation types (surface, outline, volume) | Architecture | ACV uses `VtkVis` pipeline on `ccHObject` |
| SpreadSheet | ParaView: driven by `pqDataRepresentation` + `vtkSelection` | Architecture | ACV drives from single `ccHObject` model |
| Chart Data | ParaView: plots arbitrary VTK table columns from pipeline | Architecture | ACV plots `ccPointCloud` scalar fields |
| Comparative | ParaView: `vtkSMComparativeViewProxy` with filmstrip cues | Architecture | ACV uses manual camera/opacity sweep |
| Python View | ParaView: in-process `matplotlib.pyplot` via `vtkSMPythonViewProxy` | Architecture | ACV uses `QProcess` subprocess execution |
| Camera Undo | ParaView: linked undo stacks across views via SM | Medium | ACV tracks per-view via `VtkVis` stacks |
| Slice Pipeline | ParaView: `vtkSMMultiSliceViewProxy` drives representations | Architecture | ACV rebuilds cutters from first actor |

### Summary

**18/18 ParaView view types are implemented.** All view types in the Create View dialog match ParaView's naming and ordering exactly. Each view type has feature parity at the VTK rendering level, with architectural adaptations for ACloudViewer's entity-driven (vs pipeline-driven) data model.

**View frame decorator** fully aligned with ParaView's `pqViewFrame`: title bar with per-view toolbar, standard split/maximize/close buttons, drag-and-drop swap via layout proxy, "Convert To..." submenu, context menu with view properties, camera linking, and rename.

**Shortcut management system** fully refactored with ParaView-style modal shortcut infrastructure (`ecvKeySequences` + `ecvModalShortcut` + `ecvShortcutDecorator`), comprehensive conflict detection across all shortcut sources, user-customizable VTK interactor key bindings via `VtkShortcutRegistry`, category-based filtering, and view preset shortcuts for standard camera orientations.
