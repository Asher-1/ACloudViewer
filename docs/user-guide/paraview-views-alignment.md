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
| 2 | SpreadSheet View | `SpreadSheetView` | **Implemented** | `ecvSpreadSheetView` (point clouds only) |
| 3 | Line Chart View | `XYChartView` | **Implemented** | `vtkChartView::LINE_CHART` |
| 4 | Bar Chart View | `XYBarChartView` | **Implemented** | `vtkChartView::BAR_CHART` |
| 5 | Histogram View | `XYHistogramChartView` | **Implemented** | `vtkChartView::HISTOGRAM` |
| 6 | Eye Dome Lighting | `RenderViewWithEDL` | **Implemented** | `vtkGLView::enableEDL()` |
| 7 | Orthographic Slice View | `OrthographicSliceView` | **Implemented** | 4-pane split with ortho cameras |
| 8 | Slice View | `MultiSlice` | **Implemented** | Single `vtkImplicitPlaneWidget2` (vs ParaView multi-slice) |
| 9 | Box Chart View | `BoxChartView` | **Implemented** | `vtkChartView::BOX_CHART` with `vtkChartBox` |
| 10 | Parallel Coordinates View | `ParallelCoordinatesChartView` | **Implemented** | `vtkChartView::PARALLEL_COORDINATES` |
| 11 | Plot Matrix View | `PlotMatrixView` | **Implemented** | `vtkChartView::PLOT_MATRIX` with `vtkScatterPlotMatrix` |
| 12 | Point Chart View | `XYPointChartView` | **Implemented** | `vtkChartView::POINT_CHART` |
| 13 | Quartile Chart View | `QuartileChartView` | Not implemented | — |
| 14 | Image Chart View | `ImageChartView` | Not implemented | — |
| 15 | Render View (Comparative) | `ComparativeRenderView` | Not implemented | — |
| 16 | Bar Chart View (Comparative) | `ComparativeXYBarChartView` | Not implemented | — |
| 17 | Line Chart View (Comparative) | `ComparativeXYChartView` | Not implemented | — |
| 18 | Python View | `PythonView` | Not implemented | — |

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

**Purpose:** Tabular display of point cloud attributes.

**Data sources:** Point clouds (`ccPointCloud`)

**Columns displayed:** Index, X, Y, Z, R, G, B (if colors), Nx, Ny, Nz (if normals), scalar fields.

**Features:**
- Filter rows by text search
- Export to CSV
- Sort by column header click
- Auto-updates when entity selection changes

**Usage:**
1. Load a point cloud
2. Select it in the DB tree
3. Create a SpreadSheet View in an empty cell
4. Data populates automatically

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

**Purpose:** Enhanced depth perception for point cloud visualization.

**Data sources:** Same as Render View

**Usage:** Creates a Render View with EDL post-processing enabled. Uses `vtkEDLShading` render pass for enhanced depth cues without explicit normals.

---

### 7. Orthographic Slice View

**Purpose:** Four-pane orthographic view showing XY, XZ, YZ slices + free 3D view.

**Data sources:** Same as Render View

**Layout:**
```
┌────────┬────────┐
│  XY    │  XZ    │
│ (top)  │(front) │
├────────┼────────┤
│  YZ    │  3D    │
│(side)  │(free)  │
└────────┴────────┘
```

Each sub-view is a full `vtkGLView` with parallel projection locked to the respective axis. The 4th view is a standard perspective 3D view.

---

### 8. Slice View

**Purpose:** Interactive clipping plane for sectional views.

**Data sources:** Same as Render View (best with meshes)

**Usage:**
1. Load a mesh or point cloud
2. Create a Slice View
3. An interactive plane widget appears
4. Drag/rotate the plane to clip geometry
5. Actors are clipped in real-time via `vtkClippingPlane`

**Note:** ACloudViewer uses a single `vtkImplicitPlaneWidget2` (one plane). ParaView's MultiSlice supports multiple orthogonal planes with configurable positions.

---

### 9. Box Chart View

**Purpose:** Display statistical distribution (min, Q1, median, Q3, max) as box-and-whisker plots for each selected scalar field.

**Data sources:** Point clouds with scalar fields or colors

**Usage:**
1. Load a point cloud with multiple scalar fields
2. Select the entity in the DB tree
3. Create a Box Chart View
4. Select fields from the Fields list (multi-select supported)
5. Each selected field renders as a color-coded box-and-whisker column

**How it works:**
- Uses VTK's `vtkChartBox` + `vtkPlotBox` internally
- Automatically computes the five-number summary from sampled data
- Color-coded columns match the field palette
- Ideal for comparing distributions of different scalar fields at a glance

**Controls:** Same as Line Chart View (Pan, Zoom, Reset Zoom, PNG/CSV export).

---

### 10. Parallel Coordinates View

**Purpose:** Multi-axis polyline visualization for high-dimensional scalar field analysis.

**Data sources:** Point clouds with 2+ scalar fields

**Usage:**
1. Load a point cloud with multiple scalar fields
2. Select the entity, create a Parallel Coordinates View
3. Select 2 or more fields from the Fields list
4. Each field becomes a vertical axis; each sampled point draws a polyline across all axes

**How it works:**
- Uses VTK's `vtkChartParallelCoordinates` + `vtkPlotParallelCoordinates`
- Lines are semi-transparent (alpha 0.3) to handle dense data without visual overload
- Point clouds >10k points are subsampled for performance
- Useful for identifying clusters, correlations, and outliers across multiple dimensions

**Controls:** Same as Line Chart View.

---

### 11. Point Chart View (Scatter)

**Purpose:** Scatter plot of scalar field values vs point index, using discrete point markers.

**Data sources:** Point clouds with scalar fields or colors

**Usage:** Same workflow as Line Chart View, but renders as scattered dots instead of connected lines.

**Key differences from Line Chart:**
- Uses `vtkChart::POINTS` plot type instead of `vtkChart::LINE`
- Wider marker width (3.0px) for visibility
- Better for spotting discrete distributions and outliers

**Controls:** Same as Line Chart View.

---

### 12. Plot Matrix View

**Purpose:** NxN scatter plot matrix for exploring pairwise relationships between scalar fields.

**Data sources:** Point clouds with 2+ scalar fields

**Usage:**
1. Load a point cloud with multiple scalar fields
2. Select the entity, create a Plot Matrix View
3. Select 2 or more fields from the Fields list
4. An NxN grid appears: scatter plots on off-diagonal cells, histograms on the diagonal

**How it works:**
- Uses VTK's `vtkScatterPlotMatrix` internally
- Click any off-diagonal cell to see that pair highlighted as the "active plot" in the top-right
- Diagonal cells show histograms of each individual field
- Great for identifying correlations, clusters, and outlier patterns across multiple dimensions

**Controls:** Click individual cells; Pan/Zoom on sub-charts.

---

## Not Yet Implemented Views

### Quartile Chart / Image Chart
Specialized chart types for quartile bands and image-based data visualization respectively.

### Comparative Views
Side-by-side parameter sweep grids. Each cell renders the same pipeline with different parameter values.

### Python View
Programmable view using Python scripts for custom rendering. Uses `vtkPythonView`.

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
