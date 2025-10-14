// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef DATAFILTER_H
#define DATAFILTER_H

#include <vtkApproximatingSubdivisionFilter.h>
#include <vtkCleanPolyData.h>
#include <vtkClipPolyData.h>
#include <vtkContourFilter.h>
#include <vtkCutter.h>
#include <vtkDecimatePro.h>
#include <vtkDelaunay2D.h>
#include <vtkDensifyPolyData.h>
#include <vtkExtractEdges.h>
#include <vtkGenericContourFilter.h>
#include <vtkGlyph3D.h>
#include <vtkHedgeHog.h>
#include <vtkHull.h>
#include <vtkMarchingCubes.h>
#include <vtkMarchingSquares.h>
#include <vtkProteinRibbonFilter.h>
#include <vtkQuadricDecimation.h>
#include <vtkRibbonFilter.h>
#include <vtkRuledSurfaceFilter.h>
#include <vtkShrinkFilter.h>
#include <vtkSmartPointer.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkSplineFilter.h>
#include <vtkStreamTracer.h>
#include <vtkStripper.h>
#include <vtkThreshold.h>
#include <vtkTriangleFilter.h>
#include <vtkTubeFilter.h>
#include <vtkVertexGlyphFilter.h>

#include <QObject>
#include <QRunnable>

#include "../qPCL.h"
#include "signalledrunable.h"

namespace VtkUtils {

enum FilterType {
    FT_Clip,
    FT_Cut,
    FT_Slice,
    FT_Isosurface,
    FT_Threshold,
    FT_Streamline,
    FT_Smooth,
    FT_Decimate,
    FT_Count
};

QString QPCL_ENGINE_LIB_API filterName(FilterType type);

class QPCL_ENGINE_LIB_API AbstractDataFilter : public SignalledRunnable {
public:
    AbstractDataFilter() {}
    virtual ~AbstractDataFilter() {}
};

template <class Filter,
          class InputDataObject = vtkPolyData,
          class OutputDataObject = vtkPolyData>
class DataFilter : public AbstractDataFilter {
public:
    DataFilter() { m_filter = Filter::New(); }

    virtual void run() {
        m_filter->SetInputData(m_inputData);
        m_filter->Update();
        emit finished();
    }

    virtual void setInput(InputDataObject* input) { m_inputData = input; }

    OutputDataObject* output() const { return m_filter->GetOutput(); }

    Filter* filter() const { return m_filter; }

protected:
    vtkSmartPointer<Filter> m_filter;
    InputDataObject* m_inputData;
};

class QPCL_ENGINE_LIB_API ClipFilter : public DataFilter<vtkClipPolyData> {};
class QPCL_ENGINE_LIB_API CutterFilter : public DataFilter<vtkCutter> {};
class QPCL_ENGINE_LIB_API SliceFilter : public DataFilter<vtkCutter> {};
class QPCL_ENGINE_LIB_API DecimateProFilter
    : public DataFilter<vtkDecimatePro> {};
class QPCL_ENGINE_LIB_API SmoothFilter
    : public DataFilter<vtkSmoothPolyDataFilter> {};
class QPCL_ENGINE_LIB_API StreamTracerFilter
    : public DataFilter<vtkStreamTracer> {};
class QPCL_ENGINE_LIB_API IsosurfaceFilter
    : public DataFilter<vtkContourFilter> {};
class QPCL_ENGINE_LIB_API ExtractEdgesFilter
    : public DataFilter<vtkExtractEdges> {};
class QPCL_ENGINE_LIB_API TubeFilter : public DataFilter<vtkTubeFilter> {};
class QPCL_ENGINE_LIB_API Delaunay2DFilter : public DataFilter<vtkDelaunay2D> {
};
class QPCL_ENGINE_LIB_API Glyph3DFilter : public DataFilter<vtkGlyph3D> {};
class QPCL_ENGINE_LIB_API RuledSurfaceFilter
    : public DataFilter<vtkRuledSurfaceFilter> {};
class QPCL_ENGINE_LIB_API VertexGlyphFilter
    : public DataFilter<vtkVertexGlyphFilter> {};

}  // namespace VtkUtils
#endif  // DATAFILTER_H
