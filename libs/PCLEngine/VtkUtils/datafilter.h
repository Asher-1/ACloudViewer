#ifndef DATAFILTER_H
#define DATAFILTER_H

#include <QRunnable>
#include <QObject>

#include "signalledrunable.h"

#include <vtkSmartPointer.h>
#include <vtkClipPolyData.h>
#include <vtkCutter.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkDecimatePro.h>
#include <vtkThreshold.h>
#include <vtkStreamTracer.h>
#include <vtkContourFilter.h>
#include <vtkExtractEdges.h>
#include <vtkTubeFilter.h>
#include <vtkDelaunay2D.h>
#include <vtkGlyph3D.h>
#include <vtkRuledSurfaceFilter.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkApproximatingSubdivisionFilter.h>
#include <vtkCleanPolyData.h>
#include <vtkGenericContourFilter.h>
#include <vtkDensifyPolyData.h>
#include <vtkHedgeHog.h>
#include <vtkMarchingCubes.h>
#include <vtkMarchingSquares.h>
#include <vtkRibbonFilter.h>
#include <vtkHull.h>
#include <vtkSplineFilter.h>
#include <vtkShrinkFilter.h>
#include <vtkStripper.h>
#include <vtkProteinRibbonFilter.h>
#include <vtkQuadricDecimation.h>
#include <vtkTriangleFilter.h>

#include "../qPCL.h"

namespace VtkUtils
{

enum FilterType { FT_Clip, FT_Cut, FT_Slice, FT_Isosurface,
                  FT_Threshold, FT_Streamline, FT_Smooth, FT_Decimate, FT_Count };

QString QPCL_ENGINE_LIB_API filterName(FilterType type);

class QPCL_ENGINE_LIB_API AbstractDataFilter : public SignalledRunnable
{
public:
    AbstractDataFilter() {}
    virtual ~AbstractDataFilter() {}
};

template <class Filter, class InputDataObject = vtkPolyData, class OutputDataObject = vtkPolyData>
class DataFilter : public AbstractDataFilter
{
public:
    DataFilter()
    {
        m_filter = Filter::New();
    }

    virtual void run()
    {
        m_filter->SetInputData(m_inputData);
        m_filter->Update();
        emit finished();
    }

    virtual void setInput(InputDataObject* input)
    {
        m_inputData = input;
    }

    OutputDataObject* output() const
    {
        return m_filter->GetOutput();
    }

    Filter* filter() const
    {
        return m_filter;
    }

protected:
    vtkSmartPointer<Filter> m_filter;
    InputDataObject* m_inputData;
};

class QPCL_ENGINE_LIB_API ClipFilter : public DataFilter<vtkClipPolyData> {};
class QPCL_ENGINE_LIB_API CutterFilter : public DataFilter<vtkCutter> {};
class QPCL_ENGINE_LIB_API SliceFilter : public DataFilter<vtkCutter> {};
class QPCL_ENGINE_LIB_API DecimateProFilter : public DataFilter<vtkDecimatePro> {};
class QPCL_ENGINE_LIB_API SmoothFilter : public DataFilter<vtkSmoothPolyDataFilter> {};
class QPCL_ENGINE_LIB_API StreamTracerFilter : public DataFilter<vtkStreamTracer> {};
class QPCL_ENGINE_LIB_API IsosurfaceFilter : public DataFilter<vtkContourFilter> {};
class QPCL_ENGINE_LIB_API ExtractEdgesFilter : public DataFilter<vtkExtractEdges> {};
class QPCL_ENGINE_LIB_API TubeFilter : public DataFilter<vtkTubeFilter> {};
class QPCL_ENGINE_LIB_API Delaunay2DFilter : public DataFilter<vtkDelaunay2D> {};
class QPCL_ENGINE_LIB_API Glyph3DFilter : public DataFilter<vtkGlyph3D> {};
class QPCL_ENGINE_LIB_API RuledSurfaceFilter : public DataFilter<vtkRuledSurfaceFilter> {};
class QPCL_ENGINE_LIB_API VertexGlyphFilter : public DataFilter<vtkVertexGlyphFilter> {};

} // namespace VtkUtils
#endif // DATAFILTER_H
