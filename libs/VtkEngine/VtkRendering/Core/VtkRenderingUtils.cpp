// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file VtkRenderingUtils.cpp
 * @brief Implementation of VTK rendering utilities (actors, colors, scalar
 * bars).
 */

#include "VtkRenderingUtils.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>
#include <ecvGLMatrix.h>

// CV_DB_LIB
#include <LineSet.h>
#include <ecvCameraSensor.h>
#include <ecvColorScale.h>
#include <ecvDisplayTools.h>
#include <ecvDrawContext.h>
#include <ecvGBLSensor.h>
#include <ecvScalarField.h>

// VTK Extensions
#include <VTKExtensions/Utility/vtkDiscretizableColorTransferFunctionCustom.h>
#include <VTKExtensions/Views/vtkContext2DScalarBarActor.h>
#include <VTKExtensions/Views/vtkScalarBarActorCustom.h>
#include <VTKExtensions/Views/vtkScalarBarRepresentationCustom.h>
#include <VTKExtensions/Widgets/vtkScalarBarWidgetCustom.h>

#include "VtkUtils/vtkutils.h"

// VTK
#include <vtkAbstractWidget.h>
#include <vtkActor.h>
#include <vtkAnnotatedCubeActor.h>
#include <vtkAppendPolyData.h>
#include <vtkAxesActor.h>
#include <vtkCaptionActor2D.h>
#include <vtkCellData.h>
#include <vtkCubeSource.h>
#include <vtkLine.h>
#include <vtkLineSource.h>
#include <vtkMapper.h>
#include <vtkPoints.h>
#include <vtkPropAssembly.h>
#include <vtkProperty2D.h>
#include <vtkTextProperty.h>
#include <vtkUnsignedCharArray.h>

#include <list>

namespace VtkRendering {

// =====================================================================
// VTK Actor Creation
// =====================================================================

void CreateActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet>& data,
                               vtkSmartPointer<vtkLODActor>& actor,
                               bool use_scalars) {
    if (!actor) actor = vtkSmartPointer<vtkLODActor>::New();

    auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(data);

    if (use_scalars) {
        auto* scalars = data->GetPointData()->GetScalars();
        if (scalars) {
            double minmax[2];
            scalars->GetRange(minmax);
            mapper->SetScalarRange(minmax);
            mapper->SetScalarModeToUsePointData();

            auto* poly = vtkPolyData::SafeDownCast(data);
            bool interp = poly && (poly->GetNumberOfCells() !=
                                   poly->GetNumberOfVerts());
            mapper->SetInterpolateScalarsBeforeMapping(interp ? 1 : 0);
            mapper->ScalarVisibilityOn();
        }
    }

    actor->SetNumberOfCloudPoints(static_cast<int>(
            std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
    actor->GetProperty()->SetInterpolationToFlat();
    actor->SetMapper(mapper);
}

void CreateActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet>& data,
                               vtkSmartPointer<vtkActor>& actor,
                               bool use_scalars) {
    if (!actor) actor = vtkSmartPointer<vtkActor>::New();

    auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(data);

    if (use_scalars) {
        auto* scalars = data->GetPointData()->GetScalars();
        if (scalars) {
            double minmax[2];
            scalars->GetRange(minmax);
            mapper->SetScalarRange(minmax);
            mapper->SetScalarModeToUsePointData();
            auto* poly = vtkPolyData::SafeDownCast(data);
            bool interp = poly && (poly->GetNumberOfCells() !=
                                   poly->GetNumberOfVerts());
            mapper->SetInterpolateScalarsBeforeMapping(interp ? 1 : 0);
            mapper->ScalarVisibilityOn();
        }
    }

    actor->GetProperty()->SetInterpolationToFlat();
    actor->SetMapper(mapper);
}

// =====================================================================
// Internal line/point helpers
// =====================================================================

static vtkSmartPointer<vtkPoints> GetVtkPointsFromLineSet(
        const cloudViewer::geometry::LineSet& lineset) {
    vtkSmartPointer<vtkPoints> linePoints = vtkSmartPointer<vtkPoints>::New();
    linePoints->SetNumberOfPoints(
            static_cast<vtkIdType>(2 * lineset.lines_.size()));
    for (std::size_t i = 0; i < lineset.lines_.size(); ++i) {
        auto segment = lineset.GetLineCoordinate(i);
        linePoints->SetPoint(static_cast<vtkIdType>(2 * i),
                             segment.first.data());
        linePoints->SetPoint(static_cast<vtkIdType>(2 * i + 1),
                             segment.second.data());
    }
    return linePoints;
}

static bool GetVtkPointsAndLinesFromLineSet(
        const cloudViewer::geometry::LineSet& lineset,
        vtkSmartPointer<vtkPoints> points,
        vtkSmartPointer<vtkCellArray> lines,
        vtkSmartPointer<vtkUnsignedCharArray> colors) {
    if (!points || !lines) return false;

    bool has_color = false;
    if (lineset.hasColors()) {
        has_color = true;
        colors->SetNumberOfComponents(3);
        colors->SetName("Colors");
        colors->SetNumberOfTuples(
                static_cast<vtkIdType>(lineset.points_.size()));
    }

    points->SetNumberOfPoints(static_cast<vtkIdType>(lineset.points_.size()));
    for (std::size_t i = 0; i < lineset.points_.size(); ++i) {
        Eigen::Vector3d p = lineset.points_[i];
        points->SetPoint(static_cast<vtkIdType>(i), p.data());
    }

    for (std::size_t i = 0; i < lineset.lines_.size(); ++i) {
        vtkSmartPointer<vtkLine> segment = vtkSmartPointer<vtkLine>::New();
        Eigen::Vector2i segIndex = lineset.lines_[i];
        segment->GetPointIds()->SetId(0, segIndex(0));
        segment->GetPointIds()->SetId(1, segIndex(1));
        lines->InsertNextCell(segment);
        if (has_color) {
            ecvColor::Rgb color = ecvColor::Rgb::FromEigen(lineset.colors_[i]);
            colors->InsertTuple3(static_cast<vtkIdType>(i), color.r, color.g,
                                 color.b);
        }
    }
    return true;
}

static vtkSmartPointer<vtkPolyData> CreateLine(
        vtkSmartPointer<vtkPoints> points) {
    vtkSmartPointer<vtkLineSource> lineSource =
            vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoints(points);
    lineSource->Update();
    return lineSource->GetOutput();
}

static vtkSmartPointer<vtkPolyData> CreateLine(
        vtkSmartPointer<vtkPoints> points,
        vtkSmartPointer<vtkCellArray> lines,
        vtkSmartPointer<vtkUnsignedCharArray> colors) {
    if (points->GetNumberOfPoints() == 0 && lines->GetNumberOfCells() == 0 &&
        colors->GetNumberOfTuples() == 0) {
        return nullptr;
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    if (points->GetNumberOfPoints() > 0) polyData->SetPoints(points);
    if (lines->GetNumberOfCells() > 0) polyData->SetLines(lines);
    if (colors->GetNumberOfTuples() > 0)
        polyData->GetCellData()->SetScalars(colors);
    return polyData;
}

static vtkSmartPointer<vtkPolyData> CreateCoordinateFromLineSet(
        const cloudViewer::geometry::LineSet& lineset) {
    assert(lineset.lines_.size() == 3);

    vtkSmartPointer<vtkPoints> lineR = vtkSmartPointer<vtkPoints>::New();
    lineR->SetNumberOfPoints(2);
    auto segR = lineset.GetLineCoordinate(0);
    lineR->SetPoint(0, segR.first.data());
    lineR->SetPoint(1, segR.second.data());
    vtkSmartPointer<vtkPolyData> rData = CreateLine(lineR);
    SetPolyDataColor(rData, ecvColor::red, false);

    vtkSmartPointer<vtkPoints> lineUp = vtkSmartPointer<vtkPoints>::New();
    lineUp->SetNumberOfPoints(2);
    auto segUp = lineset.GetLineCoordinate(1);
    lineUp->SetPoint(0, segUp.first.data());
    lineUp->SetPoint(1, segUp.second.data());
    vtkSmartPointer<vtkPolyData> uData = CreateLine(lineUp);
    SetPolyDataColor(uData, ecvColor::yellow, false);

    vtkSmartPointer<vtkPoints> lineV = vtkSmartPointer<vtkPoints>::New();
    lineV->SetNumberOfPoints(2);
    auto segV = lineset.GetLineCoordinate(2);
    lineV->SetPoint(0, segV.first.data());
    lineV->SetPoint(1, segV.second.data());
    vtkSmartPointer<vtkPolyData> vData = CreateLine(lineV);
    SetPolyDataColor(vData, ecvColor::green, false);

    vtkSmartPointer<vtkAppendPolyData> appendFilter =
            vtkSmartPointer<vtkAppendPolyData>::New();
    appendFilter->AddInputData(rData);
    appendFilter->AddInputData(uData);
    appendFilter->AddInputData(vData);
    appendFilter->Update();
    return appendFilter->GetOutput();
}

// =====================================================================
// Public helpers
// =====================================================================

void SetPolyDataColor(vtkSmartPointer<vtkPolyData> polyData,
                      const ecvColor::Rgb& color,
                      bool is_cell) {
    vtkSmartPointer<vtkUnsignedCharArray> colors =
            vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("Colors");

    if (is_cell) {
        colors->SetNumberOfTuples(polyData->GetPolys()->GetNumberOfCells());
        for (vtkIdType i = 0; i < polyData->GetPolys()->GetNumberOfCells();
             ++i) {
            colors->InsertTuple3(i, color.r, color.g, color.b);
        }
        polyData->GetCellData()->SetScalars(colors);
    } else {
        colors->SetNumberOfTuples(polyData->GetNumberOfPoints());
        for (vtkIdType i = 0; i < polyData->GetNumberOfPoints(); ++i) {
            colors->InsertTuple3(i, color.r, color.g, color.b);
        }
        polyData->GetPointData()->SetScalars(colors);
    }
}

void AddPolyDataCell(vtkSmartPointer<vtkPolyData> polyData) {
    vtkSmartPointer<vtkCellArray> cellArray =
            vtkSmartPointer<vtkCellArray>::New();
    vtkIdType n_points = polyData->GetNumberOfPoints();
    if (n_points == 8) {
        vtkIdType cellId1[3] = {0, 1, 3};
        cellArray->InsertNextCell(3, cellId1);
        vtkIdType cellId2[3] = {7, 6, 4};
        cellArray->InsertNextCell(3, cellId2);
        polyData->SetPolys(cellArray);
    } else if (n_points == 14) {
        vtkIdType cellId1[3] = {0, 1, 3};
        cellArray->InsertNextCell(3, cellId1);
        vtkIdType cellId2[3] = {7, 6, 4};
        cellArray->InsertNextCell(3, cellId2);
        vtkIdType cellId3[3] = {8, 9, 11};
        cellArray->InsertNextCell(3, cellId3);
        polyData->SetPolys(cellArray);
    }
}

bool TransformPolyData(vtkSmartPointer<vtkPolyData> polyData,
                       const ccGLMatrixd& trans) {
    if (!polyData) return false;
    vtkSmartPointer<vtkPoints> points = polyData->GetPoints();
    if (!points || points->GetNumberOfPoints() == 0) return false;
    return TransformVtkPoints(points, trans);
}

bool TransformVtkPoints(vtkSmartPointer<vtkPoints> points,
                        const ccGLMatrixd& trans) {
    if (!points || points->GetNumberOfPoints() == 0) return false;
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
        double* P = points->GetPoint(i);
        trans.apply(P);
    }
    return true;
}

// =====================================================================
// Line Set Conversion
// =====================================================================

vtkSmartPointer<vtkPolyData> CreatePolyDataFromLineSet(
        const cloudViewer::geometry::LineSet& lineset, bool useLineSource) {
    if (useLineSource) {
        return CreateLine(GetVtkPointsFromLineSet(lineset));
    } else {
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> lines =
                vtkSmartPointer<vtkCellArray>::New();
        vtkSmartPointer<vtkUnsignedCharArray> colors =
                vtkSmartPointer<vtkUnsignedCharArray>::New();
        if (GetVtkPointsAndLinesFromLineSet(lineset, points, lines, colors)) {
            return CreateLine(points, lines, colors);
        }
        return nullptr;
    }
}

// =====================================================================
// Geometric Primitive Creation
// =====================================================================

vtkSmartPointer<vtkPolyData> CreateCube(double width,
                                        double height,
                                        double depth,
                                        const ccGLMatrixd& trans) {
    vtkSmartPointer<vtkPolyData> data = CreateCube(width, height, depth);
    if (!TransformPolyData(data, trans)) {
        CVLog::Error("[VtkRendering::CreateCube] Creating cube failed!");
        return nullptr;
    }
    return data;
}

vtkSmartPointer<vtkPolyData> CreateCube(double width,
                                        double height,
                                        double depth) {
    vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
    cube->SetXLength(width);
    cube->SetYLength(height);
    cube->SetZLength(depth);
    cube->Update();
    return cube->GetOutput();
}

// =====================================================================
// Sensor Visualization
// =====================================================================

vtkSmartPointer<vtkPolyData> CreateGBLSensor(const ccGBLSensor* gBLSensor) {
    assert(gBLSensor);
    auto linePoints =
            cloudViewer::geometry::LineSet::CreateFromOrientedBoundingBox(
                    gBLSensor->getSensorHead());

    vtkSmartPointer<vtkPolyData> headLinesData =
            CreatePolyDataFromLineSet(*linePoints, false);
    vtkSmartPointer<vtkPolyData> legLinesData =
            CreatePolyDataFromLineSet(gBLSensor->getSensorLegLines(), false);
    vtkSmartPointer<vtkPolyData> axisLinesData =
            CreatePolyDataFromLineSet(gBLSensor->getSensorAxis(), false);

    vtkSmartPointer<vtkAppendPolyData> appendFilter =
            vtkSmartPointer<vtkAppendPolyData>::New();
    appendFilter->AddInputData(headLinesData);
    appendFilter->AddInputData(legLinesData);
    appendFilter->AddInputData(axisLinesData);
    appendFilter->Update();
    return appendFilter->GetOutput();
}

vtkSmartPointer<vtkPolyData> CreateCameraSensor(
        const ccCameraSensor* cameraSensor,
        const ecvColor::Rgb& lineColor,
        const ecvColor::Rgb& planeColor) {
    assert(cameraSensor);

    vtkSmartPointer<vtkPolyData> nearPlaneLinesData =
            CreatePolyDataFromLineSet(cameraSensor->getNearPlane());
    AddPolyDataCell(nearPlaneLinesData);
    SetPolyDataColor(nearPlaneLinesData, planeColor, false);

    vtkSmartPointer<vtkPolyData> sideLinesData =
            CreatePolyDataFromLineSet(cameraSensor->getSideLines());
    SetPolyDataColor(sideLinesData, lineColor, false);

    vtkSmartPointer<vtkPolyData> arrowLinesData =
            CreatePolyDataFromLineSet(cameraSensor->getArrow());
    AddPolyDataCell(arrowLinesData);
    SetPolyDataColor(arrowLinesData, lineColor, false);

    vtkSmartPointer<vtkPolyData> axisLinesData =
            CreateCoordinateFromLineSet(cameraSensor->getAxis());

    vtkSmartPointer<vtkAppendPolyData> appendFilter =
            vtkSmartPointer<vtkAppendPolyData>::New();
    appendFilter->AddInputData(nearPlaneLinesData);
    appendFilter->AddInputData(sideLinesData);
    appendFilter->AddInputData(arrowLinesData);
    appendFilter->AddInputData(axisLinesData);
    appendFilter->Update();
    return appendFilter->GetOutput();
}

// =====================================================================
// Coordinate System
// =====================================================================

vtkSmartPointer<vtkPropAssembly> CreateCoordinate(double axesLength,
                                                  const std::string& xLabel,
                                                  const std::string& yLabel,
                                                  const std::string& zLabel,
                                                  const std::string& xPlus,
                                                  const std::string& xMinus,
                                                  const std::string& yPlus,
                                                  const std::string& yMinus,
                                                  const std::string& zPlus,
                                                  const std::string& zMinus) {
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    axes->SetXAxisLabelText(xLabel.c_str());
    axes->SetYAxisLabelText(yLabel.c_str());
    axes->SetZAxisLabelText(zLabel.c_str());

    axes->SetNormalizedShaftLength(0.85, 0.85, 0.85);
    axes->SetNormalizedTipLength(0.15, 0.15, 0.15);
    axes->SetShaftTypeToLine();

    axes->GetXAxisTipProperty()->SetColor(1.0, 0.0, 0.0);
    axes->GetXAxisShaftProperty()->SetColor(1.0, 0.0, 0.0);
    axes->GetYAxisTipProperty()->SetColor(0.0, 1.0, 0.0);
    axes->GetYAxisShaftProperty()->SetColor(0.0, 1.0, 0.0);
    axes->GetZAxisTipProperty()->SetColor(0.0, 0.0, 1.0);
    axes->GetZAxisShaftProperty()->SetColor(0.0, 0.0, 1.0);

    axes->SetTotalLength(axesLength, axesLength, axesLength);

    vtkSmartPointer<vtkAnnotatedCubeActor> cube =
            vtkSmartPointer<vtkAnnotatedCubeActor>::New();
    cube->SetXPlusFaceText(xPlus.c_str());
    cube->SetXMinusFaceText(xMinus.c_str());
    cube->SetYPlusFaceText(yPlus.c_str());
    cube->SetYMinusFaceText(yMinus.c_str());
    cube->SetZPlusFaceText(zPlus.c_str());
    cube->SetZMinusFaceText(zMinus.c_str());
    cube->SetXFaceTextRotation(180);
    cube->SetYFaceTextRotation(180);
    cube->SetZFaceTextRotation(-90);
    cube->SetFaceTextScale(0.50);
    cube->GetCubeProperty()->SetColor(0.8, 0.8, 0.8);
    cube->GetTextEdgesProperty()->SetLineWidth(1);
    cube->GetTextEdgesProperty()->SetDiffuse(0);
    cube->GetTextEdgesProperty()->SetAmbient(1);
    cube->GetTextEdgesProperty()->SetColor(0.2, 0.2, 0.2);
    vtkMapper::SetResolveCoincidentTopologyToPolygonOffset();

    cube->GetXPlusFaceProperty()->SetColor(1.0, 0.0, 0.0);
    cube->GetXPlusFaceProperty()->SetInterpolationToFlat();
    cube->GetXMinusFaceProperty()->SetColor(1.0, 0.0, 0.0);
    cube->GetXMinusFaceProperty()->SetInterpolationToFlat();
    cube->GetYPlusFaceProperty()->SetColor(0.0, 1.0, 0.0);
    cube->GetYPlusFaceProperty()->SetInterpolationToFlat();
    cube->GetYMinusFaceProperty()->SetColor(0.0, 1.0, 0.0);
    cube->GetYMinusFaceProperty()->SetInterpolationToFlat();
    cube->GetZPlusFaceProperty()->SetColor(0.0, 0.0, 1.0);
    cube->GetZPlusFaceProperty()->SetInterpolationToFlat();
    cube->GetZMinusFaceProperty()->SetColor(0.0, 0.0, 1.0);
    cube->GetZMinusFaceProperty()->SetInterpolationToFlat();

    vtkSmartPointer<vtkTextProperty> tpropX =
            vtkSmartPointer<vtkTextProperty>::New();
    tpropX->ShadowOn();
    tpropX->SetFontFamilyToArial();
    tpropX->BoldOn();
    tpropX->SetFontSize(14);
    tpropX->SetColor(1.0, 0.0, 0.0);
    axes->GetXAxisCaptionActor2D()->SetCaptionTextProperty(tpropX);

    vtkSmartPointer<vtkTextProperty> tpropY =
            vtkSmartPointer<vtkTextProperty>::New();
    tpropY->ShadowOn();
    tpropY->SetFontFamilyToArial();
    tpropY->BoldOn();
    tpropY->SetFontSize(14);
    tpropY->SetColor(0.0, 1.0, 0.0);
    axes->GetYAxisCaptionActor2D()->SetCaptionTextProperty(tpropY);

    vtkSmartPointer<vtkTextProperty> tpropZ =
            vtkSmartPointer<vtkTextProperty>::New();
    tpropZ->ShadowOn();
    tpropZ->SetFontFamilyToArial();
    tpropZ->BoldOn();
    tpropZ->SetFontSize(14);
    tpropZ->SetColor(0.0, 0.0, 1.0);
    axes->GetZAxisCaptionActor2D()->SetCaptionTextProperty(tpropZ);

    vtkSmartPointer<vtkPropAssembly> assembly =
            vtkSmartPointer<vtkPropAssembly>::New();
    assembly->AddPart(cube);
    assembly->AddPart(axes);
    return assembly;
}

// =====================================================================
// Scalar Bar (migrated from PclTools)
// =====================================================================

static const double c_log10 = log(10.0);

struct vlabel {
    int yPos;
    int yMin;
    int yMax;
    double val;
    vlabel(int y, int y1, int y2, double v)
        : yPos(y), yMin(y1), yMax(y2), val(v) {
        assert(y2 >= y1);
    }
};

using vlabelSet = std::list<vlabel>;

static void ConvertToLogScale(ScalarType& dispMin, ScalarType& dispMax) {
    ScalarType absDispMin = (dispMax < 0 ? std::min(-dispMax, -dispMin)
                                         : std::max<ScalarType>(dispMin, 0));
    ScalarType absDispMax = std::max(std::abs(dispMin), std::abs(dispMax));
    dispMin = std::log10(
            std::max(absDispMin, std::numeric_limits<ScalarType>::epsilon()));
    dispMax = std::log10(
            std::max(absDispMax, std::numeric_limits<ScalarType>::epsilon()));
}

using vlabelPair = std::pair<vlabelSet::iterator, vlabelSet::iterator>;

static vlabelPair GetVLabelsAround(int y, vlabelSet& set) {
    if (set.empty()) {
        return vlabelPair(set.end(), set.end());
    }
    vlabelSet::iterator it1 = set.begin();
    if (y < it1->yPos) {
        return vlabelPair(set.end(), it1);
    }
    vlabelSet::iterator it2 = it1;
    ++it2;
    for (; it2 != set.end(); ++it2, ++it1) {
        if (y <= it2->yPos) return vlabelPair(it1, it2);
    }
    return vlabelPair(it1, set.end());
}

bool UpdateScalarBar(vtkAbstractWidget* widget,
                     const CC_DRAW_CONTEXT& CONTEXT) {
    if (!widget) return false;
    std::string viewID = CVTools::FromQString(CONTEXT.viewID);
    const ccScalarField* sf = CONTEXT.sfColorScaleToDisplay;
    if (!sf || !sf->getColorScale()) {
        return false;
    }

    bool logScale = sf->logScale();
    bool symmetricalScale = sf->symmetricalScale();
    bool alwaysShowZero = sf->isZeroAlwaysShown();

    ccColorScale::LabelSet keyValues;
    bool customLabels = false;
    try {
        ccColorScale::Shared colorScale = sf->getColorScale();
        if (colorScale && colorScale->customLabels().size() >= 2) {
            keyValues = colorScale->customLabels();
            if (alwaysShowZero) keyValues.insert(0.0);
            customLabels = true;
        } else if (!logScale) {
            keyValues.insert(sf->displayRange().min());
            keyValues.insert(sf->displayRange().start());
            keyValues.insert(sf->displayRange().stop());
            keyValues.insert(sf->displayRange().max());
            keyValues.insert(sf->saturationRange().min());
            keyValues.insert(sf->saturationRange().start());
            keyValues.insert(sf->saturationRange().stop());
            keyValues.insert(sf->saturationRange().max());
            if (symmetricalScale)
                keyValues.insert(-sf->saturationRange().max());
            if (alwaysShowZero) keyValues.insert(0.0);
        } else {
            ScalarType minDisp = sf->displayRange().min();
            ScalarType maxDisp = sf->displayRange().max();
            ConvertToLogScale(minDisp, maxDisp);
            keyValues.insert(minDisp);
            keyValues.insert(maxDisp);
            ScalarType startDisp = sf->displayRange().start();
            ScalarType stopDisp = sf->displayRange().stop();
            ConvertToLogScale(startDisp, stopDisp);
            keyValues.insert(startDisp);
            keyValues.insert(stopDisp);
            keyValues.insert(sf->saturationRange().min());
            keyValues.insert(sf->saturationRange().start());
            keyValues.insert(sf->saturationRange().stop());
            keyValues.insert(sf->saturationRange().max());
        }
    } catch (const std::bad_alloc&) {
        return false;
    }

    {
        for (ccColorScale::LabelSet::iterator it = keyValues.begin();
             it != keyValues.end(); ++it) {
#if defined(CV_WINDOWS) && defined(_MSC_VER)
            if (!_finite(it->value))
#else
            if (!std::isfinite(it->value))
#endif
            {
                bool minusInf = (it->value < 0);
                keyValues.erase(it);
                if (minusInf)
                    keyValues.insert({std::numeric_limits<ScalarType>::lowest(),
                                      "-Inf"});
                else
                    keyValues.insert(
                            {std::numeric_limits<ScalarType>::max(), "+Inf"});
                it = keyValues.begin();
            }
        }
    }

    if (!customLabels && !sf->areNaNValuesShownInGrey()) {
        if (!logScale) {
            for (ccColorScale::LabelSet::iterator it = keyValues.begin();
                 it != keyValues.end();) {
                if (!sf->displayRange().isInRange(
                            static_cast<ScalarType>(it->value)) &&
                    (!alwaysShowZero || it->value != 0)) {
                    ccColorScale::LabelSet::iterator toDelete = it;
                    ++it;
                    keyValues.erase(toDelete);
                } else {
                    ++it;
                }
            }
        } else {
            ScalarType dispMin = sf->displayRange().start();
            ScalarType dispMax = sf->displayRange().stop();
            ConvertToLogScale(dispMin, dispMax);
            for (ccColorScale::LabelSet::iterator it = keyValues.begin();
                 it != keyValues.end();) {
                if (it->value >= dispMin && it->value <= dispMax) {
                    ++it;
                } else {
                    ccColorScale::LabelSet::iterator toDelete = it;
                    ++it;
                    keyValues.erase(toDelete);
                }
            }
        }
    }

    std::vector<ccColorScale::Label> sortedKeyValues(keyValues.begin(),
                                                     keyValues.end());
    double maxRange =
            sortedKeyValues.back().value - sortedKeyValues.front().value;

    const ecvGui::ParamStruct& displayParams =
            ecvDisplayTools::GetDisplayParameters();
    const ecvColor::Rgbub& textColor = displayParams.textDefaultCol;
    const ccScalarField::Histogram histogram = sf->getHistogram();
    bool showHistogram = (displayParams.colorScaleShowHistogram && !logScale &&
                          histogram.maxValue != 0 && histogram.size() > 1);

    float renderZoom = CONTEXT.renderZoom;
    QFont font = ecvDisplayTools::GetTextDisplayFont();
    const int strHeight =
            static_cast<int>(displayParams.defaultFontSize * renderZoom);
    const int scaleWidth =
            static_cast<int>(displayParams.colorScaleRampWidth * renderZoom);
    const int scaleMaxHeight =
            (keyValues.size() > 1
                     ? std::max(
                               CONTEXT.glH - static_cast<int>(140 * renderZoom),
                               2 * strHeight)
                     : scaleWidth);

    vlabelSet drawnLabels;
    {
        drawnLabels.emplace_back(0, 0, strHeight,
                                 sortedKeyValues.front().value);
        if (keyValues.size() > 1) {
            drawnLabels.emplace_back(scaleMaxHeight, scaleMaxHeight - strHeight,
                                     scaleMaxHeight,
                                     sortedKeyValues.back().value);
        }
        if (keyValues.size() > 2) {
            assert(maxRange > 0.0);
            const int minGap = strHeight;
            for (size_t i = 1; i < keyValues.size() - 1; ++i) {
                int yScale = static_cast<int>(
                        (sortedKeyValues[i].value - sortedKeyValues[0].value) *
                        scaleMaxHeight / maxRange);
                vlabelPair nLabels = GetVLabelsAround(yScale, drawnLabels);
                assert(nLabels.first != drawnLabels.end() &&
                       nLabels.second != drawnLabels.end());
                if ((nLabels.first == drawnLabels.end() ||
                     nLabels.first->yMax <= yScale - minGap) &&
                    (nLabels.second == drawnLabels.end() ||
                     nLabels.second->yMin >= yScale + minGap)) {
                    drawnLabels.insert(nLabels.second,
                                       vlabel(yScale, yScale - strHeight / 2,
                                              yScale + strHeight / 2,
                                              sortedKeyValues[i].value));
                }
            }
        }
        if (!customLabels && drawnLabels.size() > 1) {
            const int minGap = strHeight * 2;
            size_t drawnLabelsBefore = 0;
            size_t drawnLabelsAfter = drawnLabels.size();
            while (drawnLabelsAfter > drawnLabelsBefore) {
                drawnLabelsBefore = drawnLabelsAfter;
                vlabelSet::iterator it1 = drawnLabels.begin();
                vlabelSet::iterator it2 = it1;
                ++it2;
                for (; it2 != drawnLabels.end(); ++it2) {
                    if (it1->yMax + 2 * minGap < it2->yMin) {
                        double val = (it1->val + it2->val) / 2.0;
                        int yScale = static_cast<int>(
                                (val - sortedKeyValues[0].value) *
                                scaleMaxHeight / maxRange);
                        drawnLabels.insert(
                                it2, vlabel(yScale, yScale - strHeight / 2,
                                            yScale + strHeight / 2, val));
                    }
                    it1 = it2;
                }
                drawnLabelsAfter = drawnLabels.size();
            }
        }
    }

    vtkScalarBarWidgetCustom* scalarBarWidget =
            vtkScalarBarWidgetCustom::SafeDownCast(widget);
    if (!scalarBarWidget) return false;

    vtkContext2DScalarBarActor* lutActor =
            vtkContext2DScalarBarActor::SafeDownCast(
                    scalarBarWidget->GetScalarBarActor());
    if (!lutActor) {
        scalarBarWidget->Off();
        return false;
    }

    vtkScalarBarRepresentationCustom* rep =
            vtkScalarBarRepresentationCustom::SafeDownCast(
                    scalarBarWidget->GetRepresentation());
    if (rep) {
        rep->SetWindowLocation(
                vtkScalarBarRepresentationCustom::LowerRightCorner);
    }

    VTK_CREATE(vtkDiscretizableColorTransferFunctionCustom, lut);
    {
        lut->Build();
        for (int j = 0; j < scaleMaxHeight; ++j) {
            double baseValue = sortedKeyValues.front().value +
                               (j * maxRange) / scaleMaxHeight;
            double value = baseValue;
            if (logScale) value = std::exp(value * c_log10);
            const ecvColor::Rgb* col =
                    sf->getColor(static_cast<ScalarType>(value));
            if (!col) {
                if (customLabels) {
                    assert(sf->getColorScale() &&
                           !sf->getColorScale()->isRelative());
                    col = sf->getColorScale()->getColorByValue(
                            value, &ecvColor::lightGrey);
                } else {
                    col = &ecvColor::lightGrey;
                }
            }
            assert(col);
            Eigen::Vector3d rgb = ecvColor::Rgb::ToEigen(*col);
            lut->AddRGBPoint(value, rgb(0), rgb(1), rgb(2));
        }
    }

    const char* sfName = sf->getName();
    QString sfTitle(sfName);
    if (sfName) {
        if (sf->getGlobalShift() != 0) sfTitle += QString("[Shifted]");
        if (logScale) sfTitle += QString("[Log scale]");
    }
    lutActor->SetTitle(CVTools::FromQString(sfTitle).c_str());
    lutActor->SetLookupTable(lut);
    lutActor->SetOutlineScalarBar(showHistogram);
    lutActor->SetScalarBarLength(0.8);
    lutActor->SetScalarBarThickness(scaleWidth);
    lutActor->SetTitleJustification(VTK_TEXT_CENTERED);
    lutActor->SetForceHorizontalTitle(true);
    lutActor->SetDrawColorBar(1);
    lutActor->SetDrawTickLabels(1);
    lutActor->SetNumberOfTicks(static_cast<int>(drawnLabels.size()));
    lutActor->SetTextPositionToPrecedeScalarBar();
    lutActor->GetLabelTextProperty()->SetFontSize(
            static_cast<int>(displayParams.defaultFontSize * renderZoom));
    lutActor->SetDrawFrame(1);
    Eigen::Vector3d col = ecvColor::Rgb::ToEigen(textColor);
    lutActor->GetFrameProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetAnnotationTextProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetTitleTextProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetLabelTextProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetFrameProperty()->SetLineWidth(2.0f * renderZoom);
    const unsigned precision = displayParams.displayedNumPrecision;
    const char format = (sf->logScale() ? 'E' : 'f');
    QString formatInfo =
            QString("%1.%2%3").arg(precision).arg(precision).arg(format);
    const std::string labelFormat = "%" + formatInfo.toStdString();
    lutActor->SetRangeLabelFormat(labelFormat.c_str());
    scalarBarWidget->SetScalarBarActor(lutActor);
    scalarBarWidget->On();
    scalarBarWidget->Modified();
    return true;
}

}  // namespace VtkRendering
