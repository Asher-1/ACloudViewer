//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                    COPYRIGHT: CLOUDVIEWER  project                     #
//#                                                                        #
//##########################################################################

#include "PclTools.h"

#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"

#include "VtkUtils/vtkutils.h"

// CV_CORE_LIB
#include <CVTools.h>
#include <ecvGLMatrix.h>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvPlane.h>
#include <LineSet.h>
#include <ecvGBLSensor.h>
#include <ecvCameraSensor.h>
#include <ecvColorScale.h>
#include <ecvScalarField.h>
#include <ecvDisplayTools.h>

#include <vtkLine.h>
#include <vtkCellData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkAppendPolyData.h>
#include <vtkPlaneSource.h>
#include <vtkProperty.h>
#include <vtkActor.h>
#include <vtkLineSource.h>
#include <vtkLODActor.h>
#include <vtkProperty2D.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolygon.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkDataSetMapper.h>
#include <vtkUnstructuredGrid.h>

#include <vtkTexture.h>
#include <vtkAxesActor.h>
#include <vtkPropAssembly.h>
#include <vtkCaptionActor2D.h>
#include <vtkAnnotatedCubeActor.h>

#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
#include <pcl/visualization/vtk/vtkVertexBufferObjectMapper.h>
#endif

#include <VTKExtensions/Views/vtkScalarBarActorCustom.h>
#include <VTKExtensions/Utility/vtkDiscretizableColorTransferFunctionCustom.h>
#include <VTKExtensions/Views/vtkContext2DScalarBarActor.h>
#include <VTKExtensions/Views/vtkScalarBarRepresentationCustom.h>
#include <VTKExtensions/Widgets/vtkScalarBarWidgetCustom.h>

#if defined(_WIN32)
  // Remove macros defined in Windows.h
#undef near
#undef far
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
void PclTools::CreateActorFromVTKDataSet(
	const vtkSmartPointer<vtkDataSet> &data,
	vtkSmartPointer<vtkLODActor> &actor,
	bool use_scalars,
	bool use_vbos)
{
	// If actor is not initialized, initialize it here
	if (!actor)
		actor = vtkSmartPointer<vtkLODActor>::New();

#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
	if (use_vbos)
	{
		vtkSmartPointer<vtkVertexBufferObjectMapper> mapper = vtkSmartPointer<vtkVertexBufferObjectMapper>::New();

		mapper->SetInput(data);

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}

		actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}
	else
#endif
	{
		vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION < 6
		mapper->SetInput(data);
#else
		mapper->SetInputData(data);
#endif

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
		mapper->ImmediateModeRenderingOff();
#endif

		actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////
void PclTools::CreateActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet> &data,
	vtkSmartPointer<vtkActor> &actor, bool use_scalars, bool use_vbos)
{
	// If actor is not initialized, initialize it here
	if (!actor)
		actor = vtkSmartPointer<vtkActor>::New();

#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
	if (use_vbos)
	{
		vtkSmartPointer<vtkVertexBufferObjectMapper> mapper = vtkSmartPointer<vtkVertexBufferObjectMapper>::New();

		mapper->SetInput(data);

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}

		//actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}
	else
#endif
	{
		vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION < 6
		mapper->SetInput(data);
#else
		mapper->SetInputData(data);
#endif

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
		mapper->ImmediateModeRenderingOff();
#endif

		//actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}

	//actor->SetNumberOfCloudPoints (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10));
	actor->GetProperty()->SetInterpolationToFlat();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void PclTools::AllocVtkUnstructuredGrid(vtkSmartPointer<vtkUnstructuredGrid> &polydata)
{
	polydata = vtkSmartPointer<vtkUnstructuredGrid>::New();
}

vtkSmartPointer<vtkPoints> PclTools::GetVtkPointsFromLineSet(
        const cloudViewer::geometry::LineSet &lineset)
{
    vtkSmartPointer<vtkPoints> linePoints = vtkSmartPointer<vtkPoints>::New();
    linePoints->SetNumberOfPoints(static_cast<vtkIdType>(2 * lineset.lines_.size()));
    for (std::size_t i = 0; i < lineset.lines_.size(); ++i)
    {
        std::pair<Eigen::Vector3d, Eigen::Vector3d> segment = lineset.getLineCoordinate(i);
        linePoints->SetPoint(static_cast<vtkIdType>(2 * i), segment.first.data());
        linePoints->SetPoint(static_cast<vtkIdType>(2 * i + 1), segment.second.data());
    }
    return linePoints;
}

bool PclTools::GetVtkPointsAndLinesFromLineSet(const cloudViewer::geometry::LineSet &lineset,
                                               vtkSmartPointer<vtkPoints> points,
                                               vtkSmartPointer<vtkCellArray> lines,
                                               vtkSmartPointer<vtkUnsignedCharArray> colors)
{
    if (!points || !lines)
    {
        return false;
    }

    bool has_color = false;
    if (lineset.hasColors())
    {
      has_color = true;
      colors->SetNumberOfComponents (3);
      colors->SetName ("Colors");
      colors->SetNumberOfTuples(static_cast<vtkIdType>(lineset.points_.size()));
    }

    points->SetNumberOfPoints(static_cast<vtkIdType>(lineset.points_.size()));
    for (std::size_t i = 0; i < lineset.points_.size(); ++i)
    {
        Eigen::Vector3d p = lineset.points_[i];
        points->SetPoint(static_cast<vtkIdType>(i), p.data());
    }

    for (std::size_t i = 0; i < lineset.lines_.size(); ++i)
    {
        vtkSmartPointer<vtkLine> segment = vtkSmartPointer<vtkLine>::New();
        Eigen::Vector2i segIndex = lineset.lines_[i];
        segment->GetPointIds()->SetId(0, segIndex(0));
        segment->GetPointIds()->SetId(1, segIndex(1));
        lines->InsertNextCell(segment);
        if (has_color)
        {
            ecvColor::Rgb color = ecvColor::Rgb::FromEigen(lineset.colors_[i]);
            colors->InsertTuple3(static_cast<vtkIdType>(i), color.r, color.g, color.b);
        }
    }

    return true;
}

vtkSmartPointer<vtkPolyData>
PclTools::CreateLine(vtkSmartPointer<vtkPoints> points)
{
	vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
	lineSource->SetPoints(points);
	lineSource->Update();
    return lineSource->GetOutput();
}

vtkSmartPointer<vtkPolyData> PclTools::CreateLine(vtkSmartPointer<vtkPoints> points,
                                                  vtkSmartPointer<vtkCellArray> lines,
                                                  vtkSmartPointer<vtkUnsignedCharArray> colors)
{
    if (points->GetNumberOfPoints() == 0 &&
        lines->GetNumberOfCells() == 0 &&
        colors->GetNumberOfTuples() == 0)
    {
        return nullptr;
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    if (points->GetNumberOfPoints() > 0)
    {
        polyData->SetPoints(points);
    }
    if (lines->GetNumberOfCells() > 0)
    {
        polyData->SetLines(lines);
    }
    if (colors->GetNumberOfTuples() > 0)
    {
        polyData->GetCellData()->SetScalars(colors);
    }
    return polyData;
}

void PclTools::SetPolyDataColor(vtkSmartPointer<vtkPolyData> polyData,
                                const ecvColor::Rgb& color, bool is_cell)
{
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
    colors->SetNumberOfComponents (3);
    colors->SetName("Colors");

    if (is_cell) {
        colors->SetNumberOfTuples(polyData->GetPolys()->GetNumberOfCells());
        for (vtkIdType i = 0; i < polyData->GetPolys()->GetNumberOfCells(); ++i) {
            colors->InsertTuple3(i, color.r, color.g, color.b);
        }
        polyData->GetCellData()->SetScalars(colors);
    } else {
        colors->SetNumberOfTuples(polyData->GetNumberOfPoints());
        for (vtkIdType i = 0; i < polyData->GetNumberOfPoints(); ++i) {
            colors->InsertTuple3(i, color.r, color.g, color.b);
        }
        polyData->GetPointData ()->SetScalars (colors);
    }
}

void PclTools::AddPolyDataCell(vtkSmartPointer<vtkPolyData> polyData)
{
    vtkSmartPointer<vtkCellArray> cellArray = vtkSmartPointer<vtkCellArray>::New ();
    vtkIdType n_points = polyData->GetNumberOfPoints();
    if (n_points == 8) {
        vtkIdType cellId1[3] = { 0, 1, 3 };
        cellArray->InsertNextCell(3, cellId1);
        vtkIdType cellId2[3] = { 7, 6, 4 };
        cellArray->InsertNextCell(3, cellId2);
        polyData->SetPolys(cellArray);
    } else if (n_points == 14) {
        vtkIdType cellId1[3] = { 0, 1, 3 };
        cellArray->InsertNextCell(3, cellId1);
        vtkIdType cellId2[3] = { 7, 6, 4 };
        cellArray->InsertNextCell(3, cellId2);

        vtkIdType cellId3[3] = { 8, 9, 11 };
        cellArray->InsertNextCell(3, cellId3);
        polyData->SetPolys(cellArray);
    }
}

vtkSmartPointer<vtkPolyData> PclTools::CreateCoordinateFromLineSet(const cloudViewer::geometry::LineSet &lineset)
{
    assert(lineset.lines_.size() == 3);

    // right vector
    vtkSmartPointer<vtkPoints> lineR = vtkSmartPointer<vtkPoints>::New();
    lineR->SetNumberOfPoints(2);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> segmentR = lineset.getLineCoordinate(0);
    lineR->SetPoint(static_cast<vtkIdType>(0), segmentR.first.data());
    lineR->SetPoint(static_cast<vtkIdType>(1), segmentR.second.data());
    vtkSmartPointer<vtkPolyData> rLinesData = CreateLine(lineR);
    SetPolyDataColor(rLinesData, ecvColor::red, false);

    // up vector
    vtkSmartPointer<vtkPoints> lineUp = vtkSmartPointer<vtkPoints>::New();
    lineUp->SetNumberOfPoints(2);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> segmentUp = lineset.getLineCoordinate(1);
    lineUp->SetPoint(static_cast<vtkIdType>(0), segmentUp.first.data());
    lineUp->SetPoint(static_cast<vtkIdType>(1), segmentUp.second.data());
    vtkSmartPointer<vtkPolyData> uLinesData = CreateLine(lineUp);
    SetPolyDataColor(uLinesData, ecvColor::yellow, false);

    // view vector
    vtkSmartPointer<vtkPoints> lineV = vtkSmartPointer<vtkPoints>::New();
    lineV->SetNumberOfPoints(2);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> segmentV = lineset.getLineCoordinate(2);
    lineV->SetPoint(static_cast<vtkIdType>(0), segmentV.first.data());
    lineV->SetPoint(static_cast<vtkIdType>(1), segmentV.second.data());
    vtkSmartPointer<vtkPolyData> vLinesData = CreateLine(lineV);
    SetPolyDataColor(vLinesData, ecvColor::green, false);

    vtkSmartPointer<vtkAppendPolyData> appendFilter =
            vtkSmartPointer<vtkAppendPolyData>::New();
    appendFilter->AddInputData(rLinesData);
    appendFilter->AddInputData(uLinesData);
    appendFilter->AddInputData(vLinesData);
    appendFilter->Update();
    return appendFilter->GetOutput();
}

vtkSmartPointer<vtkPolyData> PclTools::CreateGBLSensor(const ccGBLSensor *gBLSensor)
{
    assert(gBLSensor);
    auto linePoints = cloudViewer::geometry::LineSet::CreateFromOrientedBoundingBox(gBLSensor->getSensorHead());

    // sensor head lines
    vtkSmartPointer<vtkPolyData> headLinesData = CreatePolyDataFromLineSet(*linePoints, false);

    // sensor leg lines
    vtkSmartPointer<vtkPolyData> legLinesData = CreatePolyDataFromLineSet(gBLSensor->getSensorLegLines(), false);

    // sensor axis lines
    vtkSmartPointer<vtkPolyData> axisLinesData = CreatePolyDataFromLineSet(gBLSensor->getSensorAxis(), false);

    vtkSmartPointer<vtkAppendPolyData> appendFilter =
            vtkSmartPointer<vtkAppendPolyData>::New();
    appendFilter->AddInputData(headLinesData);
    appendFilter->AddInputData(legLinesData);
    appendFilter->AddInputData(axisLinesData);
    appendFilter->Update();
    return appendFilter->GetOutput();
}

vtkSmartPointer<vtkPolyData> PclTools::CreateCameraSensor(const ccCameraSensor *cameraSensor,
                                                          const ecvColor::Rgb& lineColor,
                                                          const ecvColor::Rgb& planeColor)
{
    assert(cameraSensor);

    // near plane
    vtkSmartPointer<vtkPolyData> nearPlaneLinesData = CreatePolyDataFromLineSet(cameraSensor->getNearPlane());
    AddPolyDataCell(nearPlaneLinesData);
    SetPolyDataColor(nearPlaneLinesData, planeColor, false);

    // side lines
    vtkSmartPointer<vtkPolyData> sideLinesData = CreatePolyDataFromLineSet(cameraSensor->getSideLines());
    SetPolyDataColor(sideLinesData, lineColor, false);

    // arrow lines
    vtkSmartPointer<vtkPolyData> arrowLinesData = CreatePolyDataFromLineSet(cameraSensor->getArrow());
    AddPolyDataCell(arrowLinesData);
    SetPolyDataColor(arrowLinesData, lineColor, false);

    // axis lines
    vtkSmartPointer<vtkPolyData> axisLinesData = CreateCoordinateFromLineSet(cameraSensor->getAxis());

    vtkSmartPointer<vtkAppendPolyData> appendFilter =
            vtkSmartPointer<vtkAppendPolyData>::New();
    appendFilter->AddInputData(nearPlaneLinesData);
    appendFilter->AddInputData(sideLinesData);
    appendFilter->AddInputData(arrowLinesData);
    appendFilter->AddInputData(axisLinesData);
    appendFilter->Update();
    return appendFilter->GetOutput();
}

vtkSmartPointer<vtkPolyData> PclTools::CreatePlane(const pcl::ModelCoefficients &coefficients)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    plane->SetNormal (coefficients.values[0], coefficients.values[1], coefficients.values[2]);

    double norm_sqr = coefficients.values[0] * coefficients.values[0]
                    + coefficients.values[1] * coefficients.values[1]
                    + coefficients.values[2] * coefficients.values[2];

    plane->Push (-coefficients.values[3] / sqrt(norm_sqr));
    plane->Update ();
    return (plane->GetOutput ());
}

vtkSmartPointer<vtkPolyData> PclTools::CreatePlane(const pcl::ModelCoefficients &coefficients,
                                                double x, double y, double z, double scale)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();

    double norm_sqr = 1.0 / (coefficients.values[0] * coefficients.values[0] +
                             coefficients.values[1] * coefficients.values[1] +
                             coefficients.values[2] * coefficients.values[2]);

    plane->SetNormal(coefficients.values[0], coefficients.values[1],
                     coefficients.values[2]);
    double t = x * coefficients.values[0] + y * coefficients.values[1] +
               z * coefficients.values[2] + coefficients.values[3];
    x -= coefficients.values[0] * t * norm_sqr;
    y -= coefficients.values[1] * t * norm_sqr;
    z -= coefficients.values[2] * t * norm_sqr;

    plane->SetCenter (x, y, z);
    plane->Update ();

    Eigen::Vector3d p1,p2;
    Eigen::Vector3d n;
    n.x() = x + coefficients.values[0];
    n.y() = y + coefficients.values[1];
    n.z() = z + coefficients.values[2];

    p1.x() = x + 1.0; //
    p1.y() = y;
    p1.z() = (coefficients.values[3] - coefficients.values[0]*p1.x() - coefficients.values[1]*p1.y())/coefficients.values[2];

    p2 = n.cross(p1);

    p1.normalize();
    p2.normalize();

    p1 = p1*scale;
    p2 = p2*scale;

    double point1[3];
    double point2[3];

    point1[0] = p1.x();
    point1[1] = p1.y();
    point1[2] = p1.z();

    point2[0] = p2.x();
    point2[1] = p2.y();
    point2[2] = p2.z();

    plane->SetOrigin(x, y, z);
    plane->SetPoint1(point1);
    plane->SetPoint2(point2);

    plane->Update();

    return plane->GetOutput();
}


vtkSmartPointer<vtkPropAssembly> PclTools::CreateCoordinate(double axesLength,
                                                            const std::string& xLabel,
                                                            const std::string& yLabel,
                                                            const std::string& zLabel,
                                                            const std::string& xPlus,
                                                            const std::string& xMinus,
                                                            const std::string& yPlus,
                                                            const std::string& yMinus,
                                                            const std::string& zPlus,
                                                            const std::string& zMinus)
{
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    axes->SetShaftTypeToCylinder();
    axes->SetXAxisLabelText(xLabel.c_str());
    axes->SetYAxisLabelText(yLabel.c_str());
    axes->SetZAxisLabelText(zLabel.c_str());
    axes->GetXAxisTipProperty()->SetColor(1.0, 0.0, 0.0);
    axes->GetXAxisShaftProperty()->SetColor(1.0, 0.0, 0.0);
    axes->GetYAxisTipProperty()->SetColor(1.0, 1.0, 0.0);
    axes->GetYAxisShaftProperty()->SetColor(1.0, 1.0, 0.0);
    axes->GetZAxisTipProperty()->SetColor(0.0, 1.0, 0.0);
    axes->GetZAxisShaftProperty()->SetColor(0.0, 1.0, 0.0);
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
    cube->SetFaceTextScale(0.65);
    cube->GetCubeProperty()->SetColor(0.5, 1, 1);
    cube->GetTextEdgesProperty()->SetLineWidth(1);
    cube->GetTextEdgesProperty()->SetDiffuse(0);
    cube->GetTextEdgesProperty()->SetAmbient(1);
    cube->GetTextEdgesProperty()->SetColor(0.1800, 0.2800, 0.2300);
    // this static function improves the appearance of the text edges
    // since they are overlaid on a surface rendering of the cube's faces
    vtkMapper::SetResolveCoincidentTopologyToPolygonOffset();

    cube->GetXPlusFaceProperty()->SetColor(1.0, 0.0, 0.0);
    cube->GetXPlusFaceProperty()->SetInterpolationToFlat();
    cube->GetXMinusFaceProperty()->SetColor(1.0, 0.0, 0.0);
    cube->GetXMinusFaceProperty()->SetInterpolationToFlat();
    cube->GetYPlusFaceProperty()->SetColor(1.0, 1.0, 0.0);
    cube->GetYPlusFaceProperty()->SetInterpolationToFlat();
    cube->GetYMinusFaceProperty()->SetColor(1.0, 1.0, 0.0);
    cube->GetYMinusFaceProperty()->SetInterpolationToFlat();
    cube->GetZPlusFaceProperty()->SetColor(0.0, 1.0, 0.0);
    cube->GetZPlusFaceProperty()->SetInterpolationToFlat();
    cube->GetZMinusFaceProperty()->SetColor(0.0, 1.0, 0.0);
    cube->GetZMinusFaceProperty()->SetInterpolationToFlat();

    vtkSmartPointer<vtkTextProperty> tprop4 = vtkSmartPointer<vtkTextProperty>::New();
    tprop4->ShadowOn();
    tprop4->SetFontFamilyToArial();

    // tprop.SetFontFamilyToTimes();
    axes->GetXAxisCaptionActor2D()->SetCaptionTextProperty(tprop4);
    //
    vtkSmartPointer<vtkTextProperty> tprop2 = vtkSmartPointer<vtkTextProperty>::New();
    tprop2->ShallowCopy(tprop4);
    axes->GetYAxisCaptionActor2D()->SetCaptionTextProperty(tprop2);
    //
    vtkSmartPointer<vtkTextProperty> tprop3 = vtkSmartPointer<vtkTextProperty>::New();
    tprop3->ShallowCopy(tprop4);
    axes->GetZAxisCaptionActor2D()->SetCaptionTextProperty(tprop3);

    vtkSmartPointer<vtkPropAssembly> assembly = vtkSmartPointer<vtkPropAssembly>::New();
    assembly->AddPart(cube);
    assembly->AddPart(axes);
    return assembly;
}

//For log scale inversion
const double c_log10 = log(10.0);

//structure for recursive display of labels
struct vlabel
{
	int yPos; 		/**< label center pos **/
	int yMin; 		/**< label 'ROI' min **/
	int yMax; 		/**< label 'ROI' max **/
	double val; 	/**< label value **/

	//default constructor
	vlabel(int y, int y1, int y2, double v) : yPos(y), yMin(y1), yMax(y2), val(v) { assert(y2 >= y1); }
};

//! A set of 'vlabel' structures
using vlabelSet = std::list<vlabel>;

//Convert standard range to log scale
void ConvertToLogScale(ScalarType& dispMin, ScalarType& dispMax)
{
	ScalarType absDispMin = (dispMax < 0 ? std::min(-dispMax, -dispMin) : std::max<ScalarType>(dispMin, 0));
	ScalarType absDispMax = std::max(std::abs(dispMin), std::abs(dispMax));
	dispMin = std::log10(std::max(absDispMin, std::numeric_limits<ScalarType>::epsilon()));
	dispMax = std::log10(std::max(absDispMax, std::numeric_limits<ScalarType>::epsilon()));
}

//helper: returns the neighbouring labels at a given position
//(first: above label, second: below label)
//Warning: set must be already sorted!
using vlabelPair = std::pair<vlabelSet::iterator, vlabelSet::iterator>;

static vlabelPair GetVLabelsAround(int y, vlabelSet& set)
{
	if (set.empty())
	{
		return vlabelPair(set.end(), set.end());
	}
	else
	{
		vlabelSet::iterator it1 = set.begin();
		if (y < it1->yPos)
		{
			return vlabelPair(set.end(), it1);
		}
		vlabelSet::iterator it2 = it1; ++it2;
		for (; it2 != set.end(); ++it2, ++it1)
		{
			if (y <= it2->yPos) // '<=' to make sure the last label stays at the top!
				return vlabelPair(it1, it2);
		}
		return vlabelPair(it1, set.end());
	}
}

bool PclTools::UpdateScalarBar(vtkAbstractWidget* widget, const CC_DRAW_CONTEXT & CONTEXT)
{
    if (!widget) return false;
    std::string viewID = CVTools::FromQString(CONTEXT.viewID);
    const ccScalarField* sf = CONTEXT.sfColorScaleToDisplay;
    if (!sf || !sf->getColorScale())
    {
        return false;
    }

    bool logScale = sf->logScale();
    bool symmetricalScale = sf->symmetricalScale();
    bool alwaysShowZero = sf->isZeroAlwaysShown();

    //set of particular values
    //DGM: we work with doubles for maximum accuracy
    ccColorScale::LabelSet keyValues;
    bool customLabels = false;
    try
    {
        ccColorScale::Shared colorScale = sf->getColorScale();
        if (colorScale && colorScale->customLabels().size() >= 2)
        {
            keyValues = colorScale->customLabels();

            if (alwaysShowZero)
            {
                keyValues.insert(0.0);
            }

            customLabels = true;
        }
        else if (!logScale)
        {
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

            if (alwaysShowZero)
                keyValues.insert(0.0);
        }
        else
        {
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
    }
    catch (const std::bad_alloc&)
    {
        //not enough memory
        return false;
    }

    //magic fix (for infinite values!)
    {
        for (ccColorScale::LabelSet::iterator it = keyValues.begin(); it != keyValues.end(); ++it)
        {
#if defined(CV_WINDOWS) && defined(_MSC_VER)
            if (!_finite(*it))
#else
            if (!std::isfinite(*it))
#endif
            {
                bool minusInf = (*it < 0);
                keyValues.erase(it);
                if (minusInf)
                    keyValues.insert(-std::numeric_limits<ScalarType>::max());
                else
                    keyValues.insert(std::numeric_limits<ScalarType>::max());
                it = keyValues.begin(); //restart the process (easier than trying to be intelligent here ;)
            }
        }
    }

    // Internally, the elements in a set are already sorted
    // std::sort(keyValues.begin(), keyValues.end());

    if (!customLabels && !sf->areNaNValuesShownInGrey())
    {
        //remove 'hidden' values
        if (!logScale)
        {
            for (ccColorScale::LabelSet::iterator it = keyValues.begin(); it != keyValues.end(); )
            {
                if (!sf->displayRange().isInRange(static_cast<ScalarType>(*it)) && (!alwaysShowZero || *it != 0)) //we keep zero if the user has explicitely asked for it!
                {
                    ccColorScale::LabelSet::iterator toDelete = it;
                    ++it;
                    keyValues.erase(toDelete);
                }
                else
                {
                    ++it;
                }
            }
        }
        else
        {
            //convert actual display range to log scale
            //(we can't do the opposite, otherwise we get accuracy/round-off issues!)
            ScalarType dispMin = sf->displayRange().start();
            ScalarType dispMax = sf->displayRange().stop();
            ConvertToLogScale(dispMin, dispMax);

            for (ccColorScale::LabelSet::iterator it = keyValues.begin(); it != keyValues.end(); )
            {
                if (*it >= dispMin && *it <= dispMax)
                {
                    ++it;
                }
                else
                {
                    ccColorScale::LabelSet::iterator toDelete = it;
                    ++it;
                    keyValues.erase(toDelete);
                }
            }
        }
    }

    std::vector<double> sortedKeyValues(keyValues.begin(), keyValues.end());
    double maxRange = sortedKeyValues.back() - sortedKeyValues.front();


    const ecvGui::ParamStruct& displayParams = ecvDisplayTools::GetDisplayParameters();
    //default color: text color
    const ecvColor::Rgbub& textColor = displayParams.textDefaultCol;
    //histogram?
    const ccScalarField::Histogram histogram = sf->getHistogram();
    bool showHistogram = (displayParams.colorScaleShowHistogram && !logScale && histogram.maxValue != 0 && histogram.size() > 1);

    //display area
    float renderZoom = CONTEXT.renderZoom;
    QFont font = ecvDisplayTools::GetTextDisplayFont(); //takes rendering zoom into account!
    const int strHeight = static_cast<int>(displayParams.defaultFontSize * renderZoom); //QFontMetrics(font).height() --> always returns the same value?!
    const int scaleWidth = static_cast<int>(displayParams.colorScaleRampWidth * renderZoom);
    const int scaleMaxHeight = (keyValues.size() > 1 ?
        std::max(CONTEXT.glH - static_cast<int>(140 * renderZoom), 2 * strHeight) : scaleWidth); //if 1 value --> we draw a cube

    //list of labels to draw
    vlabelSet drawnLabels;
    {
        //add first label
        drawnLabels.emplace_back(0, 0, strHeight, sortedKeyValues.front());

        if (keyValues.size() > 1)
        {
            //add last label
            drawnLabels.emplace_back(scaleMaxHeight, scaleMaxHeight - strHeight, scaleMaxHeight, sortedKeyValues.back());
        }

        //we try to display the other keyPoints (if any)
        if (keyValues.size() > 2)
        {
            assert(maxRange > 0.0);
            const int minGap = strHeight;
            for (size_t i = 1; i < keyValues.size() - 1; ++i)
            {
                int yScale = static_cast<int>((sortedKeyValues[i] - sortedKeyValues[0]) * scaleMaxHeight / maxRange);
                vlabelPair nLabels = GetVLabelsAround(yScale, drawnLabels);

                assert(nLabels.first != drawnLabels.end() && nLabels.second != drawnLabels.end());
                if ((nLabels.first == drawnLabels.end() || nLabels.first->yMax <= yScale - minGap)
                    && (nLabels.second == drawnLabels.end() || nLabels.second->yMin >= yScale + minGap))
                {
                    //insert it at the right place (so as to keep a sorted list!)
                    drawnLabels.insert(nLabels.second, vlabel(yScale, yScale - strHeight / 2, yScale + strHeight / 2, sortedKeyValues[i]));
                }
            }
        }

        //now we recursively display labels for which we have some room left
        if (!customLabels && drawnLabels.size() > 1)
        {
            const int minGap = strHeight * 2;

            size_t drawnLabelsBefore = 0; //just to init the loop
            size_t drawnLabelsAfter = drawnLabels.size();

            //proceed until no more label can be inserted
            while (drawnLabelsAfter > drawnLabelsBefore)
            {
                drawnLabelsBefore = drawnLabelsAfter;

                vlabelSet::iterator it1 = drawnLabels.begin();
                vlabelSet::iterator it2 = it1; ++it2;
                for (; it2 != drawnLabels.end(); ++it2)
                {
                    if (it1->yMax + 2 * minGap < it2->yMin)
                    {
                        //insert label
                        double val = (it1->val + it2->val) / 2.0;
                        int yScale = static_cast<int>((val - sortedKeyValues[0]) * scaleMaxHeight / maxRange);

                        //insert it at the right place (so as to keep a sorted list!)
                        drawnLabels.insert(it2, vlabel(yScale, yScale - strHeight / 2, yScale + strHeight / 2, val));
                    }
                    it1 = it2;
                }

                drawnLabelsAfter = drawnLabels.size();
            }
        }

    }

    // start draw scalar bar!
    vtkScalarBarWidgetCustom* scalarBarWidget = vtkScalarBarWidgetCustom::SafeDownCast(widget);
    if (!scalarBarWidget)
    {
        return false;
    }
    vtkContext2DScalarBarActor* lutActor = vtkContext2DScalarBarActor::SafeDownCast(scalarBarWidget->GetScalarBarActor());
    if (!lutActor)
    {
        scalarBarWidget->Off();
        return false;
    }

    vtkScalarBarRepresentationCustom* rep = vtkScalarBarRepresentationCustom::SafeDownCast(scalarBarWidget->GetRepresentation());
    if (rep)
    {
        rep->SetWindowLocation(vtkScalarBarRepresentationCustom::LowerRightCorner);
    }

    VTK_CREATE(vtkDiscretizableColorTransferFunctionCustom, lut);
    {
        //lut->SetNumberOfColors(scaleMaxHeight);
        //lut->SetNodeValue();
        //lut->SetNumberOfTableValues(static_cast<vtkIdType>(scaleMaxHeight));
        //lut->SetTableRange(sortedKeyValues.front(), sortedKeyValues.back());
        //lut->SetNumberOfIndexedColors(scaleMaxHeight);
        //lut->SetUseAboveRangeColor(1);
        //lut->SetUseBelowRangeColor(1);
        //lut->SetDiscretize(1);
        lut->Build();
        //if (logScale)
        //{
        //	lut->SetScaleToLog10();
        //}
        //else
        //{
        //	lut->SetScaleToLinear();
        //}

        for (int j = 0; j < scaleMaxHeight; ++j)
        {
            double baseValue = sortedKeyValues.front() + (j * maxRange) / scaleMaxHeight;
            double value = baseValue;
            if (logScale)
            {
                value = std::exp(value*c_log10);
            }
            const ecvColor::Rgb* col = sf->getColor(static_cast<ScalarType>(value));
            if (!col)
            {
                //special case: if we have user-defined labels, we want all the labels to be displayed with their associated color
                if (customLabels)
                {
                    assert(sf->getColorScale() && !sf->getColorScale()->isRelative());
                    col = sf->getColorScale()->getColorByValue(value, &ecvColor::lightGrey);
                }
                else
                {
                    col = &ecvColor::lightGrey;
                }
            }
            assert(col);
            Eigen::Vector3d rgb = ecvColor::Rgb::ToEigen(*col);
            //lut->SetTableValue(j, rgb(0), rgb(1), rgb(2), 1);
            lut->AddRGBPoint(value, rgb(0), rgb(1), rgb(2));
        }

    }

    // Scalar field name
    const char* sfName = sf->getName();
    QString sfTitle(sfName);
    if (sfName)
    {
        if (sf->getGlobalShift() != 0)
            sfTitle += QString("[Shifted]");
        if (logScale)
            sfTitle += QString("[Log scale]");
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
    //lutActor->SetNumberOfLabels(20);
    lutActor->SetNumberOfTicks(static_cast<int>(drawnLabels.size()));
    //lutActor->SetAutomaticLabelFormat(1);
    //lutActor->SetAutomaticAnnotations(1)
    //lutActor->SetAddRangeAnnotations(1);
    lutActor->SetTextPositionToPrecedeScalarBar();
    //lutActor->SetTextPositionToSucceedScalarBar();
    lutActor->GetLabelTextProperty()->SetFontSize(static_cast<int>(displayParams.defaultFontSize * renderZoom));
    lutActor->SetDrawFrame(1);
    Eigen::Vector3d col = ecvColor::Rgb::ToEigen(textColor);
    lutActor->GetFrameProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetAnnotationTextProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetTitleTextProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetLabelTextProperty()->SetColor(col(0), col(1), col(2));
    lutActor->GetFrameProperty()->SetLineWidth(2.0f * renderZoom);
    //lutActor->SetDrawBackground(1);
    //lutActor->GetBackgroundProperty()->SetColor(1., 1., 1.);
    //precision (same as color scale)
    const unsigned precision = displayParams.displayedNumPrecision;
    //format
    const char format = (sf->logScale() ? 'E' : 'f');
    QString formatInfo = QString("%1.%2%3").arg(precision).arg(precision).arg(format);
    const std::string labelFormat = "%" + formatInfo.toStdString();
    lutActor->SetRangeLabelFormat(labelFormat.c_str());
    scalarBarWidget->SetScalarBarActor(lutActor);
    scalarBarWidget->On();
    scalarBarWidget->Modified();
    return true;
}

bool PclTools::TransformPolyData(vtkSmartPointer<vtkPolyData> polyData, const ccGLMatrixd &trans)
{
    if (!polyData)
    {
        return false;
    }
    vtkSmartPointer<vtkPoints> points = polyData->GetPoints();
    if(!points || points->GetNumberOfPoints() == 0)
    {
        return false;
    }

    return TransformVtkPoints(points, trans);

}

bool PclTools::TransformVtkPoints(vtkSmartPointer<vtkPoints> points, const ccGLMatrixd &trans)
{
    if(!points || points->GetNumberOfPoints() == 0)
    {
        return false;
    }

    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
        double * P = points->GetPoint(i);
        trans.apply(P);
    }

    return true;
}

vtkSmartPointer<vtkPolyData> PclTools::CreateCube(double width, double height, double depth, const ccGLMatrixd &trans)
{
    vtkSmartPointer<vtkPolyData> data = CreateCube(width, height, depth);
    if (!TransformPolyData(data, trans))
    {
        CVLog::Error("[PclTools::CreateCube] Creating cube failed!");
        return nullptr;
    }
    return data;
}

vtkSmartPointer<vtkPolyData> PclTools::CreateCube(double width, double height, double depth)
{
    vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New ();
    cube->SetXLength (width);
    cube->SetYLength (height);
    cube->SetZLength (depth);
    cube->Update();
    return cube->GetOutput();
}

vtkSmartPointer<vtkPolyData> PclTools::CreatePolyDataFromLineSet(const cloudViewer::geometry::LineSet &lineset, bool useLineSource/* = true*/)
{
    if (useLineSource) {
        return CreateLine(GetVtkPointsFromLineSet(lineset));
    } else {
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
        vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
        if (GetVtkPointsAndLinesFromLineSet(lineset, points, lines, colors)) {
            return CreateLine(points, lines, colors);
        } else {
            return nullptr;
        }
    }
}
