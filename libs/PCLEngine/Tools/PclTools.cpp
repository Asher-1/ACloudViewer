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
#include <ecvColorScale.h>
#include <ecvScalarField.h>
#include <ecvDisplayTools.h>

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
#include <pcl/visualization/vtk/vtkVertexBufferObjectMapper.h>

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

vtkSmartPointer<vtkDataSet>
PclTools::CreateLine(vtkSmartPointer<vtkPoints> points)
{
	vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
	lineSource->SetPoints(points);
	lineSource->Update();
	return (lineSource->GetOutput());
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
	std::string viewID = CVTools::fromQString(CONTEXT.viewID);
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
	lutActor->SetTitle(CVTools::fromQString(sfTitle).c_str());
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
	//lutActor->SetAutomaticAnnotations(1)£»
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
