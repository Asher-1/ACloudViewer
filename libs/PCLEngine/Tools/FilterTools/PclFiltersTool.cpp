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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#include "PclFiltersTool.h" 

//Local
#include "PclUtils/cc2sm.h"
#include "PclUtils/PCLVis.h"
#include "PclUtils/PCLConv.h"
#include "Tools/ecvTools.h"

// Vtk Filters
#include "cvGenericFilter.h"
#include "cvClipFilter.h"
#include "cvSliceFilter.h"
#include "cvDecimateFilter.h"
#include "cvIsoSurfaceFilter.h"
#include "cvThresholdFilter.h"
#include "cvSmoothFilter.h"
#include "cvProbeFilter.h"
#include "cvStreamlineFilter.h"
#include "cvGlyphFilter.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVGeom.h>
#include <CVTools.h>

// CV_DB_LIB
#include <ecvHObject.h>
#include <ecvDisplayTools.h>

// VTK
#include <vtkRenderWindowInteractor.h>

// QT
#include <QDir>
#include <QFileInfo>

#define DEFAULT_POINT 0
#define SELECTED_POINT -2
#define GROUND_POINT -1

using namespace PclUtils;

PclFiltersTool::PclFiltersTool(FilterType type)
	: ecvGenericFiltersTool(type)
	, m_filter(nullptr)
{
	//this->registerFilter();
}

PclFiltersTool::PclFiltersTool(ecvGenericVisualizer3D* viewer, FilterType type)
	: ecvGenericFiltersTool(type)
	, m_filter(nullptr)
{
	this->initialize(viewer);
	//this->registerFilter();
}

PclFiltersTool::~PclFiltersTool()
{
	if (m_filter)
	{
		delete m_filter;
		m_filter = nullptr;
	}
}

////////////////////Initialization///////////////////////////
void PclFiltersTool::initialize(ecvGenericVisualizer3D* viewer)
{
	assert(viewer);
	this->setVisualizer(viewer);

	switch (m_filterType)
	{
	case ecvGenericFiltersTool::CLIP_FILTER:
		m_filter = new cvClipFilter();
		break;
	case ecvGenericFiltersTool::SLICE_FILTER:
		m_filter = new cvSliceFilter();
		break;
	case ecvGenericFiltersTool::DECIMATE_FILTER:
		m_filter = new cvDecimateFilter();
		break;
	case ecvGenericFiltersTool::ISOSURFACE_FILTER:
		m_filter = new cvIsoSurfaceFilter();
		break;
	case ecvGenericFiltersTool::THRESHOLD_FILTER:
		m_filter = new cvThresholdFilter();
		break;
	case ecvGenericFiltersTool::SMOOTH_FILTER:
		m_filter = new cvSmoothFilter();
		break;
	case ecvGenericFiltersTool::PROBE_FILTER:
		m_filter = new cvProbeFilter();
		break;
	case ecvGenericFiltersTool::STREAMLINE_FILTER:
		m_filter = new cvStreamlineFilter();
		break;
	case ecvGenericFiltersTool::GLYPH_FILTER:
		m_filter = new cvGlyphFilter();
		break;
	default:
		CVLog::Error(QString("unknown filter type"));
		break;
	}

	if (m_filter && m_viewer)
	{
		m_filter->setUpViewer(m_viewer);
	}

	resetMode();
	m_cloudLabel = nullptr;
}

void PclFiltersTool::setVisualizer(ecvGenericVisualizer3D* viewer)
{
	if (viewer)
	{
		m_viewer = reinterpret_cast<PCLVis*>(viewer);
		if (!m_viewer)
		{
			CVLog::Warning("[PclAnnotationTool::setVisualizer] viewer is Null!");
		}
	}
	else
	{
		CVLog::Warning("[PclFiltersTool::setVisualizer] viewer is Null!");
	}
}

void PclFiltersTool::showInteractor(bool state)
{
	if (!m_filter) return;
	m_filter->showInteractor(state);
	update();
}

void PclFiltersTool::showOutline(bool state)
{
	if (!m_filter) return;
	m_filter->showOutline(state);
}

ccHObject * PclFiltersTool::getOutput() const
{
	if (!m_filter) return nullptr;
	return m_filter->getOutput();
}

void PclFiltersTool::getOutput(
	std::vector<ccHObject*>& outputSlices, 
	std::vector<ccPolyline*>& outputContours) const
{
	if (!m_filter) return;
	m_filter->getOutput(outputSlices, outputContours);
}

void PclFiltersTool::setNegative(bool state)
{
	if (!m_filter) return;
	m_filter->setNegative(state);
}

QWidget * PclFiltersTool::getFilterWidget()
{
	return m_filter->topLevelWidget();
}

const ccBBox & PclFiltersTool::getBox()
{
	if (!m_filter) return m_box;
	m_filter->getInteractorBounds(m_box);
	return m_box;
}

void PclFiltersTool::setBox(const ccBBox & box)
{
	m_box = box;
}

void PclFiltersTool::shift(const CCVector3 & v)
{
	if (!m_filter) return;

	m_filter->shift(CCVector3d::fromArray(v.u));
}

void PclFiltersTool::set(const ccBBox & extents, const ccGLMatrix & transformation)
{
}

void PclFiltersTool::get(ccBBox & extents, ccGLMatrix & transformation)
{
	if (!m_filter) return;
	ccGLMatrixd trans;
	m_filter->getInteractorInfos(extents, trans);
	transformation = ccGLMatrix(trans.data());
}

bool PclFiltersTool::setInputData(ccHObject* entity, int viewPort)
{
	m_associatedEntity = entity;
	if (!m_filter) return false;
	return m_filter->setInput(entity);
}

bool PclFiltersTool::start()
{

	if (!m_filter) return false;
	m_filter->apply();
	m_filter->start();
	return true;
}

void PclFiltersTool::registerFilter()
{
	//if (m_viewer)
	//{
	//	m_viewer->setInteractorEnabled(true);
	//	connect(m_viewer, &PCLVis::interactorPickedEvent, this, &PclFiltersTool::pickedEventProcess);
	//	connect(m_viewer, &PCLVis::interactorKeyboardEvent, this, &PclFiltersTool::keyboardEventProcess);
	//	connect(m_viewer, &PCLVis::interactorAreaPickedEvent, this, &PclFiltersTool::areaPickingEventProcess);
	//}
	
}

void PclFiltersTool::unregisterFilter()
{
	//if (m_viewer)
	//{
	//	m_viewer->setInteractorEnabled(false);
	//	disconnect(m_viewer, &PCLVis::interactorPickedEvent, this, &PclFiltersTool::pickedEventProcess);
	//	disconnect(m_viewer, &PCLVis::interactorKeyboardEvent, this, &PclFiltersTool::keyboardEventProcess);
	//	disconnect(m_viewer, &PCLVis::interactorAreaPickedEvent, this, &PclFiltersTool::areaPickingEventProcess);
	//}

	resetMode();
	clear();
	update();
}

//////////////////////////////////////////////////////////////

////////////////////Register callback////////////////////////
void PclFiltersTool::intersectMode()
{
	m_intersectMode = true;
	m_unionMode = false;
	m_trimMode = false;
}

void PclFiltersTool::unionMode()
{
	m_intersectMode = false;
	m_unionMode = true;
	m_trimMode = false;
}

void PclFiltersTool::trimMode()
{
	m_intersectMode = false;
	m_unionMode = false;
	m_trimMode = true;
}

void PclFiltersTool::resetMode()
{
	m_intersectMode = false;
	m_unionMode = false;
	m_trimMode = false;
}

////////////////////Callback function///////////////////////////
void PclFiltersTool::pointPickingProcess(int index)
{

}

void PclFiltersTool::areaPickingEventProcess(const std::vector<int>& new_selected_slice)
{
	if (new_selected_slice.empty() || !m_viewer) return;

	int s = m_viewer->getRenderWindowInteractor()->GetShiftKey();
	int a = m_viewer->getRenderWindowInteractor()->GetControlKey();

	// remove ground points
	vector<int> selected_slice;
	for (auto x : new_selected_slice) {
		if (m_cloudLabel[x] != GROUND_POINT) {
			selected_slice.push_back(x);
		}
	}

	if (s && a || m_intersectMode) { // intersection
		m_last_selected_slice = ecvTools::IntersectionVector(m_last_selected_slice, selected_slice);
	}
	else if (s || m_unionMode) { // union
		m_last_selected_slice = ecvTools::UnionVector(m_last_selected_slice, selected_slice);
	}
	else if (a || m_trimMode) { // remove
		m_last_selected_slice = ecvTools::DiffVector(m_last_selected_slice, selected_slice);
	}
	else { // new
		m_last_selected_slice = selected_slice;
	}

	update();
}

void PclFiltersTool::pickedEventProcess(vtkActor* actor) 
{
}

void PclFiltersTool::keyboardEventProcess(const std::string& symKey)
{
	// delete annotation
	if (symKey == "Delete") {

	}
}
///////////////////////////////////////////////////////////////

void PclFiltersTool::setPointSize(const std::string & viewID, int viewPort)
{
	m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, viewID, viewPort);
}

void PclFiltersTool::reset()
{
	if (!m_filter) return;
	m_filter->reset();
}

void PclFiltersTool::restore()
{
	if (!m_filter) return;
	m_filter->restoreOrigin();
}

void PclFiltersTool::clear()
{
	if (m_filter)
	{
		m_filter->clearAllActor();
	}
}