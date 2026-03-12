// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file VtkFiltersTool.cpp
 * @brief Implementation of VTK filter tools pipeline.
 */

#include "VtkFiltersTool.h"
#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// Local
#include "Tools/Common/ecvTools.h"
#include "Visualization/VtkVis.h"

// Vtk Filters
#include "cvClipFilter.h"
#include "cvDecimateFilter.h"
#include "cvGenericFilter.h"
#include "cvGlyphFilter.h"
#include "cvIsoSurfaceFilter.h"
#include "cvProbeFilter.h"
#include "cvSliceFilter.h"
#include "cvSmoothFilter.h"
#include "cvStreamlineFilter.h"
#include "cvThresholdFilter.h"

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVLog.h>
#include <CVTools.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvHObject.h>

// VTK
#include <vtkRenderWindowInteractor.h>

// QT
#include <QDir>
#include <QFileInfo>

#define DEFAULT_POINT 0
#define SELECTED_POINT -2
#define GROUND_POINT -1

using namespace Visualization;

VtkFiltersTool::VtkFiltersTool(FilterType type)
    : ecvGenericFiltersTool(type), m_filter(nullptr) {}

VtkFiltersTool::VtkFiltersTool(ecvGenericVisualizer3D* viewer, FilterType type)
    : ecvGenericFiltersTool(type), m_filter(nullptr) {
    this->initialize(viewer);
}

VtkFiltersTool::~VtkFiltersTool() {
    if (m_filter) {
        delete m_filter;
        m_filter = nullptr;
    }
}

////////////////////Initialization///////////////////////////
void VtkFiltersTool::initialize(ecvGenericVisualizer3D* viewer) {
    assert(viewer);
    this->setVisualizer(viewer);

    switch (m_filterType) {
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

    if (m_filter && m_viewer) {
        m_filter->setUpViewer(m_viewer);
    }

    resetMode();
    m_cloudLabel = nullptr;
}

void VtkFiltersTool::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (viewer) {
        m_viewer = reinterpret_cast<VtkVis*>(viewer);
        if (!m_viewer) {
            CVLog::Warning("[VtkFiltersTool::setVisualizer] viewer is Null!");
        }
    } else {
        CVLog::Warning("[VtkFiltersTool::setVisualizer] viewer is Null!");
    }
}

void VtkFiltersTool::showInteractor(bool state) {
    if (!m_filter) return;
    m_filter->showInteractor(state);
    update();
}

void VtkFiltersTool::showOutline(bool state) {
    if (!m_filter) return;
    m_filter->showOutline(state);
}

ccHObject* VtkFiltersTool::getOutput() const {
    if (!m_filter) return nullptr;
    return m_filter->getOutput();
}

void VtkFiltersTool::getOutput(std::vector<ccHObject*>& outputSlices,
                               std::vector<ccPolyline*>& outputContours) const {
    if (!m_filter) return;
    m_filter->getOutput(outputSlices, outputContours);
}

void VtkFiltersTool::setNegative(bool state) {
    if (!m_filter) return;
    m_filter->setNegative(state);
}

QWidget* VtkFiltersTool::getFilterWidget() {
    return m_filter->topLevelWidget();
}

const ccBBox& VtkFiltersTool::getBox() {
    if (!m_filter) return m_box;
    m_filter->getInteractorBounds(m_box);
    return m_box;
}

void VtkFiltersTool::setBox(const ccBBox& box) { m_box = box; }

void VtkFiltersTool::shift(const CCVector3& v) {
    if (!m_filter) return;

    m_filter->shift(CCVector3d::fromArray(v.u));
}

void VtkFiltersTool::set(const ccBBox& extents,
                         const ccGLMatrix& transformation) {}

void VtkFiltersTool::get(ccBBox& extents, ccGLMatrix& transformation) {
    if (!m_filter) return;
    ccGLMatrixd trans;
    m_filter->getInteractorInfos(extents, trans);
    transformation = ccGLMatrix(trans.data());
}

bool VtkFiltersTool::setInputData(ccHObject* entity, int viewport) {
    m_associatedEntity = entity;
    if (!m_filter) return false;
    return m_filter->setInput(entity);
}

bool VtkFiltersTool::start() {
    if (!m_filter) return false;
    m_filter->apply();
    m_filter->start();
    return true;
}

void VtkFiltersTool::unregisterFilter() {
    resetMode();
    clear();
    update();
}

//////////////////////////////////////////////////////////////

////////////////////Register callback////////////////////////
void VtkFiltersTool::intersectMode() {
    m_intersectMode = true;
    m_unionMode = false;
    m_trimMode = false;
}

void VtkFiltersTool::unionMode() {
    m_intersectMode = false;
    m_unionMode = true;
    m_trimMode = false;
}

void VtkFiltersTool::trimMode() {
    m_intersectMode = false;
    m_unionMode = false;
    m_trimMode = true;
}

void VtkFiltersTool::resetMode() {
    m_intersectMode = false;
    m_unionMode = false;
    m_trimMode = false;
}

////////////////////Callback function///////////////////////////
void VtkFiltersTool::areaPickingEventProcess(
        const std::vector<int>& new_selected_slice) {
    if (new_selected_slice.empty() || !m_viewer) return;

    int s = m_viewer->getRenderWindowInteractor()->GetShiftKey();
    int a = m_viewer->getRenderWindowInteractor()->GetControlKey();

    // remove ground points
    std::vector<int> selected_slice;
    for (auto x : new_selected_slice) {
        if (m_cloudLabel[x] != GROUND_POINT) {
            selected_slice.push_back(x);
        }
    }

    if ((s && a) || m_intersectMode) {  // intersection
        m_last_selected_slice = ecvTools::IntersectionVector(
                m_last_selected_slice, selected_slice);
    } else if (s || m_unionMode) {  // union
        m_last_selected_slice =
                ecvTools::UnionVector(m_last_selected_slice, selected_slice);
    } else if (a || m_trimMode) {  // remove
        m_last_selected_slice =
                ecvTools::DiffVector(m_last_selected_slice, selected_slice);
    } else {  // new
        m_last_selected_slice = selected_slice;
    }

    update();
}

///////////////////////////////////////////////////////////////

void VtkFiltersTool::setPointSize(const std::string& viewID, int viewport) {
    m_viewer->setPointCloudRenderingProperties(
            Visualization::VtkVis::PCL_VISUALIZER_POINT_SIZE, 5, viewID,
            viewport);
}

void VtkFiltersTool::reset() {
    if (!m_filter) return;
    m_filter->reset();
}

void VtkFiltersTool::restore() {
    if (!m_filter) return;
    m_filter->restoreOrigin();
}

void VtkFiltersTool::clear() {
    if (m_filter) {
        m_filter->clearAllActor();
    }
}
