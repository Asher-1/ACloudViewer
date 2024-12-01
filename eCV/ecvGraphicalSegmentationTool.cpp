// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: EDF R&D / DAHAI LU                                 #
// #                                                                        #
// ##########################################################################

#include "ecvGraphicalSegmentationTool.h"

#include "ecvGraphicalSegmentationOptionsDlg.h"

// LOCAL
#include "MainWindow.h"
#include "ecvItemSelectionDlg.h"

// CV_CORE_LIB
#include <ManualSegmentationTools.h>
#include <SquareMatrix.h>

// ECV_DB_LIB
#include <CVLog.h>
#include <ecv2DViewportObject.h>
#include <ecvDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// for the helper (apply)
#include <ecv2DLabel.h>
#include <ecvCameraSensor.h>
#include <ecvGBLSensor.h>
#include <ecvSubMesh.h>

// CVPluginAPI
#include <ecvMainAppInterface.h>

// Qt
#include <QInputDialog>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>

// System
#include <assert.h>

#if defined(_OPENMP)
// OpenMP
#include <omp.h>
#endif

ccGraphicalSegmentationTool::ccGraphicalSegmentationTool(QWidget* parent)
    : ccOverlayDialog(parent),
      Ui::GraphicalSegmentationDlg(),
      m_somethingHasChanged(false),
      m_state(0),
      m_segmentationPoly(nullptr),
      m_polyVertices(nullptr),
      m_rectangularSelection(false),
      m_deleteHiddenParts(false) {
    // Set QDialog background as transparent (DGM: doesn't work over an OpenGL
    // context)
    // setAttribute(Qt::WA_NoSystemBackground);

    setupUi(this);

    connect(inButton, &QToolButton::clicked, this,
            &ccGraphicalSegmentationTool::segmentIn);
    connect(outButton, &QToolButton::clicked, this,
            &ccGraphicalSegmentationTool::segmentOut);
    connect(razButton, &QToolButton::clicked, this,
            &ccGraphicalSegmentationTool::reset);
    connect(validButton, &QToolButton::clicked, this,
            &ccGraphicalSegmentationTool::apply);
    connect(validAndDeleteButton, &QToolButton::clicked, this,
            &ccGraphicalSegmentationTool::applyAndDelete);
    connect(cancelButton, &QToolButton::clicked, this,
            &ccGraphicalSegmentationTool::cancel);
    connect(pauseButton, &QToolButton::toggled, this,
            &ccGraphicalSegmentationTool::pauseSegmentation);

    // selection modes
    connect(actionSetPolylineSelection, &QAction::triggered, this,
            &ccGraphicalSegmentationTool::doSetPolylineSelection);
    connect(actionSetRectangularSelection, &QAction::triggered, this,
            &ccGraphicalSegmentationTool::doSetRectangularSelection);
    // import/export options
    connect(actionUseExistingPolyline, &QAction::triggered, this,
            &ccGraphicalSegmentationTool::doActionUseExistingPolyline);
    connect(actionExportSegmentationPolyline, &QAction::triggered, this,
            &ccGraphicalSegmentationTool::doExportSegmentationPolyline);

    // add shortcuts
    addOverridenShortcut(Qt::Key_Space);   // space bar for the "pause" button
    addOverridenShortcut(Qt::Key_Escape);  // escape key for the "cancel" button
    addOverridenShortcut(Qt::Key_Return);  // return key for the "apply" button
    addOverridenShortcut(
            Qt::Key_Delete);  // delete key for the "apply and delete" button
    addOverridenShortcut(Qt::Key_Tab);  // tab key to switch between rectangular
                                        // and polygonal selection modes
    addOverridenShortcut(Qt::Key_I);    //'I' key for the "segment in" button
    addOverridenShortcut(Qt::Key_O);    //'O' key for the "segment out" button
    connect(this, &ccOverlayDialog::shortcutTriggered, this,
            &ccGraphicalSegmentationTool::onShortcutTriggered);

    QMenu* selectionModeMenu = new QMenu(this);
    selectionModeMenu->addAction(actionSetPolylineSelection);
    selectionModeMenu->addAction(actionSetRectangularSelection);
    selectionModelButton->setDefaultAction(actionSetPolylineSelection);
    selectionModelButton->setMenu(selectionModeMenu);

    QMenu* importExportMenu = new QMenu(this);
    importExportMenu->addAction(actionUseExistingPolyline);
    importExportMenu->addAction(actionExportSegmentationPolyline);
    loadSaveToolButton->setMenu(importExportMenu);

    m_polyVertices = new ccPointCloud("vertices");
    m_segmentationPoly = new ccPolyline(m_polyVertices);
    m_segmentationPoly->setForeground(true);
    m_segmentationPoly->setColor(ecvColor::green);
    m_segmentationPoly->showColors(true);
    m_segmentationPoly->set2DMode(true);
    m_segmentationPoly->setClosed(true);
    allowPolylineExport(false);
}

void ccGraphicalSegmentationTool::allowPolylineExport(bool state) {
    if (state) {
        actionExportSegmentationPolyline->setEnabled(true);
    } else {
        loadSaveToolButton->setDefaultAction(actionUseExistingPolyline);
        actionExportSegmentationPolyline->setEnabled(false);
    }
}

ccGraphicalSegmentationTool::~ccGraphicalSegmentationTool() {
    if (m_segmentationPoly) delete m_segmentationPoly;
    m_segmentationPoly = nullptr;

    if (m_polyVertices) delete m_polyVertices;
    m_polyVertices = nullptr;
}

void ccGraphicalSegmentationTool::onShortcutTriggered(int key) {
    switch (key) {
        case Qt::Key_Space:
            pauseButton->toggle();
            return;

        case Qt::Key_I:
            inButton->click();
            return;

        case Qt::Key_O:
            outButton->click();
            return;

        case Qt::Key_Return:
            validButton->click();
            return;
        case Qt::Key_Delete:
            validAndDeleteButton->click();
            return;
        case Qt::Key_Escape:
            cancelButton->click();
            return;

        case Qt::Key_Tab:
            if (m_rectangularSelection)
                doSetPolylineSelection();
            else
                doSetRectangularSelection();
            return;

        default:
            // nothing to do
            break;
    }
}

bool ccGraphicalSegmentationTool::linkWith(QWidget* win) {
    assert(m_segmentationPoly);

    if (!ccOverlayDialog::linkWith(win)) {
        return false;
    }

    if (ecvDisplayTools::TheInstance()) {
        connect(ecvDisplayTools::TheInstance(),
                SIGNAL(leftButtonClicked(int, int)), this,
                SLOT(addPointToPolyline(int, int)));
        connect(ecvDisplayTools::TheInstance(),
                SIGNAL(rightButtonClicked(int, int)), this,
                SLOT(closePolyLine(int, int)));
        connect(ecvDisplayTools::TheInstance(),
                SIGNAL(mouseMoved(int, int, Qt::MouseButtons)), this,
                SLOT(updatePolyLine(int, int, Qt::MouseButtons)));
        connect(ecvDisplayTools::TheInstance(), SIGNAL(buttonReleased()), this,
                SLOT(closeRectangle()));
    }

    return true;
}

bool ccGraphicalSegmentationTool::start() {
    assert(m_polyVertices && m_segmentationPoly);

    m_segmentationPoly->clear();
    m_polyVertices->clear();
    allowPolylineExport(false);

    // ecvDisplayTools::AddToOwnDB(m_segmentationPoly);
    ecvDisplayTools::SetPickingMode(ecvDisplayTools::NO_PICKING);
    pauseSegmentationMode(false);

    m_somethingHasChanged = false;

    reset();

    return ccOverlayDialog::start();
}

void ccGraphicalSegmentationTool::prepareEntityForRemoval(
        ccHObject* entity, bool unallocateVisibilityArrays) {
    if (!entity) {
        assert(false);
        return;
    }

    // restore the display state of the entity
    entity->popDisplayState();

    if (unallocateVisibilityArrays) {
        ccGenericPointCloud* asCloud =
                ccHObjectCaster::ToGenericPointCloud(entity);
        if (asCloud) {
            asCloud->unallocateVisibilityArray();
        }
    }

    // specific case: we may have automatically hidden the mesh or the polyline
    // associated to a cloud
    if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccGenericPointCloud* cloud = static_cast<ccGenericPointCloud*>(entity);

        ccGenericMesh* associatedMesh = nullptr;
        if (ccGenericMesh::IsCloudVerticesOfMesh(cloud, &associatedMesh) &&
            associatedMesh) {
            associatedMesh->popDisplayState();
            return;
        }

        ccPolyline* associatedPolyline = nullptr;
        if (ccPolyline::IsCloudVerticesOfPolyline(cloud, &associatedPolyline) &&
            associatedPolyline) {
            associatedPolyline->popDisplayState();
            return;
        }
    }
}

void ccGraphicalSegmentationTool::removeAllEntities() {
    for (QSet<ccHObject*>::const_iterator p = m_toSegment.constBegin();
         p != m_toSegment.constEnd(); ++p) {
        ccHObject* entity = *p;

        prepareEntityForRemoval(entity, true);
    }

    setDrawFlag(true);  // for update afterforwards
    m_toSegment.clear();
}

void ccGraphicalSegmentationTool::stop(bool accepted) {
    assert(m_segmentationPoly);

    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::DisplayNewMessage(
                "Segmentation [OFF]", ecvDisplayTools::UPPER_CENTER_MESSAGE,
                false, 2, ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);

        ecvDisplayTools::SetInteractionMode(
                ecvDisplayTools::TRANSFORM_CAMERA());
        ecvDisplayTools::GetCurrentScreen()->setMouseTracking(false);
        ecvDisplayTools::SetPickingMode(ecvDisplayTools::DEFAULT_PICKING);
        resetSegmentation();
        ecvDisplayTools::SetRedrawRecursive(false);
        ecvDisplayTools::RedrawDisplay(true, false);
    }
    ccOverlayDialog::stop(accepted);
}

void ccGraphicalSegmentationTool::setDrawFlag(bool state /* = true*/) {
    for (QSet<ccHObject*>::iterator p = m_toSegment.begin();
         p != m_toSegment.end(); ++p) {
        (*p)->setRedrawFlagRecursive(state);
    }
}

void ccGraphicalSegmentationTool::reset() {
    if (m_somethingHasChanged) {
        for (QSet<ccHObject*>::const_iterator p = m_toSegment.constBegin();
             p != m_toSegment.constEnd(); ++p) {
            ccHObjectCaster::ToGenericPointCloud(*p)->resetVisibilityArray();
        }

        if (ecvDisplayTools::GetCurrentScreen()) {
            resetSegmentation();
            ecvDisplayTools::SetRedrawRecursive(false);
            setDrawFlag(true);
            ecvDisplayTools::RedrawDisplay(false);
        }

        m_somethingHasChanged = false;
    }

    razButton->setEnabled(false);
    validButton->setEnabled(false);
    validAndDeleteButton->setEnabled(false);
    loadSaveToolButton->setDefaultAction(actionUseExistingPolyline);
}

bool ccGraphicalSegmentationTool::addEntity(ccHObject* entity,
                                            bool silent /*=false*/) {
    bool result = false;
    if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccGenericPointCloud* cloud =
                ccHObjectCaster::ToGenericPointCloud(entity);

        ccGenericMesh* associatedMesh = nullptr;
        if (ccGenericMesh::IsCloudVerticesOfMesh(cloud, &associatedMesh)) {
            assert(nullptr != associatedMesh);
            if (m_toSegment.contains(associatedMesh)) {
                if (!silent) {
                    CVLog::Warning(QString("[Graphical Segmentation Tool] The "
                                           "mesh associated to cloud %1 is "
                                           "already selected")
                                           .arg(cloud->getName()));
                }
                return false;
            }

            // hide the associated mesh, as it will also be (graphically)
            // segmented
            associatedMesh->pushDisplayState();
            associatedMesh->setVisible(false);
        }

        ccPolyline* associatedPolyline = nullptr;
        if (ccPolyline::IsCloudVerticesOfPolyline(cloud, &associatedPolyline)) {
            assert(nullptr != associatedPolyline);
            if (m_toSegment.contains(associatedPolyline)) {
                if (!silent) {
                    CVLog::Warning(QString("[Graphical Segmentation Tool] The "
                                           "polyline associated to cloud %1 is "
                                           "already selected")
                                           .arg(cloud->getName()));
                }
                return false;
            }

            // hide the associated polyline, as it will also be (graphically)
            // segmented
            associatedPolyline->pushDisplayState();
            associatedPolyline->setVisible(false);
        }

        m_toSegment.insert(cloud);
        cloud->pushDisplayState();
        cloud->setVisible(true);
        cloud->setEnabled(true);
    } else if (entity->isKindOf(CV_TYPES::MESH)) {
        if (entity->isKindOf(CV_TYPES::PRIMITIVE)) {
            if (!silent) {
                CVLog::Warning(
                        "[ccGraphicalSegmentationTool] Can't segment "
                        "primitives "
                        "yet! Sorry...");
            }
            return false;
        }
        if (entity->isKindOf(CV_TYPES::SUB_MESH)) {
            if (!silent) {
                CVLog::Warning(
                        "[ccGraphicalSegmentationTool] Can't segment "
                        "sub-meshes! "
                        "Select the parent mesh...");
            }
            return false;
        } else {
            ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(entity);
            assert(mesh);

            ccGenericPointCloud* vertices = mesh->getAssociatedCloud();
            if (!vertices) {
                assert(false);
                return false;
            }

            // Make sure the vertices of this mesh are not already in the 'to
            // segment' list
            if (m_toSegment.contains(vertices)) {
                // let's remove the vertices
                mesh->pushDisplayState();  // just in case the vertices were
                                           // inserted before the mesh)
                vertices->popDisplayState();
                m_toSegment.remove(vertices);
            }

            m_toSegment.insert(mesh);
            mesh->pushDisplayState();
            mesh->setVisible(true);
            mesh->setEnabled(true);
            result = true;
        }
    } else if (entity->isKindOf(CV_TYPES::POLY_LINE)) {
        ccPolyline* poly = ccHObjectCaster::ToPolyline(entity);
        assert(poly);

        ccGenericPointCloud* verticesCloud =
                dynamic_cast<ccGenericPointCloud*>(poly->getAssociatedCloud());
        if (!verticesCloud) {
            assert(false);
            return false;
        }

        // Make sure the vertices of this polyline are not already in the 'to
        // segment' list
        if (verticesCloud && m_toSegment.contains(verticesCloud)) {
            // let's remove the vertices
            poly->pushDisplayState();  // just in case the vertices were
                                       // inserted before the polyline)
            verticesCloud->popDisplayState();
            m_toSegment.remove(verticesCloud);
        }

        m_toSegment.insert(poly);
        poly->pushDisplayState();
        poly->setVisible(true);
        poly->setEnabled(true);

        result = true;
    } else if (entity->isA(CV_TYPES::HIERARCHY_OBJECT)) {
        // automatically add entity's children
        for (unsigned i = 0; i < entity->getChildrenNumber(); ++i)
            result |= addEntity(entity->getChild(i));
    }

    return result;
}

unsigned ccGraphicalSegmentationTool::getNumberOfValidEntities() const {
    return static_cast<unsigned>(m_toSegment.size());
}

void ccGraphicalSegmentationTool::updatePolyLine(int x,
                                                 int y,
                                                 Qt::MouseButtons buttons) {
    // process not started yet?
    if ((m_state & RUNNING) == 0) {
        return;
    }

    assert(m_polyVertices);
    assert(m_segmentationPoly);

    unsigned vertCount = m_polyVertices->size();

    // new point (expressed relatively to the screen center)
    CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(x, y);
    CCVector3 P(static_cast<PointCoordinateType>(pos2D.x),
                static_cast<PointCoordinateType>(pos2D.y), 0);

    if (m_state & RECTANGLE) {
        // we need 4 points for the rectangle!
        if (vertCount != 4) m_polyVertices->resize(4);

        const CCVector3* A = m_polyVertices->getPointPersistentPtr(0);
        CCVector3* B = const_cast<CCVector3*>(
                m_polyVertices->getPointPersistentPtr(1));
        CCVector3* C = const_cast<CCVector3*>(
                m_polyVertices->getPointPersistentPtr(2));
        CCVector3* D = const_cast<CCVector3*>(
                m_polyVertices->getPointPersistentPtr(3));
        *B = CCVector3(A->x, P.y, 0);
        *C = P;
        *D = CCVector3(P.x, A->y, 0);

        if (vertCount != 4) {
            m_segmentationPoly->clear();
            if (!m_segmentationPoly->addPointIndex(0, 4)) {
                CVLog::Error("Out of memory!");
                allowPolylineExport(false);
                return;
            }
            m_segmentationPoly->setClosed(true);
        }
    } else if (m_state & POLYLINE) {
        if (vertCount < 2) return;
        // we replace last point by the current one
        CCVector3* lastP = const_cast<CCVector3*>(
                m_polyVertices->getPointPersistentPtr(vertCount - 1));
        *lastP = P;
    }

    updateSegmentation();
}

void ccGraphicalSegmentationTool::addPointToPolyline(int x, int y) {
    if ((m_state & STARTED) == 0) {
        return;
    }

    assert(m_polyVertices);
    assert(m_segmentationPoly);
    unsigned vertCount = m_polyVertices->size();

    // particular case: we close the rectangular selection by a 2nd click
    if (m_rectangularSelection && vertCount == 4 && (m_state & RUNNING)) return;

    // new point
    // QPointF pos2D = ecvDisplayTools::ToCenteredGLCoordinates(x, y);
    CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(x, y, 0);
    CCVector3 P(static_cast<PointCoordinateType>(pos2D.x),
                static_cast<PointCoordinateType>(pos2D.y), 0);

    // CTRL key pressed at the same time?
    bool ctrlKeyPressed = m_rectangularSelection ||
                          ((QApplication::keyboardModifiers() &
                            Qt::ControlModifier) == Qt::ControlModifier);

    // start new polyline?
    if (((m_state & RUNNING) == 0) || vertCount == 0 || ctrlKeyPressed) {
        // reset state
        m_state = (ctrlKeyPressed ? RECTANGLE : POLYLINE);
        m_state |= (STARTED | RUNNING);
        // reset polyline
        m_polyVertices->clear();
        if (!m_polyVertices->reserve(2)) {
            CVLog::Error("Out of memory!");
            allowPolylineExport(false);
            return;
        }
        // we add the same point twice (the last point will be used for display
        // only)
        m_polyVertices->addPoint(P);
        m_polyVertices->addPoint(P);
        m_segmentationPoly->clear();
        m_segmentationPoly->setClosed(true);
        if (!m_segmentationPoly->addPointIndex(0, 2)) {
            CVLog::Error("Out of memory!");
            allowPolylineExport(false);
            return;
        }
    } else  // next points in "polyline mode" only
    {
        // we were already in 'polyline' mode?
        if (m_state & POLYLINE) {
            if (!m_polyVertices->reserve(vertCount + 1)) {
                CVLog::Error("Out of memory!");
                allowPolylineExport(false);
                return;
            }

            // we replace last point by the current one
            CCVector3* lastP = const_cast<CCVector3*>(
                    m_polyVertices->getPointPersistentPtr(vertCount - 1));
            *lastP = P;

            // and add a new (equivalent) one
            m_polyVertices->addPoint(P);
            if (!m_segmentationPoly->addPointIndex(vertCount)) {
                CVLog::Error("Out of memory!");
                return;
            }
            m_segmentationPoly->setClosed(true);
        } else  // we must change mode
        {
            assert(false);  // we shouldn't fall here?!
            m_state &= (~RUNNING);
            addPointToPolyline(x, y);
            return;
        }
    }

    updateSegmentation();
}

void ccGraphicalSegmentationTool::closeRectangle() {
    // only for rectangle selection in RUNNING mode
    if ((m_state & RECTANGLE) == 0 || (m_state & RUNNING) == 0) return;

    assert(m_segmentationPoly);
    unsigned vertCount = m_segmentationPoly->size();
    if (vertCount < 4) {
        // first point only? we keep the real time update mechanism
        if (m_rectangularSelection) return;
        m_segmentationPoly->clear();
        m_polyVertices->clear();
        allowPolylineExport(false);
    } else {
        allowPolylineExport(true);
    }

    // stop
    m_state &= (~RUNNING);

    updateSegmentation();
}

void ccGraphicalSegmentationTool::closePolyLine(int, int) {
    // only for polyline in RUNNING mode
    if ((m_state & POLYLINE) == 0 || (m_state & RUNNING) == 0) return;

    assert(m_segmentationPoly);
    unsigned vertCount = m_segmentationPoly->size();
    if (vertCount < 4) {
        m_segmentationPoly->clear();
        m_polyVertices->clear();
    } else {
        // remove last point!
        m_segmentationPoly->resize(vertCount - 1);  // can't fail --> smaller
        m_segmentationPoly->setClosed(true);
    }

    // stop
    m_state &= (~RUNNING);

    // set the default import/export icon to 'export' mode
    loadSaveToolButton->setDefaultAction(actionExportSegmentationPolyline);
    allowPolylineExport(m_segmentationPoly->size() > 1);

    updateSegmentation();
}

void ccGraphicalSegmentationTool::updateSegmentation() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        resetSegmentation();
        WIDGETS_PARAMETER param(m_segmentationPoly,
                                WIDGETS_TYPE::WIDGET_POLYLINE_2D);
        param.opacity = 1.0;
        ecvDisplayTools::DrawWidgets(param, true);
    }
}

void ccGraphicalSegmentationTool::resetSegmentation() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                  m_segmentationPoly->getViewId()));
    }
}

void ccGraphicalSegmentationTool::segmentIn() { segment(true); }

void ccGraphicalSegmentationTool::segmentOut() { segment(false); }

void ccGraphicalSegmentationTool::segment(bool keepPointsInside) {
    if (!ecvDisplayTools::GetCurrentScreen()) return;

    if (!m_segmentationPoly) {
        CVLog::Error("No polyline defined!");
        return;
    }

    if (!m_segmentationPoly->isClosed()) {
        CVLog::Error(
                "Define and/or close the segmentation polygon first! (right "
                "click to close)");
        return;
    }

    // viewing parameters
    ccGLCameraParameters camera;
    ecvDisplayTools::GetGLCameraParameters(camera);

    // for each selected entity
    for (QSet<ccHObject*>::const_iterator p = m_toSegment.constBegin();
         p != m_toSegment.constEnd(); ++p) {
        ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(*p);
        assert(cloud);

        ccGenericPointCloud::VisibilityTableType& visibilityArray =
                cloud->getTheVisibilityArray();
        assert(!visibilityArray.empty());

        unsigned cloudSize = cloud->size();

        // we project each point and we check if it falls inside the
        // segmentation polyline
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int i = 0; i < static_cast<int>(cloudSize); ++i) {
            if (visibilityArray[i] == POINT_VISIBLE) {
                const CCVector3* P3D = cloud->getPoint(i);
                CCVector3d Q2D;
                camera.project(*P3D, Q2D);

                CCVector2 P2D(static_cast<PointCoordinateType>(Q2D.x),
                              static_cast<PointCoordinateType>(Q2D.y));

                bool pointInside =
                        cloudViewer::ManualSegmentationTools::isPointInsidePoly(
                                P2D, m_segmentationPoly);

                visibilityArray[i] =
                        (keepPointsInside != pointInside ? POINT_HIDDEN
                                                         : POINT_VISIBLE);
            }
        }
    }

    m_somethingHasChanged = true;
    validButton->setEnabled(true);
    validAndDeleteButton->setEnabled(true);
    razButton->setEnabled(true);
    ecvDisplayTools::SetRedrawRecursive(false);
    setDrawFlag(true);
    ecvDisplayTools::RedrawDisplay();
    pauseSegmentationMode(true, false);
}

void ccGraphicalSegmentationTool::pauseSegmentationMode(
        bool state, bool only2D /* = true*/) {
    assert(m_polyVertices && m_segmentationPoly);

    if (!ecvDisplayTools::GetMainWindow()) return;

    if (state /*=activate pause mode*/) {
        m_state = PAUSED;
        if (m_polyVertices->size() != 0) {
            m_segmentationPoly->clear();
            m_polyVertices->clear();
            allowPolylineExport(false);
        }
        ecvDisplayTools::SetInteractionMode(
                ecvDisplayTools::TRANSFORM_CAMERA());
        ecvDisplayTools::GetCurrentScreen()->setMouseTracking(false);
        ecvDisplayTools::DisplayNewMessage(
                "Segmentation [PAUSED]", ecvDisplayTools::UPPER_CENTER_MESSAGE,
                false, 3600, ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
        ecvDisplayTools::DisplayNewMessage(
                "Unpause to segment again",
                ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 3600,
                ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
    } else {
        m_state = STARTED;
        ecvDisplayTools::SetInteractionMode(
                ecvDisplayTools::INTERACT_SEND_ALL_SIGNALS);
        if (m_rectangularSelection) {
            ecvDisplayTools::DisplayNewMessage(
                    "Segmentation [ON] (rectangular selection)",
                    ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 3600,
                    ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
            ecvDisplayTools::DisplayNewMessage(
                    "Left click: set opposite corners",
                    ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 3600,
                    ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
        } else {
            ecvDisplayTools::DisplayNewMessage(
                    "Segmentation [ON] (polygonal selection)",
                    ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 3600,
                    ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
            ecvDisplayTools::DisplayNewMessage(
                    "Left click: add contour points / Right click: close",
                    ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 3600,
                    ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
        }
    }

    // update mini-GUI
    pauseButton->blockSignals(true);
    pauseButton->setChecked(state);
    pauseButton->blockSignals(false);

    resetSegmentation();
    ecvDisplayTools::SetRedrawRecursive(false);
    ecvDisplayTools::RedrawDisplay(only2D, state);
}

void ccGraphicalSegmentationTool::doSetPolylineSelection() {
    if (!m_rectangularSelection) return;

    selectionModelButton->setDefaultAction(actionSetPolylineSelection);

    m_rectangularSelection = false;
    if (m_state != PAUSED) {
        pauseSegmentationMode(true);
        pauseSegmentationMode(false);
    }

    ecvDisplayTools::DisplayNewMessage(
            QString(),
            ecvDisplayTools::UPPER_CENTER_MESSAGE);  // clear the area
    ecvDisplayTools::DisplayNewMessage(
            "Segmentation [ON] (polygonal selection)",
            ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 3600,
            ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
    ecvDisplayTools::DisplayNewMessage(
            "Left click: add contour points / Right click: close",
            ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 3600,
            ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
    ecvDisplayTools::SetRedrawRecursive(false);
    ecvDisplayTools::RedrawDisplay(true, false);
}

void ccGraphicalSegmentationTool::doSetRectangularSelection() {
    if (m_rectangularSelection) return;

    selectionModelButton->setDefaultAction(actionSetRectangularSelection);

    m_rectangularSelection = true;
    if (m_state != PAUSED) {
        pauseSegmentationMode(true);
        pauseSegmentationMode(false);
    }

    ecvDisplayTools::DisplayNewMessage(
            QString(),
            ecvDisplayTools::UPPER_CENTER_MESSAGE);  // clear the area
    ecvDisplayTools::DisplayNewMessage(
            "Segmentation [ON] (rectangular selection)",
            ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 3600,
            ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
    ecvDisplayTools::DisplayNewMessage(
            "Right click: set opposite corners",
            ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 3600,
            ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);
    ecvDisplayTools::SetRedrawRecursive(false);
    ecvDisplayTools::RedrawDisplay(true, false);
}

void ccGraphicalSegmentationTool::doActionUseExistingPolyline() {
    if (!ecvDisplayTools::GetMainWindow()) {
        assert(false);
        return;
    }

    MainWindow* mainWindow = MainWindow::TheInstance();
    if (mainWindow) {
        ccHObject* root = mainWindow->dbRootObject();
        ccHObject::Container polylines;
        if (root) {
            root->filterChildren(polylines, true, CV_TYPES::POLY_LINE);
        }

        if (!polylines.empty()) {
            int index = ccItemSelectionDlg::SelectEntity(polylines, 0, this);
            if (index < 0) return;
            assert(index >= 0 && index < static_cast<int>(polylines.size()));
            assert(polylines[index]->isA(CV_TYPES::POLY_LINE));
            ccPolyline* poly = static_cast<ccPolyline*>(polylines[index]);

            // look for an associated viewport
            ccHObject::Container viewports;
            if (poly->filterChildren(viewports, false,
                                     CV_TYPES::VIEWPORT_2D_OBJECT, true) == 1) {
                // shall we apply this viewport?
                if (QMessageBox::question(
                            ecvDisplayTools::GetMainWindow(),
                            "Associated viewport",
                            "The selected polyline has an associated viewport: "
                            "do you want to apply it?",
                            QMessageBox::Yes,
                            QMessageBox::No) == QMessageBox::Yes) {
                    ecvDisplayTools::SetViewportParameters(
                            static_cast<cc2DViewportObject*>(viewports.front())
                                    ->getParameters());
                    ecvDisplayTools::SetRedrawRecursive(false);
                    ecvDisplayTools::RedrawDisplay(false);
                    // m_associatedWin->redraw(false);
                }
            }

            cloudViewer::GenericIndexedCloudPersist* vertices =
                    poly->getAssociatedCloud();
            bool mode3D = !poly->is2DMode();

            // viewing parameters (for conversion from 3D to 2D)
            ccGLCameraParameters camera;
            ecvDisplayTools::GetGLCameraParameters(camera);
            // const double half_w = camera.viewport[2] / 2.0;
            // const double half_h = camera.viewport[3] / 2.0;

            // force polygonal selection mode
            doSetPolylineSelection();
            m_segmentationPoly->clear();
            m_polyVertices->clear();
            allowPolylineExport(false);

            // duplicate polyline 'a minima' (only points and indexes + closed
            // state)
            if (m_polyVertices->reserve(vertices->size() +
                                        (poly->isClosed() ? 0 : 1)) &&
                m_segmentationPoly->reserve(poly->size() +
                                            (poly->isClosed() ? 0 : 1))) {
                for (unsigned i = 0; i < vertices->size(); ++i) {
                    CCVector3 P = *vertices->getPoint(i);
                    if (mode3D) {
                        CCVector3d Q2D;
                        camera.project(P, Q2D);

                        P.x = static_cast<PointCoordinateType>(Q2D.x);
                        P.y = static_cast<PointCoordinateType>(Q2D.y);
                        P.z = 0;
                    }
                    m_polyVertices->addPoint(P);
                }
                for (unsigned j = 0; j < poly->size(); ++j) {
                    m_segmentationPoly->addPointIndex(
                            poly->getPointGlobalIndex(j));
                }

                m_segmentationPoly->setClosed(poly->isClosed());
                if (m_segmentationPoly->isClosed()) {
                    // stop (but we can't all pauseSegmentationMode as it would
                    // remove the current polyline)
                    m_state &= (~RUNNING);
                    allowPolylineExport(m_segmentationPoly->size() > 1);
                } else if (vertices->size()) {
                    // we make as if the segmentation was in progress
                    pauseSegmentationMode(false);
                    unsigned lastIndex = vertices->size() - 1;
                    m_polyVertices->addPoint(
                            *m_polyVertices->getPoint(lastIndex));
                    m_segmentationPoly->addPointIndex(lastIndex + 1);
                    m_segmentationPoly->setClosed(true);
                    m_state |= (POLYLINE | RUNNING);
                }

                m_rectangularSelection = false;
                updateSegmentation();
            } else {
                CVLog::Error("Not enough memory!");
            }
        } else {
            CVLog::Error("No polyline in DB!");
        }
    }
}

static unsigned s_polylineExportCount = 0;
void ccGraphicalSegmentationTool::doExportSegmentationPolyline() {
    MainWindow* mainWindow = MainWindow::TheInstance();
    if (mainWindow && m_segmentationPoly) {
        bool mode2D = false;
        // #ifdef ALLOW_2D_OR_3D_EXPORT
        QMessageBox messageBox(0);
        messageBox.setWindowTitle("Choose export type");
        messageBox.setText(
                "Export polyline in:\n - 2D (with coordinates relative to the "
                "screen)\n - 3D (with coordinates relative to the segmented "
                "entities)");
        QPushButton* button2D = new QPushButton("2D");
        QPushButton* button3D = new QPushButton("3D");
        messageBox.addButton(button2D, QMessageBox::AcceptRole);
        messageBox.addButton(button3D, QMessageBox::AcceptRole);
        messageBox.addButton(QMessageBox::Cancel);
        messageBox.setDefaultButton(button3D);
        messageBox.exec();
        if (messageBox.clickedButton() ==
            messageBox.button(QMessageBox::Cancel)) {
            // process cancelled by user
            return;
        }
        mode2D = (messageBox.clickedButton() == button2D);
        // #endif

        ccPolyline* poly = new ccPolyline(*m_segmentationPoly);

        // if the polyline is 2D and we export the polyline in 3D, we must
        // project its vertices
        if (!mode2D) {
            // get current display parameters
            ccGLCameraParameters camera;
            ecvDisplayTools::GetGLCameraParameters(camera);
            const int height = camera.viewport[3];

            // project the 2D polyline in 3D
            cloudViewer::GenericIndexedCloudPersist* vertices =
                    poly->getAssociatedCloud();
            ccPointCloud* verticesPC = dynamic_cast<ccPointCloud*>(vertices);
            if (verticesPC) {
                for (unsigned i = 0; i < vertices->size(); ++i) {
                    CCVector3* Pscreen =
                            const_cast<CCVector3*>(verticesPC->getPoint(i));
                    CCVector3d Q3D;
                    ecvDisplayTools::GetClick3DPos(
                            (int)Pscreen->x, height - (int)Pscreen->y, Q3D);
                    *Pscreen = CCVector3::fromArray(Q3D.u);
                }
                verticesPC->invalidateBoundingBox();
            } else {
                assert(false);
                CVLog::Warning(
                        "[Segmentation] Failed to convert 2D polyline to 3D! "
                        "(internal inconsistency)");
                mode2D = false;
            }

            // export Global Shift & Scale info (if any)
            bool hasGlobalShift = false;
            CCVector3d globalShift(0, 0, 0);
            double globalScale = 1.0;
            {
                for (QSet<ccHObject*>::const_iterator it =
                             m_toSegment.constBegin();
                     it != m_toSegment.constEnd(); ++it) {
                    ccShiftedObject* shifted = ccHObjectCaster::ToShifted(*it);
                    bool isShifted = (shifted && shifted->isShifted());
                    if (isShifted) {
                        globalShift = shifted->getGlobalShift();
                        globalScale = shifted->getGlobalScale();
                        hasGlobalShift = true;
                        break;
                    }
                }
            }

            if (hasGlobalShift && m_toSegment.size() != 1) {
                hasGlobalShift = (QMessageBox::question(
                                          MainWindow::TheInstance(),
                                          "Apply Global Shift",
                                          "At least one of the segmented "
                                          "entity has been shifted. Apply the "
                                          "same shift to the polyline?",
                                          QMessageBox::Yes,
                                          QMessageBox::No) == QMessageBox::Yes);
            }

            if (hasGlobalShift) {
                poly->setGlobalShift(globalShift);
                poly->setGlobalScale(globalScale);
            }
        }

        QString polyName = QString("Segmentation polyline #%1")
                                   .arg(++s_polylineExportCount);
        poly->setName(polyName);
        poly->setEnabled(
                false);  // we don't want it to appear while the segmentation
                         // mode is enabled! (anyway it's 2D only...)
        poly->set2DMode(mode2D);
        poly->setClosed(true);
        poly->setColor(ecvColor::yellow);  // we use a different color so as to
                                           // differentiate them from the active
                                           // polyline!

        // save associated viewport
        cc2DViewportObject* viewportObject =
                new cc2DViewportObject(polyName + QString(" viewport"));
        viewportObject->setParameters(ecvDisplayTools::GetViewportParameters());
        // viewportObject->setDisplay(m_associatedWin);
        poly->addChild(viewportObject);

        mainWindow->addToDB(poly, false, false, false);
        CVLog::Print(QString("[Segmentation] Polyline exported (%1 vertices)")
                             .arg(poly->size()));
    }
}

void ccGraphicalSegmentationTool::apply() {
    m_deleteHiddenParts = false;
    stop(true);
}

void ccGraphicalSegmentationTool::applyAndDelete() {
    m_deleteHiddenParts = true;
    stop(true);
}

void ccGraphicalSegmentationTool::cancel() {
    reset();
    m_deleteHiddenParts = false;
    stop(false);
}

static void RemoveUnusedLabelsAndUpdateTheOthers(
        std::set<cc2DLabel*>& watchedLabels,
        ccHObject* entity,
        const std::vector<int>& newIndexesOfRemainingPointsOrTriangles,
        ecvMainAppInterface* app) {
    if (!app) {
        assert(false);
        return;
    }

    std::set<cc2DLabel*>::iterator it = watchedLabels.begin();
    while (it != watchedLabels.end()) {
        cc2DLabel* label = *it;
        assert(label);
        for (unsigned i = 0; i < label->size(); ++i) {
            cc2DLabel::PickedPoint& pp = label->getPickedPoint(i);
            if (pp.entity() == entity) {
                if (pp.index < newIndexesOfRemainingPointsOrTriangles.size() &&
                    newIndexesOfRemainingPointsOrTriangles[pp.index] >= 0) {
                    // update the 'pointer'
                    pp.index = newIndexesOfRemainingPointsOrTriangles[pp.index];
                } else {
                    // delete the label
                    ccHObject* labelParent = label->getParent();
                    ecvMainAppInterface::ccHObjectContext parentContext;
                    bool saveContext = (labelParent != entity &&
                                        !entity->isAncestorOf(labelParent));
                    if (saveContext)
                        parentContext = app->removeObjectTemporarilyFromDBTree(
                                labelParent);
                    labelParent->removeChild(label);
                    if (saveContext)
                        app->putObjectBackIntoDBTree(labelParent,
                                                     parentContext);

                    label = nullptr;
                    it = watchedLabels.erase(it);
                    break;
                }
            }
        }

        if (label) {
            // keep the label and move on
            ++it;
        }
    }
}

bool ccGraphicalSegmentationTool::applySegmentation(
        ecvMainAppInterface* app, ccHObject::Container& newEntities) {
    if (!app) {
        assert(false);
        return false;
    }

    bool cantModifyPolylinesWarningIssued = false;

    // specific case: labels
    std::set<cc2DLabel*> watchedLabels;
    try {
        if (app->dbRootObject()) {
            ccHObject::Container loadedLabels;
            app->dbRootObject()->filterChildren(loadedLabels, true,
                                                CV_TYPES::LABEL_2D);

            for (ccHObject* labelEntity : loadedLabels) {
                cc2DLabel* label = static_cast<cc2DLabel*>(labelEntity);
                if (!label->getParent()) {
                    // sanity check: should never happen
                    assert(false);
                    continue;
                }
                for (unsigned i = 0; i < label->size(); ++i) {
                    const cc2DLabel::PickedPoint& pp = label->getPickedPoint(i);
                    if (m_toSegment.contains(pp.entity())) {
                        // we will watch this label as it may be deprecated by
                        // the segmentation process
                        watchedLabels.insert(label);
                        break;
                    }
                }
            }
        }
    } catch (const std::bad_alloc&) {
        // not enough memory
        CVLog::Error(tr("Not enough memory"));
        return false;
    }

    for (QSet<ccHObject*>::iterator p = m_toSegment.begin();
         p != m_toSegment.end();) {
        ccHObject* entity = (*p);

        // check first if we can modify this entity directly or if there might
        // be dire consequences...
        bool canModify = true;
        if (entity->isLocked()) {
            // we can't delete this entity
            CVLog::Warning("Entity " + entity->getName() +
                           " is locked. We won't be able to modify it");
            canModify = false;
        }

        if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccGenericPointCloud* cloud =
                    static_cast<ccGenericPointCloud*>(entity);
            if (cloud->size() == 0) {
                // ignore this cloud
                CVLog::Warning("Cloud " + cloud->getName() +
                               " is empty. We will ignore it");
                continue;
            }
            if (canModify) {
                // check that the point cloud is not the vertices of a mesh or
                // of a polyline
                if (ccGenericMesh::IsCloudVerticesOfMesh(cloud)) {
                    // we can't delete this cloud
                    CVLog::Warning("Cloud " + cloud->getName() +
                                   " seems to be the vertices of a mesh. We "
                                   "won't be able to modify it");
                    canModify = false;
                } else if (ccPolyline::IsCloudVerticesOfPolyline(cloud)) {
                    // we can't delete this cloud
                    CVLog::Warning("Cloud " + cloud->getName() +
                                   " seems to be the vertices of a polyine. We "
                                   "won't be able to modify it");
                    canModify = false;
                }
            }
        } else if (entity->isA(
                           CV_TYPES::MESH))  // TODO: sub-meshes and primitives
                                             // are not handled for now
        {
            ccGenericMesh* mesh = static_cast<ccGenericMesh*>(entity);
            if (mesh->size() == 0 || mesh->getAssociatedCloud()->size() == 0) {
                // ignore this mesh
                CVLog::Warning("Mesh " + mesh->getName() +
                               " is empty. We will ignore it");
                continue;
            }
        } else if (entity->isKindOf(CV_TYPES::POLY_LINE)) {
            ccPolyline* poly = static_cast<ccPolyline*>(entity);
            if (poly->size() == 0 || poly->getAssociatedCloud()->size() == 0) {
                // ignore this polyline
                CVLog::Warning("Polyline " + poly->getName() +
                               " is empty. We will ignore it");
                continue;
            }

            // can't modify polylines yet
            if (!cantModifyPolylinesWarningIssued) {
                CVLog::Warning(
                        "Can't modify polylines. A new polyline will be "
                        "created.");
                cantModifyPolylinesWarningIssued = true;
            }
            canModify = false;
        } else {
            // can't change this entity anyway
            continue;
        }

        if (entity->isKindOf(CV_TYPES::POINT_CLOUD) ||
            entity->isKindOf(CV_TYPES::MESH)) {
            // we temporarily detach the entity, as it may undergo
            // 'severe' modifications (octree deletion, etc.) --> see
            // ccPointCloud::createNewCloudFromVisibilitySelection
            ecvMainAppInterface::ccHObjectContext objContext =
                    app->removeObjectTemporarilyFromDBTree(entity);

            bool removeSelectedElementsFromEntity =
                    (canModify && !m_deleteHiddenParts);

            // apply segmentation
            ccHObject* segmentationResult = nullptr;
            bool deleteOriginalEntity = (canModify && m_deleteHiddenParts);
            if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
                ccGenericPointCloud* cloud =
                        ccHObjectCaster::ToGenericPointCloud(entity);

                std::vector<int> newIndexesOfRemainingPoints;
                ccGenericPointCloud* segmentedCloud =
                        cloud->createNewCloudFromVisibilitySelection(
                                removeSelectedElementsFromEntity, nullptr,
                                deleteOriginalEntity
                                        ? nullptr
                                        : &newIndexesOfRemainingPoints);
                if (segmentedCloud) {
                    if (segmentedCloud->size() == 0) {
                        // empty result: we ignore it
                        delete segmentedCloud;
                        segmentedCloud = nullptr;
                    } else if (segmentedCloud == cloud) {
                        // specific case: all points were selected, nothing to
                        // do
                        app->putObjectBackIntoDBTree(entity, objContext);
                        ++p;
                        continue;
                    } else  // we have a new entity
                    {
                        segmentationResult = segmentedCloud;

                        deleteOriginalEntity |= (cloud->size() == 0);

                        if (removeSelectedElementsFromEntity &&
                            !deleteOriginalEntity)  // if we have removed points
                                                    // from the original entity
                        {
                            // be smart and keep only the necessary labels
                            RemoveUnusedLabelsAndUpdateTheOthers(
                                    watchedLabels, cloud,
                                    newIndexesOfRemainingPoints, app);
                        }
                    }
                }
            } else if (entity->isA(CV_TYPES::MESH)) {
                ccMesh* mesh = ccHObjectCaster::ToMesh(entity);

                std::vector<int> newIndexesOfRemainingTriangles;
                ccMesh* segmentatedMesh = mesh->createNewMeshFromSelection(
                        removeSelectedElementsFromEntity,
                        deleteOriginalEntity ? nullptr
                                             : &newIndexesOfRemainingTriangles,
                        true);

                if (segmentatedMesh) {
                    if (segmentatedMesh->size() == 0) {
                        // empty result: we ignore it
                        delete segmentatedMesh;
                        segmentatedMesh = nullptr;
                    } else if (segmentatedMesh == mesh) {
                        // specific case: all triangles were selected, nothing
                        // to do
                        app->putObjectBackIntoDBTree(entity, objContext);
                        ++p;
                        continue;
                    } else  // we have a new entity
                    {
                        segmentationResult = segmentatedMesh;

                        deleteOriginalEntity |= (mesh->size() == 0);

                        if (removeSelectedElementsFromEntity &&
                            !deleteOriginalEntity) {
                            // be smart and keep only the necessary labels
                            RemoveUnusedLabelsAndUpdateTheOthers(
                                    watchedLabels, mesh,
                                    newIndexesOfRemainingTriangles, app);
                        }
                    }
                }
            } else {
                // we only expect clouds or meshes here
                assert(false);
            }

            if (segmentationResult)  // we have a result (= a new entity)
            {
                // update suffix
                {
                    QSettings settings;
                    settings.beginGroup(ccGraphicalSegmentationOptionsDlg::
                                                SegmentationToolOptionsKey());
                    QString segmentedSuffix =
                            settings.value(ccGraphicalSegmentationOptionsDlg::
                                                   SegmentedSuffixKey(),
                                           ".segmented")
                                    .toString();
                    settings.endGroup();

                    QString resultName = entity->getName();
                    if (!resultName.endsWith(segmentedSuffix)) {
                        resultName += segmentedSuffix;
                    }
                    segmentationResult->setName(resultName);

                    if (segmentationResult->isKindOf(CV_TYPES::MESH) &&
                        entity->isKindOf(CV_TYPES::MESH)) {
                        // update the mesh vertices as well
                        ccGenericMesh* mesh =
                                ccHObjectCaster::ToGenericMesh(entity);
                        ccGenericMesh* resultMesh =
                                ccHObjectCaster::ToGenericMesh(
                                        segmentationResult);
                        QString verticesName =
                                mesh->getAssociatedCloud()->getName();
                        if (!verticesName.endsWith(segmentedSuffix)) {
                            verticesName += segmentedSuffix;
                        }
                        resultMesh->getAssociatedCloud()->setName(verticesName);
                    }
                }

                if (removeSelectedElementsFromEntity &&
                    !deleteOriginalEntity)  // if we were able to modify the
                                            // original entity
                {
                    // update the name of the original entity
                    QSettings settings;
                    settings.beginGroup(ccGraphicalSegmentationOptionsDlg::
                                                SegmentationToolOptionsKey());
                    QString remainingSuffix =
                            settings.value(ccGraphicalSegmentationOptionsDlg::
                                                   RemainingSuffixKey(),
                                           ".remaining")
                                    .toString();
                    settings.endGroup();
                    if (!entity->getName().endsWith(remainingSuffix)) {
                        entity->setName(entity->getName() + remainingSuffix);
                    }
                    if (entity->isKindOf(CV_TYPES::MESH)) {
                        // update the mesh vertices as well
                        ccGenericMesh* mesh =
                                ccHObjectCaster::ToGenericMesh(entity);
                        QString verticesName =
                                mesh->getAssociatedCloud()->getName();
                        if (!verticesName.endsWith(remainingSuffix)) {
                            mesh->getAssociatedCloud()->setName(
                                    verticesName + remainingSuffix);
                        }
                    }

                    // specific case: deprecate GBL sensors' depth buffer
                    ccHObject::Container gblSensors;
                    entity->filterChildren(gblSensors, false,
                                           CV_TYPES::GBL_SENSOR);
                    for (ccHObject* child : gblSensors) {
                        ccGBLSensor* sensor =
                                ccHObjectCaster::ToGBLSensor(child);
                        // clear the associated depth buffer of the original
                        // sensor (deprecated)
                        sensor->clearDepthBuffer();
                        assert(entity->isKindOf(CV_TYPES::POINT_CLOUD));
                    }
                }

                // we look for first non-mesh or non-cloud parent
                ccHObject* resultParent = objContext.parent;
                while (resultParent &&
                       (resultParent->isKindOf(CV_TYPES::MESH) ||
                        resultParent->isKindOf(CV_TYPES::POINT_CLOUD))) {
                    resultParent = resultParent->getParent();
                }
                if (resultParent) {
                    resultParent->addChild(segmentationResult);
                }

                // segmentationResult->setDisplay_recursive(entity->getDisplay());
                // segmentationResult->prepareDisplayForRefresh_recursive();

                app->addToDB(segmentationResult, false, true, false, true);

                newEntities.push_back(segmentationResult);
            }

            if (!deleteOriginalEntity) {
                app->putObjectBackIntoDBTree(entity, objContext);
                ++p;
            } else {
                // remove all labels that depend on this entity
                std::set<cc2DLabel*>::iterator it = watchedLabels.begin();
                while (it != watchedLabels.end()) {
                    cc2DLabel* label = *it;
                    assert(label);
                    for (unsigned i = 0; i < label->size(); ++i) {
                        cc2DLabel::PickedPoint& pp = label->getPickedPoint(i);
                        if (pp.entity() == entity) {
                            // delete the label
                            ccHObject* labelParent = label->getParent();
                            ecvMainAppInterface::ccHObjectContext parentContext;
                            bool saveContext =
                                    (labelParent != entity &&
                                     !entity->isAncestorOf(labelParent));
                            if (saveContext)
                                parentContext =
                                        app->removeObjectTemporarilyFromDBTree(
                                                labelParent);
                            labelParent->removeChild(label);
                            if (saveContext)
                                app->putObjectBackIntoDBTree(labelParent,
                                                             parentContext);

                            label = nullptr;
                            it = watchedLabels.erase(it);
                            break;
                        }
                    }

                    if (label) {
                        // keep the label and move on
                        ++it;
                    }
                }

                prepareEntityForRemoval(entity, false);

                p = m_toSegment.erase(p);

                delete entity;  // TODO: should we wait that all entities are
                                // processed before removing it?
                entity = nullptr;
            }
        } else if (entity->isKindOf(CV_TYPES::POLY_LINE)) {
            ccPolyline* poly = static_cast<ccPolyline*>(entity);
            ccHObject* polyParent = poly->getParent();
            if (!polyParent) {
                polyParent = app->dbRootObject();
            }
            assert(polyParent);

            std::vector<ccPolyline*> polylines;
            if (poly->createNewPolylinesFromSelection(polylines)) {
                for (ccPolyline* p : polylines) {
                    // p->setDisplay_recursive(poly->getDisplay());
                    if (polyParent) polyParent->addChild(p);
                    app->addToDB(p, false, true, false, true);
                    newEntities.push_back(p);
                }
            }

            ++p;
        } else {
            assert(false);
            ++p;
        }
    }

    removeAllEntities();

    return true;
}
