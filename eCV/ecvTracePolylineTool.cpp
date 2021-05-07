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

#include "ecvTracePolylineTool.h"

//Local
#include "MainWindow.h"

//common
#include <ecvPickingHub.h>

// CV_CORE_LIB
#include <CVLog.h>
#include <SquareMatrix.h>
#include <ManualSegmentationTools.h>

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvPolyline.h>
#include <ecvGenericPointCloud.h>
#include <ecvPointCloud.h>
#include <ecvMesh.h>
#include <ecvHObjectCaster.h>
#include <ecv2DViewportObject.h>

//Qt
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QProgressDialog>
#include <QApplication>

//System
#include <assert.h>
#include <iostream>

ccTracePolylineTool::SegmentGLParams::SegmentGLParams(int x , int y)
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::GetGLCameraParameters(params);
		CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(x, y);
		clickPos = CCVector2d(pos2D.x, pos2D.y);
	}
}

ccTracePolylineTool::ccTracePolylineTool(ccPickingHub* pickingHub, QWidget* parent)
	: ccOverlayDialog(parent)
	, Ui::TracePolyLineDlg()
    , m_polyTip(nullptr)
    , m_polyTipVertices(nullptr)
    , m_poly3D(nullptr)
    , m_poly3DVertices(nullptr)
	, m_done(false)
	, m_pickingHub(pickingHub)
{
	assert(pickingHub);

	setupUi(this);
	setWindowFlags(Qt::FramelessWindowHint | Qt::Tool);

    connect(saveToolButton,         &QToolButton::clicked, this, &ccTracePolylineTool::exportLine);
    connect(resetToolButton,		&QToolButton::clicked, this, &ccTracePolylineTool::resetLine);
    connect(continueToolButton,     &QToolButton::clicked, this, &ccTracePolylineTool::continueEdition);
    connect(validButton,			&QToolButton::clicked, this, &ccTracePolylineTool::apply);
    connect(cancelButton,			&QToolButton::clicked, this, &ccTracePolylineTool::cancel);
    connect(widthSpinBox,			static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &ccTracePolylineTool::onWidthSizeChanged);

	//add shortcuts
	addOverridenShortcut(Qt::Key_Escape); //escape key for the "cancel" button
	addOverridenShortcut(Qt::Key_Return); //return key for the "apply" button
    connect(this, &ccTracePolylineTool::shortcutTriggered, this, &ccTracePolylineTool::onShortcutTriggered);

	m_polyTipVertices = new ccPointCloud("Tip vertices");
	m_polyTipVertices->reserve(2);
	m_polyTipVertices->addPoint(CCVector3(0, 0, 0));
	m_polyTipVertices->addPoint(CCVector3(1, 1, 1));
	m_polyTipVertices->setEnabled(false);

	m_polyTip = new ccPolyline(m_polyTipVertices);
	m_polyTip->setForeground(true);
	m_polyTip->setTempColor(ecvColor::green);
	m_polyTip->set2DMode(true);
	m_polyTip->reserve(2);
	m_polyTip->addPointIndex(0, 2);
	m_polyTip->setWidth(widthSpinBox->value() < 2 ? 0 : widthSpinBox->value()); //'1' is equivalent to the default line size
	m_polyTip->addChild(m_polyTipVertices);

	validButton->setEnabled(false);
}

ccTracePolylineTool::~ccTracePolylineTool()
{
	if (m_polyTip)
		delete m_polyTip;
	//DGM: already a child of m_polyTip
	//if (m_polyTipVertices)
	//	delete m_polyTipVertices;

	if (m_poly3D)
		delete m_poly3D;
	//DGM: already a child of m_poly3D
	//if (m_poly3DVertices)
	//	delete m_poly3DVertices;
}

void ccTracePolylineTool::onShortcutTriggered(int key)
{
	switch (key)
	{
	case Qt::Key_Return:
		apply();
		return;

	case Qt::Key_Escape:
		cancel();
		return;

	default:
		//nothing to do
		break;
	}
}

ccPolyline* ccTracePolylineTool::polylineOverSampling(unsigned steps) const
{
	if (!m_poly3D || !m_poly3DVertices || m_segmentParams.size() != m_poly3DVertices->size())
	{
		assert(false);
        return nullptr;
	}

	if (steps <= 1)
	{
		//nothing to do
        return nullptr;
	}

	ccHObject::Container clouds;
	 ecvDisplayTools::GetSceneDB()->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD, false);
	ccHObject::Container meshes;
	 ecvDisplayTools::GetSceneDB()->filterChildren(meshes, true, CV_TYPES::MESH, false);

	if (clouds.empty() && meshes.empty())
	{
		//no entity is currently displayed?!
		assert(false);
        return nullptr;
	}

	unsigned n_verts = m_poly3DVertices->size();
	unsigned n_segments = m_poly3D->size() - (m_poly3D->isClosed() ? 0 : 1);
	unsigned end_size = n_segments * steps + (m_poly3D->isClosed() ? 0 : 1);

	ccPointCloud* newVertices = new ccPointCloud();
	ccPolyline* newPoly = new ccPolyline(newVertices);
	newPoly->addChild(newVertices);

	if (	!newVertices->reserve(end_size)
		||	!newPoly->reserve(end_size) )
	{
		CVLog::Warning("[ccTracePolylineTool::PolylineOverSampling] Not enough memory");
		delete newPoly;
        return nullptr;
	}
	newVertices->importParametersFrom(m_poly3DVertices);
	newVertices->setName(m_poly3DVertices->getName());
	newVertices->setEnabled(m_poly3DVertices->isEnabled());
	newPoly->importParametersFrom(*m_poly3D);

	QProgressDialog pDlg(QString("Oversampling"), "Cancel", 0, static_cast<int>(end_size), ecvDisplayTools::GetCurrentScreen() ? ecvDisplayTools::GetCurrentScreen() : 0);
	pDlg.show();
	QCoreApplication::processEvents();

	for (unsigned i = 0; i < n_segments; ++i)
	{
		const CCVector3* p1 = m_poly3DVertices->getPoint(i);
		newVertices->addPoint(*p1);

		unsigned i2 = (i + 1) % n_verts;
		CCVector2d v = m_segmentParams[i2].clickPos - m_segmentParams[i].clickPos;
		v /= steps;

		for (unsigned j = 1; j < steps; j++)
		{
			CCVector2d vj = m_segmentParams[i].clickPos + v * j;

			CCVector3 nearestPoint;
			double nearestElementSquareDist = -1.0;

			//for each cloud
			for (size_t c = 0; c < clouds.size(); ++c)
			{
				ccGenericPointCloud* cloud = static_cast<ccGenericPointCloud*>(clouds[c]);
				
				int nearestPointIndex = -1;
				double nearestSquareDist = 0;
				if (cloud->pointPicking(vj,
										m_segmentParams[i2].params,
										nearestPointIndex,
										nearestSquareDist,
										snapSizeSpinBox->value(),
										snapSizeSpinBox->value(),
										true))
				{
					if (nearestElementSquareDist < 0 || nearestSquareDist < nearestElementSquareDist)
					{
						nearestElementSquareDist = nearestSquareDist;
						nearestPoint = *cloud->getPoint(nearestPointIndex);
					}
				}
			}

			//for each mesh
			for (size_t m = 0; m < meshes.size(); ++m)
			{
				ccGenericMesh* mesh = static_cast<ccGenericMesh*>(meshes[m]);
				int nearestTriIndex = -1;
				double nearestSquareDist = 0;
				CCVector3d _nearestPoint;

				if (mesh->trianglePicking(	vj,
											m_segmentParams[i2].params,
											nearestTriIndex,
											nearestSquareDist,
											_nearestPoint))
				{
					if (nearestElementSquareDist < 0 || nearestSquareDist < nearestElementSquareDist)
					{
						nearestElementSquareDist = nearestSquareDist;
						nearestPoint = CCVector3::fromArray(_nearestPoint.u);
					}
				}
			}

			if (nearestElementSquareDist >= 0)
			{
				newVertices->addPoint(nearestPoint);
			}

			if (pDlg.wasCanceled())
			{
				steps = 0; //quick finish ;)
				break;
			}
			pDlg.setValue(pDlg.value() + 1);
		}
	}

	//add last point
	if (!m_poly3D->isClosed())
	{
		newVertices->addPoint(*m_poly3DVertices->getPoint(n_verts - 1));
	}

	newVertices->shrinkToFit();
	newPoly->addPointIndex(0, newVertices->size());

	return newPoly;
}

bool ccTracePolylineTool::linkWith(QWidget* win)
{
	assert(m_polyTip);
	assert(!m_poly3D);

	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}

	if (ecvDisplayTools::TheInstance())
	{
        connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::rightButtonClicked, this, &ccTracePolylineTool::closePolyLine);
        connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::mouseMoved,		  this, &ccTracePolylineTool::updatePolyLineTip);
	}

	return true;
}

static int s_defaultPickingRadius = 1;
static int s_overSamplingCount = 1;
bool ccTracePolylineTool::start()
{
	assert(m_polyTip);
	assert(!m_poly3D);

	if (!ecvDisplayTools::GetCurrentScreen())
	{
		CVLog::Warning("[Trace Polyline Tool] No associated window!");
		return false;
	}

	if (m_pickingHub)
	{
		m_pickingHub->removeListener(this);
	}
	ecvDisplayTools::SetPickingMode(ecvDisplayTools::NO_PICKING);
	ecvDisplayTools::SetInteractionMode(ecvDisplayTools::TRANSFORM_CAMERA()
										| ecvDisplayTools::INTERACT_SIG_RB_CLICKED
										| ecvDisplayTools::INTERACT_CTRL_PAN
										| ecvDisplayTools::INTERACT_SIG_MOUSE_MOVED);
	ecvDisplayTools::GetCurrentScreen()->setCursor(Qt::CrossCursor);

	snapSizeSpinBox->blockSignals(true);
	snapSizeSpinBox->setValue(s_defaultPickingRadius);
	snapSizeSpinBox->blockSignals(false);

	oversampleSpinBox->blockSignals(true);
	oversampleSpinBox->setValue(s_overSamplingCount);
	oversampleSpinBox->blockSignals(false);

	resetLine(); //to reset the GUI

	return ccOverlayDialog::start();
}

void ccTracePolylineTool::stop(bool accepted)
{
	assert(m_polyTip);

	if (m_pickingHub)
	{
		m_pickingHub->removeListener(this);
	}

	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::DisplayNewMessage("Polyline tracing [OFF]",
			ecvDisplayTools::UPPER_CENTER_MESSAGE,
			false, 2,
			ecvDisplayTools::MANUAL_SEGMENTATION_MESSAGE);

		//m_associatedWin->setUnclosable(false);
		resetTip();
		ecvDisplayTools::SetInteractionMode(ecvDisplayTools::TRANSFORM_CAMERA());
		ecvDisplayTools::GetCurrentScreen()->setCursor(Qt::ArrowCursor);
	}

	s_defaultPickingRadius = snapSizeSpinBox->value();
	s_overSamplingCount = oversampleSpinBox->value();

	ccOverlayDialog::stop(accepted);
}

void ccTracePolylineTool::updatePolyLineTip(int x, int y, Qt::MouseButtons buttons)
{
	if (!ecvDisplayTools::GetCurrentScreen())
	{
		assert(false);
		return;
	}
	
	if (buttons != Qt::NoButton)
	{
		//nothing to do (just hide the tip)
		if (m_polyTip->isEnabled())
		{
			m_polyTip->setEnabled(false);
			updateTip();
		}
		return;
	}

	if (!m_poly3DVertices || m_poly3DVertices->size() == 0)
	{
		//there should be at least one point already picked!
		return;
	}

	if (m_done)
	{
		// when it is done do nothing
		return;
	}

	assert(m_polyTip && m_polyTipVertices && m_polyTipVertices->size() == 2);

	//we replace the last point by the new one
	{
		CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(x, y);
		CCVector3 P2D(	static_cast<PointCoordinateType>(pos2D.x),
						static_cast<PointCoordinateType>(pos2D.y),
						0);

		CCVector3* lastP = const_cast<CCVector3*>(m_polyTipVertices->getPointPersistentPtr(1));
		*lastP = P2D;
	}

	//just in case (e.g. if the view has been rotated or zoomed)
	//we also update the first vertex position!
	{
		const CCVector3* P3D = m_poly3DVertices->getPoint(m_poly3DVertices->size() - 1);

		ccGLCameraParameters camera;
		ecvDisplayTools::GetGLCameraParameters(camera);

		CCVector3d A2D;
		camera.project(*P3D, A2D);

		CCVector3* firstP = const_cast<CCVector3*>(m_polyTipVertices->getPointPersistentPtr(0));
		*firstP = CCVector3(static_cast<PointCoordinateType>(A2D.x), //we convert A2D to centered coordinates (no need to apply high DPI scale or anything!)
							static_cast<PointCoordinateType>(A2D.y),
							0);

	}

	m_polyTip->setEnabled(true);

	updateTip();
}

void ccTracePolylineTool::onItemPicked(const PickedItem& pi)
{
	if (!ecvDisplayTools::GetCurrentScreen())
	{
		assert(false);
		return;
	}

	if (!pi.entity)
	{
		//means that the mouse has been clicked but no point was found!
		return;
	} 
	else
	{
		// avoid rendering this entity again.
		pi.entity->setRedraw(false);
	}

	//if the 3D polyline doesn't exist yet, we create it
	if (!m_poly3D || !m_poly3DVertices)
	{
		m_poly3DVertices = new ccPointCloud("Vertices");
		m_poly3DVertices->setEnabled(false);

		m_poly3D = new ccPolyline(m_poly3DVertices);
		m_poly3D->setTempColor(ecvColor::green);
		m_poly3D->set2DMode(false);
		m_poly3D->addChild(m_poly3DVertices);
		m_poly3D->setWidth(widthSpinBox->value() < 2 ? 0 : widthSpinBox->value()); //'1' is equivalent to the default line size

		ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(pi.entity);
		if (cloud)
		{
			//copy the first clicked entity's global shift & scale
			m_poly3D->setGlobalShift(cloud->getGlobalShift());
			m_poly3D->setGlobalScale(cloud->getGlobalScale());
		}

		m_segmentParams.resize(0); //just in case

		//ecvDisplayTools::AddToOwnDB(m_poly3D);
	}

	//try to add one more point
	if (	!m_poly3DVertices->reserve(m_poly3DVertices->size() + 1)
		||	!m_poly3D->reserve(m_poly3DVertices->size() + 1))
	{
		CVLog::Error("Not enough memory");
		return;
	}

	try
	{
		m_segmentParams.reserve(m_segmentParams.size() + 1);
	}
	catch (const std::bad_alloc&)
	{
		CVLog::Error("Not enough memory");
		return;
	}

	m_poly3DVertices->addPoint(pi.P3D);
	m_poly3D->addPointIndex(m_poly3DVertices->size() - 1);
	m_segmentParams.emplace_back(pi.clickPoint.x(), pi.clickPoint.y());

	//we replace the first point of the tip by this new point
	{
		CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(pi.clickPoint.x(), pi.clickPoint.y());
		CCVector3 P2D(	static_cast<PointCoordinateType>(pos2D.x),
						static_cast<PointCoordinateType>(pos2D.y),
						0);

		CCVector3* firstTipPoint = const_cast<CCVector3*>(m_polyTipVertices->getPointPersistentPtr(0));
		*firstTipPoint = P2D;
		m_polyTip->setEnabled(false); //don't need to display it for now
	}

	updatePoly3D();
}

void ccTracePolylineTool::closePolyLine(int, int)
{
	// CTRL + right click = panning
	if (!m_poly3D || (QApplication::keyboardModifiers() & Qt::ControlModifier))
	{
		return;
	}

	unsigned vertCount = m_poly3D->size();
	if (vertCount < 2)
	{
		//discard this polyline
		resetLine();
	}
	else
	{
		//hide the tip
		if (m_polyTip)
		{
			m_polyTip->setEnabled(false);
		}
		//update the GUI
		validButton->setEnabled(true);
		saveToolButton->setEnabled(true);
		resetToolButton->setEnabled(true);
		continueToolButton->setEnabled(true);
		if (m_pickingHub)
		{
			m_pickingHub->removeListener(this);
		}
		ecvDisplayTools::SetPickingMode(ecvDisplayTools::NO_PICKING); //no more picking
		m_done = true;

		updateTip();
	}
}

void ccTracePolylineTool::restart(bool reset)
{
	if (m_poly3D)
	{
		if (reset)
		{
			//discard this polyline
			resetPoly3D();

			if (m_polyTip)
			{
				// hide the tip
				m_polyTip->setEnabled(false);
			}

			delete m_poly3D;
			m_segmentParams.resize(0);
			//delete m_poly3DVertices;
            m_poly3D = nullptr;
            m_poly3DVertices = nullptr;
		}
		else
		{
			if (m_polyTip)
			{
				//show the tip
				m_polyTip->setEnabled(true);
			}
		}
	}

	//enable picking
	if (m_pickingHub && !m_pickingHub->addListener(
		this, true/*, true, ccGLWindow::POINT_PICKING*/))
	{
		CVLog::Error("The picking mechanism is already in use. Close the tool using it first.");
	}

	if (m_polyTip)
	{
		updateTip();
	}
	
	validButton->setEnabled(false);
	saveToolButton->setEnabled(false);
	resetToolButton->setEnabled(false);
	continueToolButton->setEnabled(false);
	m_done = false;
}

void ccTracePolylineTool::exportLine()
{
	if (!m_poly3D)
	{
		return;
	}

	unsigned overSampling = static_cast<unsigned>(oversampleSpinBox->value());
	if (overSampling > 1)
	{
		ccPolyline* poly = polylineOverSampling(overSampling);
		if (poly)
		{
			resetPoly3D();

			delete m_poly3D;
			m_segmentParams.resize(0);
            m_poly3DVertices = nullptr;
			m_poly3D = poly;
		}
	}

	m_poly3D->setTempColor(ecvColor::green);
	m_poly3D->setColor(ecvColor::green);
	if (MainWindow::TheInstance())
	{
		MainWindow::TheInstance()->addToDB(m_poly3D);
	}
	else
	{
		assert(false);
	}

    m_poly3D = nullptr;
	m_segmentParams.resize(0);
    m_poly3DVertices = nullptr;

	resetLine(); //to update the GUI
}

void ccTracePolylineTool::apply()
{
	exportLine();
	stop(true);
}

void ccTracePolylineTool::cancel()
{
	resetLine();

	stop(false);
}

void ccTracePolylineTool::onWidthSizeChanged(int width)
{
	if (m_poly3D)
	{
		m_poly3D->setWidth(width);
	}
	if (m_polyTip)
	{
		m_polyTip->setWidth(width);
	}
	
	if (ecvDisplayTools::GetCurrentScreen())
	{
		if (m_poly3D)
		{
			updatePoly3D();
		}
		if (m_polyTip)
		{
			updateTip();
		}
	}
}

// widget method
void ccTracePolylineTool::resetPoly3D()
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(
            WIDGETS_TYPE::WIDGET_POLYLINE, m_poly3D->getViewId()), true);
	}
}

void ccTracePolylineTool::updatePoly3D()
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::DrawWidgets(WIDGETS_PARAMETER(m_poly3D, WIDGETS_TYPE::WIDGET_POLYLINE), true);
	}
}

void ccTracePolylineTool::resetTip()
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(
            WIDGETS_TYPE::WIDGET_POLYLINE, m_polyTip->getViewId()), true);
	}
}

void ccTracePolylineTool::updateTip()
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::DrawWidgets(WIDGETS_PARAMETER(m_polyTip, WIDGETS_TYPE::WIDGET_POLYLINE), true);
	}
}
