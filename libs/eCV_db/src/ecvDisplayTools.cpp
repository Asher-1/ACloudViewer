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

#ifdef USE_VLD
//VLD
#include <vld.h>
#endif

// CV_CORE_LIB
#include <CVTools.h>

// LOCAL
#include "ecvBBox.h"
#include "ecvRenderingTools.h"
#include "ecvDisplayTools.h"
#include "ecvSingleton.h"
#include "ecvClipBox.h"
#include "ecvPointCloud.h"
#include "ecvSphere.h"
#include "ecvSubMesh.h"
#include "ecvCameraSensor.h"
#include "ecvHObjectCaster.h"
#include "ecvInteractor.h"
#include "ecv2DLabel.h"
#include "ecv2DViewportLabel.h"
#include "ecvPolyline.h"

#include "ecvGenericVisualizer.h"
#include "ecvGenericVisualizer2D.h"
#include "ecvGenericVisualizer3D.h"

// QT
#include <QString>
#include <QDesktopWidget>
#include <QMainWindow>
#include <QApplication>
#include <QScreen>
#include <QSettings>
#include <QMessageBox>
#include <QPushButton>
#include <QLayout>

// SYSTEM
#include <assert.h>

 //unique display tools instance
static ecvSingleton<ecvDisplayTools> s_tools;

bool ecvDisplayTools::USE_2D = true;
bool ecvDisplayTools::USE_VTK_PICK = false;

static const QString DEBUG_LAYER_ID = "DEBUG_LAYER";

//default interaction flags
ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::PAN_ONLY() { ecvDisplayTools::INTERACTION_FLAGS flags = INTERACT_PAN | INTERACT_ZOOM_CAMERA | INTERACT_2D_ITEMS | INTERACT_CLICKABLE_ITEMS; return flags; }
ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::TRANSFORM_CAMERA() { ecvDisplayTools::INTERACTION_FLAGS flags = INTERACT_ROTATE | PAN_ONLY(); return flags; }
ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::TRANSFORM_ENTITIES() { ecvDisplayTools::INTERACTION_FLAGS flags = INTERACT_ROTATE | INTERACT_PAN | INTERACT_ZOOM_CAMERA | INTERACT_TRANSFORM_ENTITIES | INTERACT_CLICKABLE_ITEMS; return flags; }

/*** Persistent settings ***/

static const char c_ps_groupName[] = "ECVWindow";
static const char c_ps_perspectiveView[] = "perspectiveView";
static const char c_ps_objectMode[] = "objectCenteredView";
static const char c_ps_sunLight[] = "sunLightEnabled";
static const char c_ps_customLight[] = "customLightEnabled";
static const char c_ps_pivotVisibility[] = "pivotVisibility";
static const char c_ps_stereoGlassType[] = "stereoGlassType";

//Vaious overlay elements dimensions
static const double CC_DISPLAYED_PIVOT_RADIUS_PERCENT = 0.8; //percentage of the smallest screen dimension
static const double CC_DISPLAYED_CUSTOM_LIGHT_LENGTH = 10.0;
static const float  CC_DISPLAYED_TRIHEDRON_AXES_LENGTH = 25.0f;
static const float  CC_DISPLAYED_CENTER_CROSS_LENGTH = 10.0f;

//Max click duration for enabling picking mode (in ms)
static const int CC_MAX_PICKING_CLICK_DURATION_MS = 200;

//Unique GL window ID
static int s_GlWindowNumber = 0;

void ecvDisplayTools::Init(ecvDisplayTools* displayTools, QMainWindow* win, bool stereoMode)
{
	//should be called only once!
	if (s_tools.instance)
	{
		assert(false);
		return;
	}

	s_tools.instance = displayTools;
	ecvGenericDisplayTools::SetInstance(s_tools.instance);

	//start internal timer
	s_tools.instance->m_timer.start();

	SetMainWindow(win);
	// register current instance visualizer only once
    s_tools.instance->registerVisualizer(win, stereoMode);

	s_tools.instance->m_uniqueID = ++s_GlWindowNumber; //GL window unique ID
	s_tools.instance->m_lastMousePos = QPoint(-1, -1);
	s_tools.instance->m_lastMouseMovePos = QPoint(-1, -1);
	s_tools.instance->m_validModelviewMatrix = false;
	s_tools.instance->m_validProjectionMatrix = false;
	s_tools.instance->m_cameraToBBCenterDist = 0.0;
	s_tools.instance->m_shouldBeRefreshed = false;
	s_tools.instance->m_mouseMoved = false;
	s_tools.instance->m_mouseButtonPressed = false;

	s_tools.instance->m_bbHalfDiag = 0.0;
	s_tools.instance->m_interactionFlags = TRANSFORM_CAMERA();
	s_tools.instance->m_pickingMode = NO_PICKING;
	s_tools.instance->m_pickingModeLocked = false;
	s_tools.instance->m_lastClickTime_ticks = 0;

	s_tools.instance->m_sunLightEnabled = true;
	s_tools.instance->m_customLightEnabled = false;
	s_tools.instance->m_clickableItemsVisible = false;
	s_tools.instance->m_alwaysUseFBO = false;
	s_tools.instance->m_updateFBO = true;
	s_tools.instance->m_winDBRoot = nullptr;
	s_tools.instance->m_globalDBRoot = nullptr; //external DB
	s_tools.instance->m_removeFlag = false;
	s_tools.instance->m_font = QFont();
	s_tools.instance->m_pivotVisibility = PIVOT_SHOW_ON_MOVE;
	s_tools.instance->m_pivotSymbolShown = false;
	s_tools.instance->m_allowRectangularEntityPicking = false;
	s_tools.instance->m_rectPickingPoly = nullptr;
	s_tools.instance->m_overridenDisplayParametersEnabled = false;
	s_tools.instance->m_displayOverlayEntities = true;
	s_tools.instance->m_bubbleViewModeEnabled = false;
	s_tools.instance->m_bubbleViewFov_deg = 90.0f;
	s_tools.instance->m_touchInProgress = false;
	s_tools.instance->m_touchBaseDist = 0.0;
	s_tools.instance->m_scheduledFullRedrawTime = 0;
	s_tools.instance->m_exclusiveFullscreen = false;
	s_tools.instance->m_showDebugTraces = false;
	s_tools.instance->m_pickRadius = DefaultPickRadius;
	s_tools.instance->m_autoRefresh = false;
	s_tools.instance->m_hotZone = nullptr;
	s_tools.instance->m_showCursorCoordinates = false;
	s_tools.instance->m_autoPickPivotAtCenter = false;
	s_tools.instance->m_ignoreMouseReleaseEvent = false;
	s_tools.instance->m_rotationAxisLocked = false;
	s_tools.instance->m_lockedRotationAxis = CCVector3d(0, 0, 1);

	//GL window own DB
	s_tools.instance->m_winDBRoot = new ccHObject(QString("DB.3DView_%1").arg(s_tools.instance->m_uniqueID));

	//matrices
	s_tools.instance->m_viewportParams.viewMat.toIdentity();
    s_tools.instance->m_viewportParams.setCameraCenter(CCVector3d(0.0, 0.0, 1.0)); //don't position the camera on the pivot by default!
	s_tools.instance->m_viewMatd.toIdentity();
	s_tools.instance->m_projMatd.toIdentity();

	//default modes
	SetPickingMode(DEFAULT_PICKING);
	SetInteractionMode(TRANSFORM_CAMERA());

	//auto-load previous perspective settings
	{
		QSettings settings;
		settings.beginGroup(c_ps_groupName);

		//load parameters
		bool perspectiveView = settings.value(c_ps_perspectiveView, false).toBool();
		//DGM: we force object-centered view by default now, as the viewer-based perspective is too dependent
		//on what is displayed (so restoring this parameter at next startup is rarely a good idea)
		bool objectCenteredView = /*settings.value(c_ps_objectMode, true ).toBool()*/true;
		int pivotVisibility = settings.value(c_ps_pivotVisibility, PIVOT_HIDE).toInt();

		settings.endGroup();

		//pivot visibility
		switch (pivotVisibility)
		{
		case PIVOT_HIDE:
			SetPivotVisibility(PIVOT_HIDE);
			break;
		case PIVOT_SHOW_ON_MOVE:
			SetPivotVisibility(PIVOT_SHOW_ON_MOVE);
			break;
		case PIVOT_ALWAYS_SHOW:
			SetPivotVisibility(PIVOT_ALWAYS_SHOW);
			break;
		}

		//apply saved parameters
		SetPerspectiveState(perspectiveView, objectCenteredView);
	}

	s_tools.instance->m_deferredPickingTimer.setSingleShot(true);
	s_tools.instance->m_deferredPickingTimer.setInterval(100);

	//signal/slot connections
	connect(s_tools.instance, &ecvDisplayTools::itemPickedFast, s_tools.instance, 
		&ecvDisplayTools::onItemPickedFast, Qt::DirectConnection);
	connect(GetVisualizer3D(), &ecvGenericVisualizer3D::interactorPointPickedEvent, s_tools.instance, &ecvDisplayTools::onPointPicking);

	connect(&s_tools.instance->m_scheduleTimer, &QTimer::timeout, s_tools.instance, &ecvDisplayTools::checkScheduledRedraw);
	connect(&s_tools.instance->m_deferredPickingTimer, &QTimer::timeout, s_tools.instance, &ecvDisplayTools::doPicking);
}

ecvDisplayTools* ecvDisplayTools::TheInstance()
{
	if (!s_tools.instance) { CVLog::Warning("ecvDisplayTools must be initialized"); return nullptr; }
	return s_tools.instance;
}

void ecvDisplayTools::ReleaseInstance()
{
	if (s_tools.instance)
	{
		s_tools.release();
	}
}

ecvDisplayTools:: ~ecvDisplayTools()
{
	cancelScheduledRedraw();

	delete m_winDBRoot;
	m_winDBRoot = nullptr;

	delete m_rectPickingPoly;
	m_rectPickingPoly = nullptr;

	delete m_hotZone;
	m_hotZone = nullptr;
}

void ecvDisplayTools::checkScheduledRedraw()
{
	if (m_scheduledFullRedrawTime && m_timer.elapsed() > m_scheduledFullRedrawTime)
	{
		//clean the outdated messages
		{
			std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
			qint64 currentTime_sec = m_timer.elapsed() / 1000;
			//CVLog::PrintDebug(QString("[paintGL] Current time: %1.").arg(currentTime_sec));

			while (it != m_messagesToDisplay.end())
			{
				//no more valid? we delete the message
				if (it->messageValidity_sec < currentTime_sec)
				{
					RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, it->message));
					it = m_messagesToDisplay.erase(it);
				}
				else
				{
					++it;
				}
			}
		}
	}
}

void ecvDisplayTools::cancelScheduledRedraw()
{
	m_scheduledFullRedrawTime = 0;
	m_scheduleTimer.stop();
}

void ecvDisplayTools::scheduleFullRedraw(unsigned maxDelay_ms)
{
	m_scheduledFullRedrawTime = m_timer.elapsed() + maxDelay_ms;

	if (!m_scheduleTimer.isActive())
	{
		m_scheduleTimer.start(500);
	}
}

void ecvDisplayTools::onPointPicking(const CCVector3 &p, int index, const std::string& id)
{
	m_last_picked_point = p;
	m_last_point_index = index;
	m_last_picked_id = id.c_str();
#ifdef QT_DEBUG
	CVLog::Print(QString("current selected index is %1").arg(index));
	CVLog::Print(QString("current selected entity id is %1").arg(m_last_picked_id));
	CVLog::Print(QString("current selected point coord is [%1, %2, %3]").arg(p.x).arg(p.y).arg(p.z));
#endif // !QDEBUG

	if (m_last_picked_id.isEmpty())
	{
		PICKING_MODE pickingMode = PICKING_MODE::ENTITY_PICKING;
		PickingParameters params(pickingMode, 0, 0, m_pickRadius, m_pickRadius);
		ProcessPickingResult(params, nullptr, -1);
	}
	else
	{
		if (ecvDisplayTools::USE_VTK_PICK)
		{
			doPicking();
		}
	}

}

void ecvDisplayTools::doPicking()
{
	int x = m_lastMousePos.x();
	int y = m_lastMousePos.y();

	if (x < 0 || y < 0)
	{
		assert(false);
		return;
	}

	if ((m_pickingMode != NO_PICKING)
		|| (m_interactionFlags & INTERACT_2D_ITEMS))
	{
		if (m_interactionFlags & INTERACT_2D_ITEMS)
		{
			//label selection
			UpdateActiveItemsList(x, y, false);
			if (!m_activeItems.empty() && m_activeItems.size() == 1)
			{
				ccInteractor* pickedObj = m_activeItems.front();
				cc2DLabel* label = dynamic_cast<cc2DLabel*>(pickedObj);
				if (label && !label->isSelected())
				{
					// warning deprecated!
					emit s_tools.instance->entitySelectionChanged(label);
					QApplication::processEvents();
				}
			}
		}
		else
		{
			assert(m_activeItems.empty());
		}

		if (m_activeItems.empty() && m_pickingMode != NO_PICKING)
		{
			//perform standard picking
			PICKING_MODE pickingMode = m_pickingMode;

			//shift+Alt = point/triangle picking
			if (pickingMode == ENTITY_PICKING && (QApplication::keyboardModifiers() & Qt::AltModifier))
			{
                pickingMode = LABEL_PICKING;
            }
            else if(pickingMode == ENTITY_PICKING && (QApplication::keyboardModifiers() & Qt::ControlModifier))
            {
                pickingMode = POINT_OR_TRIANGLE_PICKING;
            }

			PickingParameters params(pickingMode, x, y, m_pickRadius, m_pickRadius);
			StartPicking(params);
		}
	}

}

void ecvDisplayTools::onWheelEvent(float wheelDelta_deg)
{
	//in perspective mode, wheel event corresponds to 'walking'
	if (m_viewportParams.perspectiveView)
	{
		//to zoom in and out we simply change the fov in bubble-view mode!
		if (m_bubbleViewModeEnabled)
		{
			SetBubbleViewFov(m_bubbleViewFov_deg - wheelDelta_deg / 3.6f); //1 turn = 100 degrees
		}
		else
		{
			//convert degrees in 'constant' walking speed in ... pixels ;)
			const double& deg2PixConversion = GetDisplayParameters().zoomSpeed;
			double delta = deg2PixConversion * 
				static_cast<double>(wheelDelta_deg) * m_viewportParams.pixelSize;

			//if we are (clearly) outisde of the displayed objects bounding-box
			if (m_cameraToBBCenterDist > m_bbHalfDiag)
			{
				//we go faster if we are far from the entities
				delta *= 1.0 + std::log(m_cameraToBBCenterDist / m_bbHalfDiag);
			}

//			MoveCamera(0.0f, 0.0f, static_cast<float>(-delta));
		}
	}
	else //ortho. mode
	{
		//convert degrees in zoom 'power'
		static const float c_defaultDeg2Zoom = 20.0f;
		float zoomFactor = std::pow(1.1f, wheelDelta_deg / c_defaultDeg2Zoom);
		UpdateZoom(zoomFactor);
	}

	UpdateDisplayParameters();
}

bool ecvDisplayTools::ProcessClickableItems(int x, int y)
{
	if (s_tools.instance->m_clickableItems.empty())
	{
		return false;
	}

		// correction for HD screens
    const int retinaScale = GetDevicePixelRatio();
    x *= retinaScale;
    y *= retinaScale;

    ClickableItem::Role clickedItem = ClickableItem::NO_ROLE;
    for (std::vector<ClickableItem>::const_iterator it = s_tools.instance->m_clickableItems.begin();
         it != s_tools.instance->m_clickableItems.end(); ++it) {
        if (it->area.contains(x, y)) {
            clickedItem = it->role;
            break;
        }
    }

	switch (clickedItem)
	{
	case ClickableItem::NO_ROLE:
	{
		//nothing to do
	}
	break;

	case ClickableItem::INCREASE_POINT_SIZE:
	{
		SetPointSize(s_tools.instance->m_viewportParams.defaultPointSize + 1.0f);
		SetRedrawRecursive(false);
		RedrawDisplay();
	}
	return true;

	case ClickableItem::DECREASE_POINT_SIZE:
	{
		SetPointSize(s_tools.instance->m_viewportParams.defaultPointSize - 1.0f);
		SetRedrawRecursive(false);
		RedrawDisplay();
	}
	return true;

	case ClickableItem::INCREASE_LINE_WIDTH:
	{
		SetLineWidth(s_tools.instance->m_viewportParams.defaultLineWidth + 1.0f);
		SetRedrawRecursive(false);
		RedrawDisplay();
	}
	return true;

	case ClickableItem::DECREASE_LINE_WIDTH:
	{
		SetLineWidth(s_tools.instance->m_viewportParams.defaultLineWidth - 1.0f);
		SetRedrawRecursive(false);
		RedrawDisplay();
	}
	return true;

	case ClickableItem::LEAVE_BUBBLE_VIEW_MODE:
	{
		SetBubbleViewMode(false);
		RedrawDisplay();
	}
	return true;

	case ClickableItem::LEAVE_FULLSCREEN_MODE:
	{
		if (s_tools.instance->m_win)
		{
            emit s_tools.instance->exclusiveFullScreenToggled(false);
		}
		
	}
	return true;

	default:
	{
		//unhandled item?!
		assert(false);
	}
	break;
	}

	return false;
}

void ecvDisplayTools::SetPointSize(float size, bool silent, int viewport)
{
	float newSize = std::max(std::min(size, MAX_POINT_SIZE_F), MIN_POINT_SIZE_F);
	if (!silent)
	{
		CVLog::Print(QString("New point size: %1").arg(newSize));
	}

	if (s_tools.instance->m_viewportParams.defaultPointSize != newSize)
	{
		s_tools.instance->m_viewportParams.defaultPointSize = newSize;
		SetPointSizeRecursive(static_cast<int>(newSize));

		if (!silent)
		{
			ecvDisplayTools::DisplayNewMessage(QString("New default point size: %1").arg(newSize),
				ecvDisplayTools::LOWER_LEFT_MESSAGE, //DGM HACK: we cheat and use the same 'slot' as the window size
				false,
				2,
				SCREEN_SIZE_MESSAGE);
		}
	}
}

void ecvDisplayTools::SetPointSizeRecursive(int size)
{
	// we draw 3D entities
	if (s_tools.instance->m_globalDBRoot)
	{
		s_tools.instance->m_globalDBRoot->setPointSizeRecursive(size);
	}

	if (s_tools.instance->m_winDBRoot)
	{
		s_tools.instance->m_winDBRoot->setPointSizeRecursive(size);
	}
}

void ecvDisplayTools::SetLineWidth(float width, bool silent, int viewport)
{
	float newWidth = std::max(std::min(width, MAX_LINE_WIDTH_F), MIN_LINE_WIDTH_F);
	if (!silent)
	{
		CVLog::Print(QString("New line with: %1").arg(newWidth));
	}

	if (s_tools.instance->m_viewportParams.defaultLineWidth != newWidth)
	{
		s_tools.instance->m_viewportParams.defaultLineWidth = newWidth;
		SetLineWithRecursive(newWidth);
		if (!silent)
		{
			ecvDisplayTools::DisplayNewMessage(QString("New default line width: %1").arg(newWidth),
				ecvDisplayTools::LOWER_LEFT_MESSAGE, //DGM HACK: we cheat and use the same 'slot' as the window size
				false,
				2,
				SCREEN_SIZE_MESSAGE);
		}
	}
}

void ecvDisplayTools::SetLineWithRecursive(PointCoordinateType with)
{
	// we draw 3D entities
	if (s_tools.instance->m_globalDBRoot)
	{
		s_tools.instance->m_globalDBRoot->setLineWidthRecursive(with);
	}

	if (s_tools.instance->m_winDBRoot)
	{
		s_tools.instance->m_winDBRoot->setLineWidthRecursive(with);
	}
}

void ecvDisplayTools::StartPicking(PickingParameters& params)
{
	//correction for HD screens
	const int retinaScale = GetDevicePixelRatio();
	params.centerX *= retinaScale;
	params.centerY *= retinaScale;

	if (!s_tools.instance->m_globalDBRoot && !s_tools.instance->m_winDBRoot)
	{
		//we must always emit a signal!
		ProcessPickingResult(params, nullptr, -1);
		return;
	}

	if (params.mode == POINT_OR_TRIANGLE_PICKING
		|| params.mode == POINT_PICKING
		|| params.mode == TRIANGLE_PICKING
		|| params.mode == LABEL_PICKING // = spawn a label on the clicked point or triangle
		)
	{
		//CPU-based point picking
		StartCPUBasedPointPicking(params);
	}
	else
	{
		StartOpenGLPicking(params);
	}
}

void ecvDisplayTools::ProcessPickingResult(const PickingParameters& params,
	ccHObject* pickedEntity,
	int pickedItemIndex,
	const CCVector3* nearestPoint/*=0*/,
	const std::unordered_set<int>* selectedIDs/*=0*/)
{
	//standard "entity" picking
	if (params.mode == ENTITY_PICKING)
	{
		emit s_tools.instance->entitySelectionChanged(pickedEntity);
	}
	//rectangular "entity" picking
	else if (params.mode == ENTITY_RECT_PICKING)
	{
		if (selectedIDs)
			emit s_tools.instance->entitiesSelectionChanged(*selectedIDs);
		else
			assert(false);
	}
	//3D point or triangle picking
	else if (params.mode == POINT_PICKING
		|| params.mode == TRIANGLE_PICKING
		|| params.mode == POINT_OR_TRIANGLE_PICKING)
	{
		assert(pickedEntity == nullptr || pickedItemIndex >= 0);
		assert(nearestPoint);

		emit s_tools.instance->itemPicked(pickedEntity, static_cast<unsigned>(pickedItemIndex),
			params.centerX, params.centerY, *nearestPoint);
	}
	//fast picking (labels, interactors, etc.)
	else if (params.mode == FAST_PICKING)
	{
		emit s_tools.instance->itemPickedFast(pickedEntity, pickedItemIndex, params.centerX, params.centerY);
	}
	else if (params.mode == LABEL_PICKING)
	{
		if (s_tools.instance->m_globalDBRoot && pickedEntity && pickedItemIndex >= 0)
		{
			//qint64 stopTime = m_timer.nsecsElapsed();
			//CVLog::Print(QString("[Picking] entity ID %1 - item #%2 (time: %3 ms)").arg(selectedID).arg(pickedItemIndex).arg((stopTime-startTime) / 1.0e6));

			//auto spawn the right label
			cc2DLabel* label = nullptr;
			if (pickedEntity->isKindOf(CV_TYPES::POINT_CLOUD))
			{
				label = new cc2DLabel();
				label->addPickedPoint(ccHObjectCaster::ToGenericPointCloud(pickedEntity), pickedItemIndex);
				pickedEntity->addChild(label);
			}
			else if (pickedEntity->isKindOf(CV_TYPES::MESH))
			{
				label = new cc2DLabel();
				ccGenericMesh *mesh = ccHObjectCaster::ToGenericMesh(pickedEntity);
				ccGenericPointCloud *cloud = mesh->getAssociatedCloud();
				assert(cloud);
				cloudViewer::VerticesIndexes *vertexIndexes = mesh->getTriangleVertIndexes(pickedItemIndex);
				label->addPickedPoint(cloud, vertexIndexes->i1);
				label->addPickedPoint(cloud, vertexIndexes->i2);
				label->addPickedPoint(cloud, vertexIndexes->i3);
				cloud->addChild(label);
				if (!cloud->isEnabled())
				{
					cloud->setVisible(false);
					cloud->setEnabled(true);
				}
			}

			if (label)
			{
				label->setVisible(true);
				label->setPosition(static_cast<float>(params.centerX + 20) / s_tools.instance->m_glViewport.width(),
					static_cast<float>(params.centerY + 20) / s_tools.instance->m_glViewport.height());
				emit s_tools.instance->newLabel(static_cast<ccHObject*>(label));
				QApplication::processEvents();
			}
		}
	}
}

void ecvDisplayTools::SetZNearCoef(double coef)
{
	if (coef <= 0.0 || coef >= 1.0)
	{
		CVLog::Warning("[ecvDisplayTools::setZNearCoef] Invalid coef. value!");
		return;
	}

	if (s_tools.instance->m_viewportParams.zNearCoef != coef)
	{
		//update param
		s_tools.instance->m_viewportParams.zNearCoef = coef;
		//and camera state (if perspective view is 'on')
		if (s_tools.instance->m_viewportParams.perspectiveView)
		{
			//invalidateViewport();
			//invalidateVisualization();

			//DGM: we update the projection matrix directly so as to get an up-to-date estimation of zNear
			UpdateProjectionMatrix();

			SetCameraClip(s_tools.instance->m_viewportParams.zNear,
				s_tools.instance->m_viewportParams.zFar);

			Deprecate3DLayer();

			DisplayNewMessage(QString("Near clipping = %1% of max depth (= %2)")
				.arg(s_tools.instance->m_viewportParams.zNearCoef * 100.0, 0, 'f', 1)
				.arg(s_tools.instance->m_viewportParams.zNear),
				ecvDisplayTools::LOWER_LEFT_MESSAGE, //DGM HACK: we cheat and use the same 'slot' as the window size
				false,
				2,
				SCREEN_SIZE_MESSAGE);
		}

		emit s_tools.instance->zNearCoefChanged(coef);
		emit s_tools.instance->cameraParamChanged();
	}
}

//DGM: WARNING: OpenGL picking with the picking buffer is depreacted.
//We need to get rid of this code or change it to color-based selection...
void ecvDisplayTools::StartOpenGLPicking(const PickingParameters& params)
{
	if (!params.pickInLocalDB && !params.pickInSceneDB)
	{
		assert(false);
		return;
	}

	//setup rendering context
	unsigned short flags = CC_DRAW_FOREGROUND;

	switch (params.mode)
	{
	case FAST_PICKING:
		flags |= CC_DRAW_FAST_NAMES_ONLY;
	case ENTITY_PICKING:
	case ENTITY_RECT_PICKING:
		flags |= CC_DRAW_ENTITY_NAMES;
		break;
	default:
		//unhandled mode?!
		assert(false);
		//we must always emit a signal!
		ProcessPickingResult(params, nullptr, -1);
		return;
	}

	//OpenGL picking
	assert(!s_tools.instance->m_captureMode.enabled);

	//process hits
	std::unordered_set<int> selectedIDs;
	int pickedItemIndex = -1;
	int selectedID = -1;
	ccHObject* pickedEntity = nullptr;

	CCVector3 P(0, 0, 0);
	CCVector3* pickedPoint = nullptr;

	if (s_tools.instance->m_last_point_index >= 0)
	{
		pickedEntity = GetPickedEntity(params);
		if (pickedEntity)
		{
			selectedID = pickedEntity->getUniqueID();
			selectedIDs.insert(selectedID);
			pickedItemIndex = s_tools.instance->m_last_point_index;
		}
	}

	if (pickedEntity && pickedItemIndex >= 0 && 
		pickedEntity->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		ccGenericPointCloud* tempEntity = ccHObjectCaster::ToGenericPointCloud(pickedEntity);
		int pNum = static_cast<int>(tempEntity->size());
		if (pickedItemIndex >= pNum)
		{
			P = s_tools.instance->m_last_picked_point;
			CVLog::Warning(QString("[ecvDisplayTools::StartOpenGLPicking] Picking Error, %1 is more than picked entity size %2").arg(pickedItemIndex).arg(tempEntity->size()));
			pickedItemIndex = pNum - 1;
		}
		else
		{
			P = *(static_cast<ccGenericPointCloud*>(pickedEntity)->getPoint(pickedItemIndex));
			// check selected point
			CCVector3 temp = P - s_tools.instance->m_last_picked_point;
			if (temp.norm() > 1)
			{
				ProcessPickingResult(params, nullptr, -1);
#ifdef QT_DEBUG
				CVLog::Warning(QString("[ecvDisplayTools::StartOpenGLPicking] droped selected point coord is [%1, %2, %3]").arg(P.x).arg(P.y).arg(P.z));
#endif // QT_DEBUG
				return;
			}

		}

		pickedPoint = &P;
	}

	//we must always emit a signal!
	ProcessPickingResult(params, pickedEntity, pickedItemIndex, pickedPoint, &selectedIDs);
}

void ecvDisplayTools::StartCPUBasedPointPicking(const PickingParameters& params)
{
	//qint64 t0 = m_timer.elapsed();
	CCVector2d clickedPos(params.centerX, s_tools.instance->m_glViewport.height() - 1 - params.centerY);

	ccHObject* nearestEntity = nullptr;
	int nearestElementIndex = -1;
	double nearestElementSquareDist = -1.0;
	CCVector3 nearestPoint(0, 0, 0);
	static const unsigned MIN_POINTS_FOR_OCTREE_COMPUTATION = 128;

	static ecvGui::ParamStruct::ComputeOctreeForPicking autoComputeOctreeThisSession = ecvGui::ParamStruct::ASK_USER;
	bool autoComputeOctree = false;
	bool firstCloudWithoutOctree = true;

	ccGLCameraParameters camera;
	GetGLCameraParameters(camera);

	if (ecvDisplayTools::USE_VTK_PICK)
	{
		int pickedIndex = -1;
		ccHObject* pickedEntity = nullptr;
		if (s_tools.instance->m_last_point_index >= 0)
		{
			pickedIndex = s_tools.instance->m_last_point_index;
			pickedEntity = GetPickedEntity(params);
		}

		if (pickedEntity && pickedIndex >= 0)
		{
			ccHObject* ent = nullptr;
			if (pickedEntity->isKindOf(CV_TYPES::POINT_CLOUD))
			{
				ent = pickedEntity;
			}
			else if (pickedEntity->isKindOf(CV_TYPES::MESH) && !pickedEntity->isA(CV_TYPES::MESH_GROUP))
			{
				ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(pickedEntity);
				ent = mesh->getAssociatedCloud();
			}
			else
			{
				return;
			}

			ccGenericPointCloud* tempEntity = ccHObjectCaster::ToGenericPointCloud(ent);
			nearestElementIndex = pickedIndex;
			int pNum = static_cast<int>(tempEntity->size());
			if (pickedIndex >= pNum)
			{
				nearestPoint = s_tools.instance->m_last_picked_point;
				CVLog::Warning(QString("[ecvDisplayTools::StartCPUBasedPointPicking] Picking Error, %1 is more than picked entity size %2").arg(pickedIndex).arg(tempEntity->size()));
				nearestElementIndex = pNum;
			}
			else
			{
				nearestPoint = *(tempEntity->getPoint(pickedIndex));
				// check selected point
				CCVector3 temp = nearestPoint - s_tools.instance->m_last_picked_point;
				if (temp.norm() > 1)
				{
                    ProcessPickingResult(params, nullptr, -1);
#ifdef QT_DEBUG
					CVLog::Warning(QString("[ecvDisplayTools::StartCPUBasedPointPicking] droped selected point coord is [%1, %2, %3]").
						arg(nearestPoint.x).arg(nearestPoint.y).arg(nearestPoint.z));
#endif // QT_DEBUG
					return;
				}
			}

			nearestEntity = pickedEntity;
    }
	}
	else
	{
		try
		{
			ccHObject::Container toProcess;
			if (s_tools.instance->m_globalDBRoot)
				toProcess.push_back(s_tools.instance->m_globalDBRoot);
			if (s_tools.instance->m_winDBRoot)
				toProcess.push_back(s_tools.instance->m_winDBRoot);

			while (!toProcess.empty())
			{
				//get next item
				ccHObject* ent = toProcess.back();
				toProcess.pop_back();

				if (!ent->isEnabled())
					continue;

				bool ignoreSubmeshes = false;

				//we look for point cloud displayed in this window
                if (ent->isKindOf(CV_TYPES::POINT_CLOUD))
                {
                    ccGenericPointCloud* cloud = static_cast<ccGenericPointCloud*>(ent);

					if (firstCloudWithoutOctree && !cloud->getOctree() && cloud->size() > MIN_POINTS_FOR_OCTREE_COMPUTATION) //no need to use the octree for a few points!
					{
						//can we compute an octree for picking?
						ecvGui::ParamStruct::ComputeOctreeForPicking behavior = GetDisplayParameters().autoComputeOctree;
						if (behavior == ecvGui::ParamStruct::ASK_USER)
						{
							//we use the persistent parameter for this session
							behavior = autoComputeOctreeThisSession;
						}

						switch (behavior)
						{
						case ecvGui::ParamStruct::ALWAYS:
							autoComputeOctree = true;
							break;

						case ecvGui::ParamStruct::ASK_USER:
						{
							QMessageBox question(QMessageBox::Question,
								"Picking acceleration",
								"Automatically compute octree(s) to accelerate the picking process?\n(this behavior can be changed later in the Display Settings)",
								QMessageBox::NoButton,
								GetCurrentScreen());

							QPushButton* yes = new QPushButton("Yes");
							question.addButton(yes, QMessageBox::AcceptRole);
							QPushButton* no = new QPushButton("No");
							question.addButton(no, QMessageBox::RejectRole);
							QPushButton* always = new QPushButton("Always");
							question.addButton(always, QMessageBox::AcceptRole);
							QPushButton* never = new QPushButton("Never");
							question.addButton(never, QMessageBox::RejectRole);

							question.exec();
							QAbstractButton* clickedButton = question.clickedButton();
							if (clickedButton == yes)
							{
								autoComputeOctree = true;
								autoComputeOctreeThisSession = ecvGui::ParamStruct::ALWAYS;
							}
							else if (clickedButton == no)
							{
								CVLog::Warning("now only support octree picking, please don't select no!");
								continue;
								autoComputeOctree = false;
								autoComputeOctreeThisSession = ecvGui::ParamStruct::NEVER;
							}
							else if (clickedButton == always || clickedButton == never)
							{
								autoComputeOctree = (clickedButton == always);
								//update the global application parameters
								ecvGui::ParamStruct params = ecvGui::Parameters();
								params.autoComputeOctree = autoComputeOctree ? ecvGui::ParamStruct::ALWAYS : ecvGui::ParamStruct::NEVER;
								ecvGui::Set(params);
								params.toPersistentSettings();
							}
						}
						break;

						case ecvGui::ParamStruct::NEVER:
							autoComputeOctree = false;
							break;
						}

						firstCloudWithoutOctree = false;
					}

					int nearestPointIndex = -1;
					double nearestSquareDist = 0.0;
					if (cloud->pointPicking(clickedPos,
						camera,
						nearestPointIndex,
						nearestSquareDist,
						params.pickWidth,
						params.pickHeight,
						autoComputeOctree && cloud->size() > MIN_POINTS_FOR_OCTREE_COMPUTATION))
					{
						if (nearestElementIndex < 0 || (nearestPointIndex >= 0 && nearestSquareDist < nearestElementSquareDist))
						{
							nearestElementSquareDist = nearestSquareDist;
							nearestElementIndex = nearestPointIndex;
							nearestPoint = *(cloud->getPoint(nearestPointIndex));
							nearestEntity = cloud;
						}
					}
				}
				else if (ent->isKindOf(CV_TYPES::MESH)
					&& !ent->isA(CV_TYPES::MESH_GROUP)) //we don't need to process mesh groups as their children will be processed later
				{
					ignoreSubmeshes = true;

					ccGenericMesh* mesh = static_cast<ccGenericMesh*>(ent);
					if (mesh->isShownAsWire())
					{
						//skip meshes that are displayed in wireframe mode
						continue;
					}

					int nearestTriIndex = -1;
					double nearestSquareDist = 0.0;
					CCVector3d P;
					if (mesh->trianglePicking(clickedPos,
						camera,
						nearestTriIndex,
						nearestSquareDist,
						P))
					{
						if (nearestElementIndex < 0 || (nearestTriIndex >= 0 && nearestSquareDist < nearestElementSquareDist))
						{
							nearestElementSquareDist = nearestSquareDist;
							nearestElementIndex = nearestTriIndex;
							nearestPoint = CCVector3::fromArray(P.u);
							nearestEntity = mesh;
						}
					}
                } else if (ent->isKindOf(CV_TYPES::SENSOR)) {
                    // only activated when ctrl and mouse pressed!
                    if (params.mode != POINT_OR_TRIANGLE_PICKING)
                    {
                        continue;
                    }

                    if (ent->isA(CV_TYPES::CAMERA_SENSOR))
                    {
                        ignoreSubmeshes = true;

                        ccCameraSensor* cameraSensor = static_cast<ccCameraSensor*>(ent);
                        if (!cameraSensor && cameraSensor->getNearPlane().isEmpty())
                        {
                            //skip meshes that are displayed in wireframe mode
                            continue;
                        }

                        QString id = ecvDisplayTools::PickObject(clickedPos.x, clickedPos.y);

                        if (id.toInt() != -1 && static_cast<int>(cameraSensor->getUniqueID()) == id.toInt())
                        {
                            nearestElementIndex = id.toInt();
                            nearestPoint = CCVector3();
                            nearestEntity = cameraSensor;
                            break;
                        }
                    }
                }

				//add children
				for (unsigned i = 0; i < ent->getChildrenNumber(); ++i)
				{
					//we ignore the sub-meshes of the current (mesh) entity
					//as their content is the same!
					if (ignoreSubmeshes
						&&	ent->getChild(i)->isKindOf(CV_TYPES::SUB_MESH)
						&& static_cast<ccSubMesh*>(ent)->getAssociatedMesh() == ent)
					{
						continue;
					}

					toProcess.push_back(ent->getChild(i));
				}
			}
		}
		catch (const std::bad_alloc&)
		{
			//not enough memory
			CVLog::Warning("[Picking][CPU] Not enough memory!");
		}
	}
	//qint64 dt = m_timer.elapsed() - t0;
	//CVLog::Print(QString("[Picking][CPU] Time: %1 ms").arg(dt));

	if (!ecvDisplayTools::USE_VTK_PICK)
	{
		s_tools.instance->m_last_point_index = nearestElementIndex;
		s_tools.instance->m_last_picked_point = nearestPoint;
		if (nearestEntity)
		{
			s_tools.instance->m_last_picked_id = QString::number(nearestEntity->getUniqueID());
		}
	}

	//we must always emit a signal!
	ProcessPickingResult(params, nearestEntity, nearestElementIndex, &nearestPoint);
}

ccHObject* ecvDisplayTools::GetPickedEntity(const PickingParameters& params)
{
	if (s_tools.instance->m_last_picked_id.isEmpty()) return nullptr;

	ccHObject* pickedEntity = nullptr;
	unsigned int selectedID = s_tools.instance->m_last_picked_id.toUInt();
	if (params.pickInSceneDB && s_tools.instance->m_globalDBRoot)
	{
		pickedEntity = s_tools.instance->m_globalDBRoot->find(selectedID);
	}
	if (!pickedEntity && params.pickInLocalDB && s_tools.instance->m_winDBRoot)
	{
		pickedEntity = s_tools.instance->m_winDBRoot->find(selectedID);
	}

	return pickedEntity;
}

QPointF ecvDisplayTools::ToCenteredGLCoordinates(int x, int y)
{
	return QPointF(x - Width() / 2, Height() / 2 - y)/* * GetDevicePixelRatio()*/;
}

CCVector3d ecvDisplayTools::ToVtkCoordinates(int x, int y, int z)
{
	CCVector3d p = CCVector3d(x*1.0, y*1.0, z*1.0);
	ToVtkCoordinates(p);
	return p;
}

void ecvDisplayTools::ToVtkCoordinates(CCVector3d & sP)
{
	sP.y = Height() - sP.y; // for vtk coordinates
    sP *= GetDevicePixelRatio();
}

void ecvDisplayTools::ToVtkCoordinates(CCVector2i &sP)
{
    sP.y = Height() - sP.y; // for vtk coordinates
    sP *= GetDevicePixelRatio();
}

void ecvDisplayTools::SetPivotVisibility(PivotVisibility vis)
{
	s_tools.instance->m_pivotVisibility = vis;

	if (vis == PivotVisibility::PIVOT_HIDE)
	{
		SetPivotVisibility(false);
	}
	else
	{
		SetPivotVisibility(true);
	}

	UpdateScreen();

	//auto-save last pivot visibility settings
	{
		QSettings settings;
		settings.beginGroup(c_ps_groupName);
		settings.setValue(c_ps_pivotVisibility, vis);
		settings.endGroup();
	}
}

void ecvDisplayTools::ResizeGL(int w, int h)
{
	//update OpenGL viewport
	SetGLViewport(0, 0, w, h);

	InvalidateVisualization();
	Deprecate3DLayer();

	if (s_tools.instance->m_hotZone)
	{
		s_tools.instance->m_hotZone->topCorner = QPoint(0, 0);
	}

	DisplayNewMessage(QString("New size = %1 * %2 (px)").
		arg(s_tools.instance->m_glViewport.width()).arg(s_tools.instance->m_glViewport.height()),
		LOWER_LEFT_MESSAGE,
		false,
		2,
		SCREEN_SIZE_MESSAGE);
}

void ecvDisplayTools::MoveCamera(float dx, float dy, float dz)
{
	if (dx != 0.0f || dy != 0.0f) //camera movement? (dz doesn't count as it only corresponds to a zoom)
	{
		//feedback for echo mode
		emit s_tools.instance->cameraDisplaced(dx, dy);
	}

	//current X, Y and Z viewing directions
	//correspond to the 'model view' matrix
	//lines.
	CCVector3d V(dx, dy, dz);
	if (!s_tools.instance->m_viewportParams.objectCenteredView)
	{
		s_tools.instance->m_viewportParams.viewMat.transposed().applyRotation(V);
	}

    SetCameraPos(s_tools.instance->m_viewportParams.getCameraCenter() + V);
}

void ecvDisplayTools::UpdateActiveItemsList(int x, int y, bool extendToSelectedLabels/*=false*/)
{
	s_tools.instance->m_activeItems.clear();

	PickingParameters params(FAST_PICKING, x, y, 2, 2);

	StartPicking(params);

	if (s_tools.instance->m_activeItems.size() == 1)
	{
		ccInteractor* pickedObj = s_tools.instance->m_activeItems.front();
		cc2DLabel* label = dynamic_cast<cc2DLabel*>(pickedObj);
		if (label)
		{
			if (!label->isSelected() || !extendToSelectedLabels)
			{
				//select it?
				//emit entitySelectionChanged(label);
				//QApplication::processEvents();
			}
			else
			{
				//we get the other selected labels as well!
				ccHObject::Container labels;
				if (s_tools.instance->m_globalDBRoot)
					s_tools.instance->m_globalDBRoot->filterChildren(labels, true, CV_TYPES::LABEL_2D);
				if (s_tools.instance->m_winDBRoot)
					s_tools.instance->m_winDBRoot->filterChildren(labels, true, CV_TYPES::LABEL_2D);

				for (auto & label : labels)
				{
					if (label->isA(CV_TYPES::LABEL_2D) && label->isVisible()) //Warning: cc2DViewportLabel is also a kind of 'CV_TYPES::LABEL_2D'!
					{
						cc2DLabel* l = static_cast<cc2DLabel*>(label);
						if (l != label && l->isSelected())
						{
							s_tools.instance->m_activeItems.push_back(l);
						}
					}
				}
			}
		}
	}
}

void ecvDisplayTools::onItemPickedFast(ccHObject* pickedEntity, int pickedItemIndex, int x, int y)
{
	if (pickedEntity)
	{
		if (pickedEntity->isA(CV_TYPES::LABEL_2D))
		{
			cc2DLabel* label = static_cast<cc2DLabel*>(pickedEntity);
			m_activeItems.push_back(label);
		}
		else if (pickedEntity->isA(CV_TYPES::CLIPPING_BOX))
		{
			ccClipBox* cbox = static_cast<ccClipBox*>(pickedEntity);
			cbox->setActiveComponent(pickedItemIndex);
			cbox->setClickedPoint(x, y, Width(), Height(), m_viewportParams.viewMat);

			m_activeItems.push_back(cbox);
		}
	}

	emit fastPickingFinished();
}

void ecvDisplayTools::UpdateScreenSize()
{
	ResizeGL(Width(), Height());
}

CCVector3d ecvDisplayTools::ConvertMousePositionToOrientation(int x, int y)
{
	double xc = static_cast<double>(Width() / 2);
	double yc = static_cast<double>(Height() / 2); //DGM FIME: is it scaled coordinates or not?!

	CCVector3d Q2D;
	if (s_tools.instance->m_viewportParams.objectCenteredView)
	{
		//project the current pivot point on screen
		ccGLCameraParameters camera;
		GetGLCameraParameters(camera);

        if (!camera.project(s_tools.instance->m_viewportParams.getPivotPoint(), Q2D))
		{
			//arbitrary direction
			return CCVector3d(0, 0, 1);
		}

		//we set the virtual rotation pivot closer to the actual one (but we always stay in the central part of the screen!)
		Q2D.x = std::min(Q2D.x, 3.0 * Width() / 4.0);
		Q2D.x = std::max(Q2D.x, Width() / 4.0);

		Q2D.y = std::min(Q2D.y, 3.0 * Height() / 4.0);
		Q2D.y = std::max(Q2D.y, Height() / 4.0);
	}
	else
	{
		Q2D.x = xc;
		Q2D.y = yc;
	}

	//invert y
	y = Height() - 1 - y;

	CCVector3d v(x - Q2D.x, y - Q2D.y, 0.0);

	v.x = std::max(std::min(v.x / xc, 1.0), -1.0);
	v.y = std::max(std::min(v.y / yc, 1.0), -1.0);

	//square 'radius'
	double d2 = v.x*v.x + v.y*v.y;

	//projection on the unit sphere
	if (d2 > 1)
	{
		double d = std::sqrt(d2);
		v.x /= d;
		v.y /= d;
	}
	else
	{
		v.z = std::sqrt(1.0 - d2);
	}

	return v;
}

void ecvDisplayTools::RotateBaseViewMat(const ccGLMatrixd& rotMat)
{
	s_tools.instance->m_viewportParams.viewMat = rotMat * s_tools.instance->m_viewportParams.viewMat;

	//we emit the 'baseViewMatChanged' signal
	emit s_tools.instance->baseViewMatChanged(s_tools.instance->m_viewportParams.viewMat);
	emit s_tools.instance->cameraParamChanged();
	InvalidateVisualization();
	//Deprecate3DLayer();
}

ccGLMatrixd ecvDisplayTools::GenerateViewMat(CC_VIEW_ORIENTATION orientation)
{
	CCVector3d eye(0, 0, 0);
	CCVector3d center(0, 0, 0);
	CCVector3d top(0, 0, 0);

	//we look at (0,0,0) by default
	switch (orientation)
	{
	case CC_TOP_VIEW:
		eye.z = 1.0;
		top.y = 1.0;
		break;
	case CC_BOTTOM_VIEW:
		eye.z = -1.0;
		top.y = 1.0;
		break;
	case CC_FRONT_VIEW:
		eye.y = -1.0;
		top.z = 1.0;
		break;
	case CC_BACK_VIEW:
		eye.y = 1.0;
		top.z = 1.0;
		break;
	case CC_LEFT_VIEW:
		eye.x = -1.0;
		top.z = 1.0;
		break;
	case CC_RIGHT_VIEW:
		eye.x = 1.0;
		top.z = 1.0;
		break;
	case CC_ISO_VIEW_1:
		eye.x = -1.0;
		eye.y = -1.0;
		eye.z = 1.0;
		top.x = 1.0;
		top.y = 1.0;
		top.z = 1.0;
		break;
	case CC_ISO_VIEW_2:
		eye.x = 1.0;
		eye.y = 1.0;
		eye.z = 1.0;
		top.x = -1.0;
		top.y = -1.0;
		top.z = 1.0;
		break;
	}

	return ccGLMatrixd::FromViewDirAndUpDir(center - eye, top);
}

void ecvDisplayTools::SetView(CC_VIEW_ORIENTATION orientation, ccBBox* bbox)
{
	switch (orientation)
	{
	case CC_TOP_VIEW:
		SetCameraPosition(0, 0, 1, 0, 0, 0, 0, 1, 0);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	case CC_BOTTOM_VIEW:
		SetCameraPosition(0, 0, -1, 0, 0, 0, 0, 1, 0);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	case CC_FRONT_VIEW:
		SetCameraPosition(0, 1, 0, 0, 0, 0, 0, 0, 1);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	case CC_BACK_VIEW:
		SetCameraPosition(0, -1, 0, 0, 0, 0, 0, 0, 1);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	case CC_LEFT_VIEW:
		SetCameraPosition(-1, 0, 0, 0, 0, 0, 0, 0, 1);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	case CC_RIGHT_VIEW:
		SetCameraPosition(1.0, 0.0, 0, 0, 0, 0, 0, 0, 1);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	case CC_ISO_VIEW_1:
		SetCameraPosition(-1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	case CC_ISO_VIEW_2:
		SetCameraPosition(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0);
		UpdateConstellationCenterAndZoom(bbox, false);
		break;
	default:
		break;
	}
}

void ecvDisplayTools::SetView(CC_VIEW_ORIENTATION orientation, bool forceRedraw/*=true*/)
{
	bool wasViewerBased = !s_tools.instance->m_viewportParams.objectCenteredView;
	if (wasViewerBased)
		SetPerspectiveState(s_tools.instance->m_viewportParams.perspectiveView, true);

	s_tools.instance->m_viewportParams.viewMat = GenerateViewMat(orientation);

	if (wasViewerBased)
		SetPerspectiveState(s_tools.instance->m_viewportParams.perspectiveView, false);

	InvalidateVisualization();
	Deprecate3DLayer();

	//we emit the 'baseViewMatChanged' signal
	emit s_tools.instance->baseViewMatChanged(s_tools.instance->m_viewportParams.viewMat);
	emit s_tools.instance->cameraParamChanged();
	if (forceRedraw)
		RedrawDisplay();
}

inline float RoundScale(float equivalentWidth)
{
	//we compute the scale granularity (to avoid width values with a lot of decimals)
	int k = static_cast<int>(std::floor(std::log(equivalentWidth) / std::log(10.0f)));
	float granularity = std::pow(10.0f, static_cast<float>(k)) / 2.0f;
	//we choose the value closest to equivalentWidth with the right granularity
	return std::floor(std::max(equivalentWidth / granularity, 1.0f))*granularity;
}

float ecvDisplayTools::ComputePerspectiveZoom()
{
	//DGM: in fact it can be useful to compute it even in ortho mode :)
	//if (!m_viewportParams.perspectiveView)
	//	return m_viewportParams.zoom;

	//we compute the zoom equivalent to the corresponding camera position (inverse of above calculus)
	float currentFov_deg = GetFov();
	if (currentFov_deg < FLT_EPSILON)
		return 1.0f;

	//Camera center to pivot vector
    double zoomEquivalentDist = (s_tools.instance->m_viewportParams.getCameraCenter() -
                                 s_tools.instance->m_viewportParams.getPivotPoint()).norm();
    if (cloudViewer::LessThanEpsilon( zoomEquivalentDist ))
		return 1.0f;

	float screenSize = std::min(s_tools.instance->m_glViewport.width(), s_tools.instance->m_glViewport.height()) * s_tools.instance->m_viewportParams.pixelSize; //see how pixelSize is computed!
    return screenSize / static_cast<float>(zoomEquivalentDist * std::tan(cloudViewer::DegreesToRadians(currentFov_deg)));
}

ccGLMatrixd & ecvDisplayTools::GetModelViewMatrix()
{
	if (!s_tools.instance->m_validModelviewMatrix)
		UpdateModelViewMatrix();

	return s_tools.instance->m_viewMatd;
}

ccGLMatrixd & ecvDisplayTools::GetProjectionMatrix()
{
	if (!s_tools.instance->m_validProjectionMatrix)
		UpdateProjectionMatrix();

	return s_tools.instance->m_projMatd;
}

ccGLMatrixd ecvDisplayTools::ComputeProjectionMatrix(bool withGLfeatures,
                                                     ProjectionMetrics* metrics/*=nullptr*/,
                                                     double* eyeOffset/*=nullptr*/)
{
	double bbHalfDiag = 1.0;
	CCVector3d bbCenter(0, 0, 0);

	//compute center of visible objects constellation
	if (s_tools.instance->m_globalDBRoot || s_tools.instance->m_winDBRoot)
	{
		//get whole bounding-box
		ccBBox box;
		GetVisibleObjectsBB(box);
		if (box.isValid())
		{
			//get bbox center
			bbCenter = CCVector3d::fromArray(box.getCenter().u);
			//get half bbox diagonal length
			bbHalfDiag = box.getDiagNormd() / 2;
		}
	}

    CCVector3d cameraCenterToBBCenter = s_tools.instance->m_viewportParams.getCameraCenter() - bbCenter;
    double cameraToBBCenterDist = cameraCenterToBBCenter.normd();

	if (metrics)
	{
		metrics->bbHalfDiag = bbHalfDiag;
        metrics->cameraToBBCenterDist = cameraToBBCenterDist;
	}

    //virtual pivot point (i.e. to handle viewer-based mode smoothly)
    CCVector3d rotationCenter = s_tools.instance->m_viewportParams.getRotationCenter();

    //compute the maximum distance between the pivot point and the farthest displayed object
    double rotationCenterToFarthestObjectDist = 0.0;
    {
        //maximum distance between the pivot point and the farthest corner of the displayed objects bounding-box
        rotationCenterToFarthestObjectDist = (bbCenter - rotationCenter).norm() + bbHalfDiag;

        //(if enabled) the pivot symbol should always be visible in object-centere view mode
        if (s_tools.instance->m_pivotSymbolShown &&
            s_tools.instance->m_pivotVisibility != PIVOT_HIDE &&
            withGLfeatures &&
            s_tools.instance->m_viewportParams.objectCenteredView)
        {
            double pivotActualRadius = CC_DISPLAYED_PIVOT_RADIUS_PERCENT *
                std::min(s_tools.instance->m_glViewport.width(),
                s_tools.instance->m_glViewport.height()) / 2;
            double pivotSymbolScale = pivotActualRadius * ComputeActualPixelSize();
            rotationCenterToFarthestObjectDist = std::max(rotationCenterToFarthestObjectDist, pivotSymbolScale);
        }

        if (withGLfeatures && s_tools.instance->m_customLightEnabled)
        {
            //distance from custom light to pivot point
            double distToCustomLight = (rotationCenter - CCVector3d::fromArray(s_tools.instance->m_customLightPos)).norm();
            rotationCenterToFarthestObjectDist = std::max(rotationCenterToFarthestObjectDist, distToCustomLight);
        }

        rotationCenterToFarthestObjectDist *= 1.01; //for round-off issues
    }

    double cameraCenterToRotationCentertDist = 0;
    if (s_tools.instance->m_viewportParams.objectCenteredView)
    {
        cameraCenterToRotationCentertDist = s_tools.instance->m_viewportParams.getFocalDistance();
    }

    //we deduce zFar
    double zNear = cameraCenterToRotationCentertDist - rotationCenterToFarthestObjectDist;
    double zFar = cameraCenterToRotationCentertDist + rotationCenterToFarthestObjectDist;

    //compute the aspect ratio
    double ar = static_cast<double>(s_tools.instance->m_glViewport.height()) /
                                    s_tools.instance->m_glViewport.width();

    ccGLMatrixd projMatrix;
    if (s_tools.instance->m_viewportParams.perspectiveView)
    {
            //DGM: the 'zNearCoef' must not be too small, otherwise the loss in accuracy
            //for the detph buffer is too high and the display is jeopardized, especially
            //for entities with large coordinates)
            //zNear = zFar * m_viewportParams.zNearCoef;
            zNear = bbHalfDiag * s_tools.instance->m_viewportParams.zNearCoef; //we want a stable value!
            //zNear = std::max(bbHalfDiag * m_viewportParams.zNearCoef, zNear); //we want a stable value!
            zFar = std::max(zNear + ZERO_TOLERANCE_D, zFar);

            double xMax = zNear * s_tools.instance->m_viewportParams.computeDistanceToHalfWidthRatio();
            double yMax = xMax * ar;

            //DGM: we now take 'frustumAsymmetry' into account (for stereo rendering)
            double frustumAsymmetry = 0.0;
//            if (eyeOffset)
//            {
//                //see 'NVIDIA 3D VISION PRO AND STEREOSCOPIC 3D' White paper (Oct 2010, p. 12)
//                double convergence = std::abs(s_tools.instance->m_viewportParams.getFocalDistance());

//                //we assume zNear = screen distance
//                //double scale = zNear * m_stereoParams.stereoStrength / m_stereoParams.screenDistance_mm;	//DGM: we don't want to depend on the cloud size anymore
//                                                                                                            //as it can produce very strange visual effects on very large clouds
//                                                                                                            //we now keep something related to the focal distance (multiplied by
//                                                                                                            //the 'zNearCoef' that can be tweaked by the user if necessary)
//                double scale = convergence * s_tools.instance->m_viewportParams.zNearCoef *
//                        s_tools.instance->m_stereoParams.stereoStrength /
//                        s_tools.instance->m_stereoParams.screenDistance_mm;
//                double eyeSeperation = s_tools.instance->m_stereoParams.eyeSeparation_mm * scale;

//                //on input 'eyeOffset' should be -1 (left) or +1 (right)
//                *eyeOffset *= eyeSeperation;

//                frustumAsymmetry = (*eyeOffset) * zNear / convergence;

//            }

            projMatrix = ecvGenericDisplayTools::Frustum(-xMax - frustumAsymmetry, xMax - frustumAsymmetry, -yMax, yMax, zNear, zFar);
    }
	else
	{
        //zNear = std::max(zNear, 0.0);
        zFar = std::max(zNear + ZERO_TOLERANCE_D, zFar);

        //CVLog::Print(QString("cameraCenterToPivotDist = %0 / zNear = %1 / zFar = %2").arg(cameraCenterToPivotDist).arg(zNear).arg(zFar));

        double xMax = std::abs(cameraCenterToRotationCentertDist) * s_tools.instance->m_viewportParams.computeDistanceToHalfWidthRatio();
        double yMax = xMax * ar;

        projMatrix = ecvGenericDisplayTools::Ortho(-xMax, xMax, -yMax, yMax, zNear, zFar);
	}
    return projMatrix;
}

void ecvDisplayTools::UpdateProjectionMatrix()
{
	ProjectionMetrics metrics;

	s_tools.instance->m_projMatd = ComputeProjectionMatrix
	(
		true,
		&metrics,
		nullptr
	); //no stereo vision by default!

	s_tools.instance->m_viewportParams.zNear = metrics.zNear;
	s_tools.instance->m_viewportParams.zFar = metrics.zFar;
	s_tools.instance->m_cameraToBBCenterDist = metrics.cameraToBBCenterDist;
	s_tools.instance->m_bbHalfDiag = metrics.bbHalfDiag;

	s_tools.instance->m_validProjectionMatrix = true;
}

CCVector3d ecvDisplayTools::GetRealCameraCenter()
{
	//the camera center is always defined in perspective mode
	if (s_tools.instance->m_viewportParams.perspectiveView)
	{
        return s_tools.instance->m_viewportParams.getCameraCenter();
	}

	//in orthographic mode, we put the camera at the center of the
	//visible objects (along the viewing direction)
	ccBBox box;
	GetVisibleObjectsBB(box);

    return CCVector3d(s_tools.instance->m_viewportParams.getCameraCenter().x,
        s_tools.instance->m_viewportParams.getCameraCenter().y,
		box.isValid() ? box.getCenter().z : 0.0);
}

ccGLMatrixd ecvDisplayTools::ComputeModelViewMatrix()
{
    ccGLMatrixd viewMatd = s_tools.instance->m_viewportParams.computeViewMatrix();

    ccGLMatrixd scaleMatd = s_tools.instance->m_viewportParams.computeScaleMatrix(s_tools.instance->m_glViewport);

    return scaleMatd * viewMatd;
}

void ecvDisplayTools::UpdateModelViewMatrix()
{
	//we save visualization matrix
    s_tools.instance->m_viewMatd = ComputeModelViewMatrix();

	s_tools.instance->m_validModelviewMatrix = true;
}

void ecvDisplayTools::SetBaseViewMat(ccGLMatrixd& mat)
{
	s_tools.instance->m_viewportParams.viewMat = mat;

	InvalidateVisualization();

	//we emit the 'baseViewMatChanged' signal
	emit s_tools.instance->baseViewMatChanged(s_tools.instance->m_viewportParams.viewMat);
	emit s_tools.instance->cameraParamChanged();
}

void ecvDisplayTools::SetPerspectiveState(bool state, bool objectCenteredView)
{
	// previous state
	bool perspectiveWasEnabled = s_tools.instance->m_viewportParams.perspectiveView;
	bool viewWasObjectCentered = s_tools.instance->m_viewportParams.objectCenteredView;

	// new state
	s_tools.instance->m_viewportParams.perspectiveView = state;
	s_tools.instance->m_viewportParams.objectCenteredView = objectCenteredView;

	//Camera center to pivot vector
    CCVector3d PC = s_tools.instance->m_viewportParams.getCameraCenter() -
        s_tools.instance->m_viewportParams.getPivotPoint();

	if (s_tools.instance->m_viewportParams.perspectiveView)
	{
		if (!perspectiveWasEnabled) //from ortho. mode to perspective view
		{
			//we compute the camera position that gives 'quite' the same view as the ortho one
			//(i.e. we replace the zoom by setting the camera at the right distance from
			//the pivot point)
			double currentFov_deg = static_cast<double>(GetFov());
            assert(cloudViewer::GreaterThanEpsilon(currentFov_deg));
            // see how pixelSize is computed!
            double screenSize = std::min(s_tools.instance->m_glViewport.width(),
					s_tools.instance->m_glViewport.height())
                    * s_tools.instance->m_viewportParams.pixelSize;
			if (screenSize > 0.0)
			{
                PC.z = screenSize / (s_tools.instance->m_viewportParams.zoom * 
					std::tan(cloudViewer::DegreesToRadians(currentFov_deg)));
			}
		}

		//display message
		DisplayNewMessage(objectCenteredView ? 
			"Centered perspective ON" :
			"Viewer-based perspective ON",
			LOWER_LEFT_MESSAGE,
			false,
			2,
			PERSPECTIVE_STATE_MESSAGE);
	}
	else
	{
        // object-centered mode is forced for otho. view
		s_tools.instance->m_viewportParams.objectCenteredView = true; 

		if (perspectiveWasEnabled) //from perspective view to ortho. view
		{
			//we compute the zoom equivalent to the corresponding camera position (inverse of above calculus)
			float newZoom = ComputePerspectiveZoom();
			SetZoom(newZoom);
		}

		DisplayNewMessage("Perspective OFF",
			LOWER_LEFT_MESSAGE,
			false,
			2,
			PERSPECTIVE_STATE_MESSAGE);
	}

	//if we change form object-based to viewer-based visualization, we must
	//'rotate' around the object (or the opposite ;)
	if (viewWasObjectCentered && !s_tools.instance->m_viewportParams.objectCenteredView)
	{
		s_tools.instance->m_viewportParams.viewMat.transposed().apply(PC); //inverse rotation
	}
	else if (!viewWasObjectCentered && s_tools.instance->m_viewportParams.objectCenteredView)
	{
		s_tools.instance->m_viewportParams.viewMat.apply(PC);
	}

    SetCameraPos(s_tools.instance->m_viewportParams.getPivotPoint() + PC);

	emit s_tools.instance->perspectiveStateChanged();
	emit s_tools.instance->cameraParamChanged();

	//auto-save last perspective settings
	{
		QSettings settings;
		settings.beginGroup(c_ps_groupName);
		//write parameters
		settings.setValue(c_ps_perspectiveView, s_tools.instance->m_viewportParams.perspectiveView);
		settings.setValue(c_ps_objectMode, s_tools.instance->m_viewportParams.objectCenteredView);
		settings.endGroup();
	}

	s_tools.instance->m_bubbleViewModeEnabled = false;
	InvalidateViewport();
	InvalidateVisualization();
	Deprecate3DLayer();
}

void ecvDisplayTools::UpdateConstellationCenterAndZoom(const ccBBox* aBox, bool redraw)
{

	if (s_tools.instance->m_bubbleViewModeEnabled)
	{
		CVLog::Warning("[updateConstellationCenterAndZoom] Not when bubble-view is enabled!");
		return;
	}

	SetZoom(1.0f);

	ccBBox zoomedBox;

	//the user has provided a valid bounding box
	if (aBox)
	{
		zoomedBox = (*aBox);
	}
	else //otherwise we'll take the default one (if possible)
	{
		GetVisibleObjectsBB(zoomedBox);
	}

	if (!zoomedBox.isValid())
		return;

	if (redraw)
	{
		InvalidateViewport();
		InvalidateVisualization();
		Deprecate3DLayer();
		RedrawDisplay();
	}

	ResetCamera(&zoomedBox);
	UpdateScreen();

	//we get the bounding-box diagonal length
	double bbDiag = static_cast<double>(zoomedBox.getDiagNorm());

    if (cloudViewer::LessThanEpsilon(bbDiag))
	{
		CVLog::Warning("[ecvDisplayTools] Entity/DB has a null bounding-box! Can't zoom in...");
		return;
	}

	//we compute the pixel size (in world coordinates)
	{
		int minScreenSize = std::min(s_tools.instance->m_glViewport.width(), s_tools.instance->m_glViewport.height());
		SetPixelSize(minScreenSize > 0 ? static_cast<float>(bbDiag / minScreenSize) : 1.0f);
	}

	//we set the pivot point on the box center
	CCVector3d P = CCVector3d::fromArray(zoomedBox.getCenter().u);
	SetPivotPoint(P);
}

void ecvDisplayTools::SetRedrawRecursive(bool redraw/* = false*/)
{
	GetSceneDB()->setRedrawFlagRecursive(redraw);
	GetOwnDB()->setRedrawFlagRecursive(redraw);
}

void ecvDisplayTools::UpdateNamePoseRecursive()
{
	GetSceneDB()->updateNameIn3DRecursive();
	GetOwnDB()->updateNameIn3DRecursive();
}

void ecvDisplayTools::SetRedrawRecursive(ccHObject * obj, bool redraw/* = false*/)
{
	assert(obj);
	obj->setRedrawFlagRecursive(redraw);
}

void ecvDisplayTools::GetVisibleObjectsBB(ccBBox& box)
{
	//compute center of visible objects constellation
	if (s_tools.instance->m_globalDBRoot)
	{
		//get whole bounding-box
		box = s_tools.instance->m_globalDBRoot->getDisplayBB_recursive(false);
	}

	//incorporate window own db
	if (s_tools.instance->m_winDBRoot)
	{
		ccBBox ownBox = s_tools.instance->m_winDBRoot->getDisplayBB_recursive(false);
		if (ownBox.isValid())
		{
			box += ownBox;
		}
	}
}

ENTITY_TYPE ecvDisplayTools::ConvertToEntityType(const CV_CLASS_ENUM& type) {
	ENTITY_TYPE entityType = ENTITY_TYPE::ECV_NONE;
	switch (type)
	{
	case CV_TYPES::HIERARCHY_OBJECT:
		entityType = ENTITY_TYPE::ECV_HIERARCHY_OBJECT;
		break;	
	case CV_TYPES::POINT_CLOUD:
		entityType = ENTITY_TYPE::ECV_POINT_CLOUD;
		break;
	case CV_TYPES::POLY_LINE:
    case CV_TYPES::LINESET:
		entityType = ENTITY_TYPE::ECV_SHAPE;
		break;		
	case CV_TYPES::LABEL_2D:
		entityType = ENTITY_TYPE::ECV_2DLABLE;
		break;	
	case CV_TYPES::VIEWPORT_2D_LABEL:
		entityType = ENTITY_TYPE::ECV_2DLABLE_VIEWPORT;
		break;
	case CV_TYPES::POINT_OCTREE:
		entityType = ENTITY_TYPE::ECV_OCTREE;
		break;	
	case CV_TYPES::POINT_KDTREE:
		entityType = ENTITY_TYPE::ECV_KDTREE;
		break;
	case CV_TYPES::FACET:
	case CV_TYPES::PRIMITIVE:
	case CV_TYPES::MESH:
    case CV_TYPES::SUB_MESH:
	case CV_TYPES::SPHERE:
	case CV_TYPES::CONE:
	case CV_TYPES::PLANE:
	case CV_TYPES::CYLINDER:
	case CV_TYPES::TORUS:
	case CV_TYPES::EXTRU:
	case CV_TYPES::DISH:
	case CV_TYPES::BOX:
    case CV_TYPES::COORDINATESYSTEM:
	case CV_TYPES::QUADRIC:
		entityType = ENTITY_TYPE::ECV_MESH;
		break;
	case CV_TYPES::IMAGE:
		entityType = ENTITY_TYPE::ECV_IMAGE;
		break;
	case CV_TYPES::SENSOR:
	case CV_TYPES::GBL_SENSOR:
	case CV_TYPES::CAMERA_SENSOR:
		entityType = ENTITY_TYPE::ECV_SENSOR;
		break;
	default:
		break;
	}
    return entityType;
}

void ecvDisplayTools::DisplayOverlayEntities(bool state)
{
    s_tools.instance->m_displayOverlayEntities = state;
    if (!state) {
        ClearBubbleView();
    }
}

void ecvDisplayTools::SetSceneDB(ccHObject * root)
{
	s_tools.instance->m_globalDBRoot = root;
	ZoomGlobal();
}

void ecvDisplayTools::AddToOwnDB(ccHObject* obj, bool noDependency/*=true*/)
{
	if (!obj)
	{
		assert(false);
		return;
	}

	if (s_tools.instance->m_winDBRoot)
	{
		s_tools.instance->m_winDBRoot->addChild(obj, noDependency ?
			ccHObject::DP_NONE : ccHObject::DP_PARENT_OF_OTHER);
	}
	else
	{
		CVLog::Error("[ecvDisplayTools::addToOwnDB] Window has no DB!");
	}
}

void ecvDisplayTools::RemoveFromOwnDB(ccHObject* obj)
{
	if (s_tools.instance->m_winDBRoot)
		s_tools.instance->m_winDBRoot->removeChild(obj);
}

void ecvDisplayTools::SetRemoveViewIDs(std::vector<removeInfo> & removeinfos)
{
	if (removeinfos.size() > 0)
	{
		s_tools.instance->m_removeInfos = removeinfos;
		s_tools.instance->m_removeFlag = true;
	}
	else 
	{
		s_tools.instance->m_removeFlag = false;
    }
}

void ecvDisplayTools::ZoomCamera(double zoomFactor, int viewport)
{
    TheInstance()->zoomCamera(zoomFactor, viewport);
    if (!TheInstance()->m_viewportParams.perspectiveView)
    {
        TheInstance()->UpdateZoom(static_cast<float>(zoomFactor));
    }
    UpdateDisplayParameters();
}

void ecvDisplayTools::SetInteractionMode(INTERACTION_FLAGS flags)
{
	s_tools.instance->m_interactionFlags = flags;

	//we need to explicitely enable 'mouse tracking' to track the mouse when no button is clicked
#ifdef CV_GL_WINDOW_USE_QWINDOW
	if (m_parentWidget)
	{
		m_parentWidget->setMouseTracking(flags & (INTERACT_CLICKABLE_ITEMS | INTERACT_SIG_MOUSE_MOVED));
	}
#else
		GetCurrentScreen()->setMouseTracking(flags & (INTERACT_CLICKABLE_ITEMS | INTERACT_SIG_MOUSE_MOVED));
#endif

	if ((flags & INTERACT_CLICKABLE_ITEMS) == 0)
	{
		//auto-hide the embedded icons if they are disabled
		s_tools.instance->m_clickableItemsVisible = false;
	}
}

CCVector3d ecvDisplayTools::GetCurrentViewDir()
{
	//view direction is (the opposite of) the 3rd line of the current view matrix
	const double* M = s_tools.instance->m_viewportParams.viewMat.data();
	CCVector3d axis(-M[2], -M[6], -M[10]);
	axis.normalize();

	return axis;
}

CCVector3d ecvDisplayTools::GetCurrentUpDir()
{
	//otherwise up direction is the 2nd line of the current view matrix
	const double* M = s_tools.instance->m_viewportParams.viewMat.data();
	CCVector3d axis(M[1], M[5], M[9]);
	axis.normalize();

	return axis;
}

float ecvDisplayTools::GetFov()
{
	return (s_tools.instance->m_bubbleViewModeEnabled ? 
			s_tools.instance->m_bubbleViewFov_deg :
			s_tools.instance->m_viewportParams.fov_deg);
}

void ecvDisplayTools::SetupProjectiveViewport(const ccGLMatrixd& cameraMatrix,
    float fov_deg/*=0.0f*/,
	float ar/*=1.0f*/,
	bool viewerBasedPerspective/*=true*/,
	bool bubbleViewMode/*=false*/)
{
	//perspective (viewer-based by default)
	if (bubbleViewMode)
		SetBubbleViewMode(true);
	else
		SetPerspectiveState(true, !viewerBasedPerspective);

	//field of view (= OpenGL 'fovy' but in degrees)
	if (fov_deg > 0.0f)
	{
		SetFov(fov_deg);
	}

	//aspect ratio
	SetAspectRatio(ar);

	//set the camera matrix 'translation' as OpenGL camera center
	CCVector3d T = cameraMatrix.getTranslationAsVec3D();
    CCVector3d UP = cameraMatrix.getColumnAsVec3D(1);
    cameraMatrix.applyRotation(UP.data());
    SetCameraPosition(T.data(), UP.data());
	SetCameraPos(T);
	if (viewerBasedPerspective && s_tools.instance->m_autoPickPivotAtCenter)
	{
		SetPivotPoint(T);
	}

	//apply orientation matrix
	ccGLMatrixd trans = cameraMatrix;
	trans.clearTranslation();
	trans.invert();
	SetBaseViewMat(trans);

    ResetCameraClippingRange();
    UpdateScreen();
}

void ecvDisplayTools::SetAspectRatio(float ar)
{
	if (ar < 0.0f)
	{
		CVLog::Warning("[ecvDisplayTools::setAspectRatio] Invalid AR value!");
		return;
	}

	if (s_tools.instance->m_viewportParams.cameraAspectRatio != ar)
	{
		//update param
		s_tools.instance->m_viewportParams.cameraAspectRatio = ar;

        //and camera state
        InvalidateViewport();
        InvalidateVisualization();
        Deprecate3DLayer();
	}
}

void ecvDisplayTools::SetFov(float fov_deg)
{
    if (cloudViewer::LessThanEpsilon(fov_deg) || fov_deg > 180.0f)
	{
		CVLog::Warning("[ecvDisplayTools::setFov] Invalid FOV value!");
		return;
	}

	//derivation if we are in bubble-view mode
	if (s_tools.instance->m_bubbleViewModeEnabled)
	{
		SetBubbleViewFov(fov_deg);
	}
	else if (s_tools.instance->m_viewportParams.fov_deg != fov_deg)
	{
		//update param
		s_tools.instance->m_viewportParams.fov_deg = fov_deg;
		//and camera state (if perspective view is 'on')
		{
			SetCameraFovy(fov_deg);
			InvalidateViewport();
			InvalidateVisualization();
			Deprecate3DLayer();

			DisplayNewMessage(QString("F.O.V. = %1 deg.").arg(fov_deg, 0, 'f', 1),
				LOWER_LEFT_MESSAGE, //DGM HACK: we cheat and use the same 'slot' as the window size
				false,
				2,
				SCREEN_SIZE_MESSAGE);
		}

		emit s_tools.instance->fovChanged(s_tools.instance->m_viewportParams.fov_deg);
		emit s_tools.instance->cameraParamChanged();
	}
}

void ecvDisplayTools::DisplayNewMessage(const QString& message,
	MessagePosition pos,
	bool append/*=false*/,
	int displayMaxDelay_sec/*=2*/,
	MessageType type/*=CUSTOM_MESSAGE*/)
{
	if (message.isEmpty())
	{
		if (!append)
		{
			std::list<MessageToDisplay>::iterator it = s_tools.instance->m_messagesToDisplay.begin();
			while (it != s_tools.instance->m_messagesToDisplay.end())
			{
				//same position? we remove the message
				if (it->position == pos)
				{
					RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, it->message));
					it = s_tools.instance->m_messagesToDisplay.erase(it);
				}
				else
					++it;
			}
		}
		else
		{
			CVLog::Warning("[ecvDisplayTools::DisplayNewMessage] Appending an empty message has no effect!");
		}
		return;
	}

	//shall we replace the equivalent message(if any)?
	if (!append)
	{
		//only if type is not 'custom'
		if (type != CUSTOM_MESSAGE)
		{
			for (std::list<MessageToDisplay>::iterator it = 
				s_tools.instance->m_messagesToDisplay.begin(); 
				it != s_tools.instance->m_messagesToDisplay.end();)
			{
				//same type? we remove it
				if (it->type == type)
				{
					RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, it->message));
					it = s_tools.instance->m_messagesToDisplay.erase(it);
				}
				else
					++it;
			}
		}
	}
	else
	{
		if (pos == SCREEN_CENTER_MESSAGE)
		{
			CVLog::Warning("[ecvDisplayTools::DisplayNewMessage] Append is not supported for center screen messages!");
		}
	}

	MessageToDisplay mess;
	mess.message = message;
	mess.messageValidity_sec = s_tools.instance->m_timer.elapsed() / 1000 + displayMaxDelay_sec;
	mess.position = pos;
	mess.type = type;
	s_tools.instance->m_messagesToDisplay.push_back(mess);
	//CVLog::Print(QString("[DisplayNewMessage] New message valid until %1 s.").arg(mess.messageValidity_sec));
}

void ecvDisplayTools::SetPivotPoint(const CCVector3d& P,
	bool autoUpdateCameraPos/*=false*/,
	bool verbose/*=false*/)
{
	if (autoUpdateCameraPos &&
		(!s_tools.instance->m_viewportParams.perspectiveView || 
			s_tools.instance->m_viewportParams.objectCenteredView))
	{
		//compute the equivalent camera center
        CCVector3d dP = s_tools.instance->m_viewportParams.getPivotPoint() - P;
		CCVector3d MdP = dP; s_tools.instance->m_viewportParams.viewMat.applyRotation(MdP);
        CCVector3d newCameraPos = s_tools.instance->m_viewportParams.getCameraCenter() + MdP - dP;
		SetCameraPos(newCameraPos);
	}

    s_tools.instance->m_viewportParams.setPivotPoint(P, true);
	SetAutoUpateCameraPos(autoUpdateCameraPos);
	SetCenterOfRotation(P);

    emit s_tools.instance->pivotPointChanged(s_tools.instance->m_viewportParams.getPivotPoint());
	emit s_tools.instance->cameraParamChanged();

	if (verbose)
	{
		const unsigned& precision = GetDisplayParameters().displayedNumPrecision;
		DisplayNewMessage(QString(), LOWER_LEFT_MESSAGE, false); //clear previous message
		DisplayNewMessage(QString("Point (%1 ; %2 ; %3) set as rotation center")
			.arg(P.x, 0, 'f', precision)
			.arg(P.y, 0, 'f', precision)
			.arg(P.z, 0, 'f', precision),
			LOWER_LEFT_MESSAGE, true);
		RedrawDisplay(true, false);
	}

	//s_tools.instance->m_autoPivotCandidate = CCVector3d(0, 0, 0);
	s_tools.instance->m_autoPivotCandidate = P;
	InvalidateViewport();
	InvalidateVisualization();

}

void ecvDisplayTools::SetAutoPickPivotAtCenter(bool state)
{
	if (s_tools.instance->m_autoPickPivotAtCenter != state)
	{
		s_tools.instance->m_autoPickPivotAtCenter = state;

		if (state)
		{
			//force 3D redraw to update the coordinates of the 'auto' pivot center
			s_tools.instance->m_autoPivotCandidate = CCVector3d(0, 0, 0);
			//RedrawDisplay(false);
		}
	}
}

void ecvDisplayTools::GetContext(CC_DRAW_CONTEXT& CONTEXT)
{
	//display size
	CONTEXT.glW = s_tools.instance->m_glViewport.width();
	CONTEXT.glH = s_tools.instance->m_glViewport.height();
	CONTEXT.devicePixelRatio = static_cast<float>(GetDevicePixelRatio());
	CONTEXT.drawingFlags = 0;

	const ecvGui::ParamStruct& guiParams = GetDisplayParameters();

	//decimation options
	CONTEXT.decimateCloudOnMove = guiParams.decimateCloudOnMove;
	CONTEXT.minLODPointCount = guiParams.minLoDCloudSize;
	CONTEXT.minLODTriangleCount = guiParams.minLoDMeshSize;
	CONTEXT.higherLODLevelsAvailable = false;
	CONTEXT.moreLODPointsAvailable = false;
	CONTEXT.currentLODLevel = 0;

	//scalar field color-bar
	CONTEXT.sfColorScaleToDisplay = nullptr;

	//point picking
	CONTEXT.labelMarkerSize = static_cast<float>(guiParams.labelMarkerSize * ComputeActualPixelSize());
	CONTEXT.labelMarkerTextShift_pix = 5; //5 pixels shift

	//text display
	CONTEXT.dispNumberPrecision = guiParams.displayedNumPrecision;
	//label opacity
	CONTEXT.labelOpacity = guiParams.labelOpacity;

	//default colors
	CONTEXT.pointsDefaultCol = guiParams.pointsDefaultCol;
	CONTEXT.textDefaultCol = guiParams.textDefaultCol;
	CONTEXT.labelDefaultBkgCol = guiParams.labelBackgroundCol;
	CONTEXT.labelDefaultMarkerCol = guiParams.labelMarkerCol;
	CONTEXT.bbDefaultCol = guiParams.bbDefaultCol;

	//display acceleration
	CONTEXT.useVBOs = guiParams.useVBOs;

	//other options
	CONTEXT.drawRoundedPoints = guiParams.drawRoundedPoints;
}

bool ecvDisplayTools::RenderToFile(QString filename,
	float zoomFactor/*=1.0*/,
	bool dontScaleFeatures/*=false*/,
	bool renderOverlayItems/*=false*/)
{
	if (filename.isEmpty() || zoomFactor < 1.0e-2f)
	{
		return false;
	}

    QImage outputImage = RenderToImage(static_cast<int>(zoomFactor),
                                       renderOverlayItems, false, 0);

	if (outputImage.isNull())
	{
		//an error occurred (message should have already been issued!)
		return false;
	}

	if (GetDisplayParameters().drawRoundedPoints)
	{
		//convert the image to plain RGB to avoid issues with points transparency when saving to PNG
        outputImage = outputImage.convertToFormat(QImage::Format_RGB32);
	}

	bool success = outputImage.convertToFormat(QImage::Format_RGB32).save(filename);
	if (success)
	{
		CVLog::Print(QString("[Snapshot] File '%1' saved! (%2 x %3 pixels)").arg(filename).arg(outputImage.width()).arg(outputImage.height()));
	}
	else
	{
		CVLog::Print(QString("[Snapshot] Failed to save file '%1'!").arg(filename));
	}

	return success;
}

void ecvDisplayTools::SetCameraPos(const CCVector3d& P)
{
    if ((s_tools.instance->m_viewportParams.getCameraCenter() - P).norm2d() != 0.0)
	{
        s_tools.instance->m_viewportParams.setCameraCenter(P);
        SetCameraPosition(P);
        emit s_tools.instance->cameraPosChanged(s_tools.instance->m_viewportParams.getCameraCenter());
		emit s_tools.instance->cameraParamChanged();
		InvalidateViewport();
		InvalidateVisualization();
		Deprecate3DLayer();
	}
}

const ecvGui::ParamStruct& ecvDisplayTools::GetDisplayParameters()
{
	return s_tools.instance->m_overridenDisplayParametersEnabled ? 
		s_tools.instance->m_overridenDisplayParameters : ecvGui::Parameters();
}

void ecvDisplayTools::GetGLCameraParameters(ccGLCameraParameters & params)
{
	// get/compute the modelview matrix
	{
		GetViewMatrix(params.modelViewMat.data());
	}

	// get/compute the projection matrix
	{
		GetProjectionMatrix(params.projectionMat.data());
	}

    ccGLMatrixd rotationMat;
    rotationMat.setRotation(ccGLMatrixd::ToEigenMatrix3(params.modelViewMat).data());
    s_tools.instance->m_viewportParams.viewMat = rotationMat;
	double nearFar[2];
	GetCameraClip(nearFar);
	s_tools.instance->m_viewportParams.zNear = nearFar[0];
	s_tools.instance->m_viewportParams.zFar = nearFar[1];
    s_tools.instance->m_viewportParams.fov_deg = static_cast<float>(GetCameraFovy());
	params.fov_deg = s_tools.instance->m_viewportParams.fov_deg;

	params.viewport[0] = 0;
	params.viewport[1] = 0;
	params.viewport[2] = Width() * GetDevicePixelRatio();
	params.viewport[3] = Height() * GetDevicePixelRatio();
	SetGLViewport(QRect(0, 0, Width(), Height()));

	params.perspective = s_tools.instance->m_viewportParams.perspectiveView;
	params.pixelSize = s_tools.instance->m_viewportParams.pixelSize;
}

void ecvDisplayTools::SetDisplayParameters(const ecvGui::ParamStruct &params)
{
	s_tools.instance->m_overridenDisplayParametersEnabled = true;
	s_tools.instance->m_overridenDisplayParameters = params;
	ecvGui::Set(params);
}

void ecvDisplayTools::UpdateDisplayParameters()
{
	// set camera near and far
	double nearFar[2];
	GetCameraClip(nearFar);
	s_tools.instance->m_viewportParams.zNear = nearFar[0];
	s_tools.instance->m_viewportParams.zFar = nearFar[1];

    ccGLMatrixd viewMat;
    GetViewMatrix(viewMat.data());
    ccGLMatrixd rotationMat;
    rotationMat.setRotation(ccGLMatrixd::ToEigenMatrix3(viewMat).data());
    s_tools.instance->m_viewportParams.viewMat = rotationMat;

	// set camera fov
    s_tools.instance->m_viewportParams.fov_deg = static_cast<float>(GetCameraFovy());

	if (s_tools.instance->m_viewportParams.perspectiveView)
	{
		s_tools.instance->m_viewportParams.zoom = ComputePerspectiveZoom();
	}

	// set camera pos
	double pos[3];
	GetCameraPos(pos);
    s_tools.instance->m_viewportParams.setCameraCenter(CCVector3d::fromArray(pos), true);

	// set camera focal
	double focal[3];
	GetCameraFocal(focal);
	s_tools.instance->m_viewportParams.focal = CCVector3d::fromArray(focal);

	// set camera pos
	double up[3];
	GetCameraUp(up);
	s_tools.instance->m_viewportParams.up = CCVector3d::fromArray(up);
}

void ecvDisplayTools::SetViewportParameters(const ecvViewportParameters& params)
{
	ecvViewportParameters oldParams = s_tools.instance->m_viewportParams;
	s_tools.instance->m_viewportParams = params;
	if (params.perspectiveView)
	{
		SetCameraFovy(params.fov_deg);
		SetCameraClip(params.zNear, params.zFar);
	}

    SetCameraPosition(params.getCameraCenter().u, params.focal.u, params.up.u);
	//Update();

	InvalidateViewport();
	InvalidateVisualization();
	Deprecate3DLayer();

	emit s_tools.instance->baseViewMatChanged(s_tools.instance->m_viewportParams.viewMat);
    emit s_tools.instance->pivotPointChanged(s_tools.instance->m_viewportParams.getPivotPoint());
    emit s_tools.instance->cameraPosChanged(s_tools.instance->m_viewportParams.getCameraCenter());
	emit s_tools.instance->fovChanged(s_tools.instance->m_viewportParams.fov_deg);
	emit s_tools.instance->cameraParamChanged();
}

const ecvViewportParameters & ecvDisplayTools::GetViewportParameters()
{
	UpdateDisplayParameters();
	return s_tools.instance->m_viewportParams;
}

void ecvDisplayTools::SetBubbleViewMode(bool state)
{
	//Backup the camera center before entering this mode!
	bool bubbleViewModeWasEnabled = s_tools.instance->m_bubbleViewModeEnabled;
	if (!s_tools.instance->m_bubbleViewModeEnabled && state)
	{
		s_tools.instance->m_preBubbleViewParameters = s_tools.instance->m_viewportParams;
	}

	if (state)
	{
		//bubble-view mode = viewer-based perspective mode
		//setPerspectiveState must be called first as it
		//automatically deactivates bubble-view mode!
		SetPerspectiveState(true, false);

		s_tools.instance->m_bubbleViewModeEnabled = true;

		//when entering this mode, we reset the f.o.v.
		s_tools.instance->m_bubbleViewFov_deg = 0.0f; //to trick the signal emission mechanism
		SetBubbleViewFov(90.0f);
	}
	else if (bubbleViewModeWasEnabled)
	{
		s_tools.instance->m_bubbleViewModeEnabled = false;
		SetPerspectiveState(s_tools.instance->m_preBubbleViewParameters.perspectiveView, 
			s_tools.instance->m_preBubbleViewParameters.objectCenteredView);

		//when leaving this mode, we restore the original camera center
		SetViewportParameters(s_tools.instance->m_preBubbleViewParameters);
	}
}

void ecvDisplayTools::SetBubbleViewFov(float fov_deg)
{
	if (fov_deg < FLT_EPSILON || fov_deg > 180.0f)
		return;

	if (fov_deg != s_tools.instance->m_bubbleViewFov_deg)
	{
		s_tools.instance->m_bubbleViewFov_deg = fov_deg;

		if (s_tools.instance->m_bubbleViewModeEnabled)
		{
			InvalidateViewport();
			InvalidateVisualization();
			Deprecate3DLayer();
			emit s_tools.instance->fovChanged(s_tools.instance->m_bubbleViewFov_deg);
			emit s_tools.instance->cameraParamChanged();
		}
	}
}

void ecvDisplayTools::SetPixelSize(float pixelSize) 
{
	if (s_tools.instance->m_viewportParams.pixelSize != pixelSize)
	{
		s_tools.instance->m_viewportParams.pixelSize = pixelSize;
	}
	InvalidateViewport();
	InvalidateVisualization();
	Deprecate3DLayer();
}

void ecvDisplayTools::SetZoom(float value)
{
	//zoom should be avoided in bubble-view mode
	assert(!s_tools.instance->m_bubbleViewModeEnabled);

	if (value < CC_GL_MIN_ZOOM_RATIO)
		value = CC_GL_MIN_ZOOM_RATIO;
	else if (value > CC_GL_MAX_ZOOM_RATIO)
		value = CC_GL_MAX_ZOOM_RATIO;

	if (s_tools.instance->m_viewportParams.zoom != value)
	{
		s_tools.instance->m_viewportParams.zoom = value;
		InvalidateViewport();
		InvalidateVisualization();
		//Deprecate3DLayer();
	}
}

void ecvDisplayTools::UpdateZoom(float zoomFactor)
{
	//no 'zoom' in viewer based perspective
	assert(!s_tools.instance->m_viewportParams.perspectiveView);

	if (zoomFactor > 0.0f && zoomFactor != 1.0f)
	{
		SetZoom(s_tools.instance->m_viewportParams.zoom*zoomFactor);
	}
}

void ecvDisplayTools::SetPickingMode(PICKING_MODE mode/*=DEFAULT_PICKING*/)
{
	//is the picking mode locked?
	if (s_tools.instance->m_pickingModeLocked)
	{
		if ((mode != s_tools.instance->m_pickingMode) && (mode != DEFAULT_PICKING))
			CVLog::Warning("[ecvDisplayTools::setPickingMode] Picking mode is locked! Can't change it...");
		return;
	}

	switch (mode)
	{
	case DEFAULT_PICKING:
		mode = ENTITY_PICKING;
	case NO_PICKING:
	case ENTITY_PICKING:
		GetCurrentScreen()->setCursor(QCursor(Qt::ArrowCursor));
		break;
	case POINT_OR_TRIANGLE_PICKING:
	case TRIANGLE_PICKING:
	case POINT_PICKING:
		GetCurrentScreen()->setCursor(QCursor(Qt::PointingHandCursor));
		break;
	default:
		break;
	}

	s_tools.instance->m_pickingMode = mode;

	//CVLog::Warning(QString("[%1] Picking mode set to: ").arg(m_uniqueID) + ToString(m_pickingMode));
}

double ecvDisplayTools::ComputeActualPixelSize()
{
	if (!s_tools.instance->m_viewportParams.perspectiveView)
	{
		return static_cast<double>(s_tools.instance->m_viewportParams.pixelSize / s_tools.instance->m_viewportParams.zoom);
	}

	int minScreenDim = std::min(s_tools.instance->m_glViewport.width(), s_tools.instance->m_glViewport.height());
	if (minScreenDim <= 0)
		return 1.0;

	//Camera center to pivot vector
    double zoomEquivalentDist = (s_tools.instance->m_viewportParams.getCameraCenter() -
                                 s_tools.instance->m_viewportParams.getPivotPoint()).norm();

	double currentFov_deg = static_cast<double>(GetFov());
	return zoomEquivalentDist * std::tan(std::min(currentFov_deg, 75.0) * CV_DEG_TO_RAD) / minScreenDim; //tan(75) = 3.73 (then it quickly increases!)
}

void ecvDisplayTools::ShowPivotSymbol(bool state)
{
	//is the pivot really going to be drawn?
	if (state && !s_tools.instance->m_pivotSymbolShown &&
		s_tools.instance->m_viewportParams.objectCenteredView &&
		s_tools.instance->m_pivotVisibility != PIVOT_HIDE)
	{
		InvalidateViewport();
		Deprecate3DLayer();
	}

	s_tools.instance->m_pivotSymbolShown = state;
}

int ecvDisplayTools::GetFontPointSize()
{
	return (s_tools.instance->m_captureMode.enabled ? 
		FontSizeModifier(GetDisplayParameters().defaultFontSize, s_tools.instance->m_captureMode.zoomFactor) :
		GetDisplayParameters().defaultFontSize) * GetDevicePixelRatio();
}

int ecvDisplayTools::GetLabelFontPointSize()
{
	return (s_tools.instance->m_captureMode.enabled ? 
		FontSizeModifier(GetDisplayParameters().labelFontSize, s_tools.instance->m_captureMode.zoomFactor) : 
		GetDisplayParameters().labelFontSize) * GetDevicePixelRatio();
}

QFont ecvDisplayTools::GetLabelDisplayFont()
{
	QFont font = s_tools.instance->m_font;
	font.setPointSize(GetLabelFontPointSize());
	return font;
}

void ecvDisplayTools::SetFocusToScreen()
{
#ifdef CV_WINDOWS
	POINT lpPoint;
	POINT oldlpPoint;
	QPoint globalOldPoint = QCursor::pos();
	oldlpPoint.x = globalOldPoint.x();
	oldlpPoint.y = globalOldPoint.y();

	QRect screenRect = GetScreenRect();
	QPoint clickPos = screenRect.center();
	lpPoint.x = clickPos.x();
	lpPoint.y = clickPos.y();

	SetCursorPos(lpPoint.x, lpPoint.y);
	mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, 0, 0, 0, 0);
	Sleep(20);
	SetCursorPos(oldlpPoint.x, oldlpPoint.y);
#else
	CVLog::Warning("only supported in windows!");
#endif

	if (GetCurrentScreen())
	{
		GetCurrentScreen()->setFocus();
		if (GetCurrentScreen()->parentWidget())
		{
			GetCurrentScreen()->parentWidget()->setFocus();
		}
	}
}

void ecvDisplayTools::ToBeRefreshed()
{
	s_tools.instance->m_shouldBeRefreshed = true;

	InvalidateViewport();
	InvalidateVisualization();
}

void ecvDisplayTools::RefreshDisplay(bool only2D/*=false*/, bool forceRedraw/* = true*/)
{
	if (s_tools.instance->m_shouldBeRefreshed)
	{
		RedrawDisplay(only2D, forceRedraw);
	}
}

void ecvDisplayTools::RedrawDisplay(bool only2D/*=false*/, bool forceRedraw/* = true*/)
{
	//visual traces
	if (s_tools.instance->m_showDebugTraces)
	{
		// clear history
		RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
		if (!s_tools.instance->m_diagStrings.isEmpty())
		{
			QStringList::iterator it = s_tools.instance->m_diagStrings.begin();
			while (it != s_tools.instance->m_diagStrings.end())
			{
				//no more valid? we delete the message
				RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
				it = s_tools.instance->m_diagStrings.erase(it);
			}
		}

		s_tools.instance->m_diagStrings << QString("only2D : %1").arg(only2D ? "true" : "false");
		s_tools.instance->m_diagStrings << QString("ForceRedraw : %1").arg(forceRedraw ? "true" : "false");
	}
	else
	{
		RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
		if (!s_tools.instance->m_diagStrings.isEmpty())
		{
			QStringList::iterator it = s_tools.instance->m_diagStrings.begin();
			while (it != s_tools.instance->m_diagStrings.end())
			{
				//no more valid? we delete the message
				RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
				it = s_tools.instance->m_diagStrings.erase(it);
			}
		}
	}

	CheckIfRemove();
	if (s_tools.instance->m_removeAllFlag)
	{
		Update();
		return;
	}

	//we update font size (for text display)
	SetFontPointSize(GetFontPointSize());

	if (!only2D)
	{
		//force the 3D layer to be redrawn
		Deprecate3DLayer();
	}

	bool drawBackground = false;
	bool draw3DPass = false;
	bool drawForeground = true;
	bool draw3DCross = GetDisplayParameters().displayCross;

	//here are all the reasons for which we would like to update the main 3D layer
	if (s_tools.instance->m_updateFBO ||
		s_tools.instance->m_captureMode.enabled )
	{
		//we must update the FBO (or display without FBO)
		drawBackground = true;
		draw3DPass = true;
	}

	//context initialization
	CC_DRAW_CONTEXT CONTEXT;
	GetContext(CONTEXT);

	//clean the outdated messages
	{
		std::list<MessageToDisplay>::iterator it = s_tools.instance->m_messagesToDisplay.begin();
		qint64 currentTime_sec = s_tools.instance->m_timer.elapsed() / 1000;
		//CVLog::PrintDebug(QString("[paintGL] Current time: %1.").arg(currentTime_sec));

		while (it != s_tools.instance->m_messagesToDisplay.end())
		{
			//no more valid? we delete the message
			if (it->messageValidity_sec < currentTime_sec)
			{
				RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, it->message));
				it = s_tools.instance->m_messagesToDisplay.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	//backup the current viewport
	QRect originViewport = s_tools.instance->m_glViewport;
	bool modifiedViewport = false;

	/******************/
	/*** BACKGROUND ***/
	/******************/
	//shall we clear the background (depth and/or color)
	if (drawBackground)
	{
		if (s_tools.instance->m_showDebugTraces)
		{
			s_tools.instance->m_diagStrings << "draw background";
		}

		CONTEXT.clearColorLayer = true;
		CONTEXT.clearDepthLayer = true;
		DrawBackground(CONTEXT);
	}

	/*********************/
	/*** MAIN 3D LAYER ***/
	/*********************/
	if (draw3DPass)
	{
		if (s_tools.instance->m_showDebugTraces)
		{
			s_tools.instance->m_diagStrings << "draw 3D";
		}

		CONTEXT.forceRedraw = forceRedraw;
		Draw3D(CONTEXT);
	}

	//display traces
	if (s_tools.instance->m_showDebugTraces)
	{
		if (!s_tools.instance->m_diagStrings.isEmpty())
		{
			int x = s_tools.instance->m_glViewport.width() / 2 - 100;
			int y = 0;

			//draw black background
			{
				int height = (s_tools.instance->m_diagStrings.size() + 1) * 10;
				WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, DEBUG_LAYER_ID);
				param.color = ecvColor::dark;
				param.color.a = 0.5f;
				param.rect = QRect(x, s_tools.instance->m_glViewport.height() - y - height, 200, height);
				DrawWidgets(param, true);
			}

			for (const QString &str : s_tools.instance->m_diagStrings)
			{
				RenderText(x + 10, y + 10, str, QFont(), ecvColor::yellow, DEBUG_LAYER_ID);
				y += 10;
			}
		}
	}

	//restore viewport if necessary
	if (modifiedViewport)
	{
		//correction for HD screens
		SetGLViewport(originViewport);
		CONTEXT.glW = originViewport.width();
		CONTEXT.glH = originViewport.height();
		modifiedViewport = false;
	}

	if (drawBackground || draw3DCross)
	{
		s_tools.instance->m_updateFBO = false;
	}

	/******************/
	/*** FOREGROUND ***/
	/******************/
	if (drawForeground)
	{
		DrawForeground(CONTEXT);
	}

	s_tools.instance->m_shouldBeRefreshed = false;

	if (false && s_tools.instance->m_autoPickPivotAtCenter
		&&	!s_tools.instance->m_mouseMoved
		&&	s_tools.instance->m_autoPivotCandidate.norm2d() != 0.0 )
	{
		SetPivotPoint(s_tools.instance->m_autoPivotCandidate, true, false);
	}

	// update canvas
	UpdateScreen();
}

void ecvDisplayTools::SetGLViewport(const QRect& rect)
{
	const int retinaScale = GetDevicePixelRatio();
	s_tools.instance->m_glViewport = QRect(rect.left() * retinaScale, rect.top() * retinaScale,
		rect.width() * retinaScale, rect.height() * retinaScale);
	InvalidateViewport();
}

void ecvDisplayTools::drawCross()
{
}

void ecvDisplayTools::drawTrihedron()
{
}

void ecvDisplayTools::Draw3D(CC_DRAW_CONTEXT& CONTEXT)
{
	CONTEXT.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
	if (s_tools.instance->m_interactionFlags & INTERACT_TRANSFORM_ENTITIES)
	{
		CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
	}

	/****************************************/
	/****    PASS: 3D/FOREGROUND/LIGHT   ****/
	/****************************************/
	if (s_tools.instance->m_customLightEnabled || s_tools.instance->m_sunLightEnabled)
	{
		CONTEXT.drawingFlags |= CC_LIGHT_ENABLED;

		//we enable absolute sun light (if activated)
		if (s_tools.instance->m_sunLightEnabled)
		{
			//glEnableSunLight();
		}
	}

	// we draw 3D entities
	if (s_tools.instance->m_globalDBRoot)
	{
		s_tools.instance->m_globalDBRoot->draw(CONTEXT);
	}

	if (s_tools.instance->m_winDBRoot)
	{
		s_tools.instance->m_winDBRoot->draw(CONTEXT);
	}

#if 0
	//do this before drawing the pivot!
	if (s_tools.instance->m_autoPickPivotAtCenter)
	{
		CCVector3d P;
		if (GetClick3DPos(s_tools.instance->m_glViewport.width() / 2, s_tools.instance->m_glViewport.height() / 2, P))
		{
			s_tools.instance->m_autoPivotCandidate = P;
		}
	}
#endif

	if (s_tools.instance->m_globalDBRoot && s_tools.instance->m_globalDBRoot->getChildrenNumber())
	{
		//draw pivot
		//DrawPivot();
	}
}

void ecvDisplayTools::HideShowEntities(const QStringList & viewIDs, ENTITY_TYPE hideShowEntityType, bool visibility)
{
	CC_DRAW_CONTEXT context;
	context.hideShowEntityType = hideShowEntityType;
	context.visible = visibility;
	for (const QString & removeViewId : viewIDs)
	{
		context.viewID = removeViewId;
		HideShowEntities(context);
    }
}

bool ecvDisplayTools::HideShowEntities(const ccHObject *obj, bool visible)
{
    if (!obj || !ecvDisplayTools::GetCurrentScreen())
    {
       return false;
    }
    CC_DRAW_CONTEXT context;
    context.viewID = QString::number(obj->getUniqueID(), 10);
    context.visible = visible;
    ecvDisplayTools::HideShowEntities(context);
    return true;
}

void ecvDisplayTools::RemoveEntities(const ccHObject* obj)
{
    if (!obj || !ecvDisplayTools::GetCurrentScreen())
    {
       return;
    }

    CC_DRAW_CONTEXT context;
    context.removeViewID = QString::number(obj->getUniqueID(), 10);
    context.removeEntityType = obj->getEntityType();
    ecvDisplayTools::RemoveEntities(context);
}

void ecvDisplayTools::RemoveEntities(const QStringList & viewIDs, ENTITY_TYPE removeEntityType)
{
	CC_DRAW_CONTEXT context;
	context.removeEntityType = removeEntityType;
	for (const QString & removeViewId : viewIDs)
	{
		context.removeViewID = removeViewId;
		RemoveEntities(context);
	}
}

void ecvDisplayTools::DrawBackground(CC_DRAW_CONTEXT& CONTEXT)
{

	CONTEXT.drawingFlags = CC_DRAW_2D;
	if (s_tools.instance->m_interactionFlags & INTERACT_TRANSFORM_ENTITIES)
	{
		CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
	}

	//clear background
	{
		if (CONTEXT.clearDepthLayer)
		{
		}
		if (CONTEXT.clearColorLayer)
		{
			const ecvGui::ParamStruct& displayParams = GetDisplayParameters();
			if (displayParams.drawBackgroundGradient)
			{
				//draw the default gradient color background
				//we use the default background color for gradient start
				const ecvColor::Rgbub& bkgCol2 = GetDisplayParameters().backgroundCol;

				//and the inverse of the text color for gradient stop
				ecvColor::Rgbub bkgCol1 = GetDisplayParameters().textDefaultCol;
				bkgCol1.r = 255 - GetDisplayParameters().textDefaultCol.r;
				bkgCol1.g = 255 - GetDisplayParameters().textDefaultCol.g;
				bkgCol1.b = 255 - GetDisplayParameters().textDefaultCol.b;
				CONTEXT.backgroundCol = bkgCol1;
				CONTEXT.backgroundCol2 = bkgCol2;
				CONTEXT.drawBackgroundGradient = true;

			}
			else
			{
				//use plain color as specified by the user
				const ecvColor::Rgbub& bkgCol = displayParams.backgroundCol;
				CONTEXT.backgroundCol = bkgCol;
				CONTEXT.backgroundCol2 = bkgCol;
				CONTEXT.drawBackgroundGradient = false;
			}

			s_tools.instance->setBackgroundColor(CONTEXT);
		}
	}

}

void ecvDisplayTools::DrawForeground(CC_DRAW_CONTEXT& CONTEXT)
{
	/****************************************/
	/****  PASS: 2D/FOREGROUND/NO LIGHT  ****/
	/****************************************/

	CONTEXT.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
	if (s_tools.instance->m_interactionFlags & INTERACT_TRANSFORM_ENTITIES)
	{
		CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
	}

	//we draw 2D entities
	if (s_tools.instance->m_globalDBRoot)
		s_tools.instance->m_globalDBRoot->draw(CONTEXT);
	if (s_tools.instance->m_winDBRoot)
		s_tools.instance->m_winDBRoot->draw(CONTEXT);

	//current displayed scalar field color ramp (if any)
	ccRenderingTools::DrawColorRamp(CONTEXT);

	s_tools.instance->m_clickableItems.clear();

	/*** overlay entities ***/
	if (s_tools.instance->m_displayOverlayEntities)
	{
		//default overlay color
		const ecvColor::Rgbub& textCol = GetDisplayParameters().textDefaultCol;

		if (!s_tools.instance->m_captureMode.enabled || s_tools.instance->m_captureMode.renderOverlayItems)
		{
			//scale: only in ortho mode
			if (!s_tools.instance->m_viewportParams.perspectiveView)
			{
				//DrawScale(textCol);
			}

			//trihedron
			//DrawTrihedron();
		}

		if (!s_tools.instance->m_captureMode.enabled)
		{
			int yStart = 0;

			//current messages (if valid)
			if (!s_tools.instance->m_messagesToDisplay.empty())
			{
				int ll_currentHeight = s_tools.instance->m_glViewport.height() - 10; //lower left
				int uc_currentHeight = 10; //upper center

				for (const auto &message : s_tools.instance->m_messagesToDisplay)
				{
					switch (message.position)
					{
					case LOWER_LEFT_MESSAGE:
					{
						RenderText(10, ll_currentHeight, message.message, s_tools.instance->m_font);
						int messageHeight = QFontMetrics(s_tools.instance->m_font).height();
						ll_currentHeight -= (messageHeight * 5) / 4; //add a 25% margin
					}
					break;
					case UPPER_CENTER_MESSAGE:
					{
						QRect rect = QFontMetrics(s_tools.instance->m_font).boundingRect(message.message);
						int x = (s_tools.instance->m_glViewport.width() - rect.width()) / 2;
						int y = uc_currentHeight + rect.height();

						RenderText(x, y, message.message, s_tools.instance->m_font);
						uc_currentHeight += (rect.height() * 5) / 4; //add a 25% margin
					}
					break;
					case SCREEN_CENTER_MESSAGE:
					{
						QFont newFont(s_tools.instance->m_font); //no need to take zoom into account!
						newFont.setPointSize(12 * GetDevicePixelRatio());
						QRect rect = QFontMetrics(newFont).boundingRect(message.message);
						//only one message supported in the screen center (for the moment ;)
						RenderText((s_tools.instance->m_glViewport.width() - rect.width()) / 2, (s_tools.instance->m_glViewport.height() - rect.height()) / 2, message.message, newFont);
					}
					break;
					}
				}
			}

			//hot-zone
			{
				s_tools.instance->DrawClickableItems(0, yStart);
			}

		}
	}

}

void ecvDisplayTools::Redraw2DLabel()
{
	//we get the other selected labels as well!
	ccHObject::Container labels;
	FilterByEntityType(labels, CV_TYPES::LABEL_2D);

	for (auto & label : labels)
	{
		//Warning: cc2DViewportLabel is also a kind of 'CV_TYPES::LABEL_2D'!
		if (label->isA(CV_TYPES::LABEL_2D) && label->isVisible())
		{
			cc2DLabel* l = static_cast<cc2DLabel*>(label);
			if (!l || (l->isDisplayedIn2D() && !l->isPointLegendDisplayed())) continue;

			CC_DRAW_CONTEXT context;
			GetContext(context);
			l->update2DLabelView(context);
		}
	}
}

void ecvDisplayTools::Update2DLabel(bool immediateUpdate/* = false*/)
{
	//we get the other selected labels as well!

	s_tools.instance->m_activeItems.clear();
	ccHObject::Container labels;
	FilterByEntityType(labels, CV_TYPES::LABEL_2D);

	for (auto & label : labels)
	{
		//Warning: cc2DViewportLabel is also a kind of 'CV_TYPES::LABEL_2D'!
		if (label->isA(CV_TYPES::LABEL_2D) && label->isVisible())
		{
			cc2DLabel* l = ccHObjectCaster::To2DLabel(label);
			if (!l || (l->isDisplayedIn2D() && !l->isPointLegendDisplayed())) continue;

			s_tools.instance->m_activeItems.push_back(l);
			if (immediateUpdate)
			{
				CC_DRAW_CONTEXT context;
				GetContext(context);
				l->update2DLabelView(context);
			}
		}
		else if (label->isA(CV_TYPES::VIEWPORT_2D_LABEL))
		{
			cc2DViewportLabel* l = ccHObjectCaster::To2DViewportLabel(label);
			if (!l) continue;
			l->clear2Dviews();
		}
	}
}

void ecvDisplayTools::Pick2DLabel(int x, int y)
{
	QString id = s_tools.instance->pick2DLabel(x, y);

	//we get the other selected labels as well!
	s_tools.instance->m_activeItems.clear();
	if (!id.isEmpty())
	{
		ccHObject::Container labels;
		FilterByEntityType(labels, CV_TYPES::LABEL_2D);
		for (auto & label : labels)
		{
			//Warning: cc2DViewportLabel is also a kind of 'CV_TYPES::LABEL_2D'!
			if (label->isA(CV_TYPES::LABEL_2D) && label->isVisible())
			{
				cc2DLabel* l = ccHObjectCaster::To2DLabel(label);
				if (QString::number(l->getUniqueID()).compare(id) == 0)
				{
					s_tools.instance->m_activeItems.push_back(l);
				}
			}
		}
	}
}

void ecvDisplayTools::FilterByEntityType(ccHObject::Container& labels, CV_CLASS_ENUM type)
{
	if (s_tools.instance->m_globalDBRoot)
		s_tools.instance->m_globalDBRoot->filterChildren(labels, true, type);
	if (s_tools.instance->m_winDBRoot)
		s_tools.instance->m_winDBRoot->filterChildren(labels, true, type);
}

void ecvDisplayTools::RenderText(
	int x, int y, const QString & str, 
	const QFont & font/*=QFont()*/,
	ecvColor::Rgbub color/* = ecvColor::defaultLabelBkgColor*/,
	QString id)
{
	CC_DRAW_CONTEXT context;
	// for T2D
	if (id.isEmpty())
	{
		context.viewID = str;
	}
	else
	{
		context.viewID = id;
	}
	
	context.textParam.text = str;
	context.textParam.display3D = false;
	context.textParam.font = font;
	context.textParam.font.setPointSize(font.pointSize());
	//QRect screen = QGuiApplication::primaryScreen()->geometry();

	context.textDefaultCol = color;
	if (context.textParam.display3D)
	{
		context.textParam.textScale = 2.0;
		CCVector3d input3D(x, s_tools.instance->m_glViewport.height() - y, 0);
		CCVector3d output2D;
		ToWorldPoint(input3D, output2D);
		context.textParam.textPos.x = output2D.x;
		context.textParam.textPos.y = output2D.y;
		context.textParam.textPos.z = output2D.z;
	}
	else
	{
		context.textParam.textPos.x = x;
		context.textParam.textPos.y = 
			s_tools.instance->m_glViewport.height() - y;
		context.textParam.textPos.z = 0;
	}
	DisplayText(context);
}

void ecvDisplayTools::RenderText(
	double x, double y, double z, 
	const QString & str, 
	const QFont & font/*=QFont()*/,
	ecvColor::Rgbub color/* = ecvColor::defaultLabelBkgColor*/,
	QString id)
{
	//get the actual viewport / matrices
	ccGLCameraParameters camera;
	ecvDisplayTools::GetViewerPos(camera.viewport);
	ecvDisplayTools::GetProjectionMatrix(camera.projectionMat.data());
	ecvDisplayTools::GetViewMatrix(camera.modelViewMat.data());

	CCVector3d Q2D(0, 0, 0);
	if (camera.project(CCVector3d(x, y, z), Q2D))
	{
		Q2D.y = s_tools.instance->m_glViewport.height() - 1 - Q2D.y;
		RenderText(Q2D.x, Q2D.y, str, font, color, id);
	}
}

void ecvDisplayTools::Display3DLabel(
	const QString& str,
	const CCVector3& pos3D, 
	const unsigned char* rgb/*=0*/, 
	const QFont& font/*=QFont()*/)
{
	ecvColor::Rgbub col(rgb ? rgb : GetDisplayParameters().textDefaultCol.rgb);
	RenderText(pos3D.x, pos3D.y, pos3D.z, str, font, col);
}

void ecvDisplayTools::DisplayText(QString text,
	int x,
	int y,
	unsigned char align/*=ALIGN_HLEFT|ALIGN_VTOP*/,
	float bkgAlpha/*=0*/,
	const unsigned char* rgbColor/*=0*/,
	const QFont* font/*=0*/,
	QString id)
{
	int x2 = x;
	int y2 = s_tools.instance->m_glViewport.height() - 1 - y;

	//actual text color
	const unsigned char* col = (rgbColor ? rgbColor : GetDisplayParameters().textDefaultCol.rgb);

	QFont realFont = (font ? *font : s_tools.instance->m_font);
	QFont textFont = realFont;
	QFontMetrics fm(textFont);
	int margin = fm.height() / 4;

	if (align != ALIGN_DEFAULT || bkgAlpha != 0.0f)
	{
		QRect rect = fm.boundingRect(text);

		//text alignment
		if (align & ALIGN_HMIDDLE)
			x2 -= rect.width() / 2;
		else if (align & ALIGN_HRIGHT)
			x2 -= rect.width();
		if (align & ALIGN_VMIDDLE)
			y2 += rect.height() / 2;
		else if (align & ALIGN_VBOTTOM)
			y2 += rect.height();

		//background is not totally transparent
		if (bkgAlpha != 0.0f)
		{
			//inverted color with a bit of transparency
			const float invertedCol[4] = { (255 - col[0]) / 255.0f,
											(255 - col[0]) / 255.0f,
											(255 - col[0]) / 255.0f,
											bkgAlpha };

			//int xB = x2 - s_tools.instance->m_glViewport.width() / 2;
			//int yB = s_tools.instance->m_glViewport.height() / 2 - y2;
			//yB += margin / 2; //empirical compensation

			int xB = x2;
			int yB = s_tools.instance->m_glViewport.height() - y2;

			WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, id);
			param.text = text;

			if (id.isEmpty())
			{
				param.viewID = text;
				RemoveWidgets(param);
			}

			param.color.r = invertedCol[0];
			param.color.g = invertedCol[1];
			param.color.b = invertedCol[2];
			param.color.a = invertedCol[3];
			param.rect = QRect(xB - margin, yB - margin, 
				rect.width() + 2 * margin, 
				static_cast<int>(rect.height() + 1.5 * margin));

			DrawWidgets(param, true);
		}
	}

	if (align & ALIGN_VBOTTOM)
		y2 -= margin; //empirical compensation
	else if (align & ALIGN_VMIDDLE)
		y2 -= margin / 2; //empirical compensation

	ecvColor::Rgbub textColor(col);
	RenderText(x2, y2, text, realFont, textColor, id);
}

void ecvDisplayTools::DisplayTexture2DPosition(QImage image, const QString& id, int x, int y, int w, int h, unsigned char alpha/*=255*/)
{
	WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_IMAGE, id);
	param.image = image;
	param.opacity = alpha/255.0f;
	param.rect = QRect(x, y, w, h);
	DrawWidgets(param, true);
}

void ecvDisplayTools::ClearBubbleView()
{
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, s_tools.instance->m_hotZone->bbv_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, s_tools.instance->m_hotZone->fs_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, s_tools.instance->m_hotZone->psi_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, s_tools.instance->m_hotZone->lsi_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, "Exit"));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, "clicked_items"));
}

void ecvDisplayTools::DrawClickableItems(int xStart0, int& yStart)
{
	const static char* CLICKED_ITEMS = "clicked_items";
	//we init the necessary parameters the first time we need them
	if (!s_tools.instance->m_hotZone)
	{
		s_tools.instance->m_hotZone = new HotZone(ecvDisplayTools::GetCurrentScreen());
	}
	// remember the last position of the 'top corner'
	s_tools.instance->m_hotZone->topCorner = QPoint(xStart0, yStart) + 
		QPoint(s_tools.instance->m_hotZone->margin, s_tools.instance->m_hotZone->margin);

	bool fullScreenEnabled = ExclusiveFullScreen();

	if (!s_tools.instance->m_clickableItemsVisible
		&& !s_tools.instance->m_bubbleViewModeEnabled
		&& !fullScreenEnabled)
	{
        ClearBubbleView();
		//nothing to do
		return;
	}

	//"exit" icon
	//static const QImage c_exitIcon = QImage(":/Resources/images/ecvExit.png").mirrored();
	int fullW = s_tools.instance->m_glViewport.width();
    int fullH = s_tools.instance->m_glViewport.height();

	// clear history
	RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, CLICKED_ITEMS));

	//draw semi-transparent background
	{
		QRect areaRect = s_tools.instance->m_hotZone->rect(s_tools.instance->m_clickableItemsVisible, 
			s_tools.instance->m_bubbleViewModeEnabled, fullScreenEnabled);
		areaRect.translate(s_tools.instance->m_hotZone->topCorner);

		WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, CLICKED_ITEMS);
		param.color = ecvColor::FromRgba(ecvColor::odarkGrey);
		param.color.a = 0.2f/*210/255.0f*/;
		int x0 = areaRect.x();
		int y0 = fullH - areaRect.y() - areaRect.height();
		param.rect = QRect(x0, y0, areaRect.width(), areaRect.height());
		DrawWidgets(param, false);
	}

	yStart = s_tools.instance->m_hotZone->topCorner.y();
	int iconSize = s_tools.instance->m_hotZone->iconSize;

	if (fullScreenEnabled)
	{
		int xStart = s_tools.instance->m_hotZone->topCorner.x();

		//label
		RenderText(xStart, yStart + s_tools.instance->m_hotZone->yTextBottomLineShift,
			s_tools.instance->m_hotZone->fs_label, 
			s_tools.instance->m_hotZone->font, ecvColor::defaultLabelBkgColor, CLICKED_ITEMS);

		//icon
		xStart += s_tools.instance->m_hotZone->fs_labelRect.width() + s_tools.instance->m_hotZone->margin;

		//"full-screen" icon
		{
			int x0 = xStart;
			int y0 = fullH - (yStart + iconSize);
			WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, CLICKED_ITEMS);
			param.color = ecvColor::FromRgba(ecvColor::ored);
			param.rect = QRect(x0, y0, iconSize, iconSize);
			DrawWidgets(param, false);

			WIDGETS_PARAMETER texParam(WIDGETS_TYPE::WIDGET_T2D, CLICKED_ITEMS);
			texParam.color = ecvColor::bright;
			texParam.text = "Exit";
			texParam.rect = QRect(x0, fullH - (yStart + 3*iconSize/4), iconSize, iconSize);
            texParam.fontSize = s_tools.instance->m_hotZone->font.pointSize();
			DrawWidgets(texParam, false);
			s_tools.instance->m_clickableItems.emplace_back(ClickableItem::LEAVE_FULLSCREEN_MODE, 
				QRect(xStart, yStart, iconSize, iconSize));
			xStart += iconSize;
		}

		yStart += iconSize;
		yStart += s_tools.instance->m_hotZone->margin;
	}

	if (s_tools.instance->m_bubbleViewModeEnabled)
	{
		int xStart = s_tools.instance->m_hotZone->topCorner.x();

		//label
		RenderText(xStart, yStart + s_tools.instance->m_hotZone->yTextBottomLineShift, 
			s_tools.instance->m_hotZone->bbv_label, s_tools.instance->m_hotZone->font);

		//icon
		xStart += s_tools.instance->m_hotZone->bbv_labelRect.width() + s_tools.instance->m_hotZone->margin;

		//"exit" icon
		{
			s_tools.instance->m_clickableItems.emplace_back(ClickableItem::LEAVE_BUBBLE_VIEW_MODE, 
				QRect(xStart, yStart, s_tools.instance->m_hotZone->iconSize, s_tools.instance->m_hotZone->iconSize));
			xStart += s_tools.instance->m_hotZone->iconSize;
		}

		yStart += s_tools.instance->m_hotZone->iconSize;
		yStart += s_tools.instance->m_hotZone->margin;
	}

	if (s_tools.instance->m_clickableItemsVisible)
	{
        ecvColor::Rgb textColor = ecvColor::Rgb(s_tools.instance->m_hotZone->color);
		WIDGETS_PARAMETER widgetParam(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, CLICKED_ITEMS);
        widgetParam.color = ecvColor::FromRgba(ecvColor::ogreen);
		WIDGETS_PARAMETER sepParam(WIDGETS_TYPE::WIDGET_POINTS_2D, CLICKED_ITEMS);
		sepParam.color = widgetParam.color;
		sepParam.color.a = 0.5f;
		
		//default point size
		{
			int xStart = s_tools.instance->m_hotZone->topCorner.x();

			RenderText(xStart, yStart + s_tools.instance->m_hotZone->yTextBottomLineShift, 
				s_tools.instance->m_hotZone->psi_label, s_tools.instance->m_hotZone->font, 
				textColor, CLICKED_ITEMS);

			//icons
			xStart += s_tools.instance->m_hotZone->psi_labelRect.width() + s_tools.instance->m_hotZone->margin;
            xStart -= iconSize;
			//"minus" icon
			{
				int x0 = xStart;
				int y0 = fullH - (yStart + iconSize/2);
				widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
				DrawWidgets(widgetParam, false);
				s_tools.instance->m_clickableItems.emplace_back(ClickableItem::DECREASE_POINT_SIZE,
					QRect(xStart, yStart, iconSize, iconSize));
				xStart += iconSize;
			}

			//separator
			{
				sepParam.radius = s_tools.instance->m_viewportParams.defaultPointSize / 2;
				int x0 = xStart + s_tools.instance->m_hotZone->margin /*s_tools.instance->m_hotZone->margin / 2*/;
				int y0 = fullH - (yStart + iconSize / 2);
				sepParam.rect = QRect(x0, y0, iconSize, iconSize);
				DrawWidgets(sepParam, false);
				xStart += s_tools.instance->m_hotZone->margin * 2;
			}

			//"plus" icon
			{
				int x0 = xStart;
				int y0 = fullH - (yStart + iconSize / 2);
				widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
				DrawWidgets(widgetParam, false);
				x0 = xStart + 3 * iconSize / 8;
				y0 = fullH - (yStart + 7*iconSize/8);
				widgetParam.rect = QRect(x0, y0, iconSize/4, iconSize);
				DrawWidgets(widgetParam, false);

				s_tools.instance->m_clickableItems.emplace_back(
					ClickableItem::INCREASE_POINT_SIZE,
					QRect(xStart, yStart, iconSize, iconSize));
				xStart += iconSize;
			}

			yStart += iconSize;
			yStart += s_tools.instance->m_hotZone->margin;
		}

		//default line size
		{
			int xStart = s_tools.instance->m_hotZone->topCorner.x();

			RenderText(xStart, yStart + s_tools.instance->m_hotZone->yTextBottomLineShift, 
				s_tools.instance->m_hotZone->lsi_label, s_tools.instance->m_hotZone->font, 
				textColor, CLICKED_ITEMS);

			//icons
			xStart += s_tools.instance->m_hotZone->lsi_labelRect.width() + s_tools.instance->m_hotZone->margin;
            xStart -= iconSize;

			//"minus" icon
			{
				int x0 = xStart;
				int y0 = fullH - (yStart + iconSize / 2);
				widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
				DrawWidgets(widgetParam, false);

				s_tools.instance->m_clickableItems.emplace_back(ClickableItem::DECREASE_LINE_WIDTH,
					QRect(xStart, yStart, iconSize, iconSize));
				xStart += iconSize;
			}

			//separator
			{
				sepParam.radius = s_tools.instance->m_viewportParams.defaultLineWidth / 2;
				int x0 = xStart + s_tools.instance->m_hotZone->margin /*s_tools.instance->m_hotZone->margin / 2*/;
				int y0 = fullH - (yStart + iconSize / 2);
				sepParam.rect = QRect(x0, y0, iconSize, iconSize);
				DrawWidgets(sepParam, false);
				xStart += s_tools.instance->m_hotZone->margin * 2;
			}

			//"plus" icon
			{
				int x0 = xStart;
				int y0 = fullH - (yStart + iconSize / 2);
				widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
				DrawWidgets(widgetParam, false);
				x0 = xStart + 3 * iconSize / 8;
				y0 = fullH - (yStart + 7 * iconSize / 8);
				widgetParam.rect = QRect(x0, y0, iconSize / 4, iconSize);
				DrawWidgets(widgetParam, false);

				s_tools.instance->m_clickableItems.emplace_back(
					ClickableItem::INCREASE_LINE_WIDTH, 
					QRect(xStart, yStart, iconSize, iconSize));
				xStart += iconSize;
			}

			yStart += iconSize;
			yStart += s_tools.instance->m_hotZone->margin;
		}
	}
}

void ecvDisplayTools::DrawScale(const ecvColor::Rgbub& color)
{
	assert(!s_tools.instance->m_viewportParams.perspectiveView); //a scale is only valid in ortho. mode!

	float scaleMaxW = s_tools.instance->m_glViewport.width() / 4.0f; //25% of screen width
	if (s_tools.instance->m_captureMode.enabled)
	{
		//DGM: we have to fall back to the case 'render zoom = 1' (otherwise we might not get the exact same aspect)
		scaleMaxW /= s_tools.instance->m_captureMode.zoomFactor;
	}
	if (s_tools.instance->m_viewportParams.zoom < CC_GL_MIN_ZOOM_RATIO)
	{
		assert(false);
		return;
	}

	//we first compute the width equivalent to 25% of horizontal screen width
	//(this is why it's only valid in orthographic mode !)
	float equivalentWidthRaw = scaleMaxW * s_tools.instance->m_viewportParams.pixelSize / s_tools.instance->m_viewportParams.zoom;
	float equivalentWidth = RoundScale(equivalentWidthRaw);

	QFont font = GetTextDisplayFont(); //we take rendering zoom into account!
	QFontMetrics fm(font);

	//we deduce the scale drawing width
	float scaleW_pix = equivalentWidth / s_tools.instance->m_viewportParams.pixelSize * s_tools.instance->m_viewportParams.zoom;
	if (s_tools.instance->m_captureMode.enabled)
	{
		//we can now safely apply the rendering zoom
		scaleW_pix *= s_tools.instance->m_captureMode.zoomFactor;
	}
	float trihedronLength = CC_DISPLAYED_TRIHEDRON_AXES_LENGTH * s_tools.instance->m_captureMode.zoomFactor;
	float dW = 2.0f * trihedronLength + 20.0f;
	float dH = std::max(fm.height() * 1.25f, trihedronLength + 5.0f);
	float w = s_tools.instance->m_glViewport.width() / 2.0f - dW;
	float h = s_tools.instance->m_glViewport.height() / 2.0f - dH;
	float tick = 3.0f * s_tools.instance->m_captureMode.zoomFactor;
}

void ecvDisplayTools::CheckIfRemove()
{
	if (s_tools.instance->m_removeAllFlag)
	{
		CC_DRAW_CONTEXT context;
		context.removeEntityType = ENTITY_TYPE::ECV_ALL;
		RemoveEntities(context);
		SetRemoveAllFlag(false);
	}
	else if (s_tools.instance->m_removeFlag)
	{
		for (const removeInfo& rmInfo : s_tools.instance->m_removeInfos)
		{
			if (rmInfo.removeType == ENTITY_TYPE::ECV_NONE) continue;
			// octree and kdtree object has been deleted before
			if (rmInfo.removeType == ENTITY_TYPE::ECV_OCTREE) continue;
			if (rmInfo.removeType == ENTITY_TYPE::ECV_KDTREE) continue;
			if (rmInfo.removeType == ENTITY_TYPE::ECV_2DLABLE) continue;
            if (rmInfo.removeType == ENTITY_TYPE::ECV_SENSOR) continue;
			
			CC_DRAW_CONTEXT context;
			context.removeEntityType = rmInfo.removeType;
			context.removeViewID = rmInfo.removeId;
			RemoveEntities(context);
			RemoveBB(context);
		}
		s_tools.instance->m_removeFlag = false;
	}
}

void ecvDisplayTools::RemoveBB(CC_DRAW_CONTEXT context)
{
	context.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
	context.removeViewID = QString("BBox-") + context.removeViewID;
	RemoveEntities(context);
}

void ecvDisplayTools::RemoveBB(const QString & viewId)
{
	CC_DRAW_CONTEXT context;
	context.removeViewID = viewId;
	RemoveBB(context);
}

void ecvDisplayTools::ChangeEntityProperties(PROPERTY_PARAM & propertyParam, bool autoUpdate/* = true*/)
{
	if (propertyParam.entity)
	{
		if (propertyParam.entity->isKindOf(CV_TYPES::PRIMITIVE))
		{
            propertyParam.entityType = ConvertToEntityType(CV_TYPES::PRIMITIVE);
		}
		else
		{
            propertyParam.entityType = ConvertToEntityType(propertyParam.entity->getClassID());
		}

		propertyParam.viewId = QString::number(propertyParam.entity->getUniqueID());
		s_tools.instance->changeEntityProperties(propertyParam);
		if (autoUpdate)
		{
			UpdateScreen();
		}
	}
}

void ecvDisplayTools::DrawWidgets(const WIDGETS_PARAMETER& param, bool update/* = false*/)
{
	ccHObject * entity = param.entity;
    int viewport = param.viewport;
	switch (param.type)
	{
	case WIDGETS_TYPE::WIDGET_COORDINATE:
		ShowOrientationMarker();
		break;
	case WIDGETS_TYPE::WIDGET_BBOX:
		break;
		
	case WIDGETS_TYPE::WIDGET_T2D:
	{
		QFont textFont = s_tools.instance->m_font;
		//QRect screen = QGuiApplication::primaryScreen()->geometry();

		//if (screen.width() > 1920 && GetDevicePixelRatio() == 1)  // for high DPI
		//{
		//	textFont.setPointSize(textFont.pointSize() * 3);
		//}

		const_cast<WIDGETS_PARAMETER*>(&param)->fontSize = textFont.pointSize();
		
		s_tools.instance->drawWidgets(param);
	}
		break;
	case WIDGETS_TYPE::WIDGET_IMAGE:
	case WIDGETS_TYPE::WIDGET_LINE_2D:
	case WIDGETS_TYPE::WIDGET_CIRCLE_2D:
	case WIDGETS_TYPE::WIDGET_POINTS_2D:
	case WIDGETS_TYPE::WIDGET_SCALAR_BAR:
	case WIDGETS_TYPE::WIDGET_POLYLINE_2D:
	case WIDGETS_TYPE::WIDGET_TRIANGLE_2D:
	case WIDGETS_TYPE::WIDGET_RECTANGLE_2D:
		s_tools.instance->drawWidgets(param);
		break;
	case WIDGETS_TYPE::WIDGET_LINE_3D:
		if (param.lineWidget.valid)
		{
			s_tools.instance->drawWidgets(param);
		}
		break;
	case WIDGETS_TYPE::WIDGET_POLYLINE:
	{
		//context initialization
		CC_DRAW_CONTEXT CONTEXT;
		GetContext(CONTEXT);
		ccPolyline* poly = ccHObjectCaster::ToPolyline(entity);
		if (poly->is2DMode())
		{
			CONTEXT.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
		}
		else
		{
			CONTEXT.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
		}
		
		if (s_tools.instance->m_interactionFlags & INTERACT_TRANSFORM_ENTITIES)
		{
			CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
		}
        CONTEXT.defaultViewPort = viewport;
		poly->draw(CONTEXT);
	}
		break;
	case WIDGETS_TYPE::WIDGET_SPHERE:
		s_tools.instance->drawWidgets(param);
		break;	
	case WIDGETS_TYPE::WIDGET_CAPTION:
		s_tools.instance->drawWidgets(param);
		break;
	case WIDGETS_TYPE::WIDGET_T3D:
		break;
	default:
		break;
	}

	if (update)
	{
		UpdateScreen();
	}
}

void ecvDisplayTools::RemoveWidgets(const WIDGETS_PARAMETER& param, bool update/* = false*/)
{
	CC_DRAW_CONTEXT context;
	switch (param.type)
	{
	case WIDGETS_TYPE::WIDGET_COORDINATE:
		break;
	case WIDGETS_TYPE::WIDGET_BBOX:
	{
		context.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
		context.viewID = QString("BBox-") + param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_POLYGONMESH:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_MESH;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_LINE_3D:
	case WIDGETS_TYPE::WIDGET_POLYLINE:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_LINES_3D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_CAPTION:
	{
		context.removeEntityType = ENTITY_TYPE::ECV_CAPTION;
        context.defaultViewPort = param.viewport;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_SCALAR_BAR:
	{
		context.removeEntityType = ENTITY_TYPE::ECV_SCALAR_BAR;
        context.defaultViewPort = param.viewport;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_SPHERE:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_IMAGE:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_IMAGE;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_POINTS_2D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_MARK_POINT;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_CIRCLE_2D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_CIRCLE_2D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	case WIDGETS_TYPE::WIDGET_TRIANGLE_2D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_TRIANGLE_2D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_POLYLINE_2D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_POLYLINE_2D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_LINE_2D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_LINES_2D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_RECTANGLE_2D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_RECTANGLE_2D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
	break;
	case WIDGETS_TYPE::WIDGET_T2D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_TEXT2D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
		break;
	case WIDGETS_TYPE::WIDGET_T3D:
	{
        context.defaultViewPort = param.viewport;
		context.removeEntityType = ENTITY_TYPE::ECV_TEXT3D;
		context.removeViewID = param.viewID;
		RemoveEntities(context);
	}
		break;
	default:
		break;
	}

	if (update)
	{
		UpdateScreen();
	}
}

bool ecvDisplayTools::GetClick3DPos(int x, int y, CCVector3d& P3D)
{
	ccGLCameraParameters camera;
	GetGLCameraParameters(camera);

	y = s_tools.instance->m_glViewport.height() - 1 - y;

	double glDepth = GetGLDepth(x, y);
	if (glDepth == 1.0)
	{
		return false;
	}
	CCVector3d P2D(x, y, glDepth);
	return camera.unproject(P2D, P3D);
}

void ecvDisplayTools::DrawPivot()
{
	if (!s_tools.instance->m_viewportParams.objectCenteredView ||
		(s_tools.instance->m_pivotVisibility == PIVOT_HIDE) ||
		(s_tools.instance->m_pivotVisibility == PIVOT_SHOW_ON_MOVE && !s_tools.instance->m_pivotSymbolShown))
	{
		return;
	}

	//place origin on pivot point
    CCVector3d tranlateTartget = CCVector3d(s_tools.instance->m_viewportParams.getPivotPoint().x,
                                            s_tools.instance->m_viewportParams.getPivotPoint().y,
                                            s_tools.instance->m_viewportParams.getPivotPoint().z);

	//compute actual symbol radius
	double symbolRadius = CC_DISPLAYED_PIVOT_RADIUS_PERCENT * std::min(s_tools.instance->m_glViewport.width(), 
		s_tools.instance->m_glViewport.height()) / 2.0;

	//draw a small sphere
	{
		ccSphere sphere(static_cast<PointCoordinateType>(10.0 / symbolRadius));
		sphere.setColor(ecvColor::yellow);
		sphere.showColors(true);
		sphere.setVisible(true);
		sphere.setEnabled(true);
		//force lighting for proper sphere display
		CC_DRAW_CONTEXT CONTEXT;
		GetContext(CONTEXT);
		CONTEXT.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND | CC_LIGHT_ENABLED;
		sphere.draw(CONTEXT);
	}
}

void ecvDisplayTools::SetCurrentScreen(QWidget* widget) {
	s_tools.instance->m_currentScreen = widget;
	widget->update();
}
