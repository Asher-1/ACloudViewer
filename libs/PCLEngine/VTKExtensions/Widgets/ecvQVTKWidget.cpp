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

#include "ecvQVTKWidget.h"
#include "VtkUtils/utils.h"
#include "VtkUtils/vtkutils.h"
#include "VtkUtils/rendererslayoutalgo.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>
#include <CVConst.h>

// VTK
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkRendererCollection.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkLookupTable.h>
#include <vtkPNGReader.h>
#include <vtkImageData.h>
#include <vtkLogoRepresentation.h>
#include <vtkLogoWidget.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkScalarBarWidget.h>
#include <vtkScalarBarActor.h>
#include <vtkScalarBarRepresentation.h>
#include <vtkColorTransferFunction.h>
#include <vtkIdFilter.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkClipPolyData.h>
#include <vtkConeSource.h>
#include <vtkDelaunay2D.h>
#include <vtkAngleRepresentation2D.h>

#include <pcl/visualization/vtk/vtkVertexBufferObjectMapper.h>

// ECV_DB_LIB
#include <ecvMainAppInterface.h>
#include <ecvDisplayTools.h>
#include <ecvInteractor.h>
#include <ecvPolyline.h>
#include <ecvPointCloud.h>

// QT
#include <QWidget>
#include <QMainWindow>
#include <QHBoxLayout>
#include <QApplication>
#include <QLayout>
#include <QMessageBox>
#include <QMimeData>
#include <QPushButton>
#include <QSettings>
#include <QTouchEvent>
#include <QWheelEvent>

#ifdef USE_VLD
#include <vld.h>
#endif

// macroes
#ifndef VTK_CREATE
#define VTK_CREATE(TYPE, NAME) vtkSmartPointer<TYPE> NAME = vtkSmartPointer<TYPE>::New()
#endif


class VtkWidgetPrivate
{
public:
	VtkWidgetPrivate(ecvQVTKWidget* q);
	~VtkWidgetPrivate();

	void init();
	void configRenderer(vtkRenderer* renderer);
	void layoutRenderers();

	ecvQVTKWidget* q_ptr;
	QColor backgroundColor = Qt::black;
	bool multiViewports = false;
	vtkRenderer* defaultRenderer = nullptr;
	vtkSmartPointer<vtkOrientationMarkerWidget> orientationMarkerWidget = nullptr;

	QList<vtkRenderer*> renderers;
	QList<vtkProp*> actors;
	QList<vtkProp*> props;

	double bounds[6];
};

VtkWidgetPrivate::VtkWidgetPrivate(ecvQVTKWidget *q) : q_ptr(q)
{
	init();
}

VtkWidgetPrivate::~VtkWidgetPrivate()
{
}

void VtkWidgetPrivate::configRenderer(vtkRenderer *renderer)
{
	if (!renderer)
		return;

	double bgclr[3];
	Utils::vtkColor(backgroundColor, bgclr);

	renderer->SetBackground(bgclr);
}

static int columnCount(int count)
{
	int cols = 1;
	while (true) {
		if ((cols * cols) >= count)
			return cols;
		++cols;
	}
	return cols;
}

void VtkWidgetPrivate::layoutRenderers()
{
	switch (renderers.size()) {
	case 1:
		VtkUtils::layoutRenderers<1>(renderers);
		break;

	case 2:
		VtkUtils::layoutRenderers<2>(renderers);
		break;

	case 3:
		VtkUtils::layoutRenderers<3>(renderers);
		break;

	case 4:
		VtkUtils::layoutRenderers<4>(renderers);
		break;

	case 5:
		VtkUtils::layoutRenderers<5>(renderers);
		break;

	case 6:
		VtkUtils::layoutRenderers<6>(renderers);
		break;

	case 7:
		VtkUtils::layoutRenderers<7>(renderers);
		break;

	case 8:
		VtkUtils::layoutRenderers<8>(renderers);
		break;

	case 9:
		VtkUtils::layoutRenderers<9>(renderers);
		break;

	case 10:
		VtkUtils::layoutRenderers<10>(renderers);
		break;

	default:
		VtkUtils::layoutRenderers<-1>(renderers);
	}
}

void VtkWidgetPrivate::init()
{
	layoutRenderers();
}

//Max click duration for enabling picking mode (in ms)
//static const int CC_MAX_PICKING_CLICK_DURATION_MS = 200;
static const int CC_MAX_PICKING_CLICK_DURATION_MS = 350;
ecvQVTKWidget::ecvQVTKWidget(QMainWindow* parentWindow, ecvMainAppInterface* app, ecvDisplayTools* tools)
	: QVTKWidget(parentWindow)
	, m_axesWidget(nullptr)
	, m_logoWidget(nullptr)
	, m_interactor(nullptr)
	, m_scalarbarWidget(nullptr)
	, m_dataObject(nullptr)
	, m_modelActor(nullptr)
	, m_win(parentWindow)
	, m_app(app)
	, m_tools(tools)
{
	this->setWindowTitle("3D View");

	//drag & drop handling
	setAcceptDrops(true);
	setAttribute(Qt::WA_AcceptTouchEvents, true);
	//setAttribute(Qt::WA_OpaquePaintEvent, true);
	vtkObject::GlobalWarningDisplayOff();
	d_ptr = new VtkWidgetPrivate(this);
}

ecvQVTKWidget::~ecvQVTKWidget()
{
	if (d_ptr)
	{
		delete d_ptr;
		d_ptr = nullptr;
	}
}

vtkSmartPointer<vtkLookupTable> ecvQVTKWidget::createLookupTable(double min, double max)
{
	double hsv1[3];
	double hsv2[3];
	Utils::qColor2HSV(m_color1, hsv1);
	Utils::qColor2HSV(m_color2, hsv2);

	VTK_CREATE(vtkLookupTable, lut);
	lut->SetHueRange(hsv1[0], hsv2[0]);
	lut->SetSaturationRange(hsv1[1], hsv2[1]);
	lut->SetValueRange(hsv1[2], hsv2[2]);
	lut->SetTableRange(min, max);
	lut->Build();

	return lut;
}

void ecvQVTKWidget::initVtk(vtkSmartPointer<vtkRenderWindowInteractor> interactor, bool useVBO)
{
	this->m_useVBO = useVBO;
	this->m_interactor = interactor;

	this->m_render = this->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
	this->m_camera = m_render->GetActiveCamera();
	this->m_renders = this->GetRenderWindow()->GetRenderers();
}

void ecvQVTKWidget::transformCameraView(const double * viewMat)
{
	vtkSmartPointer<vtkTransform> viewTransform = vtkSmartPointer<vtkTransform>::New();
	viewTransform->SetMatrix(viewMat);
	vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
	cam->ApplyTransform(viewTransform);
	this->m_render->SetActiveCamera(cam);
	this->m_render->Render();
}

void ecvQVTKWidget::transformCameraProjection(const double * projMat)
{
	vtkSmartPointer<vtkMatrix4x4> ProjTransform = vtkSmartPointer<vtkMatrix4x4>::New();
	ProjTransform->Determinant(projMat);
	this->m_camera->SetExplicitProjectionTransformMatrix(ProjTransform);
}

void ecvQVTKWidget::toWorldPoint(const CCVector3d& input2D, CCVector3d& output3D)
{
	m_render->SetDisplayPoint(input2D.x, input2D.y, input2D.z);
	m_render->DisplayToWorld();
	const double* world = m_render->GetWorldPoint();
	for (int i = 0; i < 3; i++)
	{
		output3D.u[i] = world[i] / world[3];
	}
}

void ecvQVTKWidget::toWorldPoint(const CCVector3& input2D, CCVector3d& output3D)
{
	toWorldPoint(CCVector3d::fromArray(input2D.u), output3D);
}

void ecvQVTKWidget::toDisplayPoint(const CCVector3d & worldPos, CCVector3d & displayPos)
{
	m_render->SetWorldPoint(worldPos.x, worldPos.y, worldPos.z, 1.0);
	m_render->WorldToDisplay();
	displayPos.x = (m_render->GetDisplayPoint())[0];
	displayPos.y = (m_render->GetDisplayPoint())[1];
	displayPos.z = (m_render->GetDisplayPoint())[2];
}

void ecvQVTKWidget::toDisplayPoint(const CCVector3 & worldPos, CCVector3d & displayPos)
{
	m_render->SetWorldPoint(worldPos.x, worldPos.y, worldPos.z, 1.0);
	m_render->WorldToDisplay();
	displayPos.x = (m_render->GetDisplayPoint())[0];
	displayPos.y = (m_render->GetDisplayPoint())[1];
	displayPos.z = (m_render->GetDisplayPoint())[2];
}

void ecvQVTKWidget::setCameraPosition(const CCVector3d & pos)
{
	vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
	cam->SetPosition(pos.x, pos.y, pos.z);
	this->m_render->SetActiveCamera(cam);
	this->m_render->Render();
}

void ecvQVTKWidget::setCameraFocalPoint(const CCVector3d & pos)
{
	vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
	cam->SetFocalPoint(pos.x, pos.y, pos.z);
	this->m_render->SetActiveCamera(cam);
	this->m_render->Render();
}

void ecvQVTKWidget::setCameraViewUp(const CCVector3d & pos)
{
	vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
	cam->SetViewUp(pos.x, pos.y, pos.z);
	this->m_render->SetActiveCamera(cam);
	this->m_render->Render();
}

void ecvQVTKWidget::setBackgroundColor(const ecvColor::Rgbf & bkg1, const ecvColor::Rgbf & bkg2, bool gradient)
{
	m_render->SetBackground2(bkg2.r, bkg2.g, bkg2.b);
	m_render->SetBackground(bkg1.r, bkg1.g, bkg1.b);
	m_render->SetGradientBackground(gradient);
}

void ecvQVTKWidget::setMultiViewports(bool multi)
{
	if (d_ptr->multiViewports != multi) {
		d_ptr->multiViewports = multi;
	}
}

bool ecvQVTKWidget::multiViewports() const
{
	return d_ptr->multiViewports;
}

void ecvQVTKWidget::addActor(vtkProp* actor, const QColor& clr)
{
	if (!actor || d_ptr->actors.contains(actor))
		return;

	double vtkClr[3];
	Utils::vtkColor(clr, vtkClr);

	d_ptr->actors.append(actor);

	if (!d_ptr->multiViewports) {
		if (d_ptr->renderers.isEmpty()) {
			vtkRenderer* renderer = vtkRenderer::New();
			renderer->SetBackground(vtkClr);
			d_ptr->configRenderer(renderer);
			renderer->AddActor(actor);
			GetRenderWindow()->AddRenderer(renderer);
			d_ptr->renderers.append(renderer);
			renderer->ResetCamera();
		}
		else {
			defaultRenderer()->SetBackground(vtkClr);
			defaultRenderer()->AddActor(actor);
		}
	}
	else {
		if (!defaultRendererTaken()) {
			defaultRenderer()->SetBackground(vtkClr);
			defaultRenderer()->AddActor(actor);
		}
		else {
			vtkRenderer* renderer = vtkRenderer::New();
			renderer->SetBackground(vtkClr);
			d_ptr->configRenderer(renderer);
			renderer->AddActor(actor);
			GetRenderWindow()->AddRenderer(renderer);
			d_ptr->renderers.append(renderer);
			d_ptr->layoutRenderers();
			renderer->ResetCamera();
		}
	}
}

void ecvQVTKWidget::addViewProp(vtkProp* prop)
{
	if (!prop || d_ptr->props.contains(prop))
		return;

	d_ptr->props.append(prop);

	if (!d_ptr->multiViewports) {
		if (d_ptr->renderers.isEmpty()) {
			vtkRenderer* renderer = vtkRenderer::New();
			d_ptr->configRenderer(renderer);
			renderer->AddViewProp(prop);
			GetRenderWindow()->AddRenderer(renderer);
			d_ptr->renderers.append(renderer);
			renderer->ResetCamera();
		}
		else {
			defaultRenderer()->AddViewProp(prop);
		}
	}
	else {

		if (!defaultRendererTaken()) {
			defaultRenderer()->AddViewProp(prop);
		}
		else {
			vtkRenderer* renderer = vtkRenderer::New();
			d_ptr->configRenderer(renderer);
			renderer->AddViewProp(prop);
			GetRenderWindow()->AddRenderer(renderer);
			d_ptr->renderers.append(renderer);
			d_ptr->layoutRenderers();
			renderer->ResetCamera();
		}
	}
}

QList<vtkProp*> ecvQVTKWidget::actors() const
{
	return d_ptr->actors;
}

void ecvQVTKWidget::setActorsVisible(bool visible)
{
	foreach(auto actor, d_ptr->actors)
		actor->SetVisibility(visible);
}

void ecvQVTKWidget::setActorVisible(vtkProp* actor, bool visible)
{
	actor->SetVisibility(visible);
}

bool ecvQVTKWidget::actorVisible(vtkProp* actor)
{
	return actor->GetVisibility();
}

void ecvQVTKWidget::setBackgroundColor(const QColor& clr)
{
	if (d_ptr->backgroundColor != clr) {
		d_ptr->backgroundColor = clr;

		foreach(vtkRenderer* renderer, d_ptr->renderers)
			d_ptr->configRenderer(renderer);

#if 0
		vtkRendererCollection* renderers = GetRenderWindow()->GetRenderers();
		vtkRenderer* renderer = renderers->GetFirstRenderer();
		while (renderer) {
			renderer = renderers->GetNextItem();
		}
#endif
		update();
	}
}

QColor ecvQVTKWidget::backgroundColor() const
{
	return d_ptr->backgroundColor;
}

vtkRenderer* ecvQVTKWidget::defaultRenderer()
{
	VtkUtils::vtkInitOnce(&d_ptr->defaultRenderer);
	GetRenderWindow()->AddRenderer(d_ptr->defaultRenderer);
	if (!d_ptr->renderers.contains(d_ptr->defaultRenderer))
		d_ptr->renderers.append(d_ptr->defaultRenderer);
	return d_ptr->defaultRenderer;
}

bool ecvQVTKWidget::defaultRendererTaken() const
{
	if (!d_ptr->defaultRenderer)
		return false;
	return d_ptr->defaultRenderer->GetActors()->GetNumberOfItems() != 0;
}

void ecvQVTKWidget::setBounds(double* bounds)
{
	Utils::ArrayAssigner<double, 6> aa;
	aa(bounds, d_ptr->bounds);
}

double ecvQVTKWidget::xMin() const
{
	return d_ptr->bounds[0];
}

double ecvQVTKWidget::xMax() const
{
	return d_ptr->bounds[1];
}

double ecvQVTKWidget::yMin() const
{
	return d_ptr->bounds[2];
}

double ecvQVTKWidget::yMax() const
{
	return d_ptr->bounds[3];
}

double ecvQVTKWidget::zMin() const
{
	return d_ptr->bounds[4];
}

double ecvQVTKWidget::zMax() const
{
	return d_ptr->bounds[5];
}

// event processing
void ecvQVTKWidget::mousePressEvent(QMouseEvent *event)
{
	m_tools->m_mouseMoved = false;
	m_tools->m_mouseButtonPressed = true;
	m_tools->m_ignoreMouseReleaseEvent = false;
	m_tools->m_lastMousePos = event->pos();

	if (!ecvDisplayTools::USE_VTK_PICK)
	{
		m_tools->m_last_point_index = -1;
		m_tools->m_last_picked_id = QString();
	}

	if ((event->buttons() & Qt::RightButton))
	{
		//right click = panning (2D translation)
		if ((m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_PAN) || 
			((QApplication::keyboardModifiers() & Qt::ControlModifier) && 
			(m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_CTRL_PAN)))
		{
			QApplication::setOverrideCursor(QCursor(Qt::SizeAllCursor));
		}

		if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_SIG_RB_CLICKED)
		{
			emit m_tools->rightButtonClicked(event->x(), event->y());
		}
	}
	else if (event->buttons() & Qt::LeftButton)
	{
		m_tools->m_lastClickTime_ticks = m_tools->m_timer.elapsed(); //in msec

		//left click = rotation
		if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_ROTATE)
		{
			QApplication::setOverrideCursor(QCursor(Qt::PointingHandCursor));
		}

		if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_SIG_LB_CLICKED)
		{
			emit m_tools->leftButtonClicked(event->x(), event->y());
		}

		//do this before drawing the pivot!
		if (m_tools->m_autoPickPivotAtCenter)
		{
			CCVector3d P;
			//if (m_tools->GetClick3DPos(m_tools->m_glViewport.width() / 2, m_tools->m_glViewport.height() / 2, P))
			if (m_tools->GetClick3DPos(event->x(), event->y(), P))
			{
				ecvDisplayTools::SetPivotPoint(P, true, false);
			}
		}
	}
	else
	{
	}

	QVTKWidget::mousePressEvent(event);
}

void ecvQVTKWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
	m_tools->m_deferredPickingTimer.stop(); //prevent the picking process from starting
	m_tools->m_ignoreMouseReleaseEvent = true;

	const int x = event->x();
	const int y = event->y();

	CCVector3d P;
	if (ecvDisplayTools::GetClick3DPos(x, y, P))
	{
		ecvDisplayTools::SetPivotPoint(P, true, true);
	}

	emit m_tools->doubleButtonClicked(event->x(), event->y());

	QVTKWidget::mouseDoubleClickEvent(event);
}

void ecvQVTKWidget::wheelEvent(QWheelEvent * event)
{
	bool doRedraw = false;
	Qt::KeyboardModifiers keyboardModifiers = QApplication::keyboardModifiers();

	if (keyboardModifiers & Qt::AltModifier)
	{
		event->accept();

		//same shortcut as Meshlab: change the point size
		float sizeModifier = (event->delta() < 0 ? -1.0f : 1.0f);
		ecvDisplayTools::SetPointSize(m_tools->m_viewportParams.defaultPointSize + sizeModifier);
		ecvDisplayTools::SetRedrawRecursive(false);
		ecvDisplayTools::RedrawDisplay();
		doRedraw = true;
	}
	else if (keyboardModifiers & Qt::ControlModifier)
	{
		event->accept();
		if (m_tools->m_viewportParams.perspectiveView)
		{
			//same shortcut as Meshlab: change the zNear value
			static const int MAX_INCREMENT = 150;
			int increment = ecvViewportParameters::ZNearCoefToIncrement(m_tools->m_viewportParams.zNearCoef, MAX_INCREMENT + 1);
			int newIncrement = std::min(std::max(0, increment + (event->delta() < 0 ? -1 : 1)), MAX_INCREMENT); //the zNearCoef must be < 1! 
			if (newIncrement != increment)
			{
				double newCoef = ecvViewportParameters::IncrementToZNearCoef(newIncrement, MAX_INCREMENT + 1);
				ecvDisplayTools::SetZNearCoef(newCoef);
				ecvDisplayTools::SetRedrawRecursive(false);
				ecvDisplayTools::RedrawDisplay();
				doRedraw = true;
			}
		}
	}
	else if (keyboardModifiers & Qt::ShiftModifier)
	{
		event->accept();
		if (m_tools->m_viewportParams.perspectiveView)
		{
			//same shortcut as Meshlab: change the fov value
			float newFOV = (m_tools->m_viewportParams.fov + (event->delta() < 0 ? -1.0f : 1.0f));
			newFOV = std::min(std::max(1.0f, newFOV), 180.0f);
			if (newFOV != m_tools->m_viewportParams.fov)
			{
				ecvDisplayTools::SetFov(newFOV);
				ecvDisplayTools::SetRedrawRecursive(false);
				ecvDisplayTools::RedrawDisplay();
				doRedraw = true;
			}
		}
	}
	else if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_ZOOM_CAMERA)
	{
		QVTKWidget::wheelEvent(event);

		//see QWheelEvent documentation ("distance that the wheel is rotated, in eighths of a degree")
		float wheelDelta_deg = static_cast<float>(event->delta()) / 8;
		m_tools->onWheelEvent(wheelDelta_deg);
		emit m_tools->mouseWheelRotated(wheelDelta_deg);
		emit m_tools->cameraParamChanged();

		doRedraw = true;
		event->accept();
	}

	if (doRedraw)
	{
		// update label and 3D name if visible
		emit m_tools->labelmove2D(0, 0, 0, 0);
		ecvDisplayTools::UpdateNamePoseRecursive();
		m_tools->Update2DLabel(true);
		ecvDisplayTools::Update();
	}
}

void ecvQVTKWidget::mouseMoveEvent(QMouseEvent *event)
{
	if (!((m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_ROTATE) &&
		(m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES)))
	{
		if ((m_tools->m_interactionFlags & ecvDisplayTools::TRANSFORM_CAMERA()))
		{
			QVTKWidget::mouseMoveEvent(event);
			m_tools->UpdateDisplayParameters();
		}
	}

	const int x = event->x();
	const int y = event->y();
	// update mouse coordinate in status bar
	m_tools->m_lastMouseMovePos = event->pos();
	emit m_tools->mousePosChanged(event->pos());

	if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_SIG_MOUSE_MOVED)
	{
		emit m_tools->mouseMoved(x, y, event->buttons());
		event->accept();
	}

	//no button pressed
	if (event->buttons() == Qt::NoButton)
	{
		if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_CLICKABLE_ITEMS)
		{
			//what would be the size of the 'hot zone' if it was displayed with all options
			if (!m_tools->m_hotZone)
			{
				m_tools->m_hotZone = new ecvDisplayTools::HotZone(this);
			}
			QRect areaRect = m_tools->m_hotZone->rect(true, m_tools->m_bubbleViewModeEnabled, ecvDisplayTools::ExclusiveFullScreen());

			const int retinaScale = ecvDisplayTools::GetDevicePixelRatio();
			bool inZone = (x * retinaScale * 3 < m_tools->m_hotZone->topCorner.x() + areaRect.width() * 4   //25% margin
				&& y * retinaScale * 2 < m_tools->m_hotZone->topCorner.y() + areaRect.height() * 4); //50% margin

			if (inZone != m_tools->m_clickableItemsVisible)
			{
				m_tools->m_clickableItemsVisible = inZone;
				ecvDisplayTools::RedrawDisplay(true, false);
			}

			event->accept();
		}

		//display the 3D coordinates of the pixel below the mouse cursor (if possible)
		if (m_tools->m_showCursorCoordinates)
		{
			CCVector3d P;
			QString message = QString("2D (%1 ; %2)").arg(x).arg(y);
			if ( ecvDisplayTools::GetClick3DPos(x, y, P))
			{
				message += QString(" --> 3D (%1 ; %2 ; %3)").arg(P.x).arg(P.y).arg(P.z);
			}
			ecvDisplayTools::DisplayNewMessage(message, ecvDisplayTools::LOWER_LEFT_MESSAGE, false, 5, ecvDisplayTools::SCREEN_SIZE_MESSAGE);
			ecvDisplayTools::RedrawDisplay(true);
		}

		//don't need to process any further
		return;
	}

	int dx = x - m_tools->m_lastMousePos.x();
	int dy = y - m_tools->m_lastMousePos.y();

	if ((event->buttons() & Qt::RightButton))
	{
		// update 2D labels
		if (abs(dx) > 0 || abs(dy) > 0)
		{
			// update label and 3D name if visible
			m_tools->Update2DLabel(true);
			emit m_tools->labelmove2D(x, y, 0, 0);
			ecvDisplayTools::UpdateNamePoseRecursive();
		}
	}
	else if ((event->buttons() & Qt::MidButton))
	{
		//right button = panning / translating
		if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_PAN)
		{
			//displacement vector (in "3D")
			double pixSize = ecvDisplayTools::ComputeActualPixelSize();
			CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
			if (!m_tools->m_viewportParams.perspectiveView)
			{
				u.y *= m_tools->m_viewportParams.orthoAspectRatio;
			}

			const int retinaScale = ecvDisplayTools::GetDevicePixelRatio();
			u *= retinaScale;

			bool entityMovingMode = (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES)
				|| ((QApplication::keyboardModifiers() & Qt::ControlModifier) && m_tools->m_customLightEnabled);
			if (entityMovingMode)
			{
				//apply inverse view matrix
				m_tools->m_viewportParams.viewMat.transposed().applyRotation(u);

				if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES)
				{
					emit m_tools->translation(u);
				}
				else if (m_tools->m_customLightEnabled)
				{
					//update custom light position
					m_tools->m_customLightPos[0] += static_cast<float>(u.x);
					m_tools->m_customLightPos[1] += static_cast<float>(u.y);
					m_tools->m_customLightPos[2] += static_cast<float>(u.z);
					ecvDisplayTools::InvalidateViewport();
					ecvDisplayTools::Deprecate3DLayer();
				}
			}
			else //camera moving mode
			{
				if (m_tools->m_viewportParams.objectCenteredView)
				{
					//inverse displacement in object-based mode
					u = -u;
				}
				ecvDisplayTools::MoveCamera(static_cast<float>(u.x), static_cast<float>(u.y), static_cast<float>(u.z));
			}

		} //if (m_interactionFlags & INTERACT_PAN)

		if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_2D_ITEMS)
		{
			//on the first time, let's check if the mouse is on a (selected) 2D item
			if (!m_tools->m_mouseMoved)
			{
				if (m_tools->m_pickingMode != ecvDisplayTools::NO_PICKING
					//DGM: in fact we still need to move labels in those modes below (see the 'Point Picking' tool of CLOUDVIEWER  for instance)
					//&&	m_pickingMode != POINT_PICKING
					//&&	m_pickingMode != TRIANGLE_PICKING
					//&&	m_pickingMode != POINT_OR_TRIANGLE_PICKING
					&& (QApplication::keyboardModifiers() == Qt::NoModifier
						|| QApplication::keyboardModifiers() == Qt::ControlModifier))
				{
					ecvDisplayTools::UpdateActiveItemsList(m_tools->m_lastMousePos.x(), m_tools->m_lastMousePos.y(), true);
				}
			}
		}

		if (abs(dx) > 0 || abs(dy) > 0)
		{
			emit m_tools->labelmove2D(x, y, dx, dy);
			ecvDisplayTools::UpdateNamePoseRecursive();
			//specific case: move active item(s)
			if (!m_tools->m_activeItems.empty())
			{
				updateActivateditems(x, y, dx, dy, !ecvDisplayTools::USE_2D);
			}
		}
	}
	else if (event->buttons() & Qt::LeftButton) //rotation
	{
		m_tools->scheduleFullRedraw(1000);

		if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_2D_ITEMS)
		{
			//on the first time, let's check if the mouse is on a (selected) 2D item
			if (!m_tools->m_mouseMoved)
			{
				if (m_tools->m_pickingMode != ecvDisplayTools::NO_PICKING
					//DGM: in fact we still need to move labels in those modes below (see the 'Point Picking' tool of CLOUDVIEWER  for instance)
					//&&	m_pickingMode != POINT_PICKING
					//&&	m_pickingMode != TRIANGLE_PICKING
					//&&	m_pickingMode != POINT_OR_TRIANGLE_PICKING
					&& (QApplication::keyboardModifiers() == Qt::NoModifier
						|| QApplication::keyboardModifiers() == Qt::ControlModifier))
				{
					ecvDisplayTools::UpdateActiveItemsList(m_tools->m_lastMousePos.x(), m_tools->m_lastMousePos.y(), true);
				}
			}
		}
		else
		{
			//assert(m_tools->m_activeItems.empty());
			m_tools->m_activeItems.clear();
		}

		// update label and 3D name if visible
		if (abs(dx) > 0 || abs(dy) > 0)
		{
			emit m_tools->labelmove2D(x, y, dx, dy);
			ecvDisplayTools::UpdateNamePoseRecursive();
		}
		
		//specific case: move active item(s)
		if (!m_tools->m_activeItems.empty())
		{
			if (abs(dx) > 0 || abs(dy) > 0)
			{
				updateActivateditems(x, y, dx, dy, !ecvDisplayTools::USE_2D);
			}
		}
		else
		{
			//specific case: rectangular polyline drawing (for rectangular area selection mode)
			if (m_tools->m_allowRectangularEntityPicking
				&& (m_tools->m_pickingMode == ecvDisplayTools::ENTITY_PICKING || m_tools->m_pickingMode == ecvDisplayTools::ENTITY_RECT_PICKING)
				&& (m_tools->m_rectPickingPoly || (QApplication::keyboardModifiers() & Qt::AltModifier)))
			{
				//first time: initialization of the rectangle
				if (!m_tools->m_rectPickingPoly)
				{
					ccPointCloud* vertices = new ccPointCloud("rect.vertices");
					m_tools->m_rectPickingPoly = new ccPolyline(vertices);
					m_tools->m_rectPickingPoly->addChild(vertices);
					if (vertices->reserve(4) && m_tools->m_rectPickingPoly->addPointIndex(0, 4))
					{
						m_tools->m_rectPickingPoly->setForeground(true);
						m_tools->m_rectPickingPoly->setColor(ecvColor::green);
						m_tools->m_rectPickingPoly->showColors(true);
						m_tools->m_rectPickingPoly->set2DMode(true);
						m_tools->m_rectPickingPoly->setVisible(true);
						//QPointF posA = ecvDisplayTools::ToCenteredGLCoordinates(m_tools->m_lastMousePos.x(), m_tools->m_lastMousePos.y());
						CCVector3d pos3D = ecvDisplayTools::ToVtkCoordinates(m_tools->m_lastMousePos.x(), m_tools->m_lastMousePos.y());

						CCVector3 A(static_cast<PointCoordinateType>(pos3D.x),
							static_cast<PointCoordinateType>(pos3D.y), pos3D.z);
						//we add 4 times the same point (just to fill the cloud!)
						vertices->addPoint(A);
						vertices->addPoint(A);
						vertices->addPoint(A);
						vertices->addPoint(A);
						m_tools->m_rectPickingPoly->setClosed(true);
						ecvDisplayTools::AddToOwnDB(m_tools->m_rectPickingPoly, false);
					}
					else
					{
						CVLog::Warning("[ ecvQVTKWidget::mouseMoveEvent] Failed to create seleciton polyline! Not enough memory!");
						delete m_tools->m_rectPickingPoly;
						m_tools->m_rectPickingPoly = nullptr;
						vertices = nullptr;
					}
				}

				if (m_tools->m_rectPickingPoly)
				{
					CVLib::GenericIndexedCloudPersist* vertices = 
						m_tools->m_rectPickingPoly->getAssociatedCloud();
					assert(vertices);
					CCVector3* B = const_cast<CCVector3*>(vertices->getPointPersistentPtr(1));
					CCVector3* C = const_cast<CCVector3*>(vertices->getPointPersistentPtr(2));
					CCVector3* D = const_cast<CCVector3*>(vertices->getPointPersistentPtr(3));
					//QPointF posD = ecvDisplayTools::ToCenteredGLCoordinates(event->x(), event->y());
					CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(event->x(), event->y());
					B->x = C->x = static_cast<PointCoordinateType>(pos2D.x);
					C->y = D->y = static_cast<PointCoordinateType>(pos2D.y);
				}
			}
			//standard rotation around the current pivot
			else if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_ROTATE)
			{
				// choose the right rotation mode
				enum RotationMode { StandardMode, BubbleViewMode, LockedAxisMode };
				RotationMode rotationMode = StandardMode;
				if ((m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES) !=
					ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES)
				{
					if (m_tools->m_bubbleViewModeEnabled)
						rotationMode = BubbleViewMode;
					else if (m_tools->m_rotationAxisLocked)
						rotationMode = LockedAxisMode;
				}

				ccGLMatrixd rotMat;
				switch (rotationMode)
				{
				case BubbleViewMode:
				{
					QPoint posDelta = m_tools->m_lastMousePos - event->pos();

					if (std::abs(posDelta.x()) != 0)
					{
						double delta_deg = (posDelta.x() * static_cast<double>(m_tools->m_bubbleViewFov_deg)) / height();
						// rotation about the sensor Z axis
						CCVector3d axis = m_tools->m_viewportParams.viewMat.getColumnAsVec3D(2);
						rotMat.initFromParameters(delta_deg * CV_DEG_TO_RAD, axis, CCVector3d(0, 0, 0));
					}

					if (std::abs(posDelta.y()) != 0)
					{
						double delta_deg = (posDelta.y() * static_cast<double>(m_tools->m_bubbleViewFov_deg)) / height();
						// rotation about the local X axis
						ccGLMatrixd rotX;
						rotX.initFromParameters(delta_deg * CV_DEG_TO_RAD, CCVector3d(1, 0, 0), CCVector3d(0, 0, 0));
						rotMat = rotX * rotMat;
					}
				}
				break;

				case StandardMode:
				{
					static CCVector3d s_lastMouseOrientation;
					if (!m_tools->m_mouseMoved)
					{
						//on the first time, we must compute the previous orientation (the camera hasn't moved yet)
						s_lastMouseOrientation = ecvDisplayTools::ConvertMousePositionToOrientation(
							m_tools->m_lastMousePos.x(), m_tools->m_lastMousePos.y());
					}

					CCVector3d currentMouseOrientation = ecvDisplayTools::ConvertMousePositionToOrientation(x, y);
					rotMat = ccGLMatrixd::FromToRotation(s_lastMouseOrientation, currentMouseOrientation);
					s_lastMouseOrientation = currentMouseOrientation;
				}
				break;

				case LockedAxisMode:
				{
					//apply rotation about the locked axis
					CCVector3d axis = m_tools->m_lockedRotationAxis;
					ecvDisplayTools::GetBaseViewMat().applyRotation(axis);

					//determine whether we are in a side or top view
					bool topView = (std::abs(axis.z) > 0.5);

					//m_tools->m_viewportParams.objectCenteredView
					ccGLCameraParameters camera;
					ecvDisplayTools::GetGLCameraParameters(camera);

					if (topView)
					{
						//rotation origin
						CCVector3d C2D;
						if (m_tools->m_viewportParams.objectCenteredView)
						{
							//project the current pivot point on screen
							camera.project(m_tools->m_viewportParams.pivotPoint, C2D);
							C2D.z = 0.0;
						}
						else
						{
							C2D = CCVector3d(width() / 2.0, height() / 2.0, 0.0);
						}

						CCVector3d previousMousePos(static_cast<double>(m_tools->m_lastMousePos.x()), 
							static_cast<double>(height() - m_tools->m_lastMousePos.y()), 0.0);
						CCVector3d currentMousePos(static_cast<double>(x), static_cast<double>(height() - y), 0.0);

						CCVector3d a = (currentMousePos - C2D);
						CCVector3d b = (previousMousePos - C2D);
						CCVector3d u = a * b;
						double u_norm = std::abs(u.z); //a and b are in the XY plane
						if (u_norm > 1.0e-6)
						{
							double sin_angle = u_norm / (a.norm() * b.norm());

							//determine the rotation direction
							if (u.z * m_tools->m_lockedRotationAxis.z > 0)
							{
								sin_angle = -sin_angle;
							}

							double angle_rad = asin(sin_angle); //in [-pi/2 ; pi/2]
							rotMat.initFromParameters(angle_rad, axis, CCVector3d(0, 0, 0));
						}
					}
					else //side view
					{
						//project the current pivot point on screen
						CCVector3d A2D, B2D;
						if (camera.project(m_tools->m_viewportParams.pivotPoint, A2D)
							&& camera.project(m_tools->m_viewportParams.pivotPoint + 
								m_tools->m_viewportParams.zFar * m_tools->m_lockedRotationAxis, B2D))
						{
							CCVector3d lockedRotationAxis2D = B2D - A2D;
							lockedRotationAxis2D.z = 0; //just in case
							lockedRotationAxis2D.normalize();

							CCVector3d mouseShift(static_cast<double>(dx), -static_cast<double>(dy), 0.0);
							mouseShift -= mouseShift.dot(lockedRotationAxis2D) * lockedRotationAxis2D; //we only keep the orthogonal part
							double angle_rad = 2.0 * M_PI * mouseShift.norm() / (width() + height());
							if ((lockedRotationAxis2D * mouseShift).z > 0.0)
							{
								angle_rad = -angle_rad;
							}

							rotMat.initFromParameters(angle_rad, axis, CCVector3d(0, 0, 0));
						}
					}
				}
				break;

				default:
					assert(false);
					break;
				}

				if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES)
				{
					rotMat = m_tools->m_viewportParams.viewMat.transposed() * rotMat * m_tools->m_viewportParams.viewMat;
					//feedback for 'interactive transformation' mode
					emit m_tools->rotation(rotMat);
				}
				else
				{
					ecvDisplayTools::RotateBaseViewMat(rotMat);

					//ecvDisplayTools::ShowPivotSymbol(true);
					QApplication::changeOverrideCursor(QCursor(Qt::ClosedHandCursor));

					 //feedback for 'echo' mode
					emit m_tools->viewMatRotated(rotMat);
				}

			}
		}
	}

	m_tools->m_mouseMoved = true;
	m_tools->m_lastMousePos = event->pos();
	emit m_tools->cameraParamChanged();
	event->accept();
}

void ecvQVTKWidget::updateActivateditems(int x, int y, int dx, int dy, bool updatePosition)
{
	if (updatePosition)
	{
		//displacement vector (in "3D")
		double pixSize = ecvDisplayTools::ComputeActualPixelSize();
		CCVector3d u(dx*pixSize, -dy * pixSize, 0.0);
		m_tools->m_viewportParams.viewMat.transposed().applyRotation(u);

		const int retinaScale = ecvDisplayTools::GetDevicePixelRatio();
		u *= retinaScale;

		for (auto &activeItem : m_tools->m_activeItems)
		{

			if (activeItem->move2D(
				x * retinaScale, y * retinaScale,
				dx * retinaScale, dy * retinaScale,
				ecvDisplayTools::GlWidth(),
				ecvDisplayTools::GlHeight()))
			{
				ecvDisplayTools::InvalidateViewport();
			}
			else if (activeItem->move3D(u))
			{
				ecvDisplayTools::InvalidateViewport();
				ecvDisplayTools::Deprecate3DLayer();
			}
		}
	}

	ecvDisplayTools::Redraw2DLabel();
}

void ecvQVTKWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if (m_tools->m_interactionFlags & ecvDisplayTools::TRANSFORM_CAMERA())
	{
		QVTKWidget::mouseReleaseEvent(event);
	}

	if (m_tools->m_ignoreMouseReleaseEvent)
	{
		m_tools->m_ignoreMouseReleaseEvent = false;
		return;
	}
	bool mouseHasMoved = m_tools->m_mouseMoved;

	//reset to default state
	m_tools->m_mouseButtonPressed = false;
	m_tools->m_mouseMoved = false;
	QApplication::restoreOverrideCursor();

	if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_SIG_BUTTON_RELEASED)
	{
		event->accept();
		emit m_tools->buttonReleased();
	}

	if (m_tools->m_pivotSymbolShown)
	{
		if (m_tools->m_pivotVisibility == 
			ecvDisplayTools::PIVOT_SHOW_ON_MOVE)
		{
			ecvDisplayTools::ToBeRefreshed();
		}
		ecvDisplayTools::ShowPivotSymbol(
			m_tools->m_pivotVisibility == 
			ecvDisplayTools::PIVOT_ALWAYS_SHOW);
	}

	if ((event->button() == Qt::MidButton))
	{
		if (mouseHasMoved)
		{
			event->accept();
			m_tools->m_activeItems.clear();
			//ecvDisplayTools::ToBeRefreshed();
		}
		else if (m_tools->m_interactionFlags & ecvDisplayTools::INTERACT_2D_ITEMS)
		{
			//interaction with 2D item(s)
			ecvDisplayTools::UpdateActiveItemsList(event->x(), event->y(), false);
			if (!m_tools->m_activeItems.empty())
			{
				ccInteractor* item = m_tools->m_activeItems.front();
				m_tools->m_activeItems.clear();
				if (item->acceptClick(event->x(), height() - 1 - event->y(), Qt::MidButton))
				{
					event->accept();
				}
			}
		}
	}
	else if (event->button() == Qt::LeftButton)
	{
		if (mouseHasMoved)
		{
			//if a rectangular picking area has been defined
			if (m_tools->m_rectPickingPoly)
			{
				CVLib::GenericIndexedCloudPersist* vertices = m_tools->m_rectPickingPoly->getAssociatedCloud();
				assert(vertices);
				const CCVector3* A = vertices->getPointPersistentPtr(0);
				const CCVector3* C = vertices->getPointPersistentPtr(2);

				int pickX = static_cast<int>(A->x + C->x) / 2;
				int pickY = static_cast<int>(A->y + C->y) / 2;
				int pickW = static_cast<int>(std::abs(C->x - A->x));
				int pickH = static_cast<int>(std::abs(C->y - A->y));

				ecvDisplayTools::RemoveFromOwnDB(m_tools->m_rectPickingPoly);
				m_tools->m_rectPickingPoly = nullptr;
				vertices = nullptr;

				ecvDisplayTools::PickingParameters params(ecvDisplayTools::ENTITY_RECT_PICKING, 
					pickX + ecvDisplayTools::Width() / 2, ecvDisplayTools::Height() / 2 - pickY, pickW, pickH);
				ecvDisplayTools::StartPicking(params);
				ecvDisplayTools::ToBeRefreshed();
			}

			event->accept();
		}
		else
		{
			//picking?
			if (m_tools->m_timer.elapsed() < m_tools->m_lastClickTime_ticks + CC_MAX_PICKING_CLICK_DURATION_MS) //in msec
			{
				int x = m_tools->m_lastMousePos.x();
				int y = m_tools->m_lastMousePos.y();

				// first test if the user has clicked on a particular item on the screen
				if (!ecvDisplayTools::ProcessClickableItems(x, y))
				{
					m_tools->m_lastMousePos = event->pos(); // just in case (it should be already at this position)
					m_tools->m_deferredPickingTimer.start();
				}
			}
		}

		m_tools->m_activeItems.clear();
	}

	ecvDisplayTools::RefreshDisplay(true);
}

void ecvQVTKWidget::dragEnterEvent(QDragEnterEvent * event)
{
	const QMimeData* mimeData = event->mimeData();
	if (mimeData->hasFormat("text/uri-list"))
		event->acceptProposedAction();

	QVTKWidget::dragEnterEvent(event);
}

void ecvQVTKWidget::dropEvent(QDropEvent * event)
{
	const QMimeData* mimeData = event->mimeData();

	if (mimeData && mimeData->hasFormat("text/uri-list"))
	{
		QStringList fileNames;
		for (const QUrl &url : mimeData->urls()) {
			QString fileName = url.toLocalFile();
			fileNames.append(fileName);
#ifdef QT_DEBUG
			CVLog::Print(QString("File dropped: %1").arg(fileName));
#endif
		}

		if (!fileNames.empty())
		{
			emit m_tools->filesDropped(fileNames);
		}

		event->acceptProposedAction();
	}

	QVTKWidget::dropEvent(event);
	event->ignore();
}

bool ecvQVTKWidget::event(QEvent* evt)
{
	switch (evt->type())
	{
		//Gesture start/stop
		case QEvent::TouchBegin:
		case QEvent::TouchEnd:
		{
			QTouchEvent* touchEvent = static_cast<QTouchEvent*>(evt);
			touchEvent->accept();
			m_tools->m_touchInProgress = (evt->type() == QEvent::TouchBegin);
			m_tools->m_touchBaseDist = 0.0;
			CVLog::PrintDebug(QString("Touch event %1").arg(m_tools->m_touchInProgress ? "begins" : "ends"));
		}
		break;

		case QEvent::Close:
		{
			if (m_unclosable)
			{
				evt->ignore();
			}
			else
			{
				evt->accept();
			}
		}
		return true;

		case QEvent::DragEnter:
		{
			dragEnterEvent(static_cast<QDragEnterEvent*>(evt));
		}
		return true;

		case QEvent::Drop:
		{
			dropEvent(static_cast<QDropEvent*>(evt));
		}
		return true;

		case QEvent::TouchUpdate:
		{
			//Gesture update
			if (m_tools->m_touchInProgress && !m_tools->m_viewportParams.perspectiveView)
			{
				QTouchEvent* touchEvent = static_cast<QTouchEvent*>(evt);
				const QList<QTouchEvent::TouchPoint>& touchPoints = touchEvent->touchPoints();
				if (touchPoints.size() == 2)
				{
					QPointF D = (touchPoints[1].pos() - touchPoints[0].pos());
					qreal dist = std::sqrt(D.x()*D.x() + D.y()*D.y());
					if (m_tools->m_touchBaseDist != 0.0)
					{
						float zoomFactor = dist / m_tools->m_touchBaseDist;
						ecvDisplayTools::UpdateZoom(zoomFactor);
					}
					m_tools->m_touchBaseDist = dist;
					evt->accept();
					break;
				}
			}
			CVLog::PrintDebug(QString("Touch update (%1 points)").arg(static_cast<QTouchEvent*>(evt)->touchPoints().size()));
		}
		break;

		case QEvent::Resize:
		{
			QSize newSize = static_cast<QResizeEvent*>(evt)->size();
			ecvDisplayTools::ResizeGL(newSize.width(), newSize.height());
			evt->accept();
		}
		break;

		default:
		{
			//CVLog::Print("Unhandled event: %i", evt->type());
		}
		break;

	}

	return QVTKWidget::event(evt);
}

void ecvQVTKWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key())
	{
	case Qt::Key_Escape:
	{
		m_app->toggleExclusiveFullScreen(false);
		break;
	}

	default:
		QVTKWidget::keyPressEvent(event);
	}
}