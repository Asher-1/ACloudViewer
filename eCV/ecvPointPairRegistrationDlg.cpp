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

#include <ecvPointPairRegistrationDlg.h>

// LOCAL
#include "MainWindow.h"
#include "ecvAskThreeDoubleValuesDlg.h"

// COMMON
#include <ecvPickingHub.h>

// CV_CORE_LIB
#include <RegistrationTools.h>
#include <GeometricalAnalysisTools.h>

//ECV_DB_LIB
#include <ecvGenericPointCloud.h>
#include <ecv2DLabel.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvSphere.h>
#include <ecvDisplayTools.h>

// ECV_IO_LIB
#include <ecvGlobalShiftManager.h>

// QT
#include <QMdiSubWindow>
#include <QMessageBox>
#include <QToolButton>
#include <QSettings>

//default position of each columns in the aligned and ref. table widgets
static const int XYZ_COL_INDEX			= 0;
static const int RMS_COL_INDEX			= 3;
static const int DEL_BUTTON_COL_INDEX	= 4;

//minimum number of pairs to let the user click on the align button
static const unsigned MIN_PAIRS_COUNT = 3;

ccPointPairRegistrationDlg::ccPointPairRegistrationDlg(
	ccPickingHub* pickingHub, 
	ecvMainAppInterface* app, 
	QWidget* parent/*=0*/)
	: ccOverlayDialog(parent)
	, m_alignedPoints("aligned points")
	, m_refPoints("reference points")
	, m_refLabels("reference labels")
	, m_alignedLabels("aligned labels")
	, m_paused(false)
	, m_pickingHub(pickingHub)
	, m_app(app)
{
	assert(m_pickingHub);

	setupUi(this);

	//restore from persistent settings
	{
		QSettings settings;
		settings.beginGroup("PointPairAlign");
		bool pickSpheres    = settings.value("PickSpheres",  useSphereToolButton->isChecked()).toBool();
		double sphereRadius = settings.value("SphereRadius", radiusDoubleSpinBox->value()).toDouble();
		int maxRMS          = settings.value("MaxRMS",       maxRmsSpinBox->value()).toInt();
		bool adjustScale    = settings.value("AdjustScale",  adjustScaleCheckBox->isChecked()).toBool();
		bool autoUpdateZoom = settings.value("AutoUpdateZom",autoZoomCheckBox->isChecked()).toBool();
		settings.endGroup();

		useSphereToolButton->setChecked(pickSpheres);
		radiusDoubleSpinBox->setValue(sphereRadius);
		maxRmsSpinBox->setValue(maxRMS);
		adjustScaleCheckBox->setChecked(adjustScale);
		autoZoomCheckBox->setChecked(autoUpdateZoom);
	}

    connect(showAlignedCheckBox,	&QCheckBox::toggled,	this,	&ccPointPairRegistrationDlg::showAlignedEntities);
    connect(showReferenceCheckBox,	&QCheckBox::toggled,	this,	&ccPointPairRegistrationDlg::showReferenceEntities);

    connect(typeAlignToolButton,	&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::addManualAlignedPoint);
    connect(typeRefToolButton,		&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::addManualRefPoint);

    connect(unstackAlignToolButton,	&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::unstackAligned);
    connect(unstackRefToolButton,	&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::unstackRef);

    connect(alignToolButton,		&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::align);
    connect(resetToolButton,		&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::reset);

    connect(validToolButton,		&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::apply);
    connect(cancelToolButton,		&QToolButton::clicked,	this,	&ccPointPairRegistrationDlg::cancel);

    connect(adjustScaleCheckBox,	&QCheckBox::toggled,	this,	&ccPointPairRegistrationDlg::updateAlignInfo);
    connect(TxCheckBox,				&QCheckBox::toggled,	this,	&ccPointPairRegistrationDlg::updateAlignInfo);
    connect(TyCheckBox,				&QCheckBox::toggled,	this,	&ccPointPairRegistrationDlg::updateAlignInfo);
    connect(TzCheckBox,				&QCheckBox::toggled,	this,	&ccPointPairRegistrationDlg::updateAlignInfo);
    connect(rotComboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),	this, &ccPointPairRegistrationDlg::updateAlignInfo);

	m_alignedPoints.setEnabled(true);
	m_alignedPoints.setVisible(false);

	m_refPoints.setEnabled(true);
	m_refPoints.setVisible(false);
}

ccPointPairRegistrationDlg::EntityContext::EntityContext(ccHObject* ent)
	: entity(ent)
	, wasVisible(entity ? entity->isVisible() : false)
	, wasEnabled(entity ? entity->isEnabled() : false)
	, wasSelected(entity ? entity->isSelected() : false)
{
}

void ccPointPairRegistrationDlg::EntityContext::restore()
{
	if (!entity)
		return;

	entity->setVisible(wasVisible);
	entity->setEnabled(wasEnabled);
	entity->setSelected(wasSelected);
	if (ecvDisplayTools::GetMainScreen())
	{
		ecvDisplayTools::SetRedrawRecursive(false);
		ecvDisplayTools::RedrawDisplay(false, false);
	}
}

void ccPointPairRegistrationDlg::EntityContexts::fill(const ccHObject::Container& entities)
{
	clear();
	isShifted = false;

	for (ccHObject* entity : entities)
	{
		if (!entity)
		{
			assert(false);
			continue;
		}

		if (!isShifted)
		{
			ccShiftedObject* shiftedEntity = ccHObjectCaster::ToShifted(entity);
			if (shiftedEntity && shiftedEntity->isShifted())
			{
				isShifted = true;
				shift = shiftedEntity->getGlobalShift(); //we can only consider the first shift!
			}
		}

		insert(entity, EntityContext(entity));
	}
}

void ccPointPairRegistrationDlg::clear()
{
	alignToolButton->setEnabled(false);
	validToolButton->setEnabled(false);
	while (alignedPointsTableWidget->rowCount() != 0)
		alignedPointsTableWidget->removeRow(alignedPointsTableWidget->rowCount() - 1);
	while (refPointsTableWidget->rowCount() != 0)
		refPointsTableWidget->removeRow(refPointsTableWidget->rowCount() - 1);

	for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
	{
		it.key()->setSelected(true);
		it.key()->showSF(false);
	}	
	for (auto it = m_referenceEntities.begin(); it != m_referenceEntities.end(); ++it)
	{
		it.key()->setSelected(true);
		it.key()->showSF(false);
	}

	if (ecvDisplayTools::GetCurrentScreen())
	{
		// clear aligned markers
		for (unsigned i = 0; i < m_alignedLabels.getChildrenNumber(); ++i)
		{
			ccHObject* child = m_alignedLabels.getChild(i);
			if (child && child->isKindOf(CV_TYPES::LABEL_2D))
			{
				cc2DLabel* alignedLabel = ccHObjectCaster::To2DLabel(child);
				alignedLabel->clearLabel();
			}
			else // probably sphere
			{
				updateSphereMarks(child, true);
			}
		}
		// clear reference markers
		for (unsigned i = 0; i < m_refLabels.getChildrenNumber(); ++i)
		{
			ccHObject* child = m_refLabels.getChild(i);
			if (child && child->isKindOf(CV_TYPES::LABEL_2D))
			{
				cc2DLabel* refLabel = ccHObjectCaster::To2DLabel(child);
				refLabel->clearLabel();
			}
			else // probably sphere
			{
				updateSphereMarks(child, true);
			}
		}
	}

	m_alignedPoints.removeAllChildren();
	m_alignedLabels.removeAllChildren();
	m_alignedPoints.resize(0);
	m_alignedPoints.setGlobalShift(0, 0, 0);
	m_alignedPoints.setGlobalScale(1.0);
	m_alignedEntities.clear();
	m_refPoints.removeAllChildren();
	m_refLabels.removeAllChildren();
	m_refPoints.resize(0);
	m_refPoints.setGlobalShift(0, 0, 0);
	m_refPoints.setGlobalScale(1.0);
	m_referenceEntities.clear();
}

bool ccPointPairRegistrationDlg::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}

	m_alignedEntities.restoreAll();
	m_referenceEntities.restoreAll();

	return true;
}

bool ccPointPairRegistrationDlg::start()
{
	assert(!m_alignedEntities.empty());
	if (ecvDisplayTools::GetCurrentScreen())
	{
		if (!m_pickingHub->addListener(this, true))
		{
			CVLog::Error("Picking mechanism is already in use! Close the other tool first, and then restart this one.");
			return false;
		}

		connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::labelmove2D,
			this, &ccPointPairRegistrationDlg::label2DMove);
	}
	return ccOverlayDialog::start();
}

void ccPointPairRegistrationDlg::stop(bool state)
{
	updateAlignInfo();

	disconnect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::labelmove2D,
		this, &ccPointPairRegistrationDlg::label2DMove);
	m_pickingHub->removeListener(this);

	// clear the area
	ecvDisplayTools::DisplayNewMessage(QString(), ecvDisplayTools::UPPER_CENTER_MESSAGE);
	ecvDisplayTools::DisplayNewMessage(QString(), ecvDisplayTools::LOWER_LEFT_MESSAGE);
	ccOverlayDialog::stop(state);
}

static void SetEnabled_recursive(ccHObject* ent)
{
	assert(ent);
	ent->setEnabled(true);
	if (ent->getParent())
		SetEnabled_recursive(ent->getParent());
}

void ccPointPairRegistrationDlg::label2DMove(int x, int y, int dx, int dy)
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		const int retinaScale = ecvDisplayTools::GetDevicePixelRatio();
		for (unsigned i = 0; i < m_alignedLabels.getChildrenNumber(); ++i)
		{
			ccHObject* child = m_alignedLabels.getChild(i);
			if (child && child->isKindOf(CV_TYPES::LABEL_2D))
			{
				cc2DLabel* alignLabel = ccHObjectCaster::To2DLabel(child);
				if (alignLabel)
				{
					if (abs(dx) > 0 || abs(dy) > 0)
					{
						alignLabel->move2D(
							x * retinaScale, y * retinaScale,
							dx * retinaScale, dy * retinaScale,
							ecvDisplayTools::GlWidth(),
							ecvDisplayTools::GlHeight());
					}

					CC_DRAW_CONTEXT context;
					ecvDisplayTools::GetContext(context);
					alignLabel->update2DLabelView(context);
				}
			}
			else // probably sphere
			{
				if (child)
				{
					child->updateNameIn3DRecursive();
				}
			}
		}
		for (unsigned i = 0; i < m_refLabels.getChildrenNumber(); ++i)
		{
			ccHObject* child = m_refLabels.getChild(i);
			if (child && child->isKindOf(CV_TYPES::LABEL_2D))
			{
				cc2DLabel* refLabel = ccHObjectCaster::To2DLabel(child);
				if (refLabel)
				{
					if (abs(dx) > 0 || abs(dy) > 0)
					{
						refLabel->move2D(
							x * retinaScale, y * retinaScale,
							dx * retinaScale, dy * retinaScale,
							ecvDisplayTools::GlWidth(),
							ecvDisplayTools::GlHeight());
					}

					CC_DRAW_CONTEXT context;
					ecvDisplayTools::GetContext(context);
					refLabel->update2DLabelView(context);
				}
			}
			else // probably sphere
			{
				if (child)
				{
					child->updateNameIn3DRecursive();
				}
			}
		}
	}
}

bool ccPointPairRegistrationDlg::init(QWidget* win,
	const ccHObject::Container& alignedEntities,
	const ccHObject::Container* referenceEntities/*=nullptr*/)
{
	if (!win)
	{
		assert(false);
		return false;
	}

	clear();

	if (alignedEntities.empty())
	{
		CVLog::Error("[PointPairRegistration] Need at least one aligned entity!");
		return false;
	}

	m_alignedEntities.fill(alignedEntities);
	if (referenceEntities)
	{
		m_referenceEntities.fill(*referenceEntities);
	}

	//create dedicated 3D view
	if (!m_associatedWin)
	{
		linkWith(win);
		assert(m_associatedWin);
	}

	//add aligned entity to display
	ecvViewportParameters originViewportParams;
	bool hasOriginViewportParams = false;
	for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
	{
		ccHObject* aligned = it.key();
		if (ecvDisplayTools::GetCurrentScreen())
		{
			hasOriginViewportParams = true;
			originViewportParams = ecvDisplayTools::GetViewportParameters();
		}
		//DGM: it's already in the global DB!
		//m_associatedWin->addToOwnDB(aligned);
		aligned->setVisible(true);
		SetEnabled_recursive(aligned);
		//SetVisible_recursive(aligned);
	}

	//add reference entity (if any) to display
	for (auto it = m_referenceEntities.begin(); it != m_referenceEntities.end(); ++it)
	{
		ccHObject* reference = it.key();
		if (!hasOriginViewportParams && ecvDisplayTools::GetCurrentScreen())
		{
			hasOriginViewportParams = true;
			originViewportParams = ecvDisplayTools::GetViewportParameters();
		}
		//DGM: it's already in the global DB!
		//m_associatedWin->addToOwnDB(reference);
		reference->setVisible(true);
		SetEnabled_recursive(reference);
		//SetVisible_recursive(reference);
	}

	showReferenceCheckBox->setChecked(!m_referenceEntities.empty());
	showReferenceCheckBox->setEnabled(!m_referenceEntities.empty());
	showAlignedCheckBox->setChecked(true);

	ecvDisplayTools::GetCurrentScreen()->showMaximized();
	resetTitle();
	ecvDisplayTools::DisplayNewMessage(QString(), ecvDisplayTools::LOWER_LEFT_MESSAGE);
	ecvDisplayTools::DisplayNewMessage("(you can add points 'manually' if necessary)",
		ecvDisplayTools::LOWER_LEFT_MESSAGE, true, 3600);
	ecvDisplayTools::DisplayNewMessage(
		QString("Pick equivalent points on both clouds (at least %1 pairs - mind the order)").arg(MIN_PAIRS_COUNT),
		ecvDisplayTools::LOWER_LEFT_MESSAGE, true, 3600);

	ecvDisplayTools::SetRedrawRecursive(false);
	if (hasOriginViewportParams)
	{
		ecvDisplayTools::SetViewportParameters(originViewportParams);
		ecvDisplayTools::RedrawDisplay(true);
	}
	else
	{
		ecvDisplayTools::ZoomGlobal();
	}

	onPointCountChanged();
	return true;
}

static QString s_aligned_tooltip = QObject::tr("Whether the point is expressed in the entity original coordinate system (before being shifted by CV) or not");
static double  s_last_ax = 0;
static double  s_last_ay = 0;
static double  s_last_az = 0;
static bool    s_lastAlignePointIsGlobal = true;
void ccPointPairRegistrationDlg::addManualAlignedPoint()
{
	ccAskThreeDoubleValuesDlg ptsDlg("x", "y", "z", -1.0e12, 1.0e12, s_last_ax, s_last_ay, s_last_az, 8, "Add aligned point", this);

	//if the aligned entity is shifted, the user has the choice to input virtual point either
	//in the original coordinate system or the shifted one
	if (m_alignedEntities.isShifted)
		ptsDlg.showCheckbox("Not shifted", s_lastAlignePointIsGlobal, s_aligned_tooltip);

	if (!ptsDlg.exec())
		return;

	//save values for current session
	s_last_ax = ptsDlg.doubleSpinBox1->value();
	s_last_ay = ptsDlg.doubleSpinBox2->value();
	s_last_az = ptsDlg.doubleSpinBox3->value();
	bool shifted = true;
	if (m_alignedEntities.isShifted)
	{
		s_lastAlignePointIsGlobal = ptsDlg.getCheckboxState();
		shifted = !s_lastAlignePointIsGlobal;
	}

	CCVector3d P(s_last_ax,s_last_ay,s_last_az);

	addAlignedPoint(P, nullptr, shifted);
}

static double s_last_rx = 0;
static double s_last_ry = 0;
static double s_last_rz = 0;
static bool s_lastRefPointisGlobal = true;
void ccPointPairRegistrationDlg::addManualRefPoint()
{
	ccAskThreeDoubleValuesDlg ptsDlg("x", "y", "z", -1.0e12, 1.0e12, s_last_rx, s_last_ry, s_last_rz, 8, "Add reference point", this);
	
	//if the reference entity is shifted, the user has the choice to input virtual
	//points either in the original coordinate system or the shifted one
	//(if there's no reference entity, we use a 'global'	one by default)
	if (m_referenceEntities.isShifted)
		ptsDlg.showCheckbox("Not shifted", s_lastRefPointisGlobal, s_aligned_tooltip);

	if (!ptsDlg.exec())
		return;

	//save values for current session
	s_last_rx = ptsDlg.doubleSpinBox1->value();
	s_last_ry = ptsDlg.doubleSpinBox2->value();
	s_last_rz = ptsDlg.doubleSpinBox3->value();
	bool shifted = (!m_referenceEntities.empty());
	if (m_referenceEntities.isShifted)
	{
		s_lastRefPointisGlobal = ptsDlg.getCheckboxState();
		shifted = !s_lastRefPointisGlobal;
	}

	CCVector3d P(s_last_rx, s_last_ry, s_last_rz);

	addReferencePoint(P, nullptr, shifted);
}

void ccPointPairRegistrationDlg::pause(bool state)
{
	m_paused = state;
	setDisabled(state);
}

bool ccPointPairRegistrationDlg::convertToSphereCenter(CCVector3d& P, ccHObject* entity, PointCoordinateType& sphereRadius)
{
	sphereRadius = -PC_ONE;
	if (	!entity
		||	!useSphereToolButton->isChecked()
		||	!entity->isKindOf(CV_TYPES::POINT_CLOUD) ) //only works with cloud right now
	{
		//nothing to do
		return true;
	}

	//we'll now try to detect the sphere
	double searchRadius = radiusDoubleSpinBox->value();
	double maxRMSPercentage = maxRmsSpinBox->value() / 100.0;
	ccGenericPointCloud* cloud = static_cast<ccGenericPointCloud*>(entity);
	assert(cloud);

	//crop points inside a box centered on the current point
	ccBBox box;
	box.add(CCVector3::fromArray((P - CCVector3d(1,1,1)*searchRadius).u));
	box.add(CCVector3::fromArray((P + CCVector3d(1,1,1)*searchRadius).u));
	cloudViewer::ReferenceCloud* part = cloud->crop(box,true);

	bool success = false;
	if (part && part->size() > 16)
	{
		PointCoordinateType radius;
		CCVector3 C;
		double rms;
		ecvProgressDialog pDlg(true, this);
		//first roughly search for the sphere
		if (cloudViewer::GeometricalAnalysisTools::DetectSphereRobust(part, 0.5, C, radius, rms, &pDlg, 0.9) == cloudViewer::GeometricalAnalysisTools::NoError)
		{
			if (radius / searchRadius < 0.5 || radius / searchRadius > 2.0)
			{
				CVLog::Warning(QString("[ccPointPairRegistrationDlg] Detected sphere radius (%1) is too far from search radius!").arg(radius));
			}
			else
			{
				//now look again (more precisely)
				{
					delete part;
					box.clear();
					box.add(C - CCVector3(1, 1, 1)*radius*static_cast<PointCoordinateType>(1.05)); //add 5%
					box.add(C + CCVector3(1, 1, 1)*radius*static_cast<PointCoordinateType>(1.05)); //add 5%
					part = cloud->crop(box, true);
					if (part && part->size() > 16)
						cloudViewer::GeometricalAnalysisTools::DetectSphereRobust(part, 0.5, C, radius, rms, &pDlg, 0.99);
				}
				CVLog::Print(QString("[ccPointPairRegistrationDlg] Detected sphere radius = %1 (rms = %2)").arg(radius).arg(rms));
				if (radius / searchRadius < 0.5 || radius / searchRadius > 2.0)
				{
					CVLog::Warning("[ccPointPairRegistrationDlg] Sphere radius is too far from search radius!");
				}
				else if (rms / searchRadius >= maxRMSPercentage)
				{
					CVLog::Warning("[ccPointPairRegistrationDlg] RMS is too high!");
				}
				else
				{
					sphereRadius = radius;
					P = CCVector3d::fromArray(C.u);
					success = true;
				}
			}
		}
		else
		{
			CVLog::Warning("[ccPointPairRegistrationDlg] Failed to fit a sphere around the picked point!");
		}
	}
	else
	{
		//not enough memory? No points inside the 
		CVLog::Warning("[ccPointPairRegistrationDlg] Failed to crop points around the picked point?!");
	}

	if (part)
		delete part;

	return success;
}

void ccPointPairRegistrationDlg::onItemPicked(const PickedItem& pi)
{
	if (!ecvDisplayTools::GetCurrentScreen())
		return;
	
	//no point picking when paused!
	if (m_paused)
		return;

	if (!pi.entity)
		return;

	CCVector3d pin = CCVector3d::fromArray(pi.P3D.u);

	if (m_alignedEntities.contains(pi.entity))
	{
		addAlignedPoint(pin, pi.entity, true); //picked points are always shifted by default
	}
	else if (m_referenceEntities.contains(pi.entity))
	{
		addReferencePoint(pin, pi.entity, true); //picked points are always shifted by default
	}
	else
	{
		//assert(false);
		CVLog::Warning(QString("pick wrong entity : [%1]").arg(pi.entity->getName()));
		return;
	}
}

void ccPointPairRegistrationDlg::onPointCountChanged()
{
	bool canAlign = (m_alignedPoints.size() == m_refPoints.size() && m_refPoints.size() >= MIN_PAIRS_COUNT);
	alignToolButton->setEnabled(canAlign);
	validToolButton->setEnabled(false);

	unstackAlignToolButton->setEnabled(m_alignedPoints.size() != 0);
	unstackRefToolButton->setEnabled(m_refPoints.size() != 0);

	updateAlignInfo();
}

static QToolButton* CreateDeleteButton()
{
	QToolButton* delButton = new QToolButton();
	delButton->setIcon(QIcon(":/Resources/images/smallCancel.png"));
	return delButton;
}

static cc2DLabel* CreateLabel(cc2DLabel* label, ccPointCloud* cloud, unsigned pointIndex, QString pointName)
{
	assert(label);
	label->addPickedPoint(cloud, pointIndex);
	label->setName(pointName);
	label->setVisible(true);
	label->setEnabled(true);
	label->setDisplayedIn2D(false);
	label->displayPointLegend(true);

	return label;
}

static cc2DLabel* CreateLabel(ccPointCloud* cloud, unsigned pointIndex, QString pointName)
{
	return CreateLabel(new cc2DLabel, cloud, pointIndex, pointName);
}

void ccPointPairRegistrationDlg::onDelButtonPushed()
{
	QObject* senderButton = sender();

	//go through all the buttons and find which one has been pushed!
	bool alignedPoint = true;
	int pointIndex = -1;
	//test 'aligned' buttons first
	{
		for (int i = 0; i < alignedPointsTableWidget->rowCount(); ++i)
		{
			if (alignedPointsTableWidget->cellWidget(i, DEL_BUTTON_COL_INDEX) == senderButton)
			{
				pointIndex = i;
				break;
			}
		}
	}

	if (pointIndex < 0)
	{
		//test reference points if necessary
		alignedPoint = false;
		for (int i = 0; i < refPointsTableWidget->rowCount(); ++i)
		{
			if (refPointsTableWidget->cellWidget(i, DEL_BUTTON_COL_INDEX) == senderButton)
			{
				pointIndex = i;
				break;
			}
		}
	}

	if (pointIndex < 0)
	{
		assert(false);
		return;
	}

	if (alignedPoint)
		removeAlignedPoint(pointIndex);
	else
		removeRefPoint(pointIndex);
}

void ccPointPairRegistrationDlg::addPointToTable(QTableWidget* tableWidget, int rowIndex, const CCVector3d& P, QString pointName)
{
	assert(tableWidget);
	if (!tableWidget)
		return;

	//add corresponding row in table
	tableWidget->setRowCount(std::max<int>(rowIndex + 1, tableWidget->rowCount()));
	tableWidget->setVerticalHeaderItem(rowIndex, new QTableWidgetItem(pointName));

	//add point coordinates
	for (int d = 0; d < 3; ++d)
	{
		QTableWidgetItem* item = new QTableWidgetItem();
		item->setData(Qt::EditRole, QString::number(P.u[d], 'f', 6));
		tableWidget->setItem(rowIndex, XYZ_COL_INDEX + d, item);
	}

	//add 'remove' button
	{
		if (rowIndex == 0)
			tableWidget->setColumnWidth(DEL_BUTTON_COL_INDEX, 20);
		//QTableWidgetItem* item = new QTableWidgetItem();
		//tableWidget->setItem(rowIndex, DEL_BUTTON_COL_INDEX, item);
		QToolButton* delButton = CreateDeleteButton();
		connect(delButton, &QToolButton::clicked, this, &ccPointPairRegistrationDlg::onDelButtonPushed);
		tableWidget->setCellWidget(rowIndex, DEL_BUTTON_COL_INDEX, delButton);
	}
}

bool ccPointPairRegistrationDlg::addReferencePoint(CCVector3d& Pin, ccHObject* entity/*=0*/, bool shifted/*=true*/)
{
	assert(entity == nullptr || m_referenceEntities.contains(entity));

	ccGenericPointCloud* cloud = entity ? 
		ccHObjectCaster::ToGenericPointCloud(entity) : nullptr;

	//first point?
	if (m_refPoints.size() == 0)
	{
		if (entity) //picked point
		{
			//simply copy the cloud global shift/scale
			if (cloud)
			{
				m_refPoints.setGlobalScale(cloud->getGlobalScale());
				m_refPoints.setGlobalShift(cloud->getGlobalShift());
			}
		}
		else //virtual point
		{
			m_refPoints.setGlobalScale(1.0);
			m_refPoints.setGlobalShift(0, 0, 0);

			if (!shifted)
			{
				//test that the input point has not too big coordinates
				bool shiftEnabled = false;
				CCVector3d Pshift(0, 0, 0);
				double scale = 1.0;
				//we use the aligned shift by default (if any)
				ccGenericPointCloud* alignedCloud = ccHObjectCaster::ToGenericPointCloud(entity);
				if (alignedCloud && alignedCloud->isShifted())
				{
					Pshift = alignedCloud->getGlobalShift();
					scale = alignedCloud->getGlobalScale();
					shiftEnabled = true;
				}
				if (ecvGlobalShiftManager::Handle(Pin, 0, ecvGlobalShiftManager::DIALOG_IF_NECESSARY, shiftEnabled, Pshift, nullptr, &scale))
				{
					m_refPoints.setGlobalShift(Pshift);
					m_refPoints.setGlobalScale(scale);
				}
			}
		}
	}

	PointCoordinateType sphereRadius = -PC_ONE;
	if (!convertToSphereCenter(Pin, entity, sphereRadius))
		return false;

	//transform the input point in the 'global world' by default
	if (shifted && cloud)
	{
		Pin = cloud->toGlobal3d<double>(Pin);
	}

	//check that we don't duplicate points
	for (unsigned i = 0; i < m_refPoints.size(); ++i)
	{
		//express the 'Pi' point in the current global coordinate system
		CCVector3d Pi = m_refPoints.toGlobal3d<PointCoordinateType>(*m_refPoints.getPoint(i));
        if (cloudViewer::LessThanEpsilon((Pi - Pin).norm()))
		{
			CVLog::Error("Point already picked or too close to an already selected one!");
			return false;
		}
	}

	//add point to the 'reference' set
	unsigned newPointIndex = m_refPoints.size();
	if (newPointIndex == m_refPoints.capacity() && !m_refPoints.reserve(newPointIndex + 1))
	{
		CVLog::Error("Not enough memory?!");
		return false;
	}

	//shift point to the local coordinate system before pushing it
	CCVector3 P = m_refPoints.toLocal3pc<double>(Pin);
	m_refPoints.addPoint(P);

	QString pointName = QString("R%1").arg(newPointIndex);

	//add corresponding row in table
	addPointToTable(refPointsTableWidget, newPointIndex, Pin, pointName);

	//eventually add a label (or a sphere)
	if (sphereRadius <= 0)
	{
		cc2DLabel* label = CreateLabel(&m_refPoints, newPointIndex, pointName);
		m_refLabels.addChild(label);
		label->updateLabel();
	}
	else
	{
		ccGLMatrix trans;
		trans.setTranslation(P);
		ccSphere* sphere = new ccSphere(sphereRadius, &trans, pointName);
		sphere->showNameIn3D(true);
		sphere->setTempColor(ecvColor::yellow, true);
		m_refLabels.addChild(sphere);
		updateSphereMarks(sphere, false);
	}

	onPointCountChanged();

	return true;
}

bool ccPointPairRegistrationDlg::addAlignedPoint(CCVector3d& Pin, ccHObject* entity/*=0*/, bool shifted/*=0*/)
{
	//if the input point is not shifted, we shift it to the aligned coordinate system
	assert(entity == nullptr || m_alignedEntities.contains(entity));

	//first point?
	if (m_alignedPoints.size() == 0)
	{
		//simply copy the cloud global shift/scale
		ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
		if (cloud)
		{
			m_alignedPoints.setGlobalScale(cloud->getGlobalScale());
			m_alignedPoints.setGlobalShift(cloud->getGlobalShift());
		}
	}

	PointCoordinateType sphereRadius = -PC_ONE;
	if (!convertToSphereCenter(Pin, entity, sphereRadius))
		return false;

	//transform the input point in the 'global world' by default
	if (shifted)
		Pin = m_alignedPoints.toGlobal3d<double>(Pin);

	//check that we don't duplicate points
	for (unsigned i = 0; i < m_alignedPoints.size(); ++i)
	{
		CCVector3d Pi = m_alignedPoints.toGlobal3d<PointCoordinateType>(*m_alignedPoints.getPoint(i));
        if (cloudViewer::LessThanEpsilon((Pi-Pin).norm()))
		{
			CVLog::Error("Point already picked or too close to an already selected one!");
			return false;
		}
	}

	unsigned newPointIndex = m_alignedPoints.size();
	if (newPointIndex == m_alignedPoints.capacity() && 
		!m_alignedPoints.reserve(newPointIndex+1))
	{
		CVLog::Error("Not enough memory?!");
		return false;
	}

	//shift point to the local coordinate system before pushing it
	CCVector3 P = m_alignedPoints.toLocal3pc<double>(Pin);
	m_alignedPoints.addPoint(P);
	
	QString pointName = QString("A%1").arg(newPointIndex);
	
	//add corresponding row in table
	addPointToTable(alignedPointsTableWidget, newPointIndex, Pin, pointName);

	//eventually add a label (or a sphere)
	if (sphereRadius <= 0)
	{
		cc2DLabel* label = CreateLabel(&m_alignedPoints, newPointIndex, pointName);
		m_alignedLabels.addChild(label);
		label->updateLabel();
	}
	else
	{
		ccGLMatrix trans;
		trans.setTranslation(P);
		ccSphere* sphere = new ccSphere(sphereRadius, &trans, pointName);
		sphere->showNameIn3D(true);
		sphere->setTempColor(ecvColor::red, true);
		m_alignedLabels.addChild(sphere);
		updateSphereMarks(sphere, false);
	}

	onPointCountChanged();

	return true;
}

void ccPointPairRegistrationDlg::unstackAligned()
{
	unsigned pointCount = m_alignedPoints.size();
	if (pointCount == 0) //nothing to do
		return;

	assert(alignedPointsTableWidget->rowCount() > 0);
	alignedPointsTableWidget->removeRow(alignedPointsTableWidget->rowCount()-1);

	assert(m_alignedLabels.getChildrenNumber() == pointCount);
	pointCount--;

	// remove label
	updateAlignedMarkers(pointCount);

	//remove point
	m_alignedPoints.resize(pointCount);

	onPointCountChanged();
}

void ccPointPairRegistrationDlg::unstackRef()
{
	unsigned pointCount = m_refPoints.size();
	if (pointCount == 0)
		return;

	assert(refPointsTableWidget->rowCount() > 0);
	refPointsTableWidget->removeRow(refPointsTableWidget->rowCount() - 1);

	//remove label
	assert(m_refLabels.getChildrenNumber() == pointCount);
	pointCount--;

	// remove label
	updateRefMarkers(pointCount);

	// remove point
	m_refPoints.resize(pointCount);

	if (pointCount == 0)
	{
		//reset global shift (if any)
		m_refPoints.setGlobalShift(0, 0, 0);
		m_refPoints.setGlobalScale(1.0);
	}

	onPointCountChanged();
}

void ccPointPairRegistrationDlg::removeAlignedPoint(int index, bool autoRemoveDualPoint/*=false*/)
{
	if (index >= static_cast<int>(m_alignedPoints.size()))
	{
		CVLog::Error("[ccPointPairRegistrationDlg::removeAlignedPoint] Invalid index!");
		assert(false);
		return;
	}

	int pointCount = static_cast<int>(m_alignedPoints.size());

	//remove the label (or sphere)
	updateAlignedMarkers(index);

	//remove array row
	alignedPointsTableWidget->removeRow(index);

	//shift points & rename labels
	for (int i = index + 1; i < pointCount; ++i)
	{
		*const_cast<CCVector3*>(m_alignedPoints.getPoint(i - 1)) = *m_alignedPoints.getPoint(i);

		//new name
		QString pointName = QString("A%1").arg(i - 1);
		//update the label (if any)
		ccHObject* child = m_alignedLabels.getChild(i - 1);
		if (child)
		{
			if (child->isKindOf(CV_TYPES::LABEL_2D))
			{
				cc2DLabel* label = static_cast<cc2DLabel*>(child);
				label->clear();
				CreateLabel(label, &m_alignedPoints, static_cast<unsigned>(i - 1), pointName);
				label->updateLabel();
			}
			else //probably a sphere
			{
				child->setName(pointName);
				updateSphereMarks(child, false);
			}
		}
		//update array
		alignedPointsTableWidget->setVerticalHeaderItem(i - 1, new QTableWidgetItem(pointName));
	}
	m_alignedPoints.invalidateBoundingBox();

	pointCount--;
	assert(pointCount >= 0);
	m_alignedPoints.resize(static_cast<unsigned>(pointCount));

	if (m_alignedPoints.size() == 0)
	{
		//reset global shift (if any)
		m_alignedPoints.setGlobalShift(0, 0, 0);
		m_alignedPoints.setGlobalScale(1.0);
	}

	onPointCountChanged();

	//auto-remove the other point?
	if (autoRemoveDualPoint
		&&	index < static_cast<int>(m_refPoints.size())
		&& QMessageBox::question(0, "Remove dual point", 
			"Remove the equivalent reference point as well?", 
			QMessageBox::Yes, QMessageBox::No) == QMessageBox::Yes)
	{
		removeRefPoint(index, false);
	}
}

void ccPointPairRegistrationDlg::removeRefPoint(int index, bool autoRemoveDualPoint/*=false*/)
{
	if (index >= static_cast<int>(m_refPoints.size()))
	{
		CVLog::Error("[ccPointPairRegistrationDlg::removeRefPoint] Invalid index!");
		assert(false);
		return;
	}

	int pointCount = static_cast<int>(m_refPoints.size());

	//remove the label (or sphere)
	updateRefMarkers(index);
	//remove array row
	refPointsTableWidget->removeRow(index);

	//shift points & rename labels
	for (int i = index + 1; i < pointCount; ++i)
	{
		*const_cast<CCVector3*>(m_refPoints.getPoint(i - 1)) = *m_refPoints.getPoint(i);

		//new name
		QString pointName = QString("R%1").arg(i - 1);
		//update the label (if any)
		ccHObject* child = m_refLabels.getChild(i - 1);
		if (child)
		{
			if (child->isKindOf(CV_TYPES::LABEL_2D))
			{
				cc2DLabel* label = static_cast<cc2DLabel*>(child);
				label->clear();
				CreateLabel(label, &m_refPoints, 
					static_cast<unsigned>(i - 1), pointName);
				label->updateLabel();
			}
			else //probably a sphere
			{
				child->setName(pointName);
				updateSphereMarks(child, false);
			}
		}

		//update array
		refPointsTableWidget->setVerticalHeaderItem(i - 1, new QTableWidgetItem(pointName));
	}
	m_refPoints.invalidateBoundingBox();

	pointCount--;
	assert(pointCount >= 0);
	m_refPoints.resize(static_cast<unsigned>(pointCount));

	if (m_refPoints.size() == 0)
	{
		//reset global shift (if any)
		m_refPoints.setGlobalShift(0, 0, 0);
		m_refPoints.setGlobalScale(1.0);
	}

	onPointCountChanged();

	//auto-remove the other point?
	if (	autoRemoveDualPoint
		&&	index < static_cast<int>(m_alignedPoints.size())
		&&	QMessageBox::question(0, "Remove dual point", 
			"Remove the equivalent aligned point as well?",
			QMessageBox::Yes, QMessageBox::No) == QMessageBox::Yes)
	{
		removeAlignedPoint(index,false);
	}
}

void ccPointPairRegistrationDlg::updateSphereMarks(ccHObject* obj, bool remove)
{
	if (obj->isA(CV_TYPES::SPHERE))
	{
		bool forceRedraw = !remove;
		CC_DRAW_CONTEXT context;
		context.forceRedraw = forceRedraw;
		if (remove)
		{
			context.removeEntityType = ENTITY_TYPE::ECV_MESH;
            context.removeViewID = obj->getViewId();
			ecvDisplayTools::RemoveEntities(context);
			obj->showNameIn3D(false);
		}
		else
		{
			obj->showNameIn3D(true);
		}

		context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
		obj->draw(context);
		context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
		obj->draw(context);
	}
}

void ccPointPairRegistrationDlg::updateAlignedMarkers(int index)
{
	if (index < 0) { return; }

	if (ecvDisplayTools::GetCurrentScreen())
	{
		ccHObject* child = m_alignedLabels.getChild(index);
		if (child && child->isKindOf(CV_TYPES::LABEL_2D))
		{
			cc2DLabel* label = ccHObjectCaster::To2DLabel(child);
			label->clearLabel();
		}
		else //  probably sphere
		{
			updateSphereMarks(child, true);
		}
		m_alignedLabels.removeChild(index);
	}
}

void ccPointPairRegistrationDlg::updateRefMarkers(int index)
{
	if (index < 0) { return; }

	if (ecvDisplayTools::GetCurrentScreen())
	{
		ccHObject* child = m_refLabels.getChild(index);
		if (child && child->isKindOf(CV_TYPES::LABEL_2D))
		{
			cc2DLabel* label = ccHObjectCaster::To2DLabel(child);
			label->clearLabel();
		}
		else
		{
			updateSphereMarks(child, true);
		}
		m_refLabels.removeChild(index);
	}
}

void ccPointPairRegistrationDlg::showAlignedEntities(bool state)
{
	if (m_alignedEntities.empty())
		return;

	for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
		it.key()->setVisible(state);
	m_alignedPoints.setEnabled(state);

	for (unsigned i = 0; i < m_alignedLabels.getChildrenNumber(); ++i)
	{
		ccHObject* child = m_alignedLabels.getChild(i);
		if (child && child->isKindOf(CV_TYPES::LABEL_2D))
		{
			cc2DLabel* alignLabel = ccHObjectCaster::To2DLabel(child);
			alignLabel->setEnabled(state);
			alignLabel->updateLabel();
		}
		else
		{
			updateSphereMarks(child, !state);
		}
	}

	ecvDisplayTools::SetRedrawRecursive(false);

	if (autoZoomCheckBox->isChecked())
	{
		ecvDisplayTools::ZoomGlobal();
	}
	else
	{
		ecvDisplayTools::RedrawDisplay();
	}
}

void ccPointPairRegistrationDlg::showReferenceEntities(bool state)
{
	if (m_referenceEntities.empty())
		return;

	for (auto it = m_referenceEntities.begin(); it != m_referenceEntities.end(); ++it)
		it.key()->setVisible(state);
	m_refPoints.setEnabled(state);

	for (unsigned i = 0; i < m_refLabels.getChildrenNumber(); ++i)
	{
		ccHObject* child = m_refLabels.getChild(i);
		if (child && child->isKindOf(CV_TYPES::LABEL_2D))
		{
			cc2DLabel* refLabel = ccHObjectCaster::To2DLabel(child);
			refLabel->setEnabled(state);
			refLabel->updateLabel();
		}
		else
		{
			updateSphereMarks(child, !state);
		}
	}

	ecvDisplayTools::SetRedrawRecursive(false);

	if (autoZoomCheckBox->isChecked())
	{
		ecvDisplayTools::ZoomGlobal();
	}
	else
	{
		ecvDisplayTools::RedrawDisplay();
	}
}

bool ccPointPairRegistrationDlg::callHornRegistration(cloudViewer::PointProjectionTools::Transformation& trans, double& rms, bool autoUpdateTab)
{
	if (m_alignedEntities.empty())
	{
		assert(false);
		return false;
	}

	if (m_alignedPoints.size() != m_refPoints.size() || m_refPoints.size() < MIN_PAIRS_COUNT)
	{
		assert(false);
		CVLog::Error(QString("Need at least %1 points for each entity (and the same number of points in both subsets)!").arg(MIN_PAIRS_COUNT));
		return false;
	}

	//fixed scale?
	bool adjustScale = adjustScaleCheckBox->isChecked();

	//call Horn registration method
	if (!cloudViewer::HornRegistrationTools::FindAbsoluteOrientation(&m_alignedPoints, &m_refPoints, trans, !adjustScale))
	{
		CVLog::Error("Registration failed! (points are aligned?)");
		return false;
	}

	//apply constraints (if any)
	{
		int filters = 0;
		switch (rotComboBox->currentIndex())
		{
		case 1:
			filters |= cloudViewer::RegistrationTools::SKIP_RYZ;
			break;
		case 2:
			filters |= cloudViewer::RegistrationTools::SKIP_RXZ;
			break;
		case 3:
			filters |= cloudViewer::RegistrationTools::SKIP_RXY;
			break;
		default:
			//nothing to do
			break;
		}

		if (!TxCheckBox->isChecked())
			filters |= cloudViewer::RegistrationTools::SKIP_TX;
		if (!TyCheckBox->isChecked())
			filters |= cloudViewer::RegistrationTools::SKIP_TY;
		if (!TzCheckBox->isChecked())
			filters |= cloudViewer::RegistrationTools::SKIP_TZ;

		if (filters != 0)
		{
			cloudViewer::RegistrationTools::FilterTransformation(trans, filters, trans);
		}
	}

	//compute RMS
	rms = cloudViewer::HornRegistrationTools::ComputeRMS(&m_alignedPoints, &m_refPoints, trans);

	if (autoUpdateTab)
	{
		//display resulting RMS in colums
		if (rms >= 0)
		{
			assert(m_alignedPoints.size() == m_refPoints.size());
			for (unsigned i = 0; i < m_alignedPoints.size(); ++i)
			{
				const CCVector3* Ri = m_refPoints.getPoint(i);
				const CCVector3* Li = m_alignedPoints.getPoint(i);
				CCVector3d Lit = trans.apply(*Li);
				double dist = (Ri->toDouble() - Lit).norm();

				QTableWidgetItem* itemA = new QTableWidgetItem();
				itemA->setData(Qt::EditRole, dist);
				alignedPointsTableWidget->setItem(i, RMS_COL_INDEX, itemA);
				QTableWidgetItem* itemR = new QTableWidgetItem();
				itemR->setData(Qt::EditRole, dist);
				refPointsTableWidget->setItem(i, RMS_COL_INDEX, itemR);
			}
		}
		else
		{
			//clear RMS columns
			clearRMSColumns();
		}
	}

	return true;
}

void ccPointPairRegistrationDlg::clearRMSColumns()
{
	for (int i=0; alignedPointsTableWidget->rowCount(); ++i)
		alignedPointsTableWidget->setItem(i,RMS_COL_INDEX,new QTableWidgetItem());
	for (int i=0; refPointsTableWidget->rowCount(); ++i)
		refPointsTableWidget->setItem(i,RMS_COL_INDEX,new QTableWidgetItem());
}

void ccPointPairRegistrationDlg::resetTitle()
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::DisplayNewMessage(QString(), ecvDisplayTools::UPPER_CENTER_MESSAGE, false);
		ecvDisplayTools::DisplayNewMessage("[Point-pair registration]", ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 3600);
	}
}

void ccPointPairRegistrationDlg::updateAlignInfo()
{
	//reset title
	resetTitle();

	cloudViewer::PointProjectionTools::Transformation trans;
	double rms;

	if (	m_alignedPoints.size() == m_refPoints.size()
		&&	m_refPoints.size() >= MIN_PAIRS_COUNT
		&&	callHornRegistration(trans, rms, true))
	{
		QString rmsString = QString("Achievable RMS: %1").arg(rms);
		ecvDisplayTools::DisplayNewMessage(rmsString, ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 60 * 60);
		resetToolButton->setEnabled(true);
		validToolButton->setEnabled(true);
	}
	else
	{
		resetToolButton->setEnabled(false);
		validToolButton->setEnabled(false);
	}

	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::SetRedrawRecursive(false);
		ecvDisplayTools::RedrawDisplay(true);
	}
}

void ccPointPairRegistrationDlg::align()
{
	cloudViewer::PointProjectionTools::Transformation trans;
	double rms;

	//reset title
	resetTitle();
	ecvDisplayTools::SetRedrawRecursive(false);
	ecvDisplayTools::RedrawDisplay(true, false);

	if (callHornRegistration(trans, rms, true))
	{
		if (rms >= 0)
		{
			QString rmsString = QString("Current RMS: %1").arg(rms);
			CVLog::Print(QString("[PointPairRegistration] ") + rmsString);
			ecvDisplayTools::DisplayNewMessage(rmsString, ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 60 * 60);
		}
		else
		{
			CVLog::Warning("[PointPairRegistration] Internal error (negative RMS?!)");
			return;
		}

		//apply (scaled) transformation (if not fixed)
		bool adjustScale = adjustScaleCheckBox->isChecked();
		if (adjustScale)
		{
			if (trans.R.isValid())
				trans.R.scale(trans.s);

			QString scaleString = QString("Scale: %1").arg(trans.s);
			CVLog::Print(QString("[PointPairRegistration] ")+scaleString);
		}
		else
		{
			CVLog::Print(QString("[PointPairRegistration] Scale: fixed (1.0)"));
		}

		ccGLMatrix transMat = FromCCLibMatrix<double, float>(trans.R, trans.T);
		//...virtually
		m_transMatHistory = transMat;
		transformAlignedEntity(transMat, true);

		ecvDisplayTools::SetRedrawRecursive(false);
		for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
			it.key()->setRedrawFlagRecursive(true);

		// update aligned sphere markers
		for (unsigned i = 0; i < m_alignedLabels.getChildrenNumber(); ++i)
		{
			ccHObject* child = m_alignedLabels.getChild(i);
			if (child && !child->isKindOf(CV_TYPES::LABEL_2D))
			{
				child->setRedrawFlagRecursive(true);
			}
		}

		//force clouds visibility
		{
			//we don't want the window zoom to change or the window to be be redrawn
			if (!showAlignedCheckBox->isChecked())
				showAlignedCheckBox->setChecked(true);
			if (!showReferenceCheckBox->isChecked())
				showReferenceCheckBox->setChecked(true);
			//restore window ref
		}

		if (autoZoomCheckBox->isChecked())
		{
			zoomGlobalOnRegistrationEntities();
		}
		
		if (ecvDisplayTools::GetCurrentScreen())
		{
			ecvDisplayTools::RedrawDisplay();
		}

		updateAllMarkers(1.0f / static_cast<float>(trans.s));

		resetToolButton->setEnabled(true);
		alignToolButton->setEnabled(false);
		validToolButton->setEnabled(true);
	}
}

void ccPointPairRegistrationDlg::updateAllMarkers(float markerSize)
{
	//DGM: we have to 'counter-scale' the markers (otherwise they might appear very big or very small!)
	for (unsigned i = 0; i < m_alignedLabels.getChildrenNumber(); ++i)
	{
		ccHObject* child = m_alignedLabels.getChild(i);
		if (child->isA(CV_TYPES::LABEL_2D))
		{
			static_cast<cc2DLabel*>(child)->setRelativeMarkerScale(markerSize);
			child->setEnabled(true);
			static_cast<cc2DLabel*>(child)->updateLabel();
		}
		else //  probably sphere
		{
			child->updateNameIn3DRecursive();
		}
	}

	// DGM: we have to reset the reference markers scale
	for (unsigned i = 0; i < m_refLabels.getChildrenNumber(); ++i)
	{
		ccHObject* child = m_refLabels.getChild(i);
		if (child->isA(CV_TYPES::LABEL_2D))
		{
			static_cast<cc2DLabel*>(child)->setRelativeMarkerScale(markerSize);
			child->setEnabled(true);
			static_cast<cc2DLabel*>(child)->updateLabel();
			//child->setEnabled(true);
			//CC_DRAW_CONTEXT context;
			//ecvDisplayTools::GetContext(context);
			//static_cast<cc2DLabel*>(child)->update2DLabelView(context);
		}
		else //  probably sphere
		{
			child->updateNameIn3DRecursive();
		}
	}
}

void ccPointPairRegistrationDlg::transformAlignedEntity(const ccGLMatrix &transMat, bool apply/* = true*/)
{
	assert(!m_alignedEntities.empty());
	//we temporarily detach entity, as it may undergo
	//"severe" modifications (octree deletion, etc.) --> see ccHObject::applyGLTransformation
	for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
	{
		ecvMainAppInterface::ccHObjectContext objContext;
		if (m_app)
			objContext = m_app->removeObjectTemporarilyFromDBTree(it.key());
		it.key()->applyGLTransformation_recursive(apply ? &transMat : nullptr);
		if (m_app)
			m_app->putObjectBackIntoDBTree(it.key(), objContext);

		if (!apply)
		{
            ecvDisplayTools::RemoveBB(it.key()->getViewId());
		}
	}
	m_alignedPoints.applyGLTransformation_recursive(apply ? &transMat : nullptr);
}

void ccPointPairRegistrationDlg::reset()
{
	if (m_alignedEntities.empty())
		return;

	transformAlignedEntity(m_transMatHistory.inverse(), true);
	m_transMatHistory.toIdentity();

	// update aligned sphere markers
	for (unsigned i = 0; i < m_alignedLabels.getChildrenNumber(); ++i)
	{
		ccHObject* child = m_alignedLabels.getChild(i);
		if (child && !child->isKindOf(CV_TYPES::LABEL_2D))
		{
			child->setRedrawFlagRecursive(true);
		}
	}

	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::SetRedrawRecursive(false);
		for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
		{
			it.key()->setRedrawFlagRecursive(true);
		}

		if (autoZoomCheckBox->isChecked())
		{
			zoomGlobalOnRegistrationEntities();
		}
		ecvDisplayTools::RedrawDisplay();
	}

	updateAllMarkers(1.0);

	updateAlignInfo();

	alignToolButton->setEnabled(true);
	resetToolButton->setEnabled(false);
}

void ccPointPairRegistrationDlg::zoomGlobalOnRegistrationEntities()
{
	ccHObject tempGroup("TempGroup");

	for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
	{
		tempGroup.addChild(it.key(), ccHObject::DP_NONE);
	}
	for (auto it = m_referenceEntities.begin(); it != m_referenceEntities.end(); ++it)
	{
		tempGroup.addChild(it.key(), ccHObject::DP_NONE);
	}

	ccBBox bbox;
	if (tempGroup.getChildrenNumber() != 0)
	{
		bbox = tempGroup.getDisplayBB_recursive(false);
	}
	if (bbox.isValid())
	{
		ecvDisplayTools::UpdateConstellationCenterAndZoom(&bbox, false);
	}
}

void ccPointPairRegistrationDlg::apply()
{
	cloudViewer::PointProjectionTools::Transformation trans;
	double rms = -1.0;

	// restore current alignedPoints
	const ccGLMatrix& transMat = m_transMatHistory.inverse();
	m_alignedPoints.applyGLTransformation_recursive(&transMat);
	
	if (callHornRegistration(trans, rms, false))
	{
		QStringList summary;
		if (rms >= 0)
		{
			QString rmsString = QString("Final RMS: %1").arg(rms);
			CVLog::Print(QString("[PointPairRegistration] ")+rmsString);
			summary << rmsString;
			summary << "----------------";
		}

		//apply (scaled) transformation (if not fixed)
		bool adjustScale = adjustScaleCheckBox->isChecked();
		if (adjustScale && trans.R.isValid())
		{
			trans.R.scale(trans.s);
		}
		ccGLMatrix transMat = FromCCLibMatrix<double,float>(trans.R,trans.T);
		
		//...for real this time!
		transformAlignedEntity(transMat, false);

		summary << QString("Transformation matrix");
		summary << transMat.toString(3,'\t'); //low precision, just for display
		summary << "----------------";

		CVLog::Print("[PointPairRegistration] Applied transformation matrix:");
		CVLog::Print(transMat.toString(12,' ')); //full precision
		
		if (adjustScale)
		{
			QString scaleString = QString("Scale: %1 (already integrated in above matrix!)").arg(trans.s);
			CVLog::Warning(QString("[PointPairRegistration] ") + scaleString);
			summary << scaleString;
		}
		else
		{
			CVLog::Print(QString("[PointPairRegistration] Scale: fixed (1.0)"));
			summary << "Scale: fixed (1.0)";
		}
		summary << "----------------";

		//pop-up summary
		summary << "Refer to Console (F8) for more details";
		QMessageBox::information(this, "Align info", summary.join("\n"));

		// don't forget global shift
		bool alwaysDropShift = false;
		bool firstQuestion = true;
		for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
		{
			ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(it.key());
			if (cloud)
			{
				if (m_refPoints.isShifted())
				{
					const CCVector3d& Pshift = m_refPoints.getGlobalShift();
					const double& scale = m_refPoints.getGlobalScale();
					cloud->setGlobalShift(Pshift);
					cloud->setGlobalScale(scale);
					CVLog::Warning(tr("[PointPairRegistration] Aligned entity global shift has been updated to match the reference: (%1,%2,%3) [x%4]").
						arg(Pshift.x).arg(Pshift.y).arg(Pshift.z).arg(scale));
				}
				else if (cloud->isShifted()) //we'll ask the user first before dropping the shift information on the aligned cloud
				{
					if (firstQuestion)
					{
						alwaysDropShift = (QMessageBox::question(this, 
							tr("Drop shift information?"),
							tr("Aligned cloud is shifted but reference cloud is not: drop global shift information?"), 
							QMessageBox::Yes, QMessageBox::No) == QMessageBox::Yes);
						firstQuestion = false;
					}

					if (alwaysDropShift)
					{
						cloud->setGlobalShift(0, 0, 0);
						cloud->setGlobalScale(1.0);
						CVLog::Warning(tr("[PointPairRegistration] Cloud %1: global shift has been reset to match the reference!").arg(cloud->getName()));
					}
				}
			}
		}
	}
	else
	{
		CVLog::Warning(QString("[PointPairRegistration] Failed to register entities?!"));
	}

	//save persistent settings
	{
		QSettings settings;
		settings.beginGroup("PointPairAlign");
		settings.setValue("PickSpheres",  useSphereToolButton->isChecked());
		settings.setValue("SphereRadius", radiusDoubleSpinBox->value());
		settings.setValue("MaxRMS", maxRmsSpinBox->value());
		settings.setValue("AdjustScale",  adjustScaleCheckBox->isChecked());
		settings.setValue("AutoUpdateZom",autoZoomCheckBox->isChecked());
		settings.endGroup();
	}
	
	stop(true);
}

void ccPointPairRegistrationDlg::cancel()
{
	for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
		it.key()->enableGLTransformation(false);

	transformAlignedEntity(m_transMatHistory.inverse(), true);
	ecvDisplayTools::SetRedrawRecursive(false);
	for (auto it = m_alignedEntities.begin(); it != m_alignedEntities.end(); ++it)
		it.key()->setRedrawFlagRecursive(true);

	ecvDisplayTools::RedrawDisplay();
	stop(false);
}
