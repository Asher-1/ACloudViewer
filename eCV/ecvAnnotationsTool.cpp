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

#include "ecvAnnotationsTool.h"

// LOCAL
#include "MainWindow.h"
#include "ecvEntityAction.h"
#include "ecvFileUtils.h"
#include "ecvSettingManager.h"
#include "ecvPersistentSettings.h"

// ECV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>
#include <CVConst.h>

// ECV_DB_LIB
#include <ecvPolyline.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvGenericAnnotationTool.h>

// QT
#include <QMessageBox>

ecvAnnotationsTool::ecvAnnotationsTool(QWidget* parent)
	: ccOverlayDialog(parent)
	, Ui::AnnotationsDlg()
	, m_entityContainer("entities")
	, m_editMode(false)
{
	setupUi(this);
	m_disabledCombEvent = false;

	connect(importClassSetsButton,			SIGNAL(clicked()),				this, SLOT(importClassesFromFile()));
	connect(saveButton,						SIGNAL(clicked()),				this, SLOT(saveAnnotations()));
	connect(resetButton,					SIGNAL(clicked()),				this, SLOT(reset()));
	connect(closeButton,					SIGNAL(clicked()),				this, SLOT(closeDialog()));
	connect(exportCloudWithAnnotations,		SIGNAL(clicked()),				this, SLOT(exportAnnotationToSF()));

	connect(editToolButton,					SIGNAL(toggled(bool)),			this, SLOT(toggleEditMode(bool)));
	
	connect(pauseToolButton,				SIGNAL(toggled(bool)),			this, SLOT(toggleInteractors(bool)));
	connect(showBoxToolButton,				SIGNAL(toggled(bool)),			this, SLOT(toggleBox(bool)));
	connect(showOriginToolButton,			SIGNAL(toggled(bool)),			this, SLOT(toggleOrigin(bool)));
	
	connect(minusXShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftXMinus()));
	connect(plusXShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftXPlus()));
	connect(minusYShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftYMinus()));
	connect(plusYShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftYPlus()));
	connect(minusZShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftZMinus()));
	connect(plusZShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftZPlus()));

	viewButtonsFrame->setEnabled(true);
	connect(viewUpToolButton,				SIGNAL(clicked()),				this, SLOT(setTopView()));
	connect(viewDownToolButton,				SIGNAL(clicked()),				this, SLOT(setBottomView()));
	connect(viewFrontToolButton,			SIGNAL(clicked()),				this, SLOT(setFrontView()));
	connect(viewBackToolButton,				SIGNAL(clicked()),				this, SLOT(setBackView()));
	connect(viewLeftToolButton,				SIGNAL(clicked()),				this, SLOT(setLeftView()));
	connect(viewRightToolButton,			SIGNAL(clicked()),				this, SLOT(setRightView()));

	connect(newModeButton,			SIGNAL(clicked()),					this, SLOT(onNewMode()));
	connect(unionModeButton,		SIGNAL(clicked()),					this, SLOT(onUnionMode()));
	connect(trimModeButton,			SIGNAL(clicked()),					this, SLOT(onTrimMode()));
	connect(intersectModeButton,	SIGNAL(clicked()),					this, SLOT(onIntersectMode()));
	connect(labelSelectedButton,	SIGNAL(clicked()),					this, SLOT(onLabelSelected()));
	connect(labelsComboBox,			SIGNAL(currentIndexChanged(int)),	this, SLOT(onLabelChanged(int)));

	// add shortcuts
	addOverridenShortcut(Qt::Key_I);      //'I' key for the "intersect selection mode" button
	addOverridenShortcut(Qt::Key_U);      //'U' key for the "union selection mode" button
	addOverridenShortcut(Qt::Key_N);      //'N' key for the "new selection mode" button
	addOverridenShortcut(Qt::Key_T);      //'T' key for the "trim selection mode" button
	addOverridenShortcut(Qt::Key_R);      //'R' key for the "reset annotations" button
	addOverridenShortcut(Qt::Key_S);      //'S' key for the "save annotation file" button
	addOverridenShortcut(Qt::Key_H);	  //'H' key for the "show or hide annotation mode" button
	addOverridenShortcut(Qt::Key_L);	  //'L' key for the "label selected with current class sets" button
	addOverridenShortcut(Qt::Key_Space);  //space bar for the "pause" button
	connect(this, SIGNAL(shortcutTriggered(int)), this, SLOT(onShortcutTriggered(int)));
}

ecvAnnotationsTool::~ecvAnnotationsTool()
{
	releaseAssociatedEntities();
}

bool ecvAnnotationsTool::setAnnotationsTool(ecvGenericAnnotationTool * annotationTool)
{
	if (annotationTool)
	{
		m_annotationTool = annotationTool;
		connect(m_annotationTool, &ecvGenericAnnotationTool::objectPicked, this, &ecvAnnotationsTool::onItemPicked);
		return true;
	}
	return false;
}

void ecvAnnotationsTool::onShortcutTriggered(int key)
{
	switch (key)
	{
	case Qt::Key_H:
		showBoxToolButton->toggle();
		return;

	case Qt::Key_Space:
		pauseToolButton->toggle();
		return;

	case Qt::Key_L:
		labelSelectedButton->click();
		return;

	case Qt::Key_R:
		resetButton->click();
		return;

	case Qt::Key_S:
		saveButton->click();
		return;

	case Qt::Key_I:
		intersectModeButton->click();
		return;

	case Qt::Key_U:
		unionModeButton->click();
		return;

	case Qt::Key_N:
		newModeButton->click();
		return;
	case Qt::Key_T:
		trimModeButton->click();
		return;

	default:
		//nothing to do
		break;
	}
}

void ecvAnnotationsTool::onNewMode()
{
	if (m_annotationTool)
	{
		m_annotationTool->resetMode();
	}
}

void ecvAnnotationsTool::onUnionMode()
{
	if (m_annotationTool)
	{
		m_annotationTool->unionMode();
	}
}

void ecvAnnotationsTool::onTrimMode()
{
	if (m_annotationTool)
	{
		m_annotationTool->trimMode();
	}
}

void ecvAnnotationsTool::onIntersectMode()
{
	if (m_annotationTool)
	{
		m_annotationTool->intersectMode();
	}
}

void ecvAnnotationsTool::onLabelSelected()
{
	onLabelChanged(0);
}

void ecvAnnotationsTool::onLabelChanged(int index)
{
	int curIndex = labelsComboBox->currentIndex();
	if (curIndex < 0 || m_disabledCombEvent)
	{
		return;
	}

	QColor backColor;
	if (curIndex == 0)
	{
		backColor = QColor(Qt::white);
	}
	else
	{
		ecvColor::Rgb col = ecvColor::LookUpTable::at(static_cast<size_t>(curIndex));
		backColor = QColor(col.r, col.g, col.b);
	}

	QPalette pal = labelsComboBox->palette();
	pal.setColor(QPalette::Base, backColor);
	labelsComboBox->setPalette(pal);

	QString text = labelsComboBox->currentText();
	labelsComboBox->clearFocus();

	if (m_annotationTool)
	{
		if (m_editMode)
		{
			editToolButton->toggle();
			m_annotationTool->selectExistedAnnotation(text.toStdString());
		}
		else
		{
			m_annotationTool->changeAnnotationType(text.toStdString());
		}
	}
}

void ecvAnnotationsTool::onItemPicked(bool isPicked)
{
}

void ecvAnnotationsTool::toggleInteractors(bool state)
{
	if (m_annotationTool)
	{
		m_annotationTool->toggleInteractor();
	}
}

void ecvAnnotationsTool::saveAnnotations()
{
	if (m_annotationTool)
	{
		m_annotationTool->exportAnnotations();
	}
}

void ecvAnnotationsTool::importClassesFromFile()
{
	if (m_annotationTool)
	{
		//default output path (+ filename)
		QString currentPath = ecvSettingManager::getValue(ecvPS::LoadFile(), ecvPS::CurrentPath(), ecvFileUtils::defaultDocPath()).toString();
		QString filters = "*.classes";
		QString selectedFilter = filters;
		QString selectedFilename = QFileDialog::getOpenFileName(
			this,
			tr("import class sets"),
			currentPath,
			filters,
			&selectedFilter);

		if (selectedFilename.isEmpty())
		{
			//process cancelled by the user
			return;
		}

		// we update current file path
		currentPath = QFileInfo(selectedFilename).absolutePath();
		ecvSettingManager::setValue(ecvPS::LoadFile(), ecvPS::CurrentPath(), currentPath);
		if (!m_annotationTool->loadClassesFromFile(CVTools::fromQString(selectedFilename)))
		{
			return;
		}

		std::vector<std::string> labels;
		m_annotationTool->getAnnotationLabels(labels);
		updateLabelsCombox(labels);
	}
}

void ecvAnnotationsTool::toggleBox(bool state)
{
	if (!m_annotationTool)
	{
		CVLog::Warning("[ecvAnnotationsTool::toggleBox] annotations tool has not been initialized!");
		return;
	}

	if (state)
	{
		m_annotationTool->showAnnotation();
	}
	else
	{
		m_annotationTool->hideAnnotation();
	}
	ecvDisplayTools::UpdateScreen();
}

void ecvAnnotationsTool::toggleOrigin(bool dummy)
{
	if (!m_annotationTool)
	{
		CVLog::Warning("[ecvAnnotationsTool::toggleBox] annotations tool has not been initialized!");
		return;
	}

	if (showOriginToolButton->isChecked())
	{
		m_annotationTool->showOrigin();
	}
	else
	{
		m_annotationTool->hideOrigin();
	}
	ecvDisplayTools::UpdateScreen();
}

void ecvAnnotationsTool::releaseAssociatedEntities()
{
	m_entityContainer.removeAllChildren();
}

bool ecvAnnotationsTool::addAssociatedEntity(ccHObject* entity)
{
	if (!entity)
	{
		assert(false);
		return false;
	}

	if (!m_annotationTool)
	{
		CVLog::Error(QString("[ecvAnnotationsTool::addAssociatedEntity] No associated annotation Tool!"));
		return false;
	}

	if (entity->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		m_entityContainer.addChild(entity, ccHObject::DP_NONE);
		classSetsGroupBox->setEnabled(true);
	}

	if (entity->getBB_recursive().isValid())
	{
		reset();
	}

	//force visibility
	entity->setVisible(true);
	entity->setEnabled(true);
	return true;
}

unsigned ecvAnnotationsTool::getNumberOfAssociatedEntity() const
{
	return m_entityContainer.getChildrenNumber();
}

void ecvAnnotationsTool::toggleEditMode(bool state)
{
	m_editMode = state;
}

bool ecvAnnotationsTool::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}
	
	return true;
}

bool ecvAnnotationsTool::start()
{
	assert(!m_processing);
	if (!m_annotationTool)
		return false;

	ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_entityContainer.getFirstChild());
	bool suceess = cloud && m_annotationTool->setInputCloud(cloud);
	if (!suceess)
	{
		CVLog::Error("no point cloud can be processed!");
		return false;
	}

	std::vector<std::string> labels;
	m_annotationTool->getAnnotationLabels(labels);
	updateLabelsCombox(labels);
	if (pauseToolButton->isChecked())
	{
		pauseToolButton->setChecked(false);
	}

	if (showOriginToolButton->isChecked())
	{
		showOriginToolButton->setChecked(false);
	}

	// custom settings according to different annotation mode
	switch (m_annotationTool->getAnnotationMode())
	{
	case ecvGenericAnnotationTool::BOUNDINGBOX:
		exportCloudWithAnnotations->setEnabled(false);
		break;
	case ecvGenericAnnotationTool::SEMANTICS:
		exportCloudWithAnnotations->setEnabled(true);
		break;
	default:
		break;
	}

	m_annotationTool->start();
	return ccOverlayDialog::start();
}

void ecvAnnotationsTool::stop(bool state)
{
	if (m_annotationTool)
	{
		if (!pauseToolButton->isChecked())
		{
			pauseToolButton->setChecked(true);
		}

		if (!showOriginToolButton->isChecked())
		{
			showOriginToolButton->setChecked(true);
		}

		m_annotationTool->stop();
		delete m_annotationTool;
		m_annotationTool = nullptr;
	}
	releaseAssociatedEntities();
	ccOverlayDialog::stop(state);
}

void ecvAnnotationsTool::updateLabelsCombox(const std::vector<std::string>& labels)
{
	if (labels.empty()) return;

	// just for avoid dummy triggering event
	m_disabledCombEvent = true;
	labelsComboBox->clear();
	for (size_t i = 0; i < labels.size(); ++i)
	{
		QString name = CVTools::toQString(labels[i]);
		labelsComboBox->addItem(name);
		if (i == 0)
		{
			continue;
		}
		ecvColor::Rgb col = ecvColor::LookUpTable::at(i);
		QColor backColor(col.r, col.g, col.b);
		labelsComboBox->setItemData(static_cast<int>(i), backColor, Qt::BackgroundColorRole);
	}

	QPalette pal = labelsComboBox->palette();
	pal.setColor(QPalette::Base, QColor(Qt::white));
	labelsComboBox->setPalette(pal);
	m_disabledCombEvent = false;
}

void ecvAnnotationsTool::shiftBox(unsigned char dim, bool minus)
{
}

void ecvAnnotationsTool::reset()
{
	m_box.clear();
	if (m_entityContainer.getChildrenNumber())
	{
		m_box = m_entityContainer.getBB_recursive();
	}
	if (m_annotationTool)
	{
		m_annotationTool->reset();
	}
}

void ecvAnnotationsTool::closeDialog()
{
	if (QMessageBox::question(this,
		tr("Quit"), tr("Are you sure you want to quit Annotation?"),
		QMessageBox::Ok, QMessageBox::Cancel) == QMessageBox::Ok)
	{
		stop(true);
	}
}

void ecvAnnotationsTool::exportAnnotationToSF()
{
	if (m_annotationTool)
	{
		std::vector<int> annotations;
		if (!m_annotationTool->getCurrentAnnotations(annotations))
		{
			CVLog::Warning("[ecvAnnotationsTool::exportAnnotationToSF] export Annotation To SF failed!");
			return;
		}
		std::vector<std::vector<int>> annosVector;
		annosVector.push_back(annotations);
		std::vector<std::vector<ScalarType>> scalarsVector;
		ccEntityAction::ConvertToScalarType<int>(annosVector, scalarsVector);

		ccHObject::Container container;
		container.push_back(m_entityContainer.getFirstChild());
		if (!ccEntityAction::importToSF(container, scalarsVector, "Clusters"))
		{
			CVLog::Error("[ecvAnnotationsTool::exportAnnotationToSF] import sf failed!");
		}
		else
		{
			m_entityContainer.getFirstChild()->showSF(false);
			CVLog::Print("[ecvAnnotationsTool::exportAnnotationToSF] "
				"Export annotations to sf successfully, please change the colors mode to scalar filed!");
		}
	}
}

void ecvAnnotationsTool::setTopView()
{
	setView(CC_TOP_VIEW);
}

void ecvAnnotationsTool::setBottomView()
{
	setView(CC_BOTTOM_VIEW);
}

void ecvAnnotationsTool::setFrontView()
{
	setView(CC_FRONT_VIEW);
}

void ecvAnnotationsTool::setBackView()
{
	setView(CC_BACK_VIEW);
}

void ecvAnnotationsTool::setLeftView()
{
	setView(CC_LEFT_VIEW);
}

void ecvAnnotationsTool::setRightView()
{
	setView(CC_RIGHT_VIEW);
}


ccBBox ecvAnnotationsTool::getSelectedEntityBbox()
{
	ccBBox box;
	if (getNumberOfAssociatedEntity() != 0)
	{
		box = m_entityContainer.getDisplayBB_recursive(false);
	}
	return box;
}

void ecvAnnotationsTool::setView(CC_VIEW_ORIENTATION orientation)
{
	ccBBox* bbox = nullptr;
	ccBBox box = getSelectedEntityBbox();
	if (box.isValid())
	{
		bbox = &box;
	}
	ecvDisplayTools::SetView(orientation, bbox);
}