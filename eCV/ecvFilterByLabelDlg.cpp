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
//#                  COPYRIGHT: Daniel Girardeau-Montaut                   #
//#                                                                        #
//##########################################################################

#include "ecvFilterByLabelDlg.h"

// LOCAL
#include "MainWindow.h"
#include "db_tree/ecvDBRoot.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>
#include <ClassMap.h>

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvSubMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvDisplayTools.h>
#include <ecvHObjectCaster.h>

// QT
#include <QCheckBox>

// SYSTEM
#include <unordered_set>

ecvFilterByLabelDlg::ecvFilterByLabelDlg(QWidget* parent)
	: ccOverlayDialog(parent)
	, Ui::FilterByLabelDialog()
	, m_mode(CANCEL)
	, m_minVald(0.0)
	, m_maxVald(1.0)
{
	setupUi(this);

	connect(splitPushButton, &QAbstractButton::clicked, this, &ecvFilterByLabelDlg::onSplit);
	connect(exportSelectedToolButton, &QAbstractButton::clicked, this, &ecvFilterByLabelDlg::onExportSelected);
	connect(exportUnselectedToolButton, &QAbstractButton::clicked, this, &ecvFilterByLabelDlg::onExportUnSelected);
	connect(selectAllRadioButton, &QAbstractButton::clicked, this, &ecvFilterByLabelDlg::selectAllClasses);
	connect(unselectAllRadioButton, &QAbstractButton::clicked, this, &ecvFilterByLabelDlg::selectAllClasses);
	connect(cancelToolButton, &QAbstractButton::clicked, this, &ecvFilterByLabelDlg::cancel);
	connect(toggleSelectedToolButton, &QAbstractButton::clicked, this, &ecvFilterByLabelDlg::toggleSelectedVisibility);
}

ecvFilterByLabelDlg::~ecvFilterByLabelDlg()
{
}

bool ecvFilterByLabelDlg::start()
{
	if (!ecvDisplayTools::GetCurrentScreen())
		return false;

	ccHObject* ent = m_toFilter.first;
	ccPointCloud* pc = m_toFilter.second;
	if (!ent || !pc)
	{
		return false;
	}

	labelGroupBox->setEnabled(true);
	CVLib::ScalarField* sf = pc->getCurrentDisplayedScalarField();
	Q_ASSERT(sf->currentSize() == pc->size());

	clear();
	std::unordered_set<size_t> labels_set;
	for (unsigned j = 0; j < pc->size(); j++)
	{
		ScalarType value = sf->getValue(j);
		size_t valueId = static_cast<size_t>(value);
		if (value - valueId != 0)
		{
			CVLog::Error("[ecvFilterByLabelDlg::start] only support int type scalar fields!");
			clear();
			return false;
		}

		labels_set.insert(valueId);
	}
	m_labels.assign(labels_set.begin(), labels_set.end());
	std::sort(m_labels.begin(), m_labels.end());
	createCheckboxesWithLabels();

	return ccOverlayDialog::start();
}

void ecvFilterByLabelDlg::createCheckboxesWithLabels()
{
	if (m_labels.empty())
	{
		return;
	}

	if (gridLayout && !gridLayout->isEmpty())
	{
		clearLayoutWidgets(gridLayout);
	}

	for (size_t i = 0; i < m_labels.size(); ++i)
	{
		size_t label = m_labels[i];
		const std::string& labelName = ClassMap::SemanticMap[label];
		QCheckBox* labelCheckBox = new QCheckBox(labelGroupBox);
		labelCheckBox->setObjectName(QString::fromUtf8(labelName.c_str()));
		labelCheckBox->setText(QString::fromUtf8(labelName.c_str()));
		bool isChecked = label >= m_minVald && label <= m_maxVald;
		if (label == 0)
		{
			isChecked = false;
		}
		
		labelCheckBox->setChecked(isChecked);
		ecvColor::Rgb col;
		if (m_toFilter.second)
		{
			col = *m_toFilter.second->getScalarValueColor(static_cast<ScalarType>(label));
		}
		else
		{
			col = ecvColor::LookUpTable::at(label);
		}

		QString styleSheet = QString("background-color: rgb(%1, %2, %3, %4)").arg(col.r).arg(col.g).arg(col.b).arg(125);
		labelCheckBox->setStyleSheet(styleSheet);
		//labelCheckBox->setAutoFillBackground(true);
		//QPalette pal = labelCheckBox->palette();
		//QColor backColor(col.r, col.g, col.b);
		//pal.setColor(QPalette::Base, backColor);
		//labelCheckBox->setPalette(pal);

		int rowIndex = static_cast<int>(i / 2);
		int colIndex = static_cast<int>(i % 2);

		gridLayout->addWidget(labelCheckBox, rowIndex, colIndex, 1, 1,
			Qt::AlignLeft | Qt::AlignVCenter);
	}
}

void ecvFilterByLabelDlg::stop(bool state)
{
	ccOverlayDialog::stop(state);
}

bool ecvFilterByLabelDlg::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}

	return true;
}

void ecvFilterByLabelDlg::getSelectedFilterClasses(std::vector<ScalarType>& filteredClasses)
{
	if (!filteredClasses.empty())
	{
		filteredClasses.clear();
	}
	QList<QCheckBox*> list = labelGroupBox->findChildren<QCheckBox*>();

	foreach(QCheckBox* ncheckBox, list)
	{
		if (ncheckBox && ncheckBox->isChecked())
		{
			int index = ClassMap::FindindexByValue(CVTools::fromQString(ncheckBox->text()));
			filteredClasses.push_back(static_cast<ScalarType>(index));
		}
	}
}

void ecvFilterByLabelDlg::selectAllClasses()
{
	bool state = selectAllRadioButton->isChecked();
	QList<QCheckBox*> list = labelGroupBox->findChildren<QCheckBox*>();

	foreach(QCheckBox* ncheckBox, list)
	{
		if (ncheckBox)
		{
			ncheckBox->setChecked(state);
		}
	}
}

void ecvFilterByLabelDlg::toggleSelectedVisibility()
{
}

void ecvFilterByLabelDlg::cancel()
{
	stop(false);
	clear();
}

void ecvFilterByLabelDlg::clear()
{
	m_labels.clear();
}

bool ecvFilterByLabelDlg::setInputEntity(ccHObject* entity)
{
	ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
	if (!cloud || !cloud->isKindOf(CV_TYPES::POINT_CLOUD))
		return false;

	ccPointCloud* pc = static_cast<ccPointCloud*>(cloud);
	ccScalarField* sf = pc->getCurrentDisplayedScalarField();
	if (!sf)
	{
		CVLog::Warning(tr("Entity [%1] has no active scalar field !").arg(pc->getName()));
		return false;
	}
	else
	{
		m_minVald = static_cast<double>(sf->displayRange().start());
		m_maxVald = static_cast<double>(sf->displayRange().stop());
		if (m_maxVald >= 20)
		{
			CVLog::Warning(tr("Entity [%1] scalar field value range is bigger than 20!").arg(pc->getName()));
			return false;
		}
	}

	int pointNumber = static_cast<int>(pc->size());
	if (pointNumber < 10)
	{
		CVLog::Warning(
			QString("[ecvFilterByLabelDlg::setInputEntity] "
				"Skip entity [%1] as the point number of it is %2 lower than min limit 10!")
			.arg(entity->getName(), pointNumber));
		return false;
	}

	m_toFilter = EntityAndVerticesType(entity, pc);

	return true;
}

void ecvFilterByLabelDlg::apply()
{
	assert(m_mode != ecvFilterByLabelDlg::CANCEL);
	ccHObject* ent = m_toFilter.first;
	ccPointCloud* pc = m_toFilter.second;
	if (!ent || !pc)
	{
		return;
	}

	std::vector<ScalarType> selectedLabels;
	getSelectedFilterClasses(selectedLabels);
	if (selectedLabels.empty())
	{
		CVLog::Warning("[ecvFilterByLabelDlg::apply] no filter labels selected, please select some and try again!");
		return;
	}

	//we set as output (OUT) the currently displayed scalar field
	int outSfIdx = pc->getCurrentDisplayedScalarFieldIndex();
	assert(outSfIdx >= 0);
	pc->setCurrentOutScalarField(outSfIdx);

	ccHObject* resultInside = nullptr;
	ccHObject* resultOutside = nullptr;
	ccHObject::Container results;
	if (ent->isKindOf(CV_TYPES::MESH))
	{
		pc->hidePointsByScalarValue(selectedLabels);

		if (ecvFilterByLabelDlg::SPLIT == m_mode || 
			ecvFilterByLabelDlg::EXPORT_SELECTED == m_mode)
		{
			if (ent->isA(CV_TYPES::MESH)/*|| ent->isKindOf(CV_TYPES::PRIMITIVE)*/) //TODO
				resultInside = ccHObjectCaster::ToMesh(ent)->createNewMeshFromSelection(false);
			else if (ent->isA(CV_TYPES::SUB_MESH))
				resultInside = ccHObjectCaster::ToSubMesh(ent)->createNewSubMeshFromSelection(false);
		}

		if (ecvFilterByLabelDlg::SPLIT == m_mode || 
			ecvFilterByLabelDlg::EXPORT_UNSELECTED == m_mode)
		{
			pc->invertVisibilityArray();
			if (ent->isA(CV_TYPES::MESH)/*|| ent->isKindOf(CV_TYPES::PRIMITIVE)*/) //TODO
				resultOutside = ccHObjectCaster::ToMesh(ent)->createNewMeshFromSelection(false);
			else if (ent->isA(CV_TYPES::SUB_MESH))
				resultOutside = ccHObjectCaster::ToSubMesh(ent)->createNewSubMeshFromSelection(false);
		}

		pc->unallocateVisibilityArray();
	}
	else if (ent->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		if (ecvFilterByLabelDlg::SPLIT == m_mode ||
			ecvFilterByLabelDlg::EXPORT_SELECTED == m_mode)
		{
			//shortcut, as we know here that the point cloud is a "ccPointCloud"
			resultInside = pc->filterPointsByScalarValue(selectedLabels, false);
		}

		if (ecvFilterByLabelDlg::SPLIT == m_mode ||
			ecvFilterByLabelDlg::EXPORT_UNSELECTED == m_mode)
		{
			resultOutside = pc->filterPointsByScalarValue(selectedLabels, true);
		}
	}

	if (resultInside)
	{
		ent->setEnabled(false);
		//resultInside->setDisplay(ent->getDisplay());
		//resultInside->prepareDisplayForRefresh();
		MainWindow::TheInstance()->addToDB(resultInside);

		results.push_back(resultInside);
	}
	if (resultOutside)
	{
		ent->setEnabled(false);
		//resultOutside->setDisplay(ent->getDisplay());
		//resultOutside->prepareDisplayForRefresh();
		resultOutside->setName(resultOutside->getName() + ".outside");
		MainWindow::TheInstance()->addToDB(resultOutside);

		results.push_back(resultOutside);
	}

	if (!results.empty())
	{
		CVLog::Warning(tr("Previously selected entities (sources) have been hidden!"));
		if (MainWindow::TheInstance()->db())
		{
			MainWindow::TheInstance()->db()->selectEntities(results);
		}
	}

	stop(true);
	clear();
	return;
}

void ecvFilterByLabelDlg::clearLayoutWidgets(QLayout *layout)
{
	if (gridLayout && !gridLayout->isEmpty())
	{
		QLayoutItem *item;
		while ((item = layout->takeAt(0)) != 0) {
			// delete widget
			if (item->widget()) {
				delete item->widget();
				//item->widget()->deleteLater();
			}
			// delete layout
			QLayout *childLayout = item->layout();
			if (childLayout) {
				clearLayoutWidgets(childLayout);
			}
			delete item;
		}
	}
}
