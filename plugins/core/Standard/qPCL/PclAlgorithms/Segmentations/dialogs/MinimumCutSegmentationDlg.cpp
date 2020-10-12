//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER PLUGIN: qPCL                         #
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
//#                         COPYRIGHT: Asher                               #
//#                                                                        #
//##########################################################################
//
#include "MinimumCutSegmentationDlg.h"

// ECV_DB_LIB
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>
#include <ecv2DLabel.h>
#include <ecvDisplayTools.h>
#include <ecvGLMatrix.h>

MinimumCutSegmentationDlg::MinimumCutSegmentationDlg(ecvMainAppInterface* app)
	: QDialog(app ? app->getActiveWindow() : 0)
	, Ui::MinimumCutSegmentationDlg()
	, m_app(app)
{
	setupUi(this);

	connect(label2DCloudComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onLabelChanged(int)));

}

void MinimumCutSegmentationDlg::onLabelChanged(int dummy)
{
	updateForeGroundPoint();
}

void MinimumCutSegmentationDlg::refreshLabelComboBox()
{
	if (m_app)
	{
		//add list of labels to the combo-boxes
		ccHObject::Container labels;
		if (m_app->dbRootObject())
			m_app->dbRootObject()->filterChildren(labels, true, CV_TYPES::LABEL_2D);

		unsigned cloudCount = 0;
		label2DCloudComboBox->clear();
		for (size_t i = 0; i < labels.size(); ++i)
		{
			if (labels[i]->isA(CV_TYPES::LABEL_2D)) // as filterChildren only test 'isKindOf'
			{
				QString name = getEntityName(labels[i]);
				QVariant uniqueID(labels[i]->getUniqueID());
				label2DCloudComboBox->addItem(name, uniqueID);
				++cloudCount;
			}
		}

		if (cloudCount >= 1 && m_app)
		{
			//return the 2D Label currently selected in the combox box
			selectLabel2DCheckBox->setChecked(true);
			label2DCloudComboBox->setEnabled(true);
			updateForeGroundPoint();
		}
		else
		{
			selectLabel2DCheckBox->setChecked(false);
			label2DCloudComboBox->setEnabled(false);
		}
	}

}

void MinimumCutSegmentationDlg::updateForeGroundPoint()
{
	if (!label2DCloudComboBox || !m_app)
	{
		return;
	}
	cc2DLabel* label = get2DLabelFromCombo(label2DCloudComboBox, m_app->dbRootObject());
	if (!label)
	{
		return;
	}
	const CCVector3* center = label->getPickedPoint(0).cloud->getPoint(label->getPickedPoint(0).index);
	if (!center)
	{
		return;
	}
	
	cxAxisDoubleSpinBox->setValue(center->x);
	cyAxisDoubleSpinBox->setValue(center->y);
	czAxisDoubleSpinBox->setValue(center->z);
}

QString MinimumCutSegmentationDlg::getEntityName(ccHObject* obj)
{
	if (!obj)
	{
		assert(false);
		return QString();
	}

	QString name = obj->getName();
	if (name.isEmpty())
		name = tr("unnamed");
	name += QString(" [ID %1]").arg(obj->getUniqueID());

	return name;
}

cc2DLabel* MinimumCutSegmentationDlg::get2DLabelFromCombo(QComboBox* comboBox, ccHObject* dbRoot)
{
	assert(comboBox && dbRoot);
	if (!comboBox || !dbRoot)
	{
		return nullptr;
	}

	//return the cloud currently selected in the combox box
	int index = comboBox->currentIndex();
	if (index < 0)
	{
		return nullptr;
	}
	unsigned uniqueID = comboBox->itemData(index).toUInt();
	ccHObject* item = dbRoot->find(uniqueID);
	if (!item || !item->isA(CV_TYPES::LABEL_2D))
	{
		return nullptr;
	}
	return static_cast<cc2DLabel*>(item);
}
