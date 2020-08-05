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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#include "SACSegmentationDlg.h"

SACSegmentationDlg::SACSegmentationDlg(QWidget* parent)
	: QDialog(parent, Qt::Tool)
	, Ui::SACSegmentationDlg()
{
	setupUi(this);
	initParameters();
	connect(modelTypeCombo, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, 
		static_cast<void (SACSegmentationDlg::*)(int)>(&SACSegmentationDlg::modelsChanged));
}

void SACSegmentationDlg::modelsChanged(int currentIndex)
{
	switch (modelTypeCombo->currentIndex())
	{
	case 5:		// SACMODEL_CYLINDER
	case 6:		// SACMODEL_CONE
	case 11:	// SACMODEL_NORMAL_PLANE
	case 12:	// SACMODEL_NORMAL_SPHERE
	case 16:	// SACMODEL_NORMAL_PARALLEL_PLANE
	{
		normalDisWeightLabel->setEnabled(true);
		normalDisWeightSpinBox->setEnabled(true);
	}
		break;
	default:
	{
		normalDisWeightLabel->setEnabled(false);
		normalDisWeightSpinBox->setEnabled(false);
	}
		break;
	}
}
void SACSegmentationDlg::updateModelTypeComboBox(const QStringList& fields)
{
	modelTypeCombo->clear();
	for (int i = 0; i < fields.size(); i++)
	{
		modelTypeCombo->addItem(fields[i], i);
		modelTypeCombo->setItemText(i, fields[i]);
	}
}

void SACSegmentationDlg::updateMethodTypeComboBox(const QStringList& fields)
{
	methodTypeCombo->clear();
	for (int i = 0; i < fields.size(); i++)
	{
		methodTypeCombo->addItem(fields[i], i);
		methodTypeCombo->setItemText(i, fields[i]);
	}
}

void SACSegmentationDlg::initParameters()
{
	QStringList methodFields;
	QStringList modelFields;
	if (modelFields.isEmpty())
	{
		modelFields				<<
			tr("SACMODEL_PLANE") <<
			tr("SACMODEL_LINE") <<
			tr("SACMODEL_CIRCLE2D") <<
			tr("SACMODEL_CIRCLE3D") <<
			tr("SACMODEL_SPHERE") <<
			tr("SACMODEL_CYLINDER") <<
			tr("SACMODEL_CONE") <<
			tr("SACMODEL_TORUS") <<
			tr("SACMODEL_PARALLEL_LINE") <<
			tr("SACMODEL_PERPENDICULAR_PLANE") <<
			tr("SACMODEL_PARALLEL_LINES") <<
			tr("SACMODEL_NORMAL_PLANE") <<
			tr("SACMODEL_NORMAL_SPHERE") <<
			tr("SACMODEL_REGISTRATION") <<
			tr("SACMODEL_REGISTRATION_2D") <<
			tr("SACMODEL_PARALLEL_PLANE") <<
			tr("SACMODEL_NORMAL_PARALLEL_PLANE") <<
			tr("SACMODEL_STICK");
	}
	if (methodFields.isEmpty())
	{
		methodFields			<<
			tr("SAC_RANSAC")	<<
			tr("SAC_LMEDS")		<<
			tr("SAC_MSAC")		<<
			tr("SAC_RRANSAC")	<<
			tr("SAC_RMSAC")		<<
			tr("SAC_MLESAC")	<<
			tr("SAC_PROSAC");
	}

	//update the combo box
	updateModelTypeComboBox(modelFields);
	updateMethodTypeComboBox(methodFields);
}
