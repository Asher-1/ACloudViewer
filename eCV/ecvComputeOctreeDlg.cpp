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

#include "ecvComputeOctreeDlg.h"

//Local
#include "ecvBoundingBoxEditorDlg.h"

//ECV_DB_LIB
#include <ecvOctree.h>

ccComputeOctreeDlg::ccComputeOctreeDlg(const ccBBox& baseBBox, double minCellSize, QWidget* parent/*=0*/)
	: QDialog(parent)
	, Ui::ComputeOctreeDialog()
	, m_bbEditorDlg(0)
{
	setupUi(this);

	headerLabel->setText(QString("Max subdivision level: %1").arg(ccOctree::MAX_OCTREE_LEVEL));

	//minimum cell size
	if (minCellSize > 0.0)
	{
		cellSizeDoubleSpinBox->setMinimum(minCellSize);
		cellSizeDoubleSpinBox->setMaximum(1.0e9);
	}
	else
	{
		CVLog::Warning("[ccComputeOctreeDlg] Invalid minimum cell size specified!");
		cellSizeRadioButton->setEnabled(false);	
	}

	//custom bbox editor
	if (baseBBox.isValid())
	{
		m_bbEditorDlg = new ccBoundingBoxEditorDlg(this);
		m_bbEditorDlg->setBaseBBox(baseBBox,true);
		m_bbEditorDlg->forceKeepSquare(true);
		connect(customBBToolButton, &QAbstractButton::clicked, m_bbEditorDlg, &ccBoundingBoxEditorDlg::exec);
	}
	else
	{
		CVLog::Warning("[ccComputeOctreeDlg] Invalid base bounding-box specified!");
		customBBRadioButton->setEnabled(false);
	}
}

ccComputeOctreeDlg::ComputationMode ccComputeOctreeDlg::getMode() const
{
	//defaultRadioButton
	if (cellSizeRadioButton->isChecked())
		return MIN_CELL_SIZE;
	else if (customBBRadioButton->isChecked())
		return CUSTOM_BBOX;

	assert(defaultRadioButton->isChecked());
	return DEFAULT;
}

double ccComputeOctreeDlg::getMinCellSize() const
{
	return cellSizeDoubleSpinBox->value();
}

ccBBox ccComputeOctreeDlg::getCustomBBox() const
{
	return (m_bbEditorDlg ? m_bbEditorDlg->getBox() : ccBBox());
}
