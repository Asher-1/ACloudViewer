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
#include "DONSegmentationDlg.h"

DONSegmentationDlg::DONSegmentationDlg(QWidget* parent)
	: QDialog(parent, Qt::Tool)
	, Ui::DONSegmentationDlg()
{
	setupUi(this);
	buttonGroup->setExclusive(true);
	buttonGroup->setId(curvatureRadioButton, 0);
	buttonGroup->setId(xRadioButton, 1);
	buttonGroup->setId(yRadioButton, 2);
	buttonGroup->setId(zRadioButton, 3);
}

const QString DONSegmentationDlg::getComparisonField()
{
	return buttonGroup->checkedButton()->text();
}

void DONSegmentationDlg::getComparisonTypes(QStringList& types)
{
	types.clear();
	if (equalCheckBox->isChecked())
	{
		if (greaterCheckBox->isChecked())
		{
			types << "GE";
		}

		if (lessThanCheckBox->isChecked())
		{
			types << "LE";
		}

		if (!greaterCheckBox->isChecked() && !lessThanCheckBox->isChecked())
		{
			types << "EQ";
		}
	}
	else
	{
		if (greaterCheckBox->isChecked())
		{
			types << "GT";
		}

		if (lessThanCheckBox->isChecked())
		{
			types << "LT";
		}
	}
}
