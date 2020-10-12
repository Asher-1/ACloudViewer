//##########################################################################
//#                                                                        #
//#            CLOUDVIEWER PLUGIN: ColorimetricSegmenter                   #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#    COPYRIGHT:	Tri-Thien TRUONG, Ronan COLLIER, Mathieu LETRONE       #
//#                                                                        #
//##########################################################################

#include "HSVDialog.h"

//Local
#include "HSV.h"

// eCV
#include <ecvPickingHub.h>

// ECV_DB_LIB
#include <ecvGenericPointCloud.h>

//Qt
#include <QCheckBox>

/*
	Constructor
*/
HSVDialog::HSVDialog(ccPickingHub* pickingHub, QWidget* parent)
	: QDialog(parent)
	, Ui::HSVDialog()
	, m_pickingHub(pickingHub)
{
	assert(pickingHub);

	setModal(false);
	setupUi(this);

	red->setValue(0);
	green->setValue(0);
	blue->setValue(0);

	//link between Ui and actions
	connect(pointPickingButton_first, &QCheckBox::toggled, this, &HSVDialog::pickPoint);
	connect(red, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &HSVDialog::updateValues);
	connect(green, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &HSVDialog::updateValues);
	connect(blue, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &HSVDialog::updateValues);
		
	//auto disable picking mode on quit
	connect(this, &QDialog::finished, [&]()
	{
		//if (pointPickingButton_first->isChecked()) pointPickingButton_first->setChecked(false); 
		if (m_pickingHub)
			m_pickingHub->removeListener(this);
	}
	);
}

/*
	Method for the picking point functionnality
*/
void HSVDialog::pickPoint(bool state)
{
	if (!m_pickingHub)
	{
		return;
	}
	
	if (state)
	{
		if (!m_pickingHub->addListener(this, true))
		{
			CVLog::Error("Can't start the picking process (another tool is using it)");
			state = false;
		}
	}
	else
	{
		m_pickingHub->removeListener(this);
	}
	
	pointPickingButton_first->blockSignals(true);
	pointPickingButton_first->setChecked(state);
	pointPickingButton_first->blockSignals(false);
}

/*
	Method applied after a point is picked by picking point functionnality
*/
void HSVDialog::onItemPicked(const PickedItem& pi)
{
	assert(pi.entity);

	if (pi.entity->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		//Get RGB values of the picked point
		ccGenericPointCloud* cloud = static_cast<ccGenericPointCloud*>(pi.entity);
		const ecvColor::Rgb& rgb = cloud->getPointColor(pi.itemIndex);
		if (pointPickingButton_first->isChecked())
		{
			CVLog::Print("Point picked");

			//blocking signals to avoid updating 2 times hsv values for nothing
			red->blockSignals(true);
			green->blockSignals(true);

			red->setValue(rgb.r);
			green->setValue(rgb.g);
			blue->setValue(rgb.b);

			red->blockSignals(false);
			green->blockSignals(false);

			pointPickingButton_first->setChecked(false);
		}
	}
}

/*
	Method applied after entering a value in RGB text fields
*/
void HSVDialog::updateValues()
{
	ecvColor::Rgb rgb(red->value(), green->value(), blue->value());

	Hsv hsv_values(rgb);
	hue_first->setValue(hsv_values.h);
	sat_first->setValue(hsv_values.s);
	val_first->setValue(hsv_values.v);
}
