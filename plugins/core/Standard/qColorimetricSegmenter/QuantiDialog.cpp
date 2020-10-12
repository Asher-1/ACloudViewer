#include "QuantiDialog.h"

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

QuantiDialog::QuantiDialog(QWidget* parent)
	: QDialog(parent)
	, Ui::QuantiDialog()
{
	setupUi(this);

	connect(area_quanti, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &QuantiDialog::updateLabelValues);
}
/*
	Method applied after entering a value in RGB text fields
*/
void QuantiDialog::updateLabelValues()
{
	int value = area_quanti->value();
    nb_color_label->setText(QString::number(value*value*value));
}
