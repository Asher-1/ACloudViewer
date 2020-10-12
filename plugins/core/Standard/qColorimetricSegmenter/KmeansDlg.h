#pragma once

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

#include <ui_KmeansDlg.h>

//Qt
#include <QDialog>

class KmeansDlg : public QDialog, public Ui::KmeansDialog
{
	Q_OBJECT

public:
	explicit KmeansDlg(QWidget* parent = nullptr);
	
};
