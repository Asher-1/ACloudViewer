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
#ifndef Q_PCL_PLUGIN_REGIONGROWING_DLG_HEADER
#define Q_PCL_PLUGIN_REGIONGROWING_DLG_HEADER

#include <ui_RegionGrowingSegmentationDlg.h>

//Qt
#include <QDialog>

//system
#include <vector>

class RegionGrowingSegmentationDlg : public QDialog, public Ui::RegionGrowingSegmentationDlg
{
public:
	explicit RegionGrowingSegmentationDlg(QWidget* parent = 0);
};

#endif // Q_PCL_PLUGIN_REGIONGROWING_DLG_HEADER
