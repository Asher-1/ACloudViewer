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
#ifndef Q_PCL_PLUGIN_EUCLIDEANCLUSTER_DLG_HEADER
#define Q_PCL_PLUGIN_EUCLIDEANCLUSTER_DLG_HEADER

#include <ui_EuclideanClusterDlg.h>

//Qt
#include <QDialog>

//system
#include <vector>

class EuclideanClusterDlg : public QDialog, public Ui::EuclideanClusterDlg
{
public:
	explicit EuclideanClusterDlg(QWidget* parent = 0);
};

#endif // Q_PCL_PLUGIN_EUCLIDEANCLUSTER_DLG_HEADER
