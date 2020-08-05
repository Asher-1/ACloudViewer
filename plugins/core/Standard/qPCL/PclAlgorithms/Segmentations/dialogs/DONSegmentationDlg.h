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
#ifndef Q_PCL_PLUGIN_DONSEGMENTATION_DLG_HEADER
#define Q_PCL_PLUGIN_DONSEGMENTATION_DLG_HEADER

#include <ui_DONSegmentationDlg.h>

//Qt
#include <QDialog>

//system
#include <vector>

class DONSegmentationDlg : public QDialog, public Ui::DONSegmentationDlg
{
public:
	explicit DONSegmentationDlg(QWidget* parent = 0);

	const QString getComparisonField();
	void getComparisonTypes(QStringList& types);
};

#endif // Q_PCL_PLUGIN_DONSEGMENTATION_DLG_HEADER
