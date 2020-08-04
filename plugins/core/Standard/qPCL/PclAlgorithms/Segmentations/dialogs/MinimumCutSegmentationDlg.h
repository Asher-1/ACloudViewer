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
#ifndef Q_PCL_PLUGIN_MINIMUMCUT_DLG_HEADER
#define Q_PCL_PLUGIN_MINIMUMCUT_DLG_HEADER

#include <ui_MinimumCutSegmentationDlg.h>

//Qt
#include <QDialog>

//system
#include <vector>

class ecvMainAppInterface;
class cc2DLabel;
class ccHObject;

class MinimumCutSegmentationDlg : public QDialog, public Ui::MinimumCutSegmentationDlg
{
	Q_OBJECT
public:
	explicit MinimumCutSegmentationDlg(ecvMainAppInterface* app);

	void refreshLabelComboBox();
	
public slots:
	void updateForeGroundPoint();
	void onLabelChanged(int);

protected:

	//! Gives access to the application (data-base, UI, etc.)
	ecvMainAppInterface* m_app;

	QString getEntityName(ccHObject* obj);

	cc2DLabel* get2DLabelFromCombo(QComboBox* comboBox, ccHObject* dbRoot);
};

#endif // Q_PCL_PLUGIN_MINIMUMCUT_DLG_HEADER
