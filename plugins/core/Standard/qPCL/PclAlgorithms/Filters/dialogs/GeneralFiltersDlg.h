//##########################################################################
//#                                                                        #
//#                        CLOUDVIEWER PLUGIN: qPCL                        #
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
#ifndef Q_PCL_PLUGIN_GENERALFILTERS_DIALOG_HEADER
#define Q_PCL_PLUGIN_GENERALFILTERS_DIALOG_HEADER

#include <ui_GeneralFiltersDlg.h>

#include <CVGeom.h>

//Qt
#include <QDialog>

class ecvMainAppInterface;
class ccHObject;
class ccPolyline;

class GeneralFiltersDlg : public QDialog, public Ui::GeneralFiltersDlg
{
public:
	explicit GeneralFiltersDlg(ecvMainAppInterface* app);

	ccPolyline* getPolyline();
	void getContour(std::vector<CCVector3> &contour);
	void refreshPolylineComboBox();

	const QString getComparisonField(float &minValue, float &maxValue);
	void getComparisonTypes(QStringList& types);

private:

	//! Gives access to the application (data-base, UI, etc.)
	ecvMainAppInterface* m_app;

	QString getEntityName(ccHObject* obj);

	ccPolyline* getPolylineFromCombo(QComboBox* comboBox, ccHObject* dbRoot);

};

#endif // Q_PCL_PLUGIN_GENERALFILTERS_DIALOG_HEADER
