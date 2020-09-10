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

#ifndef ECV_DISPLAY_OPTIONS_DIALOG_HEADER
#define ECV_DISPLAY_OPTIONS_DIALOG_HEADER

// LOCAL
#include "ecvOptions.h"

// CV_CORE_LIB
#include <CVPlatform.h>

// ECV_DB_LIB
#include <ecvGuiParameters.h>

//Qt
#include <QDialog>

//system
#include <cassert>
#include <ui_displayOptionsDlg.h>

//! Dialog to setup display settings
class ccDisplayOptionsDlg : public QDialog, public Ui::DisplayOptionsDlg
{
	Q_OBJECT

public:
	explicit ccDisplayOptionsDlg(QWidget* parent);
	~ccDisplayOptionsDlg() override = default;

signals:
	void aspectHasChanged();

public slots:
	void changeBackgroundColor();

protected slots:
	void changeLightDiffuseColor();
	void changeLightAmbientColor();
	void changeLightSpecularColor();
	void changeMeshFrontDiffuseColor();
	void changeMeshBackDiffuseColor();
	void changeMeshSpecularColor();
	void changePointsColor();
	void changeTextColor();
	void changeLabelBackgroundColor();
	void changeLabelMarkerColor();
	void changeMaxMeshSize(double);
	void changeMaxCloudSize(double);
	void changeVBOUsage();
	void changeColorScaleRampWidth(int);
	void changeBBColor();
	void changeDefaultFontSize(int);
	void changeLabelFontSize(int);
	void changeNumberPrecision(int);
	void changeLabelOpacity(int);
	void changeLabelMarkerSize(int);

	void changeZoomSpeed(double);

	void changeAutoComputeOctreeOption(int);

	void doAccept();
	void doReject();
	void apply();
	void reset();

protected:

	//! Refreshes dialog to reflect new parameters values
	void refresh();

	QColor lightDiffuseColor;
	QColor lightAmbientColor;
	QColor lightSpecularColor;
	QColor meshFrontDiff;
	QColor meshBackDiff;
	QColor meshSpecularColor;
	QColor pointsDefaultCol;
	QColor textDefaultCol;
	QColor backgroundCol;
	QColor labelBackgroundCol;
	QColor labelMarkerCol;
	QColor bbDefaultCol;

	//! Current GUI parameters
	ecvGui::ParamStruct parameters;
	//! Current options
	ecvOptions options;

	//! Old parameters (for restore)
	ecvGui::ParamStruct oldParameters;
	//! Old options (for restore)
	ecvOptions oldOptions;
};

#endif
