// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "CVAppCommon.h"
#include "ecvOptions.h"

// CV_CORE_LIB
#include <CVPlatform.h>

// ECV_DB_LIB
#include <ecvGuiParameters.h>

// Qt
#include <QDialog>

// system
#include <cassert>

namespace Ui {
class DisplayOptionsDlg;
}

//! Dialog to setup display settings
class CVAPPCOMMON_LIB_API ccDisplayOptionsDlg : public QDialog {
    Q_OBJECT

public:
    explicit ccDisplayOptionsDlg(QWidget* parent);
    ~ccDisplayOptionsDlg() override;

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
    
    void changeAppStyle(int);

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
    
    //! Default application style index (for reset)
    int m_defaultAppStyleIndex;

private:
    Ui::DisplayOptionsDlg* m_ui;
    
    //! Populate application style combo box
    void populateAppStyleComboBox();
};
