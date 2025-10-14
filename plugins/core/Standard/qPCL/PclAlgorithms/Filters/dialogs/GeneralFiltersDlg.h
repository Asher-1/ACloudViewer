// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_GENERALFILTERS_DIALOG_HEADER
#define Q_PCL_PLUGIN_GENERALFILTERS_DIALOG_HEADER

#include <CVGeom.h>
#include <ui_GeneralFiltersDlg.h>

// Qt
#include <QDialog>

class ecvMainAppInterface;
class ccHObject;
class ccPolyline;

class GeneralFiltersDlg : public QDialog, public Ui::GeneralFiltersDlg {
public:
    explicit GeneralFiltersDlg(ecvMainAppInterface* app);

    ccPolyline* getPolyline();
    void getContour(std::vector<CCVector3>& contour);
    void refreshPolylineComboBox();

    const QString getComparisonField(float& minValue, float& maxValue);
    void getComparisonTypes(QStringList& types);

private:
    //! Gives access to the application (data-base, UI, etc.)
    ecvMainAppInterface* m_app;

    QString getEntityName(ccHObject* obj);

    ccPolyline* getPolylineFromCombo(QComboBox* comboBox, ccHObject* dbRoot);
};

#endif  // Q_PCL_PLUGIN_GENERALFILTERS_DIALOG_HEADER
