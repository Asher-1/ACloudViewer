// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_MINIMUMCUT_DLG_HEADER
#define Q_PCL_PLUGIN_MINIMUMCUT_DLG_HEADER

#include <ui_MinimumCutSegmentationDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class ecvMainAppInterface;
class cc2DLabel;
class ccHObject;

class MinimumCutSegmentationDlg : public QDialog,
                                  public Ui::MinimumCutSegmentationDlg {
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

#endif  // Q_PCL_PLUGIN_MINIMUMCUT_DLG_HEADER
