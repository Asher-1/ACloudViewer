// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// dialog
#include "ecvHObject.h"
#include "ui_poissonReconParametersDlg.h"

//! Wrapper to the "Poisson Surface Reconstruction (Version 9)" algorithm
/** "Poisson Surface Reconstruction", M. Kazhdan, M. Bolitho, and H. Hoppe
        Symposium on Geometry Processing (June 2006), pages 61--70
        http://www.cs.jhu.edu/~misha/Code/PoissonRecon/
**/

class ccPointCloud;
class ecvPoissonReconDlg : public QDialog, public Ui::PoissonReconParamDialog {
    Q_OBJECT
public:
    explicit ecvPoissonReconDlg(QWidget* parent = 0);

    bool start();
    bool addEntity(ccHObject* ent);
    ccHObject::Container& getReconstructions();

protected:
    bool showDialog();
    bool doComputation();
    void updateParams();
    void adjustParams(ccPointCloud* cloud);

private:
    QWidget* m_app;
    bool m_applyAllClouds;
    ccHObject::Container m_clouds;
    ccHObject::Container m_result;
    std::vector<bool> m_normalsMask;
};
