// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QHOUGH_NORMALS_HEADER
#define QHOUGH_NORMALS_HEADER

#include <QDialog>

namespace Ui {
class HoughNormalsDialog;
}

class qHoughNormalsDialog : public QDialog {
public:
    //! Default constructor
    explicit qHoughNormalsDialog(QWidget* parent = nullptr);

    ~qHoughNormalsDialog();

    // Settings
    struct Parameters {
        int K = 100;
        int T = 1000;
        int n_phi = 15;
        int n_rot = 5;
        bool use_density = false;
        float tol_angle_rad = 0.79f;
        int k_density = 5;
    };

    void setParameters(const Parameters& params);
    void getParameters(Parameters& params);

private:
    Ui::HoughNormalsDialog* m_ui;
};

#endif  // QHOUGH_NORMALS_HEADER
