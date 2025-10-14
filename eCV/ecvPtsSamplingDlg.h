// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_POINTS_SAMPLING_DLG_HEADER
#define CC_POINTS_SAMPLING_DLG_HEADER

#include <ui_ptsSamplingDlg.h>

//! Dialog: points sampling on a mesh
class ccPtsSamplingDlg : public QDialog, public Ui::PointsSamplingDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccPtsSamplingDlg(QWidget* parent = 0);

    bool generateNormals() const;
    bool interpolateRGB() const;
    bool interpolateTexture() const;

    bool useDensity() const;
    double getDensityValue() const;
    unsigned getPointsNumber() const;

    void setPointsNumber(int count);
    void setDensityValue(double density);
    void setGenerateNormals(bool state);
    void setUseDensity(bool state);
};

#endif  // CC_POINTS_SAMPLING_DLG_HEADER
