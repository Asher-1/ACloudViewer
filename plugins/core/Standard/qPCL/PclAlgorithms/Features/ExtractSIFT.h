// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../BasePclModule.h"

// Qt
#include <QString>

class SIFTExtractDlg;

//! SIFT keypoints extraction
class ExtractSIFT : public BasePclModule {
public:
    ExtractSIFT();
    virtual ~ExtractSIFT();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    SIFTExtractDlg* m_dialog;
    int m_nr_octaves;
    float m_min_scale;
    int m_nr_scales_per_octave;
    float m_min_contrast;
    bool m_use_min_contrast;
    QString m_field_to_use;
    std::string m_field_to_use_no_space;

    enum Modes { RGB, SCALAR_FIELD };
    Modes m_mode;
};
