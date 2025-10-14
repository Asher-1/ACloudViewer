// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef MLSSMOOTHINGUPSAMPLING_H
#define MLSSMOOTHINGUPSAMPLING_H

#include "BasePclModule.h"

class MLSDialog;

namespace PCLModules {
struct MLSParameters;
}

class MLSSmoothingUpsampling : public BasePclModule {
public:
    MLSSmoothingUpsampling();
    virtual ~MLSSmoothingUpsampling();

protected:
    // inherited from BasePclModule
    int openInputDialog();
    int compute();
    void getParametersFromDialog();

    MLSDialog* m_dialog;
    bool m_dialogHasParent;
    PCLModules::MLSParameters*
            m_parameters;  // We directly store all the parameters here
};

#endif  // MLSSMOOTHINGUPSAMPLING_H
