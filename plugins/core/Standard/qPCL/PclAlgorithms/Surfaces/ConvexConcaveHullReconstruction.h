// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_CONVEXCONCAVERECONSTRUCTION_HEADER
#define Q_PCL_PLUGIN_CONVEXCONCAVERECONSTRUCTION_HEADER

#include "BasePclModule.h"

// Qt
#include <QString>

class ConvexConcaveHullDlg;

//! Convex Concave Hull Reconstruction
class ConvexConcaveHullReconstruction : public BasePclModule {
public:
    ConvexConcaveHullReconstruction();
    virtual ~ConvexConcaveHullReconstruction();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    ConvexConcaveHullDlg* m_dialog;
    int m_dimension;
    float m_alpha;
};

#endif  // Q_PCL_PLUGIN_CONVEXCONCAVERECONSTRUCTION_HEADER
