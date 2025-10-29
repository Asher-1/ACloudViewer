// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cloudViewer
#include <RegistrationTools.h>

// ECV_DB_LIB
#include <ecvGLMatrix.h>

class QWidget;
class QStringList;
class ccHObject;

//! Registration tools wrapper
class ccRegistrationTools {
public:
    //! Applies ICP registration on two entities
    /** \warning Automatically samples points on meshes if necessary (see code
     *for magic numbers ;)
     **/
    static bool
    ICP(ccHObject* data,
        ccHObject* model,
        ccGLMatrix& transMat,
        double& finalScale,
        double& finalRMS,
        unsigned& finalPointCount,
        const cloudViewer::ICPRegistrationTools::Parameters& inputParameters,
        bool useDataSFAsWeights = false,
        bool useModelSFAsWeights = false,
        QWidget* parent = nullptr);
};
