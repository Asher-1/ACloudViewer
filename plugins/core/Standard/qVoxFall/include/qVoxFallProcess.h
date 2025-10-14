// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_VOXFALL_PROCESS_HEADER
#define Q_VOXFALL_PROCESS_HEADER

// Local
#include "qVoxFallDialog.h"

// DB
#include <ecvPointCloud.h>

class ecvMainAppInterface;

//! VoxFall process
/** See "VoxFall: Non-Parametric Volumetric Change Detection for Rockfalls",
        Farmakis, I., Guccione, D.E., Thoeni, K. and Giacomini, A., 2024,
        Computers and Geosciences
**/
class qVoxFallProcess {
public:
    static bool Compute(const qVoxFallDialog& dlg,
                        QString& errorMessage,
                        bool allowDialogs,
                        QWidget* parentWidget = nullptr,
                        ecvMainAppInterface* app = nullptr);
};

#endif  // Q_VOXFALL_PROCESS_HEADER
