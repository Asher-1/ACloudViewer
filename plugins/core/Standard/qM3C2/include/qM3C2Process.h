// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_M3C2_PROCESS_HEADER
#define Q_M3C2_PROCESS_HEADER

// Local
#include "qM3C2Dialog.h"

class ecvMainAppInterface;

//! M3C2 process
/** See "Accurate 3D comparison of complex topography with terrestrial laser
scanner: application to the Rangitikei canyon (N-Z)", Lague, D., Brodu, N. and
Leroux, J., 2013, ISPRS journal of Photogrammmetry and Remote Sensing
**/
class qM3C2Process {
public:
    static bool Compute(const qM3C2Dialog& dlg,
                        QString& errorMessage,
                        ccPointCloud*& outputCloud,
                        bool allowDialogs,
                        QWidget* parentWidget = nullptr,
                        ecvMainAppInterface* app = nullptr);
};

#endif  // Q_M3C2_PROCESS_HEADER
