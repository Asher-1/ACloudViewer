// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericCameraTool.h"

#include "ecvDisplayTools.h"

// Qt includes.
#include <QDebug>
#include <QPointer>
#include <QSettings>
#include <QString>
#include <QToolButton>

// STL
#include <sstream>
#include <string>

ecvGenericCameraTool::CameraInfo ecvGenericCameraTool::OldCameraParam =
        ecvGenericCameraTool::CameraInfo();
ecvGenericCameraTool::CameraInfo ecvGenericCameraTool::CurrentCameraParam =
        ecvGenericCameraTool::CameraInfo();

//-----------------------------------------------------------------------------
ecvGenericCameraTool::ecvGenericCameraTool() {}

//-----------------------------------------------------------------------------
ecvGenericCameraTool::~ecvGenericCameraTool() {}

//-----------------------------------------------------------------------------
void ecvGenericCameraTool::setAutoPickPivotAtCenter(bool state) {
    ecvDisplayTools::SetAutoPickPivotAtCenter(state);
}

//-----------------------------------------------------------------------------
void ecvGenericCameraTool::saveCameraConfiguration(const std::string& file) {}

//-----------------------------------------------------------------------------
void ecvGenericCameraTool::loadCameraConfiguration(const std::string& file) {}