// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvWidgetsInterface.h>

#include "../qPCL.h"

class QPCL_ENGINE_LIB_API VtkWidgetsFactory {
public:
    VtkWidgetsFactory() = default;
    ~VtkWidgetsFactory() = default;

    static DBLib::ecvWidgetsInterface::Shared GetFilterWidgetInterface();
    static DBLib::ecvWidgetsInterface::Shared GetSmallWidgetsInterface();

private:
};
