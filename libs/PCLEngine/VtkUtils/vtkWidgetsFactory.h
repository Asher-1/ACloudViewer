// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef VTK_WIDGETS_FACTORY_H
#define VTK_WIDGETS_FACTORY_H

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

#endif  // VTK_WIDGETS_FACTORY_H
