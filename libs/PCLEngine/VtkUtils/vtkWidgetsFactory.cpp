// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkWidgetsFactory.h"

#include "VtkWidgets/filters/widgetsFiltersInterface.h"
#include "VtkWidgets/smallWidgets/smallWidgetsInterface.h"

using namespace DBLib;

ecvWidgetsInterface::Shared VtkWidgetsFactory::GetFilterWidgetInterface() {
    return ecvWidgetsInterface::Shared(new FiltersWidgetInterface());
}

ecvWidgetsInterface::Shared VtkWidgetsFactory::GetSmallWidgetsInterface() {
    return ecvWidgetsInterface::Shared(new SmallWidgetsInterface());
}