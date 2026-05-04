// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericDisplayTools.h"

#include "ecvDisplayTools.h"
#include "ecvViewManager.h"

ecvGenericDisplayTools::ecvGenericDisplayTools() {}

ecvGenericDisplayTools* ecvGenericDisplayTools::GetInstance() {
    return static_cast<ecvGenericDisplayTools*>(
            ecvViewManager::instance().displayTools());
}
