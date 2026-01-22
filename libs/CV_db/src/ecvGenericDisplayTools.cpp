// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericDisplayTools.h"

static ecvGenericDisplayTools* s_genericTools = nullptr;

ecvGenericDisplayTools::ecvGenericDisplayTools() {}

void ecvGenericDisplayTools::SetInstance(ecvGenericDisplayTools* tool) {
    s_genericTools = tool;
}

ecvGenericDisplayTools* ecvGenericDisplayTools::GetInstance() {
    return s_genericTools;
}
