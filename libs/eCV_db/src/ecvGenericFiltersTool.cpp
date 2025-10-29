// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericFiltersTool.h"

#include "ecvDisplayTools.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"

ecvGenericFiltersTool::ecvGenericFiltersTool(FilterType type)
    : m_filterType(type), m_associatedEntity(nullptr) {}

void ecvGenericFiltersTool::update() { ecvDisplayTools::UpdateScreen(); }