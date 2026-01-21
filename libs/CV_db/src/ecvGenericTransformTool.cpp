// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericTransformTool.h"

#include "ecvHObject.h"

ecvGenericTransformTool::ecvGenericTransformTool()
    : m_associatedEntity(nullptr) {}

bool ecvGenericTransformTool::setInputData(ccHObject* entity, int viewport) {
    if (!entity) {
        return false;
    }
    m_associatedEntity = entity;
    return true;
}
