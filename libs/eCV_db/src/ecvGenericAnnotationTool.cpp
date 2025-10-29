// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericAnnotationTool.h"

#include "ecvPointCloud.h"

ecvGenericAnnotationTool::ecvGenericAnnotationTool(AnnotationMode mode)
    : m_annotationMode(mode), m_associatedCloud(nullptr) {}

ccPointCloud* ecvGenericAnnotationTool::vertices() {
    return static_cast<ccPointCloud*>(m_associatedCloud);
}
