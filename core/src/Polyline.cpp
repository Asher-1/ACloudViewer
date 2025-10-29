// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Polyline.h"

using namespace cloudViewer;

Polyline::Polyline(GenericIndexedCloudPersist* associatedCloud)
    : ReferenceCloud(associatedCloud), m_isClosed(false) {}

void Polyline::clear(bool /*unusedParam*/) {
    ReferenceCloud::clear();
    m_isClosed = false;
}
