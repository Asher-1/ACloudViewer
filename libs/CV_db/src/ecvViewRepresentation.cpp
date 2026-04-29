// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewRepresentation.h"

#include "ecvGenericPointCloud.h"
#include "ecvHObject.h"

ecvViewRepresentation::ecvViewRepresentation(ccHObject* entity,
                                             ecvGenericGLDisplay* view)
    : m_entity(entity), m_view(view) {}

bool ecvViewRepresentation::isVisible() const {
    if (m_visibilityOverride.has_value()) {
        return m_visibilityOverride.value();
    }
    return m_entity ? m_entity->isVisible() : false;
}

void ecvViewRepresentation::setVisible(bool v) {
    m_visibilityOverride = v;
    m_dirty = true;
}

void ecvViewRepresentation::clearVisibilityOverride() {
    m_visibilityOverride.reset();
    m_dirty = true;
}

void ecvViewRepresentation::setProperties(const Properties& props) {
    m_properties = props;
    m_dirty = true;
}

float ecvViewRepresentation::effectiveOpacity() const {
    if (m_properties.opacity.has_value()) {
        return m_properties.opacity.value();
    }
    if (m_entity) {
        return m_entity->getOpacity();
    }
    return 1.0f;
}

float ecvViewRepresentation::effectivePointSize() const {
    if (m_properties.pointSize.has_value()) {
        return m_properties.pointSize.value();
    }
    if (m_entity && m_entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        auto* cloud =
                static_cast<ccGenericPointCloud*>(m_entity);
        return static_cast<float>(cloud->getPointSize());
    }
    return 1.0f;
}
