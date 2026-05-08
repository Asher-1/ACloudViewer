// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewRepresentation.h"

#include "ecvGenericPointCloud.h"
#include "ecvHObject.h"
#include "ecvRepresentationManager.h"

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
    ecvRepresentationManager::instance().notifyChanged(this);
}

void ecvViewRepresentation::clearVisibilityOverride() {
    m_visibilityOverride.reset();
    m_dirty = true;
    ecvRepresentationManager::instance().notifyChanged(this);
}

void ecvViewRepresentation::setProperties(const Properties& props) {
    m_properties = props;
    m_dirty = true;
    ecvRepresentationManager::instance().notifyChanged(this);
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
        auto* cloud = static_cast<ccGenericPointCloud*>(m_entity);
        return static_cast<float>(cloud->getPointSize());
    }
    return 1.0f;
}

float ecvViewRepresentation::effectiveLineWidth() const {
    if (m_properties.lineWidth.has_value()) {
        return m_properties.lineWidth.value();
    }
    return 2.0f;
}

ecvViewRepresentation::RenderMode ecvViewRepresentation::effectiveRenderMode()
        const {
    if (m_properties.renderMode.has_value()) {
        return m_properties.renderMode.value();
    }
    return RenderMode::Inherit;
}

bool ecvViewRepresentation::effectiveEdgeVisibility() const {
    if (m_properties.edgeVisibility.has_value()) {
        return m_properties.edgeVisibility.value();
    }
    return false;
}

int ecvViewRepresentation::effectiveScalarFieldIndex() const {
    if (m_properties.scalarFieldIndex.has_value()) {
        return m_properties.scalarFieldIndex.value();
    }
    return -1;
}

bool ecvViewRepresentation::effectiveShowScalarField() const {
    if (m_properties.showScalarField.has_value()) {
        return m_properties.showScalarField.value();
    }
    if (m_entity) {
        return m_entity->sfShown();
    }
    return false;
}

bool ecvViewRepresentation::effectiveShowColors() const {
    if (m_properties.showColors.has_value()) {
        return m_properties.showColors.value();
    }
    if (m_entity) {
        return m_entity->colorsShown();
    }
    return true;
}

bool ecvViewRepresentation::effectiveShowNormals() const {
    if (m_properties.showNormals.has_value()) {
        return m_properties.showNormals.value();
    }
    if (m_entity) {
        return m_entity->normalsShown();
    }
    return false;
}

float ecvViewRepresentation::effectiveNormalScale() const {
    if (m_properties.normalScale.has_value()) {
        return m_properties.normalScale.value();
    }
    return 1.0f;
}
