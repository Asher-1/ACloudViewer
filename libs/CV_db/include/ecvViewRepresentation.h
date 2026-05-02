// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <optional>

#include "CV_db.h"

class ccHObject;
class ecvGenericGLDisplay;

/// Per-(entity, view) display state.
///
/// Tracks how a single data object appears in one specific view,
/// allowing different visualization properties (opacity, point size,
/// render mode, etc.) in each window.
///
/// Design reference: ParaView vtkSMRepresentationProxy.
class CV_DB_LIB_API ecvViewRepresentation {
public:
    enum class RenderMode : int {
        Inherit = -1,
        Points = 0,
        Wireframe = 1,
        Surface = 2,
        SurfaceWithEdges = 3
    };

    ecvViewRepresentation(ccHObject* entity, ecvGenericGLDisplay* view);
    ~ecvViewRepresentation() = default;

    ccHObject* getEntity() const { return m_entity; }
    ecvGenericGLDisplay* getView() const { return m_view; }

    // -- Visibility (per-view override) --

    bool isVisible() const;
    void setVisible(bool v);
    bool hasVisibilityOverride() const {
        return m_visibilityOverride.has_value();
    }
    void clearVisibilityOverride();

    // -- Per-view display properties --

    struct Properties {
        std::optional<float> opacity;
        std::optional<float> pointSize;
        std::optional<float> lineWidth;
        std::optional<RenderMode> renderMode;
        std::optional<bool> edgeVisibility;
        std::optional<int> scalarFieldIndex;
        std::optional<bool> showScalarField;
        std::optional<bool> showColors;
        std::optional<bool> showNormals;
        std::optional<float> normalScale;
    };

    const Properties& properties() const { return m_properties; }
    Properties& properties() { return m_properties; }
    void setProperties(const Properties& props);

    float effectiveOpacity() const;
    float effectivePointSize() const;
    float effectiveLineWidth() const;
    RenderMode effectiveRenderMode() const;
    bool effectiveEdgeVisibility() const;
    int effectiveScalarFieldIndex() const;
    bool effectiveShowScalarField() const;
    bool effectiveShowColors() const;
    bool effectiveShowNormals() const;
    float effectiveNormalScale() const;

    // -- Dirty state (needs VTK actor update) --

    bool isDirty() const { return m_dirty; }
    void setDirty(bool d = true) { m_dirty = d; }

private:
    ccHObject* m_entity;
    ecvGenericGLDisplay* m_view;
    std::optional<bool> m_visibilityOverride;
    Properties m_properties;
    bool m_dirty = true;
};
