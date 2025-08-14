// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/visualization/rendering/Material.h"

namespace cloudViewer {
namespace t {
namespace geometry {

/// \class DrawableGeometry
///
/// \brief Mix-in class for geometry types that can be visualized
class DrawableGeometry {
public:
    DrawableGeometry() {}
    ~DrawableGeometry() {}

    /// Check if a material has been applied to this Geometry with SetMaterial.
    bool HasMaterial() const { return material_.IsValid(); }

    /// Get material associated with this Geometry.
    visualization::rendering::Material &GetMaterial() { return material_; }

    /// Get const reference to material associated with this Geometry
    const visualization::rendering::Material &GetMaterial() const {
        return material_;
    }

    /// Set the material properties associate with this Geometry
    void SetMaterial(const visualization::rendering::Material &material) {
        material_ = material;
    }

private:
    /// Material associated with this geometry
    visualization::rendering::Material material_;
};

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
