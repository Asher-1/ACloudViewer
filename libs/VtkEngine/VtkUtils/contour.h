// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file contour.h
/// @brief Surface widget with contour plane and isosurface rendering.

#include "surface.h"
#include "vector4f.h"

namespace VtkUtils {
class ContourPrivate;
/// @class Contour
/// @brief Surface widget extended with contour plane and isosurface controls.
class QVTK_ENGINE_LIB_API Contour : public Surface {
    Q_OBJECT
public:
    explicit Contour(QWidget* parent = nullptr);
    ~Contour();

    /// @param vectors Vector4F data for contour (x,y,z,v)
    void setVectors(const QList<Vector4F>& vectors);

    /// @param num Number of contour levels
    void setNumberOfContours(int num);
    /// @return Number of contour levels
    int numberOfContours() const;

    /// @param visible Show/hide contour plane
    void setPlaneVisible(bool visible = true);
    /// @return true if plane visible
    bool planeVisible() const;

    /// @param distance Plane distance for contour
    void setPlaneDistance(qreal distance);
    /// @return Current plane distance
    bool planeDistance() const;

protected:
    void renderSurface();

private:
    ContourPrivate* d_ptr;
    Q_DISABLE_COPY(Contour)
};

}  // namespace VtkUtils
