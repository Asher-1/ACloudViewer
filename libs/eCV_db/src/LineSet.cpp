// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LineSet.h"
#include "ecvBBox.h"
#include "ecvOrientedBBox.h"
#include "ecvDisplayTools.h"

#include <numeric>

namespace cloudViewer {
namespace geometry {

LineSet &LineSet::clear() {
    points_.clear();
    lines_.clear();
    colors_.clear();
    return *this;
}

ccBBox LineSet::getOwnBB(bool withGLFeatures)
{
    return GetAxisAlignedBoundingBox();
}

void LineSet::drawMeOnly(CC_DRAW_CONTEXT &context)
{
    bool is_empty = !HasPoints() || !HasLines();

    if (is_empty)
        return;

    if (MACRO_Draw3D(context) && ecvDisplayTools::GetMainScreen())
    {
        if (isColorOverridden())
        {
            context.defaultPolylineColor = getTempColor();
        }/*
        else if (colorsShown() && HasColors())
        {
            context.defaultPolylineColor = ecvColor::Rgb::FromEigen(colors_[0]);
        }*/
        context.currentLineWidth = 1;
        ecvDisplayTools::Draw(context, this);
    }
}


Eigen::Vector3d LineSet::GetMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3d LineSet::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3d LineSet::GetCenter() const { return ComputeCenter(points_); }

ccBBox LineSet::GetAxisAlignedBoundingBox() const {
    return ccBBox::CreateFromPoints(points_);
}

ecvOrientedBBox LineSet::GetOrientedBoundingBox() const {
    return ecvOrientedBBox::CreateFromPoints(points_);
}

LineSet &LineSet::Transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, points_);
    return *this;
}

LineSet &LineSet::Translate(const Eigen::Vector3d &translation, bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

LineSet &LineSet::Scale(const double s, const Eigen::Vector3d &center) {
    ScalePoints(s, points_, center);
    return *this;
}

LineSet &LineSet::Rotate(const Eigen::Matrix3d &R, const Eigen::Vector3d &center) {
    RotatePoints(R, points_, center);
    return *this;
}

LineSet &LineSet::operator+=(const LineSet &lineset) {
    if (lineset.IsEmpty()) return (*this);
    size_t old_point_num = points_.size();
    size_t add_point_num = lineset.points_.size();
    size_t new_point_num = old_point_num + add_point_num;
    size_t old_line_num = lines_.size();
    size_t add_line_num = lineset.lines_.size();
    size_t new_line_num = old_line_num + add_line_num;

    if ((!HasLines() || HasColors()) && lineset.HasColors()) {
        colors_.resize(new_line_num);
        for (size_t i = 0; i < add_line_num; i++) {
            colors_[old_line_num + i] = lineset.colors_[i];
        }
    } else {
        colors_.clear();
    }
    points_.resize(new_point_num);
    for (size_t i = 0; i < add_point_num; i++) {
        points_[old_point_num + i] = lineset.points_[i];
    }
    lines_.resize(new_line_num);
    for (size_t i = 0; i < add_line_num; i++) {
        lines_[old_line_num + i] =
                Eigen::Vector2i(lineset.lines_[i](0) + (int)old_point_num,
                                lineset.lines_[i](1) + (int)old_point_num);
    }
    return (*this);
}

LineSet LineSet::operator+(const LineSet &lineset) const {
    return (LineSet(*this) += lineset);
}

}  // namespace geometry
}  // namespace cloudViewer
