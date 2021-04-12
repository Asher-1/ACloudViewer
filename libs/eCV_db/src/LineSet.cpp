// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
    return getAxisAlignedBoundingBox();
}

void LineSet::drawMeOnly(CC_DRAW_CONTEXT &context)
{
    bool is_empty = !hasPoints() || !hasLines();

    if (is_empty)
        return;

    if (MACRO_Draw3D(context) && ecvDisplayTools::GetMainScreen())
    {
        if (isColorOverriden())
        {
            context.defaultPolylineColor = getTempColor();
        }
        else if (colorsShown() && hasColors())
        {
            context.defaultPolylineColor = ecvColor::Rgb::FromEigen(colors_[0]);
        }
        context.currentLineWidth = 1;
        ecvDisplayTools::Draw(context, this);
    }
}


Eigen::Vector3d LineSet::getMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3d LineSet::getMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3d LineSet::getGeometryCenter() const { return ComputeCenter(points_); }

ccBBox LineSet::getAxisAlignedBoundingBox() const {
    return ccBBox::CreateFromPoints(points_);
}

ecvOrientedBBox LineSet::getOrientedBoundingBox() const {
    return ecvOrientedBBox::CreateFromPoints(points_);
}

LineSet &LineSet::transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, points_);
    return *this;
}

LineSet &LineSet::translate(const Eigen::Vector3d &translation, bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

LineSet &LineSet::scale(const double s, const Eigen::Vector3d &center) {
    ScalePoints(s, points_, center);
    return *this;
}

LineSet &LineSet::rotate(const Eigen::Matrix3d &R, const Eigen::Vector3d &center) {
    RotatePoints(R, points_, center);
    return *this;
}

LineSet &LineSet::operator+=(const LineSet &lineset) {
    if (lineset.isEmpty()) return (*this);
    size_t old_point_num = points_.size();
    size_t add_point_num = lineset.points_.size();
    size_t new_point_num = old_point_num + add_point_num;
    size_t old_line_num = lines_.size();
    size_t add_line_num = lineset.lines_.size();
    size_t new_line_num = old_line_num + add_line_num;

    if ((!hasLines() || hasColors()) && lineset.hasColors()) {
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
