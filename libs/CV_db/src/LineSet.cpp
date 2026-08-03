// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LineSet.h"

#include <QDataStream>
#include <QFile>
#include <numeric>

#include "ecvBBox.h"
#include "ecvDrawContext.h"
#include "ecvGenericGLDisplay.h"
#include "ecvOrientedBBox.h"

namespace cloudViewer {
namespace geometry {

namespace {

constexpr short kLineSetFileVersion = 40;

}  // namespace

LineSet::LineSet(const char* name) : ccShiftedObject(name) {
    set2DMode(false);
    setTransformFlag(true);
    setForeground(true);
    setVisible(true);
    lockVisibility(false);
    setColor(ecvColor::white);
    showColors(true);
    setWidth(0);
}

LineSet::LineSet(const std::vector<Eigen::Vector3d>& points,
                 const std::vector<Eigen::Vector2i>& lines,
                 const char* name)
    : LineSet(name) {
    points_ = points;
    lines_ = lines;
}

LineSet::LineSet(const LineSet& lineset) : ccShiftedObject(lineset) {
    points_ = lineset.points_;
    lines_ = lineset.lines_;
    colors_ = lineset.colors_;
    importParametersFrom(lineset);
}

LineSet& LineSet::operator=(const LineSet& lineset) {
    if (this != &lineset) {
        ccShiftedObject::operator=(lineset);
        points_ = lineset.points_;
        lines_ = lineset.lines_;
        colors_ = lineset.colors_;
        importParametersFrom(lineset);
    }
    return *this;
}

void LineSet::importParametersFrom(const LineSet& lineset) {
    set2DMode(lineset.m_mode2D);
    setTransformFlag(lineset.m_needTransform);
    setForeground(lineset.m_foreground);
    setVisible(lineset.isVisible());
    lockVisibility(lineset.isVisibilityLocked());
    setColor(lineset.m_rgbColor);
    setWidth(lineset.m_width);
    showColors(lineset.colorsShown());
    setGlobalScale(lineset.getGlobalScale());
    setGlobalShift(lineset.getGlobalShift());
    setGLTransformationHistory(lineset.getGLTransformationHistory());
    setMetaData(lineset.metaData());
}

void LineSet::set2DMode(bool state) { m_mode2D = state; }

void LineSet::setForeground(bool state) { m_foreground = state; }

void LineSet::setWidth(PointCoordinateType width) { m_width = width; }

void LineSet::setGlobalShift(const CCVector3d& shift) {
    ccShiftedObject::setGlobalShift(shift);
}

void LineSet::setGlobalScale(double scale) {
    ccShiftedObject::setGlobalScale(scale);
}

void LineSet::applyGLTransformation(const ccGLMatrix& trans) {
    ccHObject::applyGLTransformation(trans);
}

ccBBox LineSet::getOwnBB(bool withGLFeatures) {
    ccBBox box = GetAxisAlignedBoundingBox();
    box.setValidity((!is2DMode() || !withGLFeatures) && HasLines());
    return box;
}

void LineSet::drawBB(CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col) {
    if (!HasLines()) {
        return;
    }
    if (is2DMode()) {
        ccBBox box = GetAxisAlignedBoundingBox();
        if (box.isValid()) {
            box.draw(context, col);
        }
        return;
    }
    ccHObject::drawBB(context, col);
}

PointCoordinateType LineSet::computeLength() const {
    PointCoordinateType length = 0;
    for (size_t i = 0; i < lines_.size(); ++i) {
        const auto seg = GetLineCoordinate(i);
        length += static_cast<PointCoordinateType>(
                (seg.second - seg.first).norm());
    }
    return length;
}

void LineSet::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (!HasLines()) return;

    bool draw = false;
    if (MACRO_Draw3D(context)) {
        draw = !m_mode2D;
    } else if (m_mode2D) {
        const bool drawFG = MACRO_Foreground(context);
        draw = ((drawFG && m_foreground) || (!drawFG && !m_foreground));
    }

    if (!draw || !context.display) return;

    if (isColorOverridden()) {
        context.defaultPolylineColor = getTempColor();
    } else if (colorsShown()) {
        context.defaultPolylineColor = m_rgbColor;
    } else if (HasColors()) {
        context.defaultPolylineColor =
                ecvColor::Rgb::FromEigen(colors_.front());
    }

    if (m_width != 0) {
        context.currentLineWidth = m_width;
    } else {
        context.currentLineWidth = context.defaultLineWidth;
    }

    context.display->draw(context, this);
}

LineSet& LineSet::clear() {
    points_.clear();
    lines_.clear();
    colors_.clear();
    return *this;
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

LineSet& LineSet::Transform(const Eigen::Matrix4d& transformation) {
    TransformPoints(transformation, points_);
    return *this;
}

LineSet& LineSet::Translate(const Eigen::Vector3d& translation, bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

LineSet& LineSet::Scale(const double s, const Eigen::Vector3d& center) {
    ScalePoints(s, points_, center);
    return *this;
}

LineSet& LineSet::Rotate(const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& center) {
    RotatePoints(R, points_, center);
    return *this;
}

LineSet& LineSet::operator+=(const LineSet& lineset) {
    if (lineset.IsEmpty()) return (*this);
    const size_t old_point_num = points_.size();
    const size_t add_point_num = lineset.points_.size();
    const size_t new_point_num = old_point_num + add_point_num;
    const size_t old_line_num = lines_.size();
    const size_t add_line_num = lineset.lines_.size();
    const size_t new_line_num = old_line_num + add_line_num;

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
        lines_[old_line_num + i] = Eigen::Vector2i(
                lineset.lines_[i](0) + static_cast<int>(old_point_num),
                lineset.lines_[i](1) + static_cast<int>(old_point_num));
    }
    return (*this);
}

LineSet LineSet::operator+(const LineSet& lineset) const {
    return (LineSet(*this) += lineset);
}

short LineSet::minimumFileVersion_MeOnly() const {
    return std::max(kLineSetFileVersion,
                    ccHObject::minimumFileVersion_MeOnly());
}

bool LineSet::toFile_MeOnly(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));
    if (dataVersion < kLineSetFileVersion) {
        assert(false);
        return false;
    }

    if (!ccHObject::toFile_MeOnly(out, dataVersion)) return false;

    QDataStream stream(&out);
    stream.setVersion(QDataStream::Qt_5_0);

    stream << static_cast<quint32>(points_.size());
    for (const auto& p : points_) {
        stream << p.x() << p.y() << p.z();
    }

    stream << static_cast<quint32>(lines_.size());
    for (const auto& ln : lines_) {
        stream << ln.x() << ln.y();
    }

    stream << static_cast<quint32>(colors_.size());
    for (const auto& c : colors_) {
        stream << c.x() << c.y() << c.z();
    }

    saveShiftInfoToFile(out);

    stream << m_rgbColor.r << m_rgbColor.g << m_rgbColor.b;
    stream << m_mode2D << m_foreground << m_needTransform << m_width;

    return true;
}

bool LineSet::fromFile_MeOnly(QFile& in,
                              short dataVersion,
                              int flags,
                              LoadedIDMap& oldToNewIDMap) {
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    if (dataVersion < kLineSetFileVersion) return false;

    QDataStream stream(&in);
    stream.setVersion(QDataStream::Qt_5_0);

    quint32 pointCount = 0;
    stream >> pointCount;
    points_.resize(pointCount);
    for (quint32 i = 0; i < pointCount; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        stream >> x >> y >> z;
        points_[i] = Eigen::Vector3d(x, y, z);
    }

    quint32 lineCount = 0;
    stream >> lineCount;
    lines_.resize(lineCount);
    for (quint32 i = 0; i < lineCount; ++i) {
        int a = 0;
        int b = 0;
        stream >> a >> b;
        lines_[i] = Eigen::Vector2i(a, b);
    }

    quint32 colorCount = 0;
    stream >> colorCount;
    colors_.resize(colorCount);
    for (quint32 i = 0; i < colorCount; ++i) {
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        stream >> r >> g >> b;
        colors_[i] = Eigen::Vector3d(r, g, b);
    }

    loadShiftInfoFromFile(in);

    stream >> m_rgbColor.r >> m_rgbColor.g >> m_rgbColor.b;
    stream >> m_mode2D >> m_foreground >> m_needTransform >> m_width;

    return true;
}

}  // namespace geometry
}  // namespace cloudViewer
