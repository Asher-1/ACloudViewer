// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "CV_db.h"
#include "ecvShiftedObject.h"

class ccMesh;
class ccBBox;
class ecvOrientedBBox;
class ccPointCloud;
namespace cloudViewer {
namespace geometry {

class TetraMesh;

/// \class LineSet
///
/// \brief A set of independent line segments in 2D or 3D.
///
/// General-purpose line entity (multi-segment, optionally disconnected).
/// Mirrors ccPolyline display properties: 2D overlay mode, color, line width,
/// foreground/background draw order.
class CV_DB_LIB_API LineSet : public ccShiftedObject {
public:
    /// \brief Default constructor.
    explicit LineSet(const char* name = "LineSet");

    /// \brief Parameterized constructor.
    LineSet(const std::vector<Eigen::Vector3d>& points,
            const std::vector<Eigen::Vector2i>& lines,
            const char* name = "LineSet");

    /// \brief Copy constructor.
    LineSet(const LineSet& lineset);

    ~LineSet() override = default;

    LineSet& operator=(const LineSet& lineset);

    // inherited methods (ccHObject)
    bool isSerializable() const override { return true; }

    CV_CLASS_ENUM getClassID() const override { return CV_TYPES::LINESET; }

    ccBBox getOwnBB(bool withGLFeatures = false) override;

    void drawBB(CC_DRAW_CONTEXT& context,
                const ecvColor::Rgb& col) override;

    void applyGLTransformation(const ccGLMatrix& trans) override;

    // inherited methods (ccDrawableObject)
    bool hasColors() const override { return true; }

    // inherited methods (ccShiftedObject)
    void setGlobalShift(const CCVector3d& shift) override;
    void setGlobalScale(double scale) override;

    void set2DMode(bool state);
    bool is2DMode() const { return m_mode2D; }

    void setTransformFlag(bool state) { m_needTransform = state; }
    bool needTransform() const { return m_needTransform; }

    void setForeground(bool state);
    bool isForeground() const { return m_foreground; }

    void setColor(const ecvColor::Rgb& col) {
        enableTempColor(false);
        m_rgbColor = col;
    }
    const ecvColor::Rgb& getColor() const { return m_rgbColor; }

    void setWidth(PointCoordinateType width);
    PointCoordinateType getWidth() const { return m_width; }

    PointCoordinateType computeLength() const;
    unsigned segmentCount() const {
        return static_cast<unsigned>(lines_.size());
    }

    void importParametersFrom(const LineSet& lineset);

protected:
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;

    void drawMeOnly(CC_DRAW_CONTEXT& context) override;

public:
    LineSet& clear();
    bool IsEmpty() const override { return !HasPoints(); }
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    ccBBox GetAxisAlignedBoundingBox() const override;
    ecvOrientedBBox GetOrientedBoundingBox() const override;
    LineSet& Transform(const Eigen::Matrix4d& transformation) override;
    LineSet& Translate(const Eigen::Vector3d& translation,
                       bool relative = true) override;
    LineSet& Scale(const double s, const Eigen::Vector3d& center) override;
    LineSet& Rotate(const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& center) override;

    LineSet& operator+=(const LineSet& lineset);
    LineSet operator+(const LineSet& lineset) const;

    bool HasPoints() const { return !points_.empty(); }
    bool HasLines() const { return HasPoints() && !lines_.empty(); }

    /// Per-line RGB colors (optional; size must match lines_ when used).
    bool HasColors() const {
        return HasLines() && colors_.size() == lines_.size();
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> GetLineCoordinate(
            size_t line_index) const {
        return std::make_pair(points_[lines_[line_index][0]],
                              points_[lines_[line_index][1]]);
    }

    LineSet& PaintUniformColor(const Eigen::Vector3d& color) {
        setColor(ecvColor::Rgb::FromEigen(color));
        showColors(true);
        colors_.clear();
        return *this;
    }

    static std::shared_ptr<LineSet> CreateFromPointCloudCorrespondences(
            const ccPointCloud& cloud0,
            const ccPointCloud& cloud1,
            const std::vector<std::pair<int, int>>& correspondences);

    static std::shared_ptr<LineSet> CreateFromOrientedBoundingBox(
            const ecvOrientedBBox& box);

    static std::shared_ptr<LineSet> CreateFromAxisAlignedBoundingBox(
            const ccBBox& box);

    static std::shared_ptr<LineSet> CreateFromTriangleMesh(const ccMesh& mesh);

    static std::shared_ptr<LineSet> CreateFromTetraMesh(const TetraMesh& mesh);

    static std::shared_ptr<LineSet> CreateCameraVisualization(
            int view_width_px,
            int view_height_px,
            const Eigen::Matrix3d& intrinsic,
            const Eigen::Matrix4d& extrinsic,
            double scale = 1.0);

public:
    /// Points coordinates.
    std::vector<Eigen::Vector3d> points_;
    /// Lines denoted by the index of points forming the line.
    std::vector<Eigen::Vector2i> lines_;
    /// RGB colors of lines.
    std::vector<Eigen::Vector3d> colors_;

protected:
    ecvColor::Rgb m_rgbColor;
    PointCoordinateType m_width;
    bool m_mode2D;
    bool m_needTransform;
    bool m_foreground;
};

}  // namespace geometry
}  // namespace cloudViewer
