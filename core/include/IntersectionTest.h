// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_INTERSECTION_TEST_HEADER
#define CV_INTERSECTION_TEST_HEADER

#include <Eigen/Dense>

#include "CVCoreLib.h"

namespace cloudViewer {
namespace utility {

class CV_CORE_LIB_API IntersectionTest {
public:
    static bool AABBAABB(const Eigen::Vector3d& min0,
                         const Eigen::Vector3d& max0,
                         const Eigen::Vector3d& min1,
                         const Eigen::Vector3d& max1);

    static bool TriangleTriangle3d(const Eigen::Vector3d& p0,
                                   const Eigen::Vector3d& p1,
                                   const Eigen::Vector3d& p2,
                                   const Eigen::Vector3d& q0,
                                   const Eigen::Vector3d& q1,
                                   const Eigen::Vector3d& q2);

    static bool TriangleAABB(const Eigen::Vector3d& box_center,
                             const Eigen::Vector3d& box_half_size,
                             const Eigen::Vector3d& vert0,
                             const Eigen::Vector3d& vert1,
                             const Eigen::Vector3d& vert2);

    /// Tests if the given four points all lie on the same plane.
    static bool PointsCoplanar(const Eigen::Vector3d& p0,
                               const Eigen::Vector3d& p1,
                               const Eigen::Vector3d& p2,
                               const Eigen::Vector3d& p3);

    /// Computes the minimum distance between two lines. The first line is
    /// defined by 3D points \param p0 and \param p1, the second line is defined
    /// by 3D points \param q0 and \param q1. The returned distance is negative
    /// if no minimum distance can be computed. This implementation is based on
    /// the description of Paul Bourke
    /// (http://paulbourke.net/geometry/pointlineplane/).
    static double LinesMinimumDistance(const Eigen::Vector3d& p0,
                                       const Eigen::Vector3d& p1,
                                       const Eigen::Vector3d& q0,
                                       const Eigen::Vector3d& q1);

    /// Computes the minimum distance between two line segments. The first line
    /// segment is defined by 3D points \param p0 and \param p1, the second line
    /// segment is defined by 3D points \param q0 and \param q1. This
    /// implementation is based on the description of David Eberly
    /// (https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf).
    static double LineSegmentsMinimumDistance(const Eigen::Vector3d& p0,
                                              const Eigen::Vector3d& p1,
                                              const Eigen::Vector3d& q0,
                                              const Eigen::Vector3d& q1);
};

}  // namespace utility
}  // namespace cloudViewer

#endif  // CV_INTERSECTION_TEST_HEADER
