// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file CurveFitting.h
 * @brief Curve fitting utilities for point clouds (B-spline, line, polynomial).
 */

// LOCAL
#include <PclUtils/PCLCloud.h>
#include <PclUtils/PCLConv.h>

// CV_CORE_LIB
#include <CVConst.h>
#include <CVLog.h>

#include <vector>

class ccPolyline;
class ccPointCloud;

namespace CurveFittingTool {

/**
 * @class CurveFitting
 * @brief Fits curves (B-spline, line, polynomial) to point cloud data.
 */
class CurveFitting {
public:
    CurveFitting();
    ~CurveFitting();

    /// @param cloud Input point cloud.
    /// @return Fitted B-spline polyline, or nullptr on failure.
    static ccPolyline *BsplineFitting(const ccPointCloud &cloud);

    /// @param input_cloud Input point cloud.
    void setInputcloud(PointCloudT::Ptr input_cloud);
    /// @param x_resolution Grid resolution in x.
    /// @param y_resolution Grid resolution in y.
    /// @param x_mean Output x mean values.
    /// @param y_mean Output y mean values.
    /// @param z_mean Output z mean values.
    /// @param new_cloud Output gridded cloud.
    void grid_mean_xyz(double x_resolution,
                       double y_resolution,
                       std::vector<double> &x_mean,
                       std::vector<double> &y_mean,
                       std::vector<double> &z_mean,
                       PointCloudT::Ptr &new_cloud);
    /// @param x Input x coordinates.
    /// @param y Input y coordinates.
    /// @param k Output slope (y=kx+b).
    /// @param b Output intercept.
    void line_fitting(std::vector<double> x,
                      std::vector<double> y,
                      double &k,
                      double &b);  // y=kx+b
    /// @param x Input x coordinates.
    /// @param y Input y coordinates.
    /// @param a Output coefficient (y=a*x^2+b*x+c).
    /// @param b Output coefficient.
    /// @param c Output coefficient.
    void polynomial2D_fitting(std::vector<double> x,
                              std::vector<double> y,
                              double &a,
                              double &b,
                              double &c);  // y=a*x^2+b*x+c;
    /// @param x Input x coordinates.
    /// @param y Input y coordinates.
    /// @param z Input z coordinates.
    /// @param a Output coefficient (z=a*(x^2+y^2)+b*sqrt(x^2+y^2)+c).
    /// @param b Output coefficient.
    /// @param c Output coefficient.
    void polynomial3D_fitting(std::vector<double> x,
                              std::vector<double> y,
                              std::vector<double> z,
                              double &a,
                              double &b,
                              double &c);  // z=a*(x^2+y^2)+b*sqrt(x^2+y^2)+c
    /// @param outCurve Output curve point cloud.
    /// @param step_ Sampling step for curve points.
    void getPolynomial3D(PointCloudT::Ptr outCurve, double step_ = 0.5);

private:
    PointCloudT::Ptr cloud;
    PointT point_min;
    PointT point_max;
    double a_3d;
    double b_3d;
    double c_3d;
    double k_line;
    double b_line;
};
}  // namespace CurveFittingTool
