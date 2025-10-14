// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QPCL_CURVEFITTING_HEADER
#define QPCL_CURVEFITTING_HEADER

// LOCAL
#include "PclUtils/PCLCloud.h"
#include "PclUtils/PCLConv.h"
#include "qPCL.h"

// CV_CORE_LIB
#include <CVConst.h>
#include <CVLog.h>

#include <vector>

class ccPolyline;
class ccPointCloud;

namespace CurveFittingTool {

class QPCL_ENGINE_LIB_API CurveFitting {
public:
    CurveFitting();
    ~CurveFitting();

    static ccPolyline *BsplineFitting(const ccPointCloud &cloud);

    void setInputcloud(PointCloudT::Ptr input_cloud);
    void grid_mean_xyz(double x_resolution,
                       double y_resolution,
                       std::vector<double> &x_mean,
                       std::vector<double> &y_mean,
                       std::vector<double> &z_mean,
                       PointCloudT::Ptr &new_cloud);
    void line_fitting(std::vector<double> x,
                      std::vector<double> y,
                      double &k,
                      double &b);  // y=kx+b
    void polynomial2D_fitting(std::vector<double> x,
                              std::vector<double> y,
                              double &a,
                              double &b,
                              double &c);  // y=a*x^2+b*x+c;
    void polynomial3D_fitting(std::vector<double> x,
                              std::vector<double> y,
                              std::vector<double> z,
                              double &a,
                              double &b,
                              double &c);  // z=a*(x^2+y^2)+b*sqrt(x^2+y^2)+c
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

#endif  // QPCL_CURVEFITTING_HEADER
