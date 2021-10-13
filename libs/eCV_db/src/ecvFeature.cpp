// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

#include "ecvFeature.h"

#include <Logging.h>
#include <Parallel.h>

#include "ecvKDTreeFlann.h"
#include "ecvPointCloud.h"

// EIGEN
#include <Eigen/Dense>

namespace cloudViewer {

namespace {
using namespace utility;

Eigen::Vector4d ComputePairFeatures(const Eigen::Vector3d &p1,
                                    const Eigen::Vector3d &n1,
                                    const Eigen::Vector3d &p2,
                                    const Eigen::Vector3d &n2) {
    Eigen::Vector4d result;
    Eigen::Vector3d dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) {
        return Eigen::Vector4d::Zero();
    }
    auto n1_copy = n1;
    auto n2_copy = n2;
    double angle1 = n1_copy.dot(dp2p1) / result(3);
    double angle2 = n2_copy.dot(dp2p1) / result(3);
    if (acos(fabs(angle1)) > acos(fabs(angle2))) {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 *= -1.0;
        result(2) = -angle2;
    } else {
        result(2) = angle1;
    }
    auto v = dp2p1.cross(n1_copy);
    double v_norm = v.norm();
    if (v_norm == 0.0) {
        return Eigen::Vector4d::Zero();
    }
    v /= v_norm;
    auto w = n1_copy.cross(v);
    result(1) = v.dot(n2_copy);
    result(0) = atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
    return result;
}

std::shared_ptr<Feature> ComputeSPFHFeature(
        const ccPointCloud &input,
        const geometry::KDTreeFlann &kdtree,
        const geometry::KDTreeSearchParam &search_param) {
    auto feature = cloudViewer::make_shared<Feature>();
    feature->Resize(33, (int)input.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
    for (int i = 0; i < (int)input.size(); i++) {
        const auto &point = *input.getPoint(static_cast<unsigned int>(i));
        const auto &normal = input.getPointNormal(i);
        std::vector<int> indices;
        std::vector<double> distance2;
        if (kdtree.Search(CCVector3d::fromArray(point), search_param, indices,
                          distance2) > 1) {
            // only compute SPFH feature when a point has neighbors
            double hist_incr = 100.0 / (double)(indices.size() - 1);
            for (size_t k = 1; k < indices.size(); k++) {
                // skip the point itself, compute histogram
                auto pf = ComputePairFeatures(
                        CCVector3d::fromArray(point),
                        CCVector3d::fromArray(normal),
                        CCVector3d::fromArray(*input.getPoint(indices[k])),
                        CCVector3d::fromArray(
                                input.getPointNormal(indices[k])));
                int h_index = (int)(floor(11 * (pf(0) + M_PI) / (2.0 * M_PI)));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index, i) += hist_incr;
                h_index = (int)(floor(11 * (pf(1) + 1.0) * 0.5));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index + 11, i) += hist_incr;
                h_index = (int)(floor(11 * (pf(2) + 1.0) * 0.5));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index + 22, i) += hist_incr;
            }
        }
    }
    return feature;
}

}  // unnamed namespace

namespace utility {
std::shared_ptr<Feature> ComputeFPFHFeature(
        const ccPointCloud &input,
        const geometry::KDTreeSearchParam
                &search_param /* = geometry::KDTreeSearchParamKNN()*/) {
    auto feature = cloudViewer::make_shared<Feature>();
    feature->Resize(33, (int)input.size());
    if (!input.hasNormals()) {
        utility::LogError(
                "[ComputeFPFHFeature] Failed because input point cloud has no "
                "normal.");
    }
    geometry::KDTreeFlann kdtree(input);
    auto spfh = ComputeSPFHFeature(input, kdtree, search_param);
    if (spfh == nullptr) {
        utility::LogError("Internal error: SPFH feature is nullptr.");
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
    for (int i = 0; i < (int)input.size(); i++) {
        std::vector<int> indices;
        std::vector<double> distance2;
        if (kdtree.Search(input.getEigenPoint(static_cast<size_t>(i)),
                          search_param, indices, distance2) > 1) {
            double sum[3] = {0.0, 0.0, 0.0};
            for (size_t k = 1; k < indices.size(); k++) {
                // skip the point itself
                double dist = distance2[k];
                if (dist == 0.0) continue;
                for (int j = 0; j < 33; j++) {
                    double val = spfh->data_(j, indices[k]) / dist;
                    sum[j / 11] += val;
                    feature->data_(j, i) += val;
                }
            }
            for (int j = 0; j < 3; j++)
                if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
            for (int j = 0; j < 33; j++) {
                feature->data_(j, i) *= sum[j / 11];
                // The commented line is the fpfh function in the paper.
                // But according to PCL implementation, it is skipped.
                // Our initial test shows that the full fpfh function in the
                // paper seems to be better than PCL implementation. Further
                // test required.
                feature->data_(j, i) += spfh->data_(j, i);
            }
        }
    }
    return feature;
}

}  // namespace utility
}  // namespace cloudViewer
