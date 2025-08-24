// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
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

std::shared_ptr<Feature> Feature::SelectByIndex(
        const std::vector<size_t> &indices, bool invert /* = false */) const {
    auto output = std::make_shared<Feature>();

    std::vector<bool> mask = std::vector<bool>(data_.cols(), invert);
    size_t n_output_features = 0;
    for (size_t i : indices) {
        if (i < mask.size()) {
            if (mask[i] == invert) {
                mask[i] = !invert;
                n_output_features++;
            }
        } else {
            utility::LogWarning(
                    "[SelectByIndex] contains index {} that is "
                    "not within the bounds",
                    (int)i);
        }
    }
    if (invert) {
        n_output_features = data_.cols() - n_output_features;
    }

    output->Resize(data_.rows(), n_output_features);

    for (size_t i = 0, current_col_feature = 0;
         i < static_cast<size_t>(data_.cols()); i++) {
        if (mask[i]) {
            output->data_.col(current_col_feature++) = data_.col(i);
        }
    }

    utility::LogDebug(
            "[SelectByIndex] Feature group down sampled from {:d} features to "
            "{:d} features.",
            (int)data_.cols(), (int)output->data_.cols());

    return output;
}

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

CorrespondenceSet CorrespondencesFromFeatures(const Feature &source_features,
                                              const Feature &target_features,
                                              bool mutual_filter,
                                              float mutual_consistent_ratio) {
    const int num_searches = mutual_filter ? 2 : 1;

    // Access by reference, since Eigen Matrix could be copied
    std::array<std::reference_wrapper<const Feature>, 2> features{
            std::reference_wrapper<const Feature>(source_features),
            std::reference_wrapper<const Feature>(target_features)};
    std::array<int, 2> num_pts{int(source_features.data_.cols()),
                               int(target_features.data_.cols())};
    std::vector<CorrespondenceSet> corres(num_searches);

    const int kMaxThreads = utility::EstimateMaxThreads();
    const int kOuterThreads = std::min(kMaxThreads, num_searches);
    const int kInnerThreads = std::max(kMaxThreads / num_searches, 1);
    (void)kOuterThreads;  // Avoids compiler warning if OpenMP is disabled
    (void)kInnerThreads;
#pragma omp parallel for num_threads(kOuterThreads)
    for (int k = 0; k < num_searches; ++k) {
        geometry::KDTreeFlann kdtree(features[1 - k]);

        int num_pts_k = num_pts[k];
        corres[k] = CorrespondenceSet(num_pts_k);
#pragma omp parallel for num_threads(kInnerThreads)
        for (int i = 0; i < num_pts_k; i++) {
            std::vector<int> corres_tmp(1);
            std::vector<double> dist_tmp(1);

            kdtree.SearchKNN(Eigen::VectorXd(features[k].get().data_.col(i)), 1,
                             corres_tmp, dist_tmp);
            int j = corres_tmp[0];
            corres[k][i] = Eigen::Vector2i(i, j);
        }
    }

    // corres[0]: corres_ij, corres[1]: corres_ji
    if (!mutual_filter) return corres[0];

    // should not use parallel for due to emplace back
    CorrespondenceSet corres_mutual;
    int num_src_pts = num_pts[0];
    for (int i = 0; i < num_src_pts; ++i) {
        int j = corres[0][i](1);
        if (corres[1][j](1) == i) {
            corres_mutual.emplace_back(i, j);
        }
    }

    // Empirically mutual correspondence set should not be too small
    if (int(corres_mutual.size()) >=
        int(mutual_consistent_ratio * num_src_pts)) {
        utility::LogDebug("{:d} correspondences remain after mutual filter",
                          corres_mutual.size());
        return corres_mutual;
    }
    utility::LogWarning(
            "Too few correspondences ({:d}) after mutual filter, fall back to "
            "original correspondences.",
            corres_mutual.size());
    return corres[0];
}

}  // namespace utility
}  // namespace cloudViewer
