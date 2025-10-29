// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Core>
#include <vector>

namespace fgr {

typedef std::vector<Eigen::Vector3f> Points;
typedef std::vector<Eigen::VectorXf> Features;
typedef std::pair<int, int> Pair;
typedef std::vector<Pair> Correspondences;

class CApp {
public:
    CApp(double div_factor =
                 1.4,  // Division factor used for graduated non-convexity
         bool use_absolute_scale =
                 false,  // Measure distance in absolute scale (true) or in
                         // scale relative to the diameter of the model (false)
         double max_corr_dist = 0.025,  // Maximum correspondence distance (also
                                        // see comment of USE_ABSOLUTE_SCALE)
         int iteration_number = 64,     // Maximum number of iteration
         float tuple_scale =
                 0.95f,  // Similarity measure used for tuples of feature points
         size_t tuple_max_cnt = 1000)  // Maximum tuple numbers
        : div_factor_(div_factor),
          use_absolute_scale_(use_absolute_scale),
          max_corr_dist_(max_corr_dist),
          iteration_number_(iteration_number),
          tuple_scale_(std::min(1.0f, tuple_scale)),
          tuple_max_cnt_(tuple_max_cnt) {}

    // Reference must be loaded first
    void LoadFeature(const Points& pts, const Features& feat);

    // Normalize scale of points.
    // X' = (X-\mu)/scale
    void NormalizePoints();

    void AdvancedMatching(bool crossCheck = true, bool tupleConstraint = true);

    bool OptimizePairwise(bool decrease_mu = true);

    Eigen::Matrix4f GetOutputTrans() const;

private:
    // containers
    std::vector<Points> pointClouds_;
    std::vector<Features> features_;
    Eigen::Matrix4f transOutput_;
    Correspondences correspondances_;

    // for normalization
    Eigen::Vector3f means_[2];
    float globalScale_ = 1.0f;
    float startScale_ = 1.0f;

    double div_factor_ = 0.0;
    bool use_absolute_scale_ = false;
    double max_corr_dist_ = 0.0;
    int iteration_number_ = 0;
    float tuple_scale_ = 0.0f;
    size_t tuple_max_cnt_ = 0;
};

}  // namespace fgr
