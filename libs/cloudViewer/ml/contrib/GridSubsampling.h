// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdint>
#include <set>

#include "ml/contrib/Cloud.h"

namespace cloudViewer {
namespace ml {
namespace contrib {

class SampledData {
public:
    // Elements
    // ********

    int count;
    PointXYZ point;
    std::vector<float> features;
    std::vector<std::unordered_map<int, int>> labels;

    // Methods
    // *******

    // Constructor
    SampledData() {
        count = 0;
        point = PointXYZ();
    }

    SampledData(const size_t fdim, const size_t ldim) {
        count = 0;
        point = PointXYZ();
        features = std::vector<float>(fdim);
        labels = std::vector<std::unordered_map<int, int>>(ldim);
    }

    // Method Update
    void update_all(const PointXYZ p,
                    std::vector<float>::iterator f_begin,
                    std::vector<int>::iterator l_begin) {
        count += 1;
        point += p;
        transform(features.begin(), features.end(), f_begin, features.begin(),
                  std::plus<float>());
        int i = 0;
        for (std::vector<int>::iterator it = l_begin;
             it != l_begin + labels.size(); ++it) {
            labels[i][*it] += 1;
            i++;
        }
        return;
    }

    void update_features(const PointXYZ p,
                         std::vector<float>::iterator f_begin) {
        count += 1;
        point += p;
        transform(features.begin(), features.end(), f_begin, features.begin(),
                  std::plus<float>());
        return;
    }

    void update_classes(const PointXYZ p, std::vector<int>::iterator l_begin) {
        count += 1;
        point += p;
        int i = 0;
        for (std::vector<int>::iterator it = l_begin;
             it != l_begin + labels.size(); ++it) {
            labels[i][*it] += 1;
            i++;
        }
        return;
    }

    void update_points(const PointXYZ p) {
        count += 1;
        point += p;
        return;
    }
};

void grid_subsampling(std::vector<PointXYZ>& original_points,
                      std::vector<PointXYZ>& subsampled_points,
                      std::vector<float>& original_features,
                      std::vector<float>& subsampled_features,
                      std::vector<int>& original_classes,
                      std::vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose);

void batch_grid_subsampling(std::vector<PointXYZ>& original_points,
                            std::vector<PointXYZ>& subsampled_points,
                            std::vector<float>& original_features,
                            std::vector<float>& subsampled_features,
                            std::vector<int>& original_classes,
                            std::vector<int>& subsampled_classes,
                            std::vector<int>& original_batches,
                            std::vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p);

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
