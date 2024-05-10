// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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

#include <cstdio>
#include <vector>

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;

    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Flann [filename]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;
    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    auto new_cloud_ptr = std::make_shared<ccPointCloud>();
    if (io::ReadPointCloud(argv[1], *new_cloud_ptr)) {
        utility::LogInfo("Successfully read {}", argv[1]);
    } else {
        utility::LogWarning("Failed to read {}", argv[1]);
        return 1;
    }

    if ((int)new_cloud_ptr->size() < 100) {
        utility::LogWarning("Boring point cloud.");
        return 1;
    }

    if (!new_cloud_ptr->hasColors()) {
        new_cloud_ptr->resizeTheRGBTable();
    }

    int nn = std::min(20, (int)new_cloud_ptr->size() - 1);
    cloudViewer::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*new_cloud_ptr);
    std::vector<int> new_indices_vec(nn);
    std::vector<double> new_dists_vec(nn);
    kdtree.SearchKNN(new_cloud_ptr->getEigenPoint(0), nn, new_indices_vec,
                     new_dists_vec);

    for (size_t i = 0; i < new_indices_vec.size(); i++) {
        utility::LogInfo("{:d}, {:f}", (int)new_indices_vec[i],
                         sqrt(new_dists_vec[i]));
        new_cloud_ptr->setPointColor(
                static_cast<unsigned int>(new_indices_vec[i]),
                ecvColor::Rgb(255, 0, 0));
    }

    new_cloud_ptr->setPointColor(0, ecvColor::Rgb(0, 255, 0));

    float r = float(sqrt(new_dists_vec[nn - 1]) * 2.0);
    int k = kdtree.SearchRadius(new_cloud_ptr->getEigenPoint(99), r,
                                new_indices_vec, new_dists_vec);


    utility::LogInfo("======== {:d}, {:f} ========", k, r);
    for (int i = 0; i < k; i++) {
        utility::LogInfo("{:d}, {:f}", (int)new_indices_vec[i],
                         sqrt(new_dists_vec[i]));
        new_cloud_ptr->setPointColor(
                static_cast<unsigned int>(new_dists_vec[i]),
                ecvColor::Rgb(0, 0, 255));
    }
    new_cloud_ptr->setPointColor(99, ecvColor::Rgb(0, 255, 255));

    k = kdtree.Search(new_cloud_ptr->getEigenPoint(199),
                      geometry::KDTreeSearchParamRadius(r), new_indices_vec,
                      new_dists_vec);

    utility::LogInfo("======== {:d}, {:f} ========", k, r);
    for (int i = 0; i < k; i++) {
        utility::LogInfo("{:d}, {:f}", (int)new_indices_vec[i],
                         sqrt(new_dists_vec[i]));
        new_cloud_ptr->setPointColor(
                static_cast<unsigned int>(new_indices_vec[i]),
                ecvColor::Rgb(0, 0, 255));
    }
    new_cloud_ptr->setPointColor(199, ecvColor::Rgb(0, 255, 255));

    visualization::DrawGeometries({new_cloud_ptr}, "KDTreeFlann", 1600, 900);
    return 0;
}
