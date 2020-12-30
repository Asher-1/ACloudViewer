// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#include <thread>

#include <Console.h>
#include <FileSystem.h>
#include <IJsonConvertible.h>

#include <ecvBBox.h>
#include <ecvOrientedBBox.h>
#include <ecvPointCloud.h>
#include "io/PointCloudIO.h"
#include <ecvKDTreeFlann.h>

namespace cloudViewer {

class AlignmentSession : public CVLib::utility::IJsonConvertible {
public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    std::shared_ptr<ccPointCloud> source_ptr_;  // Original source pointcloud
    std::shared_ptr<ccPointCloud> target_ptr_;  // Original target pointcloud
    std::vector<size_t> source_indices_;  // Manually annotated point indices
    std::vector<size_t> target_indices_;  // Manually annotated point indices
    Eigen::Matrix4d_u transformation_;    // Current alignment result
    double voxel_size_ = -1.0;
    double max_correspondence_distance_ = -1.0;
    bool with_scaling_ = true;
};

}  // namespace cloudViewer
