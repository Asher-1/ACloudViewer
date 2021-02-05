// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
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

#include "t/geometry/Image.h"

#include "core/Dtype.h"
#include "core/ShapeUtil.h"
#include "core/Tensor.h"

#include <Image.h>
#include <Console.h>

namespace cloudViewer {
namespace t {
namespace geometry {
using namespace cloudViewer;

Image::Image(int64_t rows,
             int64_t cols,
             int64_t channels,
             core::Dtype dtype,
             const core::Device &device)
    : Geometry(Geometry::GeometryType::Image, 2) {
    if (rows < 0) {
        utility::LogError("rows must be >= 0, but got {}.", rows);
    }
    if (cols < 0) {
        utility::LogError("cols must be >= 0, but got {}.", cols);
    }
    if (channels <= 0) {
        utility::LogError("channels must be > 0, but got {}.", channels);
    }
    data_ = core::Tensor({rows, cols, channels}, dtype, device);
}

Image::Image(const core::Tensor &tensor)
    : Geometry(Geometry::GeometryType::Image, 2) {
    if (!tensor.IsContiguous()) {
        utility::LogError("Input tensor must be contiguous.");
    }
    if (tensor.NumDims() == 2) {
        data_ = tensor.Reshape(
                core::shape_util::Concat(tensor.GetShape(), {1}));
    } else if (tensor.NumDims() == 3) {
        data_ = tensor;
    } else {
        utility::LogError("Input tensor must be 2-D or 3-D, but got shape {}.",
                          tensor.GetShape().ToString());
    }
}

Image Image::FromLegacyImage(const cloudViewer::geometry::Image &image_legacy,
                             const core::Device &device) {
    static const std::unordered_map<int, core::Dtype> kBytesToDtypeMap = {
            {1, core::Dtype::UInt8},
            {2, core::Dtype::UInt16},
            {4, core::Dtype::Float32},
    };

    if (image_legacy.isEmpty()) {
        return Image(0, 0, 1, core::Dtype::Float32, device);
    }

    auto iter = kBytesToDtypeMap.find(image_legacy.bytes_per_channel_);
    if (iter == kBytesToDtypeMap.end()) {
        utility::LogError("[Image] unsupported image bytes_per_channel ({})",
                          image_legacy.bytes_per_channel_);
    }

    core::Dtype dtype = iter->second;

    Image image(image_legacy.height_, image_legacy.width_,
                image_legacy.num_of_channels_, dtype, device);

    size_t num_bytes = image_legacy.height_ * image_legacy.BytesPerLine();
    core::MemoryManager::MemcpyFromHost(image.data_.GetDataPtr(), device,
                                        image_legacy.data_.data(), num_bytes);
    return image;
}

cloudViewer::geometry::Image Image::ToLegacyImage() const {
    auto dtype = GetDtype();
    if (!(dtype == core::Dtype::UInt8 || dtype == core::Dtype::UInt16 ||
          dtype == core::Dtype::Float32))
        utility::LogError("Legacy image does not support data type {}.",
                          dtype.ToString());
    if (!data_.IsContiguous()) {
        utility::LogError("Image tensor must be contiguous.");
    }
    cloudViewer::geometry::Image image_legacy;
    image_legacy.Prepare(static_cast<int>(GetCols()),
                         static_cast<int>(GetRows()),
                         static_cast<int>(GetChannels()),
                         static_cast<int>(dtype.ByteSize()));
    size_t num_bytes = image_legacy.height_ * image_legacy.BytesPerLine();
    core::MemoryManager::MemcpyToHost(image_legacy.data_.data(),
                                      data_.GetDataPtr(), data_.GetDevice(),
                                      num_bytes);
    return image_legacy;
}

std::string Image::ToString() const {
    return fmt::format("Image[size={{{},{}}}, channels={}, {}, {}]", GetRows(),
                       GetCols(), GetChannels(), GetDtype().ToString(),
                       GetDevice().ToString());
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
