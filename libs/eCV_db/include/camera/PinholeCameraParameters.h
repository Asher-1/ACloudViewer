// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
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

#pragma once

#include "eCV_db.h"

#include <memory>
#include <vector>

#include "camera/PinholeCameraIntrinsic.h"

namespace cloudViewer {
namespace camera {

/// \class PinholeCameraParameters
///
/// \brief Contains both intrinsic and extrinsic pinhole camera parameters.
class ECV_DB_LIB_API PinholeCameraParameters :
	public cloudViewer::utility::IJsonConvertible {
public:
    // Must comment it due to unreferenced symbols when linked
    // CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief Default Constructor.
    PinholeCameraParameters();
    virtual ~PinholeCameraParameters() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// PinholeCameraIntrinsic object.
    PinholeCameraIntrinsic intrinsic_;
    /// Camera extrinsic parameters.
    Eigen::Matrix4d_u extrinsic_;

    std::string texture_file_;
};
}  // namespace camera
}  // namespace cloudViewer
