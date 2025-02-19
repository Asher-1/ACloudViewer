// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io -
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

#include <fmt/format.h>

#include <Eigen/Core>

#include "CVCoreLib.h"
#include "Eigen.h"

/// @cond
namespace Json {
class Value;
}  // namespace Json
/// @endcond

namespace cloudViewer {
namespace utility {

/// \brief Parse string and conver to Json::value. Throws exception if the
/// conversion is invalid.
///
/// \param json_str String containing json value.
/// \return A Json object.
Json::Value CV_CORE_LIB_API StringToJson(const std::string &json_str);

/// \brief Serialize a Json::Value to a string.
///
/// \param json The Json::Value object to be converted.
/// \return A string containing the json value.
std::string CV_CORE_LIB_API JsonToString(const Json::Value &json);

/// Class IJsonConvertible defines the behavior of a class that can convert
/// itself to/from a json::Value.
class CV_CORE_LIB_API IJsonConvertible {
public:
    virtual ~IJsonConvertible() {}

public:
    virtual bool ConvertToJsonValue(Json::Value &value) const = 0;
    virtual bool ConvertFromJsonValue(const Json::Value &value) = 0;

    /// Convert to a styled string representation of JSON data for display
    virtual std::string ToString() const;

public:
    static bool EigenVector3dFromJsonArray(Eigen::Vector3d &vec,
                                           const Json::Value &value);
    static bool EigenVector3dToJsonArray(const Eigen::Vector3d &vec,
                                         Json::Value &value);
    static bool EigenVector4dFromJsonArray(Eigen::Vector4d &vec,
                                           const Json::Value &value);
    static bool EigenVector4dToJsonArray(const Eigen::Vector4d &vec,
                                         Json::Value &value);
    static bool EigenMatrix3dFromJsonArray(Eigen::Matrix3d &mat,
                                           const Json::Value &value);
    static bool EigenMatrix3dToJsonArray(const Eigen::Matrix3d &mat,
                                         Json::Value &value);
    static bool EigenMatrix4dFromJsonArray(Eigen::Matrix4d &mat,
                                           const Json::Value &value);
    static bool EigenMatrix4dToJsonArray(const Eigen::Matrix4d &mat,
                                         Json::Value &value);
    static bool EigenMatrix4dFromJsonArray(Eigen::Matrix4d_u &mat,
                                           const Json::Value &value);
    static bool EigenMatrix4dToJsonArray(const Eigen::Matrix4d_u &mat,
                                         Json::Value &value);
    static bool EigenMatrix6dFromJsonArray(Eigen::Matrix6d &mat,
                                           const Json::Value &value);
    static bool EigenMatrix6dToJsonArray(const Eigen::Matrix6d &mat,
                                         Json::Value &value);
    static bool EigenMatrix6dFromJsonArray(Eigen::Matrix6d_u &mat,
                                           const Json::Value &value);
    static bool EigenMatrix6dToJsonArray(const Eigen::Matrix6d_u &mat,
                                         Json::Value &value);
};

}  // namespace utility
}  // namespace cloudViewer

namespace fmt {
template <>
struct formatter<Json::Value> {
    template <typename FormatContext>
    auto format(const Json::Value &value, FormatContext &ctx) const
            -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}",
                         cloudViewer::utility::JsonToString(value));
    }

    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
};

}  // namespace fmt
