// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "CV_io.h"

// CV_CORE_LIB
#include <IJsonConvertible.h>

// SYSTEM
#include <string>

namespace cloudViewer {
namespace io {

/// The general entrance for reading an IJsonConvertible from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool CV_IO_LIB_API
ReadIJsonConvertible(const std::string &filename,
                     cloudViewer::utility::IJsonConvertible &object);

/// The general entrance for writing an IJsonConvertible to a file
/// The function calls write functions based on the extension name of filename.
/// \return return true if the write function is successful, false otherwise.
bool CV_IO_LIB_API
WriteIJsonConvertible(const std::string &filename,
                      const cloudViewer::utility::IJsonConvertible &object);

bool CV_IO_LIB_API
ReadIJsonConvertibleFromJSON(const std::string &filename,
                             cloudViewer::utility::IJsonConvertible &object);

bool CV_IO_LIB_API WriteIJsonConvertibleToJSON(
        const std::string &filename,
        const cloudViewer::utility::IJsonConvertible &object);

bool CV_IO_LIB_API ReadIJsonConvertibleFromJSONString(
        const std::string &json_string,
        cloudViewer::utility::IJsonConvertible &object);

bool CV_IO_LIB_API WriteIJsonConvertibleToJSONString(
        std::string &json_string,
        const cloudViewer::utility::IJsonConvertible &object);

/// String to and from enum mapping, based on
/// https://github.com/nlohmann/json/blob/master/include/nlohmann/detail/macro_scope.hpp
/// (MIT license)
/// If you have an enum:
/// enum IMAGE_FORMAT {FORMAT_PNG,  FORMAT_JPG};
/// Use as STRINGIFY_ENUM(IMAGE_FORMAT, {
///      {FORMAT_INVALID, nullptr},
///      {FORMAT_PNG, "png"},
///      {FORMAT_JPG, "jpg"}
///      })
/// in the cpp file and
/// DECLARE_STRINGIFY_ENUM(IMAGE_FORMAT)
/// in the header file. This creates the functions
/// - enum_to_string(const ENUM_TYPE &e) -> std::string
/// - enum_from_string(const std::string &str, ENUM_TYPE &e) -> void
/// for conversion between the enum and string. Invalid string values are mapped
/// to the first specified option in the macro.
#define DECLARE_STRINGIFY_ENUM(ENUM_TYPE)                        \
    std::string enum_to_string(ENUM_TYPE e);                     \
    void enum_from_string(const std::string &str, ENUM_TYPE &e); \
    inline auto format_as(ENUM_TYPE e) { return enum_to_string(e); }

#define STRINGIFY_ENUM(ENUM_TYPE, ...)                                      \
    std::string enum_to_string(ENUM_TYPE e) {                               \
        static_assert(std::is_enum<ENUM_TYPE>::value,                       \
                      #ENUM_TYPE " must be an enum!");                      \
        static const std::pair<ENUM_TYPE, std::string> m[] = __VA_ARGS__;   \
        auto it = std::find_if(                                             \
                std::begin(m), std::end(m),                                 \
                [e](const std::pair<ENUM_TYPE, std::string> &es_pair)       \
                        -> bool { return es_pair.first == e; });            \
        return ((it != std::end(m)) ? it : std::begin(m))->second;          \
    }                                                                       \
    void enum_from_string(const std::string &str, ENUM_TYPE &e) {           \
        static_assert(std::is_enum<ENUM_TYPE>::value,                       \
                      #ENUM_TYPE " must be an enum!");                      \
        static const std::pair<ENUM_TYPE, std::string> m[] = __VA_ARGS__;   \
        auto it = std::find_if(                                             \
                std::begin(m), std::end(m),                                 \
                [&str](const std::pair<ENUM_TYPE, std::string> &es_pair)    \
                        -> bool { return es_pair.second == str; });         \
        e = ((it != std::end(m)) ? it : std::begin(m))->first;              \
        cloudViewer::utility::LogDebug("{} -> {}", str, enum_to_string(e)); \
    }

}  // namespace io
}  // namespace cloudViewer
