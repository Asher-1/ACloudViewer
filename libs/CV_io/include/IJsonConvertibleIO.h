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

/**
 * @file IJsonConvertibleIO.h
 * @brief JSON serialization I/O utilities
 *
 * Provides functions for reading and writing objects that implement the
 * IJsonConvertible interface, enabling JSON-based serialization and
 * deserialization. Also includes utility macros for enum-string conversion.
 */

namespace cloudViewer {
namespace io {

/**
 * @brief Read JSON-convertible object from file (general entrance)
 *
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input file path
 * @param object Output object implementing IJsonConvertible
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
ReadIJsonConvertible(const std::string &filename,
                     cloudViewer::utility::IJsonConvertible &object);

/**
 * @brief Write JSON-convertible object to file (general entrance)
 *
 * Automatically selects appropriate writer based on file extension.
 * @param filename Output file path
 * @param object Object to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
WriteIJsonConvertible(const std::string &filename,
                      const cloudViewer::utility::IJsonConvertible &object);

/**
 * @brief Read JSON-convertible object from JSON file
 * @param filename Input JSON file path
 * @param object Output object implementing IJsonConvertible
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
ReadIJsonConvertibleFromJSON(const std::string &filename,
                             cloudViewer::utility::IJsonConvertible &object);

/**
 * @brief Write JSON-convertible object to JSON file
 * @param filename Output JSON file path
 * @param object Object to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteIJsonConvertibleToJSON(
        const std::string &filename,
        const cloudViewer::utility::IJsonConvertible &object);

/**
 * @brief Read JSON-convertible object from JSON string
 * @param json_string Input JSON string
 * @param object Output object implementing IJsonConvertible
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadIJsonConvertibleFromJSONString(
        const std::string &json_string,
        cloudViewer::utility::IJsonConvertible &object);

/**
 * @brief Write JSON-convertible object to JSON string
 * @param json_string Output JSON string
 * @param object Object to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteIJsonConvertibleToJSONString(
        std::string &json_string,
        const cloudViewer::utility::IJsonConvertible &object);

/**
 * @defgroup EnumStringConversion Enum-String Conversion Macros
 * @brief Macros for bidirectional enum-string conversion
 *
 * Based on nlohmann/json macro implementation (MIT license).
 * @see
 * https://github.com/nlohmann/json/blob/master/include/nlohmann/detail/macro_scope.hpp
 *
 * Usage example:
 * @code
 * // In header file:
 * enum IMAGE_FORMAT { FORMAT_INVALID, FORMAT_PNG, FORMAT_JPG };
 * DECLARE_STRINGIFY_ENUM(IMAGE_FORMAT)
 *
 * // In cpp file:
 * STRINGIFY_ENUM(IMAGE_FORMAT, {
 *     {FORMAT_INVALID, nullptr},
 *     {FORMAT_PNG, "png"},
 *     {FORMAT_JPG, "jpg"}
 * })
 * @endcode
 *
 * This creates two functions:
 * - `enum_to_string(const ENUM_TYPE &e) -> std::string`
 * - `enum_from_string(const std::string &str, ENUM_TYPE &e) -> void`
 *
 * Invalid string values are mapped to the first specified enum value.
 * @{
 */
/**
 * @brief Declare enum-to-string conversion functions in header
 *
 * Use this macro in header files to declare conversion functions.
 * Must be paired with STRINGIFY_ENUM in the implementation file.
 * @param ENUM_TYPE The enum type to declare conversions for
 */
#define DECLARE_STRINGIFY_ENUM(ENUM_TYPE)                        \
    std::string enum_to_string(ENUM_TYPE e);                     \
    void enum_from_string(const std::string &str, ENUM_TYPE &e); \
    inline auto format_as(ENUM_TYPE e) { return enum_to_string(e); }

/**
 * @brief Define enum-to-string conversion functions in implementation
 *
 * Use this macro in cpp files to implement conversion functions.
 * Must be paired with DECLARE_STRINGIFY_ENUM in the header file.
 * @param ENUM_TYPE The enum type to define conversions for
 * @param ... Initializer list of {enum_value, "string"} pairs
 */
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

/** @} */  // end of EnumStringConversion group

}  // namespace io
}  // namespace cloudViewer
