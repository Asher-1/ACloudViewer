// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <boost/filesystem.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "util/endian.h"
#include "util/file.h"
#include "util/logging.h"
#include "util/string.h"

namespace colmap {

#ifndef STRINGIFY
#define STRINGIFY(s) STRINGIFY_(s)
#define STRINGIFY_(s) #s
#endif  // STRINGIFY

// Append trailing slash to string if it does not yet end with a slash.

// Join multiple paths into one path.
template <typename... T>
std::string JoinPaths(T const&... paths);

// Return list of files in directory.
std::vector<std::string> GetFileList(const std::filesystem::path& path);

// GetRecursiveFileList/GetDirList moved to util/file.h (upstream parity,
// returning std::filesystem::path); the recursive-directory variant below is
// a fork-only extension.
// Return list of directories, recursively in all sub-directories.
std::vector<std::string> GetRecursiveDirList(const std::filesystem::path& path);

// Get the size in bytes of a file.
size_t GetFileSize(const std::filesystem::path& path);

// Print first-order heading with over- and underscores to `std::cout`.
void PrintHeading1(const std::string& heading);

// Print second-order heading with underscores to `std::cout`.
void PrintHeading2(const std::string& heading);

// Check if vector contains elements.
template <typename T>
bool VectorContainsValue(const std::vector<T>& vector, const T value);

template <typename T>
bool VectorContainsDuplicateValues(const std::vector<T>& vector);

// Parse CSV line to a list of values.
template <typename T>
std::vector<T> CSVToVector(const std::string& csv);

// Concatenate values in list to comma-separated list.
template <typename T>
std::string VectorToCSV(const std::vector<T>& values);

// Read contiguous binary blob from file.
template <typename T>
void ReadBinaryBlob(const std::filesystem::path& path, std::vector<T>* data);

// Write contiguous binary blob to file.
template <typename T>
void WriteBinaryBlob(const std::filesystem::path& path,
                     const std::vector<T>& data);

// ReadTextFileLines moved to util/file.{h,cc} (upstream parity).

// Remove an argument from the list of command-line arguments.
void RemoveCommandLineArgument(const std::string& arg, int* argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::string JoinPathString(T const& path) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::filesystem::path>) {
        return path.string();
    } else {
        // Preserve the historical boost behavior: joining with an absolute
        // path argument appends rather than resets (std::filesystem differs).
        return boost::filesystem::path(path).string();
    }
}

template <typename... T>
std::string JoinPaths(T const&... paths) {
    boost::filesystem::path result;
    int unpack[]{0, (result = result /
                              boost::filesystem::path(JoinPathString(paths)),
                     0)...};
    static_cast<void>(unpack);
    return result.string();
}

template <typename T>
bool VectorContainsValue(const std::vector<T>& vector, const T value) {
    return std::find_if(vector.begin(), vector.end(), [value](const T element) {
               return element == value;
           }) != vector.end();
}

template <typename T>
bool VectorContainsDuplicateValues(const std::vector<T>& vector) {
    std::vector<T> unique_vector = vector;
    return std::unique(unique_vector.begin(), unique_vector.end()) !=
           unique_vector.end();
}

template <typename T>
std::string VectorToCSV(const std::vector<T>& values) {
    std::string string;
    for (const T value : values) {
        string += std::to_string(value) + ", ";
    }
    return string.substr(0, string.length() - 2);
}

template <typename T>
void ReadBinaryBlob(const std::filesystem::path& path, std::vector<T>* data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.is_open()) << path;
    file.seekg(0, std::ios::end);
    const size_t num_bytes = file.tellg();
    CHECK_EQ(num_bytes % sizeof(T), 0);
    data->resize(num_bytes / sizeof(T));
    file.seekg(0, std::ios::beg);
    ReadBinaryLittleEndian<T>(&file, data);
}

template <typename T>
void WriteBinaryBlob(const std::filesystem::path& path,
                     const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    WriteBinaryLittleEndian<T>(&file, data);
}

}  // namespace colmap
