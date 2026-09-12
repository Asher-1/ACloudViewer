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
#include "util/logging.h"
#include "util/string.h"

namespace colmap {

#ifndef STRINGIFY
#define STRINGIFY(s) STRINGIFY_(s)
#define STRINGIFY_(s) #s
#endif  // STRINGIFY

enum class CopyType { COPY, HARD_LINK, SOFT_LINK };

// Append trailing slash to string if it does not yet end with a slash.
std::string EnsureTrailingSlash(const std::string& str);

// Check whether file name has the file extension (case insensitive).
bool HasFileExtension(const std::filesystem::path& file_name,
                      const std::string& ext);

// Add a file extension, e.g., "file" + ".jpg" -> "file.jpg".
std::filesystem::path AddFileExtension(std::filesystem::path path,
                                       const std::string& ext);

// Split the path into its root and extension, for example,
// "dir/file.jpg" into "dir/file" and ".jpg".
void SplitFileExtension(const std::string& path,
                        std::string* root,
                        std::string* ext);

// Copy or link file from source to destination path
void FileCopy(const std::filesystem::path& src_path,
              const std::filesystem::path& dst_path,
              CopyType type = CopyType::COPY);

// Check if the path points to an existing directory.
bool ExistsFile(const std::filesystem::path& path);

// Check if the path points to an existing directory.
bool ExistsDir(const std::filesystem::path& path);

// Check if the path points to an existing file or directory.
bool ExistsPath(const std::filesystem::path& path);

// Create the directory if it does not exist. If "recursive" is true, all
// missing parent directories are created as well (upstream parity).
void CreateDirIfNotExists(const std::filesystem::path& path,
                          bool recursive = false);

// Upstream parity (dbb41680 util/file.h): check that an open file stream
// is usable, with a message that points at the path.
#define THROW_CHECK_FILE_OPEN(file, path)  \
    THROW_CHECK((file).is_open())          \
            << "Could not open " << (path) \
            << ". Is the path a directory or does the parent dir not exist?";

// Extract the base name of a path, e.g., "image.jpg" for "/dir/image.jpg".
std::string GetPathBaseName(const std::filesystem::path& path);

// Get the path of the parent directory for the given path.
std::string GetParentDir(const std::filesystem::path& path);

// Get the relative path between from and to. Both the from and to paths must
// exist.
std::string GetRelativePath(const std::filesystem::path& from,
                            const std::filesystem::path& to);

// Join multiple paths into one path.
template <typename... T>
std::string JoinPaths(T const&... paths);

// Return list of files in directory.
std::vector<std::string> GetFileList(const std::filesystem::path& path);

// Return list of files, recursively in all sub-directories.
std::vector<std::string> GetRecursiveFileList(
        const std::filesystem::path& path);

// Return list of directories, recursively in all sub-directories.
std::vector<std::string> GetDirList(const std::filesystem::path& path);

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

// Read each line of a text file into a separate element. Empty lines are
// ignored and leading/trailing whitespace is removed.
std::vector<std::string> ReadTextFileLines(const std::filesystem::path& path);

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
