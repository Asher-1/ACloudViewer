// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace colmap {

enum class CopyType { COPY, HARD_LINK, SOFT_LINK };

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
std::filesystem::path GetParentDir(const std::filesystem::path& path);

// Get the relative path between from and to. Both the from and to paths must
// exist.
std::string GetRelativePath(const std::filesystem::path& from,
                            const std::filesystem::path& to);

// Normalize the path by removing repeated separators and dots and, on Windows,
// replacing \\ separators by / (upstream parity).
std::string NormalizePath(const std::filesystem::path& path);

// Get the normalized relative path of full_path w.r.t. base_path
// (upstream parity).
std::string GetNormalizedRelativePath(const std::filesystem::path& full_path,
                                      const std::filesystem::path& base_path);

// Return list of files, recursively in all sub-directories (upstream parity).
std::vector<std::filesystem::path> GetRecursiveFileList(
        const std::filesystem::path& path);

// Return list of directories in the given directory (upstream parity).
std::vector<std::filesystem::path> GetDirList(
        const std::filesystem::path& path);

// Gets current user's home directory from environment variables.
// Returns null if it cannot be resolved (upstream parity).
std::optional<std::filesystem::path> HomeDir();

// Read contiguous binary blob from file (upstream parity; the fork has no
// span type, so the write side takes a vector).
void ReadBinaryBlob(const std::filesystem::path& path, std::vector<char>* data);

// Write contiguous binary blob to file.
void WriteBinaryBlob(const std::filesystem::path& path,
                     const std::vector<char>& data);

// Read each line of a text file into a separate element. Empty lines are
// ignored and leading/trailing whitespace is removed (upstream parity).
std::vector<std::string> ReadTextFileLines(const std::filesystem::path& path);

// Detect if given string is a URI
// (i.e., starts with http://, https://, file://). Declared in
// util/download.h in this fork (the download surface lives there).

}  // namespace colmap
