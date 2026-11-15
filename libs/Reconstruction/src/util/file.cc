// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "util/file.h"

#include <fstream>
#include <optional>

// glibc declares environ with C linkage via <unistd.h>; keep the
// declaration compatible regardless of the include order.
extern "C" char** environ;

#include "util/logging.h"
#include "util/string.h"

namespace colmap {

std::string EnsureTrailingSlash(const std::string& str) {
  if (str.length() > 0) {
    if (str.back() != '/') {
      return str + "/";
    }
  } else {
    return str + "/";
  }
  return str;
}

bool HasFileExtension(const std::filesystem::path& file_name,
                      const std::string& ext) {
  CHECK(!ext.empty());
  CHECK_EQ(ext.at(0), '.');
  std::string ext_lower = ext;
  StringToLower(&ext_lower);
  const std::string name = file_name.string();
  if (name.size() >= ext_lower.size()) {
    std::string suffix = name.substr(name.size() - ext_lower.size(),
                                     ext_lower.size());
    StringToLower(&suffix);
    if (suffix == ext_lower) return true;
  }
  return false;
}

std::filesystem::path AddFileExtension(std::filesystem::path path,
                                       const std::string& ext) {
  path += ext;
  return path;
}

void SplitFileExtension(const std::string& path, std::string* root,
                        std::string* ext) {
  const auto parts = StringSplit(path, ".");
  CHECK_GT(parts.size(), 0);
  if (parts.size() == 1) {
    *root = parts[0];
    *ext = "";
  } else {
    *root = "";
    for (size_t i = 0; i < parts.size() - 1; ++i) {
      *root += parts[i] + ".";
    }
    *root = root->substr(0, root->length() - 1);
    if (parts.back() == "") {
      *ext = "";
    } else {
      *ext = "." + parts.back();
    }
  }
}

void FileCopy(const std::filesystem::path& src_path,
              const std::filesystem::path& dst_path,
              CopyType type) {
  switch (type) {
    case CopyType::COPY:
      std::filesystem::copy_file(src_path, dst_path);
      break;
    case CopyType::HARD_LINK:
      std::filesystem::create_hard_link(src_path, dst_path);
      break;
    case CopyType::SOFT_LINK:
      std::filesystem::create_symlink(src_path, dst_path);
      break;
  }
}

bool ExistsFile(const std::filesystem::path& path) {
  return std::filesystem::is_regular_file(path);
}

bool ExistsDir(const std::filesystem::path& path) {
  return std::filesystem::is_directory(path);
}

bool ExistsPath(const std::filesystem::path& path) {
  return std::filesystem::exists(path);
}

void CreateDirIfNotExists(const std::filesystem::path& path,
                          bool recursive) {
  if (ExistsDir(path)) {
    return;
  }
  if (recursive) {
    THROW_CHECK(std::filesystem::create_directories(path))
            << "Could not create directory: " << path;
  } else {
    THROW_CHECK(std::filesystem::create_directory(path))
            << "Could not create directory: " << path;
  }
}

std::string GetPathBaseName(const std::filesystem::path& path) {
  // Upstream parity (dbb41680): normalize first, then take the filename of
  // the path (or of its parent for trailing-separator directories). Note the
  // legacy fork behavior additionally split on backslashes on POSIX; the
  // platform separator is authoritative per upstream.
  const std::filesystem::path fs_path(NormalizePath(path));
  if (fs_path.has_filename()) {
    return fs_path.filename().string();
  } else {  // It is a directory.
    return fs_path.parent_path().filename().string();
  }
}

std::filesystem::path GetParentDir(const std::filesystem::path& path) {
  // Preserve the historical behavior verified by misc_test: parent of "/"
  // is "" rather than "/" (std::filesystem differs on this edge case).
  if (path == "/") {
    return {};
  }
  return path.parent_path();
}

std::string NormalizePath(const std::filesystem::path& path) {
  std::string normalized_path = path.lexically_normal().string();
  if constexpr (std::filesystem::path::preferred_separator == '\\') {
    normalized_path = StringReplace(normalized_path, "\\", "/");
  }
  return normalized_path;
}

std::string GetNormalizedRelativePath(const std::filesystem::path& full_path,
                                      const std::filesystem::path& base_path) {
  return NormalizePath(full_path.lexically_proximate(base_path));
}

std::vector<std::filesystem::path> GetRecursiveFileList(
        const std::filesystem::path& path) {
  std::vector<std::filesystem::path> file_list;
  for (auto it = std::filesystem::recursive_directory_iterator(path);
       it != std::filesystem::recursive_directory_iterator();
       ++it) {
    if (std::filesystem::is_regular_file(*it)) {
      const std::filesystem::path file_path = *it;
      file_list.push_back(file_path);
    }
  }
  return file_list;
}

std::vector<std::filesystem::path> GetDirList(
        const std::filesystem::path& path) {
  std::vector<std::filesystem::path> dir_list;
  for (auto it = std::filesystem::directory_iterator(path);
       it != std::filesystem::directory_iterator();
       ++it) {
    if (std::filesystem::is_directory(*it)) {
      const std::filesystem::path dir_path = *it;
      dir_list.push_back(dir_path);
    }
  }
  return dir_list;
}

namespace {

// Non-MSVC replacement for std::getenv_s (upstream parity): iterates environ
// because std::getenv_s is not available on all platforms.
std::optional<std::string> GetEnvSafe(const char* key) {
#ifdef _MSC_VER
  size_t size = 0;
  getenv_s(&size, nullptr, 0, key);
  if (size == 0) {
    return std::nullopt;
  }
  std::string value(size, ' ');
  getenv_s(&size, value.data(), size, key);
  THROW_CHECK_EQ(value.back(), '\0');
  return value.substr(0, size - 1);
#else
  const std::string_view key_sv(key);
  for (char** env = environ; *env; ++env) {
    const std::string_view key_value(*env);
    if (key_sv.size() < key_value.size() &&
        key_value.substr(0, key_sv.size()) == key_sv &&
        key_value[key_sv.size()] == '=') {
      return std::string(key_value.substr(
          key_sv.size() + 1, key_value.size() - key_sv.size() - 1));
    }
  }
  return std::nullopt;
#endif
}

}  // namespace

std::optional<std::filesystem::path> HomeDir() {
#ifdef _MSC_VER
  std::optional<std::string> userprofile = GetEnvSafe("USERPROFILE");
  if (userprofile.has_value()) {
    return *userprofile;
  }
  const std::optional<std::string> homedrive = GetEnvSafe("HOMEDRIVE");
  const std::optional<std::string> homepath = GetEnvSafe("HOMEPATH");
  if (!homedrive.has_value() || !homepath.has_value()) {
    return std::nullopt;
  }
  return std::filesystem::path(*homedrive) / std::filesystem::path(*homepath);
#else
  std::optional<std::string> home = GetEnvSafe("HOME");
  if (!home.has_value()) {
    return std::nullopt;
  }
  return *home;
#endif
}

void ReadBinaryBlob(const std::filesystem::path& path,
                    std::vector<char>* data) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  THROW_CHECK_FILE_OPEN(file, path);
  file.seekg(0, std::ios::end);
  const size_t num_bytes = file.tellg();
  try {
    data->resize(num_bytes);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to allocate " << num_bytes
               << " bytes for binary blob from " << path << ": " << e.what();
    throw;
  }
  file.seekg(0, std::ios::beg);
  file.read(data->data(), num_bytes);
}

void WriteBinaryBlob(const std::filesystem::path& path,
                     const std::vector<char>& data) {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  file.write(data.data(), data.size());
}

std::vector<std::string> ReadTextFileLines(const std::filesystem::path& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    lines.push_back(line);
  }

  return lines;
}

std::string GetRelativePath(const std::filesystem::path& from,
                            const std::filesystem::path& to) {
  // std::filesystem::relative weakly-canonicalizes both paths and builds
  // the '..'-based relative path (replaces the historical boost-based
  // implementation adapted from stackoverflow).
  return std::filesystem::relative(from, to).string();
}


}  // namespace colmap
