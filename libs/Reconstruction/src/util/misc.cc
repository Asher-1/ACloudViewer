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
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "util/misc.h"

#include <cstdarg>

#include <boost/algorithm/string.hpp>

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
  // boost::filesystem::path is not constructible from std::filesystem::path
  // (would require two user-defined conversions), so convert explicitly.
  const boost::filesystem::path src = src_path.string();
  const boost::filesystem::path dst = dst_path.string();
  switch (type) {
    case CopyType::COPY:
      boost::filesystem::copy_file(src, dst);
      break;
    case CopyType::HARD_LINK:
      boost::filesystem::create_hard_link(src, dst);
      break;
    case CopyType::SOFT_LINK:
      boost::filesystem::create_symlink(src, dst);
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
  const std::vector<std::string> names =
      StringSplit(StringReplace(path.string(), "\\", "/"), "/");
  if (names.size() > 1 && names.back() == "") {
    return names[names.size() - 2];
  } else {
    return names.back();
  }
}

std::string GetParentDir(const std::filesystem::path& path) {
  // Preserve the historical boost behavior: parent of "/" is "" rather
  // than "/" (std::filesystem differs on this edge case).
  return boost::filesystem::path(path.string()).parent_path().string();
}

std::string GetRelativePath(const std::filesystem::path& from,
                            const std::filesystem::path& to) {
  // This implementation is adapted from:
  // https://stackoverflow.com/questions/10167382
  // A native implementation in boost::filesystem is only available starting
  // from boost version 1.60.
  using namespace boost::filesystem;

  path from_path = canonical(path(from.string()));
  path to_path = canonical(path(to.string()));

  // Start at the root path and while they are the same then do nothing then
  // when they first diverge take the entire from path, swap it with '..'
  // segments, and then append the remainder of the to path.
  path::const_iterator from_iter = from_path.begin();
  path::const_iterator to_iter = to_path.begin();

  // Loop through both while they are the same to find nearest common directory
  while (from_iter != from_path.end() && to_iter != to_path.end() &&
         (*to_iter) == (*from_iter)) {
    ++to_iter;
    ++from_iter;
  }

  // Replace from path segments with '..' (from => nearest common directory)
  path rel_path;
  while (from_iter != from_path.end()) {
    rel_path /= "..";
    ++from_iter;
  }

  // Append the remainder of the to path (nearest common directory => to)
  while (to_iter != to_path.end()) {
    rel_path /= *to_iter;
    ++to_iter;
  }

  return rel_path.string();
}

std::vector<std::string> GetFileList(const std::filesystem::path& path) {
  std::vector<std::string> file_list;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (std::filesystem::is_regular_file(entry.path())) {
      file_list.push_back(entry.path().string());
    }
  }
  return file_list;
}

std::vector<std::string> GetRecursiveFileList(const std::filesystem::path& path) {
  std::vector<std::string> file_list;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (std::filesystem::is_regular_file(entry.path())) {
      file_list.push_back(entry.path().string());
    }
  }
  return file_list;
}

std::vector<std::string> GetDirList(const std::filesystem::path& path) {
  std::vector<std::string> dir_list;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (std::filesystem::is_directory(entry.path())) {
      dir_list.push_back(entry.path().string());
    }
  }
  return dir_list;
}

std::vector<std::string> GetRecursiveDirList(const std::filesystem::path& path) {
  std::vector<std::string> dir_list;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (std::filesystem::is_directory(entry.path())) {
      dir_list.push_back(entry.path().string());
    }
  }
  return dir_list;
}

size_t GetFileSize(const std::filesystem::path& path) {
  std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);
  CHECK(file.is_open()) << path;
  return file.tellg();
}

void PrintHeading1(const std::string& heading) {
  std::cout << std::endl << std::string(78, '=') << std::endl;
  std::cout << heading << std::endl;
  std::cout << std::string(78, '=') << std::endl << std::endl;
}

void PrintHeading2(const std::string& heading) {
  std::cout << std::endl << heading << std::endl;
  std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
}

template <>
std::vector<std::string> CSVToVector(const std::string& csv) {
  auto elems = StringSplit(csv, ",;");
  std::vector<std::string> values;
  values.reserve(elems.size());
  for (auto& elem : elems) {
    StringTrim(&elem);
    if (elem.empty()) {
      continue;
    }
    values.push_back(elem);
  }
  return values;
}

template <>
std::vector<int> CSVToVector(const std::string& csv) {
  auto elems = StringSplit(csv, ",;");
  std::vector<int> values;
  values.reserve(elems.size());
  for (auto& elem : elems) {
    StringTrim(&elem);
    if (elem.empty()) {
      continue;
    }
    try {
      values.push_back(std::stoi(elem));
    } catch (const std::invalid_argument&) {
      return std::vector<int>(0);
    }
  }
  return values;
}

template <>
std::vector<float> CSVToVector(const std::string& csv) {
  auto elems = StringSplit(csv, ",;");
  std::vector<float> values;
  values.reserve(elems.size());
  for (auto& elem : elems) {
    StringTrim(&elem);
    if (elem.empty()) {
      continue;
    }
    try {
      values.push_back(std::stod(elem));
    } catch (const std::invalid_argument&) {
      return std::vector<float>(0);
    }
  }
  return values;
}

template <>
std::vector<double> CSVToVector(const std::string& csv) {
  auto elems = StringSplit(csv, ",;");
  std::vector<double> values;
  values.reserve(elems.size());
  for (auto& elem : elems) {
    StringTrim(&elem);
    if (elem.empty()) {
      continue;
    }
    try {
      values.push_back(std::stold(elem));
    } catch (const std::invalid_argument&) {
      return std::vector<double>(0);
    }
  }
  return values;
}

std::vector<std::string> ReadTextFileLines(const std::filesystem::path& path) {
  std::ifstream file(path);
  CHECK(file.is_open()) << path;

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

void RemoveCommandLineArgument(const std::string& arg, int* argc, char** argv) {
  for (int i = 0; i < *argc; ++i) {
    if (argv[i] == arg) {
      for (int j = i + 1; j < *argc; ++j) {
        argv[i] = argv[j];
      }
      *argc -= 1;
      break;
    }
  }
}

}  // namespace colmap
