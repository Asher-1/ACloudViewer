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













std::vector<std::string> GetFileList(const std::filesystem::path& path) {
  std::vector<std::string> file_list;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (std::filesystem::is_regular_file(entry.path())) {
      file_list.push_back(entry.path().string());
    }
  }
  return file_list;
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
