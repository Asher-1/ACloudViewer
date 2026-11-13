// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

// Exercise the include order where the Windows SDK precedes this header.
#ifndef CopyFile
#define CopyFile TestWin32CopyFileAlias
#endif
#include "FileSystem.h"

namespace fs = cloudViewer::utility::filesystem;
static int failures = 0;
#define CHECK(condition)                                                 \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #condition);                                    \
            ++failures;                                                  \
        }                                                                \
    } while (0)

static void CheckDirectories(const std::filesystem::path& path) {
    const std::string root = path.u8string() + ".dir";
    const auto directory = std::filesystem::u8path(root) /
                           std::filesystem::u8path(u8"\u76ee\u5f55");
    const auto leaf = directory / std::filesystem::u8path(u8"\u6a21\u578b.bin");
    CHECK(fs::MakeDirectoryHierarchy(directory.u8string()));
    CHECK(fs::IsDirectory(directory.u8string()));
    CHECK(fs::DirectoryExists(directory.u8string()));
    CHECK(fs::DirectoryIsEmpty(directory.u8string()));
    CHECK(!fs::IsFile(directory.u8string()));
    CHECK(fs::JoinPath(root, u8"\u76ee\u5f55") == directory.u8string());
    CHECK(fs::JoinPath(std::vector<std::string>{root, u8"\u76ee\u5f55",
                                                u8"\u6a21\u578b.bin"}) ==
          leaf.u8string());
    {
        auto output = fs::OpenOutputStream(leaf.u8string(), std::ios::binary);
        output << "unicode directory";
        CHECK(output.good());
    }
    CHECK(fs::FileExists(leaf.u8string()));
    CHECK(fs::IsFile(leaf.u8string()));
    CHECK(!fs::IsDirectory(leaf.u8string()));
    CHECK(!fs::DirectoryIsEmpty(directory.u8string()));
    std::vector<std::string> subdirs, files;
    CHECK(fs::ListDirectory(root, subdirs, files));
    CHECK(subdirs.size() == 1 && files.empty());
    if (!subdirs.empty())
        CHECK(std::filesystem::u8path(subdirs[0]) == directory);
    subdirs.clear();
    CHECK(fs::ListDirectory(directory.u8string(), subdirs, files));
    CHECK(subdirs.empty() && files.size() == 1);
    if (!files.empty()) CHECK(std::filesystem::u8path(files[0]) == leaf);
    const std::string copied = root + "/copied";
    CHECK(fs::CopyDir(directory.u8string(), copied));
    const std::string copied_leaf = fs::JoinPath(copied, u8"\u6a21\u578b.bin");
    std::vector<char> bytes;
    CHECK(fs::FReadToBuffer(copied_leaf, bytes, nullptr));
    CHECK(std::string(bytes.begin(), bytes.end()) == "unicode directory");
    CHECK(fs::CopyA(directory.u8string(), root + "/recursive"));
    CHECK(fs::CopyFile(leaf.u8string(), root + "/file-copy"));
    CHECK(fs::FileExists(root + "/file-copy"));
    CHECK(fs::RemoveFile(copied_leaf));
    CHECK(fs::DirectoryIsEmpty(copied));
    const std::string previous = fs::GetWorkingDirectory();
    CHECK(fs::ChangeWorkingDirectory(directory.u8string()));
    CHECK(std::filesystem::u8path(fs::GetWorkingDirectory()) == directory);
    CHECK(fs::ChangeWorkingDirectory(previous));
    CHECK(std::filesystem::is_directory(
            std::filesystem::u8path(fs::GetTempDirectoryPath())));
    CHECK(fs::DeleteDirectory(root));
    CHECK(!fs::DirectoryExists(root));
}

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      std::filesystem::u8path(
                              u8"filesystem-\u6587\u4ef6-" +
                              std::to_string(std::chrono::steady_clock::now()
                                                     .time_since_epoch()
                                                     .count()));
    FILE* file = fs::FOpen(path.u8string(), "w+b");
    CHECK(file != nullptr);
    if (!file) return 1;
    const int64_t beyond_long = (int64_t{1} << 31) + 17;
    // Seeking without writing exercises 64-bit offsets without allocating
    // gigabytes of storage on Windows filesystems lacking sparse-file mode.
    CHECK(fs::FSeek(file, beyond_long));
    CHECK(fs::FTell(file) == beyond_long);
    size_t size = 1;
    CHECK(fs::FSize(file, size) && size == 0);
    CHECK(fs::FTell(file) == beyond_long);
    CHECK(fs::FSeek(file, 0));
    CHECK(std::fwrite("abc", 1, 3, file) == 3);
    CHECK(fs::FSize(file, size) && size == 3);
    CHECK(fs::FTell(file) == 3);
    CHECK(fs::FSeek(file, -1, SEEK_END));
    CHECK(std::fgetc(file) == 'c');
    CHECK(!fs::FSeek(nullptr, 0));
    CHECK(fs::FTell(nullptr) == -1);
    CHECK(!fs::FSize(nullptr, size));
    std::fclose(file);
    std::vector<char> bytes;
    std::string error;
    CHECK(fs::FReadToBuffer(path.u8string(), bytes, &error));
    CHECK(std::string(bytes.begin(), bytes.end()) == "abc");
    fs::CFile cfile;
    CHECK(cfile.Open(path.u8string(), "rb"));
    CHECK(cfile.GetFileSize() == 3 && cfile.CurPos() == 0);
    CHECK(fs::FSeek(cfile.GetFILE(), beyond_long));
    CHECK(cfile.CurPos() == beyond_long);
    CHECK(cfile.GetFileSize() == 3 && cfile.CurPos() == beyond_long);
    cfile.Close();
    fs::ReadOnlyMappedFile mapping;
    CHECK(mapping.Open(path.u8string()) == mapping.IsSupported());
    if (mapping.Data()) {
        CHECK(mapping.Size() == 3);
        CHECK(std::string(reinterpret_cast<const char*>(mapping.Data()),
                          mapping.Size()) == "abc");
    }
    CHECK(mapping.Open(path.u8string()) == mapping.IsSupported());
    CHECK(!mapping.Open(path.u8string() + ".missing"));
    CHECK(mapping.Data() == nullptr && mapping.Size() == 0);
    mapping.Close();
    mapping.Close();
    {
        auto output = fs::OpenOutputStream(path.u8string(), std::ios::binary);
        CHECK(output.is_open());
        output.write("streams", 7);
        CHECK(output.good());
    }
    {
        auto output = fs::OpenOutputStream(path.u8string(),
                                           std::ios::binary | std::ios::app);
        CHECK(output.is_open());
        output.write("xyz", 3);
    }
    {
        auto input = fs::OpenInputStream(path.u8string(),
                                         std::ios::binary | std::ios::ate);
        CHECK(input.is_open() && input.tellg() == std::streampos(10));
        input.seekg(0);
        std::string content;
        input >> content;
        CHECK(content == "streamsxyz");
    }
    CHECK(!fs::OpenInputStream(path.u8string() + ".missing").is_open());
    CHECK(!fs::OpenOutputStream(path.u8string() + ".missing/child").is_open());
    CHECK(fs::RemoveFile(path.u8string()));
    CheckDirectories(path);
    return failures ? 1 : 0;
}
