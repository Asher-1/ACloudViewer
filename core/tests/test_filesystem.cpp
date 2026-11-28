// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

// Exercise the include order where the Windows SDK precedes this header.
#ifndef CopyFile
#define CopyFile TestWin32CopyFileAlias
#endif
#include "FileSystem.h"

namespace fs = cloudViewer::utility::filesystem;
class FileSystemTest : public ::testing::Test {
protected:
    std::filesystem::path path;
    void SetUp() override {
        path = std::filesystem::temp_directory_path() /
               std::filesystem::u8path(
                       u8"filesystem-\u6587\u4ef6-" +
                       std::to_string(std::chrono::steady_clock::now()
                                              .time_since_epoch()
                                              .count()));
    }
    void TearDown() override {
        std::error_code error;
        std::filesystem::remove(path, error);
        std::filesystem::remove_all(
                std::filesystem::u8path(path.u8string() + ".dir"), error);
    }
    void WriteFixture(const std::string& bytes) {
        auto output = fs::OpenOutputStream(path.u8string(), std::ios::binary);
        ASSERT_TRUE(output.is_open());
        output.write(bytes.data(), bytes.size());
        ASSERT_TRUE(output.good());
    }
};

TEST_F(FileSystemTest, UnicodeDirectoryTraversalCopyAndRemoval) {
    const std::string root = path.u8string() + ".dir";
    const auto directory = std::filesystem::u8path(root) /
                           std::filesystem::u8path(u8"\u76ee\u5f55");
    const auto leaf = directory / std::filesystem::u8path(u8"\u6a21\u578b.bin");
    EXPECT_TRUE(fs::MakeDirectoryHierarchy(directory.u8string()));
    EXPECT_TRUE(fs::IsDirectory(directory.u8string()));
    EXPECT_TRUE(fs::DirectoryExists(directory.u8string()));
    EXPECT_TRUE(fs::DirectoryIsEmpty(directory.u8string()));
    EXPECT_TRUE(!fs::IsFile(directory.u8string()));
    EXPECT_TRUE(fs::JoinPath(root, u8"\u76ee\u5f55") == directory.u8string());
    EXPECT_TRUE(fs::JoinPath(std::vector<std::string>{root, u8"\u76ee\u5f55",
                                                      u8"\u6a21\u578b.bin"}) ==
                leaf.u8string());
    {
        auto output = fs::OpenOutputStream(leaf.u8string(), std::ios::binary);
        output << "unicode directory";
        EXPECT_TRUE(output.good());
    }
    EXPECT_TRUE(fs::FileExists(leaf.u8string()));
    EXPECT_TRUE(fs::IsFile(leaf.u8string()));
    EXPECT_TRUE(!fs::IsDirectory(leaf.u8string()));
    EXPECT_TRUE(!fs::DirectoryIsEmpty(directory.u8string()));
    std::vector<std::string> subdirs, files;
    EXPECT_TRUE(fs::ListDirectory(root, subdirs, files));
    EXPECT_TRUE(subdirs.size() == 1 && files.empty());
    if (!subdirs.empty())
        EXPECT_TRUE(std::filesystem::u8path(subdirs[0]) == directory);
    subdirs.clear();
    EXPECT_TRUE(fs::ListDirectory(directory.u8string(), subdirs, files));
    EXPECT_TRUE(subdirs.empty() && files.size() == 1);
    if (!files.empty()) EXPECT_TRUE(std::filesystem::u8path(files[0]) == leaf);
    const std::string copied = root + "/copied";
    EXPECT_TRUE(fs::CopyDir(directory.u8string(), copied));
    const std::string copied_leaf = fs::JoinPath(copied, u8"\u6a21\u578b.bin");
    std::vector<char> bytes;
    EXPECT_TRUE(fs::FReadToBuffer(copied_leaf, bytes, nullptr));
    EXPECT_TRUE(std::string(bytes.begin(), bytes.end()) == "unicode directory");
    EXPECT_TRUE(fs::CopyA(directory.u8string(), root + "/recursive"));
    EXPECT_TRUE(fs::CopyFile(leaf.u8string(), root + "/file-copy"));
    EXPECT_TRUE(fs::FileExists(root + "/file-copy"));
    EXPECT_TRUE(fs::RemoveFile(copied_leaf));
    EXPECT_TRUE(fs::DirectoryIsEmpty(copied));
    const std::string previous = fs::GetWorkingDirectory();
    EXPECT_TRUE(fs::ChangeWorkingDirectory(directory.u8string()));
    EXPECT_TRUE(std::filesystem::u8path(fs::GetWorkingDirectory()) ==
                directory);
    EXPECT_TRUE(fs::ChangeWorkingDirectory(previous));
    EXPECT_TRUE(std::filesystem::is_directory(
            std::filesystem::u8path(fs::GetTempDirectoryPath())));
    EXPECT_TRUE(fs::DeleteDirectory(root));
    EXPECT_TRUE(!fs::DirectoryExists(root));
}

TEST_F(FileSystemTest, LargeOffsetsAndFileSizePreservePosition) {
    FILE* file = fs::FOpen(path.u8string(), "w+b");
    ASSERT_NE(file, nullptr);
    const auto closeFile =
            std::unique_ptr<FILE, decltype(&std::fclose)>(file, &std::fclose);
    const int64_t beyond_long = (int64_t{1} << 31) + 17;
    // Seeking without writing exercises 64-bit offsets without allocating
    // gigabytes of storage on Windows filesystems lacking sparse-file mode.
    EXPECT_TRUE(fs::FSeek(file, beyond_long));
    EXPECT_TRUE(fs::FTell(file) == beyond_long);
    size_t size = 1;
    EXPECT_TRUE(fs::FSize(file, size) && size == 0);
    EXPECT_TRUE(fs::FTell(file) == beyond_long);
    EXPECT_TRUE(fs::FSeek(file, 0));
    EXPECT_TRUE(std::fwrite("abc", 1, 3, file) == 3);
    EXPECT_TRUE(fs::FSize(file, size) && size == 3);
    EXPECT_TRUE(fs::FTell(file) == 3);
    EXPECT_TRUE(fs::FSeek(file, -1, SEEK_END));
    EXPECT_TRUE(std::fgetc(file) == 'c');
    EXPECT_TRUE(!fs::FSeek(nullptr, 0));
    EXPECT_TRUE(fs::FTell(nullptr) == -1);
    EXPECT_TRUE(!fs::FSize(nullptr, size));
}

TEST_F(FileSystemTest, BufferedMappedAndCFileReads) {
    ASSERT_NO_FATAL_FAILURE(WriteFixture("abc"));
    const int64_t beyond_long = (int64_t{1} << 31) + 17;
    std::vector<char> bytes;
    std::string error;
    EXPECT_TRUE(fs::FReadToBuffer(path.u8string(), bytes, &error));
    EXPECT_TRUE(std::string(bytes.begin(), bytes.end()) == "abc");
    fs::CFile cfile;
    EXPECT_TRUE(cfile.Open(path.u8string(), "rb"));
    EXPECT_TRUE(cfile.GetFileSize() == 3 && cfile.CurPos() == 0);
    EXPECT_TRUE(fs::FSeek(cfile.GetFILE(), beyond_long));
    EXPECT_TRUE(cfile.CurPos() == beyond_long);
    EXPECT_TRUE(cfile.GetFileSize() == 3 && cfile.CurPos() == beyond_long);
    cfile.Close();
    fs::ReadOnlyMappedFile mapping;
    EXPECT_TRUE(mapping.Open(path.u8string()) == mapping.IsSupported());
    if (mapping.Data()) {
        EXPECT_TRUE(mapping.Size() == 3);
        EXPECT_TRUE(std::string(reinterpret_cast<const char*>(mapping.Data()),
                                mapping.Size()) == "abc");
    }
    EXPECT_TRUE(mapping.Open(path.u8string()) == mapping.IsSupported());
    EXPECT_TRUE(!mapping.Open(path.u8string() + ".missing"));
    EXPECT_TRUE(mapping.Data() == nullptr && mapping.Size() == 0);
    mapping.Close();
    mapping.Close();
}

TEST_F(FileSystemTest, UnicodeStreamsAppendAndMissingFiles) {
    {
        auto output = fs::OpenOutputStream(path.u8string(), std::ios::binary);
        EXPECT_TRUE(output.is_open());
        output.write("streams", 7);
        EXPECT_TRUE(output.good());
    }
    {
        auto output = fs::OpenOutputStream(path.u8string(),
                                           std::ios::binary | std::ios::app);
        EXPECT_TRUE(output.is_open());
        output.write("xyz", 3);
    }
    {
        auto input = fs::OpenInputStream(path.u8string(),
                                         std::ios::binary | std::ios::ate);
        EXPECT_TRUE(input.is_open() && input.tellg() == std::streampos(10));
        input.seekg(0);
        std::string content;
        input >> content;
        EXPECT_TRUE(content == "streamsxyz");
    }
    EXPECT_TRUE(!fs::OpenInputStream(path.u8string() + ".missing").is_open());
    EXPECT_TRUE(!fs::OpenOutputStream(path.u8string() + ".missing/child")
                         .is_open());
    EXPECT_TRUE(fs::RemoveFile(path.u8string()));
}
