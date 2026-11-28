// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVTools.h>
#include <FileSystem.h>
#include <gtest/gtest.h>

#include <QTemporaryDir>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include "common/data_root_util.hpp"

namespace {
namespace fs = cloudViewer::utility::filesystem;
class DataRoot : public testing::Test {
protected:
    void SetUp() override { saved = fs::GetEnvVar("CLOUDVIEWER_DATA_ROOT"); }
    void TearDown() override {
        EXPECT_TRUE(fs::SetEnvVar("CLOUDVIEWER_DATA_ROOT", saved));
    }
    std::string saved;
};

TEST_F(DataRoot, SharedModelDirectoriesUseApplicationRoot) {
    const auto root = aicore::locate_data_root();
    ASSERT_FALSE(root.empty());
    EXPECT_EQ(root, fs::LocateDataRoot());
    EXPECT_EQ(aicore::extract_model_dir("da3_models"),
              (std::filesystem::u8path(root) / "extract" / "da3_models")
                      .lexically_normal()
                      .generic_u8string());
    EXPECT_NE(aicore::extract_model_dir("freesplatter_models")
                      .find("freesplatter_models"),
              std::string::npos);
}

TEST_F(DataRoot, UnicodeOverrideRoundTripsAndNormalizesModelPaths) {
    const std::string root =
            u8"cache \u6a21\u578b/\U0001f9e0/space/../selected";
    EXPECT_EQ(CVTools::Wchar2Char(CVTools::Char2Wchar(root.c_str()).c_str()),
              root);
    ASSERT_TRUE(fs::SetEnvVar("CLOUDVIEWER_DATA_ROOT", root));
    EXPECT_EQ(fs::GetEnvVar("CLOUDVIEWER_DATA_ROOT"), root);
    EXPECT_EQ(aicore::locate_data_root(), root);
    EXPECT_EQ(aicore::extract_model_dir("da3_models"),
              (std::filesystem::u8path(root) / "extract" / "da3_models")
                      .lexically_normal()
                      .generic_u8string());
}

TEST_F(DataRoot, UnicodeFilesPreserveBytesAndPrivatePermissions) {
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const std::string path = directory.path().toUtf8().toStdString() +
                             u8"/\u6a21\u578b-\U0001f9e0.bin";
    using File = std::unique_ptr<std::FILE, decltype(&std::fclose)>;
    File file(fs::FOpenForWrite(path), std::fclose);
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(std::fwrite("cache", 1, 5, file.get()), 5u);
    file.reset();
    file.reset(fs::FOpen(path, "rb"));
    ASSERT_NE(file, nullptr);
    char bytes[5] = {};
    ASSERT_EQ(std::fread(bytes, 1, 5, file.get()), 5u);
    EXPECT_EQ(std::string(bytes, 5), "cache");
#ifndef _WIN32
    EXPECT_EQ(std::filesystem::status(std::filesystem::u8path(path))
                              .permissions() &
                      std::filesystem::perms::others_write,
              std::filesystem::perms::none);
#endif
}

TEST_F(DataRoot, EmptyOverrideFallsBackToHome) {
    ASSERT_TRUE(fs::SetEnvVar("CLOUDVIEWER_DATA_ROOT", {}));
    EXPECT_EQ(aicore::locate_data_root(),
              fs::GetHomeDirectory() + "/cloudViewer_data");
}
}  // namespace
