// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVTools.h>
#include <FileSystem.h>

#include <QTemporaryDir>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "common/data_root_util.hpp"
#include "tests/common/test_macros.hpp"

static int failures = 0;

namespace {
void setOverride(const std::string& value) {
    AICORE_CHECK(cloudViewer::utility::filesystem::SetEnvVar(
            "CLOUDVIEWER_DATA_ROOT", value));
}
}  // namespace

int main() {
    namespace fs = cloudViewer::utility::filesystem;
    const std::string saved = fs::GetEnvVar("CLOUDVIEWER_DATA_ROOT");
    const std::string home_root = aicore::locate_data_root();
    AICORE_CHECK(!home_root.empty());
    AICORE_CHECK(home_root == fs::LocateDataRoot());

    const std::string da3 = aicore::extract_model_dir("da3_models");
    AICORE_CHECK(da3 ==
                 (std::filesystem::u8path(home_root) / "extract" / "da3_models")
                         .lexically_normal()
                         .generic_u8string());
    AICORE_CHECK(aicore::extract_model_dir("freesplatter_models")
                         .find("freesplatter_models") != std::string::npos);

    // Cover spaces, Chinese and supplementary Unicode with the shared
    // conversion functions, including UTF-16 surrogate pairs on Windows.
    const std::string expected =
            u8"cache \u6a21\u578b/\U0001f9e0/space/../selected";
    AICORE_CHECK(CVTools::Wchar2Char(
                         CVTools::Char2Wchar(expected.c_str()).c_str()) ==
                 expected);
    setOverride(expected);
    AICORE_CHECK(fs::GetEnvVar("CLOUDVIEWER_DATA_ROOT") == expected);
    AICORE_CHECK(aicore::locate_data_root() == expected);
    AICORE_CHECK(aicore::extract_model_dir("da3_models") ==
                 (std::filesystem::u8path(expected) / "extract" / "da3_models")
                         .lexically_normal()
                         .generic_u8string());
    QTemporaryDir temporary;
    AICORE_CHECK(temporary.isValid());
    const std::string output = temporary.path().toUtf8().toStdString() +
                               u8"/\u6a21\u578b-\U0001f9e0.bin";
    std::FILE* file = fs::FOpenForWrite(output);
    AICORE_CHECK(file != nullptr);
    if (file) {
        AICORE_CHECK(std::fwrite("cache", 1, 5, file) == 5);
        std::fclose(file);
        file = fs::FOpen(output, "rb");
        AICORE_CHECK(file != nullptr);
        if (file) {
            char bytes[5] = {};
            AICORE_CHECK(std::fread(bytes, 1, 5, file) == 5);
            AICORE_CHECK(std::string(bytes, 5) == "cache");
            std::fclose(file);
        }
#ifndef _WIN32
        AICORE_CHECK((std::filesystem::status(std::filesystem::u8path(output))
                              .permissions() &
                      std::filesystem::perms::others_write) ==
                     std::filesystem::perms::none);
#endif
    }
    setOverride({});
    AICORE_CHECK(aicore::locate_data_root() ==
                 fs::GetHomeDirectory() + "/cloudViewer_data");
    setOverride(saved);

    std::fprintf(stderr, "data_root ok (default=%s)\n", home_root.c_str());
    return failures == 0 ? 0 : 1;
}
