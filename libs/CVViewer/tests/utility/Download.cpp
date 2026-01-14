// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/utility/Download.h"

#include "cloudViewer/data/Dataset.h"
#include "cloudViewer/utility/FileSystem.h"
#include "cloudViewer/utility/Helper.h"
#include "cloudViewer/utility/Logging.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(Downloader, DownloadAndVerify) {
    std::string url =
            "https://github.com/isl-org/cloudViewer_downloads/releases/download/"
            "data-manager/test_data_00.zip";
    std::string md5 = "996987b27c4497dbb951ec056c9684f4";

    std::string prefix = "temp_test";
    std::string file_dir = data::LocateDataRoot() + "/" + prefix;
    std::string file_path = file_dir + "/" + "test_data_00.zip";
    EXPECT_TRUE(utility::filesystem::DeleteDirectory(file_dir));

    // This download shall work.
    EXPECT_EQ(utility::DownloadFromURL(url, md5, file_dir), file_path);
    EXPECT_TRUE(utility::filesystem::DirectoryExists(file_dir));
    EXPECT_TRUE(utility::filesystem::FileExists(file_path));
    EXPECT_EQ(utility::GetMD5(file_path), md5);

    // This download shall be skipped as the file already exists (look at the
    // message).
    EXPECT_EQ(utility::DownloadFromURL(url, md5, file_dir), file_path);

    // Mismatch md5.
    EXPECT_ANY_THROW(utility::DownloadFromURL(
            url, "00000000000000000000000000000000", file_dir));

    // Clean up.
    EXPECT_TRUE(utility::filesystem::DeleteDirectory(file_dir));
}

}  // namespace tests
}  // namespace cloudViewer
