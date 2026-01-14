// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/utility/Extract.h"

#include "cloudViewer/data/Dataset.h"
#include "cloudViewer/utility/Download.h"
#include "cloudViewer/utility/FileSystem.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(Extract, ExtractFromZIP) {
    // Directory relative to `data_root`, where files will be temp. downloaded
    // for this test.
    const std::string prefix = "test_extract";
    const std::string extract_dir = data::LocateDataRoot() + "/" + prefix;
    EXPECT_TRUE(utility::filesystem::DeleteDirectory(extract_dir));

    // Download the `test_data_v2_00.zip` test data.
    std::string url =
            "https://github.com/isl-org/cloudViewer_downloads/releases/"
            "download/"
            "data-manager/test_data_v2_00.zip";
    std::string md5 = "bc47a5e33d33e717259e3a37fa5eebef";
    std::string file_path = extract_dir + "/test_data_v2_00.zip";
    // This download shall work.
    EXPECT_EQ(utility::DownloadFromURL(url, md5, extract_dir), file_path);

    // Extract the test zip file.
    EXPECT_NO_THROW(utility::Extract(file_path, extract_dir));
    url = "https://github.com/isl-org/cloudViewer_downloads/releases/download/"
          "data-manager/test_data_v2_00.tar.xz";
    md5 = "7c682c7af4ef9bda1fc854b008ae2bef";
    file_path = extract_dir + "/test_data_v2_00.tar.xz";
    EXPECT_EQ(utility::DownloadFromURL(url, md5, extract_dir), file_path);

    // Currently only `.zip` files are supported.
    EXPECT_ANY_THROW(utility::Extract(file_path, extract_dir));

    // Clean up.
    EXPECT_TRUE(utility::filesystem::DeleteDirectory(extract_dir));
}

}  // namespace tests
}  // namespace cloudViewer
