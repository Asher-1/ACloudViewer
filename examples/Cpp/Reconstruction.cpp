// ----------------------------------------------------------------------------
// -                        ErowCloudViewer: asher-1.github.io -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "CloudViewer.h"
#include "pipelines/database.h"

using namespace cloudViewer;

int clean_database(const std::string& database_path, const std::string& type) {
    return CleanDatabase(database_path, type);
}

int create_database(const std::string& database_path) {
    return CreateDatabase(database_path);
}

int merge_database(const std::string& database_path1,
                   const std::string& database_path2,
                   const std::string& merged_database_path) {
    return MergeDatabase(database_path1, database_path2, merged_database_path);
}

int main(int argc, char **argv) {
    cloudViewer::utility::SetVerbosityLevel(
            cloudViewer::utility::VerbosityLevel::Debug);

    if (argc < 2) {
        // clang-format off
        cloudViewer::utility::LogInfo("Usage:");
        cloudViewer::utility::LogInfo("    > Reconstruction pipeline");
        cloudViewer::utility::LogInfo("    > option <options>");
        cloudViewer::utility::LogInfo("    > database_path <path>");
        // clang-format on
        return 1;
    }

    std::string option = argv[1];

    int flag = 1;
    if (option == "clean") {
        std::string type = argv[2];
        std::string database_path = argv[3];
        flag = clean_database(database_path, type);
    } else if (option == "create") {
        std::string database_path = argv[2];
        flag = create_database(database_path);
    } else if (option == "merge") {
        std::string database_path1 = argv[2];
        std::string database_path2 = argv[3];
        std::string merged_database_path = argv[4];
        flag = merge_database(database_path1, database_path2, merged_database_path);
    } else {
        cloudViewer::utility::LogError("unsupported option: {}", option);
    }

    if (flag == 0) {
        cloudViewer::utility::LogInfo("success");
    } else {
        cloudViewer::utility::LogInfo("failed!");
    }

    return flag;
}
