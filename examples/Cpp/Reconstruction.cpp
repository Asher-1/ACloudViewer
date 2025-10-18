// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
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

int main(int argc, char** argv) {
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
        flag = merge_database(database_path1, database_path2,
                              merged_database_path);
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
