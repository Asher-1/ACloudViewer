// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pipelines/database.h"

#include "exe/database.h"
#include "pipelines/option_utils.h"

namespace cloudViewer {

int CleanDatabase(const std::string& database_path,
                  const std::string& clean_type) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    // supported type {all, images, features, matches}
    parser.registerOption("type", &clean_type);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunDatabaseCleaner(parser.getArgc(), parser.getArgv());
}

int CreateDatabase(const std::string& database_path) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunDatabaseCreator(parser.getArgc(), parser.getArgv());
}

int MergeDatabase(const std::string& database_path1,
                  const std::string& database_path2,
                  const std::string& merged_database_path) {
    OptionsParser parser;
    parser.registerOption("database_path1", &database_path1);
    parser.registerOption("database_path2", &database_path2);
    parser.registerOption("merged_database_path", &merged_database_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunDatabaseMerger(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer
