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
// Author: Asher (Dahai Lu)

#include "pipelines/database.h"

#include "exe/database.h"
#include "pipelines/option_utils.h"

namespace cloudViewer {

int CleanDatabase(const std::string& database_path, const std::string& clean_type) {
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
