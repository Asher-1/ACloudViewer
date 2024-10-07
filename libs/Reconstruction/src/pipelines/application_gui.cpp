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
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "pipelines/application_gui.h"

#include "exe/gui.h"
#include "pipelines/option_utils.h"

void InitQtResources() {
#ifdef GUI_ENABLED
    Q_INIT_RESOURCE(resources);
#endif
}

namespace cloudViewer {

int GraphicalUserInterface(const std::string& database_path,
                           const std::string& image_path,
                           const std::string& import_path) {
    InitQtResources();
    OptionsParser parser;
    if (!database_path.empty()) {
        parser.registerOption("database_path", &database_path);
    }
    if (!image_path.empty()) {
        parser.registerOption("image_path", &image_path);
    }
    if (!import_path.empty()) {
        parser.registerOption("import_path", &import_path);
    }

    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunGraphicalUserInterface(parser.getArgc(),
                                             parser.getArgv());
}

int GenerateProject(const std::string& output_path,
                    const std::string& quality /*= "high"*/) {
    OptionsParser parser;
    parser.registerOption("output_path", &output_path);
    // supported {low, medium, high, extreme}
    parser.registerOption("quality", &quality);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunProjectGenerator(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer
