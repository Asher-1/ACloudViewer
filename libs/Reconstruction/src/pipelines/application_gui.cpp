// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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
