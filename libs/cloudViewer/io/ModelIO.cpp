// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "io/ModelIO.h"

#include <FileSystem.h>
#include <Logging.h>

#include <unordered_map>

namespace cloudViewer {
namespace io {

bool ReadModelUsingAssimp(const std::string& filename,
                          visualization::rendering::TriangleMeshModel& model,
                          const ReadTriangleModelOptions& params /*={}*/);

bool ReadTriangleModel(const std::string& filename,
                       visualization::rendering::TriangleMeshModel& model,
                       ReadTriangleModelOptions params /*={}*/) {
    if (params.print_progress) {
        auto progress_text = std::string("Reading model file") + filename;
        auto pbar = utility::ConsoleProgressBar(100, progress_text, true);
        params.update_progress = [pbar](double percent) mutable -> bool {
            pbar.setCurrentCount(size_t(percent));
            return true;
        };
    }
    return ReadModelUsingAssimp(filename, model, params);
}

}  // namespace io
}  // namespace cloudViewer
