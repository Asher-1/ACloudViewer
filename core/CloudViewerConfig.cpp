// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewerConfig.h"

#include <Logging.h>

namespace cloudViewer {

void PrintCloudViewerVersion() {
    utility::LogInfo("CloudViewer {}", CLOUDVIEWER_VERSION);
}

std::string GetCloudViewerVersion() { return std::string(CLOUDVIEWER_VERSION); }

std::string GetBuildInfo() {
#ifdef CUDA_ENABLED
    const std::string cuda_info = "with CUDA";
#else
    const std::string cuda_info = "without CUDA";
#endif
    return "Commit " + CLOUDVIEWER_GIT_COMMIT_ID + " on " +
           CLOUDVIEWER_GIT_COMMIT_DATE + " " + cuda_info;
}

std::string GetCloudViewerBuildInfo() {
    return "CloudViewer " + GetCloudViewerVersion() + " " + GetBuildInfo();
}

}  // namespace cloudViewer
