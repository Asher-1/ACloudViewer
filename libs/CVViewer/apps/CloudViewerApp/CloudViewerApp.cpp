// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewerApp.h"

#include <string>

#include <CloudViewer.h>
#include "cloudViewer/visualization/app/Viewer.h"

using namespace cloudViewer::visualization::app;

#if __APPLE__
// CloudViewerApp_mac.mm
#else
int main(int argc, const char *argv[]) {
    RunViewer(argc, argv);
    return 0;
}
#endif  // __APPLE__
