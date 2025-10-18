// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>

#ifdef GUI_ENABLED
#include <QApplication>

#include "ui/main_window.h"
#else
// Dummy QApplication class when GUI is disabled
class QApplication {
public:
    QApplication(int argc, char** argv) {}
};
#endif

namespace colmap {

#if defined(CUDA_ENABLED) || !defined(OPENGL_ENABLED)
const bool kUseOpenGL = false;
#else
const bool kUseOpenGL = true;
#endif

int RunGraphicalUserInterface(int argc, char** argv);
int RunProjectGenerator(int argc, char** argv);

}  // namespace colmap
