// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Window.h"

int main(int argc, char **argv)
{
   QApplication app(argc, argv);
   app.setApplicationVersion("1.0");
   app.setApplicationName("Bob's Badass App");

   Window window;
   window.show();

   return app.exec();
}
