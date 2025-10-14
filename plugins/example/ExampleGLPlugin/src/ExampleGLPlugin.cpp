// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ExampleGLPlugin.h"

#include "Bilateral.h"

ExampleGLPlugin::ExampleGLPlugin(QObject *parent)
    : QObject(parent),
      ccGLPluginInterface(":/CC/plugin/ExampleGLPlugin/info.json") {}

ccGlFilter *ExampleGLPlugin::getFilter() { return Example::getBilateral(); }
