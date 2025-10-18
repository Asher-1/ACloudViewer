// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ccGLPluginInterface.h"

/** Replace 'ExampleGLPlugin' by your own plugin class name throughout and then
        check 'ExampleGLPlugin.cpp' for more directions.

        Each plugin requires an info.json file to provide information about
itself - the name, authors, maintainers, icon, etc..

        The one method you are required to implement is getFilter(). This method
        registers your GL filter with ACloudViewer.
**/

//! Example GL Plugin
class ExampleGLPlugin : public QObject, public ccGLPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccGLPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.ExampleGL" FILE
                          "../info.json")

public:
    explicit ExampleGLPlugin(QObject *parent = nullptr);
    ~ExampleGLPlugin() override = default;

    // Inherited from ccGLFilterPluginInterface
    ccGlFilter *getFilter() override;
};
