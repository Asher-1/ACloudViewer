// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ExampleIOPlugin.h"

#include "FooFilter.h"

ExampleIOPlugin::ExampleIOPlugin(QObject* parent)
    : QObject(parent),
      ccIOPluginInterface(":/CC/plugin/ExampleIOPlugin/info.json") {}

void ExampleIOPlugin::registerCommands(ccCommandLineInterface* cmd) {
    // If you want to register this plugin for the command line, create a
    // ccCommandLineInterface::Command and add it here. e.g.:
    //
    // cmd->registerCommand( ccCommandLineInterface::Command::Shared( new
    // FooCommand ) );
}

ccIOPluginInterface::FilterList ExampleIOPlugin::getFilters() {
    return {
            FileIOFilter::Shared(new FooFilter),
    };
}
