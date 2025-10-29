// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #                 ACloudViewer PLUGIN: qSTEPCADImport                 #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 of the License.               #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                          COPYRIGHT: EDF R&D                            #
// #                                                                        #
// ##########################################################################

#include <ecvIOPluginInterface.h>

//! Step file import plugin
class qStepCADImport : public QObject, public ccIOPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccIOPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qStepCADImport" FILE
                          "../info.json")

public:
    //! Default constructor
    explicit qStepCADImport(QObject *parent = nullptr);

    // inherited from ccIOPluginInterface
    void registerCommands(ccCommandLineInterface *inCmdLine) override;
    FilterList getFilters() override;
};
