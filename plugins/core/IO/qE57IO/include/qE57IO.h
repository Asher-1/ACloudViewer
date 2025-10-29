// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #                              CloudViewer                              #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: ACloudViewer project                            #
// #                                                                        #
// ##########################################################################

#include <ecvIOPluginInterface.h>

class qE57IO : public QObject, public ccIOPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccIOPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qE57IO" FILE
                          "../info.json")

public:
    explicit qE57IO(QObject *parent = nullptr);

    void registerCommands(ccCommandLineInterface *cmd) override;

    FilterList getFilters() override;
};
