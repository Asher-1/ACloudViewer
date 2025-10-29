// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #                   CloudViewer PLUGIN: qPhotoScanIO                    #
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
// #                  COPYRIGHT: Daniel Girardeau-Montaut                   #
// #                                                                        #
// ##########################################################################

#include <ecvIOPluginInterface.h>

//! PhotoScan
class qPhotoscanIO : public QObject, public ccIOPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccIOPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qPhotoscanIO" FILE
                          "../info.json")

public:
    explicit qPhotoscanIO(QObject *parent = nullptr);

    ~qPhotoscanIO() override = default;

    // inherited from ccIOPluginInterface
    FilterList getFilters() override;
};
