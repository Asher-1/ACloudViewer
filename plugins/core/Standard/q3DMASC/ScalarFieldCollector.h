// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #                     ACLOUDVIEWER PLUGIN: q3DMASC                       #
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
// #                 COPYRIGHT: Dimitri Lague / CNRS / UEB                  #
// #                                                                        #
// ##########################################################################

// Qt
#include <QMap>

// system
#include <set>

class ccPointCloud;

namespace cloudViewer {
class ScalarField;
};

//! SF collector
/** For tracking the creation and removing a set of scalar fields
 **/
class SFCollector {
public:
    enum Behavior { ALWAYS_KEEP, CAN_REMOVE, ALWAYS_REMOVE };

    void push(ccPointCloud* cloud,
              cloudViewer::ScalarField* sf,
              Behavior behavior);

    void releaseSFs(bool keepByDefault);

    bool setBehavior(cloudViewer::ScalarField* sf, Behavior behavior);

    struct SFDesc {
        ccPointCloud* cloud = nullptr;
        cloudViewer::ScalarField* sf = nullptr;
        Behavior behavior = CAN_REMOVE;
    };

    using Map = QMap<cloudViewer::ScalarField*, SFDesc>;
    Map scalarFields;
};
