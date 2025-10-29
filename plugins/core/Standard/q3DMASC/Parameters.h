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

namespace masc {
struct RandomTreesParams {
    int maxDepth = 25;  // To be left as a parameter of the training plugin
                        // (default 25)
    int minSampleCount =
            1;  // To be left as a parameter of the training plugin (default 1)
    // int maxCategories = 0;		//Normally not important as there�s no
    // categorical variable
    int activeVarCount = 0;  // Use 0 as the default parameter (works best)
    int maxTreeCount =
            100;  // Left as a parameter of the training plugin (default: 100)
};

struct TrainParameters {
    RandomTreesParams rt;
    float testDataRatio = 0.2f;  // percentage of test data
};

};  // namespace masc