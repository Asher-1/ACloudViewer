// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_RDB_FILTER_HEADER
#define CC_RDB_FILTER_HEADER

// ##########################################################################
// #                                                                        #
// #                      ACloudViewer PLUGIN: qRDBIO                    #
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
// #          COPYRIGHT: RIEGL Laser Measurement Systems GmbH               #
// #                                                                        #
// ##########################################################################

#include <FileIOFilter.h>

class RDBFilter : public FileIOFilter {
public:
    RDBFilter();

    // inherited from FileIOFilter
    CC_FILE_ERROR loadFile(const QString &fileName,
                           ccHObject &container,
                           LoadParameters &parameters) override;

    bool canSave(CV_CLASS_ENUM type,
                 bool &multiple,
                 bool &exclusive) const override;
};

#endif  // CV_RDB_FILTER_HEADER
