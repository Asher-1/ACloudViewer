// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECVFILEUTILS_H
#define ECVFILEUTILS_H
// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                             #
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
// #                    COPYRIGHT: CLOUDVIEWER  project                     #
// #                                                                        #
// ##########################################################################

#include <QStandardPaths>

namespace ecvFileUtils {
//! Shortcut for getting the documents location path
inline QString defaultDocPath() {
    // note that according to the docs the QStandardPaths::DocumentsLocation
    // path is never empty
    return QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)
            .first();
}
}  // namespace ecvFileUtils
#endif  // ECVFILEUTILS_H
