// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CCCOMPASSIMPORT_H
#define CCCOMPASSIMPORT_H

class QString;

class ecvMainAppInterface;

namespace ccCompassImport {
void importFoliations(ecvMainAppInterface *app);  // import foliation data
void importLineations(ecvMainAppInterface *app);  // import lineation data
};  // namespace ccCompassImport

#endif  // CCCOMPASSIMPORT_H
