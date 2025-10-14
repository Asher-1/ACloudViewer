// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

class QString;

class ecvMainAppInterface;

namespace ccCompassExport {
void saveCSV(ecvMainAppInterface *app, const QString &filename);
void saveSVG(ecvMainAppInterface *app, const QString &filename, float zoom);
void saveXML(ecvMainAppInterface *app, const QString &filename);
};  // namespace ccCompassExport
