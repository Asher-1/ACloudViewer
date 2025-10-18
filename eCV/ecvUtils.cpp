// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvUtils.h"

#include "ecvConsole.h"

namespace ecvUtils {
void DisplayLockedVerticesWarning(const QString &meshName,
                                  bool displayAsError) {
    QString message =
            QString("Vertices of mesh '%1' are locked (they may be shared by "
                    "multiple entities for instance).\nYou should call this "
                    "method directly on the vertices cloud.\n(warning: all "
                    "entities depending on this cloud will be impacted!)")
                    .arg(meshName);

    if (displayAsError)
        ecvConsole::Error(message);
    else
        ecvConsole::Warning(message);
}
}  // namespace ecvUtils
