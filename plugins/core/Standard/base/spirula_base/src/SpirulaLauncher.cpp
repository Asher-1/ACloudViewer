// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SpirulaLauncher.h"

#include <QFileInfo>
#include <QProcess>

namespace spirula {

bool launchGui(const QString& exePath, QString* errorOut) {
    if (exePath.isEmpty()) {
        if (errorOut) {
            *errorOut = QStringLiteral(
                    "Spirula Studio executable not found.\n"
                    "Build it with -DBUILD_SPIRULA=ON (it is copied next to "
                    "the application) or place 'spirula' on the PATH.");
        }
        return false;
    }

    QProcess process;
    process.setProgram(exePath);
    process.setWorkingDirectory(QFileInfo(exePath).absolutePath());
    // No arguments: the upstream single binary opens its official GUI.
    if (!process.startDetached()) {
        if (errorOut) {
            *errorOut = QStringLiteral("Failed to start '%1'.").arg(exePath);
        }
        return false;
    }
    return true;
}

}  // namespace spirula
