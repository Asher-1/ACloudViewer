// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QtGlobal>

#ifdef Q_OS_MAC
#include <QFileOpenEvent>
#endif

// qCC_io
#include "FileIO.h"
#include "MainWindow.h"
#include "ecvApplication.h"

ecvApplication::ecvApplication(int &argc, char **argv, bool isCommandLine)
    : ecvApplicationBase(
              argc, argv, isCommandLine, QStringLiteral("3.9.0 (Asher)")) {
    setApplicationName("ACloudViewer");

    FileIO::setWriterInfo(applicationName(), versionStr());
}

bool ecvApplication::event(QEvent *inEvent) {
#ifdef Q_OS_MAC
    switch (inEvent->type()) {
        case QEvent::FileOpen: {
            MainWindow *mainWindow = MainWindow::TheInstance();

            if (mainWindow == nullptr) {
                return false;
            }

            mainWindow->addToDB(QStringList(
                    static_cast<QFileOpenEvent *>(inEvent)->file()));
            return true;
        }

        default:
            break;
    }
#endif

    return ecvApplicationBase::event(inEvent);
}
