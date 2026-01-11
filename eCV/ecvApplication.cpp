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

// Qt
#include <QMap>

// qCC_io
#include "FileIO.h"
#include "MainWindow.h"
#include "ecvApplication.h"

//! Map between a file version, and the first version of CloudViewer that was
//! able to load it
struct FileVersionToVersion : QMap<short, QString> {
    FileVersionToVersion() {
        insert(10, "1.0 (before 2012)");
        insert(20, "2.0 (05/04/2012)");
        insert(21, "2.4 (07/02/2012)");
        insert(22, "2.4 (11/26/2012)");
        insert(23, "2.4 (02/07/2013)");
        insert(24, "2.4 (02/22/2013)");
        insert(25, "2.4 (03/16/2013)");
        insert(26, "2.4 (04/03/2013)");
        insert(27, "2.4 (04/12/2013)");
        insert(28, "2.5.0 (07/12/2013)");
        insert(29, "2.5.0 (08/14/2013)");
        insert(30, "2.5.0 (08/30/2013)");
        insert(31, "2.5.0 (09/25/2013)");
        insert(32, "2.5.1 (10/11/2013)");
        insert(33, "2.5.2 (12/19/2013)");
        insert(34, "2.5.3 (01/09/2014)");
        insert(35, "2.5.4 (02/13/2014)");
        insert(36, "2.5.5 (05/30/2014)");
        insert(37, "2.5.5 (08/24/2014)");
        insert(38, "2.6.0 (09/14/2014)");
        insert(39, "2.6.1 (01/30/2015)");
        insert(40, "2.6.2 (08/06/2015)");
        insert(41, "2.6.2 (09/01/2015)");
        insert(42, "2.6.2 (10/07/2015)");
        insert(43, "2.6.3 (01/07/2016)");
        insert(44, "2.8.0 (07/07/2016)");
        insert(45, "2.8.0 (10/06/2016)");
        insert(46, "2.8.0 (11/03/2016)");
        insert(47, "2.9.0 (12/22/2016)");
        insert(48, "2.10.0 (10/19/2018)");
        insert(49, "2.11.0 (03/31/2019)");
        insert(50, "2.11.0 (10/06/2019)");
        insert(51, "2.12.0 (03/29/2019)");
        insert(52, "2.12.0 (11/30/2020)");
        insert(53, "2.13.alpha (10/02/2022)");
        insert(54, "2.13.alpha (01/29/2023)");
        insert(55, "2.9.2 (02/19/2024)");
        insert(56, "3.9.3 (01/05/2025)");
        insert(57, "3.9.4 (01/05/2026)");
    }

    QString getMinVersion(short fileVersion) const {
        if (contains(fileVersion)) {
            return value(fileVersion);
        } else {
            return "Unknown version";
        }
    }
};
static FileVersionToVersion s_fileVersionToVersion;

QString ecvApplication::GetMinVersionForFileVersion(short fileVersion) {
    return s_fileVersionToVersion.getMinVersion(fileVersion);
}

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
