// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "G3PointDisclaimer.h"

#include "ui_G3PointDisclaimer.h"

// qCC_plugins
#include <ecvMainAppInterface.h>

// Qt
#include <QMainWindow>

bool G3PointDisclaimer::s_disclaimerAccepted = false;

G3PointDisclaimer::G3PointDisclaimer(QWidget *parent)
    : QDialog(parent), ui(new Ui::G3PointDisclaimer) {
    ui->setupUi(this);

    QString compilationInfo;
    compilationInfo += "Version " + QString(G3POINT_VERSION);
    compilationInfo += QStringLiteral("<br><i>Compiled with");

#if defined(_MSC_VER)
    compilationInfo += QStringLiteral(" MSVC %1 and").arg(_MSC_VER);
#endif

    compilationInfo += QStringLiteral(" Qt %1").arg(QT_VERSION_STR);
    compilationInfo += QStringLiteral("</i>");
    compilationInfo += " [cc " + QString(GIT_BRANCH_CC) + "/" +
                       QString(GIT_COMMMIT_HASH_CC) + "]";
    compilationInfo += " [g3point " + QString(GIT_TAG_G3POINT) + " " +
                       QString(GIT_BRANCH_G3POINT) + "/" +
                       QString(GIT_COMMMIT_HASH_G3POINT) + "]";

    ui->labelCompilationInformation->setText(compilationInfo);
}

G3PointDisclaimer::~G3PointDisclaimer() { delete ui; }

bool G3PointDisclaimer::show(ecvMainAppInterface *app) {
    if (!s_disclaimerAccepted) {
        // if the user "cancels" it, then he refuses the disclaimer
        s_disclaimerAccepted =
                G3PointDisclaimer(app ? app->getMainWindow() : 0).exec();
    }

    return s_disclaimerAccepted;
}
