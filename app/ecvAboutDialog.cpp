// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvAboutDialog.h"

#include "CommonSettings.h"
#include "MainWindow.h"
#include "ui_aboutDlg.h"

ecvAboutDialog::ecvAboutDialog(QWidget *parent)
    : QDialog(parent), mUI(new Ui::AboutDialog) {
    setAttribute(Qt::WA_DeleteOnClose);

    mUI->setupUi(this);

    QString compilationInfo;

    compilationInfo = versionLongStr(true, Settings::APP_VERSION);
    compilationInfo += QStringLiteral("<br><i>Compiled with");

#if defined(_MSC_VER)
    compilationInfo += QStringLiteral(" MSVC %1 and").arg(_MSC_VER);
#endif

    compilationInfo += QStringLiteral(" Qt %1").arg(QT_VERSION_STR);
    compilationInfo += QStringLiteral(" PCL %1").arg(PCL_VERSION);
    compilationInfo += QStringLiteral(" VTK %1").arg(VTK_VERSION);
    compilationInfo += QStringLiteral("</i>");
    compilationInfo += QStringLiteral("<br><i>Compiled Info: %1")
                               .arg(cloudViewer::GetBuildInfo().c_str());
    compilationInfo += QStringLiteral("</i>");

    QString htmlText = mUI->labelText->text();
    QString enrichedHtmlText = htmlText.arg(compilationInfo);

    mUI->labelText->setText(enrichedHtmlText);
}

ecvAboutDialog::~ecvAboutDialog() { delete mUI; }
