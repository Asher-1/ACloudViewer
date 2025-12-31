// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionToolHelper.h"

#include <CVLog.h>

#include <QApplication>
#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QSettings>
#include <QStyle>
#include <QVBoxLayout>

bool cvSelectionToolHelper::promptUser(const QString& settingsKey,
                                       const QString& title,
                                       const QString& message,
                                       QWidget* parent) {
    // Check if user has disabled this instruction
    QSettings settings;
    QString key = QString("SelectionTools/DontShowAgain/%1").arg(settingsKey);
    bool dontShow = settings.value(key, false).toBool();

    if (dontShow) {
        return false;  // Don't show dialog
    }

    // Create custom dialog (ParaView-style)
    QDialog dialog(parent);
    dialog.setWindowTitle(title);
    dialog.setModal(true);  // Modal - blocks until user responds

    QVBoxLayout* mainLayout = new QVBoxLayout(&dialog);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    mainLayout->setSpacing(15);

    // Icon + message text
    QHBoxLayout* contentLayout = new QHBoxLayout();

    // Information icon
    QLabel* iconLabel = new QLabel(&dialog);
    QIcon infoIcon =
            dialog.style()->standardIcon(QStyle::SP_MessageBoxInformation);
    iconLabel->setPixmap(infoIcon.pixmap(32, 32));
    iconLabel->setAlignment(Qt::AlignTop);
    contentLayout->addWidget(iconLabel);

    // Message text
    QLabel* textLabel = new QLabel(message, &dialog);
    textLabel->setWordWrap(true);
    textLabel->setTextFormat(Qt::RichText);
    textLabel->setMinimumWidth(400);
    contentLayout->addWidget(textLabel, 1);

    mainLayout->addLayout(contentLayout);

    // "Don't show this message again" checkbox (ParaView-style)
    QCheckBox* dontShowAgainCheckBox = new QCheckBox(
            QObject::tr("Do not show this message again"), &dialog);
    mainLayout->addWidget(dontShowAgainCheckBox);

    // OK button (ParaView uses QMessageBox::Ok | QMessageBox::Save,
    // but we simplify with just OK since Save is essentially "don't show
    // again")
    QDialogButtonBox* buttonBox =
            new QDialogButtonBox(QDialogButtonBox::Ok, &dialog);
    QObject::connect(buttonBox, &QDialogButtonBox::accepted, &dialog,
                     &QDialog::accept);
    mainLayout->addWidget(buttonBox);

    // Show dialog (modal - exec() blocks until user responds)
    int result = dialog.exec();

    // Save preference if user checked "don't show again"
    if (dontShowAgainCheckBox->isChecked()) {
        settings.setValue(key, true);
        CVLog::Print(QString("[cvSelectionToolHelper::promptUser] User checked "
                             "'don't show again' for key: %1")
                             .arg(settingsKey));
    }

    return true;
}
