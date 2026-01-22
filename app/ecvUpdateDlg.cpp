// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvUpdateDlg.h"

#include <QDebug>

#include "CommonSettings.h"
#include "MainWindow.h"
#include "ui_UpdateDialog.h"

// for application update
#include <QSimpleUpdater/include/QSimpleUpdater.h>

//==============================================================================
// ecvUpdateDlg::ecvUpdateDlg
//==============================================================================

ecvUpdateDlg::ecvUpdateDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), m_ui(new Ui::UpdateDialog) {
    m_ui->setupUi(this);

    /* QSimpleUpdater is single-instance */
    m_updater = QSimpleUpdater::getInstance();

    /* Check for updates when the "Check For Updates" button is clicked */
    connect(m_updater, SIGNAL(checkingFinished(QString)), this,
            SLOT(updateChangelog(QString)));
    connect(m_updater, SIGNAL(appcastDownloaded(QString, QByteArray)), this,
            SLOT(displayAppcast(QString, QByteArray)));

    /* React to button clicks */
    connect(m_ui->resetButton, SIGNAL(clicked()), this, SLOT(resetFields()));
    connect(m_ui->closeButton, SIGNAL(clicked()), this, SLOT(close()));
    connect(m_ui->checkButton, SIGNAL(clicked()), this,
            SLOT(checkForUpdates()));

    /* reset the UI state */
    resetFields();
}

//==============================================================================
// ecvUpdateDlg::~ecvUpdateDlg
//==============================================================================

ecvUpdateDlg::~ecvUpdateDlg() { delete m_ui; }

//==============================================================================
// ecvUpdateDlg::checkForUpdates
//==============================================================================

void ecvUpdateDlg::resetFields() {
    m_ui->installedVersion->setText(
            QString(Settings::APP_VERSION).replace("v", ""));
    m_ui->customAppcast->setChecked(false);
    m_ui->enableDownloader->setChecked(true);
    m_ui->showAllNotifcations->setChecked(false);
    m_ui->showUpdateNotifications->setChecked(true);
    m_ui->mandatoryUpdate->setChecked(false);
}

//==============================================================================
// ecvUpdateDlg::checkForUpdates
//==============================================================================

void ecvUpdateDlg::checkForUpdates() {
    /* Get settings from the UI */
    QString version = m_ui->installedVersion->text();
    bool customAppcast = m_ui->customAppcast->isChecked();
    bool downloaderEnabled = m_ui->enableDownloader->isChecked();
    bool notifyOnFinish = m_ui->showAllNotifcations->isChecked();
    bool notifyOnUpdate = m_ui->showUpdateNotifications->isChecked();
    bool mandatoryUpdate = m_ui->mandatoryUpdate->isChecked();

    /* Apply the settings */
    m_updater->setModuleVersion(Settings::UPDATE_RUL, version);
    m_updater->setNotifyOnFinish(Settings::UPDATE_RUL, notifyOnFinish);
    m_updater->setNotifyOnUpdate(Settings::UPDATE_RUL, notifyOnUpdate);
    m_updater->setUseCustomAppcast(Settings::UPDATE_RUL, customAppcast);
    m_updater->setDownloaderEnabled(Settings::UPDATE_RUL, downloaderEnabled);
    m_updater->setMandatoryUpdate(Settings::UPDATE_RUL, mandatoryUpdate);

    /* Check for updates */
    m_updater->checkForUpdates(Settings::UPDATE_RUL);
}

//==============================================================================
// ecvUpdateDlg::updateChangelog
//==============================================================================

void ecvUpdateDlg::updateChangelog(const QString& url) {
    if (url == Settings::UPDATE_RUL)
        m_ui->changelogText->setText(m_updater->getChangelog(url));
}

//==============================================================================
// ecvUpdateDlg::displayAppcast
//==============================================================================

void ecvUpdateDlg::displayAppcast(const QString& url, const QByteArray& reply) {
    if (url == Settings::UPDATE_RUL) {
        QString text =
                "This is the downloaded appcast: <p><pre>" +
                QString::fromUtf8(reply) +
                "</pre></p><p> If you need to store more information on the "
                "appcast (or use another format), just use the "
                "<b>QSimpleUpdater::setCustomAppcast()</b> function. "
                "It allows your application to interpret the appcast "
                "using your code and not QSU's code.</p>";

        m_ui->changelogText->setText(text);
    }
}
