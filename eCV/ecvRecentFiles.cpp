// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvRecentFiles.h"

#include "MainWindow.h"
#include "ecvPersistentSettings.h"
#include "ecvSettingManager.h"

// QT
#include <QAction>
#include <QDir>
#include <QFile>
#include <QMenu>
#include <QString>
#include <QStringList>

QString ecvRecentFiles::s_settingKey("RecentFiles");
// const QString SEPERATOR = ";";

ecvRecentFiles::ecvRecentFiles(QWidget *parent) : QObject(parent) {
    m_menu = new QMenu(tr("Open Recent..."), parent);
    m_menu->setIcon(QIcon(":/Resources/images/svg/pqRecentFile.png"));
    m_actionClearMenu = new QAction(tr("Clear Menu"), this);

    connect(m_actionClearMenu, &QAction::triggered, this, [this]() {
        ecvSettingManager::removeKey(ecvPS::LoadFile(), s_settingKey);

        updateMenu();
    });

    updateMenu();
}

QMenu *ecvRecentFiles::menu() { return m_menu; }

void ecvRecentFiles::addFilePath(const QString &filePath) {
    QStringList list =
            ecvSettingManager::getValue(ecvPS::LoadFile(), s_settingKey)
                    .toStringList();

    list.removeAll(filePath);
    list.prepend(filePath);

    // only save the last ten files
    if (list.count() > 10) {
        list = list.mid(0, 10);
    }
    ecvSettingManager::setValue(ecvPS::LoadFile(), s_settingKey, list);

    updateMenu();
}

void ecvRecentFiles::updateMenu() {
    m_menu->clear();

    const QStringList recentList = listRecent();

    for (const QString &recentFile : recentList) {
        QAction *recentAction = new QAction(contractFilePath(recentFile), this);

        recentAction->setData(recentFile);

        connect(recentAction, &QAction::triggered, this,
                &ecvRecentFiles::openFileFromAction);

        m_menu->addAction(recentAction);
    }

    if (!m_menu->actions().isEmpty()) {
        m_menu->addSeparator();
        m_menu->addAction(m_actionClearMenu);
    }

    m_menu->setEnabled(!m_menu->actions().isEmpty());
}

void ecvRecentFiles::openFileFromAction() {
    QAction *action = qobject_cast<QAction *>(sender());

    Q_ASSERT(action);

    QString fileName = action->data().toString();

    if (!QFile::exists(fileName)) {
        return;
    }

    QStringList fileListOfOne{fileName};

    MainWindow::TheInstance()->addToDB(fileListOfOne);
}

QStringList ecvRecentFiles::listRecent() {
    QStringList list =
            ecvSettingManager::getValue(ecvPS::LoadFile(), s_settingKey)
                    .toStringList();

    QStringList::iterator iter = list.begin();

    while (iter != list.end()) {
        const QString filePath = *iter;

        if (!QFile::exists(filePath)) {
            iter = list.erase(iter);
            continue;
        }

        ++iter;
    }

    return list;
}

QString ecvRecentFiles::contractFilePath(const QString &filePath) {
    QString homePath = QDir::toNativeSeparators(QDir::homePath());
    QString newPath = QDir::toNativeSeparators(filePath);

    if (newPath.startsWith(homePath)) {
        return newPath.replace(0, QDir::homePath().length(), '~');
    }

    return filePath;
}

QString ecvRecentFiles::expandFilePath(const QString &filePath) {
    QString newPath = QDir::toNativeSeparators(filePath);

    if (newPath.startsWith('~')) {
        QString homePath = QDir::toNativeSeparators(QDir::homePath());

        return newPath.replace(0, 1, homePath);
    }

    return filePath;
}
