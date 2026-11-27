// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Functional gate for the app-layer recent-files menu (ecvRecentFiles):
// MRU ordering, deduplication, the ten-entry cap, stale-path pruning and
// the clear-menu contract. Settings are redirected to a temporary directory
// so the host configuration is never touched.

#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QFile>
#include <QMenu>
#include <QSettings>
#include <QTemporaryDir>
#include <QTemporaryFile>
#include <QWidget>

#include "MainWindow.h"
#include "ecvPersistentSettings.h"
#include "ecvRecentFiles.h"
#include "ecvSettingManager.h"

// --- Minimal MainWindow stubs (only referenced symbols are linked) ---------
MainWindow* MainWindow::TheInstance() { return nullptr; }
void MainWindow::addToDB(const QStringList&, QString, bool) {}

namespace {

const char kSettingKey[] = "RecentFiles";

// Full paths of the file entries currently shown in the menu, in order.
QStringList menuPaths(ecvRecentFiles& recent) {
    QStringList paths;
    for (QAction* action : recent.menu()->actions()) {
        const QString path = action->data().toString();
        if (!path.isEmpty()) paths.append(path);
    }
    return paths;
}

QAction* clearAction(ecvRecentFiles& recent) {
    for (QAction* action : recent.menu()->actions()) {
        if (action->data().toString().isEmpty() && !action->isSeparator())
            return action;
    }
    return nullptr;
}

class RecentFilesTest : public ::testing::Test {
protected:
    void SetUp() override {
        ecvSettingManager::removeKey(ecvPS::LoadFile(), kSettingKey);
    }

    void TearDown() override {
        for (const QString& path : m_created) QFile::remove(path);
    }

    // Persistent temp file so listRecent()'s existence check keeps it.
    QString makeFile() {
        QTemporaryFile file;
        file.setAutoRemove(false);
        EXPECT_TRUE(file.open());
        const QString path = file.fileName();
        file.close();
        m_created.append(path);
        return path;
    }

    QWidget m_parent;
    QStringList m_created;
};

}  // namespace

TEST_F(RecentFilesTest, AddPrependsAndDeduplicates) {
    ecvRecentFiles recent(&m_parent);
    const QString a = makeFile();
    const QString b = makeFile();
    const QString c = makeFile();

    recent.addFilePath(a);
    recent.addFilePath(b);
    recent.addFilePath(c);
    recent.addFilePath(a);  // re-adding moves the entry to the front

    EXPECT_EQ(menuPaths(recent), QStringList({a, c, b}));
}

TEST_F(RecentFilesTest, KeepsOnlyTenMostRecent) {
    ecvRecentFiles recent(&m_parent);
    QStringList added;
    for (int i = 0; i < 12; ++i) {
        const QString path = makeFile();
        added.prepend(path);
        recent.addFilePath(path);
    }

    EXPECT_EQ(menuPaths(recent), added.mid(0, 10));
}

TEST_F(RecentFilesTest, MissingFilesAreDroppedFromMenu) {
    ecvRecentFiles recent(&m_parent);
    const QString existing = makeFile();

    recent.addFilePath(existing);
    recent.addFilePath(QStringLiteral("/nonexistent/acv-test-missing.ply"));

    EXPECT_EQ(menuPaths(recent), QStringList({existing}));
}

TEST_F(RecentFilesTest, ClearMenuEmptiesListAndDisablesMenu) {
    ecvRecentFiles recent(&m_parent);
    recent.addFilePath(makeFile());
    ASSERT_FALSE(menuPaths(recent).isEmpty());

    QAction* clear = clearAction(recent);
    ASSERT_NE(clear, nullptr);
    clear->trigger();

    EXPECT_TRUE(menuPaths(recent).isEmpty());
    EXPECT_FALSE(recent.menu()->isEnabled());
    EXPECT_TRUE(ecvSettingManager::getValue(ecvPS::LoadFile(), kSettingKey)
                        .toStringList()
                        .isEmpty());
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("ACloudViewerRecentFilesTests");
    QCoreApplication::setApplicationName("RecentFilesTest");
    QSettings::setPath(QSettings::NativeFormat, QSettings::UserScope,
                       settingsDir.path());
    QSettings::setPath(QSettings::NativeFormat, QSettings::SystemScope,
                       settingsDir.path());
    ecvSettingManager::Init(QStringLiteral("recent-files-test.ini"));
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ecvSettingManager::ReleaseInstance();
    return result;
}