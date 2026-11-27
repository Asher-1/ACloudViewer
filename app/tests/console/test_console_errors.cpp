// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QApplication>
#include <QMessageBox>
#include <QTimer>
#include <QWidget>
#include <thread>

#include "ecvConsole.h"

namespace {

class ConsoleErrors : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(ecvConsole::TheInstance(false), nullptr);
        // No text widget: do not install a Qt handler or touch user
        // log/settings.
        ecvConsole::Init(nullptr, &parent);
    }

    void TearDown() override { ecvConsole::ReleaseInstance(false); }

    QWidget parent;
};

TEST_F(ConsoleErrors, MainThreadErrorKeepsSynchronousModalDialog) {
    const QString message = QStringLiteral("Console error contract \u2026");
    bool returned = false;
    bool observed = false;
    QTimer observer;
    QObject::connect(&observer, &QTimer::timeout, &parent, [&] {
        for (auto* widget : QApplication::topLevelWidgets()) {
            auto* dialog = qobject_cast<QMessageBox*>(widget);
            if (!dialog || !dialog->isVisible()) continue;
            observed = true;
            EXPECT_FALSE(returned);
            EXPECT_TRUE(dialog->isModal());
            EXPECT_EQ(dialog->parentWidget(), &parent);
#ifndef Q_OS_MACOS
            // macOS renders QMessageBox as a native alert without a window
            // title, so the "Error" caption is not observable there.
            EXPECT_EQ(dialog->windowTitle(), QStringLiteral("Error"));
#endif
            EXPECT_EQ(dialog->text(), message);
            EXPECT_EQ(dialog->icon(), QMessageBox::Warning);
            dialog->done(QMessageBox::Ok);
        }
    });
    observer.start(1);
    EXPECT_FALSE(CVLog::Error(message));
    returned = true;
    observer.stop();
    EXPECT_TRUE(observed);
}

TEST_F(ConsoleErrors, WorkerErrorDoesNotOpenGuiDialog) {
    bool result = true;
    std::thread worker([&] {
        result = CVLog::Error(QStringLiteral("Worker error contract"));
    });
    worker.join();
    EXPECT_FALSE(result);
    QApplication::processEvents();
    for (auto* widget : QApplication::topLevelWidgets()) {
        EXPECT_EQ(qobject_cast<QMessageBox*>(widget), nullptr);
    }
}

TEST_F(ConsoleErrors, WarningDoesNotBecomeModal) {
    bool observed = false;
    QTimer observer;
    QObject::connect(&observer, &QTimer::timeout, &parent, [&] {
        for (auto* widget : QApplication::topLevelWidgets()) {
            if (auto* dialog = qobject_cast<QMessageBox*>(widget)) {
                observed = true;
                dialog->reject();
            }
        }
    });
    observer.start(1);
    EXPECT_FALSE(CVLog::Warning(QStringLiteral("Nonmodal warning contract")));
    QApplication::processEvents();
    observer.stop();
    EXPECT_FALSE(observed);
}

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
