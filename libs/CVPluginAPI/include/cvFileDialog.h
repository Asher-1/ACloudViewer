// SPDX-License-Identifier: GPL-2.0-or-later
// cvFileDialog.h — Cross-platform QFileDialog wrapper.
//
// On macOS, QFileDialog uses NSOpenPanel which maintains its own directory
// state via NSUserDefaults. This state can override the directory passed by
// the application, causing "last used directory" persistence to fail.
// Using DontUseNativeDialog on macOS gives us a fully Qt-controlled dialog
// where QSettings-based persistence works reliably.
//
// Usage: call
//   cvFileDialog::getOpenFileName(...) / cvFileDialog::getOpenFileNames(...) /
//   cvFileDialog::getSaveFileName(...) / cvFileDialog::getExistingDirectory(...)
//   exactly as you would with QFileDialog.

#pragma once

#include <QFileDialog>
#include <QString>
#include <QStringList>
#include <QWidget>

#include "CVPluginAPI.h"

namespace cvFileDialog {

CVPLUGIN_LIB_API QString getOpenFileName(
        QWidget *parent = nullptr,
        const QString &caption = QString(),
        const QString &dir = QString(),
        const QString &filter = QString(),
        QString *selectedFilter = nullptr,
        QFileDialog::Options options = QFileDialog::Options());

CVPLUGIN_LIB_API QStringList getOpenFileNames(
        QWidget *parent = nullptr,
        const QString &caption = QString(),
        const QString &dir = QString(),
        const QString &filter = QString(),
        QString *selectedFilter = nullptr,
        QFileDialog::Options options = QFileDialog::Options());

CVPLUGIN_LIB_API QString getSaveFileName(
        QWidget *parent = nullptr,
        const QString &caption = QString(),
        const QString &dir = QString(),
        const QString &filter = QString(),
        QString *selectedFilter = nullptr,
        QFileDialog::Options options = QFileDialog::Options());

CVPLUGIN_LIB_API QString getExistingDirectory(
        QWidget *parent = nullptr,
        const QString &caption = QString(),
        const QString &dir = QString(),
        QFileDialog::Options options = QFileDialog::Options());

}  // namespace cvFileDialog
