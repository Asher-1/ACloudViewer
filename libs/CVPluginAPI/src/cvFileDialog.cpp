// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvFileDialog.h"

namespace {

QFileDialog::Options nativeDialogOptions() {
    QFileDialog::Options opts;
#ifdef Q_OS_MACOS
    opts |= QFileDialog::DontUseNativeDialog;
#endif
    return opts;
}

}  // namespace

namespace cvFileDialog {

QString getOpenFileName(QWidget *parent,
                        const QString &caption,
                        const QString &dir,
                        const QString &filter,
                        QString *selectedFilter,
                        QFileDialog::Options options) {
    return QFileDialog::getOpenFileName(parent, caption, dir, filter,
                                        selectedFilter,
                                        options | nativeDialogOptions());
}

QStringList getOpenFileNames(QWidget *parent,
                             const QString &caption,
                             const QString &dir,
                             const QString &filter,
                             QString *selectedFilter,
                             QFileDialog::Options options) {
    return QFileDialog::getOpenFileNames(parent, caption, dir, filter,
                                         selectedFilter,
                                         options | nativeDialogOptions());
}

QString getSaveFileName(QWidget *parent,
                        const QString &caption,
                        const QString &dir,
                        const QString &filter,
                        QString *selectedFilter,
                        QFileDialog::Options options) {
    return QFileDialog::getSaveFileName(parent, caption, dir, filter,
                                        selectedFilter,
                                        options | nativeDialogOptions());
}

QString getExistingDirectory(QWidget *parent,
                             const QString &caption,
                             const QString &dir,
                             QFileDialog::Options options) {
    return QFileDialog::getExistingDirectory(parent, caption, dir,
                                             options | nativeDialogOptions());
}

}  // namespace cvFileDialog
