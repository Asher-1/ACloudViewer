// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFileDialog>
#include <QString>
#include <QStringList>
#include <QWidget>

#include "CVPluginAPI.h"

namespace cvFileDialog {

CVPLUGIN_LIB_API QString
getOpenFileName(QWidget *parent = nullptr,
                const QString &caption = QString(),
                const QString &dir = QString(),
                const QString &filter = QString(),
                QString *selectedFilter = nullptr,
                QFileDialog::Options options = QFileDialog::Options());

CVPLUGIN_LIB_API QStringList
getOpenFileNames(QWidget *parent = nullptr,
                 const QString &caption = QString(),
                 const QString &dir = QString(),
                 const QString &filter = QString(),
                 QString *selectedFilter = nullptr,
                 QFileDialog::Options options = QFileDialog::Options());

CVPLUGIN_LIB_API QString
getSaveFileName(QWidget *parent = nullptr,
                const QString &caption = QString(),
                const QString &dir = QString(),
                const QString &filter = QString(),
                QString *selectedFilter = nullptr,
                QFileDialog::Options options = QFileDialog::Options());

CVPLUGIN_LIB_API QString
getExistingDirectory(QWidget *parent = nullptr,
                     const QString &caption = QString(),
                     const QString &dir = QString(),
                     QFileDialog::Options options = QFileDialog::Options());

}  // namespace cvFileDialog
