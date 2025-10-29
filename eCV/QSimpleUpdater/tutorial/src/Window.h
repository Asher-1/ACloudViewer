// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMainWindow>
#include <QApplication>

namespace Ui
{
class Window;
}

class QSimpleUpdater;

class Window : public QMainWindow
{
   Q_OBJECT

public:
   explicit Window(QWidget *parent = 0);
   ~Window();

public slots:
   void resetFields();
   void checkForUpdates();
   void updateChangelog(const QString &url);
   void displayAppcast(const QString &url, const QByteArray &reply);

private:
   Ui::Window *m_ui;
   QSimpleUpdater *m_updater;
};
