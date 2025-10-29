// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QDialog>

class QSimpleUpdater;

namespace Ui {
class UpdateDialog;
}

class ecvUpdateDlg : public QDialog {
    Q_OBJECT

public:
    explicit ecvUpdateDlg(QWidget* parent = nullptr);
    ~ecvUpdateDlg();

public slots:
    void resetFields();
    void checkForUpdates();
    void updateChangelog(const QString& url);
    void displayAppcast(const QString& url, const QByteArray& reply);

private:
    Ui::UpdateDialog* m_ui;
    QSimpleUpdater* m_updater;
};
