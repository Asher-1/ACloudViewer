// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

class QListWidget;
class QListWidgetItem;
class QPlainTextEdit;
class QPushButton;
class QLabel;
class QSplitter;

class ecvLogManagerDlg : public QDialog {
    Q_OBJECT

public:
    explicit ecvLogManagerDlg(QWidget* parent = nullptr);

private slots:
    void refreshLogList();
    void onLogFileSelected(QListWidgetItem* current, QListWidgetItem* prev);
    void deleteSelectedLogs();
    void openLogFolder();

private:
    void setupUI();
    QString logDirectory() const;

    QSplitter* m_splitter;
    QListWidget* m_fileList;
    QPlainTextEdit* m_logViewer;
    QPushButton* m_deleteBtn;
    QPushButton* m_openFolderBtn;
    QPushButton* m_refreshBtn;
    QLabel* m_statusLabel;
};
