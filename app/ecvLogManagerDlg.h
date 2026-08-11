// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>

class QListWidget;
class QListWidgetItem;
class QPlainTextEdit;
class QToolButton;
class QLabel;
class QSplitter;
class QLineEdit;
class QComboBox;
class QToolBar;
class QAction;

class LogLevelHighlighter : public QSyntaxHighlighter {
    Q_OBJECT
public:
    explicit LogLevelHighlighter(QTextDocument* parent);

protected:
    void highlightBlock(const QString& text) override;

private:
    struct Rule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    QVector<Rule> m_rules;
};

class ecvLogManagerDlg : public QDialog {
    Q_OBJECT

public:
    explicit ecvLogManagerDlg(QWidget* parent = nullptr);

private slots:
    void refreshLogList();
    void onLogFileSelected(QListWidgetItem* current, QListWidgetItem* prev);
    void deleteSelectedLogs();
    void deleteOldLogs();
    void openLogFolder();
    void exportLog();
    void onSearchTextChanged(const QString& text);
    void findNext();
    void findPrevious();
    void onFilterChanged(int index);

private:
    void setupUI();
    void setupToolbar();
    void updateStatusBar();
    void scrollToCurrentLog();
    QString logDirectory() const;
    qint64 totalLogSize() const;

    QToolBar* m_toolbar = nullptr;
    QSplitter* m_splitter = nullptr;
    QListWidget* m_fileList = nullptr;
    QPlainTextEdit* m_logViewer = nullptr;
    QLineEdit* m_searchEdit = nullptr;
    QComboBox* m_filterCombo = nullptr;
    QLabel* m_statusLabel = nullptr;
    QLabel* m_sizeLabel = nullptr;
    LogLevelHighlighter* m_highlighter = nullptr;

    QAction* m_actRefresh = nullptr;
    QAction* m_actOpenFolder = nullptr;
    QAction* m_actExport = nullptr;
    QAction* m_actDelete = nullptr;
    QAction* m_actDeleteOld = nullptr;

    int m_totalFiles = 0;
    qint64 m_totalSize = 0;
    QString m_currentFilter;
};
