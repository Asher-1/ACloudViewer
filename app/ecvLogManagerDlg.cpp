// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvLogManagerDlg.h"

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QShortcut>
#include <QSplitter>
#include <QStyle>
#include <QTextStream>
#include <QToolBar>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

#include "ecvConsole.h"

// ============================================================================
// LogLevelHighlighter
// ============================================================================

LogLevelHighlighter::LogLevelHighlighter(QTextDocument* parent)
    : QSyntaxHighlighter(parent) {
    Rule rule;

    QTextCharFormat errorFmt;
    errorFmt.setForeground(QColor("#D32F2F"));
    errorFmt.setFontWeight(QFont::Bold);
    rule.pattern = QRegularExpression(
            R"(\[.*?\]\s*(error|ERROR|Error|fatal|FATAL|abort|ABORT|Abort trap).*)",
            QRegularExpression::CaseInsensitiveOption);
    rule.format = errorFmt;
    m_rules.append(rule);

    rule.pattern =
            QRegularExpression(R"(^\s*\[mvk-error\].*)",
                               QRegularExpression::CaseInsensitiveOption);
    rule.format = errorFmt;
    m_rules.append(rule);

    QTextCharFormat warnFmt;
    warnFmt.setForeground(QColor("#E65100"));
    rule.pattern =
            QRegularExpression(R"(\[.*?\]\s*(warning|WARNING|Warning|WARN).*)",
                               QRegularExpression::CaseInsensitiveOption);
    rule.format = warnFmt;
    m_rules.append(rule);

    rule.pattern = QRegularExpression(
            R"(^WARNING:.*)", QRegularExpression::CaseInsensitiveOption);
    rule.format = warnFmt;
    m_rules.append(rule);

    QTextCharFormat infoFmt;
    infoFmt.setForeground(QColor("#1565C0"));
    rule.pattern = QRegularExpression(
            R"(^\s*\[.*?\]\s*(Plugin found:|Plugin loaded:|Loaded backend).*)",
            QRegularExpression::CaseInsensitiveOption);
    rule.format = infoFmt;
    m_rules.append(rule);

    QTextCharFormat headerFmt;
    headerFmt.setForeground(QColor("#2E7D32"));
    headerFmt.setFontWeight(QFont::Bold);
    rule.pattern = QRegularExpression(R"(^=+$)");
    rule.format = headerFmt;
    m_rules.append(rule);

    rule.pattern = QRegularExpression(R"(^ACloudViewer Log File$)");
    rule.format = headerFmt;
    m_rules.append(rule);

    rule.pattern = QRegularExpression(R"(^Started at:.*$)");
    rule.format = headerFmt;
    m_rules.append(rule);

    rule.pattern = QRegularExpression(R"(^Log file:.*$)");
    rule.format = headerFmt;
    m_rules.append(rule);

    QTextCharFormat timestampFmt;
    timestampFmt.setForeground(QColor("#757575"));
    rule.pattern = QRegularExpression(
            R"(^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\])");
    rule.format = timestampFmt;
    m_rules.append(rule);
}

void LogLevelHighlighter::highlightBlock(const QString& text) {
    for (const Rule& rule : m_rules) {
        QRegularExpressionMatchIterator it = rule.pattern.globalMatch(text);
        while (it.hasNext()) {
            QRegularExpressionMatch match = it.next();
            setFormat(match.capturedStart(), match.capturedLength(),
                      rule.format);
        }
    }
}

// ============================================================================
// ecvLogManagerDlg
// ============================================================================

ecvLogManagerDlg::ecvLogManagerDlg(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Log Manager"));
    setWindowIcon(QIcon(":/Resources/images/svg/logManager.svg"));
    resize(1050, 680);
    setMinimumSize(700, 450);
    setupUI();
    refreshLogList();
    scrollToCurrentLog();
}

void ecvLogManagerDlg::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    setupToolbar();
    mainLayout->addWidget(m_toolbar);

    auto* searchBar = new QWidget(this);
    searchBar->setStyleSheet(
            "QWidget { background: palette(window); "
            "border-bottom: 1px solid palette(mid); }");
    auto* searchLayout = new QHBoxLayout(searchBar);
    searchLayout->setContentsMargins(8, 4, 8, 4);
    searchLayout->setSpacing(6);

    auto* searchIcon = new QLabel(searchBar);
    searchIcon->setPixmap(
            style()->standardPixmap(QStyle::SP_FileDialogContentsView));
    searchLayout->addWidget(searchIcon);

    m_searchEdit = new QLineEdit(searchBar);
    m_searchEdit->setPlaceholderText(tr("Search log content... (Ctrl+F)"));
    m_searchEdit->setClearButtonEnabled(true);
    m_searchEdit->setMinimumWidth(200);
    searchLayout->addWidget(m_searchEdit, 1);

    auto* findPrevBtn = new QToolButton(searchBar);
    findPrevBtn->setIcon(style()->standardIcon(QStyle::SP_ArrowUp));
    findPrevBtn->setToolTip(tr("Find Previous (Shift+F3)"));
    findPrevBtn->setAutoRaise(true);
    searchLayout->addWidget(findPrevBtn);

    auto* findNextBtn = new QToolButton(searchBar);
    findNextBtn->setIcon(style()->standardIcon(QStyle::SP_ArrowDown));
    findNextBtn->setToolTip(tr("Find Next (F3)"));
    findNextBtn->setAutoRaise(true);
    searchLayout->addWidget(findNextBtn);

    auto* sep = new QLabel("|", searchBar);
    sep->setStyleSheet("color: palette(mid);");
    searchLayout->addWidget(sep);

    auto* filterLabel = new QLabel(tr("Level:"), searchBar);
    searchLayout->addWidget(filterLabel);

    m_filterCombo = new QComboBox(searchBar);
    m_filterCombo->addItem(tr("All"), "");
    m_filterCombo->addItem(tr("Errors"), "error");
    m_filterCombo->addItem(tr("Warnings"), "warning");
    m_filterCombo->addItem(tr("Plugins"), "plugin");
    m_filterCombo->setMinimumWidth(100);
    searchLayout->addWidget(m_filterCombo);

    mainLayout->addWidget(searchBar);

    auto* contentWidget = new QWidget(this);
    auto* contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setContentsMargins(6, 6, 6, 0);
    contentLayout->setSpacing(0);

    m_splitter = new QSplitter(Qt::Horizontal, contentWidget);
    m_splitter->setChildrenCollapsible(false);

    auto* leftPanel = new QWidget(m_splitter);
    auto* leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(4);

    auto* listHeader = new QLabel(tr("<b>Log Files</b>"), leftPanel);
    listHeader->setStyleSheet("padding: 2px 4px;");
    leftLayout->addWidget(listHeader);

    m_fileList = new QListWidget(leftPanel);
    m_fileList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_fileList->setSortingEnabled(false);
    m_fileList->setAlternatingRowColors(true);
    m_fileList->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_fileList->setStyleSheet(
            "QListWidget { font-size: 12px; }"
            "QListWidget::item { padding: 3px 6px; }"
            "QListWidget::item:selected { "
            "  background: palette(highlight); "
            "  color: palette(highlighted-text); }");
    leftLayout->addWidget(m_fileList);

    auto* rightPanel = new QWidget(m_splitter);
    auto* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(4);

    auto* viewerHeader = new QLabel(tr("<b>Log Content</b>"), rightPanel);
    viewerHeader->setStyleSheet("padding: 2px 4px;");
    rightLayout->addWidget(viewerHeader);

    m_logViewer = new QPlainTextEdit(rightPanel);
    m_logViewer->setReadOnly(true);
    m_logViewer->setLineWrapMode(QPlainTextEdit::NoWrap);
    qtCompatSetTabStopWidth(m_logViewer, 32);
    QFont monoFont;
#ifdef Q_OS_MACOS
    monoFont = QFont("Menlo", 11);
#elif defined(Q_OS_WIN)
    monoFont = QFont("Consolas", 10);
#else
    monoFont = QFont("DejaVu Sans Mono", 10);
#endif
    monoFont.setStyleHint(QFont::Monospace);
    m_logViewer->setFont(monoFont);
    m_logViewer->setStyleSheet(
            "QPlainTextEdit { "
            "  background: #FAFBFC; "
            "  color: #24292E; "
            "  selection-background-color: #0366D6; "
            "  selection-color: white; "
            "  border: 1px solid palette(mid); "
            "  border-radius: 3px; }");
    rightLayout->addWidget(m_logViewer);

    m_highlighter = new LogLevelHighlighter(m_logViewer->document());

    m_splitter->addWidget(leftPanel);
    m_splitter->addWidget(rightPanel);
    m_splitter->setStretchFactor(0, 2);
    m_splitter->setStretchFactor(1, 5);
    m_splitter->setSizes({280, 700});

    contentLayout->addWidget(m_splitter);
    mainLayout->addWidget(contentWidget, 1);

    auto* statusBar = new QWidget(this);
    statusBar->setStyleSheet(
            "QWidget { background: palette(window); "
            "border-top: 1px solid palette(mid); }");
    auto* statusLayout = new QHBoxLayout(statusBar);
    statusLayout->setContentsMargins(10, 4, 10, 4);

    m_statusLabel = new QLabel(statusBar);
    m_statusLabel->setStyleSheet("color: palette(text); font-size: 12px;");
    statusLayout->addWidget(m_statusLabel);

    statusLayout->addStretch();

    m_sizeLabel = new QLabel(statusBar);
    m_sizeLabel->setStyleSheet("color: palette(dark); font-size: 12px;");
    statusLayout->addWidget(m_sizeLabel);

    mainLayout->addWidget(statusBar);

    connect(m_fileList, &QListWidget::currentItemChanged, this,
            &ecvLogManagerDlg::onLogFileSelected);
    connect(m_searchEdit, &QLineEdit::textChanged, this,
            &ecvLogManagerDlg::onSearchTextChanged);
    connect(m_searchEdit, &QLineEdit::returnPressed, this,
            &ecvLogManagerDlg::findNext);
    connect(findNextBtn, &QToolButton::clicked, this,
            &ecvLogManagerDlg::findNext);
    connect(findPrevBtn, &QToolButton::clicked, this,
            &ecvLogManagerDlg::findPrevious);
    connect(m_filterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ecvLogManagerDlg::onFilterChanged);

    auto* shortcutFind = new QShortcut(QKeySequence::Find, this);
    connect(shortcutFind, &QShortcut::activated, m_searchEdit,
            QOverload<>::of(&QLineEdit::setFocus));

    auto* shortcutFindNext = new QShortcut(QKeySequence::FindNext, this);
    connect(shortcutFindNext, &QShortcut::activated, this,
            &ecvLogManagerDlg::findNext);

    auto* shortcutFindPrev = new QShortcut(QKeySequence::FindPrevious, this);
    connect(shortcutFindPrev, &QShortcut::activated, this,
            &ecvLogManagerDlg::findPrevious);

    auto* shortcutRefresh = new QShortcut(QKeySequence::Refresh, this);
    connect(shortcutRefresh, &QShortcut::activated, this,
            &ecvLogManagerDlg::refreshLogList);
}

void ecvLogManagerDlg::setupToolbar() {
    m_toolbar = new QToolBar(this);
    m_toolbar->setIconSize(QSize(18, 18));
    m_toolbar->setMovable(false);
    m_toolbar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_toolbar->setStyleSheet(
            "QToolBar { spacing: 4px; padding: 2px 6px; "
            "  border-bottom: 1px solid palette(mid); }"
            "QToolButton { padding: 3px 8px; border-radius: 3px; }"
            "QToolButton:hover { background: palette(midlight); }");

    m_actRefresh = m_toolbar->addAction(
            style()->standardIcon(QStyle::SP_BrowserReload), tr("Refresh"));
    m_actRefresh->setShortcut(QKeySequence::Refresh);
    m_actRefresh->setToolTip(tr("Refresh log file list (F5)"));

    m_actOpenFolder = m_toolbar->addAction(
            style()->standardIcon(QStyle::SP_DirOpenIcon), tr("Open Folder"));
    m_actOpenFolder->setToolTip(tr("Open log directory in file manager"));

    m_actExport = m_toolbar->addAction(
            style()->standardIcon(QStyle::SP_DialogSaveButton), tr("Export"));
    m_actExport->setToolTip(tr("Export selected log to a file"));

    m_toolbar->addSeparator();

    m_actDelete = m_toolbar->addAction(
            style()->standardIcon(QStyle::SP_TrashIcon), tr("Delete"));
    m_actDelete->setToolTip(tr("Delete selected log files"));

    m_actDeleteOld = m_toolbar->addAction(
            style()->standardIcon(QStyle::SP_DialogDiscardButton),
            tr("Clean Old"));
    m_actDeleteOld->setToolTip(tr("Delete log files older than 30 days"));

    m_toolbar->addSeparator();

    auto* spacer = new QWidget(m_toolbar);
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_toolbar->addWidget(spacer);

    auto* closeAction = m_toolbar->addAction(
            style()->standardIcon(QStyle::SP_DialogCloseButton), tr("Close"));
    closeAction->setShortcut(QKeySequence::Close);

    connect(m_actRefresh, &QAction::triggered, this,
            &ecvLogManagerDlg::refreshLogList);
    connect(m_actOpenFolder, &QAction::triggered, this,
            &ecvLogManagerDlg::openLogFolder);
    connect(m_actExport, &QAction::triggered, this,
            &ecvLogManagerDlg::exportLog);
    connect(m_actDelete, &QAction::triggered, this,
            &ecvLogManagerDlg::deleteSelectedLogs);
    connect(m_actDeleteOld, &QAction::triggered, this,
            &ecvLogManagerDlg::deleteOldLogs);
    connect(closeAction, &QAction::triggered, this, &QDialog::accept);
}

QString ecvLogManagerDlg::logDirectory() const {
    return ecvConsole::getLogDirectory();
}

qint64 ecvLogManagerDlg::totalLogSize() const {
    qint64 total = 0;
    for (int i = 0; i < m_fileList->count(); ++i) {
        total += m_fileList->item(i)->data(Qt::UserRole + 1).toLongLong();
    }
    return total;
}

void ecvLogManagerDlg::updateStatusBar() {
    m_statusLabel->setText(
            tr("%1 log file(s) in %2").arg(m_totalFiles).arg(logDirectory()));
    m_sizeLabel->setText(
            tr("Total: %1").arg(QLocale().formattedDataSize(m_totalSize)));
}

void ecvLogManagerDlg::scrollToCurrentLog() {
    for (int i = 0; i < m_fileList->count(); ++i) {
        auto* item = m_fileList->item(i);
        if (item->data(Qt::UserRole + 2).toBool()) {
            m_fileList->setCurrentItem(item);
            m_fileList->scrollToItem(item);
            break;
        }
    }
}

void ecvLogManagerDlg::refreshLogList() {
    m_fileList->clear();
    m_logViewer->clear();
    m_totalFiles = 0;
    m_totalSize = 0;

    QString logDir = logDirectory();
    QDir dir(logDir);
    if (!dir.exists()) {
        m_statusLabel->setText(
                tr("Log directory does not exist: %1").arg(logDir));
        return;
    }

    QFileInfoList files = dir.entryInfoList(QStringList() << "*.log",
                                            QDir::Files, QDir::Time);

    QString currentLogPath;
    if (auto* console = ecvConsole::TheInstance(false)) {
        currentLogPath =
                QFileInfo(console->currentLogFilePath()).absoluteFilePath();
    }

    for (const QFileInfo& fi : files) {
        bool isCurrent = !currentLogPath.isEmpty() &&
                         fi.absoluteFilePath() == currentLogPath;

        QString dateStr = fi.lastModified().toString("yyyy-MM-dd  HH:mm:ss");
        QString sizeStr = QLocale().formattedDataSize(fi.size());
        QString label;
        if (isCurrent) {
            label = QString("%1\n%2    %3    [current]")
                            .arg(fi.fileName(), dateStr, sizeStr);
        } else {
            label = QString("%1\n%2    %3")
                            .arg(fi.fileName(), dateStr, sizeStr);
        }

        auto* item = new QListWidgetItem(m_fileList);
        item->setText(label);
        item->setData(Qt::UserRole, fi.absoluteFilePath());
        item->setData(Qt::UserRole + 1, fi.size());
        item->setData(Qt::UserRole + 2, isCurrent);
        item->setToolTip(fi.absoluteFilePath());

        if (isCurrent) {
            QFont f = item->font();
            f.setBold(true);
            item->setFont(f);
            item->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        } else {
            item->setIcon(style()->standardIcon(QStyle::SP_FileIcon));
        }

        m_totalSize += fi.size();
    }

    m_totalFiles = files.size();
    updateStatusBar();
}

void ecvLogManagerDlg::onLogFileSelected(QListWidgetItem* current,
                                         QListWidgetItem* /*prev*/) {
    m_logViewer->clear();
    if (!current) return;

    QString filePath = current->data(Qt::UserRole).toString();
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        m_logViewer->setPlainText(tr("Cannot open file: %1").arg(filePath));
        return;
    }

    constexpr qint64 MAX_SIZE = 4 * 1024 * 1024;
    QByteArray data;
    if (file.size() > MAX_SIZE) {
        file.seek(file.size() - MAX_SIZE);
        data = file.readAll();
        m_logViewer->setPlainText(
                tr("--- Showing last %1 of %2 ---\n\n")
                        .arg(QLocale().formattedDataSize(MAX_SIZE),
                             QLocale().formattedDataSize(file.size())) +
                QString::fromUtf8(data));
    } else {
        data = file.readAll();
        m_logViewer->setPlainText(QString::fromUtf8(data));
    }

    if (!m_currentFilter.isEmpty()) {
        onFilterChanged(m_filterCombo->currentIndex());
    }
}

void ecvLogManagerDlg::deleteSelectedLogs() {
    auto selected = m_fileList->selectedItems();
    if (selected.isEmpty()) return;

    auto reply = QMessageBox::question(
            this, tr("Delete Log Files"),
            tr("Are you sure you want to delete %1 selected log file(s)?")
                    .arg(selected.size()),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (reply != QMessageBox::Yes) return;

    int deleted = 0;
    for (auto* item : selected) {
        if (item->data(Qt::UserRole + 2).toBool()) continue;
        QString path = item->data(Qt::UserRole).toString();
        if (QFile::remove(path)) {
            ++deleted;
        }
    }

    refreshLogList();
    m_statusLabel->setText(tr("Deleted %1 file(s)").arg(deleted));
}

void ecvLogManagerDlg::deleteOldLogs() {
    QString logDir = logDirectory();
    QDir dir(logDir);
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.log",
                                            QDir::Files, QDir::Time);

    QDateTime cutoff = QDateTime::currentDateTime().addDays(-30);
    int oldCount = 0;
    for (const QFileInfo& fi : files) {
        if (fi.lastModified() < cutoff) ++oldCount;
    }

    if (oldCount == 0) {
        QMessageBox::information(this, tr("Clean Old Logs"),
                                 tr("No log files older than 30 days found."));
        return;
    }

    auto reply = QMessageBox::question(
            this, tr("Clean Old Logs"),
            tr("Delete %1 log file(s) older than 30 days?").arg(oldCount),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (reply != QMessageBox::Yes) return;

    QString currentLogPath;
    if (auto* console = ecvConsole::TheInstance(false)) {
        currentLogPath =
                QFileInfo(console->currentLogFilePath()).absoluteFilePath();
    }

    int deleted = 0;
    for (const QFileInfo& fi : files) {
        if (fi.lastModified() < cutoff &&
            fi.absoluteFilePath() != currentLogPath) {
            if (QFile::remove(fi.absoluteFilePath())) ++deleted;
        }
    }

    refreshLogList();
    m_statusLabel->setText(tr("Cleaned %1 old file(s)").arg(deleted));
}

void ecvLogManagerDlg::openLogFolder() {
    QDesktopServices::openUrl(QUrl::fromLocalFile(logDirectory()));
}

void ecvLogManagerDlg::exportLog() {
    auto* current = m_fileList->currentItem();
    if (!current) return;

    QString srcPath = current->data(Qt::UserRole).toString();
    QString fileName = QFileInfo(srcPath).fileName();
    QString destPath = QFileDialog::getSaveFileName(
            this, tr("Export Log File"), fileName,
            tr("Log Files (*.log *.txt);;All Files (*)"));
    if (destPath.isEmpty()) return;

    if (QFile::copy(srcPath, destPath)) {
        m_statusLabel->setText(tr("Exported to: %1").arg(destPath));
    } else {
        QMessageBox::warning(
                this, tr("Export Failed"),
                tr("Could not export log file to:\n%1").arg(destPath));
    }
}

void ecvLogManagerDlg::onSearchTextChanged(const QString& text) {
    if (text.isEmpty()) {
        QList<QTextEdit::ExtraSelection> empty;
        m_logViewer->setExtraSelections(empty);
        return;
    }
    findNext();
}

void ecvLogManagerDlg::findNext() {
    QString searchText = m_searchEdit->text();
    if (searchText.isEmpty()) return;

    if (!m_logViewer->find(searchText)) {
        QTextCursor cursor = m_logViewer->textCursor();
        cursor.movePosition(QTextCursor::Start);
        m_logViewer->setTextCursor(cursor);
        m_logViewer->find(searchText);
    }
}

void ecvLogManagerDlg::findPrevious() {
    QString searchText = m_searchEdit->text();
    if (searchText.isEmpty()) return;

    if (!m_logViewer->find(searchText, QTextDocument::FindBackward)) {
        QTextCursor cursor = m_logViewer->textCursor();
        cursor.movePosition(QTextCursor::End);
        m_logViewer->setTextCursor(cursor);
        m_logViewer->find(searchText, QTextDocument::FindBackward);
    }
}

void ecvLogManagerDlg::onFilterChanged(int index) {
    m_currentFilter = m_filterCombo->itemData(index).toString();
    if (m_currentFilter.isEmpty()) {
        auto* current = m_fileList->currentItem();
        if (current) {
            onLogFileSelected(current, nullptr);
        }
        return;
    }

    QString fullText = m_logViewer->toPlainText();
    if (fullText.isEmpty()) return;

    QStringList lines = fullText.split('\n');
    QStringList filtered;
    for (const QString& line : lines) {
        if (line.contains(m_currentFilter, Qt::CaseInsensitive)) {
            filtered.append(line);
        }
    }

    m_logViewer->setPlainText(
            tr("--- Filtered: %1 matching lines ---\n\n").arg(filtered.size()) +
            filtered.join('\n'));
}
