// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvLogManagerDlg.h"

#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSplitter>
#include <QUrl>
#include <QVBoxLayout>

#include "ecvConsole.h"

ecvLogManagerDlg::ecvLogManagerDlg(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Log Manager"));
    resize(900, 600);
    setupUI();
    refreshLogList();
}

void ecvLogManagerDlg::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    m_splitter = new QSplitter(Qt::Horizontal, this);

    auto* leftWidget = new QWidget(this);
    auto* leftLayout = new QVBoxLayout(leftWidget);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    auto* listLabel = new QLabel(tr("Log Files:"), leftWidget);
    m_fileList = new QListWidget(leftWidget);
    m_fileList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_fileList->setSortingEnabled(false);

    leftLayout->addWidget(listLabel);
    leftLayout->addWidget(m_fileList);

    auto* rightWidget = new QWidget(this);
    auto* rightLayout = new QVBoxLayout(rightWidget);
    rightLayout->setContentsMargins(0, 0, 0, 0);

    auto* viewerLabel = new QLabel(tr("Log Content:"), rightWidget);
    m_logViewer = new QPlainTextEdit(rightWidget);
    m_logViewer->setReadOnly(true);
    m_logViewer->setLineWrapMode(QPlainTextEdit::NoWrap);
    QFont monoFont("Courier");
    monoFont.setStyleHint(QFont::Monospace);
    monoFont.setPointSize(10);
    m_logViewer->setFont(monoFont);

    rightLayout->addWidget(viewerLabel);
    rightLayout->addWidget(m_logViewer);

    m_splitter->addWidget(leftWidget);
    m_splitter->addWidget(rightWidget);
    m_splitter->setStretchFactor(0, 1);
    m_splitter->setStretchFactor(1, 2);

    mainLayout->addWidget(m_splitter);

    m_statusLabel = new QLabel(this);
    mainLayout->addWidget(m_statusLabel);

    auto* btnLayout = new QHBoxLayout;
    m_refreshBtn = new QPushButton(tr("Refresh"), this);
    m_openFolderBtn = new QPushButton(tr("Open Folder"), this);
    m_deleteBtn = new QPushButton(tr("Delete Selected"), this);
    auto* closeBtn = new QPushButton(tr("Close"), this);

    btnLayout->addWidget(m_refreshBtn);
    btnLayout->addWidget(m_openFolderBtn);
    btnLayout->addStretch();
    btnLayout->addWidget(m_deleteBtn);
    btnLayout->addWidget(closeBtn);
    mainLayout->addLayout(btnLayout);

    connect(m_fileList, &QListWidget::currentItemChanged, this,
            &ecvLogManagerDlg::onLogFileSelected);
    connect(m_refreshBtn, &QPushButton::clicked, this,
            &ecvLogManagerDlg::refreshLogList);
    connect(m_openFolderBtn, &QPushButton::clicked, this,
            &ecvLogManagerDlg::openLogFolder);
    connect(m_deleteBtn, &QPushButton::clicked, this,
            &ecvLogManagerDlg::deleteSelectedLogs);
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::accept);
}

QString ecvLogManagerDlg::logDirectory() const {
    return ecvConsole::getLogDirectory();
}

void ecvLogManagerDlg::refreshLogList() {
    m_fileList->clear();
    m_logViewer->clear();

    QString logDir = logDirectory();
    QDir dir(logDir);
    if (!dir.exists()) {
        m_statusLabel->setText(
                tr("Log directory does not exist: %1").arg(logDir));
        return;
    }

    QFileInfoList files = dir.entryInfoList(QStringList() << "*.log",
                                            QDir::Files, QDir::Time);

    for (const QFileInfo& fi : files) {
        QString label = QString("%1  (%2)")
                                .arg(fi.fileName(),
                                     QLocale().formattedDataSize(fi.size()));
        auto* item = new QListWidgetItem(label, m_fileList);
        item->setData(Qt::UserRole, fi.absoluteFilePath());

        if (ecvConsole::TheInstance(false)) {
            auto* console = ecvConsole::TheInstance(false);
            if (console && fi.absoluteFilePath() ==
                                   QFileInfo(console->currentLogFilePath())
                                           .absoluteFilePath()) {
                QFont f = item->font();
                f.setBold(true);
                item->setFont(f);
                item->setText(
                        fi.fileName() +
                        tr("  (current)  (%1)")
                                .arg(QLocale().formattedDataSize(fi.size())));
            }
        }
    }

    m_statusLabel->setText(tr("Log directory: %1  |  %2 file(s)")
                                   .arg(logDir)
                                   .arg(files.size()));
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

    constexpr qint64 MAX_SIZE = 2 * 1024 * 1024;
    QByteArray data;
    if (file.size() > MAX_SIZE) {
        file.seek(file.size() - MAX_SIZE);
        data = file.readAll();
        m_logViewer->setPlainText(
                tr("--- File truncated (showing last %1) ---\n")
                        .arg(QLocale().formattedDataSize(MAX_SIZE)) +
                QString::fromUtf8(data));
    } else {
        data = file.readAll();
        m_logViewer->setPlainText(QString::fromUtf8(data));
    }
}

void ecvLogManagerDlg::deleteSelectedLogs() {
    auto selected = m_fileList->selectedItems();
    if (selected.isEmpty()) return;

    auto reply = QMessageBox::question(
            this, tr("Delete Logs"),
            tr("Delete %1 selected log file(s)?").arg(selected.size()),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (reply != QMessageBox::Yes) return;

    int deleted = 0;
    for (auto* item : selected) {
        QString path = item->data(Qt::UserRole).toString();
        if (QFile::remove(path)) {
            ++deleted;
        }
    }

    refreshLogList();
    m_statusLabel->setText(tr("Deleted %1 file(s)").arg(deleted));
}

void ecvLogManagerDlg::openLogFolder() {
    QString logDir = logDirectory();
    QDesktopServices::openUrl(QUrl::fromLocalFile(logDir));
}
