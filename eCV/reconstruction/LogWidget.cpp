// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LogWidget.h"

#include "util/option_manager.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

namespace cloudViewer {

LogWidget::LogWidget(QWidget* parent, const int max_num_blocks) {
    setWindowFlags(Qt::Window);
    setWindowTitle("Log");
    resize(320, parent->height());

    QGridLayout* grid = new QGridLayout(this);
    grid->setContentsMargins(5, 10, 5, 5);

    qRegisterMetaType<QTextCursor>("QTextCursor");
    qRegisterMetaType<QTextBlock>("QTextBlock");

    QTimer* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &LogWidget::Flush);
    timer->start(100);

    // Comment these lines if debugging, otherwise debug messages won't appear
    // on the console and the output is lost in the log widget when crashing
    cout_redirector_ =
            new StandardOutputRedirector<char, std::char_traits<char>>(
                    std::cout, LogWidget::Update, this);
    cerr_redirector_ =
            new StandardOutputRedirector<char, std::char_traits<char>>(
                    std::cerr, LogWidget::Update, this);
    clog_redirector_ =
            new StandardOutputRedirector<char, std::char_traits<char>>(
                    std::clog, LogWidget::Update, this);

    QHBoxLayout* left_button_layout = new QHBoxLayout();

    QPushButton* save_log_button = new QPushButton(tr("Save"), this);
    connect(save_log_button, &QPushButton::released, this, &LogWidget::SaveLog);
    left_button_layout->addWidget(save_log_button);

    QPushButton* clear_button = new QPushButton(tr("Clear"), this);
    connect(clear_button, &QPushButton::released, this, &LogWidget::Clear);
    left_button_layout->addWidget(clear_button);

    grid->addLayout(left_button_layout, 0, 0, Qt::AlignLeft);

    QHBoxLayout* right_button_layout = new QHBoxLayout();

    grid->addLayout(right_button_layout, 0, 1, Qt::AlignRight);

    text_box_ = new QPlainTextEdit(this);
    text_box_->setReadOnly(true);
    text_box_->setMaximumBlockCount(max_num_blocks);
    text_box_->setWordWrapMode(QTextOption::NoWrap);
    text_box_->setFont(QFont("Courier", 10));
    grid->addWidget(text_box_, 1, 0, 1, 2);
}

LogWidget::~LogWidget() {
    // Flush any remaining CVLog buffer content
    if (!cvlog_buffer_.empty()) {
        QString remaining = QString::fromStdString(cvlog_buffer_).trimmed();
        if (!remaining.isEmpty()) {
            CVLog::LogMessage(remaining, CVLog::LOG_STANDARD);
        }
    }

    if (log_file_.is_open()) {
        log_file_.close();
    }

    delete cout_redirector_;
    delete cerr_redirector_;
    delete clog_redirector_;
}

void LogWidget::Append(const std::string& text) {
    // Collect CVLog messages to send after releasing the lock
    // to avoid deadlock when CVLog::LogMessage triggers GUI dialogs
    QList<QPair<QString, int>> cvlog_messages;

    {
        QMutexLocker locker(&mutex_);
        text_queue_ += text;

        // Dump to log file and flush immediately to avoid data loss on crash
        if (log_file_.is_open()) {
            log_file_ << text;
            log_file_.flush();  // Immediate flush for crash safety
        }

        // Also output to CVLog for unified logging (accumulate by lines)
        cvlog_buffer_ += text;

        // Process complete lines in buffer
        size_t pos;
        while ((pos = cvlog_buffer_.find('\n')) != std::string::npos) {
            std::string line = cvlog_buffer_.substr(0, pos);
            cvlog_buffer_.erase(0, pos + 1);

            if (!line.empty()) {
                QString qline = QString::fromStdString(line);
                qline = qline.trimmed();

                if (!qline.isEmpty()) {
                    // Determine log level based on content
                    int logLevel = CVLog::LOG_STANDARD;
                    QString lowerLine = qline.toLower();

                    // Use QtCompatRegExp for Qt5/Qt6 compatibility
                    static const QtCompatRegExp errorPattern(
                            "\\b(error|fatal|exception|crash)"
                            "\\b",
                            QtCompatRegExpOption::CaseInsensitive);
                    static const QtCompatRegExp warningPattern(
                            "\\b(warning|warn|caution)\\b",
                            QtCompatRegExpOption::CaseInsensitive);

                    // Use qtCompatRegExpMatch for cross-version compatibility
                    bool hasError =
                            qtCompatRegExpMatch(errorPattern, lowerLine);
                    bool hasWarning =
                            qtCompatRegExpMatch(warningPattern, lowerLine);

                    // Also check for common log format prefixes
                    if (lowerLine.startsWith("error") ||
                        lowerLine.startsWith("[error") ||
                        lowerLine.startsWith("e ") || hasError) {
                        logLevel = CVLog::LOG_ERROR;
                    } else if (lowerLine.startsWith("warning") ||
                               lowerLine.startsWith("[warning") ||
                               lowerLine.startsWith("w ") || hasWarning) {
                        logLevel = CVLog::LOG_WARNING;
                    }

                    // Collect message to send after releasing lock
                    cvlog_messages.append(qMakePair(qline, logLevel));
                }
            }
        }
        // Also flush incomplete buffer periodically for crash safety
        // If buffer is getting large (>1KB) without newline, flush it anyway
        if (cvlog_buffer_.size() > 1024) {
            QString remaining = QString::fromStdString(cvlog_buffer_).trimmed();
            if (!remaining.isEmpty()) {
                cvlog_messages.append(
                        qMakePair(remaining, CVLog::LOG_STANDARD));
            }
            cvlog_buffer_.clear();
        }
    }  // Release lock here

    // Now send CVLog messages without holding the lock
    // This prevents deadlock when CVLog::LogMessage triggers GUI dialogs
    for (const auto& msg : cvlog_messages) {
        CVLog::LogMessage(msg.first, msg.second);
    }
}

void LogWidget::Flush() {
    QMutexLocker locker(&mutex_);

    if (text_queue_.size() > 0) {
        // Write to log widget
        text_box_->moveCursor(QTextCursor::End);
        text_box_->insertPlainText(QString::fromStdString(text_queue_));
        text_box_->moveCursor(QTextCursor::End);
        text_queue_.clear();
    }
}

void LogWidget::Clear() {
    QMutexLocker locker(&mutex_);
    text_queue_.clear();
    cvlog_buffer_.clear();
    text_box_->clear();
}

void LogWidget::Update(const char* text,
                       std::streamsize count,
                       void* log_widget_ptr) {
    std::string text_str;
    for (std::streamsize i = 0; i < count; ++i) {
        if (text[i] == '\n') {
            text_str += "\n";
        } else {
            text_str += text[i];
        }
    }

    LogWidget* log_widget = static_cast<LogWidget*>(log_widget_ptr);
    log_widget->Append(text_str);
}

void LogWidget::SaveLog() {
    const std::string log_path =
            QFileDialog::getSaveFileName(this, tr("Select path to log file"),
                                         "", tr("Log (*.log)"))
                    .toUtf8()
                    .constData();

    if (log_path == "") {
        return;
    }

    std::ofstream file(log_path, std::ios::app);
    CHECK(file.is_open()) << log_path;
    file << text_box_->toPlainText().toUtf8().constData();
}

}  // namespace cloudViewer
