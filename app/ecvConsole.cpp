// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvConsole.h"

#include "MainWindow.h"
#include "ecvHead.h"
#include "ecvPersistentSettings.h"
#include "ecvSettingManager.h"

// CV_DB_LIB
#include <ecvSingleton.h>

// CV_APP_COMMON
#include <CommonSettings.h>
#include <ecvOptions.h>

// Qt
#include <QtCompat.h>

#include <QApplication>
#include <QClipboard>
#include <QColor>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFontDatabase>
#include <QFontMetrics>
#include <QKeyEvent>
#include <QMessageBox>
#include <QPainter>
#include <QSize>
#include <QStandardPaths>
#include <QStyle>
#include <QStyleOptionViewItem>
#include <QStyledItemDelegate>
#include <QThread>
#include <QTime>

// system
#include <cassert>
#ifdef QT_DEBUG
#include <iostream>
#endif

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

/***************
 *** Globals ***
 ***************/

// unique console instance
static ecvSingleton<ecvConsole> s_console;

bool ecvConsole::s_redirectToStdOut = false;
bool ecvConsole::s_showQtMessagesInConsole = false;

// ecvCustomQListWidget

namespace {

// Codepoints that render as emoji when a color emoji font serves them:
// Misc Symbols + Dingbats (U+2600-27BF: ⚠ ☑ ❌ ✅), Symbols And Pictographs
// (U+2B00-2BFF: ⭐) and the supplementary pictograph planes (U+1F000-1FAFF).
// Everything else stays in the console font. Variation selectors and joiners
// are handled as run modifiers, not as standalone codepoints.
bool isEmojiCodepoint(uint cp) {
    return (cp >= 0x1F000 && cp <= 0x1FAFF) || (cp >= 0x2600 && cp <= 0x27BF) ||
           (cp >= 0x2B00 && cp <= 0x2BFF);
}

bool isEmojiModifier(uint cp) {
    return cp == 0xFE0F || cp == 0x200D || cp == 0x20E3;
}

}  // namespace

// Model answers (VLX-Seek reasoning steps, OCR transcriptions, ...) are
// arbitrary Unicode and regularly carry emoji presentation sequences
// (U+26A0+FE0F, U+274C, U+2705, ...). The per-glyph fontconfig fallback
// behind the console font resolves those to whatever monochrome symbol font
// sorts first (DejaVu / FreeSerif), which renders them as unrecognizable
// geometry glyphs. This delegate paints emoji runs with a color emoji font
// and keeps every other run in the console font, so the console matches the
// UTF-8 log file. Runs are painted in logical order (the console's base
// direction is LTR) while drawText keeps intra-run shaping intact. When no
// color emoji font is installed, the stock painting path is used unchanged.
class ecvConsoleItemDelegate : public QStyledItemDelegate {
public:
    explicit ecvConsoleItemDelegate(QObject* parent)
        : QStyledItemDelegate(parent) {
        const QStringList families = qtCompatFontFamilies();
        for (const QString& candidate : {QStringLiteral("Noto Color Emoji"),
                                         QStringLiteral("Segoe UI Emoji"),
                                         QStringLiteral("Apple Color Emoji")}) {
            if (!families.contains(candidate)) continue;
            m_emojiFont.setFamily(candidate);
            m_hasEmojiFont = true;
            break;
        }
    }

protected:
    void paint(QPainter* painter,
               const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        QStyleOptionViewItem opt = option;
        initStyleOption(&opt, index);
        if (!m_hasEmojiFont || !containsEmoji(opt.text)) {
            QStyledItemDelegate::paint(painter, option, index);
            return;
        }
        // Background, selection and focus through the style; the text is
        // painted run-by-run below (initStyleOption mapped the item's
        // ForegroundRole into opt.palette).
        QStyleOptionViewItem background = opt;
        background.text.clear();
        QStyle* style =
                opt.widget ? opt.widget->style() : QApplication::style();
        style->drawControl(QStyle::CE_ItemViewItem, &background, painter,
                           opt.widget);
        // Split the line into emoji vs plain-text runs. Consecutive same-kind
        // characters share one run; modifiers extend the run they follow.
        struct Run {
            QString text;
            bool emoji;
        };
        QVector<Run> runs;
        for (int i = 0; i < opt.text.size();) {
            const QChar c = opt.text.at(i);
            uint cp = c.unicode();
            int advance = 1;
            if (c.isHighSurrogate() && i + 1 < opt.text.size() &&
                opt.text.at(i + 1).isLowSurrogate()) {
                cp = QChar::surrogateToUcs4(c, opt.text.at(i + 1));
                advance = 2;
            }
            if (isEmojiModifier(cp)) {
                if (!runs.isEmpty())
                    runs.last().text += opt.text.mid(i, advance);
                else
                    runs.push_back({opt.text.mid(i, advance), false});
            } else {
                const bool emoji = isEmojiCodepoint(cp);
                if (!runs.isEmpty() && runs.last().emoji == emoji)
                    runs.last().text += opt.text.mid(i, advance);
                else
                    runs.push_back({opt.text.mid(i, advance), emoji});
            }
            i += advance;
        }
        QPalette::ColorGroup group = (opt.state & QStyle::State_Enabled)
                                             ? QPalette::Normal
                                             : QPalette::Disabled;
        if (group == QPalette::Normal && !(opt.state & QStyle::State_Active))
            group = QPalette::Inactive;
        painter->save();
        painter->setPen(
                opt.palette.color(group, (opt.state & QStyle::State_Selected)
                                                 ? QPalette::HighlightedText
                                                 : QPalette::Text));
        QFont textFont = opt.font;
        // The color emoji fonts are bitmap strikes: request the text line
        // height in pixels so the glyph boxes align with the console text.
        QFont emojiFont = m_emojiFont;
        emojiFont.setPixelSize(qMax(8, QFontMetrics(textFont).height()));
        const QRect textRect = style->subElementRect(
                QStyle::SE_ItemViewItemText, &opt, opt.widget);
        painter->setClipRect(textRect);
        int x = textRect.left();
        const int baseline = textRect.top() + QFontMetrics(textFont).ascent();
        for (const Run& run : runs) {
            painter->setFont(run.emoji ? emojiFont : textFont);
            painter->drawText(x, baseline, run.text);
            x += qRound(
                    QFontMetricsF(painter->font()).horizontalAdvance(run.text));
        }
        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem& option,
                   const QModelIndex& index) const override {
        const QSize base = QStyledItemDelegate::sizeHint(option, index);
        if (!m_hasEmojiFont) return base;
        QStyleOptionViewItem opt = option;
        initStyleOption(&opt, index);
        if (!containsEmoji(opt.text)) return base;
        QFont emojiFont = m_emojiFont;
        emojiFont.setPixelSize(qMax(8, QFontMetrics(opt.font).height()));
        return QSize(base.width(),
                     qMax(base.height(), QFontMetrics(emojiFont).height() + 2));
    }

private:
    static bool containsEmoji(const QString& text) {
        for (int i = 0; i < text.size();) {
            const QChar c = text.at(i);
            uint cp = c.unicode();
            int advance = 1;
            if (c.isHighSurrogate() && i + 1 < text.size() &&
                text.at(i + 1).isLowSurrogate()) {
                cp = QChar::surrogateToUcs4(c, text.at(i + 1));
                advance = 2;
            }
            if (isEmojiCodepoint(cp) || isEmojiModifier(cp)) return true;
            i += advance;
        }
        return false;
    }

    QFont m_emojiFont;
    bool m_hasEmojiFont = false;
};

ecvCustomQListWidget::ecvCustomQListWidget(QWidget* parent)
    : QListWidget(parent) {
    QFont mono = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    mono.setPointSize(qMax(font().pointSize(), 9));
    setFont(mono);
    // Emoji-aware line painting for model answers (see the delegate comment).
    setItemDelegate(new ecvConsoleItemDelegate(this));
}

void ecvCustomQListWidget::keyPressEvent(QKeyEvent* event) {
    if (event->matches(QKeySequence::Copy)) {
        int itemsCount = count();
        QStringList strings;
        for (int i = 0; i < itemsCount; ++i) {
            if (item(i)->isSelected()) {
                strings << item(i)->text();
            }
        }

        QApplication::clipboard()->setText(strings.join("\n"));
    } else {
        QListWidget::keyPressEvent(event);
    }
}

int ecvCustomQListWidget::sizeHintForColumn(int column) const {
    // Calculate the maximum width needed for all items
    int maxWidth = 0;
    QFontMetrics fm(font());

    for (int i = 0; i < count(); ++i) {
        QListWidgetItem* listItem = item(i);
        if (listItem) {
            QString text = listItem->text();
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
            int textWidth = fm.horizontalAdvance(text);
#else
            int textWidth = fm.width(text);
#endif
            maxWidth = qMax(maxWidth, textWidth);
        }
    }

    // Add some padding for margins and scrollbar
    return maxWidth + 40;
}

void ecvCustomQListWidget::updateHorizontalScrollRange() {
    // Force the widget to recalculate its content size
    updateGeometries();

    // Schedule a layout update
    scheduleDelayedItemsLayout();
}

// ecvConsole
ecvConsole* ecvConsole::TheInstance(bool autoInit /*=true*/) {
    if (!s_console.instance && autoInit) {
        s_console.instance = new ecvConsole;
        CVLog::RegisterInstance(s_console.instance);
    }

    return s_console.instance;
}

void ecvConsole::ReleaseInstance(bool flush /*=true*/) {
    if (flush && s_console.instance) {
        // DGM: just in case some messages are still in the queue
        s_console.instance->refresh();
    }
    CVLog::RegisterInstance(nullptr);
    s_console.release();
}

ecvConsole::ecvConsole()
    : m_textDisplay(nullptr),
      m_parentWidget(nullptr),
      m_parentWindow(nullptr),
      m_logStream(nullptr) {}

ecvConsole::~ecvConsole() {
    setLogFile(QString());  // to close/delete any active stream
}

QString ecvConsole::currentLogFilePath() const {
    if (m_logFile.isOpen()) {
        return m_logFile.fileName();
    }
    return {};
}

void myMessageOutput(QtMsgType type,
                     const QMessageLogContext& context,
                     const QString& msg) {
#ifndef QT_DEBUG
    if (!ecvConsole::QtMessagesEnabled()) {
        return;
    }

    if (type == QtDebugMsg) {
        return;
    }
#endif

    QString message =
            QString("[%1] ").arg(context.function) +
            msg;  // QString("%1 (%1:%1,
                  // %1)").arg(msg).arg(context.file).arg(context.line).arg(context.function);

    // in this function, you can write the message to any stream!
    switch (type) {
        case QtDebugMsg:
            CVLog::PrintDebug(msg);
            break;
        case QtWarningMsg:
            message.prepend("[Qt WARNING] ");
            CVLog::Warning(message);
            break;
        case QtCriticalMsg:
            message.prepend("[Qt CRITICAL] ");
            CVLog::Warning(message);
            break;
        case QtFatalMsg:
            message.prepend("[Qt FATAL] ");
            CVLog::Warning(message);
            break;
        case QtInfoMsg:
            message.prepend("[Qt INFO] ");
            CVLog::Warning(message);
            break;
    }

#ifdef QT_DEBUG
    // Also send the message to the console so we can look at the output when CC
    // has quit
    //	(in Qt Creator's Application Output for example)
    switch (type) {
        case QtDebugMsg:
        case QtWarningMsg:
        case QtInfoMsg:
            std::cout << message.toStdString() << std::endl;
            break;

        case QtCriticalMsg:
        case QtFatalMsg:
            std::cerr << message.toStdString() << std::endl;
            break;
    }

#endif
}

void ecvConsole::EnableQtMessages(bool state) {
    s_showQtMessagesInConsole = state;

    // persistent settings
    ecvSettingManager::setValue(ecvPS::Console(), "QtMessagesEnabled",
                                s_showQtMessagesInConsole);
}

void ecvConsole::Init(ecvCustomQListWidget* textDisplay /*=0*/,
                      QWidget* parentWidget /*=0*/,
                      MainWindow* parentWindow /*=0*/,
                      bool redirectToStdOut /*=false*/) {
    // should be called only once!
    if (s_console.instance) {
        assert(false);
        return;
    }

    s_console.instance = new ecvConsole;
    s_console.instance->m_textDisplay = textDisplay;
    s_console.instance->m_parentWidget = parentWidget;
    s_console.instance->m_parentWindow = parentWindow;
    s_redirectToStdOut = redirectToStdOut;

    // auto-start
    if (textDisplay) {
        // load from persistent settings
        s_showQtMessagesInConsole =
                ecvSettingManager::getValue(ecvPS::Console(),
                                            "QtMessagesEnabled", false)
                        .toBool();

        // set log file with prefix
        s_console.instance->setLogFile(Settings::LOGFILE_PREFIX);

        // install : set the callback for Qt messages
        qInstallMessageHandler(myMessageOutput);

        s_console.instance->setAutoRefresh(true);
    }

    CVLog::RegisterInstance(s_console.instance);
}

void ecvConsole::setAutoRefresh(bool state) {
    if (state) {
        connect(&m_timer, &QTimer::timeout, this, &ecvConsole::refresh);
        m_timer.start(1000);
    } else {
        m_timer.stop();
        disconnect(&m_timer, &QTimer::timeout, this, &ecvConsole::refresh);
    }
}

void ecvConsole::refresh() {
    m_mutex.lock();

    if (m_textDisplay && !m_queue.isEmpty()) {
        // Note: Message filtering based on verbosity level is now done in
        // CVLog::LogMessage(), so we just display all messages in the queue

        for (QVector<ConsoleItemType>::const_iterator it = m_queue.constBegin();
             it != m_queue.constEnd(); ++it) {
            // it->second = message severity
            int level = it->second;

            // destination: console widget (log file is already written in
            // logMessage()) it->first = message text
            QListWidgetItem* item = new QListWidgetItem(it->first);

            // set color based on the message severity
            // Error
            if ((it->second & LOG_ERROR) == LOG_ERROR) {
                item->setForeground(Qt::red);
            }
            // Warning
            else if ((it->second & LOG_WARNING) == LOG_WARNING) {
                item->setForeground(Qt::darkRed);
                // we also force the console visibility if a warning message
                // arrives!
                if (m_parentWindow) {
                    m_parentWindow->forceConsoleDisplay();
                }
            }
#ifdef QT_DEBUG
            else if (it->second & DEBUG_FLAG) {
                item->setForeground(Qt::blue);
            }
#endif

            m_textDisplay->addItem(item);
        }

        // Update horizontal scrollbar range to handle long text
        m_textDisplay->updateHorizontalScrollRange();

        m_textDisplay->scrollToBottom();
    }

    m_queue.clear();

    // Flush log file periodically (non-critical messages may not have been
    // flushed yet)
    if (m_logStream) {
        m_logFile.flush();
    }

    m_mutex.unlock();
}

void ecvConsole::logMessage(const QString& message, int level) {
    // Note: Message filtering based on verbosity level is now done in
    // CVLog::LogMessage(), so we just process all messages that reach here

    // QString line = __LINE__;
    // QString filename = __FILE__;
    // QString functionname = __FUNCTION__;

    if (s_redirectToStdOut) {
        printf("%s\n", qPrintable(message));
    }
    if (m_textDisplay || m_logStream) {
        m_mutex.lock();

        // Strip carriage returns (from CRLF on Windows) before splitting,
        // otherwise trailing \r renders as □ in QListWidget.
        QString cleaned = message;
        cleaned.remove(QChar('\r'));

        // Split multi-line messages for better display
        QStringList lines = cleaned.split('\n');

        // Write to log file immediately for crash safety (all messages)
        // UI update will still be handled by the timer for performance
        if (m_logStream) {
            // Write first line with timestamp
            QString firstLine = QStringLiteral("[") + DATETIME +
                                QStringLiteral("] ") +
                                (lines.isEmpty() ? message : lines[0]);
            *m_logStream << firstLine << QtCompat::endl;

            // Write remaining lines with indentation (no timestamp)
            for (int i = 1; i < lines.size(); ++i) {
                if (!lines[i].trimmed().isEmpty()) {
                    *m_logStream << "    " << lines[i] << QtCompat::endl;
                }
            }

            // Flush immediately for ERROR/WARNING, or every few messages for
            // others
            if ((level & LOG_ERROR) || (level & LOG_WARNING)) {
                m_logFile.flush();
            }
        }

        // Queue for UI update
        if (m_textDisplay) {
            if (lines.size() > 1) {
                // Multi-line message: add first line with timestamp
                QString firstLine = QStringLiteral("[") + DATETIME +
                                    QStringLiteral("] ") + lines[0];
                m_queue.push_back(ConsoleItemType(firstLine, level));

                // Add remaining lines with indentation (no timestamp)
                for (int i = 1; i < lines.size(); ++i) {
                    if (!lines[i].trimmed().isEmpty()) {
                        QString indentedLine =
                                QStringLiteral("    ") + lines[i];
                        m_queue.push_back(ConsoleItemType(indentedLine, level));
                    }
                }
            } else {
                // Single line message: add with timestamp
                QString formatedMessage = QStringLiteral("[") + DATETIME +
                                          QStringLiteral("] ") + message;
                m_queue.push_back(ConsoleItemType(formatedMessage, level));
            }
        }

        m_mutex.unlock();
    }
#ifdef QT_DEBUG
    else {
        // Error
        if ((level & LOG_ERROR) == LOG_ERROR) {
            if (level & LOG_VERBOSE)
                printf("ERR-DBG: ");
            else
                printf("ERR: ");
        }
        // Warning
        else if ((level & LOG_WARNING) == LOG_WARNING) {
            if (level & LOG_VERBOSE)
                printf("WARN-DBG: ");
            else
                printf("WARN: ");
        }
        // Standard
        else {
            if (level & LOG_VERBOSE)
                printf("MSG-DBG: ");
            else
                printf("MSG: ");
        }
        printf(" %s\n", qPrintable(formatedMessage));
    }
#endif

    // we display the error messages in a popup dialog
    if ((level & LOG_ERROR) == LOG_ERROR && qApp && m_parentWidget &&
        QThread::currentThread() == qApp->thread()) {
        QMessageBox::warning(m_parentWidget, "Error", message);
    }
}

QString ecvConsole::generateLogFileName(const QString& prefix) {
    // Generate timestamp in glog format: YYYYMMDD-HHMMSS
    QString timestamp =
            QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss");

    // Get process ID
    qint64 pid = getpid();

    // Generate log filename: <prefix>.<timestamp>.<pid>.log
    return QString("%1.%2.%3.log").arg(prefix).arg(timestamp).arg(pid);
}

QString ecvConsole::getLogDirectory() {
    QStringList candidatePaths;

#ifdef _WIN32
    // Windows: Use application directory first, then temp directory
    candidatePaths << QCoreApplication::applicationDirPath() + "/logs";
    candidatePaths << QStandardPaths::writableLocation(
                              QStandardPaths::TempLocation) +
                              "/ACloudViewerCache/logs";

#elif defined(__APPLE__)
    // macOS: App bundle is read-only, use standard locations
    // 1. User's log directory (~/Library/Logs/ACloudViewerCache)
    QString appLogPath =
            QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    if (!appLogPath.isEmpty()) {
        // On macOS, AppDataLocation typically points to ~/Library/Application
        // Support/ACloudViewer We want to use ~/Library/Logs/ACloudViewerCache
        // instead
        QDir appDir(appLogPath);
        appDir.cdUp();  // Go to ~/Library/Application Support
        appDir.cdUp();  // Go to ~/Library
        if (appDir.cd("Logs")) {
            candidatePaths << appDir.absolutePath() + "/ACloudViewerCache";
        }
    }

    // 2. Standard AppDataLocation as fallback
    if (!appLogPath.isEmpty()) {
        candidatePaths << appLogPath + "/logs";
    }

    // 3. User's home directory
    QString homePath =
            QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
    if (!homePath.isEmpty()) {
        candidatePaths << homePath + "/.ACloudViewerCache/logs";
    }

    // 4. Try application directory (might work if not in app bundle)
    candidatePaths << QCoreApplication::applicationDirPath() + "/logs";

    // 5. Fallback to temp directory
    candidatePaths << QStandardPaths::writableLocation(
                              QStandardPaths::TempLocation) +
                              "/ACloudViewerCache/logs";

#else
    // Linux/Unix: Try multiple locations with fallback for permission issues
    // 1. Try application directory first (for portable installations)
    candidatePaths << QCoreApplication::applicationDirPath() + "/logs";

    // 2. Try user's local data directory (usually
    // ~/.local/share/ACloudViewerCache/logs)
    QString dataPath =
            QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    if (!dataPath.isEmpty()) {
        candidatePaths << dataPath + "/logs";
    }

    // 3. Try user's home directory (hidden directory)
    QString homePath =
            QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
    if (!homePath.isEmpty()) {
        candidatePaths << homePath + "/.ACloudViewerCache/logs";
    }

    // 4. Fallback to temp directory
    candidatePaths << QStandardPaths::writableLocation(
                              QStandardPaths::TempLocation) +
                              "/ACloudViewerCache/logs";
#endif

    // Try each candidate path
    for (const QString& path : candidatePaths) {
        QDir dir(path);

        // Try to create the directory if it doesn't exist
        if (!dir.exists()) {
            if (dir.mkpath(".")) {
                // Successfully created directory
                return path;
            }
        } else {
            // Directory exists, check if writable
            QFileInfo dirInfo(path);
            if (dirInfo.isWritable()) {
                return path;
            }
        }
    }

    // If all else fails, return temp directory path (Qt should always have
    // access)
    return QStandardPaths::writableLocation(QStandardPaths::TempLocation);
}

bool ecvConsole::setLogFile(const QString& logPrefix) {
    // close previous stream (if any)
    if (m_logStream) {
        m_mutex.lock();
        delete m_logStream;
        m_logStream = nullptr;
        m_mutex.unlock();

        if (m_logFile.isOpen()) {
            m_logFile.close();
        }
    }

    if (!logPrefix.isEmpty()) {
        // Get appropriate log directory
        QString logDir = getLogDirectory();

        // Generate log file name with timestamp and PID
        QString logFileName = generateLogFileName(logPrefix);

        // Construct full log path
        QString logPath = logDir + "/" + logFileName;

        m_logFile.setFileName(logPath);
        if (!m_logFile.open(QFile::Text | QFile::WriteOnly | QFile::Append)) {
            return Error(
                    QString("[Console] Failed to open/create log file '%1'")
                            .arg(logPath));
        }

        // Log the actual log file path for user reference
        QString infoMsg =
                QString("[Console] Log file created: %1").arg(logPath);

        m_mutex.lock();
        m_logStream = new QTextStream(&m_logFile);
        // Set UTF-8 encoding for proper Chinese character support
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        m_logStream->setEncoding(QStringConverter::Utf8);
#else
        m_logStream->setCodec("UTF-8");
#endif
        // Write header to log file
        *m_logStream << "========================================"
                     << QtCompat::endl;
        *m_logStream << "ACloudViewer Log File" << QtCompat::endl;
        *m_logStream << "Started at: "
                     << QDateTime::currentDateTime().toString(
                                "yyyy-MM-dd HH:mm:ss")
                     << QtCompat::endl;
        *m_logStream << "Log file: " << logPath << QtCompat::endl;
        *m_logStream << "========================================"
                     << QtCompat::endl;
        m_mutex.unlock();

        setAutoRefresh(true);

        // Print info message to console
        Print(infoMsg);
    }

    return true;
}
