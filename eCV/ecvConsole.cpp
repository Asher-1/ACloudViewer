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

// ECV_DB_LIB
#include <ecvSingleton.h>

// CV_APP_COMMON
#include <CommonSettings.h>

// Qt
#include <QApplication>
#include <QClipboard>
#include <QColor>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QKeyEvent>
#include <QMessageBox>
#include <QStandardPaths>
#include <QTextStream>
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
ecvCustomQListWidget::ecvCustomQListWidget(QWidget* parent)
    : QListWidget(parent) {}

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

void ecvConsole::Init(QListWidget* textDisplay /*=0*/,
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
        for (QVector<ConsoleItemType>::const_iterator it = m_queue.constBegin();
             it != m_queue.constEnd(); ++it) {
            // it->second = message severity
            bool debugMessage = (it->second & LOG_VERBOSE);
#ifndef QT_DEBUG
            // skip debug message in release mode
            if (debugMessage) continue;
#endif

            // destination: console widget (log file is already written in
            // logMessage()) it->first = message text
            QListWidgetItem* item = new QListWidgetItem(it->first);

            // set color based on the message severity
            // Error
            if (it->second & LOG_ERROR) {
                item->setForeground(Qt::red);
            }
            // Warning
            else if (it->second & LOG_WARNING) {
                item->setForeground(Qt::darkRed);
                // we also force the console visibility if a warning message
                // arrives!
                if (m_parentWindow) m_parentWindow->forceConsoleDisplay();
            }
#ifdef QT_DEBUG
            else if (debugMessage) {
                item->setForeground(Qt::blue);
            }
#endif

            m_textDisplay->addItem(item);
        }

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
#ifndef QT_DEBUG
    // skip debug messages in release mode
    if (level & LOG_VERBOSE) {
        return;
    }
#endif

    // QString line = __LINE__;
    // QString filename = __FILE__;
    // QString functionname = __FUNCTION__;

    QString formatedMessage =
            QStringLiteral("[") + DATETIME + QStringLiteral("] ") + message;

    if (s_redirectToStdOut) {
        printf("%s\n", qPrintable(message));
    }
    if (m_textDisplay || m_logStream) {
        m_mutex.lock();

        // Write to log file immediately for crash safety (all messages)
        // UI update will still be handled by the timer for performance
        if (m_logStream) {
            *m_logStream << formatedMessage << endl;
            // Flush immediately for ERROR/WARNING, or every few messages for
            // others
            if ((level & LOG_ERROR) || (level & LOG_WARNING)) {
                m_logFile.flush();
            }
        }

        // Queue for UI update
        if (m_textDisplay) {
            m_queue.push_back(ConsoleItemType(formatedMessage, level));
        }

        m_mutex.unlock();
    }
#ifdef QT_DEBUG
    else {
        // Error
        if (level & LOG_ERROR) {
            if (level & LOG_VERBOSE)
                printf("ERR-DBG: ");
            else
                printf("ERR: ");
        }
        // Warning
        else if (level & LOG_WARNING) {
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
    if ((level & LOG_ERROR) && qApp && m_parentWidget &&
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
        // Write header to log file
        *m_logStream << "========================================" << endl;
        *m_logStream << "ACloudViewer Log File" << endl;
        *m_logStream << "Started at: "
                     << QDateTime::currentDateTime().toString(
                                "yyyy-MM-dd HH:mm:ss")
                     << endl;
        *m_logStream << "Log file: " << logPath << endl;
        *m_logStream << "========================================" << endl;
        m_mutex.unlock();

        setAutoRefresh(true);

        // Print info message to console
        Print(infoMsg);
    }

    return true;
}
