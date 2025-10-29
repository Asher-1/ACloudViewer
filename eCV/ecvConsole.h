// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QFile>
#include <QListWidget>
#include <QMutex>
#include <QTimer>

class MainWindow;
class QTextStream;

//! Custom QListWidget to allow for the copy of all selected elements when using
//! CTRL+C
class ecvCustomQListWidget : public QListWidget {
    Q_OBJECT

public:
    ecvCustomQListWidget(QWidget* parent = nullptr);

protected:
    void keyPressEvent(QKeyEvent* event) override;
};

//! Console
class ecvConsole : public QObject, public CVLog {
    Q_OBJECT

public:
    //! Destructor
    ~ecvConsole() override;

    //! Inits console (and optionaly associates it with a text output widget)
    /** WARNING: in release mode, no message will be output if no 'textDisplay'
            widget is defined. Moreover, error messages will only appear in a
            (blocking) QMessageBox if a 'parentWidget' widget is defined.
            In debug mode, all message are sent to system console (with
    'printf'). \param textDisplay text output widget (optional) \param
    parentWidget parent widget (optional) \param parentWindow parent window (if
    any - optional)
    **/
    static void Init(QListWidget* textDisplay = nullptr,
                     QWidget* parentWidget = nullptr,
                     MainWindow* parentWindow = nullptr,
                     bool redirectToStdOut = false);

    //! Returns the (unique) static instance
    /** \param autoInit automatically initialize the console instance (with no
     *widget!) if not done already
     **/
    static ecvConsole* TheInstance(bool autoInit = true);

    //! Releases unique instance
    static void ReleaseInstance(bool flush = true);

    //! Sets auto-refresh state
    void setAutoRefresh(bool state);

    //! Sets log file with prefix (generates timestamped log file like glog)
    /** \param logPrefix log file prefix (e.g., "ACloudviewer")
     *  \return true if successful, false otherwise
     *  Generates log file name: <prefix>.<timestamp>.<pid>.log
     **/
    bool setLogFile(const QString& logPrefix);

    //! Whether to show Qt messages (qDebug / qWarning / etc.) in Console
    static void EnableQtMessages(bool state);

    //! Returns whether to show Qt messages (qDebug / qWarning / etc.) in
    //! Console or not
    static bool QtMessagesEnabled() { return s_showQtMessagesInConsole; }

    //! Returns the parent widget (if any)
    inline QWidget* parentWidget() { return m_parentWidget; }

public slots:

    //! Refreshes console (display all messages still in queue)
    void refresh();

protected:
    //! Generate log file name with timestamp and pid
    static QString generateLogFileName(const QString& prefix);

    //! Get appropriate log directory path (handles permissions on Ubuntu)
    static QString getLogDirectory();
    //! Default constructor
    /** Constructor is protected to avoid using this object as a non static
     *class.
     **/
    ecvConsole();

    // inherited from CVLog
    void logMessage(const QString& message, int level) override;

    //! Associated text display widget
    QListWidget* m_textDisplay;

    //! Parent widget
    QWidget* m_parentWidget;

    //! Parent window (if any)
    MainWindow* m_parentWindow;

    //! Mutex for concurrent thread access to console
    QMutex m_mutex;

    //! Queue element type (message + color)
    using ConsoleItemType = QPair<QString, int>;

    //! Queue for incoming messages
    QVector<ConsoleItemType> m_queue;

    //! Timer for auto-refresh
    QTimer m_timer;

    //! Log file
    QFile m_logFile;
    //! Log file stream
    QTextStream* m_logStream;

    //! Whether to show Qt messages (qDebug / qWarning / etc.) in Console
    static bool s_showQtMessagesInConsole;
    static bool s_redirectToStdOut;
};
