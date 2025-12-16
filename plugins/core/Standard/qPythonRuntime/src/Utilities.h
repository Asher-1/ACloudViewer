// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

#include <Python.h>

#include <CVLog.h>

/// Wrapper around CVLog to have a std::cout like API
///
/// # Example
///
/// ```cpp
/// ccPrint() << "Message";
/// ```
template <enum CVLog::MessageLevelFlags level> class ccLogger
{
  public:
    ccLogger() : m_message()
    {
        m_message.reserve(255);
    }

    virtual ~ccLogger()
    {
        flush();
    }

    inline ccLogger &operator<<(const QString &message)
    {
        m_message += message;
        return *this;
    }

    inline ccLogger &operator<<(const char *message)
    {
        m_message += message;
        return *this;
    }

    inline ccLogger &operator<<(int value)
    {
        m_message += QString::number(value);
        return *this;
    }

    inline ccLogger &operator<<(const qsizetype value)
    {
        m_message += QString::number(value);
        return *this;
    }

    void flush()
    {
        CVLog::LogMessage(m_message, level);
        m_message.clear();
    }

  protected:
    QString m_message;
};

using ccVerbose = ccLogger<CVLog::MessageLevelFlags::LOG_VERBOSE>;
using ccPrint = ccLogger<CVLog::MessageLevelFlags::LOG_STANDARD>;
using ccPrintHigh = ccLogger<CVLog::MessageLevelFlags::LOG_IMPORTANT>;
using ccWarning = ccLogger<CVLog::MessageLevelFlags::LOG_WARNING>;
using ccError = ccLogger<CVLog::MessageLevelFlags::LOG_ERROR>;

/// Logger Specialized for the plugin.
///
/// It prepends every message with `[PythonRuntime]`.
///
/// # Example
///
/// ```cpp
/// plgPrint() << "Message";
/// ```
template <enum CVLog::MessageLevelFlags level> class PluginLogger : public ccLogger<level>
{
  public:
    PluginLogger() : ccLogger<level>()
    {
        this->m_message += "[PythonRuntime] ";
    }

    //    friend PluginLogger& endl(PluginLogger<level>& logger) {
    //        logger.flush();
    //        return logger;
    //    }
};

using plgVerbose = PluginLogger<CVLog::LOG_VERBOSE>;
using plgPrint = PluginLogger<CVLog::LOG_STANDARD>;
using plgPrintHigh = PluginLogger<CVLog::LOG_IMPORTANT>;
using plgWarning = PluginLogger<CVLog::LOG_WARNING>;
using plgError = PluginLogger<CVLog::LOG_ERROR>;

/// Returns a newly allocated wchar_t array (null terminated) from a QString
inline wchar_t *QStringToWcharArray(const QString &string)
{
    auto *wcharArray = new wchar_t[string.size() + 1];
    const int len = string.toWCharArray(wcharArray);
    Q_ASSERT(len <= string.size());
    wcharArray[len] = '\0';
    return wcharArray;
}

/// Logs the PYTHON_PATH the log console of ACloudViewer
inline void LogPythonPath()
{
    const wchar_t *pythonPath = Py_GetPath();
    if (pythonPath != nullptr)
    {
        size_t errPos{0};
        char *cPythonPath = Py_EncodeLocale(pythonPath, &errPos);
        if (cPythonPath)
        {
            CVLog::Print("[PythonRuntime] PythonPath is set to: %s", cPythonPath);
            PyMem_Free(cPythonPath);
        }
        else
        {
            CVLog::Print("[PythonRuntime] Failed to convert the PythonPath");
        }
    }
    else
    {
        CVLog::Print("[PythonRuntime] PythonPath is not set");
    }
}

/// Logs the PYTHON_HOME the log console of ACloudViewer
inline void LogPythonHome()
{
    const wchar_t *pythonHome = Py_GetPythonHome();
    if (pythonHome != nullptr)
    {
        size_t errPos{0};
        char *cPythonHome = Py_EncodeLocale(pythonHome, &errPos);
        if (cPythonHome)
        {
            CVLog::Print("[PythonRuntime] PythonHome is set to: %s", cPythonHome);
            PyMem_Free(cPythonHome);
        }
        else
        {
            CVLog::Print("[PythonRuntime]Failed to convert the PythonHome path");
        }
    }
    else
    {
        CVLog::Print("[PythonRuntime] PythonHome is not set");
    }
}
