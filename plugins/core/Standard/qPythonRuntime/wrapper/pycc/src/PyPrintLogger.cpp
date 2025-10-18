// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PyPrintLogger.h"

void PyPrintLogger::logMessage(const QString &message, int level)
{
    std::lock_guard<std::mutex> guard(m_lock);
    const std::string stdMsg = message.toStdString();
    py::print(stdMsg.c_str());
}
