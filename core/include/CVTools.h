// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// local
#include "CVCoreLib.h"
#include "QtCompat.h"

// system
#include <QElapsedTimer>
#include <QString>
#include <string>
#include <vector>

class CVTools {
public:
    CV_CORE_LIB_API static std::string GetFileName(const std::string file_name);
    CV_CORE_LIB_API static void TimeStart();
    CV_CORE_LIB_API static QString TimeOff();

    CV_CORE_LIB_API static QString ToNativeSeparators(const QString& path);

    // string to QString
    CV_CORE_LIB_API static QString ToQString(const std::string& s);
    // QString to string
    CV_CORE_LIB_API static std::string FromQString(const QString& qs);

    CV_CORE_LIB_API static std::string JoinStrVec(
            const std::vector<std::string>& v, std::string splitor = " ");

    // QString(Unicode) -> std::string (GBK)
    CV_CORE_LIB_API static std::string FromUnicode(const QString& qstr);

    // std::string (GBK) -> QString(Unicode)
    CV_CORE_LIB_API static QString ToUnicode(const std::string& cstr);

    CV_CORE_LIB_API static std::string ExtractDigitAlpha(
            const std::string& str);

    CV_CORE_LIB_API static int TranslateKeyCode(int key);

    CV_CORE_LIB_API static bool FileMappingReader(const std::string& filename,
                                                  void* data,
                                                  unsigned long& size);
    CV_CORE_LIB_API static bool FileMappingWriter(const std::string& filename,
                                                  const void* data,
                                                  unsigned long size);

    CV_CORE_LIB_API static bool QMappingReader(const std::string& filename,
                                               std::vector<size_t>& indices);
    CV_CORE_LIB_API static bool QMappingWriter(const std::string& filename,
                                               const void* data,
                                               std::size_t length);
    CV_CORE_LIB_API static std::wstring Char2Wchar(const char* szStr);
    CV_CORE_LIB_API static std::string Wchar2Char(const wchar_t* szStr);

private:
    CVTools() = delete;

    static QElapsedTimer s_time;
};
