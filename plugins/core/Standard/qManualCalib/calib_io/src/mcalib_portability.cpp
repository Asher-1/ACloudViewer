// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mcalib_portability.h"

#include <QString>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace mcalib {

std::string pathFromQString(const QString& path) {
    const QByteArray utf8 = path.toUtf8();
    return std::string(utf8.constData(), static_cast<size_t>(utf8.size()));
}

#if defined(_WIN32)

namespace {

std::wstring utf8ToWide(const std::string& utf8_path) {
    if (utf8_path.empty()) return {};

    const int required =
            MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, nullptr, 0);
    if (required <= 1) return {};

    std::vector<wchar_t> wide(static_cast<size_t>(required));
    MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, wide.data(),
                        required);
    return std::wstring(wide.data());
}

}  // namespace

bool openInputFile(std::ifstream& stream, const std::string& utf8_path) {
    stream.close();
    const std::wstring wide = utf8ToWide(utf8_path);
    if (wide.empty()) return false;
    stream.open(wide, std::ios::binary);
    return stream.is_open();
}

bool openOutputFile(std::ofstream& stream, const std::string& utf8_path) {
    stream.close();
    const std::wstring wide = utf8ToWide(utf8_path);
    if (wide.empty()) return false;
    stream.open(wide, std::ios::binary);
    return stream.is_open();
}

#else

bool openInputFile(std::ifstream& stream, const std::string& utf8_path) {
    stream.close();
    stream.open(utf8_path, std::ios::binary);
    return stream.is_open();
}

bool openOutputFile(std::ofstream& stream, const std::string& utf8_path) {
    stream.close();
    stream.open(utf8_path, std::ios::binary);
    return stream.is_open();
}

#endif

}  // namespace mcalib
