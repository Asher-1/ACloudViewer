// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef FILE_SYSTEM_UTILS_H
#define FILE_SYSTEM_UTILS_H

#include <cstdio>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cloudViewer {
namespace io {

#ifdef _WIN32
/**
 * @brief Open a file using UTF-8 encoded path on Windows
 * 
 * This function converts UTF-8 encoded file path to UTF-16 (wchar_t)
 * and uses _wfopen on Windows to properly handle Unicode file names.
 * 
 * @param filename UTF-8 encoded file path
 * @param mode File open mode ("r", "w", "rb", "wb", etc.)
 * @return FILE* pointer or nullptr if failed
 */
inline FILE* FOpenUTF8(const char* filename, const char* mode) {
    // Convert UTF-8 filename to UTF-16
    int wchars_num = MultiByteToWideChar(CP_UTF8, 0, filename, -1, nullptr, 0);
    if (wchars_num == 0) return nullptr;
    
    wchar_t* wfilename = new wchar_t[wchars_num];
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wfilename, wchars_num);
    
    // Convert UTF-8 mode to UTF-16
    int wmode_num = MultiByteToWideChar(CP_UTF8, 0, mode, -1, nullptr, 0);
    if (wmode_num == 0) {
        delete[] wfilename;
        return nullptr;
    }
    
    wchar_t* wmode = new wchar_t[wmode_num];
    MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, wmode_num);
    
    FILE* fp = _wfopen(wfilename, wmode);
    
    delete[] wfilename;
    delete[] wmode;
    
    return fp;
}

/**
 * @brief Open an ifstream using UTF-8 encoded path on Windows
 * 
 * @param filename UTF-8 encoded file path
 * @param mode Open mode flags
 * @return std::ifstream object
 */
inline std::ifstream IFStreamUTF8(const char* filename, 
                                  std::ios_base::openmode mode = std::ios_base::in) {
    int wchars_num = MultiByteToWideChar(CP_UTF8, 0, filename, -1, nullptr, 0);
    if (wchars_num == 0) return std::ifstream();
    
    wchar_t* wfilename = new wchar_t[wchars_num];
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wfilename, wchars_num);
    
    std::ifstream stream(wfilename, mode);
    
    delete[] wfilename;
    
    return stream;
}

/**
 * @brief Open an ofstream using UTF-8 encoded path on Windows
 * 
 * @param filename UTF-8 encoded file path
 * @param mode Open mode flags
 * @return std::ofstream object
 */
inline std::ofstream OFStreamUTF8(const char* filename, 
                                  std::ios_base::openmode mode = std::ios_base::out) {
    int wchars_num = MultiByteToWideChar(CP_UTF8, 0, filename, -1, nullptr, 0);
    if (wchars_num == 0) return std::ofstream();
    
    wchar_t* wfilename = new wchar_t[wchars_num];
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wfilename, wchars_num);
    
    std::ofstream stream(wfilename, mode);
    
    delete[] wfilename;
    
    return stream;
}

#else
// On non-Windows platforms, use standard fopen (assumes UTF-8 locale)
inline FILE* FOpenUTF8(const char* filename, const char* mode) {
    return fopen(filename, mode);
}

inline std::ifstream IFStreamUTF8(const char* filename, 
                                  std::ios_base::openmode mode = std::ios_base::in) {
    return std::ifstream(filename, mode);
}

inline std::ofstream OFStreamUTF8(const char* filename, 
                                  std::ios_base::openmode mode = std::ios_base::out) {
    return std::ofstream(filename, mode);
}
#endif

}  // namespace io
}  // namespace cloudViewer

#endif  // FILE_SYSTEM_UTILS_H
