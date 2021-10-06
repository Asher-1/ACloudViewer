// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                        -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include "CVCoreLib.h"

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY 1
#endif
#ifndef FMT_STRING_ALIAS
#define FMT_STRING_ALIAS 1
#endif
// NVCC does not support deprecated attribute on Windows prior to v11.
#if defined(__CUDACC__) && defined(_MSC_VER) && __CUDACC_VER_MAJOR__ < 11
#ifndef FMT_DEPRECATED
#define FMT_DEPRECATED
#endif
#endif
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#define DEFAULT_IO_BUFFER_SIZE 1024

// Compiler-specific function macro.
// Ref: https://stackoverflow.com/a/4384825
#ifdef _WIN32
#define __FN__ __FUNCSIG__
#else
#define __FN__ __PRETTY_FUNCTION__
#endif

// Mimic "macro in namespace" by concatenating `utility::` and a macro.
// Ref: https://stackoverflow.com/a/11791202
//
// We avoid using (format, ...) since in this case __VA_ARGS__ can be
// empty, and the behavior of pruning trailing comma with ##__VA_ARGS__ is not
// officially standard.
// Ref: https://stackoverflow.com/a/28074198
//
// __PRETTY_FUNCTION__ has to be converted, otherwise a bug regarding [noreturn]
// will be triggered.
// Ref: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94742

// LogError throws now a runtime_error with the given error message. This
// should be used if there is no point in continuing the given algorithm at
// some point and the error is not returned in another way (e.g., via a
// bool/int as return value).
//
// Usage  : utility::LogError(format_string, arg0, arg1, ...);
// Example: utility::LogError("name: {}, age: {}", "dog", 5);
#define LogError(...)                                                  \
    Logger::_LogError(__FILE__, __LINE__, (const char *)__FN__, false, \
                      __VA_ARGS__)
// Same as LogError but enforce printing the message in the console.
#define LogErrorConsole(...)                                          \
    Logger::_LogError(__FILE__, __LINE__, (const char *)__FN__, true, \
                      __VA_ARGS__)

// LogWarning is used if an error occurs, but the error is also signaled
// via a return value (i.e., there is no need to throw an exception). This
// warning should further be used, if the algorithms encounters a state
// that does not break its continuation, but the output is likely not to be
// what the user expected.
//
// Usage  : utility::LogWarning(format_string, arg0, arg1, ...);
// Example: utility::LogWarning("name: {}, age: {}", "dog", 5);
#define LogWarning(...)                                                  \
    Logger::_LogWarning(__FILE__, __LINE__, (const char *)__FN__, false, \
                        __VA_ARGS__)
// Same as LogWarning but enforce printing the message in the console.
#define LogWarningConsole(...)                                          \
    Logger::_LogWarning(__FILE__, __LINE__, (const char *)__FN__, true, \
                        __VA_ARGS__)

// LogInfo is used to inform the user with expected output, e.g, pressed a
// key in the visualizer prints helping information.
//
// Usage  : utility::LogInfo(format_string, arg0, arg1, ...);
// Example: utility::LogInfo("name: {}, age: {}", "dog", 5);
#define LogInfo(...)                                                  \
    Logger::_LogInfo(__FILE__, __LINE__, (const char *)__FN__, false, \
                     __VA_ARGS__)
// Same as LogInfo but enforce printing the message in the console.
#define LogInfoConsole(...)                                          \
    Logger::_LogInfo(__FILE__, __LINE__, (const char *)__FN__, true, \
                     __VA_ARGS__)

// LogDebug is used to print debug/additional information on the state of
// the algorithm.
//
// Usage  : utility::LogDebug(format_string, arg0, arg1, ...);
// Example: utility::LogDebug("name: {}, age: {}", "dog", 5);
#define LogDebug(...)                                                  \
    Logger::_LogDebug(__FILE__, __LINE__, (const char *)__FN__, false, \
                      __VA_ARGS__)
// Same as LogDebug but enforce printing the message in the console.
#define LogDebugConsole(...)                                          \
    Logger::_LogDebug(__FILE__, __LINE__, (const char *)__FN__, true, \
                      __VA_ARGS__)

namespace cloudViewer {
namespace utility {

enum class CV_CORE_LIB_API VerbosityLevel {
    /// LogError throws now a runtime_error with the given error message. This
    /// should be used if there is no point in continuing the given algorithm at
    /// some point and the error is not returned in another way (e.g., via a
    /// bool/int as return value).
    Error = 0,
    /// LogWarning is used if an error occurs, but the error is also signaled
    /// via a return value (i.e., there is no need to throw an exception). This
    /// warning should further be used, if the algorithms encounters a state
    /// that does not break its continuation, but the output is likely not to be
    /// what the user expected.
    Warning = 1,
    /// LogInfo is used to inform the user with expected output, e.g, pressed a
    /// key in the visualizer prints helping information.
    Info = 2,
    /// LogDebug is used to print debug/additional information on the state of
    /// the algorithm.
    Debug = 3,
};

enum class CV_CORE_LIB_API TextColor {
    Black = 0,
    Red = 1,
    Green = 2,
    Yellow = 3,
    Blue = 4,
    Magenta = 5,
    Cyan = 6,
    White = 7
};

/// Logger class should be used as a global singleton object (GetInstance()).
class CV_CORE_LIB_API Logger {
public:
    struct CV_CORE_LIB_API Impl {
        // The current print function.
        std::function<void(const std::string &)> print_fcn_;

        // The default print function (that prints to console).
        static std::function<void(const std::string &)> console_print_fcn_;

        // Verbosity level.
        VerbosityLevel verbosity_level_;

        // Colorize and reset the color of a string, does not work on Windows,
        std::string ColorString(const std::string &text,
                                TextColor text_color,
                                int highlight_text) const {
            std::ostringstream msg;
#ifndef _WIN32
            msg << fmt::sprintf("%c[%d;%dm", 0x1B, highlight_text,
                                (int)text_color + 30);
#endif
            msg << text;
#ifndef _WIN32
            msg << fmt::sprintf("%c[0;m", 0x1B);
#endif
            return msg.str();
        }
    };

public:
    Logger(Logger const &) = delete;
    void operator=(Logger const &) = delete;

    /// Get Logger global singleton instance.
    static Logger &GetInstance();

    /// Overwrite the default print function, this is useful when you want to
    /// redirect prints rather than printing to stdout. For example, in CloudViewer's
    /// python binding, the default print function is replaced with py::print().
    ///
    /// \param print_fcn The function for printing. It should take a string
    /// input and returns nothing.
    void SetPrintFunction(std::function<void(const std::string &)> print_fcn);

    /// reset the print function to the default one (print to console).
    void ResetPrintFunction();

    /// Set global verbosity level of CloudViewer.
    ///
    /// \param verbosity_level Messages with equal or less than verbosity_level
    /// verbosity will be printed.
    void SetVerbosityLevel(VerbosityLevel verbosity_level);

    /// Get global verbosity level of CloudViewer.
    VerbosityLevel GetVerbosityLevel() const;

    template <typename... Args>
    static void _LogError [[noreturn]] (const char *file_name,
                                        int line_number,
                                        const char *function_name,
                                        bool force_console_log,
                                        const char *format,
                                        Args &&... args) {
        Logger::GetInstance().VError(file_name, line_number, function_name,
                                     force_console_log, format,
                                     fmt::make_format_args(args...));
    }
    template <typename... Args>
    static void _LogWarning(const char *file_name,
                            int line_number,
                            const char *function_name,
                            bool force_console_log,
                            const char *format,
                            Args &&... args) {
        Logger::GetInstance().VWarning(file_name, line_number, function_name,
                                       force_console_log, format,
                                       fmt::make_format_args(args...));
    }
    template <typename... Args>
    static void _LogInfo(const char *file_name,
                         int line_number,
                         const char *function_name,
                         bool force_console_log,
                         const char *format,
                         Args &&... args) {
        Logger::GetInstance().VInfo(file_name, line_number, function_name,
                                    force_console_log, format,
                                    fmt::make_format_args(args...));
    }
    template <typename... Args>
    static void _LogDebug(const char *file_name,
                          int line_number,
                          const char *function_name,
                          bool force_console_log,
                          const char *format,
                          Args &&... args) {
        Logger::GetInstance().VDebug(file_name, line_number, function_name,
                                     force_console_log, format,
                                     fmt::make_format_args(args...));
    }

private:
    Logger();
    void VError [[noreturn]] (const char *file_name,
                              int line_number,
                              const char *function_name,
                              bool force_console_log,
                              const char *format,
                              fmt::format_args args) const;
    void VWarning(const char *file_name,
                  int line_number,
                  const char *function_name,
                  bool force_console_log,
                  const char *format,
                  fmt::format_args args) const;
    void VInfo(const char *file_name,
               int line_number,
               const char *function_name,
               bool force_console_log,
               const char *format,
               fmt::format_args args) const;
    void VDebug(const char *file_name,
                int line_number,
                const char *function_name,
                bool force_console_log,
                const char *format,
                fmt::format_args args) const;

private:
    std::unique_ptr<Impl> impl_;
};

/// Set global verbosity level of CloudViewer
///
/// \param level Messages with equal or less than verbosity_level verbosity will
/// be printed.
void CV_CORE_LIB_API SetVerbosityLevel(VerbosityLevel level);

/// Get global verbosity level of CloudViewer.
VerbosityLevel CV_CORE_LIB_API GetVerbosityLevel();

class CV_CORE_LIB_API VerbosityContextManager {
public:
    VerbosityContextManager(VerbosityLevel level) : level_(level) {}

    void Enter() {
        level_backup_ = Logger::GetInstance().GetVerbosityLevel();
        Logger::GetInstance().SetVerbosityLevel(level_);
    }

    void Exit() { Logger::GetInstance().SetVerbosityLevel(level_backup_); }

private:
    VerbosityLevel level_;
    VerbosityLevel level_backup_;
};

class CV_CORE_LIB_API ConsoleProgressBar {
public:
    ConsoleProgressBar(size_t expected_count,
                       const std::string &progress_info,
                       bool active = false);

    void reset(size_t expected_count,
               const std::string &progress_info,
               bool active);

    ConsoleProgressBar &operator++();

    void setCurrentCount(size_t n);

private:
    const size_t resolution_ = 40;
    size_t expected_count_;
    size_t current_count_;
    std::string progress_info_;
    size_t progress_pixel_;
    bool active_;
};

}  // namespace utility
}  // namespace cloudViewer
