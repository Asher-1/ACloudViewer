// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <glog/logging.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "util/string.h"

// Option checker macros. In contrast to glog, this function does not abort the
// program, but simply returns false on failure.
#define CHECK_OPTION_IMPL(expr) \
    __CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)
#define CHECK_OPTION(expr)                                       \
    if (!__CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)) { \
        return false;                                            \
    }
#define CHECK_OPTION_OP(name, op, val1, val2)                                \
    if (!__CheckOptionOpImpl(__FILE__, __LINE__, (val1 op val2), val1, val2, \
                             #val1, #val2, #op)) {                           \
        return false;                                                        \
    }
#define CHECK_OPTION_EQ(val1, val2) CHECK_OPTION_OP(_EQ, ==, val1, val2)
#define CHECK_OPTION_NE(val1, val2) CHECK_OPTION_OP(_NE, !=, val1, val2)
#define CHECK_OPTION_LE(val1, val2) CHECK_OPTION_OP(_LE, <=, val1, val2)
#define CHECK_OPTION_LT(val1, val2) CHECK_OPTION_OP(_LT, <, val1, val2)
#define CHECK_OPTION_GE(val1, val2) CHECK_OPTION_OP(_GE, >=, val1, val2)
#define CHECK_OPTION_GT(val1, val2) CHECK_OPTION_OP(_GT, >, val1, val2)

namespace colmap {

// Initialize glog at the beginning of the program.
void InitializeGlog(char** argv);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

const char* __GetConstFileBaseName(const char* file);

bool __CheckOptionImpl(const char* file,
                       const int line,
                       const bool result,
                       const char* expr_str);

template <typename T1, typename T2>
bool __CheckOptionOpImpl(const char* file,
                         const int line,
                         const bool result,
                         const T1& val1,
                         const T2& val2,
                         const char* val1_str,
                         const char* val2_str,
                         const char* op_str) {
    if (result) {
        return true;
    } else {
        std::cerr
                << StringPrintf(
                           "[WARNING %s:%d] Check failed: %s %s %s (%s vs. %s)",
                           __GetConstFileBaseName(file), line, val1_str, op_str,
                           val2_str, std::to_string(val1).c_str(),
                           std::to_string(val2).c_str())
                << "\n";
        return false;
    }
}

// ----------------------------------------------------------------------------
// Upstream-parity throwing checks (THROW_CHECK family). The full upstream
// LogMessageFatalThrow machinery is simplified here: these macros throw
// std::invalid_argument with a file:line prefix instead of relying on glog
// internals. Streaming attachment (THROW_CHECK(x) << msg) is not supported by
// the simplified form; ported code must fold the message into the condition
// comment or use LOG_FATAL_THROW.
// ----------------------------------------------------------------------------

inline std::string __ThrowCheckPrefix(const char* file, int line) {
    return "[" + std::string(__GetConstFileBaseName(file)) + ":" +
           std::to_string(line) + "] ";
}

inline void __ThrowCheckImpl(const bool ok, const std::string& message) {
    if (!ok) {
        throw std::invalid_argument(message);
    }
}

template <typename T>
T ThrowCheckNotNull(const char* file, int line, const char* names, T&& t) {
    if (t == nullptr) {
        throw std::invalid_argument(__ThrowCheckPrefix(file, line) + "'" +
                                    names + "' Must be non NULL");
    }
    return std::forward<T>(t);
}

// Stream buffer that throws the configured exception type on destruction,
// mirroring upstream LOG(FATAL_THROW) semantics.
template <typename T>
class LogFatalThrowStream {
public:
    LogFatalThrowStream(const char* file, int line)
        : prefix_(__ThrowCheckPrefix(file, line)) {}

    LogFatalThrowStream(const LogFatalThrowStream&) = delete;
    LogFatalThrowStream& operator=(const LogFatalThrowStream&) = delete;

    std::ostream& stream() { return stream_; }

    ~LogFatalThrowStream() noexcept(false) {
        if (std::uncaught_exceptions() == 0) {
            throw T(prefix_ + stream_.str());
        }
    }

private:
    std::string prefix_;
    std::ostringstream stream_;
};


// ----------------------------------------------------------------------------
// Streaming-capable THROW_CHECK family (upstream parity). A temporary
// ThrowCheckStream is bound to the condition; if the condition is false its
// destructor throws after the streamed context has been appended. Streaming
// returns the same object so upstream call sites compile unchanged.
// ----------------------------------------------------------------------------
class ThrowCheckStream {
 public:
    ThrowCheckStream(const bool ok, const char* file, const int line,
                     const std::string& message)
        : ok_(ok), file_(file), line_(line), message_(message) {}

    ~ThrowCheckStream() noexcept(false) {
        if (!ok_) {
            throw std::invalid_argument(
                    "[" + std::string(__GetConstFileBaseName(file_)) + ":" +
                    std::to_string(line_) + "] " + message_ +
                    stream_.str());
        }
    }

    template <typename T>
    ThrowCheckStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

    std::ostringstream& stream() { return stream_; }

 private:
    bool ok_;
    const char* file_;
    int line_;
    std::string message_;
    std::ostringstream stream_;
};

#undef THROW_CHECK
#define THROW_CHECK(condition) \
    colmap::ThrowCheckStream( \
            static_cast<bool>(condition), __FILE__, __LINE__, \
            std::string("Check failed: ") + #condition)

#undef THROW_CHECK_OP
#define THROW_CHECK_OP(op, a, b) \
    colmap::ThrowCheckStream( \
            static_cast<bool>((a) op (b)), __FILE__, __LINE__, \
            std::string("Check failed: ") + #a + " " + #op + " " + #b)

#undef THROW_CHECK_EQ
#define THROW_CHECK_EQ(a, b) THROW_CHECK_OP(==, a, b)
#undef THROW_CHECK_NE
#define THROW_CHECK_NE(a, b) THROW_CHECK_OP(!=, a, b)
#undef THROW_CHECK_LE
#define THROW_CHECK_LE(a, b) THROW_CHECK_OP(<=, a, b)
#undef THROW_CHECK_LT
#define THROW_CHECK_LT(a, b) THROW_CHECK_OP(<, a, b)
#undef THROW_CHECK_GE
#define THROW_CHECK_GE(a, b) THROW_CHECK_OP(>=, a, b)
#undef THROW_CHECK_GT
#define THROW_CHECK_GT(a, b) THROW_CHECK_OP(>, a, b)

}  // namespace colmap

#define THROW_CHECK_NOTNULL(val)                  \
    colmap::ThrowCheckNotNull(__FILE__, __LINE__, \
                              "'" #val "' Must be non NULL", (val))

#define LOG_FATAL_THROW(exception) \
    colmap::LogFatalThrowStream<exception>(__FILE__, __LINE__).stream()
