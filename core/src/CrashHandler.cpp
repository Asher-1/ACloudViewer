// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CrashHandler.h"

#include <QCoreApplication>
#include <QSysInfo>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>

#ifdef _WIN32
// dbghelp.h needs the Win32 types (CALLBACK, HANDLE, ...) from windows.h.
// The style regroups and sorts includes across blank lines, so pin the order.
// clang-format off
#include <windows.h>
#include <dbghelp.h>
// clang-format on
#else
#include <fcntl.h>
#include <unistd.h>

#include <csignal>
#if defined(__GLIBC__) || defined(__APPLE__)
#include <dlfcn.h>
#include <execinfo.h>
#define ECV_HAS_BACKTRACE 1
#endif
#endif

namespace ecvCrashHandler {

namespace {

// A crash report is evidence of an abnormal path, not a session marker. Keep
// only preallocated path/environment bytes at install; create the file lazily
// on the first abnormal write. POSIX open()/write() use those bytes directly
// from the fatal-signal path, without touching Qt or allocating memory.
int s_logFd = -1;
constexpr size_t kMaxCrashPathBytes = 4096;
constexpr size_t kMaxEnvironmentBytes = 768;
char s_logPath[kMaxCrashPathBytes] = {};
char s_environment[kMaxEnvironmentBytes] = {};
size_t s_environmentLen = 0;
volatile sig_atomic_t s_inHandler = 0;
#ifdef _WIN32
volatile LONG s_inHandlerWin = 0;
#endif

#if defined(ECV_HAS_BACKTRACE)
// Throw-site stack snapshot: __cxa_throw runs BEFORE any unwinding, so
// these frames survive the unwind that destroys the real stack. Written
// by the __cxa_throw interposer below (only once the collector is
// installed), read by the caught-exception recorder — both outside
// signal context; the spinlock only ever guards a fixed 512-byte copy.
constexpr int kMaxThrowFrames = 64;
std::atomic_flag s_throwSnapLock = ATOMIC_FLAG_INIT;
void *s_throwSnap[kMaxThrowFrames] = {};
int s_throwSnapCount = 0;
#endif

#ifndef _WIN32
void ensureCrashLogOpen() {
    if (s_logFd < 0 && s_logPath[0] != '\0') {
        s_logFd = open(s_logPath, O_WRONLY | O_CREAT | O_APPEND, 0644);
    }
}
#endif

void writeRaw(const char *text, size_t len) {
#ifdef _WIN32
    HANDLE h = CreateFileA(s_logPath, FILE_APPEND_DATA, 0, nullptr, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        SetFilePointer(h, 0, nullptr, FILE_END);
        DWORD written = 0;
        WriteFile(h, text, static_cast<DWORD>(len), &written, nullptr);
        CloseHandle(h);
    }
#else
    ensureCrashLogOpen();
    if (s_logFd >= 0) {
        if (write(s_logFd, text, len) < 0) {
            // nothing sane to do inside a fatal-signal handler
        }
    }
#endif
}

void writeLine(const char *text) {
    writeRaw(text, strlen(text));
    writeRaw("\n", 1);
}

void writeEnvironmentBlock() {
    if (s_environmentLen > 0) {
        writeRaw(s_environment, s_environmentLen);
        writeRaw("\n", 1);
    }
}

// Signal-safe POSIX timestamp: time() IS async-signal-safe; snprintf is
// NOT (locale/buffer internals). Only decimal digits are emitted.
void writeStampSignalSafe(const char *reason) {
    writeRaw("\n===== CRASH ", 13);
    writeRaw(reason, strlen(reason));
    writeRaw(" unix_time=", 11);
    char digits[24];
    const long t = static_cast<long>(time(nullptr));
    int pos = 0;
    if (t <= 0) {
        digits[pos++] = '0';
    } else {
        char tmp[24];
        int len = 0;
        long v = t;
        while (v > 0) {
            tmp[len++] = char('0' + (v % 10));
            v /= 10;
        }
        while (len > 0) digits[pos++] = tmp[--len];
    }
    digits[pos] = 0;
    writeRaw(digits, static_cast<size_t>(pos));
    writeRaw(" =====\n", 7);
}

#ifdef _WIN32
void writeStampWindows(const char *reason) {
    char stamp[160];
    std::time_t now = std::time(nullptr);
    std::tm tmBuf{};
    localtime_s(&tmBuf, &now);
    std::snprintf(stamp, sizeof(stamp),
                  "\n===== CRASH %s %04d-%02d-%02d %02d:%02d:%02d =====\n",
                  reason, tmBuf.tm_year + 1900, tmBuf.tm_mon + 1, tmBuf.tm_mday,
                  tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec);
    writeLine(stamp);
}

void writeWindowsStack() {
    void *frames[62];
    USHORT n = CaptureStackBackTrace(0, 62, frames, nullptr);
    char line[64];
    for (USHORT i = 0; i < n; ++i) {
        std::snprintf(line, sizeof(line), "#%d 0x%p\n", int(i), frames[i]);
        writeLine(line);
    }
    writeLine("===== END CRASH =====\n");
}
#endif

#if !defined(_WIN32)
extern "C" void posixSignalHandler(int sig) {
    // Re-entrant fault inside the handler: restore the default disposition
    // and re-raise so cores / OS-level reporting still happen.
    if (s_inHandler) {
        std::signal(sig, SIG_DFL);
        std::raise(sig);
        return;
    }
    s_inHandler = 1;

    const char *name = "UNKNOWN";
    switch (sig) {
        case SIGSEGV:
            name = "SIGSEGV";
            break;
        case SIGBUS:
            name = "SIGBUS";
            break;
        case SIGFPE:
            name = "SIGFPE";
            break;
        case SIGILL:
            name = "SIGILL";
            break;
        case SIGABRT:
            name = "SIGABRT";
            break;
        default:
            break;
    }
    writeStampSignalSafe(name);
    writeEnvironmentBlock();
#if defined(ECV_HAS_BACKTRACE)
    // backtrace_symbols_fd is async-signal-safe on glibc (it writes to the
    // fd directly); the stack must be captured BEFORE any further fault.
    void *frames[64];
    const int n = backtrace(frames, 64);
    if (s_logFd >= 0 && n > 0) {
        backtrace_symbols_fd(frames, n, s_logFd);
    } else
#endif
    {
        writeLine("(no backtrace support on this platform)");
    }
    writeLine("===== END CRASH =====\n");

    // Restore the default disposition and re-raise so cores / OS-level
    // reporting keep working.
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void terminateHandler() {
    if (s_inHandler) {
        // The fault path is already writing (or has written); fall through
        // to abort, which lands in the SIGABRT handler above (suppressed
        // there by the re-entrancy flag).
        std::abort();
    }
    s_inHandler = 1;
    writeStampSignalSafe("UNCAUGHT C++ EXCEPTION (std::terminate)");
    writeEnvironmentBlock();
#if defined(ECV_HAS_BACKTRACE)
    void *frames[64];
    const int n = backtrace(frames, 64);
    if (s_logFd >= 0 && n > 0) {
        backtrace_symbols_fd(frames, n, s_logFd);
    }
#endif
    writeLine("===== END CRASH =====\n");
    std::abort();  // reaches the SIGABRT handler path above
}

// Snapshot the CURRENT backtrace into the throw-site buffer (fixed-size,
// allocation-free apart from backtrace()'s lazy unwind-table init).
void snapshotThrowSite() {
    if (s_logPath[0] == '\0') {
        // Collector not installed: keep the interposition side-effect-free
        // for every other consumer of libCVCoreLib.
        return;
    }
    void *frames[kMaxThrowFrames];
    const int n = backtrace(frames, kMaxThrowFrames);
    while (s_throwSnapLock.test_and_set(std::memory_order_acquire)) {
    }
    s_throwSnapCount = n > 0 ? n : 0;
    if (s_throwSnapCount > 0) {
        std::memcpy(s_throwSnap, frames,
                    static_cast<size_t>(s_throwSnapCount) * sizeof(void *));
    }
    s_throwSnapLock.clear(std::memory_order_release);
}

// Dumps the last throw-site stack (raw addresses — same format as the
// signal path; symbolize with addr2line / llvm-symbolizer).
void writeThrowSiteStack() {
    while (s_throwSnapLock.test_and_set(std::memory_order_acquire)) {
    }
    const int n = s_throwSnapCount;
    void *frames[kMaxThrowFrames];
    std::memcpy(frames, s_throwSnap, sizeof(frames));
    s_throwSnapLock.clear(std::memory_order_release);

    if (n <= 0) {
        writeLine("(no throw-site stack captured)");
        return;
    }
    writeLine("throw-site stack (captured at __cxa_throw, before unwinding):");
    backtrace_symbols_fd(frames, n, s_logFd);
}
#endif  // !defined(_WIN32)

}  // namespace

#if defined(ECV_HAS_BACKTRACE)
// Itanium C++ ABI interposition: EVERY throw passes through __cxa_throw
// BEFORE any unwinding starts. Capturing the backtrace here is the only
// way to keep the throw-site stack available to the top-level CAUGHT-
// exception recorder — by the time main()'s catch runs, the stack has
// unwound and is unrecoverable. The definition must keep default
// visibility (core/CMakeLists.txt builds CVCoreLib with hidden
// visibility), otherwise the dynamic linker keeps resolving __cxa_throw
// to libstdc++ and the hook never fires.
extern "C" __attribute__((visibility("default"))) void __cxa_throw(
        void *thrownException, void *typeInfo, void (*destructor)(void *)) {
    using CxaThrowFn = void (*)(void *, void *, void (*)(void *));
    // Magic static: resolved once, thread-safely, outside any signal
    // context (first throw).
    static CxaThrowFn real =
            reinterpret_cast<CxaThrowFn>(dlsym(RTLD_NEXT, "__cxa_throw"));
    snapshotThrowSite();
    if (real) {
        real(thrownException, typeInfo, destructor);
    }
    // __cxa_throw never returns; reaching this line means the real symbol
    // could not be resolved — fail loudly instead of corrupting unwinding.
    std::abort();
}
#endif  // ECV_HAS_BACKTRACE

void logUnhandledException(const QString &what) {
#ifdef _WIN32
    writeStampWindows("CAUGHT C++ EXCEPTION (top-level handler)");
    writeEnvironmentBlock();
    writeLine(what.toUtf8().constData());
    writeLine(
            "(no stack: the exception was caught by the application and "
            "the stack has unwound)");
    writeLine("===== END CRASH =====\n");
#else
    writeStampSignalSafe("CAUGHT C++ EXCEPTION (top-level handler)");
    writeEnvironmentBlock();
    writeLine(what.toUtf8().constData());
#if defined(ECV_HAS_BACKTRACE)
    // The stack HAS unwound — but the interposer captured the throw site
    // before it did.
    writeThrowSiteStack();
#else
    writeLine(
            "(no stack: the exception was caught by the application and "
            "the stack has unwound)");
#endif
    writeLine("===== END CRASH =====\n");
#endif
}

void install(const QString &logFilePath) {
    const QByteArray pathUtf8 = logFilePath.toUtf8();
    std::snprintf(s_logPath, sizeof(s_logPath), "%s", pathUtf8.constData());
    const QByteArray appName = QCoreApplication::applicationName().toUtf8();
    const QByteArray osName = QSysInfo::prettyProductName().toUtf8();
    const QByteArray productType = QSysInfo::productType().toUtf8();
    const QByteArray cpuArch = QSysInfo::currentCpuArchitecture().toUtf8();
    const QByteArray buildArch = QSysInfo::buildCpuArchitecture().toUtf8();
    const int environmentLen = std::snprintf(
            s_environment, sizeof(s_environment),
            "app=%s | Qt=%s | OS=%s (%s) | arch=%s | build-abi=%s",
            appName.constData(), qVersion(), osName.constData(),
            productType.constData(), cpuArch.constData(),
            buildArch.constData());
    s_environmentLen = environmentLen > 0
                               ? std::min(static_cast<size_t>(environmentLen),
                                          sizeof(s_environment) - 1)
                               : 0;

#ifdef _WIN32
    // LPTOP_LEVEL_EXCEPTION_FILTER: unhandled SEH (access violations, ...).
    // Uncaught C++ exceptions from MSVC land here too (they surface as SEH);
    // MSVC's abort()/pure-virtual-call fast-fails bypass SEH and cannot be
    // intercepted — documented limitation.
    SetUnhandledExceptionFilter([](EXCEPTION_POINTERS * /*info*/) -> LONG {
        // Interlocked re-entrancy guard (multiple threads may fault).
        if (InterlockedExchange(&s_inHandlerWin, 1) != 0) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        writeStampWindows("UNHANDLED SEH EXCEPTION");
        writeEnvironmentBlock();
        writeWindowsStack();
        return EXCEPTION_CONTINUE_SEARCH;  // default handling (WER, ...)
    });
    // MSVC terminate hook for the C++-exception cases that do NOT become
    // SEH (e.g. noexcept violations throw before fast-fail).
    std::set_terminate([]() {
        if (InterlockedExchange(&s_inHandlerWin, 1) == 0) {
            writeStampWindows("UNCAUGHT C++ EXCEPTION (std::terminate)");
            writeEnvironmentBlock();
            writeWindowsStack();
        }
        std::abort();
    });
#else
    // Re-install support (see header): close the previous descriptor first
    // or a re-install leaks it for the lifetime of the process.
    if (s_logFd >= 0) {
        close(s_logFd);
        s_logFd = -1;
    }
    std::signal(SIGSEGV, posixSignalHandler);
    std::signal(SIGBUS, posixSignalHandler);
    std::signal(SIGFPE, posixSignalHandler);
    std::signal(SIGILL, posixSignalHandler);
    std::signal(SIGABRT, posixSignalHandler);
    std::set_terminate(terminateHandler);
#endif
}

}  // namespace ecvCrashHandler
