// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

#include "CVCoreLib.h"

namespace ecvCrashHandler {

//! Installs the process-wide crash collectors.
//!
//! Covers, through ONE unified interface:
//!  - fatal OS signals: SIGSEGV / SIGBUS / SIGFPE / SIGILL / SIGABRT
//!    (POSIX) and unhandled SEH exceptions (Windows);
//!  - uncaught C++ exceptions escaping any thread (std::set_terminate);
//!  - CAUGHT C++ exceptions reaching the app's top-level handler: the
//!    throw-site stack is snapshotted by the __cxa_throw interposer
//!    (POSIX) BEFORE unwinding and written by logUnhandledException, so
//!    even exceptions the app intercepts carry a full stack.
//!
//! On an abnormal path the collector lazily creates \p logFilePath, then
//! appends a header (reason + timestamp + the environment block captured at
//! install time) and the captured stack before restoring the default handler
//! so the process still dies the way the OS expects (cores, WER dumps, ...
//! keep working). A normal session never creates a crash file.
//!
//! Signal handlers use only preallocated UTF-8 path/environment bytes plus
//! open()/backtrace_symbols_fd()/write() on POSIX; no Qt, allocation, or
//! locale-dependent formatting occurs inside the handler.
//!
//! Re-installing replaces the previous collector; install() is NOT
//! thread-safe with respect to itself (call it once from main()).
//!
//! NOTE on Qt: Qt exposes NO public cross-platform crash/stack API
//! (qInstallMessageHandler routes log messages only). This collector is
//! therefore built on the platform primitives directly; Qt is used where
//! it IS the right tool — environment reporting (qVersion, QSysInfo) and
//! path selection at install time. For minidump-based commercial crash
//! reporting (upload, symbol server) the drop-in upgrade path is Google
//! Crashpad behind this same install() interface.
void CV_CORE_LIB_API install(const QString &logFilePath);

//! Logs a CAUGHT C++ exception into the crash log (the unified record for
//! exceptions that the app itself intercepts — e.g. the top-level handler
//! in main() — which therefore never reach std::terminate). On POSIX the
//! THROW-SITE stack (captured at __cxa_throw, before unwinding) is
//! recorded alongside the message; on Windows only the message and the
//! environment block (the stack is unrecoverable there). Safe to call
//! from the main thread.
void CV_CORE_LIB_API logUnhandledException(const QString &what);

}  // namespace ecvCrashHandler
