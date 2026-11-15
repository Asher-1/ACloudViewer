// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Child processes started by RPC handlers. Handlers run on the Qt GUI
// thread (jsonrpcserver.cpp dispatches there), so a child is never awaited:
// neither QProcess::waitForFinished() (frozen GUI and keepalive pongs) nor a
// nested QEventLoop (a second request re-enters the dispatcher, and the
// outer reply and deadline then wait for the inner child). The handler
// returns a deferred result instead; the L1 agent-integration gate enforces
// that runChildProcess() is the only way a handler starts a child.

#include <QMap>
#include <QProcessEnvironment>
#include <QString>
#include <QVariant>
#include <functional>

#include "jsonrpcserver.h"

class QObject;
class QProcess;

namespace jsonrpc {

// Environment for a headless ACloudViewer child. insert() replaces an
// inherited value; appending "QT_QPA_PLATFORM=..." to systemEnvironment()
// leaves the parent's entry first, which getenv() returns.
QProcessEnvironment headlessChildEnvironment();

// Reads params["timeout_ms"] (defaultMs when absent) into *timeoutMs. A
// value outside (0, 24 h] yields an invalid-params *error: 0 would kill the
// child at once, and a child without a deadline could hold its request
// forever.
bool childTimeout(const QMap<QString, QVariant>& params,
                  int defaultMs,
                  int* timeoutMs,
                  JsonRPCResult* error);

enum class ChildOutcome { Succeeded, Failed, StartFailed, TimedOut };

// Builds the answer from the ended child (exit code, output, result files).
using ChildFinish = std::function<JsonRPCResult(ChildOutcome, QProcess&)>;

// Starts `process` (configured, parented to the plugin) and returns a
// deferred result at once. `finish` runs on the GUI thread when the child
// exits, fails to start, or has been killed at its deadline; its result
// answers the request. The process is deleted afterwards.
JsonRPCResult runChildProcess(QProcess* process,
                              int timeoutMs,
                              ChildFinish finish);

// Kills the children of `owner` still running, without answering them: the
// plugin is stopping, and their `finish` may reference it.
void abortChildProcesses(QObject* owner);

}  // namespace jsonrpc
