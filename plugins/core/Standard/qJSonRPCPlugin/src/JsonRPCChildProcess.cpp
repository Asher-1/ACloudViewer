// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "JsonRPCChildProcess.h"

#include <QObject>
#include <QProcess>
#include <QTimer>
#include <memory>

namespace jsonrpc {

QProcessEnvironment headlessChildEnvironment() {
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
#if defined(Q_OS_MACOS)
    // .app bundles ship only the cocoa platform plugin.
    environment.remove(QStringLiteral("QT_QPA_PLATFORM"));
#elif defined(Q_OS_WIN)
    // windeployqt does not ship qoffscreen.
    environment.insert(QStringLiteral("QT_QPA_PLATFORM"),
                       QStringLiteral("minimal"));
#else
    environment.insert(QStringLiteral("QT_QPA_PLATFORM"),
                       QStringLiteral("offscreen"));
#endif
    return environment;
}

bool childTimeout(const QMap<QString, QVariant>& params,
                  int defaultMs,
                  int* timeoutMs,
                  JsonRPCResult* error) {
    constexpr qint64 kMaxMs = 24LL * 60 * 60 * 1000;
    bool ok = true;
    const qint64 value =
            params.contains(QStringLiteral("timeout_ms"))
                    ? params.value(QStringLiteral("timeout_ms")).toLongLong(&ok)
                    : defaultMs;
    if (!ok || value <= 0 || value > kMaxMs) {
        *error = JsonRPCResult::error(
                -32602, QStringLiteral("timeout_ms must be in (0, 86400000]"),
                QVariantMap{{QStringLiteral("timeout_ms"),
                             params.value(QStringLiteral("timeout_ms"))}});
        return false;
    }
    *timeoutMs = static_cast<int>(value);
    return true;
}

JsonRPCResult runChildProcess(QProcess* process,
                              int timeoutMs,
                              ChildFinish finish) {
    auto reply = std::make_shared<JsonRPCDeferred>();
    auto timedOut = std::make_shared<bool>(false);
    auto ended = std::make_shared<bool>(false);
    auto* deadline = new QTimer(process);
    deadline->setSingleShot(true);
    // finished() and errorOccurred(FailedToStart) are exclusive; the guard
    // keeps the answer single regardless.
    auto end = [process, deadline, reply, ended, finish](ChildOutcome outcome) {
        if (*ended) return;
        *ended = true;
        deadline->stop();
        reply->complete(finish(outcome, *process));
        process->deleteLater();
    };
    QObject::connect(
            process,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            process,
            [end, timedOut](int exitCode, QProcess::ExitStatus status) {
                if (*timedOut) {
                    end(ChildOutcome::TimedOut);
                } else {
                    end(status == QProcess::NormalExit && exitCode == 0
                                ? ChildOutcome::Succeeded
                                : ChildOutcome::Failed);
                }
            });
    QObject::connect(process, &QProcess::errorOccurred, process,
                     [end](QProcess::ProcessError error) {
                         if (error == QProcess::FailedToStart) {
                             end(ChildOutcome::StartFailed);
                         }
                     });
    QObject::connect(deadline, &QTimer::timeout, process, [process, timedOut] {
        *timedOut = true;
        process->kill();  // finished() ends the request
    });
    // Armed first: start() may report FailedToStart synchronously.
    deadline->start(timeoutMs);
    process->start();
    return JsonRPCResult::deferred(reply);
}

void abortChildProcesses(QObject* owner) {
    const QList<QProcess*> children = owner->findChildren<QProcess*>(
            QString(), Qt::FindDirectChildrenOnly);
    for (QProcess* process : children) {
        QObject::disconnect(process, nullptr, nullptr, nullptr);
        delete process;  // ~QProcess kills and reaps a running child
    }
}

}  // namespace jsonrpc
