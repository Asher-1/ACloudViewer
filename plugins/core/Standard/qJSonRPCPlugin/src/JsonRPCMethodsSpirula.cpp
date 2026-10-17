// Spirula Studio handlers, retained across the upstream RPC family split.
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QRegularExpression>
#include <QSet>
#include <cerrno>

#include "JsonRPCChildProcess.h"
#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"
#include "SpirulaLauncher.h"
#include "SpirulaLocator.h"
#include "SplatImporter.h"
#ifndef Q_OS_WIN
#include <csignal>
#else
#include <windows.h>
#endif
using jsonrpc::D;
using jsonrpc::runChildProcess;

JsonRPCResult JsonRPCPlugin::rpcSpirulaVersion(
        const QMap<QString, QVariant>& params) {
    Q_UNUSED(params)
    const QString exe = spirula::locateExecutable();
    if (exe.isEmpty()) {
        return JsonRPCResult::error(
                1,
                "Spirula Studio executable not found (built with "
                "BUILD_SPIRULA=ON)");
    }
    auto* process = new QProcess(this);
    process->setProgram(exe);
    process->setArguments({QStringLiteral("--version")});
    return runChildProcess(
            process, 8000,
            [exe](jsonrpc::ChildOutcome outcome, QProcess& child) {
                if (outcome != jsonrpc::ChildOutcome::Succeeded) {
                    return JsonRPCResult::error(
                            1, "spirula --version failed",
                            D("exe", exe, "stderr",
                              QString::fromUtf8(child.readAllStandardError())));
                }
                QJsonObject result;
                result["exe"] = exe;
                result["version"] =
                        QString::fromUtf8(child.readAllStandardOutput())
                                .trimmed();
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

JsonRPCResult JsonRPCPlugin::rpcSpirulaDevices(
        const QMap<QString, QVariant>& params) {
    Q_UNUSED(params)
    const QString exe = spirula::locateExecutable();
    if (exe.isEmpty()) {
        return JsonRPCResult::error(1, "Spirula Studio executable not found");
    }
    auto* process = new QProcess(this);
    process->setProgram(exe);
    process->setArguments({QStringLiteral("--device"), QStringLiteral("auto"),
                           QStringLiteral("--version")});
    return runChildProcess(
            process, 5000, [](jsonrpc::ChildOutcome outcome, QProcess& child) {
                if (outcome == jsonrpc::ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(
                            3,
                            "spirula device enumeration timed out (a "
                            "Vulkan-capable GPU/driver is required)");
                }
                if (outcome != jsonrpc::ChildOutcome::Succeeded) {
                    return JsonRPCResult::error(
                            1, "spirula device enumeration failed");
                }
                // Rows printed by upstream: "  * [0] NAME (type, 12345 MB)
                // [note]"; the note is non-empty only for unusable devices.
                static const QRegularExpression rowRe(
                        R"re(\[(-?\d+)\]\s+(\S.*?)\s+\(([^,]+),\s*(\d+)\s*MB\)(.*))re");
                QJsonArray devices;
                const QStringList rows =
                        QString::fromUtf8(child.readAllStandardOutput())
                                .split('\n');
                for (const QString& row : rows) {
                    const auto m = rowRe.match(row);
                    if (!m.hasMatch()) {
                        continue;
                    }
                    QJsonObject device;
                    device["index"] = m.captured(1).toInt();
                    device["name"] = m.captured(2).trimmed();
                    device["type"] = m.captured(3).trimmed();
                    device["vram_mb"] = m.captured(4).toLongLong();
                    device["current"] = row.trimmed().startsWith('*');
                    device["usable"] = m.captured(5).trimmed().isEmpty();
                    devices.append(device);
                }
                QJsonObject result;
                result["devices"] = devices;
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

JsonRPCResult JsonRPCPlugin::rpcSpirulaLaunch(
        const QMap<QString, QVariant>& params) {
    Q_UNUSED(params)
    const QString exe = spirula::locateExecutable();
    QString error;
    if (!spirula::launchGui(exe, &error)) {
        return JsonRPCResult::error(1, error);
    }
    QJsonObject result;
    result["launched"] = true;
    result["exe"] = exe;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

JsonRPCResult JsonRPCPlugin::rpcSpirulaCli(
        const QMap<QString, QVariant>& params) {
    const QString command = params["command"].toString();
    // Whitelist the headless subcommands; 'gui'/'sam' excluded on purpose —
    // sam opens an interactive masking session, gui is the RPC launch path.
    static const QSet<QString> kWhitelisted = {
            QStringLiteral("train"),     QStringLiteral("sfm"),
            QStringLiteral("mesh"),      QStringLiteral("geometry"),
            QStringLiteral("e57"),       QStringLiteral("lidar"),
            QStringLiteral("partition"), QStringLiteral("encode")};
    if (!kWhitelisted.contains(command)) {
        return JsonRPCResult::error(
                -32602, "Unknown or non-whitelisted spirula command",
                D("command", command));
    }
    const QString exe = spirula::locateExecutable();
    if (exe.isEmpty()) {
        return JsonRPCResult::error(1, "Spirula Studio executable not found");
    }

    QStringList argList{command};
    const QVariantList rawArgs = params["args"].toList();
    for (const QVariant& arg : rawArgs) {
        argList << arg.toString();
    }

    QJsonObject result;
    result["command"] = command;
    result["args"] = QJsonArray::fromVariantList(rawArgs);
    result["exe"] = exe;
    if (params.contains("timeout_ms")) {
        int timeoutMs = 0;
        JsonRPCResult invalid;
        if (!jsonrpc::childTimeout(params, 60000, &timeoutMs, &invalid))
            return invalid;
        auto* process = new QProcess(this);
        process->setProgram(exe);
        process->setArguments(argList);
        process->setWorkingDirectory(QFileInfo(exe).absolutePath());
        return runChildProcess(
                process, timeoutMs,
                [result, timeoutMs](jsonrpc::ChildOutcome outcome,
                                    QProcess& child) mutable {
                    if (outcome == jsonrpc::ChildOutcome::TimedOut) {
                        return JsonRPCResult::error(3,
                                                    "spirula command timed out",
                                                    D("timeout_ms", timeoutMs));
                    }
                    if (outcome == jsonrpc::ChildOutcome::StartFailed) {
                        return JsonRPCResult::error(
                                1, "Failed to start spirula command");
                    }
                    result["exit_code"] = child.exitCode();
                    result["stdout"] =
                            QString::fromUtf8(child.readAllStandardOutput());
                    result["stderr"] =
                            QString::fromUtf8(child.readAllStandardError());
                    return JsonRPCResult::success(
                            QJsonDocument(result).toVariant());
                });
    }
    qint64 pid = 0;
    if (!QProcess::startDetached(exe, argList, QFileInfo(exe).absolutePath(),
                                 &pid)) {
        return JsonRPCResult::error(1, "Failed to start spirula " + command);
    }
    result["pid"] = static_cast<double>(pid);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

JsonRPCResult JsonRPCPlugin::rpcSpirulaImport(
        const QMap<QString, QVariant>& params) {
    const QString path = params["path"].toString();
    if (path.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'path' parameter");
    }
    if (!QFileInfo::exists(path)) {
        return JsonRPCResult::error(-32602, "File not found: " + path);
    }

    const bool asMesh =
            params.value("mesh", false).toBool() ||
            path.endsWith(QStringLiteral("mesh.ply"), Qt::CaseInsensitive);
    QString error;
    ccHObject* imported = nullptr;
    if (asMesh) {
        imported = spirula::loadMesh(path, &error);
    } else {
        const QString splatPath = spirula::resolveSplatPath(path);
        if (splatPath.isEmpty()) {
            return JsonRPCResult::error(
                    1, "No splat.ply found in run directory", D("path", path));
        }
        imported = spirula::loadSplatPointCloud(splatPath, &error);
    }
    if (!imported) {
        return JsonRPCResult::error(
                1, error.isEmpty() ? QStringLiteral("Import failed") : error,
                D("path", path));
    }

    const QString uniqueName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("Spirula_") + QFileInfo(path).baseName(), m_app);
    imported->setName(uniqueName);
    m_app->addToDB(imported, true, true, false, true);
    m_app->setSelectedInDB(imported, true);
    m_app->zoomOnEntities(imported);
    m_app->refreshAll();

    QJsonObject result;
    result["entity"] = uniqueName;
    if (imported->isA(CV_TYPES::POINT_CLOUD)) {
        result["points"] = static_cast<qint64>(
                static_cast<ccPointCloud*>(imported)->size());
    }
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

JsonRPCResult JsonRPCPlugin::rpcSpirulaCancel(
        const QMap<QString, QVariant>& params) {
    const qint64 pid = params["pid"].toLongLong();
    if (pid <= 0) {
        return JsonRPCResult::error(-32602, "Missing or invalid 'pid'");
    }

    // Identity check: only ever terminate a process whose image is the
    // spirula executable — a stale/wrong pid must not kill something else.
    bool isSpirula = false;
#ifdef Q_OS_WIN
    HANDLE handle =
            OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE,
                        FALSE, static_cast<DWORD>(pid));
    if (!handle) {
        return JsonRPCResult::error(1, "Process not found",
                                    D("pid", static_cast<double>(pid)));
    }
    char imagePath[MAX_PATH] = {0};
    DWORD imageSize = MAX_PATH;
    isSpirula = QueryFullProcessImageNameA(handle, 0, imagePath, &imageSize) &&
                QString(imagePath).contains("spirula", Qt::CaseInsensitive);
    if (!isSpirula) {
        CloseHandle(handle);
        return JsonRPCResult::error(
                1, "Refusing to cancel: pid is not a spirula process",
                D("pid", static_cast<double>(pid)));
    }
    const bool terminated = TerminateProcess(handle, 1) != 0;
    CloseHandle(handle);
    if (!terminated) {
        return JsonRPCResult::error(1, "TerminateProcess failed",
                                    D("pid", static_cast<double>(pid)));
    }
#else
    QFile commFile(QStringLiteral("/proc/%1/comm").arg(pid));
    if (commFile.open(QIODevice::ReadOnly)) {
        isSpirula = commFile.readAll().trimmed().startsWith("spirula");
    }
    if (!isSpirula) {
        return JsonRPCResult::error(
                1, "Refusing to cancel: pid is not a spirula process",
                D("pid", static_cast<double>(pid)));
    }
    // SIGTERM lets the upstream trainer checkpoint before exiting.
    if (kill(static_cast<pid_t>(pid), SIGTERM) != 0) {
        return JsonRPCResult::error(
                1, "kill failed",
                D("pid", static_cast<double>(pid), "errno", errno));
    }
#endif
    QJsonObject result;
    result["cancelled"] = true;
    result["pid"] = static_cast<double>(pid);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}
