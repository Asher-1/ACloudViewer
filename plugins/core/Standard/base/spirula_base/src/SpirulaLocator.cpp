// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SpirulaLocator.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QStandardPaths>
#include <QStringList>

#ifdef QSPIRULA_BUILD_PATH
#define QSPIRULA_STRINGIFY_(x) #x
#define QSPIRULA_STRINGIFY(x) QSPIRULA_STRINGIFY_(x)
#define QSPIRULA_FALLBACK_DIR QSPIRULA_STRINGIFY(QSPIRULA_BUILD_PATH)
#endif

namespace spirula {

namespace {

//! A spirula-cuda binary is only worth picking when an NVIDIA driver is
//! actually present; otherwise the Vulkan variant covers NVIDIA GPUs too.
bool nvidiaDriverPresent() {
    QProcess probe;
    QStringList args;
    args << QStringLiteral("-L");
    probe.start(QStringLiteral("nvidia-smi"), args);
    return probe.waitForFinished(3000) &&
           probe.exitStatus() == QProcess::NormalExit && probe.exitCode() == 0;
}

QString nextToApp(const QString& exeName) {
    const QString candidate =
            QDir(QCoreApplication::applicationDirPath()).filePath(exeName);
    return QFileInfo::exists(candidate) ? candidate : QString();
}

}  // namespace

QString locateExecutable() {
    static QString cached;
    static bool resolved = false;
    if (resolved) {
        return cached;
    }
    resolved = true;

#ifdef Q_OS_WIN
    static const QLatin1String kSuffix(".exe");
#else
    static const QLatin1String kSuffix("");
#endif

    const QString cudaVariant =
            nextToApp(QStringLiteral("spirula-cuda") + kSuffix);
    if (!cudaVariant.isEmpty() && nvidiaDriverPresent()) {
        cached = cudaVariant;
        return cached;
    }

    const QString vulkanVariant =
            nextToApp(QStringLiteral("spirula") + kSuffix);
    if (!vulkanVariant.isEmpty()) {
        cached = vulkanVariant;
        return cached;
    }

#ifdef QSPIRULA_FALLBACK_DIR
    {
        const QString fallback =
                QDir(QStringLiteral(QSPIRULA_FALLBACK_DIR))
                        .filePath(QStringLiteral("spirula") + kSuffix);
        if (QFileInfo::exists(fallback)) {
            cached = fallback;
            return cached;
        }
    }
#endif

    cached = QStandardPaths::findExecutable(QStringLiteral("spirula"));
    return cached;
}

}  // namespace spirula
