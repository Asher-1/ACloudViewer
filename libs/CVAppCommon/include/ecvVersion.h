// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_VERSION_HEADER
#define CV_VERSION_HEADER

// LOCAL
#include <CVPlatform.h>

// Qt
#include <QMainWindow>

static QString versionLongStr(bool includeOS, QString verStr) {
#if defined(CV_ENV_64)
    const QString arch("64-bit");
#elif defined(CV_ENV_32)
    const QString arch("32-bit");
#else
    const QString arch("\?\?-bit");
#endif

    if (includeOS) {
#if defined(CV_WINDOWS)
        const QString platform("Windows");
#elif defined(CV_MAC_OS)
        const QString platform("macOS");
#elif defined(CV_LINUX)
        const QString platform("Linux");
#else
        const QString platform("Unknown OS");
#endif
        verStr += QStringLiteral(" [%1 %2]").arg(platform, arch);
    } else {
        verStr += QStringLiteral(" [%1]").arg(arch);
    }

#ifdef QT_DEBUG
    verStr += QStringLiteral(" [DEBUG]");
#endif

    return verStr;
};

#endif  // CV_VERSION_HEADER