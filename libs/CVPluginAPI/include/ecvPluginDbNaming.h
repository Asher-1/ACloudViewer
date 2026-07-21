// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Header-only helpers for plugin exports into the DB tree.

#pragma once

#include <ecvHObject.h>
#include <ecvMainAppInterface.h>

#include <QFileInfo>
#include <QSet>
#include <QString>
#include <algorithm>

namespace ecvPluginDbNaming {

inline QString sanitizeSegment(QString segment, int maxLen = 40) {
    segment = segment.trimmed();
    QString out;
    out.reserve(segment.size());
    for (const QChar c : segment) {
        if (c.isLetterOrNumber() || c == QLatin1Char('_') ||
            c == QLatin1Char('-')) {
            out.append(c);
        } else if (c == QLatin1Char(' ') || c == QLatin1Char('.')) {
            if (!out.isEmpty() && out.back() != QLatin1Char('_')) {
                out.append(QLatin1Char('_'));
            }
        } else {
            if (!out.isEmpty() && out.back() != QLatin1Char('_')) {
                out.append(QLatin1Char('_'));
            }
        }
    }
    while (out.startsWith(QLatin1Char('_'))) out.remove(0, 1);
    while (out.endsWith(QLatin1Char('_'))) out.chop(1);
    if (maxLen > 0 && out.length() > maxLen) {
        out = out.left(maxLen);
        while (out.endsWith(QLatin1Char('_'))) out.chop(1);
    }
    return out;
}

inline QString modelTagFromFilename(const QString& modelPath, int maxLen = 28) {
    const QString stem = QFileInfo(modelPath).completeBaseName();
    const QString tag = sanitizeSegment(stem, maxLen);
    return tag.isEmpty() ? QStringLiteral("Model") : tag;
}

inline void collectExistingNames(const ccHObject* root, QSet<QString>& names) {
    if (!root) return;
    names.insert(root->getName());
    const unsigned count = root->getChildrenNumber();
    for (unsigned i = 0; i < count; ++i) {
        collectExistingNames(root->getChild(i), names);
    }
}

inline QString makeUnique(const QString& baseName, const ccHObject* dbRoot) {
    if (baseName.isEmpty()) return baseName;
    if (!dbRoot) return baseName;

    QSet<QString> existing;
    collectExistingNames(dbRoot, existing);

    int maxSuffix = -1;
    const QString suffixPrefix = baseName + QLatin1Char('_');
    for (const QString& name : existing) {
        if (name == baseName) {
            maxSuffix = std::max(maxSuffix, 0);
            continue;
        }
        if (!name.startsWith(suffixPrefix)) continue;
        bool ok = false;
        const int suffix = name.mid(suffixPrefix.size()).toInt(&ok);
        if (ok && suffix > 0) {
            maxSuffix = std::max(maxSuffix, suffix);
        }
    }

    if (maxSuffix < 0) return baseName;
    return QStringLiteral("%1_%2").arg(baseName).arg(maxSuffix + 1, 2, 10,
                                                     QChar('0'));
}

inline QString makeUnique(const QString& baseName, ecvMainAppInterface* app) {
    if (!app) return baseName;
    return makeUnique(baseName, app->dbRootObject());
}

}  // namespace ecvPluginDbNaming
