// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewTitleRegistry.h"

#include <climits>

ecvViewTitleRegistry& ecvViewTitleRegistry::instance() {
    static ecvViewTitleRegistry s_instance;
    return s_instance;
}

QString ecvViewTitleRegistry::sanitizeName(const QString& name) {
    if (name.isEmpty()) return QString();

    QString out;
    out.reserve(name.size());
    for (const QChar& c : name) {
        if (c.isLetterOrNumber() || c == QLatin1Char('_')) out += c;
    }

    if (out.isEmpty() || out.at(0).isLetter()) return out;
    return QStringLiteral("a") + out;
}

QString ecvViewTitleRegistry::allocate(const QString& xmlLabel) {
    if (xmlLabel.isEmpty()) return QString();

    const QString prefix = sanitizeName(xmlLabel);
    if (prefix.isEmpty()) return QString();

    auto& used = m_usedSerials[prefix];
    if (!used.contains(0)) {
        used.insert(0);
        return prefix;
    }

    for (int suffix = 1; suffix < INT_MAX; ++suffix) {
        if (!used.contains(suffix)) {
            used.insert(suffix);
            return prefix + QString::number(suffix);
        }
    }
    return QString();
}

void ecvViewTitleRegistry::release(const QString& xmlLabel,
                                   const QString& title) {
    const int serial = parseSerial(xmlLabel, title);
    if (serial < 0) return;
    const QString prefix = sanitizeName(xmlLabel);
    auto it = m_usedSerials.find(prefix);
    if (it != m_usedSerials.end()) it->remove(serial);
}

int ecvViewTitleRegistry::parseSerial(const QString& xmlLabel,
                                      const QString& title) {
    const QString prefix = sanitizeName(xmlLabel);
    if (prefix.isEmpty() || title.isEmpty()) return -1;
    if (title == prefix) return 0;
    if (!title.startsWith(prefix)) return -1;

    const QString suffix = title.mid(prefix.size());
    if (suffix.isEmpty()) return -1;

    bool ok = false;
    const int serial = suffix.toInt(&ok);
    return (ok && serial > 0) ? serial : -1;
}
