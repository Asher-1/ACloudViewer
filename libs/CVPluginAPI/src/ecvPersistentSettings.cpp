// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPersistentSettings.h"

#include <QSet>
#include <QSettings>

namespace {

QSet<QString>& registeredGroups() {
    static QSet<QString> groups;
    return groups;
}

}  // namespace

void ecvPS::registerSettingsGroup(const QString& groupPrefix) {
    if (groupPrefix.isEmpty()) return;
    registeredGroups().insert(groupPrefix);
}

void ecvPS::resetAllRegistered() {
    QSettings settings;
    const QSet<QString> groups = registeredGroups();
    for (const QString& prefix : groups) {
        settings.remove(prefix);
    }
}
