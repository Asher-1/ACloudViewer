// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QSet>
#include <QString>

#include "CV_db.h"

/// ParaView Server Manager registration names for views (vtkSMCoreUtilities::
/// SanitizeName + vtkSMSessionProxyManager::GetUniqueProxyName).
/// First view of a type: "RenderView"; additional: "RenderView1",
/// "RenderView2". Serials are recycled when views are destroyed.
class CV_DB_LIB_API ecvViewTitleRegistry {
public:
    static ecvViewTitleRegistry& instance();

    /// @p xmlLabel ParaView proxy XML label (e.g. "Render View").
    QString allocate(const QString& xmlLabel);

    /// Releases a name produced by allocate() for the same @p xmlLabel.
    void release(const QString& xmlLabel, const QString& title);

    /// Same rules as vtkSMCoreUtilities::SanitizeName (alnum/underscore only).
    static QString sanitizeName(const QString& name);

private:
    ecvViewTitleRegistry() = default;

    /// 0 = bare prefix; 1+ = prefix + number (no separator).
    static int parseSerial(const QString& xmlLabel, const QString& title);

    QHash<QString, QSet<int>> m_usedSerials;
};
