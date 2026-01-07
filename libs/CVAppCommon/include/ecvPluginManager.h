// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>
#include <QVector>

#include "CVAppCommon.h"

class ccPluginInterface;

//! Simply a list of \see ccPluginInterface
using ccPluginInterfaceList = QVector<ccPluginInterface*>;

class CVAPPCOMMON_LIB_API ccPluginManager : public QObject {
    Q_OBJECT

public:
    ~ccPluginManager() override = default;

    static ccPluginManager& get();

    void setPaths(const QStringList& paths);
    QStringList pluginPaths();

    void loadPlugins();

    ccPluginInterfaceList& pluginList();

    bool isEnabled(const ccPluginInterface* plugin) const;
    void setPluginEnabled(const ccPluginInterface* plugin, bool enabled);

protected:
    explicit ccPluginManager(QObject* parent = nullptr);

private:
    void loadFromPathsAndAddToList();

    QStringList disabledPluginIIDs() const;

    QStringList m_pluginPaths;
    ccPluginInterfaceList m_pluginList;
};
