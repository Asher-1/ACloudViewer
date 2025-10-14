// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvIOPluginInterface.h"

//! CSV Matrix file (2.5D cloud)
class qCSVMatrixIO : public QObject, public ccIOPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccIOPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qCSVMatrixIO" FILE
                          "../info.json")

public:
    qCSVMatrixIO(QObject* parent = nullptr);

    virtual ~qCSVMatrixIO() override = default;

    // inherited from ccIOPluginInterface
    FilterList getFilters() override;
};
