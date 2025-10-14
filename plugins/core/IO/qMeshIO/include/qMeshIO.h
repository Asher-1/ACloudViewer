// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "ecvIOPluginInterface.h"

class qMeshIO final : public QObject, public ccIOPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccIOPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qMeshIO" FILE
                          "../info.json")

public:
    explicit qMeshIO(QObject *parent = nullptr);

private:
    void registerCommands(ccCommandLineInterface *inCmdLine) override;

    FilterList getFilters() override;
};
