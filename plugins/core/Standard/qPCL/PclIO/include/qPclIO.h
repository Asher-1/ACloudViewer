// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_IO_PLUGIN_HEADER
#define Q_PCL_IO_PLUGIN_HEADER

#include "ecvIOPluginInterface.h"

// Qt
#include <QObject>

//! PCL IO plugin (PCD format)
class qPclIO : public QObject, public ccIOPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccIOPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qPclIO" FILE
                          "../info.json")

public:
    explicit qPclIO(QObject* parent = nullptr);

    // inherited from ccIOPluginInterface
    ccIOPluginInterface::FilterList getFilters() override;

    // inherited from ccDefaultPluginInterface
    void registerCommands(ccCommandLineInterface* cmd) override;
};

#endif  // Q_PCL_IO_PLUGIN_HEADER
