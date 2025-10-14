// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_LAS_FWF_IO_PLUGIN_HEADER
#define Q_LAS_FWF_IO_PLUGIN_HEADER

#include "ecvIOPluginInterface.h"

class qLASFWFIO : public QObject, public ccIOPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccIOPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qLAS_FWF_IO" FILE
                          "../info.json")

public:
    //! Default constructor
    explicit qLASFWFIO(QObject* parent = nullptr);

    // inherited from ccPluginInterface
    virtual void registerCommands(ccCommandLineInterface* cmd) override;

    // inherited from ccIOPluginInterface
    FilterList getFilters() override;
};

#endif  // Q_LAS_FWF_IO_PLUGIN_HEADER
