// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// qCC
#include "ecvStdPluginInterface.h"

// ECV_DB_LIB
#include <ecvHObject.h>

//! M3C2 plugin
/** See "Accurate 3D comparison of complex topography with terrestrial laser
scanner: application to the Rangitikei canyon (N-Z)", Lague, D., Brodu, N. and
Leroux, J., 2013, ISPRS journal of Photogrammmetry and Remote Sensing
**/
class qM3C2Plugin : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qM3C2" FILE
                          "../info.json")

public:
    //! Default constructor
    qM3C2Plugin(QObject* parent = nullptr);

    virtual ~qM3C2Plugin() = default;

    // inherited from ccStdPluginInterface
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;
    virtual void registerCommands(ccCommandLineInterface* cmd) override;

private:
    void doAction();

    //! Default action
    QAction* m_action;

    //! Currently selected entities
    ccHObject::Container m_selectedEntities;
};
