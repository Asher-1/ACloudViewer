// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "PCVCommand.h"
#include "ecvStdPluginInterface.h"

//! Wrapper to the ShadeVis algorithm for computing Ambient Occlusion on meshes
//! and point clouds
/** "Visibility based methods and assessment for detail-recovery", M. Tarini, P.
Cignoni, R. Scopigno Proc. of Visualization 2003, October 19-24, Seattle, USA.
        http://vcg.sourceforge.net/index.php/ShadeVis
**/

class qPCV : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qPCV" FILE "../info.json")

public:
    //! Default constructor
    explicit qPCV(QObject* parent = nullptr);

    virtual ~qPCV() = default;

    // inherited from ccStdPluginInterface
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;
    virtual void registerCommands(ccCommandLineInterface* cmd) override;

protected slots:

    //! Slot called when associated ation is triggered
    void doAction();

protected:
    //! Associated action
    QAction* m_action;
};
