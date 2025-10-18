// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #                     ACLOUDVIEWER PLUGIN: q3DMASC                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                 COPYRIGHT: Dimitri Lague / CNRS / UEB                  #
// #                                                                        #
// ##########################################################################

// qCC
#include <ecvStdPluginInterface.h>

#include "Parameters.h"

//! 3DMASC plugin
/** 3D Multi-cloud, multi-Attribute, multi-Scale, multi-Class classification
 **/
class q3DMASCPlugin : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.q3DMASC" FILE "info.json")

public:
    //! Default constructor
    explicit q3DMASCPlugin(QObject* parent = nullptr);

    // inherited from ccStdPluginInterface
    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;
    void registerCommands(ccCommandLineInterface* cmd) override;

protected slots:

    void doClassifyAction();
    void saveTrainParameters(const masc::TrainParameters& params);
    void loadTrainParameters(masc::TrainParameters& params);
    void doTrainAction();

protected:
    //! Calssify action
    QAction* m_classifyAction;
    //! Train action
    QAction* m_trainAction;

    //! Currently selected entities
    ccHObject::Container m_selectedEntities;
};
