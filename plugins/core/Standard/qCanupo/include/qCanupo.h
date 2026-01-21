// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// qCC
#include <ecvStdPluginInterface.h>

// CV_DB_LIB
#include <ecvHObject.h>

//! CANUPO plugin
/** See "3D Terrestrial lidar data classification of complex natural scenes
using a multi-scale dimensionality criterion: applications in geomorphology", N.
Brodu, D. Lague, 2012, Computer Vision and Pattern Recognition
**/
class qCanupoPlugin : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qCanupo" FILE
                          "../info.json")

public:
    //! Default constructor
    qCanupoPlugin(QObject* parent = nullptr);

    // inherited from ccStdPluginInterface
    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;
    virtual void registerCommands(ccCommandLineInterface* cmd) override;

protected slots:

    void doClassifyAction();
    void doTrainAction();

protected:
    //! Calssift action
    QAction* m_classifyAction;
    //! Train action
    QAction* m_trainAction;

    //! Currently selected entities
    ccHObject::Container m_selectedEntities;
};
