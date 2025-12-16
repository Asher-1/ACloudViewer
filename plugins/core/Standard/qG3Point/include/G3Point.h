// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// CCPluginStub
#include <ecvStdPluginInterface.h>

// qCC_db
#include <ecvExternalFactory.h>

class G3PointFactory : public ccExternalFactory {
public:
    G3PointFactory(QString factoryName, ecvMainAppInterface* app)
        : ccExternalFactory(factoryName), m_app(app) {}

    virtual ccHObject* buildObject(const QString& metaName) override;

    ecvMainAppInterface* m_app;
};

class G3PointPlugin : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

    // Replace "Example" by your plugin name (IID should be unique - let's hope
    // your plugin name is unique ;) The info.json file provides information
    // about the plugin to the loading system and it is displayed in the plugin
    // information dialog.
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.G3Point" FILE
                          "../info.json")

public:
    explicit G3PointPlugin(QObject* parent = nullptr);
    virtual ~G3PointPlugin() override = default;

    // Inherited from ccStdPluginInterface
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;

private:
    //! Default action
    /** You can add as many actions as you want in a plugin.
            Each action will correspond to an icon in the dedicated
            toolbar and an entry in the plugin menu.
    **/
    QAction* m_action;
};
