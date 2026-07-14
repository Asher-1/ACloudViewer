// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_BROOM_PLUGIN_HEADER
#define Q_BROOM_PLUGIN_HEADER

#include "ecvStdPluginInterface.h"

//! CEA virtual broom plugin
class qBroom : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qBroom" FILE
                          "../info.json")

public:
    //! Default constructor
    explicit qBroom(QObject* parent = nullptr);

    virtual ~qBroom() = default;

    // inherited from ccStdPluginInterface
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;

protected:
    //! Slot called when associated ation is triggered
    void doAction();

protected:
    //! Associated action
    QAction* m_action;
};

#endif  // Q_BROOM_PLUGIN_HEADER
