// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_ANIMATION_PLUGIN_HEADER
#define Q_ANIMATION_PLUGIN_HEADER

// qCC
#include "ecvStdPluginInterface.h"

// Qt
#include <QObject>

class ccGLWindow;

// Animation plugin
class qAnimation : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qAnimation" FILE
                          "../info.json")

public:
    //! Default constructor
    qAnimation(QObject* parent = nullptr);

    virtual ~qAnimation() = default;

    // inherited from ccStdPluginInterface
    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;

private:
    void doAction();

    QAction* m_action;
};

#endif
