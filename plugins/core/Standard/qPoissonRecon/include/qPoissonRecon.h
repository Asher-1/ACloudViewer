// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_POISSON_RECON_PLUGIN_HEADER
#define Q_POISSON_RECON_PLUGIN_HEADER

#include "ecvStdPluginInterface.h"

//! Wrapper to the "Poisson Surface Reconstruction (Version 9)" algorithm
/** "Poisson Surface Reconstruction", M. Kazhdan, M. Bolitho, and H. Hoppe
        Symposium on Geometry Processing (June 2006), pages 61--70
        http://www.cs.jhu.edu/~misha/Code/PoissonRecon/
**/
class qPoissonRecon : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qPoissonRecon" FILE
                          "../info.json")

public:
    //! Default constructor
    explicit qPoissonRecon(QObject* parent = nullptr);

    virtual ~qPoissonRecon() = default;

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

#endif
