// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvStdPluginInterface.h"

//! Wrapper to the 'normals_Hough' library
//! (https://github.com/aboulch/normals_Hough)
/** "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds"
        by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing
2016, Computer Graphics Forum
**/
class qHoughNormals : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qHoughNormals" FILE
                          "../info.json")

public:
    //! Default constructor
    explicit qHoughNormals(QObject* parent = nullptr);

    virtual ~qHoughNormals() = default;

    // inherited from ccStdPluginInterface
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;

protected:
    //! Slot called when associated action is triggered
    void doAction();

protected:
    //! Associated action
    QAction* m_action;
};
