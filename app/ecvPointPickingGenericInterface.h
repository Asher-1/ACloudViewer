// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvCommon.h"
#include "ecvOverlayDialog.h"
#include "ecvPickingListener.h"

// cloudViewer
#include <CVGeom.h>

// system
#include <vector>

class ccPointCloud;
class ccHObject;
class ccPickingHub;

/** Generic interface for any dialog/graphical interactor that relies on point
 *picking.
 **/
class ccPointPickingGenericInterface : public ccOverlayDialog,
                                       public ccPickingListener {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccPointPickingGenericInterface(ccPickingHub* pickingHub,
                                            QWidget* parent = nullptr);
    //! Destructor
    ~ccPointPickingGenericInterface() override = default;

    // inherited from ccOverlayDialog
    bool linkWith(QWidget* win) override;
    bool start() override;
    void stop(bool state) override;

    //! Inherited from ccPickingListener
    void onItemPicked(const PickedItem& pi) override;

protected:
    //! Generic method to process picked items (clouds and meshes)
    virtual void processPickedPoint(const PickedItem& picked) = 0;

    //! Picking hub
    ccPickingHub* m_pickingHub;
};
