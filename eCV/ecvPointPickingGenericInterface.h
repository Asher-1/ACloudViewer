// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_POINT_PICKING_GENERIC_INTERFACE_HEADER
#define ECV_POINT_PICKING_GENERIC_INTERFACE_HEADER

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
    //! Generic method to process picked points
    /** \param cloud picked point cloud
            \param pointIndex point index in cloud
            \param x picked pixel X position
            \param y picked pixel Y position
    **/
    virtual void processPickedPoint(ccPointCloud* cloud,
                                    unsigned pointIndex,
                                    int x,
                                    int y) = 0;

    //! Picking hub
    ccPickingHub* m_pickingHub;
};

#endif  // ECV_POINT_PICKING_GENERIC_INTERFACE_HEADER
