// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// GUI
#include <ui_pointListPickingDlg.h>

// Local
#include "ecvPointPickingGenericInterface.h"

// ECV_DB_LIB
#include <ecvHObject.h>

class cc2DLabel;

//! Dialog/interactor to graphically pick a list of points
/** Options let the user export the list to an ASCII file, a new cloud, a
 *polyline, etc.
 **/
class ccPointListPickingDlg : public ccPointPickingGenericInterface,
                              public Ui::PointListPickingDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccPointListPickingDlg(ccPickingHub* pickingHub, QWidget* parent);

    //! Associates dialog with cloud
    void linkWithCloud(ccPointCloud* cloud);

protected slots:

    //! Applies changes and exit
    void applyAndExit();
    //! Cancels process and exit
    void cancelAndExit();
    //! Exports list to a new cloud
    void exportToNewCloud();
    //! Exports list to a polyline
    void exportToNewPolyline();
    //! Removes last inserted point from list
    void removeLastEntry();

    //! Exports list to an 'xyz' ASCII file
    inline void exportToASCII_xyz() {
        return exportToASCII(PLP_ASCII_EXPORT_XYZ);
    }
    //! Exports list to an 'ixyz' ASCII file
    inline void exportToASCII_ixyz() {
        return exportToASCII(PLP_ASCII_EXPORT_IXYZ);
    }
    //! Exports list to an 'gxyz' ASCII file
    inline void exportToASCII_gxyz() {
        return exportToASCII(PLP_ASCII_EXPORT_GXYZ);
    }
    //! Exports list to an 'lxyz' ASCII file
    inline void exportToASCII_lxyz() {
        return exportToASCII(PLP_ASCII_EXPORT_LXYZ);
    }

    //! Redraw window when marker size changes
    void markerSizeChanged(int);
    //! Redraw window when starting index changes
    void startIndexChanged(int);
    //! Updates point list widget
    void updateList();

protected:
    // inherited from ccPointPickingGenericInterface
    virtual void processPickedPoint(ccPointCloud* cloud,
                                    unsigned pointIndex,
                                    int x,
                                    int y) override;

    //! Gets current (visible) picked points from the associated cloud
    unsigned getPickedPoints(std::vector<cc2DLabel*>& pickedPoints);

    void clearLastLabel(ccHObject* lastVisibleLabel);
    void removeEntity(ccHObject* lastVisibleLabel);

    //! Export format
    /** See exportToASCII.
     **/
    enum ExportFormat {
        PLP_ASCII_EXPORT_XYZ,
        PLP_ASCII_EXPORT_IXYZ,
        PLP_ASCII_EXPORT_GXYZ,
        PLP_ASCII_EXPORT_LXYZ
    };

    //! Exports list to an ASCII file
    void exportToASCII(ExportFormat format);

    //! Associated cloud
    ccPointCloud* m_associatedCloud;

    //! Last existing label unique ID on load
    unsigned m_lastPreviousID;
    //! Ordered labels container
    ccHObject* m_orderedLabelsContainer;
    //! Existing picked points that the user wants to delete (for proper
    //! "cancel" mechanism)
    ccHObject::Container m_toBeDeleted;
    //! New picked points that the user has selected (for proper "cancel"
    //! mechanism)
    ccHObject::Container m_toBeAdded;
};
