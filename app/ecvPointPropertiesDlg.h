// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvPointPickingGenericInterface.h"

// CV_DB_LIB
#include <ecvGenericGLDisplay.h>

// Local
#include <ui_pointPropertiesDlg.h>

class cc2DLabel;
class cc2DViewportLabel;
class ccHObject;

//! Dialog for simple point picking (information, distance, etc.)
class ccPointPropertiesDlg : public ccPointPickingGenericInterface,
                             public Ui::PointPropertiesDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccPointPropertiesDlg(ccPickingHub* pickingHub, QWidget* parent);
    //! Default destructor
    virtual ~ccPointPropertiesDlg();

    // inherited from ccPointPickingGenericInterface
    virtual bool start() override;
    virtual void stop(bool state) override;
    virtual bool linkWith(QWidget* win) override;

protected slots:

    void onClose();
    void activatePointPropertiesDisplay();
    void activateDistanceDisplay();
    void activateAngleDisplay();
    void activate2DZonePicking();
    void initializeState();
    void exportCurrentLabel();
    void update2DZone(int x, int y, Qt::MouseButtons buttons);
    void processClickedPoint(int x, int y);
    void close2DZone();

signals:

    //! Signal emitted when a new label is created
    void newLabel(ccHObject*);

protected:
    //! Picking mode
    enum Mode { POINT_INFO, POINT_POINT_DISTANCE, POINTS_ANGLE, RECT_ZONE };

    //! Sets interaction flags on the current effective view, releasing the
    //! flags previously applied to another view. With multiple views, the
    //! effective view may change while the dialog is open (the user clicks
    //! another window); without this cleanup the old view would keep e.g.
    //! INTERACT_SEND_ALL_SIGNALS forever and stop responding to camera
    //! rotate/pan/zoom.
    void restrictViewInteraction(ecvGenericGLDisplay::INTERACTION_FLAGS flags);

    // inherited from ccPointPickingGenericInterface
    void processPickedPoint(const PickedItem& picked) override;

    //! Current picking mode
    Mode m_pickingMode;

    //! View currently carrying restricted interaction flags (or nullptr)
    ecvGenericGLDisplay* m_restrictedView = nullptr;

    //! Associated 3D label
    cc2DLabel* m_label;

    //! Associated 2D label
    cc2DViewportLabel* m_rect2DLabel;
};
