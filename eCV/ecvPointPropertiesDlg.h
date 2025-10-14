// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_POINT_PROPERTIES_DIALOG_HEADER
#define ECV_POINT_PROPERTIES_DIALOG_HEADER

#include "ecvPointPickingGenericInterface.h"

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

    // inherited from ccPointPickingGenericInterface
    void processPickedPoint(ccPointCloud* cloud,
                            unsigned pointIndex,
                            int x,
                            int y) override;

    //! Current picking mode
    Mode m_pickingMode;

    //! Associated 3D label
    cc2DLabel* m_label;

    //! Associated 2D label
    cc2DViewportLabel* m_rect2DLabel;
};

#endif  // ECV_POINT_PROPERTIES_DIALOG_HEADER
