// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "ecvOverlayDialog.h"
#include "ecvPickingListener.h"

// ECV_DB_LIB
#include <ecvHObject.h>

// SYSTEM
#include <set>

// GUI
#include <ui_tracePolylineDlg.h>

class ccPolyline;
class ccPointCloud;
class ccPickingHub;

//! Graphical Polyline Tracing tool
class ccTracePolylineTool : public ccOverlayDialog,
                            public ccPickingListener,
                            public Ui::TracePolyLineDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccTracePolylineTool(ccPickingHub* pickingHub, QWidget* parent);
    //! Destructor
    virtual ~ccTracePolylineTool();

    // inherited from ccOverlayDialog
    virtual bool linkWith(QWidget* win) override;
    virtual bool start() override;
    virtual void stop(bool accepted) override;

protected slots:

    void apply();
    void cancel();
    void exportLine();
    inline void continueEdition() { restart(false); }
    inline void resetLine() { restart(true); }

    // void handlePickedItem(ccHObject*, unsigned, int, int, const CCVector3&);
    // void addPointToPolyline(int x, int y);
    void closePolyLine(int x = 0,
                       int y = 0);  // arguments for compatibility with
                                    // ccGlWindow::rightButtonClicked signal

    void updatePolyLineTip(int x, int y, Qt::MouseButtons buttons);

    void onWidthSizeChanged(int);

    //! To capture overridden shortcuts (pause button, etc.)
    void onShortcutTriggered(int);

    //! Inherited from ccPickingListener
    virtual void onItemPicked(const PickedItem& pi) override;

protected:
    //! Restarts the edition mode
    void restart(bool reset);

    void resetTip();
    void updateTip();
    void resetPoly3D();
    void updatePoly3D();

    //! Oversamples the active 3D polyline
    ccPolyline* polylineOverSampling(unsigned steps) const;

    //! Viewport parameters (used for picking)
    struct SegmentGLParams {
        SegmentGLParams() {}
        SegmentGLParams(int x, int y);
        ccGLCameraParameters params;
        CCVector2d clickPos;
    };

    //! 2D polyline (for the currently edited part)
    ccPolyline* m_polyTip;
    //! 2D polyline vertices
    ccPointCloud* m_polyTipVertices;

    //! 3D polyline
    ccPolyline* m_poly3D;
    //! 3D polyline vertices
    ccPointCloud* m_poly3DVertices;

    //! Viewport parameters use to draw each segment of the polyline
    std::vector<SegmentGLParams> m_segmentParams;

    //! Current process state
    bool m_done;

    //! Picking hub
    ccPickingHub* m_pickingHub;
};
