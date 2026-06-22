// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include <ecvOverlayDialog.h>

// CV_CORE_LIB
#include <ScalarField.h>

// CV_DB_LIB
#include <ecvGenericGLDisplay.h>
#include <ecvHObject.h>

// QT
#include <QSet>
#include <set>

// GUI
#include <ui_graphicalSegmentationDlg.h>

class ccPolyline;
class ccPointCloud;
class ecvMainAppInterface;

//! Graphical segmentation mechanism (with polyline)
class ccGraphicalSegmentationTool : public ccOverlayDialog,
                                    public Ui::GraphicalSegmentationDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccGraphicalSegmentationTool(QWidget* parent);
    //! Destructor
    virtual ~ccGraphicalSegmentationTool();

    //! Adds an entity (and/or its children) to the 'to be segmented' pool
    /** Warning: some entities may be rejected if they are
            locked, or can't be segmented this way.
            \return whether entity has been added to the pool or not
    **/
    bool addEntity(ccHObject* anObject, bool silent = false);

    //! Returns the number of entites currently in the the 'to be segmented'
    //! pool
    unsigned getNumberOfValidEntities() const;

    //! Get a pointer to the polyline that has been segmented
    ccPolyline* getPolyLine() { return m_segmentationPoly; }

    //! Returns the active 'to be segmented' set
    QSet<ccHObject*>& entities() { return m_toSegment; }
    //! Returns the active 'to be segmented' set (const version)
    const QSet<ccHObject*>& entities() const { return m_toSegment; }

    // inherited from ccOverlayDialog
    virtual bool linkWith(QWidget* win) override;
    virtual bool start() override;
    virtual void stop(bool accepted) override;

    //! Returns whether hidden parts should be delete after segmentation
    bool deleteHiddenParts() const { return m_deleteHiddenParts; }

    //! Remove entities from the 'to be segmented' pool
    /** \warning 'unallocateVisibilityArray' will be called on all point clouds
            prior to be removed from the pool.
    **/
    void removeAllEntities();

    //! Apply segmentation and update the database (helper)
    bool applySegmentation(ecvMainAppInterface* app,
                           ccHObject::Container& newEntities);

protected slots:

    void segmentIn();
    void segmentOut();
    void exportSelection();
    void setClassificationValue();
    void reset();
    void apply();
    void applyAndDelete();
    void cancel();
    void addPointToPolyline(int x, int y);
    void closePolyLine(int x = 0,
                       int y = 0);  // arguments for compatibility with
                                    // ccGlWindow::rightButtonClicked signal
    void updateSegmentation();
    void closeRectangle();
    void updatePolyLine(int x, int y, Qt::MouseButtons buttons);
    void run();
    void stopRunning();
    void pauseSegmentationMode(bool state, bool only2D = true);
    inline void pauseSegmentation(bool state) { pauseSegmentationMode(state); }
    void resetSegmentation();
    void doSetPolylineSelection();
    void doSetRectangularSelection();
    void doActionUseExistingPolyline();
    void doExportSegmentationPolyline();

    //! To capture overridden shortcuts (pause button, etc.)
    void onShortcutTriggered(int);

    //! Prepare entity before removal
    void prepareEntityForRemoval(ccHObject* entity,
                                 bool unallocateVisibilityArrays);

    //! Whether to allow or not to exort the current segmentation polyline
    void allowPolylineExport(bool state);

signals:
    void currentScalarFieldUpdated();

protected:
    void setDrawFlag(bool state = true);

    //! Segments currently selected entities with the segmentation polyline
    void segment(bool keepPointsInside,
                 ScalarType classificationValue = NAN_VALUE,
                 bool exportSelection = false);

    //! Opens segmentation options
    void options();

    //! Set of entities to be segmented
    QSet<ccHObject*> m_toSegment;

    //! Whether something has changed or not (for proper 'cancel')
    bool m_somethingHasChanged;

    //! Process states
    enum ProcessStates {
        POLYLINE = 1,
        RECTANGLE = 2,
        //...			= 4,
        //...			= 8,
        //...			= 16,
        PAUSED = 32,
        STARTED = 64,
        RUNNING = 128,
    };

    //! Current process state
    unsigned m_state;

    //! Segmentation polyline
    ccPolyline* m_segmentationPoly;
    //! Segmentation polyline vertices
    ccPointCloud* m_polyVertices;

    //! Selection mode
    bool m_rectangularSelection;

    //! Whether to delete hidden parts after segmentation
    bool m_deleteHiddenParts;

    //! Entities created by export-selection mode, enabled when the tool closes
    std::set<ccHObject*> m_enableOnClose;

    //! Source entities hidden when export-selection mode closes
    std::set<ccHObject*> m_disableOnClose;

    //! Saved view state (restored on stop or pause)
    ecvGenericGLDisplay::INTERACTION_FLAGS m_savedInteractionMode =
            ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA;
    bool m_savedPerspectiveState = true;
    bool m_savedObjectCenteredView = true;
};
