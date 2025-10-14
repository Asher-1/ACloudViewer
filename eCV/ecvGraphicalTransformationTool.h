// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_GRAPHICAL_TRANSFORMATION_TOOL_HEADER
#define ECV_GRAPHICAL_TRANSFORMATION_TOOL_HEADER

// Local
#include <ecvOverlayDialog.h>
#include <ui_graphicalTransformationDlg.h>

// ECV_DB_LIB
#include <ecvHObject.h>

//! Dialog + mechanism for graphical transformation of entities
/** Mouse driven rotation and translation of selected entities at screen.
 **/
class ecvGenericTransformTool;
class ccGraphicalTransformationTool : public ccOverlayDialog,
                                      public Ui::GraphicalTransformationDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccGraphicalTransformationTool(QWidget* parent);
    //! Default destructor
    virtual ~ccGraphicalTransformationTool();

    // inherited from ccOverlayDialog
    virtual bool linkWith(QWidget* win) override;
    virtual bool start() override;
    virtual void stop(bool state) override;

    bool setTansformTool(ecvGenericTransformTool* tool);

    //! Adds an entity to the 'selected' entities set
    /** Only the 'selected' entities are moved.
            \return success, if the entity is eligible for graphical
    transformation
    **/
    bool addEntity(ccHObject* anObject);

    //! Returns the number of valid entities (see addEntity)
    unsigned getNumberOfValidEntities() const;

    //! Returns the 'to be transformed' entities set (see addEntity)
    const ccHObject& getValidEntities() const { return m_toTransform; }

    //! Sets the rotation center
    void setRotationCenter(CCVector3d& center);

    void exportNewEntities();

    //! Clear all variables and 'unlink' dialog
    void clear();

protected slots:

    //! Applies transformation to selected entities
    void apply();

    //! Resets transformation
    void reset();

    //! Cancels (no transformation is applied)
    void cancel();

    //! Pauses the transformation mode
    void pause(bool);

    //! To capture overridden shortcuts (pause button, etc.)
    void onShortcutTriggered(int);

    void onScaleEnabled(bool dummy);
    void onShearEnabled(bool dummy);
    void onRotationModeChanged(int dummy);
    void onTranlationModeChanged(bool dummy);

protected:
    //! List of entities to be transformed
    ccHObject m_toTransform;

    //! Rotation center
    /** The rotation center is actually the center of gravity of the selected
     *'entities'
     **/
    CCVector3d m_rotationCenter;

    ecvGenericTransformTool* m_tool;
};

#endif  // ECV_GRAPHICAL_TRANSFORMATION_TOOL_HEADER
