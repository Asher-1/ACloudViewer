// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include <ecvOverlayDialog.h>
#include <ui_measurementToolDlg.h>

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericMeasurementTools.h>

// CV_PLUGIN_API
#include <ecvPickingListener.h>

// SYSTEM
#include <vector>

class ccHObject;
class ccPickingHub;
class ecvFontPropertyWidget;

//! Dialog for managing measurement tools (Distance, Contour, Protractor)
class ecvMeasurementTool : public ccOverlayDialog,
                           public Ui::MeasurementToolDlg,
                           public ccPickingListener {
    Q_OBJECT

public:
    //! Default constructor
    explicit ecvMeasurementTool(QWidget* parent);
    //! Default destructor
    virtual ~ecvMeasurementTool();

    // inherited from ccOverlayDialog
    virtual bool linkWith(QWidget* win) override;
    virtual bool start() override;
    virtual void stop(bool state) override;

    void setMeasurementTool(ecvGenericMeasurementTools* tool);
    ecvGenericMeasurementTools* getMeasurementTool() const { return m_tool; }

    //! Adds an entity
    /** \return success, if the entity is eligible for measurement
     **/
    bool addAssociatedEntity(ccHObject* anObject);

    //! Returns the current number of associated entities
    unsigned getNumberOfAssociatedEntity() const;

    inline ccHObject::Container getOutputs() const { return m_out_entities; }

    //! Inherited from ccPickingListener
    void onItemPicked(const PickedItem& pi) override;

protected slots:
    void reset();
    void closeDialog();
    void updateMeasurementDisplay();
    void toggleWidget(bool state);
    void exportMeasurement();
    void onInstanceChanged(int index);
    void addInstance();
    void removeInstance();
    void onPointPickingRequested(int pointIndex);
    void onPointPickingCancelled();
    void onColorButtonClicked();
    void onFontPropertiesChanged();

protected:
    //! Updates the measurement result display
    void updateResultDisplay();

    //! Updates UI from current tool
    void updateUIFromTool();

    //! Updates tool from UI
    void updateToolFromUI();

    //! Releases all associated entities
    void releaseAssociatedEntities();

    //! Creates a new measurement tool instance
    ecvGenericMeasurementTools* createMeasurementTool(
            ecvGenericMeasurementTools::MeasurementType type);

    //! Updates instances combo box
    void updateInstancesComboBox();

    //! Switches to the specified tool's UI
    void switchToToolUI(ecvGenericMeasurementTools* tool);

    //! Updates color button appearance based on current color
    void updateColorButtonAppearance();

    //! Applies current color to all tool instances
    void applyColorToAllTools();

    //! Applies font properties to tool instances (all or current based on
    //! checkbox)
    void applyFontToTools();

    //! Current measurement tool
    ecvGenericMeasurementTools* m_tool;

    //! List of all measurement tool instances
    QList<ecvGenericMeasurementTools*> m_toolInstances;

    ecvGenericMeasurementTools::MeasurementType m_measurementType;

    //! Associated entities container
    ccHObject m_entityContainer;

    ccHObject::Container m_out_entities;

    //! Flag to prevent recursive updates
    bool m_updatingFromTool;

    //! Picking hub for point selection
    ccPickingHub* m_pickingHub;

    //! Current point selection mode (0=none, 1=point1, 2=point2, 3=center)
    int m_pickPointMode;

    //! Current measurement color (default: green)
    QColor m_currentColor;

    //! Font property widget
    ecvFontPropertyWidget* m_fontPropertyWidget = nullptr;

    //! VTK widget reference from linkWith (for creating shortcuts in new
    //! instances)
    QWidget* m_linkedWidget = nullptr;
};
