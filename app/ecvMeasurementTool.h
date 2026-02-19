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

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericMeasurementTools.h>

// CV_PLUGIN_API
#include <ecvPickingListener.h>

// SYSTEM
#include <vector>

class QScrollArea;
class QVBoxLayout;
class ccHObject;
class ccPickingHub;
class ecvFontPropertyWidget;

/**
 * @class ecvMeasurementTool
 * @brief Interactive measurement tool dialog
 * 
 * Overlay dialog for creating and managing various types of measurements in
 * 3D views. Supports multiple measurement types including:
 * 
 * - **Distance**: Point-to-point, point-to-plane, plane-to-plane distances
 * - **Angle**: Angular measurements (protractor)
 * - **Area**: Surface area and perimeter measurements
 * - **Volume**: 3D volume calculations
 * - **Contour**: Polyline contour measurements
 * 
 * Features:
 * - Multiple measurement instances support
 * - Interactive point picking in 3D views
 * - Customizable colors and fonts for measurement display
 * - Export measurements to file
 * - Real-time measurement updates during point selection
 * - DPI-adaptive UI layout
 * 
 * The tool integrates with CloudViewer's picking system to enable
 * interactive point/entity selection for measurement creation.
 * 
 * @see ecvGenericMeasurementTools
 * @see ccOverlayDialog
 * @see ccPickingListener
 */
class ecvMeasurementTool : public ccOverlayDialog,
                           public Ui::MeasurementToolDlg,
                           public ccPickingListener {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param parent Parent widget
     */
    explicit ecvMeasurementTool(QWidget* parent);
    
    /**
     * @brief Destructor
     * 
     * Cleans up measurement tool instances and releases associated entities.
     */
    virtual ~ecvMeasurementTool();

    /**
     * @brief Link tool to a 3D display window
     * 
     * Establishes connection with display window for rendering measurements
     * and handling picking events.
     * 
     * @param win Display window to link with
     * @return true if linked successfully
     */
    virtual bool linkWith(QWidget* win) override;
    
    /**
     * @brief Start the measurement tool
     * 
     * Activates measurement mode, enables picking, and shows the dialog.
     * @return true if started successfully
     */
    virtual bool start() override;
    
    /**
     * @brief Stop the measurement tool
     * 
     * Deactivates measurement mode, disables picking, and hides the dialog.
     * @param state Final state (true = accept, false = reject)
     */
    virtual void stop(bool state) override;

    /**
     * @brief Set current measurement tool
     * @param tool Measurement tool to set as active
     */
    void setMeasurementTool(ecvGenericMeasurementTools* tool);
    
    /**
     * @brief Get current measurement tool
     * @return Pointer to active measurement tool
     */
    ecvGenericMeasurementTools* getMeasurementTool() const { return m_tool; }

    /**
     * @brief Add entity for measurement
     * 
     * Associates an entity (point cloud, mesh, etc.) with the measurement tool.
     * Entity must be eligible for the current measurement type.
     * 
     * @param anObject Entity to associate
     * @return true if entity is eligible and was added successfully
     */
    bool addAssociatedEntity(ccHObject* anObject);

    /**
     * @brief Get number of associated entities
     * @return Count of entities currently associated with measurement
     */
    unsigned getNumberOfAssociatedEntity() const;

    /**
     * @brief Get output measurement entities
     * 
     * Returns container of measurement result entities (labels, polylines, etc.)
     * that were created by the tool.
     * 
     * @return Container of output entities
     */
    inline ccHObject::Container getOutputs() const { return m_out_entities; }

    /**
     * @brief Handle picked item event
     * 
     * Called when user picks a point/entity in 3D view. Updates measurement
     * based on picked item.
     * 
     * @param pi Picked item information
     */
    void onItemPicked(const PickedItem& pi) override;

protected slots:
    /**
     * @brief Reset measurement to initial state
     */
    void reset();
    
    /**
     * @brief Close the measurement dialog
     */
    void closeDialog();
    
    /**
     * @brief Update measurement display in 3D view
     */
    void updateMeasurementDisplay();
    
    /**
     * @brief Toggle widget visibility
     * @param state Visibility state
     */
    void toggleWidget(bool state);
    
    /**
     * @brief Export measurement to file
     */
    void exportMeasurement();
    
    /**
     * @brief Handle measurement instance change
     * @param index New instance index
     */
    void onInstanceChanged(int index);
    
    /**
     * @brief Add new measurement instance
     */
    void addInstance();
    
    /**
     * @brief Remove current measurement instance
     */
    void removeInstance();
    
    /**
     * @brief Handle point picking request
     * @param pointIndex Index of point to pick
     */
    void onPointPickingRequested(int pointIndex);
    
    /**
     * @brief Handle cancelled point picking
     */
    void onPointPickingCancelled();
    
    /**
     * @brief Handle color selection button click
     */
    void onColorButtonClicked();
    
    /**
     * @brief Handle font properties change
     */
    void onFontPropertiesChanged();

protected:
    /**
     * @brief Update measurement result display
     * 
     * Updates the numerical/text display of measurement results.
     */
    void updateResultDisplay();

    /**
     * @brief Update UI from current tool state
     * 
     * Synchronizes UI controls with current measurement tool properties.
     */
    void updateUIFromTool();

    /**
     * @brief Update tool from UI state
     * 
     * Synchronizes measurement tool properties with UI control values.
     */
    void updateToolFromUI();

    /**
     * @brief Release all associated entities
     * 
     * Clears and deletes all entities associated with measurements.
     */
    void releaseAssociatedEntities();

    /**
     * @brief Create new measurement tool instance
     * @param type Type of measurement tool to create
     * @return Pointer to new measurement tool
     */
    ecvGenericMeasurementTools* createMeasurementTool(
            ecvGenericMeasurementTools::MeasurementType type);

    /**
     * @brief Update instances combo box
     * 
     * Refreshes list of available measurement instances in UI.
     */
    void updateInstancesComboBox();

    /**
     * @brief Switch to tool's UI
     * @param tool Tool to switch to
     */
    void switchToToolUI(ecvGenericMeasurementTools* tool);

    /**
     * @brief Update color button appearance
     * @param color Color to display (uses m_currentColor if not specified)
     */
    void updateColorButtonAppearance(const QColor& color = QColor());

    /**
     * @brief Apply color to tool instances
     * 
     * Applies color to all instances or just current one based on
     * "Apply to all" checkbox state.
     * @param color Color to apply (uses m_currentColor if not specified)
     */
    void applyColorToAllTools(const QColor& color = QColor());

    /**
     * @brief Apply font properties to tool instances
     * 
     * Applies font settings to all or current tool based on checkbox.
     */
    void applyFontToTools();

    ecvGenericMeasurementTools* m_tool;  ///< Current active measurement tool

    QList<ecvGenericMeasurementTools*> m_toolInstances;  ///< All tool instances

    ecvGenericMeasurementTools::MeasurementType m_measurementType;  ///< Current measurement type

    ccHObject m_entityContainer;  ///< Container for associated entities

    ccHObject::Container m_out_entities;  ///< Output measurement entities

    bool m_updatingFromTool;  ///< Flag to prevent recursive UI updates

    ccPickingHub* m_pickingHub;  ///< Picking hub for point selection

    int m_pickPointMode;  ///< Point selection mode (0=none, 1-3=specific points)

    QColor m_currentColor;  ///< Current measurement color

    ecvFontPropertyWidget* m_fontPropertyWidget;  ///< Font property editor

    QWidget* m_linkedWidget;  ///< Linked VTK display widget

    QScrollArea* m_scrollArea;  ///< Scroll area for parameters (DPI-adaptive)

    QVBoxLayout* m_parametersLayout;  ///< Parameters layout container
};
