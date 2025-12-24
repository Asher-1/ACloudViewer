// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// Qt - must be included before qPCL.h for MOC to work correctly
#include <QDialog>
#include <QMap>
#include <QObject>
#include <QString>
#include <QVariant>
// clang-format on

#include "cvSelectionFilter.h"
#include "qPCL.h"

// Forward declarations
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QStackedWidget;
class QVBoxLayout;
class vtkPolyData;

/**
 * @brief Filter configuration dialog (ParaView-aligned)
 *
 * Provides a user interface for configuring selection filters with
 * type-specific parameters. Based on ParaView's pqFindDataWidget design.
 *
 * Supported filter types:
 * - Attribute filtering (threshold, range)
 * - Geometric filtering (bounding box, sphere)
 * - Spatial filtering (proximity, region)
 * - Topology filtering (connectivity, boundary)
 *
 * Reference: ParaView's pqFindDataWidget and pqFindDataDialog
 */
class QPCL_ENGINE_LIB_API cvFilterConfigDialog : public QDialog {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param filterType The type of filter to configure
     * @param polyData Optional mesh data for preview/validation
     * @param parent Parent widget
     */
    explicit cvFilterConfigDialog(cvSelectionFilter::FilterType filterType,
                                  vtkPolyData* polyData = nullptr,
                                  QWidget* parent = nullptr);
    ~cvFilterConfigDialog() override;

    /**
     * @brief Get the configured filter parameters
     * @return Map of parameter names to values
     */
    QMap<QString, QVariant> getParameters() const;

    /**
     * @brief Get the selected filter type
     * @return Filter type enum value
     */
    cvSelectionFilter::FilterType getFilterType() const;

private slots:
    void onFilterTypeChanged(int index);
    void onPreviewClicked();
    void onAccepted();
    void onRejected();

private:
    void setupUI();
    void setupAttributeFilterUI();
    void setupGeometricFilterUI();
    void setupSpatialFilterUI();
    void setupTopologyFilterUI();

    void loadParameters();
    void saveParameters();
    bool validateParameters();

private:
    cvSelectionFilter::FilterType m_filterType;
    vtkPolyData* m_polyData;

    // UI widgets
    QComboBox* m_filterTypeCombo;
    QStackedWidget* m_parameterStack;
    QPushButton* m_previewButton;
    QPushButton* m_okButton;
    QPushButton* m_cancelButton;

    // Attribute filter widgets
    QComboBox* m_attributeCombo;
    QDoubleSpinBox* m_minValueSpin;
    QDoubleSpinBox* m_maxValueSpin;
    QComboBox* m_operatorCombo;

    // Geometric filter widgets
    QDoubleSpinBox* m_centerXSpin;
    QDoubleSpinBox* m_centerYSpin;
    QDoubleSpinBox* m_centerZSpin;
    QDoubleSpinBox* m_radiusSpin;
    QDoubleSpinBox* m_xMinSpin;
    QDoubleSpinBox* m_xMaxSpin;
    QDoubleSpinBox* m_yMinSpin;
    QDoubleSpinBox* m_yMaxSpin;
    QDoubleSpinBox* m_zMinSpin;
    QDoubleSpinBox* m_zMaxSpin;

    // Spatial filter widgets
    QDoubleSpinBox* m_distanceSpin;
    QComboBox* m_proximityTypeCombo;

    // Topology filter widgets
    QComboBox* m_topologyTypeCombo;

    // Parameter storage
    QMap<QString, QVariant> m_parameters;
};
