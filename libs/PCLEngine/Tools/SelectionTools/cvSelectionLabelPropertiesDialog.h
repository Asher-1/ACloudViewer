// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// Qt - must be included before qPCL.h for MOC to work correctly
#include <QtGui/QColor>
#include <QtWidgets/QDialog>
#include <QtGui/QFont>
#include <QtCore/QObject>
// clang-format on

#include "qPCL.h"

// Include full definition for FontProperties struct
#include <widgets/ecvFontPropertyWidget.h>

class QSlider;
class QSpinBox;
class QDoubleSpinBox;
class QLineEdit;
class QPushButton;
class QToolButton;
class QCheckBox;

/**
 * @brief Dialog for editing selection label properties
 *
 * Based on ParaView's Selection Label Properties dialog.
 * Allows configuration of:
 * - Opacity, Point Size, Line Width
 * - Cell Label Font (family, size, color, opacity, bold/italic/shadow)
 * - Cell Label Format
 * - Point Label Font (same options)
 * - Point Label Format
 */
class QPCL_ENGINE_LIB_API cvSelectionLabelPropertiesDialog : public QDialog {
    Q_OBJECT

public:
    /**
     * @brief Label properties structure
     */
    struct LabelProperties {
        // General
        double opacity = 1.0;
        int pointSize = 5;
        int lineWidth = 2;

        // Cell Label Font
        QString cellLabelFontFamily = "Arial";
        int cellLabelFontSize = 18;
        QColor cellLabelColor = QColor(0, 255, 0);  // Green
        bool cellLabelBold = false;
        bool cellLabelItalic = false;
        bool cellLabelShadow = false;
        double cellLabelOpacity = 1.0;
        QString cellLabelHorizontalJustification =
                "Left";  // "Left", "Center", "Right"
        QString cellLabelVerticalJustification =
                "Bottom";         // "Top", "Center", "Bottom"
        QString cellLabelFormat;  // e.g., "%s" for ID

        // Point Label Font
        QString pointLabelFontFamily = "Arial";
        int pointLabelFontSize = 18;
        QColor pointLabelColor = QColor(255, 255, 0);  // Yellow
        bool pointLabelBold = false;
        bool pointLabelItalic = false;
        bool pointLabelShadow = false;
        double pointLabelOpacity = 1.0;
        QString pointLabelHorizontalJustification =
                "Left";  // "Left", "Center", "Right"
        QString pointLabelVerticalJustification =
                "Bottom";          // "Top", "Center", "Bottom"
        QString pointLabelFormat;  // e.g., "%s" for ID

        // Tooltip Settings
        bool showTooltips = true;
        int maxTooltipAttributes = 15;
    };

    explicit cvSelectionLabelPropertiesDialog(QWidget* parent = nullptr,
                                              bool isInteractive = false);
    ~cvSelectionLabelPropertiesDialog() override;

    /**
     * @brief Set the current properties
     */
    void setProperties(const LabelProperties& props);

    /**
     * @brief Get the current properties
     */
    LabelProperties properties() const;

Q_SIGNALS:
    /**
     * @brief Emitted when Apply is clicked
     */
    void propertiesApplied(const LabelProperties& props);

private Q_SLOTS:
    void onApplyClicked();
    void onResetClicked();
    void onOpacitySliderChanged(int value);
    void onCellFontPropertiesChanged();
    void onPointFontPropertiesChanged();

private:
    void setupUi();
    void loadDefaults();
    void updatePropertiesFromWidgets();

    // Helper functions to reduce code duplication (implemented in .cpp)
    static ecvFontPropertyWidget::FontProperties
    labelPropertiesToFontProperties(const LabelProperties& props,
                                    bool isCellLabel);
    static void fontPropertiesToLabelProperties(
            const ecvFontPropertyWidget::FontProperties& fontProps,
            LabelProperties& props,
            bool isCellLabel);

private:
    bool m_isInteractive;
    LabelProperties m_properties;
    LabelProperties m_defaultProperties;

    // General settings
    QSlider* m_opacitySlider = nullptr;
    QDoubleSpinBox* m_opacitySpin = nullptr;
    QSpinBox* m_pointSizeSpin = nullptr;
    QSpinBox* m_lineWidthSpin = nullptr;

    // Cell Label Font Widget
    ecvFontPropertyWidget* m_cellFontWidget = nullptr;
    QLineEdit* m_cellFormatEdit = nullptr;

    // Point Label Font Widget
    ecvFontPropertyWidget* m_pointFontWidget = nullptr;
    QLineEdit* m_pointFormatEdit = nullptr;

    // Tooltip Settings
    QCheckBox* m_showTooltipsCheckBox = nullptr;
    QSpinBox* m_maxTooltipAttributesSpin = nullptr;

    // Dialog buttons
    QToolButton* m_refreshButton = nullptr;
    QToolButton* m_saveButton = nullptr;
    QPushButton* m_applyButton = nullptr;
    QPushButton* m_resetButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPushButton* m_okButton = nullptr;
};
