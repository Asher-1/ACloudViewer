// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"

// LOCAL
#include "cvSelectionBase.h"
#include "cvSelectionData.h"

// Qt
#include <QGroupBox>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QWidget>

// Forward declarations
class cvSelectionHighlighter;
class cvSelectionTooltipHelper;
class cvViewSelectionManager;
class cvSelectionAlgebra;
class cvSelectionFilter;
class cvSelectionBookmarks;
class cvSelectionAnnotationManager;
class vtkPolyData;
class QPushButton;
class QColorDialog;
class QDoubleSpinBox;
class QCheckBox;
class QComboBox;
class ccHObject;

/**
 * @brief Comprehensive selection properties and management widget
 *
 * Inherits from QWidget and cvSelectionBase (lightweight) since it only needs
 * visualizer access for querying information. Does not need picking
 * functionality.
 *
 * This widget provides:
 * - Selection highlighting color/opacity configuration
 * - Tooltip display settings
 * - Selection statistics and detailed list
 * - Export functionality (to mesh/cloud/file)
 *
 * Based on ParaView's Selection Inspector panel:
 * - pqSelectionInspectorWidget
 * - pqSelectionManager
 * - pqOutputPort selection display
 */
class QPCL_ENGINE_LIB_API cvSelectionPropertiesWidget : public QWidget,
                                                        public cvSelectionBase {
    Q_OBJECT

public:
    explicit cvSelectionPropertiesWidget(QWidget* parent = nullptr);
    ~cvSelectionPropertiesWidget() override;

    // Inherited from cvSelectionBase:
    // void setVisualizer(ecvGenericVisualizer3D* viewer);
    // ecvGenericVisualizer3D* getVisualizer() const;
    // PclUtils::PCLVis* getPCLVis() const;

    /**
     * @brief Set the selection highlighter instance
     * @param highlighter Pointer to the highlighter
     */
    void setHighlighter(cvSelectionHighlighter* highlighter);

    /**
     * @brief Set the selection manager (provides access to utility modules)
     * @param manager Pointer to the selection manager
     */
    void setSelectionManager(cvViewSelectionManager* manager);

    /**
     * @brief Synchronize UI controls with highlighter's current settings
     * Called automatically by setHighlighter()
     */
    void syncUIWithHighlighter();

    /**
     * @brief Update with current selection data
     * @param selectionData The selection data
     * @param polyData The mesh data (optional, will be fetched if nullptr)
     * @return True if update was successful
     */
    bool updateSelection(const cvSelectionData& selectionData,
                         vtkPolyData* polyData = nullptr);

    /**
     * @brief Clear the selection information
     */
    void clearSelection();

    /**
     * @brief Get current selection data
     */
    const cvSelectionData& selectionData() const { return m_selectionData; }

    /**
     * @brief Get the selection manager
     */
    cvViewSelectionManager* selectionManager() const {
        return m_selectionManager;
    }

signals:
    /**
     * @brief Emitted when highlight color changes
     */
    void highlightColorChanged(double r,
                               double g,
                               double b,
                               int mode);  // HighlightMode

    /**
     * @brief Emitted when highlight opacity changes
     */
    void highlightOpacityChanged(double opacity, int mode);

    /**
     * @brief Emitted when tooltip display settings change
     */
    void tooltipSettingsChanged(bool showTooltips, int maxAttributes);

    /**
     * @brief Emitted when user requests algebra operation
     */
    void algebraOperationRequested(int operation);

    /**
     * @brief Emitted when user requests to save bookmark
     */
    void bookmarkRequested(const QString& name);

    /**
     * @brief Emitted when user requests to add annotation
     */
    void annotationRequested(const QString& text);

private slots:
    // Highlight configuration
    void onHoverColorClicked();
    void onPreselectedColorClicked();
    void onSelectedColorClicked();
    void onBoundaryColorClicked();

    void onHoverOpacityChanged(double value);
    void onPreselectedOpacityChanged(double value);
    void onSelectedOpacityChanged(double value);
    void onBoundaryOpacityChanged(double value);

    // Export actions
    void onExportToMeshClicked();
    void onExportToPointCloudClicked();
    void onExportToFileClicked();
    void onCopyIDsClicked();

    // Tooltip settings
    void onShowTooltipsToggled(bool checked);
    void onMaxAttributesChanged(int value);

    // List interaction
    void onSelectionListItemClicked(QListWidgetItem* item);

    // Advanced operations (new)
    void onAlgebraOperationTriggered();
    void onFilterOperationTriggered();
    void onSaveBookmarkClicked();
    void onLoadBookmarkClicked();
    void onBatchExportBookmarksClicked();
    void onAddAnnotationClicked();
    void onExtractBoundaryClicked();

private:
    void setupUi();
    void setupHighlightTab();
    void setupStatisticsTab();
    void setupExportTab();
    void setupAdvancedTab();  // New: algebra, filter, bookmarks, annotations

    void updateStatistics(vtkPolyData* polyData);
    void updateSelectionList(vtkPolyData* polyData);
    void computeBoundingBox(vtkPolyData* polyData, double bounds[6]);
    QString formatBounds(const double bounds[6]);

    void showColorDialog(const QString& title,
                         double currentColor[3],
                         int mode);

    /**
     * @brief Highlight a single item temporarily in the 3D view
     * @param id The ID of the item to highlight
     */
    void highlightSingleItem(qint64 id);

    /**
     * @brief Extract ID from list item text
     * @param itemText Text in format "ID: 123" or "ID: 123 (x, y, z)"
     * @return Extracted ID or -1 if parsing failed
     */
    qint64 extractIdFromItemText(const QString& itemText);

    /**
     * @brief Update bookmark combo box with current bookmarks
     */
    void updateBookmarkCombo();

private:
    // Helpers (m_viewer is inherited from cvGenericSelectionTool)
    cvSelectionHighlighter* m_highlighter;
    cvSelectionTooltipHelper* m_tooltipHelper;
    cvViewSelectionManager* m_selectionManager;  // Access to utility modules

    // Current selection data
    cvSelectionData m_selectionData;

    // UI components
    QTabWidget* m_tabWidget;

    // === Highlight Configuration Tab ===
    QWidget* m_highlightTab;
    // Color buttons (show current color, click to change)
    QPushButton* m_hoverColorButton;
    QPushButton* m_preselectedColorButton;
    QPushButton* m_selectedColorButton;
    QPushButton* m_boundaryColorButton;
    // Opacity spinboxes
    QDoubleSpinBox* m_hoverOpacitySpin;
    QDoubleSpinBox* m_preselectedOpacitySpin;
    QDoubleSpinBox* m_selectedOpacitySpin;
    QDoubleSpinBox* m_boundaryOpacitySpin;
    // Tooltip settings
    QCheckBox* m_showTooltipsCheckBox;
    QSpinBox* m_maxAttributesSpin;
    // Line width / point size
    QSpinBox* m_lineWidthSpin;
    QSpinBox* m_pointSizeSpin;

    // === Statistics Tab ===
    QWidget* m_statisticsTab;
    QLabel* m_countLabel;
    QLabel* m_typeLabel;
    QLabel* m_boundsLabel;
    QLabel* m_centerLabel;
    QLabel* m_volumeLabel;
    // Detailed list
    QListWidget* m_selectionListWidget;
    QLabel* m_listInfoLabel;

    // === Export Tab ===
    QWidget* m_exportTab;
    QPushButton* m_exportToMeshButton;
    QPushButton* m_exportToPointCloudButton;
    QPushButton* m_exportToFileButton;
    QPushButton* m_copyIDsButton;
    QLabel* m_exportInfoLabel;

    // === Advanced Tab === (new)
    QWidget* m_advancedTab;
    // Algebra operations
    QComboBox* m_algebraOpCombo;
    QPushButton* m_applyAlgebraButton;
    QPushButton* m_extractBoundaryButton;
    // Filtering
    QComboBox* m_filterTypeCombo;
    QPushButton* m_applyFilterButton;
    // Bookmarks
    QComboBox* m_bookmarkCombo;
    QPushButton* m_saveBookmarkButton;
    QPushButton* m_loadBookmarkButton;
    QPushButton* m_batchExportBookmarksButton;
    // Annotations
    QPushButton* m_addAnnotationButton;

    // Color storage (for color buttons)
    double m_hoverColor[3];
    double m_preselectedColor[3];
    double m_selectedColor[3];
    double m_boundaryColor[3];

    // Statistics
    int m_selectionCount;
    QString m_selectionType;
    double m_bounds[6];
    double m_center[3];
    double m_volume;
};
