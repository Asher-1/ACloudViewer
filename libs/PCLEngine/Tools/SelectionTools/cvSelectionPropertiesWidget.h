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
#include "cvSelectionLabelPropertiesDialog.h"

// Qt
#include <QColor>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTableWidget>
#include <QToolButton>
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
class QLineEdit;
class QMenu;
class ccHObject;
class QScrollArea;

/**
 * @brief Comprehensive selection properties and management widget
 * 
 * Based on ParaView's Find Data panel with:
 * - pqFindDataSelectionDisplayFrame (Selection Display)
 * - pqSelectionEditor (Selection Editor)
 * - pqFindDataCurrentSelectionFrame (Selected Data spreadsheet)
 * 
 * Layout:
 * 1. Selected Data header with action buttons
 * 2. Selection Display (collapsible) - labels, colors, label properties
 * 3. Selection Editor (collapsible) - data producer, expression, saved selections
 * 4. Selected Data table (spreadsheet view with attributes)
 */
class QPCL_ENGINE_LIB_API cvSelectionPropertiesWidget : public QWidget,
                                                        public cvSelectionBase {
    Q_OBJECT

public:
    /**
     * @brief Saved selection entry for Selection Editor
     */
    struct SavedSelection {
        QString name;          // e.g., "s0", "s1"
        QString type;          // e.g., "ID Selection"
        QColor color;          // Display color
        cvSelectionData data;  // The actual selection data
    };

    explicit cvSelectionPropertiesWidget(QWidget* parent = nullptr);
    ~cvSelectionPropertiesWidget() override;

    // cvSelectionBase interface
    void setHighlighter(cvSelectionHighlighter* highlighter);
    void setSelectionManager(cvViewSelectionManager* manager);
    void syncUIWithHighlighter();

    bool updateSelection(const cvSelectionData& selectionData,
                         vtkPolyData* polyData = nullptr);
    void clearSelection();

    const cvSelectionData& selectionData() const { return m_selectionData; }
    cvViewSelectionManager* selectionManager() const { return m_selectionManager; }

    /**
     * @brief Set the data producer name (source of selection)
     */
    void setDataProducerName(const QString& name);

signals:
    // Highlight/opacity changes
    void highlightColorChanged(double r, double g, double b, int mode);
    void highlightOpacityChanged(double opacity, int mode);

    // Selection Editor signals
    void expressionChanged(const QString& expression);
    void activateCombinedSelectionsRequested();
    void selectionAdded(const cvSelectionData& selection);
    void selectionRemoved(int index);
    void allSelectionsRemoved();

    // Find Data actions
    void freezeSelectionRequested();
    void extractSelectionRequested();
    void plotOverTimeRequested();
    void invertSelectionRequested();

    // Legacy signals
    void algebraOperationRequested(int operation);
    void bookmarkRequested(const QString& name);
    void annotationRequested(const QString& text);

private slots:
    // === Selection Display slots ===
    void onCellLabelsClicked();
    void onPointLabelsClicked();
    void onEditLabelPropertiesClicked();
    void onSelectionColorClicked();
    void onInteractiveSelectionColorClicked();
    void onEditInteractiveLabelPropertiesClicked();
    void onLabelPropertiesApplied(const cvSelectionLabelPropertiesDialog::LabelProperties& props);
    void onInteractiveLabelPropertiesApplied(const cvSelectionLabelPropertiesDialog::LabelProperties& props);

    // === Selection Editor slots ===
    void onExpressionChanged(const QString& text);
    void onAddActiveSelectionClicked();
    void onRemoveSelectedSelectionClicked();
    void onRemoveAllSelectionsClicked();
    void onActivateCombinedSelectionsClicked();
    void onSelectionEditorTableSelectionChanged();

    // === Find Data / Selected Data slots ===
    void onAttributeTypeChanged(int index);
    void onInvertSelectionToggled(bool checked);
    void onFreezeClicked();
    void onExtractClicked();
    void onPlotOverTimeClicked();
    void onToggleColumnVisibility();
    void onSpreadsheetItemClicked(QTableWidgetItem* item);

    // Legacy slots (existing functionality)
    void onHoverColorClicked();
    void onPreselectedColorClicked();
    void onSelectedColorClicked();
    void onBoundaryColorClicked();
    void onHoverOpacityChanged(double value);
    void onPreselectedOpacityChanged(double value);
    void onSelectedOpacityChanged(double value);
    void onBoundaryOpacityChanged(double value);
    void onExportToMeshClicked();
    void onExportToPointCloudClicked();
    void onExportToFileClicked();
    void onCopyIDsClicked();
    void onSelectionTableItemClicked(QTableWidgetItem* item);
    void onAlgebraOperationTriggered();
    void onFilterOperationTriggered();
    void onSaveBookmarkClicked();
    void onLoadBookmarkClicked();
    void onBatchExportBookmarksClicked();
    void onAddAnnotationClicked();
    void onExtractBoundaryClicked();

private:
    void setupUi();
    
    // ParaView-style sections
    void setupSelectedDataHeader();
    void setupSelectionDisplaySection();
    void setupSelectionEditorSection();
    void setupSelectedDataSpreadsheet();
    
    // Legacy tabs (for backwards compatibility)
    void setupStatisticsTab();
    void setupExportTab();
    void setupAdvancedTab();

    void updateStatistics(vtkPolyData* polyData);
    void updateSelectionList(vtkPolyData* polyData);
    void updateSpreadsheetData(vtkPolyData* polyData);
    void computeBoundingBox(vtkPolyData* polyData, double bounds[6]);
    QString formatBounds(const double bounds[6]);
    void showColorDialog(const QString& title, double currentColor[3], int mode);
    void highlightSingleItem(qint64 id);
    qint64 extractIdFromItemText(const QString& itemText);
    void updateBookmarkCombo();
    void updateSelectionEditorTable();
    QString generateSelectionName();
    QColor generateSelectionColor() const;

private:
    // Core components
    cvSelectionHighlighter* m_highlighter;
    cvSelectionTooltipHelper* m_tooltipHelper;
    cvViewSelectionManager* m_selectionManager;
    cvSelectionData m_selectionData;

    // Label properties (ParaView-style)
    cvSelectionLabelPropertiesDialog::LabelProperties m_labelProperties;
    cvSelectionLabelPropertiesDialog::LabelProperties m_interactiveLabelProperties;

    // Saved selections for Selection Editor
    QList<SavedSelection> m_savedSelections;
    QString m_dataProducerName;

    // Main scroll area
    QScrollArea* m_scrollArea;
    QWidget* m_scrollContent;

    // === Selected Data Header ===
    QLabel* m_selectedDataLabel;
    QPushButton* m_freezeButton;
    QPushButton* m_extractButton;
    QPushButton* m_plotOverTimeButton;

    // === Selection Display Section ===
    QGroupBox* m_selectionDisplayGroup;
    // Selection Labels
    QPushButton* m_cellLabelsButton;
    QPushButton* m_pointLabelsButton;
    QMenu* m_cellLabelsMenu;
    QMenu* m_pointLabelsMenu;
    QPushButton* m_editLabelPropertiesButton;
    // Selection Appearance
    QPushButton* m_selectionColorButton;
    // Interactive Selection
    QPushButton* m_interactiveSelectionColorButton;
    QPushButton* m_editInteractiveLabelPropertiesButton;

    // === Selection Editor Section ===
    QGroupBox* m_selectionEditorGroup;
    QLabel* m_dataProducerLabel;
    QLabel* m_dataProducerValue;
    QLabel* m_elementTypeLabel;
    QLabel* m_elementTypeValue;  // ParaView-style: shows element type in a styled label
    QLabel* m_expressionLabel;
    QLineEdit* m_expressionEdit;
    QTableWidget* m_selectionEditorTable;  // Name, Type, Color columns
    QToolButton* m_addSelectionButton;
    QToolButton* m_removeSelectionButton;
    QToolButton* m_removeAllSelectionsButton;
    QPushButton* m_activateCombinedSelectionsButton;

    // === Selected Data Spreadsheet ===
    QGroupBox* m_selectedDataGroup;
    QComboBox* m_attributeTypeCombo;
    QToolButton* m_toggleColumnVisibilityButton;
    QToolButton* m_toggleFieldDataButton;  // ParaView-style: toggle field data visibility
    QCheckBox* m_invertSelectionCheck;
    QTableWidget* m_spreadsheetTable;

    // === Legacy UI (for backwards compatibility) ===
    QTabWidget* m_tabWidget;
    QPushButton* m_hoverColorButton;
    QPushButton* m_preselectedColorButton;
    QPushButton* m_selectedColorButton;
    QPushButton* m_boundaryColorButton;
    QDoubleSpinBox* m_hoverOpacitySpin;
    QDoubleSpinBox* m_preselectedOpacitySpin;
    QDoubleSpinBox* m_selectedOpacitySpin;
    QDoubleSpinBox* m_boundaryOpacitySpin;
    QSpinBox* m_lineWidthSpin;
    QSpinBox* m_pointSizeSpin;

    QWidget* m_statisticsTab;
    QLabel* m_countLabel;
    QLabel* m_typeLabel;
    QLabel* m_boundsLabel;
    QLabel* m_centerLabel;
    QLabel* m_volumeLabel;
    QTableWidget* m_selectionTableWidget;
    QLabel* m_listInfoLabel;

    QWidget* m_exportTab;
    QPushButton* m_exportToMeshButton;
    QPushButton* m_exportToPointCloudButton;
    QPushButton* m_exportToFileButton;
    QPushButton* m_copyIDsButton;
    QLabel* m_exportInfoLabel;

    QWidget* m_advancedTab;
    QComboBox* m_algebraOpCombo;
    QPushButton* m_applyAlgebraButton;
    QPushButton* m_extractBoundaryButton;
    QComboBox* m_filterTypeCombo;
    QPushButton* m_applyFilterButton;
    QComboBox* m_bookmarkCombo;
    QPushButton* m_saveBookmarkButton;
    QPushButton* m_loadBookmarkButton;
    QPushButton* m_batchExportBookmarksButton;
    QPushButton* m_addAnnotationButton;

    // Color storage
    double m_hoverColor[3];
    double m_preselectedColor[3];
    double m_selectedColor[3];
    double m_boundaryColor[3];
    QColor m_selectionColor;
    QColor m_interactiveSelectionColor;

    // Statistics
    int m_selectionCount;
    QString m_selectionType;
    double m_bounds[6];
    double m_center[3];
    double m_volume;

    // Selection name counter
    int m_selectionNameCounter;

    // Selection colors palette (like ParaView)
    static const QColor s_selectionColors[];
    static const int s_selectionColorsCount;

    // Current label array selections (for Cell/Point Labels menus)
    QString m_currentCellLabelArray;
    QString m_currentPointLabelArray;

    // For single item highlighting with RED color
    double m_savedPreselectedColor[3];
    qint64 m_lastHighlightedId;
};
