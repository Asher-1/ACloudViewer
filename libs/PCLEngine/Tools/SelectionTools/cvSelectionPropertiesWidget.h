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
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtCore/QObject>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>
// clang-format on

// Forward declaration
class cvExpanderButton;

#include "qPCL.h"

// LOCAL
#include "cvSelectionBase.h"
#include "cvSelectionData.h"
#include "cvSelectionHighlighter.h"            // For SelectionLabelProperties
#include "cvSelectionLabelPropertiesDialog.h"  // For LabelProperties

// Forward declarations
class cvSelectionHighlighter;
class cvTooltipFormatter;
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
class QVBoxLayout;
class QSpinBox;
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
 * 3. Selection Editor (collapsible) - data producer, expression, saved
 * selections
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
    cvViewSelectionManager* selectionManager() const {
        return m_selectionManager;
    }

    /**
     * @brief Set the data producer name (source of selection)
     */
    void setDataProducerName(const QString& name);

    /**
     * @brief Refresh the data producer combo box
     * Call this when data sources change (e.g., after loading new data)
     */
    void refreshDataProducers();

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
    // Emitted when extraction is successful
    void extractedObjectReady(ccHObject* obj);
    void plotOverTimeRequested();
    void invertSelectionRequested();
    void findDataRequested(const QString& dataProducer,
                           const QString& elementType,
                           const QString& attribute,
                           const QString& op,
                           const QString& value);

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
    void onLabelPropertiesApplied(
            const cvSelectionLabelPropertiesDialog::LabelProperties& props);
    void onInteractiveLabelPropertiesApplied(
            const cvSelectionLabelPropertiesDialog::LabelProperties& props);

    // === Selection Editor slots ===
    void onExpressionChanged(const QString& text);
    void onAddActiveSelectionClicked();
    void onRemoveSelectedSelectionClicked();
    void onRemoveAllSelectionsClicked();
    void onActivateCombinedSelectionsClicked();
    void onSelectionEditorTableSelectionChanged();
    void onSelectionEditorCellClicked(int row, int column);
    void onSelectionEditorCellDoubleClicked(int row, int column);

    // === Find Data / Selected Data slots ===
    void onAttributeTypeChanged(int index);
    void onInvertSelectionToggled(bool checked);
    void onFreezeClicked();
    void onExtractClicked();
    void onPlotOverTimeClicked();
    void onToggleColumnVisibility();
    void onToggleFieldDataClicked(bool checked);

    // Create Selection (Find Data) slots
    void onDataProducerChanged(int index);
    void onElementTypeChanged(int index);
    void onFindDataClicked();
    void onResetClicked();
    void onClearClicked();
    void onAddQueryClicked();
    void onRemoveQueryClicked();
    void updateAttributeCombo();
    void updateDataProducerCombo();
    void performFindData(const QString& attribute,
                         const QString& op,
                         const QString& value,
                         bool isCell);
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
    void onSelectionTableItemClicked(QTableWidgetItem* item);
    void onAlgebraOperationTriggered();
    void onFilterOperationTriggered();
    void onSaveBookmarkClicked();
    void onLoadBookmarkClicked();
    void onBatchExportBookmarksClicked();
    void onAddAnnotationClicked();
    void onExtractBoundaryClicked();

    // Highlighter property change handlers (for external property changes)
    void onHighlighterColorChanged(int mode);
    void onHighlighterOpacityChanged(int mode);
    void onHighlighterLabelPropertiesChanged(bool interactive);

private:
    void setupUi();

    // ParaView-style sections
    void setupSelectedDataHeader();
    void setupCreateSelectionSection();
    void setupSelectionDisplaySection();
    void setupSelectionEditorSection();
    void setupSelectedDataSpreadsheet();
    void
    setupCompactStatisticsSection();  // Compact stats (replaces Statistics tab)

    void updateStatistics(vtkPolyData* polyData,
                          const cvSelectionData* customSelection = nullptr);
    void updateSelectionList(vtkPolyData* polyData);
    void updateSpreadsheetData(
            vtkPolyData* polyData,
            const cvSelectionData* customSelection = nullptr);
    void computeBoundingBox(vtkPolyData* polyData, double bounds[6]);
    QString formatBounds(const double bounds[6]);
    void showColorDialog(const QString& title,
                         double currentColor[3],
                         int mode);
    void highlightSingleItem(qint64 id);
    qint64 extractIdFromItemText(const QString& itemText);
    void updateBookmarkCombo();
    void updateSelectionEditorTable();
    QString generateSelectionName();
    QColor generateSelectionColor() const;

    // Color synchronization helper
    void syncInternalColorArray(double r, double g, double b, int mode);

    // Helper to setup collapsible QGroupBox behavior
    void setupCollapsibleGroupBox(QGroupBox* groupBox);

    // Expression evaluation helpers (ParaView-style selection algebra)
    cvSelectionData evaluateExpression(const QString& expression);
    QStringList tokenizeExpression(const QString& expression);
    cvSelectionData parseOrExpression(const QStringList& tokens, int& pos);
    cvSelectionData parseXorExpression(const QStringList& tokens, int& pos);
    cvSelectionData parseAndExpression(const QStringList& tokens, int& pos);
    cvSelectionData parseUnaryExpression(const QStringList& tokens, int& pos);
    cvSelectionData parsePrimaryExpression(const QStringList& tokens, int& pos);

private:
    // Core components
    cvSelectionHighlighter* m_highlighter;
    cvTooltipFormatter* m_tooltipFormatter;
    cvViewSelectionManager* m_selectionManager;
    cvSelectionData m_selectionData;
    QVector<qint64>
            m_originalSelectionIds;  // Store original IDs for invert toggle

    // Label properties are now stored in cvSelectionHighlighter (single source
    // of truth) Local copies only used for dialog initialization before
    // highlighter is set

    // Saved selections for Selection Editor
    QList<SavedSelection> m_savedSelections;
    QString m_dataProducerName;

    // Main scroll area
    QScrollArea* m_scrollArea;
    QWidget* m_scrollContent;

    // === Selected Data Header ===
    cvExpanderButton* m_selectedDataExpander;
    QWidget* m_selectedDataContainer;
    QLabel* m_selectedDataLabel;
    QPushButton* m_freezeButton;
    QPushButton* m_extractButton;
    QPushButton* m_plotOverTimeButton;

    // === Selection Display Section ===
    cvExpanderButton* m_selectionDisplayExpander;
    QWidget* m_selectionDisplayContainer;
    QGroupBox* m_selectionDisplayGroup;  // Legacy - may be nullptr
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
    cvExpanderButton* m_selectionEditorExpander;
    QWidget* m_selectionEditorContainer;
    QGroupBox* m_selectionEditorGroup;  // Legacy - may be nullptr
    QLabel* m_dataProducerLabel;
    QLabel* m_dataProducerValue;
    QLabel* m_elementTypeLabel;
    QLabel* m_elementTypeValue;  // ParaView-style: shows element type in a
                                 // styled label
    QLabel* m_expressionLabel;
    QLineEdit* m_expressionEdit;
    QTableWidget* m_selectionEditorTable;  // Name, Type, Color columns
    QToolButton* m_addSelectionButton;
    QToolButton* m_removeSelectionButton;
    QToolButton* m_removeAllSelectionsButton;
    QPushButton* m_activateCombinedSelectionsButton;

    // === Create Selection Section (ParaView's Find Data) ===
    cvExpanderButton* m_createSelectionExpander;
    QWidget* m_createSelectionContainer;
    QGroupBox* m_createSelectionGroup;  // Legacy - may be nullptr
    QComboBox* m_dataProducerCombo;
    QComboBox* m_elementTypeCombo;
    QComboBox* m_attributeCombo;
    QComboBox* m_operatorCombo;
    QLineEdit* m_valueEdit;
    QToolButton* m_addQueryButton;
    QToolButton* m_removeQueryButton;
    QSpinBox* m_processIdSpinBox;
    QPushButton* m_findDataButton;
    QPushButton* m_resetButton;
    QPushButton* m_clearButton;
    QVBoxLayout* m_queriesLayout;

    // === Selected Data Spreadsheet ===
    cvExpanderButton* m_selectedDataSpreadsheetExpander;
    QWidget* m_selectedDataSpreadsheetContainer;
    QGroupBox* m_selectedDataGroup;  // Legacy - may be nullptr
    QComboBox* m_attributeTypeCombo;
    QToolButton* m_toggleColumnVisibilityButton;
    QToolButton* m_toggleFieldDataButton;  // ParaView-style: toggle field data
                                           // visibility
    QCheckBox* m_invertSelectionCheck;
    QTableWidget* m_spreadsheetTable;

    // === Compact Statistics Section (ParaView-style: no tabs) ===
    cvExpanderButton* m_compactStatsExpander;
    QWidget* m_compactStatsContainer;
    QGroupBox* m_compactStatsGroup;  // Legacy - may be nullptr
    QLabel* m_countLabel;
    QLabel* m_typeLabel;
    QLabel* m_boundsLabel;
    QLabel* m_centerLabel;
    QLabel* m_volumeLabel;

    // === Optional: Tab widget pointer (set to nullptr in new design) ===
    QTabWidget* m_tabWidget;

    // === Legacy UI elements (kept for backward compatibility, may be nullptr)
    // === These are initialized to nullptr and can be conditionally shown in
    // future menus
    QTableWidget* m_selectionTableWidget;  // Legacy selection list table
    QLabel* m_listInfoLabel;               // Legacy "Showing N items" label
    QComboBox* m_algebraOpCombo;           // Algebra operations combo
    QPushButton* m_applyAlgebraButton;     // Apply algebra operation
    QPushButton* m_extractBoundaryButton;  // Extract boundary
    QComboBox* m_filterTypeCombo;          // Filter type combo
    QPushButton* m_applyFilterButton;      // Apply filter
    QComboBox* m_bookmarkCombo;            // Bookmarks combo
    QPushButton* m_saveBookmarkButton;     // Save bookmark
    QPushButton* m_loadBookmarkButton;     // Load bookmark
    QPushButton* m_batchExportBookmarksButton;  // Batch export bookmarks
    QPushButton* m_addAnnotationButton;         // Add annotation

    // Legacy color/opacity controls (may be nullptr in simplified UI)
    QPushButton* m_hoverColorButton;
    QPushButton* m_preselectedColorButton;
    QPushButton* m_selectedColorButton;
    QPushButton* m_boundaryColorButton;
    QDoubleSpinBox* m_hoverOpacitySpin;
    QDoubleSpinBox* m_preselectedOpacitySpin;
    QDoubleSpinBox* m_selectedOpacitySpin;
    QDoubleSpinBox* m_boundaryOpacitySpin;

    // Colors are now stored in cvSelectionHighlighter (single source of truth)
    // These QColor helpers provide convenient access for UI updates
    QColor getSelectionColor() const;
    QColor getInteractiveSelectionColor() const;

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
