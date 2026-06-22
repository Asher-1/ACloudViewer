// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVGeom.h>

#include <QAbstractTableModel>
#include <QHeaderView>
#include <QList>
#include <QSet>
#include <QSortFilterProxyModel>
#include <QWidget>
#include <functional>

class QTableView;
class QLabel;
class QLineEdit;
class QPushButton;
class QComboBox;
class QSpinBox;
class QToolButton;
class QMenu;
class ccHObject;
class ccGenericMesh;
class ccGenericPointCloud;
class ccPointCloud;
class ccMesh;

// ParaView-aligned SpreadSheet data model with precision control
// and column visibility, modeled after pqSpreadSheetViewModel.
class ecvSpreadSheetModel : public QAbstractTableModel {
    Q_OBJECT

public:
    explicit ecvSpreadSheetModel(QObject* parent = nullptr);

    void setEntity(ccHObject* entity);
    ccHObject* entity() const { return m_entity; }

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index,
                  int role = Qt::DisplayRole) const override;
    QVariant headerData(int section,
                        Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const override;

    QString columnName(int col) const;
    QString columnGroupName(int visibleCol) const;
    int totalColumnCount() const { return m_columns.size(); }
    bool isColumnVisible(int col) const;
    void setColumnVisible(int col, bool visible);

    int decimalPrecision() const { return m_decimalPrecision; }
    void setDecimalPrecision(int p);

    bool fixedRepresentation() const { return m_fixedRepresentation; }
    void setFixedRepresentation(bool fixed);

    enum AttributeType { POINT_DATA = 0, CELL_DATA, FIELD_DATA };
    void setAttributeType(AttributeType type);
    AttributeType attributeType() const { return m_attributeType; }

    QString getRowsAsString() const;

    void setSelectedIndices(const QSet<unsigned>& indices);
    void setSelectionOnly(bool on);
    bool selectionOnly() const { return m_selectionOnly; }

    void setGenerateCellConnectivity(bool on);
    bool generateCellConnectivity() const { return m_cellConnectivity; }

private:
    void rebuildColumns();
    void rebuildFieldData();
    QString formatValue(double val) const;

    ccHObject* m_entity = nullptr;
    ccGenericPointCloud* m_cloud = nullptr;
    ccPointCloud* m_pcCloud = nullptr;
    ccMesh* m_mesh = nullptr;
    ccGenericMesh* m_genericMesh = nullptr;

    struct ColumnDef {
        QString name;
        enum Type {
            INDEX,
            X,
            Y,
            Z,
            R,
            G,
            B,
            NX,
            NY,
            NZ,
            NORMALS_MAG,
            POINTS_MAG,
            TCOORDS_S,
            TCOORDS_T,
            TCOORDS_MAG,
            SCALAR,
            CELL_INDEX,
            V1,
            V2,
            V3,
            MAT_INDEX,
            V1_X,
            V1_Y,
            V1_Z,
            V2_X,
            V2_Y,
            V2_Z,
            V3_X,
            V3_Y,
            V3_Z
        } type;
        int sfIndex = -1;
        bool visible = true;
    };
    QVector<ColumnDef> m_columns;

    struct FieldRow {
        QString key;
        QString value;
    };
    QVector<FieldRow> m_fieldRows;

    int m_decimalPrecision = 6;
    bool m_fixedRepresentation = false;
    bool m_selectionOnly = false;
    bool m_cellConnectivity = false;
    AttributeType m_attributeType = POINT_DATA;
    QSet<unsigned> m_selectedIndices;
    QVector<unsigned> m_sortedSelection;
    QVector<int> m_visibleColMap;
    QVector<float> m_computedNormals;

    void rebuildVisibleColMap();
    void rebuildSortedSelection();
    void computeVertexNormals();

    bool hasComputedNormal(unsigned row) const {
        return (row * 3 + 2) < static_cast<unsigned>(m_computedNormals.size());
    }
    CCVector3 getNormalFromMeshForVertex(unsigned vertIdx) const;
};

class QPainter;

class ecvSpreadSheetHeaderView : public QHeaderView {
    Q_OBJECT
public:
    explicit ecvSpreadSheetHeaderView(Qt::Orientation orientation,
                                      QWidget* parent = nullptr);
    void setModel(ecvSpreadSheetModel* model) { m_sourceModel = model; }

protected:
    QSize sizeHint() const override;
    void paintEvent(QPaintEvent* event) override;
    void paintSection(QPainter* painter,
                      const QRect& rect,
                      int logicalIndex) const override;

private:
    struct GroupInfo {
        QString name;
        int startSection = -1;
        int endSection = -1;
    };
    QVector<GroupInfo> collectGroups() const;

    ecvSpreadSheetModel* m_sourceModel = nullptr;
};

// ParaView-aligned SpreadSheet View with decorator toolbar.
// Modeled after pqSpreadSheetView + pqSpreadSheetViewDecorator.
class ecvSpreadSheetView : public QWidget {
    Q_OBJECT

public:
    explicit ecvSpreadSheetView(QWidget* parent = nullptr);
    ~ecvSpreadSheetView() override;

    QString title() const;
    ecvSpreadSheetModel* getModel() const { return m_model; }

    using EntityListProvider = std::function<QList<ccHObject*>()>;
    void setEntityListProvider(EntityListProvider provider);

public slots:
    void setEntity(ccHObject* entity);
    void setSelectedPointIndices(const QSet<unsigned>& indices);

signals:
    void tableSelectionChanged(ccHObject* entity,
                               const QVector<unsigned>& indices);

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void onEntitySelectionChanged(ccHObject* entity);
    void onSourceComboChanged(int index);
    void onSourceComboAboutToShow();
    void onSearchTextChanged(const QString& text);
    void onAttributeChanged(int index);
    void onPrecisionChanged(int precision);
    void onToggleFixed(bool fixed);
    void onColumnVisibilityMenuAboutToShow();
    void onToggleSelectionOnly(bool checked);
    void onTableSelectionChanged();
    void onCellFontSizeChanged(int size);
    void onHeaderFontSizeChanged(int size);
    void onToggleCellConnectivity(bool checked);
    void onStatColumnSelected(int logicalIndex);
    void onFilterColumnChanged(int index);
    void exportToCsv();
    void exportSelectedRows();
    void copyToClipboard();

private:
    void updateStatusBar();
    void updateColumnVisibility();
    void refreshSourceCombo();

    EntityListProvider m_entityListProvider;

    // Decorator toolbar widgets (ParaView pqSpreadSheetViewDecorator pattern)
    QLabel* m_showingLabel = nullptr;
    QComboBox* m_sourceCombo = nullptr;
    QComboBox* m_attributeCombo = nullptr;
    QSpinBox* m_precisionSpin = nullptr;
    QToolButton* m_fixedRepBtn = nullptr;
    QToolButton* m_columnVisBtn = nullptr;
    QToolButton* m_selectionOnlyBtn = nullptr;
    QToolButton* m_cellConnBtn = nullptr;
    QToolButton* m_fieldDataBtn = nullptr;
    int m_prevAttributeBeforeFieldData = 0;
    QSpinBox* m_cellFontSizeSpin = nullptr;
    QSpinBox* m_headerFontSizeSpin = nullptr;
    QToolButton* m_exportBtn = nullptr;
    QMenu* m_columnVisMenu = nullptr;

    QComboBox* m_filterColumnCombo = nullptr;
    QLineEdit* m_searchEdit = nullptr;
    QTableView* m_tableView = nullptr;
    QLabel* m_statusLabel = nullptr;
    QLabel* m_statsBar = nullptr;
    ecvSpreadSheetModel* m_model = nullptr;
    QSortFilterProxyModel* m_proxyModel = nullptr;

    QString m_viewTypeKey;
    QString m_title;
};
