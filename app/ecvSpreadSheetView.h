// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QAbstractTableModel>
#include <QSortFilterProxyModel>
#include <QWidget>

class QTableView;
class QHeaderView;
class QLabel;
class QLineEdit;
class QPushButton;
class QComboBox;
class QSpinBox;
class QToolButton;
class QMenu;
class ccHObject;
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
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const override;

    QString columnName(int col) const;
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

private:
    void rebuildColumns();
    QString formatValue(double val) const;

    ccHObject* m_entity = nullptr;
    ccGenericPointCloud* m_cloud = nullptr;
    ccPointCloud* m_pcCloud = nullptr;
    ccMesh* m_mesh = nullptr;

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
            SCALAR,
            // Mesh/cell data columns
            CELL_INDEX,
            V1,
            V2,
            V3,
            MAT_INDEX
        } type;
        int sfIndex = -1;
        bool visible = true;
    };
    QVector<ColumnDef> m_columns;

    int m_decimalPrecision = 6;
    bool m_fixedRepresentation = false;
    AttributeType m_attributeType = POINT_DATA;
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

public slots:
    void setEntity(ccHObject* entity);

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void onEntitySelectionChanged(ccHObject* entity);
    void onSearchTextChanged(const QString& text);
    void onAttributeChanged(int index);
    void onPrecisionChanged(int precision);
    void onToggleFixed(bool fixed);
    void onColumnVisibilityMenuAboutToShow();
    void exportToCsv();
    void copyToClipboard();

private:
    void updateStatusBar();
    void updateColumnVisibility();

    // Decorator toolbar widgets (ParaView pqSpreadSheetViewDecorator pattern)
    QLabel* m_showingLabel = nullptr;
    QComboBox* m_sourceCombo = nullptr;
    QComboBox* m_attributeCombo = nullptr;
    QSpinBox* m_precisionSpin = nullptr;
    QToolButton* m_fixedRepBtn = nullptr;
    QToolButton* m_columnVisBtn = nullptr;
    QToolButton* m_exportBtn = nullptr;
    QMenu* m_columnVisMenu = nullptr;

    QLineEdit* m_searchEdit = nullptr;
    QTableView* m_tableView = nullptr;
    QLabel* m_statusLabel = nullptr;
    ecvSpreadSheetModel* m_model = nullptr;
    QSortFilterProxyModel* m_proxyModel = nullptr;
};
