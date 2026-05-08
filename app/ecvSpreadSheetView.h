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
class ccHObject;
class ccGenericPointCloud;
class ccPointCloud;

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

private:
    void rebuildColumns();

    ccHObject* m_entity = nullptr;
    ccGenericPointCloud* m_cloud = nullptr;
    ccPointCloud* m_pcCloud = nullptr;

    struct ColumnDef {
        QString name;
        enum Type { INDEX, X, Y, Z, R, G, B, NX, NY, NZ, SCALAR } type;
        int sfIndex = -1;
    };
    QVector<ColumnDef> m_columns;
};

class ecvSpreadSheetView : public QWidget {
    Q_OBJECT

public:
    explicit ecvSpreadSheetView(QWidget* parent = nullptr);
    ~ecvSpreadSheetView() override;

    QString title() const;

public slots:
    void setEntity(ccHObject* entity);

private slots:
    void onEntitySelectionChanged(ccHObject* entity);
    void onSearchTextChanged(const QString& text);
    void exportToCsv();

private:
    void updateStatusBar();

    QLabel* m_titleLabel = nullptr;
    QLineEdit* m_searchEdit = nullptr;
    QPushButton* m_exportBtn = nullptr;
    QTableView* m_tableView = nullptr;
    QLabel* m_statusLabel = nullptr;
    ecvSpreadSheetModel* m_model = nullptr;
    QSortFilterProxyModel* m_proxyModel = nullptr;
};
