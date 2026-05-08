// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvSpreadSheetView.h"

#include <ecvGenericPointCloud.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvViewManager.h>

#include <QApplication>
#include <QFileDialog>
#include <QFont>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableView>
#include <QTextStream>
#include <QVBoxLayout>

// ============================================================================
// Model
// ============================================================================

ecvSpreadSheetModel::ecvSpreadSheetModel(QObject* parent)
    : QAbstractTableModel(parent) {}

void ecvSpreadSheetModel::setEntity(ccHObject* entity) {
    beginResetModel();
    m_entity = entity;
    m_cloud = nullptr;
    m_pcCloud = nullptr;
    m_columns.clear();

    if (entity) {
        m_cloud = ccHObjectCaster::ToGenericPointCloud(entity);
        m_pcCloud = ccHObjectCaster::ToPointCloud(entity);
        if (m_cloud) rebuildColumns();
    }
    endResetModel();
}

void ecvSpreadSheetModel::rebuildColumns() {
    if (!m_cloud) return;

    m_columns.append({QStringLiteral("Index"), ColumnDef::INDEX, -1});
    m_columns.append({QStringLiteral("X"), ColumnDef::X, -1});
    m_columns.append({QStringLiteral("Y"), ColumnDef::Y, -1});
    m_columns.append({QStringLiteral("Z"), ColumnDef::Z, -1});

    if (m_cloud->hasColors()) {
        m_columns.append({QStringLiteral("R"), ColumnDef::R, -1});
        m_columns.append({QStringLiteral("G"), ColumnDef::G, -1});
        m_columns.append({QStringLiteral("B"), ColumnDef::B, -1});
    }

    if (m_cloud->hasNormals()) {
        m_columns.append({QStringLiteral("Nx"), ColumnDef::NX, -1});
        m_columns.append({QStringLiteral("Ny"), ColumnDef::NY, -1});
        m_columns.append({QStringLiteral("Nz"), ColumnDef::NZ, -1});
    }

    if (m_pcCloud) {
        unsigned sfCount = m_pcCloud->getNumberOfScalarFields();
        for (unsigned i = 0; i < sfCount; ++i) {
            const char* name =
                    m_pcCloud->getScalarFieldName(static_cast<int>(i));
            m_columns.append({QString::fromUtf8(name), ColumnDef::SCALAR,
                              static_cast<int>(i)});
        }
    }
}

int ecvSpreadSheetModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid() || !m_cloud) return 0;
    return static_cast<int>(m_cloud->size());
}

int ecvSpreadSheetModel::columnCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    return m_columns.size();
}

QVariant ecvSpreadSheetModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || !m_cloud) return {};
    if (role != Qt::DisplayRole) return {};

    unsigned row = static_cast<unsigned>(index.row());
    if (row >= m_cloud->size()) return {};

    const ColumnDef& col = m_columns[index.column()];
    switch (col.type) {
        case ColumnDef::INDEX:
            return row;
        case ColumnDef::X:
        case ColumnDef::Y:
        case ColumnDef::Z: {
            const CCVector3* P = m_cloud->getPoint(row);
            if (!P) return {};
            if (col.type == ColumnDef::X) return static_cast<double>(P->x);
            if (col.type == ColumnDef::Y) return static_cast<double>(P->y);
            return static_cast<double>(P->z);
        }
        case ColumnDef::R:
        case ColumnDef::G:
        case ColumnDef::B: {
            const ecvColor::Rgb& c = m_cloud->getPointColor(row);
            if (col.type == ColumnDef::R) return c.r;
            if (col.type == ColumnDef::G) return c.g;
            return c.b;
        }
        case ColumnDef::NX:
        case ColumnDef::NY:
        case ColumnDef::NZ: {
            const CCVector3& n = m_cloud->getPointNormal(row);
            if (col.type == ColumnDef::NX) return static_cast<double>(n.x);
            if (col.type == ColumnDef::NY) return static_cast<double>(n.y);
            return static_cast<double>(n.z);
        }
        case ColumnDef::SCALAR: {
            if (!m_pcCloud) return {};
            auto* sf = m_pcCloud->getScalarField(col.sfIndex);
            if (!sf || row >= sf->size()) return {};
            return static_cast<double>(sf->getValue(row));
        }
    }
    return {};
}

QVariant ecvSpreadSheetModel::headerData(int section,
                                         Qt::Orientation orientation,
                                         int role) const {
    if (role != Qt::DisplayRole) return {};
    if (orientation == Qt::Horizontal) {
        if (section >= 0 && section < m_columns.size())
            return m_columns[section].name;
    } else {
        return section;
    }
    return {};
}

QString ecvSpreadSheetModel::columnName(int col) const {
    if (col >= 0 && col < m_columns.size()) return m_columns[col].name;
    return {};
}

// ============================================================================
// View Widget
// ============================================================================

ecvSpreadSheetView::ecvSpreadSheetView(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    m_titleLabel = new QLabel(tr("SpreadSheet View - No data"), this);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setStyleSheet(
            "QLabel { background: #2d2d2d; color: #ccc; padding: 4px; "
            "font-weight: bold; }");
    layout->addWidget(m_titleLabel);

    auto* toolbar = new QWidget(this);
    auto* tbLayout = new QHBoxLayout(toolbar);
    tbLayout->setContentsMargins(4, 2, 4, 2);
    tbLayout->setSpacing(4);
    toolbar->setStyleSheet("QWidget { background: #333; }");

    m_searchEdit = new QLineEdit(toolbar);
    m_searchEdit->setPlaceholderText(tr("Filter rows..."));
    m_searchEdit->setClearButtonEnabled(true);
    m_searchEdit->setStyleSheet(
            "QLineEdit { background: #1e1e1e; color: #ddd; border: 1px solid "
            "#555; border-radius: 3px; padding: 2px 4px; }");
    tbLayout->addWidget(m_searchEdit, 1);

    m_exportBtn = new QPushButton(tr("Export CSV"), toolbar);
    m_exportBtn->setStyleSheet(
            "QPushButton { background: #3a5f8f; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background: #4a7fbf; }"
            "QPushButton:disabled { background: #2d2d2d; color: #666; }");
    m_exportBtn->setEnabled(false);
    tbLayout->addWidget(m_exportBtn);

    layout->addWidget(toolbar);

    m_model = new ecvSpreadSheetModel(this);

    m_proxyModel = new QSortFilterProxyModel(this);
    m_proxyModel->setSourceModel(m_model);
    m_proxyModel->setFilterCaseSensitivity(Qt::CaseInsensitive);
    m_proxyModel->setFilterKeyColumn(-1);

    m_tableView = new QTableView(this);
    m_tableView->setModel(m_proxyModel);
    m_tableView->setSortingEnabled(true);
    m_tableView->setAlternatingRowColors(true);
    m_tableView->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tableView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_tableView->horizontalHeader()->setStretchLastSection(true);
    m_tableView->horizontalHeader()->setSectionsClickable(true);
    m_tableView->verticalHeader()->setDefaultSectionSize(20);
    m_tableView->setStyleSheet(
            "QTableView { gridline-color: #444; background: #1e1e1e; color: "
            "#ddd; }"
            "QTableView::item:alternate { background: #252525; }"
            "QTableView::item:selected { background: #264f78; }"
            "QHeaderView::section { background: #2d2d2d; color: #ccc; "
            "padding: 3px; border: 1px solid #444; }");
    layout->addWidget(m_tableView, 1);

    m_statusLabel = new QLabel(this);
    m_statusLabel->setStyleSheet(
            "QLabel { background: #2d2d2d; color: #999; padding: 2px 6px; "
            "font-size: 11px; }");
    layout->addWidget(m_statusLabel);

    connect(m_searchEdit, &QLineEdit::textChanged, this,
            &ecvSpreadSheetView::onSearchTextChanged);
    connect(m_exportBtn, &QPushButton::clicked, this,
            &ecvSpreadSheetView::exportToCsv);
    connect(&ecvViewManager::instance(),
            &ecvViewManager::entitySelectionChanged, this,
            &ecvSpreadSheetView::onEntitySelectionChanged);

    updateStatusBar();
}

ecvSpreadSheetView::~ecvSpreadSheetView() = default;

QString ecvSpreadSheetView::title() const {
    return tr("SpreadSheet View");
}

void ecvSpreadSheetView::setEntity(ccHObject* entity) {
    m_model->setEntity(entity);
    m_exportBtn->setEnabled(entity != nullptr);

    if (entity) {
        auto* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
        if (cloud) {
            m_titleLabel->setText(
                    tr("SpreadSheet View - %1 (%2 points)")
                            .arg(entity->getName())
                            .arg(cloud->size()));
        } else {
            m_titleLabel->setText(
                    tr("SpreadSheet View - %1 (unsupported type)")
                            .arg(entity->getName()));
        }
    } else {
        m_titleLabel->setText(tr("SpreadSheet View - No data"));
    }

    updateStatusBar();
}

void ecvSpreadSheetView::onEntitySelectionChanged(ccHObject* entity) {
    setEntity(entity);
}

void ecvSpreadSheetView::onSearchTextChanged(const QString& text) {
    m_proxyModel->setFilterFixedString(text);
    updateStatusBar();
}

void ecvSpreadSheetView::exportToCsv() {
    if (!m_model->entity()) return;

    QString path = QFileDialog::getSaveFileName(
            this, tr("Export SpreadSheet to CSV"), QString(), tr("CSV (*.csv)"));
    if (path.isEmpty()) return;

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    QTextStream out(&file);

    int cols = m_model->columnCount();
    for (int c = 0; c < cols; ++c) {
        if (c > 0) out << ",";
        out << m_model->columnName(c);
    }
    out << "\n";

    int rows = m_proxyModel->rowCount();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c > 0) out << ",";
            QVariant val =
                    m_proxyModel->data(m_proxyModel->index(r, c));
            out << val.toString();
        }
        out << "\n";
    }
}

void ecvSpreadSheetView::updateStatusBar() {
    int totalRows = m_model->rowCount();
    int filteredRows = m_proxyModel->rowCount();
    int cols = m_model->columnCount();

    if (totalRows == 0) {
        m_statusLabel->setText(tr("No data loaded"));
    } else if (filteredRows < totalRows) {
        m_statusLabel->setText(
                tr("Showing %1 of %2 rows, %3 columns")
                        .arg(filteredRows)
                        .arg(totalRows)
                        .arg(cols));
    } else {
        m_statusLabel->setText(
                tr("%1 rows, %2 columns").arg(totalRows).arg(cols));
    }
}
