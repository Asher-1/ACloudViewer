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
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvViewManager.h>

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QFileDialog>
#include <QFont>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableView>
#include <QTextStream>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidgetAction>

#include <cmath>

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
    m_mesh = nullptr;
    m_columns.clear();

    if (entity) {
        m_cloud = ccHObjectCaster::ToGenericPointCloud(entity);
        m_pcCloud = ccHObjectCaster::ToPointCloud(entity);
        m_mesh = ccHObjectCaster::ToMesh(entity);
        rebuildColumns();
    }
    endResetModel();
}

void ecvSpreadSheetModel::rebuildColumns() {
    if (m_attributeType == POINT_DATA) {
        if (!m_cloud) return;

        m_columns.append({QStringLiteral("Point ID"), ColumnDef::INDEX, -1, true});
        m_columns.append({QStringLiteral("Points_X"), ColumnDef::X, -1, true});
        m_columns.append({QStringLiteral("Points_Y"), ColumnDef::Y, -1, true});
        m_columns.append({QStringLiteral("Points_Z"), ColumnDef::Z, -1, true});

        if (m_cloud->hasNormals()) {
            m_columns.append({QStringLiteral("Normals_X"), ColumnDef::NX, -1, true});
            m_columns.append({QStringLiteral("Normals_Y"), ColumnDef::NY, -1, true});
            m_columns.append({QStringLiteral("Normals_Z"), ColumnDef::NZ, -1, true});
        }

        if (m_pcCloud) {
            unsigned sfCount = m_pcCloud->getNumberOfScalarFields();
            for (unsigned i = 0; i < sfCount; ++i) {
                const char* name =
                        m_pcCloud->getScalarFieldName(static_cast<int>(i));
                m_columns.append({QString::fromUtf8(name), ColumnDef::SCALAR,
                                  static_cast<int>(i), true});
            }
        }

        if (m_cloud->hasColors()) {
            m_columns.append({QStringLiteral("RGB"), ColumnDef::R, -1, true});
        }
    } else if (m_attributeType == CELL_DATA && m_mesh) {
        unsigned triCount = m_mesh->size();
        Q_UNUSED(triCount);
        m_columns.append({QStringLiteral("Cell ID"), ColumnDef::CELL_INDEX, -1, true});
        m_columns.append({QStringLiteral("Vertex 1"), ColumnDef::V1, -1, true});
        m_columns.append({QStringLiteral("Vertex 2"), ColumnDef::V2, -1, true});
        m_columns.append({QStringLiteral("Vertex 3"), ColumnDef::V3, -1, true});
        if (m_mesh->hasMaterials()) {
            m_columns.append({QStringLiteral("Material"), ColumnDef::MAT_INDEX, -1, true});
        }
    }
}

int ecvSpreadSheetModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    if (m_attributeType == POINT_DATA && m_cloud)
        return static_cast<int>(m_cloud->size());
    if (m_attributeType == CELL_DATA && m_mesh)
        return static_cast<int>(m_mesh->size());
    return 0;
}

int ecvSpreadSheetModel::columnCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    int count = 0;
    for (const auto& col : m_columns) {
        if (col.visible) ++count;
    }
    return count;
}

bool ecvSpreadSheetModel::isColumnVisible(int col) const {
    if (col >= 0 && col < m_columns.size()) return m_columns[col].visible;
    return false;
}

void ecvSpreadSheetModel::setColumnVisible(int col, bool visible) {
    if (col >= 0 && col < m_columns.size() &&
        m_columns[col].visible != visible) {
        beginResetModel();
        m_columns[col].visible = visible;
        endResetModel();
    }
}

void ecvSpreadSheetModel::setDecimalPrecision(int p) {
    if (m_decimalPrecision != p) {
        m_decimalPrecision = p;
        emit dataChanged(index(0, 0),
                         index(rowCount() - 1, columnCount() - 1));
    }
}

void ecvSpreadSheetModel::setFixedRepresentation(bool fixed) {
    if (m_fixedRepresentation != fixed) {
        m_fixedRepresentation = fixed;
        emit dataChanged(index(0, 0),
                         index(rowCount() - 1, columnCount() - 1));
    }
}

void ecvSpreadSheetModel::setAttributeType(AttributeType type) {
    if (m_attributeType != type) {
        beginResetModel();
        m_attributeType = type;
        m_columns.clear();
        rebuildColumns();
        endResetModel();
    }
}

QString ecvSpreadSheetModel::formatValue(double val) const {
    if (std::isnan(val)) return QStringLiteral("nan");
    if (std::isinf(val)) return val > 0 ? QStringLiteral("inf") : QStringLiteral("-inf");
    if (m_fixedRepresentation) {
        return QString::number(val, 'f', m_decimalPrecision);
    }
    return QString::number(val, 'g', m_decimalPrecision);
}

QVariant ecvSpreadSheetModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) return {};
    if (role != Qt::DisplayRole) return {};

    unsigned row = static_cast<unsigned>(index.row());

    // Map visible column index to actual column
    int visCol = index.column();
    int actualCol = -1;
    int seen = 0;
    for (int i = 0; i < m_columns.size(); ++i) {
        if (m_columns[i].visible) {
            if (seen == visCol) {
                actualCol = i;
                break;
            }
            ++seen;
        }
    }
    if (actualCol < 0) return {};

    const ColumnDef& col = m_columns[actualCol];

    if (m_attributeType == POINT_DATA && m_cloud) {
        if (row >= m_cloud->size()) return {};
        switch (col.type) {
            case ColumnDef::INDEX:
                return row;
            case ColumnDef::X:
            case ColumnDef::Y:
            case ColumnDef::Z: {
                const CCVector3* P = m_cloud->getPoint(row);
                if (!P) return {};
                double v = (col.type == ColumnDef::X)   ? P->x
                           : (col.type == ColumnDef::Y) ? P->y
                                                        : P->z;
                return formatValue(v);
            }
            case ColumnDef::R: {
                const ecvColor::Rgb& c = m_cloud->getPointColor(row);
                return QString("%1").arg(c.r | (c.g << 8) | (c.b << 16));
            }
            case ColumnDef::G:
            case ColumnDef::B:
                return {};
            case ColumnDef::NX:
            case ColumnDef::NY:
            case ColumnDef::NZ: {
                const CCVector3& n = m_cloud->getPointNormal(row);
                double v = (col.type == ColumnDef::NX)   ? n.x
                           : (col.type == ColumnDef::NY) ? n.y
                                                         : n.z;
                return formatValue(v);
            }
            case ColumnDef::SCALAR: {
                if (!m_pcCloud) return {};
                auto* sf = m_pcCloud->getScalarField(col.sfIndex);
                if (!sf || row >= sf->size()) return {};
                return formatValue(static_cast<double>(sf->getValue(row)));
            }
            default:
                break;
        }
    } else if (m_attributeType == CELL_DATA && m_mesh) {
        if (row >= m_mesh->size()) return {};
        switch (col.type) {
            case ColumnDef::CELL_INDEX:
                return row;
            case ColumnDef::V1:
            case ColumnDef::V2:
            case ColumnDef::V3: {
                cloudViewer::VerticesIndexes* tri =
                        m_mesh->getTriangleVertIndexes(row);
                if (!tri) return {};
                if (col.type == ColumnDef::V1) return tri->i1;
                if (col.type == ColumnDef::V2) return tri->i2;
                return tri->i3;
            }
            case ColumnDef::MAT_INDEX:
                return static_cast<int>(m_mesh->getTriangleMtlIndex(row));
            default:
                break;
        }
    }

    return {};
}

QVariant ecvSpreadSheetModel::headerData(int section,
                                         Qt::Orientation orientation,
                                         int role) const {
    if (role != Qt::DisplayRole) return {};
    if (orientation == Qt::Horizontal) {
        int seen = 0;
        for (int i = 0; i < m_columns.size(); ++i) {
            if (m_columns[i].visible) {
                if (seen == section) return m_columns[i].name;
                ++seen;
            }
        }
    } else {
        return section;
    }
    return {};
}

QString ecvSpreadSheetModel::columnName(int col) const {
    if (col >= 0 && col < m_columns.size()) return m_columns[col].name;
    return {};
}

QString ecvSpreadSheetModel::getRowsAsString() const {
    int rows = rowCount();
    int cols = columnCount();
    if (rows == 0 || cols == 0) return {};

    QString result;
    QTextStream out(&result);

    for (int c = 0; c < cols; ++c) {
        if (c > 0) out << "\t";
        out << headerData(c, Qt::Horizontal, Qt::DisplayRole).toString();
    }
    out << "\n";

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c > 0) out << "\t";
            out << data(this->index(r, c)).toString();
        }
        out << "\n";
    }
    return result;
}

// ============================================================================
// View Widget (ParaView pqSpreadSheetView + pqSpreadSheetViewDecorator)
// ============================================================================

ecvSpreadSheetView::ecvSpreadSheetView(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // === Decorator toolbar (ParaView pqSpreadSheetViewDecorator pattern) ===
    auto* decoratorBar = new QWidget(this);
    auto* decLayout = new QHBoxLayout(decoratorBar);
    decLayout->setContentsMargins(4, 2, 4, 2);
    decLayout->setSpacing(2);

    const auto labelSS = QStringLiteral("QLabel { color: #ccc; }");
    const auto comboSS = QStringLiteral(
            "QComboBox { background: #1e1e1e; color: #ddd; border: 1px solid "
            "#555; border-radius: 3px; padding: 1px 4px; min-width: 60px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2d2d2d; color: #ddd; "
            "selection-background-color: #264f78; }");
    const auto spinSS = QStringLiteral(
            "QSpinBox { background: #1e1e1e; color: #ddd; border: 1px solid "
            "#555; border-radius: 3px; padding: 1px 2px; }");
    const auto toolBtnSS = QStringLiteral(
            "QToolButton { background: transparent; color: #ccc; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px 6px; }"
            "QToolButton:hover { background: #3a5f8f; }"
            "QToolButton:checked { background: #264f78; border-color: "
            "#4a7fbf; }");

    m_showingLabel = new QLabel(QStringLiteral("<b>Showing</b>"), decoratorBar);
    m_showingLabel->setStyleSheet(labelSS);
    decLayout->addWidget(m_showingLabel);

    m_sourceCombo = new QComboBox(decoratorBar);
    m_sourceCombo->setStyleSheet(comboSS);
    m_sourceCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    decLayout->addWidget(m_sourceCombo);

    auto* attrLabel =
            new QLabel(QStringLiteral("<b>Attribute:</b>"), decoratorBar);
    attrLabel->setStyleSheet(labelSS);
    decLayout->addWidget(attrLabel);

    m_attributeCombo = new QComboBox(decoratorBar);
    m_attributeCombo->setStyleSheet(comboSS);
    m_attributeCombo->addItem(tr("Point Data"), ecvSpreadSheetModel::POINT_DATA);
    m_attributeCombo->addItem(tr("Cell Data"), ecvSpreadSheetModel::CELL_DATA);
    m_attributeCombo->addItem(tr("Field Data"), ecvSpreadSheetModel::FIELD_DATA);
    decLayout->addWidget(m_attributeCombo);

    auto* precLabel = new QLabel(decoratorBar);
    precLabel->setText(QStringLiteral("<b>Precision:</b>"));
    precLabel->setStyleSheet(labelSS);
    decLayout->addWidget(precLabel);

    m_precisionSpin = new QSpinBox(decoratorBar);
    m_precisionSpin->setRange(1, 32);
    m_precisionSpin->setValue(6);
    m_precisionSpin->setAlignment(Qt::AlignRight);
    m_precisionSpin->setStyleSheet(spinSS);
    decLayout->addWidget(m_precisionSpin);

    m_fixedRepBtn = new QToolButton(decoratorBar);
    m_fixedRepBtn->setText(tr("Fixed"));
    m_fixedRepBtn->setToolTip(
            tr("Toggle fixed-point representation (always show precision "
               "digits)"));
    m_fixedRepBtn->setCheckable(true);
    m_fixedRepBtn->setStyleSheet(toolBtnSS);
    decLayout->addWidget(m_fixedRepBtn);

    m_columnVisMenu = new QMenu(this);
    m_columnVisBtn = new QToolButton(decoratorBar);
    m_columnVisBtn->setText(tr("Columns"));
    m_columnVisBtn->setToolTip(tr("Toggle column visibility"));
    m_columnVisBtn->setPopupMode(QToolButton::InstantPopup);
    m_columnVisBtn->setMenu(m_columnVisMenu);
    m_columnVisBtn->setStyleSheet(toolBtnSS);
    decLayout->addWidget(m_columnVisBtn);

    m_exportBtn = new QToolButton(decoratorBar);
    m_exportBtn->setText(tr("Export"));
    m_exportBtn->setToolTip(tr("Export spreadsheet to CSV"));
    m_exportBtn->setStyleSheet(toolBtnSS);
    decLayout->addWidget(m_exportBtn);

    decLayout->addStretch(1);
    decoratorBar->setStyleSheet(
            QStringLiteral("QWidget#DecoratorBar { background: #2d2d2d; }"));
    decoratorBar->setObjectName("DecoratorBar");
    layout->addWidget(decoratorBar);

    // === Search bar ===
    auto* searchBar = new QWidget(this);
    auto* searchLayout = new QHBoxLayout(searchBar);
    searchLayout->setContentsMargins(4, 2, 4, 2);
    searchLayout->setSpacing(4);
    searchBar->setStyleSheet("QWidget { background: #333; }");

    m_searchEdit = new QLineEdit(searchBar);
    m_searchEdit->setPlaceholderText(tr("Filter rows..."));
    m_searchEdit->setClearButtonEnabled(true);
    m_searchEdit->setStyleSheet(
            "QLineEdit { background: #1e1e1e; color: #ddd; border: 1px solid "
            "#555; border-radius: 3px; padding: 2px 4px; }");
    searchLayout->addWidget(m_searchEdit, 1);
    layout->addWidget(searchBar);

    // === Table view ===
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
    m_tableView->horizontalHeader()->setHighlightSections(false);
    m_tableView->verticalHeader()->setDefaultSectionSize(20);
    m_tableView->setStyleSheet(
            "QTableView { gridline-color: #444; background: #1e1e1e; color: "
            "#ddd; }"
            "QTableView::item:alternate { background: #252525; }"
            "QTableView::item:selected { background: #264f78; }"
            "QHeaderView::section { background: #2d2d2d; color: #ccc; "
            "padding: 3px; border: 1px solid #444; }");
    layout->addWidget(m_tableView, 1);

    // === Status bar ===
    m_statusLabel = new QLabel(this);
    m_statusLabel->setStyleSheet(
            "QLabel { background: #2d2d2d; color: #999; padding: 2px 6px; "
            "font-size: 11px; }");
    layout->addWidget(m_statusLabel);

    // === Connections ===
    connect(m_searchEdit, &QLineEdit::textChanged, this,
            &ecvSpreadSheetView::onSearchTextChanged);
    connect(m_exportBtn, &QToolButton::clicked, this,
            &ecvSpreadSheetView::exportToCsv);
    connect(m_attributeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ecvSpreadSheetView::onAttributeChanged);
    connect(m_precisionSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &ecvSpreadSheetView::onPrecisionChanged);
    connect(m_fixedRepBtn, &QToolButton::toggled, this,
            &ecvSpreadSheetView::onToggleFixed);
    connect(m_columnVisMenu, &QMenu::aboutToShow, this,
            &ecvSpreadSheetView::onColumnVisibilityMenuAboutToShow);
    connect(&ecvViewManager::instance(),
            &ecvViewManager::entitySelectionChanged, this,
            &ecvSpreadSheetView::onEntitySelectionChanged);

    installEventFilter(this);

    updateStatusBar();
}

ecvSpreadSheetView::~ecvSpreadSheetView() = default;

QString ecvSpreadSheetView::title() const {
    return tr("SpreadSheet View");
}

void ecvSpreadSheetView::setEntity(ccHObject* entity) {
    m_model->setEntity(entity);

    m_sourceCombo->blockSignals(true);
    m_sourceCombo->clear();
    if (entity) {
        m_sourceCombo->addItem(entity->getName());
        auto* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
        auto* mesh = ccHObjectCaster::ToMesh(entity);

        bool hasCellData = (mesh != nullptr);
        auto* itemModel = qobject_cast<QStandardItemModel*>(
                m_attributeCombo->model());
        if (itemModel) {
            auto* item = itemModel->item(1);
            if (item) {
                item->setEnabled(hasCellData);
            }
        }
        if (!hasCellData && m_attributeCombo->currentIndex() == 1) {
            m_attributeCombo->setCurrentIndex(0);
        }

        unsigned pointCount = cloud ? cloud->size() : 0;
        m_sourceCombo->setToolTip(
                tr("%1 (%2 points)")
                        .arg(entity->getName())
                        .arg(pointCount));
    } else {
        m_sourceCombo->addItem(tr("None"));
    }
    m_sourceCombo->blockSignals(false);

    updateStatusBar();
}

void ecvSpreadSheetView::onEntitySelectionChanged(ccHObject* entity) {
    setEntity(entity);
}

void ecvSpreadSheetView::onSearchTextChanged(const QString& text) {
    m_proxyModel->setFilterFixedString(text);
    updateStatusBar();
}

void ecvSpreadSheetView::onAttributeChanged(int index) {
    auto type = static_cast<ecvSpreadSheetModel::AttributeType>(
            m_attributeCombo->itemData(index).toInt());
    m_model->setAttributeType(type);
    updateStatusBar();
}

void ecvSpreadSheetView::onPrecisionChanged(int precision) {
    m_model->setDecimalPrecision(precision);
}

void ecvSpreadSheetView::onToggleFixed(bool fixed) {
    m_model->setFixedRepresentation(fixed);
}

void ecvSpreadSheetView::onColumnVisibilityMenuAboutToShow() {
    m_columnVisMenu->clear();

    int totalCols = m_model->totalColumnCount();
    for (int i = 0; i < totalCols; ++i) {
        QString name = m_model->columnName(i);
        auto* action = m_columnVisMenu->addAction(name);
        action->setCheckable(true);
        action->setChecked(m_model->isColumnVisible(i));
        connect(action, &QAction::toggled, this,
                [this, i](bool checked) {
                    m_model->setColumnVisible(i, checked);
                    updateStatusBar();
                });
    }
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
        out << m_model->headerData(c, Qt::Horizontal, Qt::DisplayRole)
                       .toString();
    }
    out << "\n";

    int rows = m_proxyModel->rowCount();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c > 0) out << ",";
            QVariant val = m_proxyModel->data(m_proxyModel->index(r, c));
            out << val.toString();
        }
        out << "\n";
    }
}

void ecvSpreadSheetView::copyToClipboard() {
    auto table = m_model->getRowsAsString();
    if (!table.isEmpty()) {
        QApplication::clipboard()->setText(table);
    }
}

bool ecvSpreadSheetView::eventFilter(QObject* obj, QEvent* event) {
    if (event->type() == QEvent::KeyPress) {
        auto* kev = static_cast<QKeyEvent*>(event);
        if (kev->matches(QKeySequence::Copy)) {
            copyToClipboard();
            return true;
        }
    }
    return QWidget::eventFilter(obj, event);
}

void ecvSpreadSheetView::updateStatusBar() {
    int totalRows = m_model->rowCount();
    int filteredRows = m_proxyModel->rowCount();
    int cols = m_model->columnCount();

    QString attrName;
    if (m_attributeCombo) {
        attrName = m_attributeCombo->currentText();
    }

    if (totalRows == 0) {
        m_statusLabel->setText(tr("No data loaded"));
    } else if (filteredRows < totalRows) {
        m_statusLabel->setText(
                tr("Showing %1 of %2 rows, %3 columns (%4)")
                        .arg(filteredRows)
                        .arg(totalRows)
                        .arg(cols)
                        .arg(attrName));
    } else {
        m_statusLabel->setText(tr("%1 rows, %2 columns (%3)")
                                       .arg(totalRows)
                                       .arg(cols)
                                       .arg(attrName));
    }
}
