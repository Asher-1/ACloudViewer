// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvSpreadSheetView.h"

#include <ecvAdvancedTypes.h>
#include <ecvGenericMesh.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>
#include <ecvViewTitleRegistry.h>

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QColor>
#include <QComboBox>
#include <QFileDialog>
#include <QFont>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPaintEvent>
#include <QPainter>
#include <QPushButton>
#include <QShortcut>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableView>
#include <QTextStream>
#include <QTimer>
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
    m_genericMesh = nullptr;
    m_columns.clear();
    m_fieldRows.clear();
    m_computedNormals.clear();

    if (entity) {
        m_cloud = ccHObjectCaster::ToGenericPointCloud(entity);
        m_pcCloud = ccHObjectCaster::ToPointCloud(entity);
        m_genericMesh = ccHObjectCaster::ToGenericMesh(entity);
        m_mesh = ccHObjectCaster::ToMesh(entity);
        if (!m_mesh && m_genericMesh)
            m_mesh = ccHObjectCaster::ToMesh(m_genericMesh);

        if (m_genericMesh && !m_cloud) {
            m_cloud = ccHObjectCaster::ToGenericPointCloud(
                    m_genericMesh->getAssociatedCloud());
        }
        if (m_genericMesh && !m_pcCloud) {
            m_pcCloud = ccHObjectCaster::ToPointCloud(
                    m_genericMesh->getAssociatedCloud());
        }
        if (!m_cloud && !m_pcCloud) {
            auto* gc = ccHObjectCaster::ToGenericPointCloud(entity);
            if (gc) {
                m_cloud = gc;
                m_pcCloud = ccHObjectCaster::ToPointCloud(gc);
            }
        }
        for (unsigned i = 0;
             (!m_genericMesh || !m_cloud) && i < entity->getChildrenNumber();
             ++i) {
            auto* child = entity->getChild(i);
            if (!m_genericMesh) {
                m_genericMesh = ccHObjectCaster::ToGenericMesh(child);
                if (!m_mesh) m_mesh = ccHObjectCaster::ToMesh(child);
            }
            if (!m_cloud) {
                m_cloud = ccHObjectCaster::ToGenericPointCloud(child);
                if (!m_pcCloud)
                    m_pcCloud = ccHObjectCaster::ToPointCloud(child);
            }
        }
        if (m_genericMesh && !m_cloud) {
            m_cloud = ccHObjectCaster::ToGenericPointCloud(
                    m_genericMesh->getAssociatedCloud());
        }
        if (m_genericMesh && !m_pcCloud) {
            m_pcCloud = ccHObjectCaster::ToPointCloud(
                    m_genericMesh->getAssociatedCloud());
        }
        if (!m_mesh && m_genericMesh)
            m_mesh = ccHObjectCaster::ToMesh(m_genericMesh);
        computeVertexNormals();
        rebuildColumns();
    }
    endResetModel();
}

CCVector3 ecvSpreadSheetModel::getNormalFromMeshForVertex(
        unsigned vertIdx) const {
    ccGenericMesh* meshSrc = m_genericMesh ? m_genericMesh : m_mesh;
    if (!meshSrc || !meshSrc->hasTriNormals()) return CCVector3(0, 0, 0);
    CCVector3 accum(0, 0, 0);
    int count = 0;
    unsigned numTris = meshSrc->size();
    for (unsigned t = 0; t < numTris; ++t) {
        auto* tri = meshSrc->getTriangleVertIndexes(t);
        if (!tri) continue;
        if (tri->i1 != vertIdx && tri->i2 != vertIdx && tri->i3 != vertIdx)
            continue;
        CCVector3 na, nb, nc;
        if (meshSrc->getTriangleNormals(t, na, nb, nc)) {
            CCVector3 vn = (tri->i1 == vertIdx)   ? na
                           : (tri->i2 == vertIdx) ? nb
                                                  : nc;
            accum += vn;
            ++count;
        }
    }
    if (count > 0) {
        accum /= static_cast<float>(count);
        float len = accum.norm();
        if (len > 1e-12f) accum /= len;
    }
    return accum;
}

void ecvSpreadSheetModel::computeVertexNormals() {
    m_computedNormals.clear();
    ccGenericMesh* meshSrc = m_genericMesh ? m_genericMesh : m_mesh;
    if (!meshSrc || !m_cloud) return;

    bool cloudHasNormals = (m_pcCloud && m_pcCloud->hasNormals()) ||
                           (m_cloud && m_cloud->hasNormals());
    if (cloudHasNormals) return;

    unsigned numVerts = m_cloud->size();
    unsigned numTris = meshSrc->size();
    if (numTris == 0) return;

    m_computedNormals.resize(numVerts * 3);
    m_computedNormals.fill(0.0f);
    QVector<int> counts(numVerts, 0);

    bool hasExplicitTriNormals = false;
    {
        auto* cm = ccHObjectCaster::ToMesh(meshSrc);
        if (cm) {
            hasExplicitTriNormals = cm->hasTriNormals();
        } else {
            hasExplicitTriNormals = meshSrc->hasTriNormals();
        }
    }

    for (unsigned t = 0; t < numTris; ++t) {
        auto* tri = meshSrc->getTriangleVertIndexes(t);
        if (!tri) continue;

        CCVector3 faceNormal(0, 0, 0);
        if (hasExplicitTriNormals) {
            CCVector3 na, nb, nc;
            if (meshSrc->getTriangleNormals(t, na, nb, nc)) {
                faceNormal = (na + nb + nc);
                float len = faceNormal.norm();
                if (len > 1e-12f) faceNormal /= len;
            }
        }
        if (faceNormal.norm2() < 1e-12f) {
            const CCVector3* A = m_cloud->getPoint(tri->i1);
            const CCVector3* B = m_cloud->getPoint(tri->i2);
            const CCVector3* C = m_cloud->getPoint(tri->i3);
            if (A && B && C) {
                CCVector3 AB = *B - *A;
                CCVector3 AC = *C - *A;
                faceNormal = AB.cross(AC);
                float len = faceNormal.norm();
                if (len > 1e-12f) faceNormal /= len;
            }
        }

        auto accum = [&](unsigned vi) {
            if (vi < numVerts) {
                unsigned b = vi * 3;
                m_computedNormals[b] += faceNormal.x;
                m_computedNormals[b + 1] += faceNormal.y;
                m_computedNormals[b + 2] += faceNormal.z;
                counts[vi]++;
            }
        };
        accum(tri->i1);
        accum(tri->i2);
        accum(tri->i3);
    }

    for (unsigned i = 0; i < numVerts; ++i) {
        if (counts[i] > 0) {
            unsigned b = i * 3;
            float inv = 1.0f / counts[i];
            float nx = m_computedNormals[b] * inv;
            float ny = m_computedNormals[b + 1] * inv;
            float nz = m_computedNormals[b + 2] * inv;
            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-12f) {
                float invL = 1.0f / len;
                nx *= invL;
                ny *= invL;
                nz *= invL;
            }
            m_computedNormals[b] = nx;
            m_computedNormals[b + 1] = ny;
            m_computedNormals[b + 2] = nz;
        }
    }
}

void ecvSpreadSheetModel::rebuildColumns() {
    if (m_attributeType == POINT_DATA) {
        if (!m_cloud) return;

        m_columns.append(
                {QStringLiteral("Point ID"), ColumnDef::INDEX, -1, true});

        bool hasNorms = m_cloud->hasNormals();
        if (!hasNorms && m_pcCloud) hasNorms = m_pcCloud->hasNormals();
        if (!hasNorms && m_mesh) {
            hasNorms = m_mesh->hasNormals() || m_mesh->hasTriNormals();
        }
        if (!hasNorms && m_genericMesh) {
            auto* cm = ccHObjectCaster::ToMesh(m_genericMesh);
            if (cm) {
                hasNorms = cm->hasNormals() || cm->hasTriNormals();
            } else {
                hasNorms = m_genericMesh->hasNormals() ||
                           m_genericMesh->hasTriNormals();
            }
        }
        if (!hasNorms && !m_computedNormals.isEmpty()) hasNorms = true;

        if (hasNorms) {
            m_columns.append(
                    {QStringLiteral("Normals"), ColumnDef::NX, -1, true});
            m_columns.append({QString(), ColumnDef::NY, -1, true});
            m_columns.append({QString(), ColumnDef::NZ, -1, true});
            m_columns.append({QStringLiteral("Normals_Magnitude"),
                              ColumnDef::NORMALS_MAG, -1, true});
        }

        m_columns.append({QStringLiteral("Points"), ColumnDef::X, -1, true});
        m_columns.append({QString(), ColumnDef::Y, -1, true});
        m_columns.append({QString(), ColumnDef::Z, -1, true});
        m_columns.append({QStringLiteral("Points_Magnitude"),
                          ColumnDef::POINTS_MAG, -1, true});

        if (m_cloud->hasColors()) {
            m_columns.append(
                    {QStringLiteral("Colors"), ColumnDef::R, -1, true});
            m_columns.append({QString(), ColumnDef::G, -1, true});
            m_columns.append({QString(), ColumnDef::B, -1, true});
        }

        if (m_mesh) {
            auto* texTable = m_mesh->getTexCoordinatesTable();
            if (texTable && texTable->size() > 0) {
                m_columns.append({QStringLiteral("TCoords"),
                                  ColumnDef::TCOORDS_S, -1, true});
                m_columns.append({QString(), ColumnDef::TCOORDS_T, -1, true});
                m_columns.append({QStringLiteral("TCoords_Magnitude"),
                                  ColumnDef::TCOORDS_MAG, -1, true});
            }
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
    } else if (m_attributeType == CELL_DATA && m_mesh) {
        m_columns.append(
                {QStringLiteral("Cell ID"), ColumnDef::CELL_INDEX, -1, true});
        m_columns.append({QStringLiteral("Vertex 1"), ColumnDef::V1, -1, true});
        m_columns.append({QStringLiteral("Vertex 2"), ColumnDef::V2, -1, true});
        m_columns.append({QStringLiteral("Vertex 3"), ColumnDef::V3, -1, true});
        if (m_cellConnectivity) {
            m_columns.append(
                    {QStringLiteral("V1_X"), ColumnDef::V1_X, -1, true});
            m_columns.append(
                    {QStringLiteral("V1_Y"), ColumnDef::V1_Y, -1, true});
            m_columns.append(
                    {QStringLiteral("V1_Z"), ColumnDef::V1_Z, -1, true});
            m_columns.append(
                    {QStringLiteral("V2_X"), ColumnDef::V2_X, -1, true});
            m_columns.append(
                    {QStringLiteral("V2_Y"), ColumnDef::V2_Y, -1, true});
            m_columns.append(
                    {QStringLiteral("V2_Z"), ColumnDef::V2_Z, -1, true});
            m_columns.append(
                    {QStringLiteral("V3_X"), ColumnDef::V3_X, -1, true});
            m_columns.append(
                    {QStringLiteral("V3_Y"), ColumnDef::V3_Y, -1, true});
            m_columns.append(
                    {QStringLiteral("V3_Z"), ColumnDef::V3_Z, -1, true});
        }
        if (m_mesh->hasMaterials()) {
            m_columns.append({QStringLiteral("Material"), ColumnDef::MAT_INDEX,
                              -1, true});
        }
    } else if (m_attributeType == FIELD_DATA) {
        rebuildFieldData();
    }
    rebuildVisibleColMap();
}

void ecvSpreadSheetModel::rebuildFieldData() {
    m_fieldRows.clear();

    if (!m_entity) return;

    auto addRow = [this](const QString& k, const QString& v) {
        FieldRow r;
        r.key = k;
        r.value = v;
        m_fieldRows.append(r);
    };

    addRow(tr("Name"), m_entity->getName());
    QString typeName = tr("Object");
    if (m_mesh)
        typeName = tr("Mesh");
    else if (m_pcCloud)
        typeName = tr("Point Cloud");
    else if (m_cloud)
        typeName = tr("Generic Point Cloud");
    addRow(tr("Type"), typeName);
    addRow(tr("Enabled"), m_entity->isEnabled() ? tr("Yes") : tr("No"));

    if (m_cloud) {
        addRow(tr("Number of Points"), QString::number(m_cloud->size()));

        CCVector3 bbMin, bbMax;
        m_cloud->getBoundingBox(bbMin, bbMax);
        addRow(tr("Bounds X"), QStringLiteral("[%1, %2]")
                                       .arg(formatValue(bbMin.x))
                                       .arg(formatValue(bbMax.x)));
        addRow(tr("Bounds Y"), QStringLiteral("[%1, %2]")
                                       .arg(formatValue(bbMin.y))
                                       .arg(formatValue(bbMax.y)));
        addRow(tr("Bounds Z"), QStringLiteral("[%1, %2]")
                                       .arg(formatValue(bbMin.z))
                                       .arg(formatValue(bbMax.z)));

        addRow(tr("Has Colors"), m_cloud->hasColors() ? tr("Yes") : tr("No"));
        bool anyNormals = m_cloud->hasNormals() ||
                          (m_genericMesh && m_genericMesh->hasNormals()) ||
                          (m_mesh && m_mesh->hasNormals()) ||
                          !m_computedNormals.isEmpty();
        addRow(tr("Has Normals"), anyNormals ? tr("Yes") : tr("No"));
    }

    if (m_pcCloud) {
        unsigned sfCount = m_pcCloud->getNumberOfScalarFields();
        addRow(tr("Scalar Fields"), QString::number(sfCount));

        for (unsigned i = 0; i < sfCount; ++i) {
            auto* sf = m_pcCloud->getScalarField(static_cast<int>(i));
            if (!sf) continue;
            sf->computeMinAndMax();
            QString sfName = QString::fromUtf8(sf->getName());
            addRow(tr("SF '%1' Range").arg(sfName),
                   QStringLiteral("[%1, %2]")
                           .arg(formatValue(sf->getMin()))
                           .arg(formatValue(sf->getMax())));
        }
    }

    if (m_mesh) {
        addRow(tr("Number of Triangles"), QString::number(m_mesh->size()));
        addRow(tr("Has Materials"),
               m_mesh->hasMaterials() ? tr("Yes") : tr("No"));
        addRow(tr("Has Per-Triangle Normals"),
               m_mesh->hasTriNormals() ? tr("Yes") : tr("No"));
    }
}

int ecvSpreadSheetModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    if (m_attributeType == FIELD_DATA) return m_fieldRows.size();
    if (m_selectionOnly && !m_selectedIndices.isEmpty()) {
        return m_selectedIndices.size();
    }
    if (m_attributeType == POINT_DATA && m_cloud)
        return static_cast<int>(m_cloud->size());
    if (m_attributeType == CELL_DATA && m_mesh)
        return static_cast<int>(m_mesh->size());
    return 0;
}

int ecvSpreadSheetModel::columnCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    if (m_attributeType == FIELD_DATA) return m_fieldRows.isEmpty() ? 0 : 2;
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
        rebuildVisibleColMap();
        endResetModel();
    }
}

void ecvSpreadSheetModel::setDecimalPrecision(int p) {
    if (m_decimalPrecision != p) {
        m_decimalPrecision = p;
        emit dataChanged(index(0, 0), index(rowCount() - 1, columnCount() - 1));
    }
}

void ecvSpreadSheetModel::setFixedRepresentation(bool fixed) {
    if (m_fixedRepresentation != fixed) {
        m_fixedRepresentation = fixed;
        emit dataChanged(index(0, 0), index(rowCount() - 1, columnCount() - 1));
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

void ecvSpreadSheetModel::setSelectedIndices(const QSet<unsigned>& indices) {
    if (m_selectedIndices != indices) {
        if (m_selectionOnly) {
            beginResetModel();
            m_selectedIndices = indices;
            rebuildSortedSelection();
            endResetModel();
        } else {
            m_selectedIndices = indices;
            rebuildSortedSelection();
            if (rowCount() > 0)
                emit dataChanged(index(0, 0),
                                 index(rowCount() - 1, columnCount() - 1),
                                 {Qt::BackgroundRole});
        }
    }
}

void ecvSpreadSheetModel::rebuildSortedSelection() {
    m_sortedSelection = m_selectedIndices.values().toVector();
    std::sort(m_sortedSelection.begin(), m_sortedSelection.end());
}

void ecvSpreadSheetModel::rebuildVisibleColMap() {
    m_visibleColMap.clear();
    for (int i = 0; i < m_columns.size(); ++i) {
        if (m_columns[i].visible) {
            m_visibleColMap.append(i);
        }
    }
}

void ecvSpreadSheetModel::setSelectionOnly(bool on) {
    if (m_selectionOnly != on) {
        beginResetModel();
        m_selectionOnly = on;
        endResetModel();
    }
}

QString ecvSpreadSheetModel::formatValue(double val) const {
    if (std::isnan(val)) return QStringLiteral("nan");
    if (std::isinf(val))
        return val > 0 ? QStringLiteral("inf") : QStringLiteral("-inf");
    if (m_fixedRepresentation) {
        return QString::number(val, 'f', m_decimalPrecision);
    }
    return QString::number(val, 'g', m_decimalPrecision);
}

QVariant ecvSpreadSheetModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) return {};

    if (m_attributeType == FIELD_DATA) {
        if (role != Qt::DisplayRole) return {};
        int row = index.row();
        if (row < 0 || row >= m_fieldRows.size()) return {};
        if (index.column() == 0) return m_fieldRows[row].key;
        if (index.column() == 1) return m_fieldRows[row].value;
        return {};
    }

    if (role == Qt::BackgroundRole && !m_selectionOnly &&
        !m_selectedIndices.isEmpty()) {
        unsigned row = static_cast<unsigned>(index.row());
        if (m_selectedIndices.contains(row)) return QColor(38, 79, 120);
        return {};
    }
    if (role != Qt::DisplayRole) return {};

    unsigned row = static_cast<unsigned>(index.row());
    if (m_selectionOnly && !m_sortedSelection.isEmpty()) {
        if (index.row() >= m_sortedSelection.size()) return {};
        row = m_sortedSelection[index.row()];
    }

    int visCol = index.column();
    int actualCol = (visCol >= 0 && visCol < m_visibleColMap.size())
                            ? m_visibleColMap[visCol]
                            : -1;
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
            case ColumnDef::R:
            case ColumnDef::G:
            case ColumnDef::B: {
                if (!m_cloud->hasColors()) return {};
                const ecvColor::Rgb& c = m_cloud->getPointColor(row);
                double v = (col.type == ColumnDef::R)   ? c.r / 255.0
                           : (col.type == ColumnDef::G) ? c.g / 255.0
                                                        : c.b / 255.0;
                return formatValue(v);
            }
            case ColumnDef::NX:
            case ColumnDef::NY:
            case ColumnDef::NZ: {
                CCVector3 n(0, 0, 0);
                if (m_pcCloud && m_pcCloud->hasNormals()) {
                    n = m_pcCloud->getPointNormal(row);
                } else if (m_cloud && m_cloud->hasNormals()) {
                    n = m_cloud->getPointNormal(row);
                } else if (hasComputedNormal(row)) {
                    unsigned b = row * 3;
                    n = CCVector3(m_computedNormals[b],
                                  m_computedNormals[b + 1],
                                  m_computedNormals[b + 2]);
                } else if ((m_genericMesh && m_genericMesh->hasNormals()) ||
                           (m_mesh && m_mesh->hasNormals())) {
                    n = getNormalFromMeshForVertex(row);
                } else {
                    return {};
                }
                double v = (col.type == ColumnDef::NX)   ? n.x
                           : (col.type == ColumnDef::NY) ? n.y
                                                         : n.z;
                return formatValue(v);
            }
            case ColumnDef::NORMALS_MAG: {
                CCVector3 n(0, 0, 0);
                if (m_pcCloud && m_pcCloud->hasNormals()) {
                    n = m_pcCloud->getPointNormal(row);
                } else if (m_cloud && m_cloud->hasNormals()) {
                    n = m_cloud->getPointNormal(row);
                } else if (hasComputedNormal(row)) {
                    unsigned b = row * 3;
                    n = CCVector3(m_computedNormals[b],
                                  m_computedNormals[b + 1],
                                  m_computedNormals[b + 2]);
                } else if ((m_genericMesh && m_genericMesh->hasNormals()) ||
                           (m_mesh && m_mesh->hasNormals())) {
                    n = getNormalFromMeshForVertex(row);
                } else {
                    return {};
                }
                return formatValue(
                        std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z));
            }
            case ColumnDef::POINTS_MAG: {
                const CCVector3* P = m_cloud->getPoint(row);
                if (!P) return {};
                return formatValue(
                        std::sqrt(P->x * P->x + P->y * P->y + P->z * P->z));
            }
            case ColumnDef::TCOORDS_S:
            case ColumnDef::TCOORDS_T: {
                if (!m_mesh) return {};
                auto* texTable = m_mesh->getTexCoordinatesTable();
                if (!texTable || row >= texTable->size()) return {};
                const TexCoords2D& tc = texTable->at(row);
                return formatValue(col.type == ColumnDef::TCOORDS_S ? tc.tx
                                                                    : tc.ty);
            }
            case ColumnDef::TCOORDS_MAG: {
                if (!m_mesh) return {};
                auto* texTable = m_mesh->getTexCoordinatesTable();
                if (!texTable || row >= texTable->size()) return {};
                const TexCoords2D& tc = texTable->at(row);
                return formatValue(std::sqrt(tc.tx * tc.tx + tc.ty * tc.ty));
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
            case ColumnDef::V1_X:
            case ColumnDef::V1_Y:
            case ColumnDef::V1_Z:
            case ColumnDef::V2_X:
            case ColumnDef::V2_Y:
            case ColumnDef::V2_Z:
            case ColumnDef::V3_X:
            case ColumnDef::V3_Y:
            case ColumnDef::V3_Z: {
                cloudViewer::VerticesIndexes* tri =
                        m_mesh->getTriangleVertIndexes(row);
                if (!tri || !m_cloud) return {};
                unsigned vi = tri->i1;
                if (col.type >= ColumnDef::V2_X && col.type <= ColumnDef::V2_Z)
                    vi = tri->i2;
                else if (col.type >= ColumnDef::V3_X)
                    vi = tri->i3;
                if (vi >= m_cloud->size()) return {};
                const CCVector3* P = m_cloud->getPoint(vi);
                if (!P) return {};
                int comp = (static_cast<int>(col.type) -
                            static_cast<int>(ColumnDef::V1_X)) %
                           3;
                return formatValue(static_cast<double>(P->u[comp]));
            }
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
        if (m_attributeType == FIELD_DATA) {
            if (section == 0) return tr("Property");
            if (section == 1) return tr("Value");
            return {};
        }
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

QString ecvSpreadSheetModel::columnGroupName(int visibleCol) const {
    if (visibleCol < 0 || visibleCol >= m_visibleColMap.size()) return {};
    int actualCol = m_visibleColMap[visibleCol];
    for (int i = actualCol; i >= 0; --i) {
        if (!m_columns[i].name.isEmpty()) return m_columns[i].name;
    }
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
// Custom Header View (ParaView multi-span column groups)
// ============================================================================

ecvSpreadSheetHeaderView::ecvSpreadSheetHeaderView(Qt::Orientation orientation,
                                                   QWidget* parent)
    : QHeaderView(orientation, parent) {
    setSectionsClickable(true);
    setHighlightSections(true);
    setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
}

QSize ecvSpreadSheetHeaderView::sizeHint() const {
    QSize sz = QHeaderView::sizeHint();
    if (m_sourceModel && orientation() == Qt::Horizontal) {
        sz.setHeight(sz.height() * 2);
    }
    return sz;
}

QVector<ecvSpreadSheetHeaderView::GroupInfo>
ecvSpreadSheetHeaderView::collectGroups() const {
    QVector<GroupInfo> groups;
    if (!m_sourceModel) return groups;
    int cnt = count();
    for (int i = 0; i < cnt; ++i) {
        QString headerText =
                m_sourceModel->headerData(i, Qt::Horizontal, Qt::DisplayRole)
                        .toString();
        QString groupName = m_sourceModel->columnGroupName(i);
        if (!headerText.isEmpty() && headerText == groupName) {
            GroupInfo g;
            g.name = groupName;
            g.startSection = i;
            g.endSection = i;
            for (int j = i + 1; j < cnt; ++j) {
                QString nh =
                        m_sourceModel
                                ->headerData(j, Qt::Horizontal, Qt::DisplayRole)
                                .toString();
                QString ng = m_sourceModel->columnGroupName(j);
                if (nh.isEmpty() && ng == groupName) {
                    g.endSection = j;
                } else {
                    break;
                }
            }
            if (g.endSection > g.startSection) {
                groups.append(g);
            }
        }
    }
    return groups;
}

void ecvSpreadSheetHeaderView::paintEvent(QPaintEvent* event) {
    QHeaderView::paintEvent(event);

    if (!m_sourceModel) return;

    auto groups = collectGroups();
    if (groups.isEmpty()) return;

    QPainter painter(viewport());
    painter.setClipping(false);

    int totalH = height();
    int topH = totalH / 2;
    QPalette pal = palette();
    QFont boldFont = painter.font();
    boldFont.setBold(true);

    for (const auto& grp : groups) {
        int x1 = sectionViewportPosition(grp.startSection);
        int x2 = sectionViewportPosition(grp.endSection) +
                 sectionSize(grp.endSection);

        QRect mergedTop(x1, 0, x2 - x1, topH);

        painter.fillRect(mergedTop, pal.brush(QPalette::Button));

        painter.setPen(pal.color(QPalette::Mid));
        painter.drawRect(mergedTop.adjusted(0, 0, -1, -1));
        painter.drawLine(mergedTop.bottomLeft(), mergedTop.bottomRight());

        painter.setFont(boldFont);
        painter.setPen(pal.color(QPalette::ButtonText));
        painter.drawText(mergedTop.adjusted(4, 0, -4, 0), Qt::AlignCenter,
                         grp.name);
    }
}

void ecvSpreadSheetHeaderView::paintSection(QPainter* painter,
                                            const QRect& rect,
                                            int logicalIndex) const {
    if (!painter || !m_sourceModel) {
        QHeaderView::paintSection(painter, rect, logicalIndex);
        return;
    }

    painter->save();

    QString groupName = m_sourceModel->columnGroupName(logicalIndex);
    QString headerText =
            m_sourceModel
                    ->headerData(logicalIndex, Qt::Horizontal, Qt::DisplayRole)
                    .toString();

    bool inGroup = false;
    int compIdx = -1;
    auto groups = collectGroups();
    for (const auto& grp : groups) {
        if (logicalIndex >= grp.startSection &&
            logicalIndex <= grp.endSection) {
            inGroup = true;
            compIdx = logicalIndex - grp.startSection;
            break;
        }
    }

    int topH = rect.height() / 2;
    int botH = rect.height() - topH;
    QRect botRect(rect.x(), rect.y() + topH, rect.width(), botH);

    QPalette pal = palette();
    QFont normalFont = painter->font();
    QFont boldFont = normalFont;
    boldFont.setBold(true);

    if (inGroup) {
        painter->fillRect(botRect, pal.brush(QPalette::Button));
        painter->setPen(pal.color(QPalette::Mid));
        painter->drawRect(botRect.adjusted(0, 0, -1, -1));

        painter->setPen(pal.color(QPalette::ButtonText));
        painter->setFont(normalFont);
        painter->drawText(botRect.adjusted(4, 0, -4, 0), Qt::AlignCenter,
                          QString::number(compIdx));
    } else {
        painter->fillRect(rect, pal.brush(QPalette::Button));
        painter->setPen(pal.color(QPalette::Mid));
        painter->drawRect(rect.adjusted(0, 0, -1, -1));

        painter->setFont(boldFont);
        painter->setPen(pal.color(QPalette::ButtonText));
        painter->drawText(rect.adjusted(4, 0, -4, 0),
                          Qt::AlignLeft | Qt::AlignVCenter, headerText);
    }

    painter->restore();
}

// ============================================================================
// View Widget (ParaView pqSpreadSheetView + pqSpreadSheetViewDecorator)
// ============================================================================

ecvSpreadSheetView::ecvSpreadSheetView(QWidget* parent) : QWidget(parent) {
    m_viewTypeKey = QStringLiteral("SpreadSheet View");
    m_title = ecvViewTitleRegistry::instance().allocate(m_viewTypeKey);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // === Decorator toolbar (ParaView pqSpreadSheetViewDecorator pattern) ===
    // Layout and widget order match pqSpreadSheetViewDecorator.ui exactly.
    auto* decoratorBar = new QWidget(this);
    decoratorBar->setObjectName("DecoratorBar");
    auto* decLayout = new QHBoxLayout(decoratorBar);
    decLayout->setContentsMargins(0, 0, 0, 0);
    decLayout->setSpacing(1);

    m_showingLabel =
            new QLabel(QStringLiteral("<b>Showing  </b>"), decoratorBar);
    decLayout->addWidget(m_showingLabel);

    m_sourceCombo = new QComboBox(decoratorBar);
    m_sourceCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_sourceCombo->addItem(tr("None"));
    decLayout->addWidget(m_sourceCombo);

    auto* attrLabel =
            new QLabel(QStringLiteral("<b>   Attribute:</b>"), decoratorBar);
    decLayout->addWidget(attrLabel);

    m_attributeCombo = new QComboBox(decoratorBar);
    m_attributeCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_attributeCombo->addItem(tr("Point Data"),
                              ecvSpreadSheetModel::POINT_DATA);
    m_attributeCombo->addItem(tr("Cell Data"), ecvSpreadSheetModel::CELL_DATA);
    m_attributeCombo->addItem(tr("Field Data"),
                              ecvSpreadSheetModel::FIELD_DATA);
    decLayout->addWidget(m_attributeCombo);

    auto* precLabel = new QLabel(decoratorBar);
    QFont precFont = precLabel->font();
    precFont.setBold(true);
    precLabel->setFont(precFont);
    precLabel->setText(tr("Precision:"));
    decLayout->addWidget(precLabel);

    m_precisionSpin = new QSpinBox(decoratorBar);
    m_precisionSpin->setRange(1, 32);
    m_precisionSpin->setValue(6);
    m_precisionSpin->setAlignment(Qt::AlignRight | Qt::AlignTrailing |
                                  Qt::AlignVCenter);
    decLayout->addWidget(m_precisionSpin);

    m_fixedRepBtn = new QToolButton(decoratorBar);
    m_fixedRepBtn->setIcon(QIcon(":/Resources/images/svg/pqFixedRepr32.png"));
    m_fixedRepBtn->setToolTip(
            tr("Switches between scientific and fixed-point representation"));
    m_fixedRepBtn->setStatusTip(
            tr("Toggle fixed-point representation (always show #Precision "
               "digits)"));
    m_fixedRepBtn->setCheckable(true);
    decLayout->addWidget(m_fixedRepBtn);

    m_selectionOnlyBtn = new QToolButton(decoratorBar);
    m_selectionOnlyBtn->setIcon(QIcon(":/Resources/images/svg/pqSelect16.png"));
    m_selectionOnlyBtn->setToolTip(tr("Show only selected elements."));
    m_selectionOnlyBtn->setCheckable(true);
    m_selectionOnlyBtn->setToolButtonStyle(Qt::ToolButtonIconOnly);
    decLayout->addWidget(m_selectionOnlyBtn);

    m_columnVisMenu = new QMenu(this);
    m_columnVisBtn = new QToolButton(decoratorBar);
    m_columnVisBtn->setIcon(
            QIcon(":/Resources/images/svg/pqRectilinearGrid16.png"));
    m_columnVisBtn->setToolTip(tr("Toggle column visibility"));
    m_columnVisBtn->setPopupMode(QToolButton::InstantPopup);
    m_columnVisBtn->setMenu(m_columnVisMenu);
    decLayout->addWidget(m_columnVisBtn);

    m_cellConnBtn = new QToolButton(decoratorBar);
    m_cellConnBtn->setIcon(
            QIcon(":/Resources/images/svg/pqProgrammableFilter.svg"));
    m_cellConnBtn->setToolTip(tr("Toggle cell connectivity visibility"));
    m_cellConnBtn->setCheckable(true);
    decLayout->addWidget(m_cellConnBtn);

    m_fieldDataBtn = new QToolButton(decoratorBar);
    m_fieldDataBtn->setIcon(QIcon(":/Resources/images/svg/pqGlobalData.svg"));
    m_fieldDataBtn->setToolTip(tr("Toggle field data visibility"));
    m_fieldDataBtn->setCheckable(true);
    decLayout->addWidget(m_fieldDataBtn);

    m_exportBtn = new QToolButton(decoratorBar);
    m_exportBtn->setIcon(QIcon(":/Resources/images/svg/pqSaveTable32.png"));
    m_exportBtn->setToolTip(tr("Export Spreadsheet"));
    {
        auto* exportMenu = new QMenu(m_exportBtn);
        exportMenu->addAction(tr("Export All to CSV"), this,
                              &ecvSpreadSheetView::exportToCsv);
        exportMenu->addAction(tr("Export Selected Rows to CSV"), this,
                              &ecvSpreadSheetView::exportSelectedRows);
        m_exportBtn->setPopupMode(QToolButton::MenuButtonPopup);
        m_exportBtn->setMenu(exportMenu);
    }
    decLayout->addWidget(m_exportBtn);

    decLayout->addStretch(1);
    layout->addWidget(decoratorBar);

    // === Row 2: Font and display options (ParaView View properties) ===
    auto* displayBar = new QWidget(this);
    displayBar->setObjectName("SpreadDisplayBar");
    auto* displayLayout = new QHBoxLayout(displayBar);
    displayLayout->setContentsMargins(2, 1, 2, 1);
    displayLayout->setSpacing(4);

    auto* cfLabel = new QLabel(tr("Cell Font:"), displayBar);
    displayLayout->addWidget(cfLabel);
    m_cellFontSizeSpin = new QSpinBox(displayBar);
    m_cellFontSizeSpin->setRange(6, 24);
    m_cellFontSizeSpin->setValue(9);
    m_cellFontSizeSpin->setFixedWidth(50);
    displayLayout->addWidget(m_cellFontSizeSpin);

    auto* hfLabel = new QLabel(tr("Header Font:"), displayBar);
    displayLayout->addWidget(hfLabel);
    m_headerFontSizeSpin = new QSpinBox(displayBar);
    m_headerFontSizeSpin->setRange(6, 24);
    m_headerFontSizeSpin->setValue(9);
    m_headerFontSizeSpin->setFixedWidth(50);
    displayLayout->addWidget(m_headerFontSizeSpin);

    displayLayout->addStretch(1);
    layout->addWidget(displayBar);

    // === Search bar with column filter ===
    auto* searchBar = new QWidget(this);
    auto* searchLayout = new QHBoxLayout(searchBar);
    searchLayout->setContentsMargins(2, 1, 2, 1);
    searchLayout->setSpacing(4);

    auto* filterLabel = new QLabel(tr("Filter:"), searchBar);
    searchLayout->addWidget(filterLabel);

    m_filterColumnCombo = new QComboBox(searchBar);
    m_filterColumnCombo->addItem(tr("All Columns"), -1);
    m_filterColumnCombo->setFixedWidth(130);
    m_filterColumnCombo->setToolTip(tr("Filter by specific column"));
    searchLayout->addWidget(m_filterColumnCombo);

    m_searchEdit = new QLineEdit(searchBar);
    m_searchEdit->setPlaceholderText(tr("Filter rows..."));
    m_searchEdit->setClearButtonEnabled(true);
    searchLayout->addWidget(m_searchEdit, 1);
    layout->addWidget(searchBar);

    // === Table view ===
    m_model = new ecvSpreadSheetModel(this);

    m_proxyModel = new QSortFilterProxyModel(this);
    m_proxyModel->setSourceModel(m_model);
    m_proxyModel->setFilterCaseSensitivity(Qt::CaseInsensitive);
    m_proxyModel->setFilterKeyColumn(-1);

    m_tableView = new QTableView(this);
    auto* customHeader =
            new ecvSpreadSheetHeaderView(Qt::Horizontal, m_tableView);
    customHeader->setModel(m_model);
    m_tableView->setHorizontalHeader(customHeader);
    m_tableView->setModel(m_proxyModel);
    m_tableView->setSortingEnabled(true);
    m_tableView->setAlternatingRowColors(true);
    m_tableView->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tableView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    customHeader->setStretchLastSection(true);
    customHeader->setSectionsClickable(true);
    customHeader->setHighlightSections(false);
    m_tableView->verticalHeader()->setDefaultSectionSize(20);
    layout->addWidget(m_tableView, 1);

    // === Statistics bar (click column header to see stats) ===
    m_statsBar = new QLabel(this);
    m_statsBar->setContentsMargins(4, 2, 4, 2);
    m_statsBar->setStyleSheet(
            QStringLiteral("QLabel { background: #2b2b2b; color: #a9b7c6; "
                           "font-family: monospace; font-size: 11px; }"));
    m_statsBar->setVisible(false);
    layout->addWidget(m_statsBar);

    // === Status bar ===
    m_statusLabel = new QLabel(this);
    m_statusLabel->setContentsMargins(4, 2, 4, 2);
    layout->addWidget(m_statusLabel);

    // === Connections ===
    connect(m_searchEdit, &QLineEdit::textChanged, this,
            &ecvSpreadSheetView::onSearchTextChanged);
    connect(m_filterColumnCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ecvSpreadSheetView::onFilterColumnChanged);
    connect(customHeader, &QHeaderView::sectionClicked, this,
            &ecvSpreadSheetView::onStatColumnSelected);
    connect(m_exportBtn, &QToolButton::clicked, this,
            &ecvSpreadSheetView::exportToCsv);
    connect(m_attributeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ecvSpreadSheetView::onAttributeChanged);
    connect(m_precisionSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &ecvSpreadSheetView::onPrecisionChanged);
    connect(m_fixedRepBtn, &QToolButton::toggled, this,
            &ecvSpreadSheetView::onToggleFixed);
    connect(m_columnVisMenu, &QMenu::aboutToShow, this,
            &ecvSpreadSheetView::onColumnVisibilityMenuAboutToShow);
    connect(m_selectionOnlyBtn, &QToolButton::toggled, this,
            &ecvSpreadSheetView::onToggleSelectionOnly);
    connect(m_cellConnBtn, &QToolButton::toggled, this,
            &ecvSpreadSheetView::onToggleCellConnectivity);
    connect(m_fieldDataBtn, &QToolButton::toggled, this, [this](bool checked) {
        if (checked) {
            m_prevAttributeBeforeFieldData = m_attributeCombo->currentIndex();
            m_attributeCombo->setCurrentIndex(
                    static_cast<int>(ecvSpreadSheetModel::FIELD_DATA));
        } else {
            m_attributeCombo->setCurrentIndex(m_prevAttributeBeforeFieldData);
        }
    });
    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ecvSpreadSheetView::onSourceComboChanged);
    m_sourceCombo->installEventFilter(this);

    connect(m_cellFontSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ecvSpreadSheetView::onCellFontSizeChanged);
    connect(m_headerFontSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ecvSpreadSheetView::onHeaderFontSizeChanged);

    auto* copyShortcut = new QShortcut(QKeySequence::Copy, m_tableView);
    connect(copyShortcut, &QShortcut::activated, this,
            &ecvSpreadSheetView::copyToClipboard);

    connect(&ecvViewManager::instance(),
            &ecvViewManager::entitySelectionChanged, this,
            &ecvSpreadSheetView::onEntitySelectionChanged);

    connect(&ecvViewManager::instance(), &ecvViewManager::pointIndicesSelected,
            this, [this](ccHObject* entity, const QSet<unsigned>& indices) {
                if (entity && m_model->entity() == entity) {
                    setSelectedPointIndices(indices);
                }
            });

    connect(m_tableView->selectionModel(),
            &QItemSelectionModel::selectionChanged, this,
            &ecvSpreadSheetView::onTableSelectionChanged);

    installEventFilter(this);

    updateStatusBar();
}

ecvSpreadSheetView::~ecvSpreadSheetView() {
    if (!m_viewTypeKey.isEmpty() && !m_title.isEmpty()) {
        ecvViewTitleRegistry::instance().release(m_viewTypeKey, m_title);
    }
}

QString ecvSpreadSheetView::title() const { return m_title; }

void ecvSpreadSheetView::setEntity(ccHObject* entity) {
    m_model->setEntity(entity);

    auto* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
    auto* mesh = ccHObjectCaster::ToMesh(entity);
    if (mesh && !cloud) {
        cloud = ccHObjectCaster::ToGenericPointCloud(
                mesh->getAssociatedCloud());
    }

    bool hasCellData = (mesh != nullptr);
    auto* itemModel =
            qobject_cast<QStandardItemModel*>(m_attributeCombo->model());
    if (itemModel) {
        auto* item = itemModel->item(1);
        if (item) {
            item->setEnabled(hasCellData);
        }
    }
    if (!hasCellData && m_attributeCombo->currentIndex() == 1) {
        m_attributeCombo->setCurrentIndex(0);
    }

    if (m_sourceCombo) {
        m_sourceCombo->blockSignals(true);
        for (int i = 0; i < m_sourceCombo->count(); ++i) {
            auto* stored = reinterpret_cast<ccHObject*>(
                    m_sourceCombo->itemData(i).value<quintptr>());
            if (stored == entity) {
                m_sourceCombo->setCurrentIndex(i);
                break;
            }
        }
        if (!entity && m_sourceCombo->count() > 0) {
            m_sourceCombo->setCurrentIndex(0);
        }
        unsigned pointCount = cloud ? cloud->size() : 0;
        m_sourceCombo->setToolTip(entity ? tr("%1 (%2 points)")
                                                   .arg(entity->getName())
                                                   .arg(pointCount)
                                         : QString());
        m_sourceCombo->blockSignals(false);
    }

    updateStatusBar();

    QTimer::singleShot(0, this, [this]() {
        m_tableView->resizeColumnsToContents();
        auto* header = m_tableView->horizontalHeader();
        if (header) header->setStretchLastSection(true);
    });
}

void ecvSpreadSheetView::setEntityListProvider(EntityListProvider provider) {
    m_entityListProvider = std::move(provider);
    refreshSourceCombo();
}

void ecvSpreadSheetView::refreshSourceCombo() {
    if (!m_sourceCombo) return;

    ccHObject* current = m_model->entity();

    m_sourceCombo->blockSignals(true);
    m_sourceCombo->clear();
    m_sourceCombo->addItem(tr("None"), QVariant::fromValue<quintptr>(0));

    if (m_entityListProvider) {
        auto entities = m_entityListProvider();
        int newIdx = 0;
        for (int i = 0; i < entities.size(); ++i) {
            ccHObject* e = entities[i];
            if (!e) continue;
            m_sourceCombo->addItem(e->getName(),
                                   QVariant::fromValue<quintptr>(
                                           reinterpret_cast<quintptr>(e)));
            if (e == current) {
                newIdx = m_sourceCombo->count() - 1;
            }
        }
        m_sourceCombo->setCurrentIndex(newIdx);
    } else if (current) {
        m_sourceCombo->addItem(current->getName(),
                               QVariant::fromValue<quintptr>(
                                       reinterpret_cast<quintptr>(current)));
        m_sourceCombo->setCurrentIndex(1);
    }

    m_sourceCombo->blockSignals(false);
}

void ecvSpreadSheetView::onSourceComboChanged(int index) {
    if (index < 0) return;
    auto ptr = m_sourceCombo->itemData(index).value<quintptr>();
    auto* entity = reinterpret_cast<ccHObject*>(ptr);
    setEntity(entity);
}

void ecvSpreadSheetView::onSourceComboAboutToShow() { refreshSourceCombo(); }

void ecvSpreadSheetView::onEntitySelectionChanged(ccHObject* entity) {
    if (!entity) return;
    refreshSourceCombo();
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

void ecvSpreadSheetView::onToggleSelectionOnly(bool checked) {
    m_model->setSelectionOnly(checked);
    if (checked) {
        m_tableView->setSelectionMode(QAbstractItemView::NoSelection);
    } else {
        m_tableView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    }
    updateStatusBar();
}

void ecvSpreadSheetView::onColumnVisibilityMenuAboutToShow() {
    m_columnVisMenu->clear();

    int totalCols = m_model->totalColumnCount();
    QString lastGroupName;
    int componentIdx = 0;
    for (int i = 0; i < totalCols; ++i) {
        QString name = m_model->columnName(i);
        if (name.isEmpty()) {
            ++componentIdx;
            name = QStringLiteral("%1 (%2)")
                           .arg(lastGroupName)
                           .arg(componentIdx);
        } else {
            lastGroupName = name;
            componentIdx = 0;
        }
        auto* action = m_columnVisMenu->addAction(name);
        action->setCheckable(true);
        action->setChecked(m_model->isColumnVisible(i));
        connect(action, &QAction::toggled, this, [this, i](bool checked) {
            m_model->setColumnVisible(i, checked);
            updateStatusBar();
        });
    }
}

void ecvSpreadSheetView::exportToCsv() {
    if (!m_model->entity()) return;

    QString path =
            QFileDialog::getSaveFileName(this, tr("Export SpreadSheet to CSV"),
                                         QString(), tr("CSV (*.csv)"));
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
    auto selModel = m_tableView->selectionModel();
    QModelIndexList selectedRows = selModel->selectedRows();
    if (selectedRows.isEmpty()) {
        auto table = m_model->getRowsAsString();
        if (!table.isEmpty()) {
            QApplication::clipboard()->setText(table);
        }
        return;
    }

    int cols = m_proxyModel->columnCount();
    QString result;
    for (int c = 0; c < cols; ++c) {
        if (c > 0) result += '\t';
        result += m_proxyModel->headerData(c, Qt::Horizontal).toString();
    }
    result += '\n';

    std::sort(selectedRows.begin(), selectedRows.end(),
              [](const QModelIndex& a, const QModelIndex& b) {
                  return a.row() < b.row();
              });
    for (const auto& idx : selectedRows) {
        for (int c = 0; c < cols; ++c) {
            if (c > 0) result += '\t';
            result += m_proxyModel->data(m_proxyModel->index(idx.row(), c))
                              .toString();
        }
        result += '\n';
    }
    QApplication::clipboard()->setText(result);
}

void ecvSpreadSheetModel::setGenerateCellConnectivity(bool on) {
    if (m_cellConnectivity == on) return;
    beginResetModel();
    m_cellConnectivity = on;
    m_columns.clear();
    rebuildColumns();
    endResetModel();
}

void ecvSpreadSheetView::onToggleCellConnectivity(bool checked) {
    m_model->setGenerateCellConnectivity(checked);
    updateStatusBar();
}

void ecvSpreadSheetView::onCellFontSizeChanged(int size) {
    QFont f = m_tableView->font();
    f.setPointSize(size);
    m_tableView->setFont(f);
    m_tableView->verticalHeader()->setDefaultSectionSize(size + 9);
}

void ecvSpreadSheetView::onHeaderFontSizeChanged(int size) {
    QFont f = m_tableView->horizontalHeader()->font();
    f.setPointSize(size);
    m_tableView->horizontalHeader()->setFont(f);
    m_tableView->verticalHeader()->setFont(f);
}

void ecvSpreadSheetView::setSelectedPointIndices(
        const QSet<unsigned>& indices) {
    m_model->setSelectedIndices(indices);
    if (!indices.isEmpty() && !m_model->selectionOnly()) {
        QVector<unsigned> sorted(indices.begin(), indices.end());
        std::sort(sorted.begin(), sorted.end());
        int firstRow = static_cast<int>(sorted.first());
        QModelIndex proxyIdx =
                m_proxyModel->mapFromSource(m_model->index(firstRow, 0));
        if (proxyIdx.isValid()) {
            m_tableView->scrollTo(proxyIdx,
                                  QAbstractItemView::PositionAtCenter);
        }
    }
    updateStatusBar();
}

void ecvSpreadSheetView::onTableSelectionChanged() {
    if (!m_model->entity()) return;

    QModelIndexList selectedRows =
            m_tableView->selectionModel()->selectedRows();
    QVector<unsigned> indices;
    indices.reserve(selectedRows.size());
    for (const auto& proxyIdx : selectedRows) {
        QModelIndex srcIdx = m_proxyModel->mapToSource(proxyIdx);
        if (srcIdx.isValid()) {
            indices.append(static_cast<unsigned>(srcIdx.row()));
        }
    }
    emit tableSelectionChanged(m_model->entity(), indices);
}

bool ecvSpreadSheetView::eventFilter(QObject* obj, QEvent* event) {
    if (obj == m_sourceCombo && event->type() == QEvent::MouseButtonPress) {
        refreshSourceCombo();
    }
    if (event->type() == QEvent::KeyPress) {
        auto* kev = static_cast<QKeyEvent*>(event);
        if (kev->matches(QKeySequence::Copy)) {
            copyToClipboard();
            return true;
        }
    }
    return QWidget::eventFilter(obj, event);
}

void ecvSpreadSheetView::onFilterColumnChanged(int index) {
    int colIdx = m_filterColumnCombo
                         ? m_filterColumnCombo->itemData(index).toInt()
                         : -1;
    m_proxyModel->setFilterKeyColumn(colIdx);
    m_proxyModel->setFilterFixedString(m_searchEdit ? m_searchEdit->text()
                                                    : QString());
    updateStatusBar();
}

void ecvSpreadSheetView::onStatColumnSelected(int logicalIndex) {
    if (!m_statsBar) return;

    int rows = m_proxyModel->rowCount();
    if (rows == 0) {
        m_statsBar->setVisible(false);
        return;
    }

    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();
    double sum = 0;
    double sumSq = 0;
    int validCount = 0;

    for (int r = 0; r < rows; ++r) {
        QVariant val = m_proxyModel->data(m_proxyModel->index(r, logicalIndex));
        bool ok = false;
        double d = val.toDouble(&ok);
        if (ok) {
            if (d < minVal) minVal = d;
            if (d > maxVal) maxVal = d;
            sum += d;
            sumSq += d * d;
            ++validCount;
        }
    }

    if (validCount == 0) {
        m_statsBar->setText(tr("Column %1: no numeric data").arg(logicalIndex));
        m_statsBar->setVisible(true);
        return;
    }

    double mean = sum / validCount;
    double variance = (validCount > 1) ? (sumSq - sum * sum / validCount) /
                                                 (validCount - 1)
                                       : 0.0;
    double stddev = std::sqrt(std::max(0.0, variance));

    QString header =
            m_proxyModel->headerData(logicalIndex, Qt::Horizontal).toString();
    m_statsBar->setText(tr("[%1]  Count: %2  |  Min: %3  |  Max: %4  |  Mean: "
                           "%5  |  StdDev: %6")
                                .arg(header)
                                .arg(validCount)
                                .arg(minVal, 0, 'g', 6)
                                .arg(maxVal, 0, 'g', 6)
                                .arg(mean, 0, 'g', 6)
                                .arg(stddev, 0, 'g', 6));
    m_statsBar->setVisible(true);
}

void ecvSpreadSheetView::exportSelectedRows() {
    auto selModel = m_tableView->selectionModel();
    QModelIndexList selectedRows = selModel->selectedRows();
    if (selectedRows.isEmpty()) {
        exportToCsv();
        return;
    }

    QString path = QFileDialog::getSaveFileName(
            this, tr("Export Selected Rows to CSV"), QString(),
            tr("CSV (*.csv)"));
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

    std::sort(selectedRows.begin(), selectedRows.end(),
              [](const QModelIndex& a, const QModelIndex& b) {
                  return a.row() < b.row();
              });
    for (const auto& idx : selectedRows) {
        for (int c = 0; c < cols; ++c) {
            if (c > 0) out << ",";
            out << m_proxyModel->data(m_proxyModel->index(idx.row(), c))
                            .toString();
        }
        out << "\n";
    }
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
        m_statusLabel->setText(tr("Showing %1 of %2 rows, %3 columns (%4)")
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

    if (m_filterColumnCombo) {
        int prevIdx = m_filterColumnCombo->currentIndex();
        m_filterColumnCombo->blockSignals(true);
        m_filterColumnCombo->clear();
        m_filterColumnCombo->addItem(tr("All Columns"), -1);
        for (int c = 0; c < cols; ++c) {
            QString hdr =
                    m_model->headerData(c, Qt::Horizontal, Qt::DisplayRole)
                            .toString();
            if (!hdr.isEmpty()) {
                m_filterColumnCombo->addItem(hdr, c);
            }
        }
        if (prevIdx >= 0 && prevIdx < m_filterColumnCombo->count())
            m_filterColumnCombo->setCurrentIndex(prevIdx);
        m_filterColumnCombo->blockSignals(false);
    }
}
