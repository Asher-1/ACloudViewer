// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvTooltipFormatter.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkStringArray.h>

// Qt
#include <QRegExp>
#include <QStringList>

//-----------------------------------------------------------------------------
cvTooltipFormatter::cvTooltipFormatter() : m_maxAttributes(15) {}

//-----------------------------------------------------------------------------
cvTooltipFormatter::~cvTooltipFormatter() {}

//-----------------------------------------------------------------------------
void cvTooltipFormatter::setMaxAttributes(int maxAttribs) {
    m_maxAttributes = maxAttribs;
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::getTooltipInfo(vtkPolyData* polyData,
                                                 vtkIdType elementId,
                                                 AssociationType association,
                                                 const QString& datasetName) {
    if (!polyData) {
        CVLog::Error("[cvTooltipFormatter] Invalid polyData");
        return QString();
    }

    if (association == POINTS) {
        return formatPointTooltip(polyData, elementId, datasetName);
    } else {
        return formatCellTooltip(polyData, elementId, datasetName);
    }
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::getPlainTooltipInfo(
        vtkPolyData* polyData,
        vtkIdType elementId,
        AssociationType association,
        const QString& datasetName) {
    // Get HTML tooltip and strip HTML tags
    QString htmlTooltip =
            getTooltipInfo(polyData, elementId, association, datasetName);

    // Simple HTML tag removal
    QString plainText = htmlTooltip;
    plainText.remove(QRegExp("<[^>]*>"));
    plainText.replace("&nbsp;", " ");

    return plainText;
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatPointTooltip(
        vtkPolyData* polyData, vtkIdType pointId, const QString& datasetName) {
    if (pointId < 0 || pointId >= polyData->GetNumberOfPoints()) {
        CVLog::Error("[cvTooltipFormatter] Invalid point ID: %lld",
                     pointId);
        return QString();
    }

    QString tooltip;

    // ParaView format: Dataset name as first line (no "Block:" prefix)
    if (!datasetName.isEmpty()) {
        tooltip += QString("<b>%1</b>").arg(datasetName);
    }

    // ParaView format: Point ID (with indent using &nbsp; for HTML)
    tooltip += QString("\n&nbsp;&nbsp;Id: %1").arg(pointId);

    // ParaView format: Coordinates with 6 significant digits (%g format, with
    // indent)
    double point[3];
    polyData->GetPoint(pointId, point);
    tooltip += QString("\n&nbsp;&nbsp;Coords: (%1, %2, %3)")
                       .arg(point[0], 0, 'g', 6)
                       .arg(point[1], 0, 'g', 6)
                       .arg(point[2], 0, 'g', 6);

    // ParaView format: Point data arrays
    vtkPointData* pointData = polyData->GetPointData();
    if (pointData) {
        int numArrays = pointData->GetNumberOfArrays();
        int displayedArrays = 0;  // Counter for limiting displayed attributes

        // Show normals if available (with indent, ParaView style)
        if (pointData->GetNormals()) {
            double* normal = pointData->GetNormals()->GetTuple3(pointId);
            tooltip += QString("\n&nbsp;&nbsp;Normals: (%1, %2, %3)")
                               .arg(normal[0], 0, 'f', 4)
                               .arg(normal[1], 0, 'f', 4)
                               .arg(normal[2], 0, 'f', 4);
            displayedArrays++;
        }

        // Show texture coordinates with material name (ParaView style)
        // ParaView uses MaterialNames from field data to name texture
        // coordinates
        vtkStringArray* materialNamesArray = nullptr;
        vtkFieldData* fieldData = polyData->GetFieldData();
        if (fieldData) {
            materialNamesArray = vtkStringArray::SafeDownCast(
                    fieldData->GetAbstractArray("MaterialNames"));
        }

        // Look for texture coordinate arrays
        bool foundTextureCoords = false;
        for (int i = 0; i < numArrays && displayedArrays < m_maxAttributes;
             ++i) {
            vtkDataArray* array = pointData->GetArray(i);
            if (!array) continue;

            QString arrayName = QString::fromUtf8(array->GetName());

            // Check if this is texture coordinates (2 or 3 components)
            int numComp = array->GetNumberOfComponents();
            if ((numComp == 2 || numComp == 3) &&
                (arrayName.contains("texture", Qt::CaseInsensitive) ||
                 arrayName.contains("tcoords", Qt::CaseInsensitive) ||
                 arrayName.contains("uv", Qt::CaseInsensitive) ||
                 arrayName == "TCoords")) {
                // Extract material name intelligently (ParaView-style)
                QString matName;

                // Strategy 1: Get from MaterialNames field data array
                // (preferred) Array name format: "TCoords0", "TCoords1", etc.
                // Extract index and look up in MaterialNames
                if (materialNamesArray &&
                    arrayName.startsWith("TCoords", Qt::CaseInsensitive)) {
                    // Extract texture coordinate index from array name
                    QString indexStr = arrayName;
                    indexStr.remove("TCoords", Qt::CaseInsensitive);
                    indexStr.remove("Coords", Qt::CaseInsensitive);
                    indexStr.remove("Texture", Qt::CaseInsensitive);
                    bool ok = false;
                    int matIndex = indexStr.toInt(&ok);

                    if (ok && matIndex >= 0 &&
                        matIndex < materialNamesArray->GetNumberOfTuples()) {
                        matName = QString::fromStdString(
                                materialNamesArray->GetValue(matIndex));
                        CVLog::PrintDebug(
                                QString("[cvTooltipFormatter] Using "
                                        "MaterialNames[%1]='%2' for texture "
                                        "coordinates")
                                        .arg(matIndex)
                                        .arg(matName));
                    }
                }

                // Strategy 2: If MaterialNames lookup failed, infer from
                // dataset name
                if (matName.isEmpty() && !datasetName.isEmpty()) {
                    // Extract base name from dataset (e.g., "wooden crate.obj"
                    // -> "crate")
                    matName = datasetName;
                    matName.replace(".obj", "", Qt::CaseInsensitive);
                    matName.replace(".ply", "", Qt::CaseInsensitive);
                    matName.replace(".stl", "", Qt::CaseInsensitive);
                    // Get last word
                    QStringList parts = matName.split(" ", Qt::SkipEmptyParts);
                    if (!parts.isEmpty()) {
                        matName = parts.last().toLower();
                    }
                }

                // Strategy 3: Final fallback
                if (matName.isEmpty()) {
                    matName = "texture";
                }

                if (numComp == 2) {
                    double* tc = array->GetTuple2(pointId);
                    tooltip += QString("\n&nbsp;&nbsp;%1: (%2, %3)")
                                       .arg(matName)
                                       .arg(tc[0], 0, 'f',
                                            5)  // 5 decimals like ParaView
                                       .arg(tc[1], 0, 'f', 5);
                } else if (numComp == 3) {
                    double* tc = array->GetTuple3(pointId);
                    tooltip += QString("\n&nbsp;&nbsp;%1: (%2, %3, %4)")
                                       .arg(matName)
                                       .arg(tc[0], 0, 'f', 5)
                                       .arg(tc[1], 0, 'f', 5)
                                       .arg(tc[2], 0, 'f', 5);
                }

                displayedArrays++;
                foundTextureCoords = true;
                break;  // Only show first texture coordinate array
            }
        }

        // Show scalars if available (after normals and texture coords)
        if (pointData->GetScalars() && displayedArrays < m_maxAttributes) {
            QString scalarName =
                    QString::fromUtf8(pointData->GetScalars()->GetName());
            // Skip if it's texture coordinates
            if (!scalarName.contains("texture", Qt::CaseInsensitive) &&
                !scalarName.contains("tcoord", Qt::CaseInsensitive)) {
                double scalar = pointData->GetScalars()->GetTuple1(pointId);
                tooltip += QString("\n&nbsp;&nbsp;%1: %2")
                                   .arg(scalarName.isEmpty() ? "Scalars"
                                                             : scalarName)
                                   .arg(formatNumber(scalar));
                displayedArrays++;
            }
        }

        // Add additional point data arrays (custom attributes)
        for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
            if (displayedArrays >= m_maxAttributes) {
                tooltip += QString("\n&nbsp;&nbsp;<i>... (%1 more attributes "
                                   "hidden)</i>")
                                   .arg(pointData->GetNumberOfArrays() - i);
                break;
            }

            vtkDataArray* array = pointData->GetArray(i);
            if (!array) continue;

            QString arrayName = QString::fromUtf8(array->GetName());

            // Skip already displayed arrays
            if (arrayName.isEmpty() || array == pointData->GetNormals() ||
                array == pointData->GetScalars() ||
                arrayName.contains("texture", Qt::CaseInsensitive) ||
                arrayName.contains("tcoord", Qt::CaseInsensitive) ||
                arrayName.contains("uv", Qt::CaseInsensitive) ||
                arrayName == "TCoords") {
                continue;
            }

            // Display this custom attribute
            QString valueStr = formatArrayValue(array, pointId);
            if (!valueStr.isEmpty()) {
                tooltip += QString("\n&nbsp;&nbsp;%1: %2")
                                   .arg(arrayName)
                                   .arg(valueStr);
                displayedArrays++;
            }
        }
    }

    // ParaView format: Field data arrays (with horizontal line separator)
    vtkFieldData* fieldDataPoint = polyData->GetFieldData();
    if (fieldDataPoint && fieldDataPoint->GetNumberOfArrays() > 0) {
        bool hasFieldData = false;
        bool isFirstField = true;  // Track if this is the first field data item
        for (int i = 0; i < fieldDataPoint->GetNumberOfArrays(); ++i) {
            vtkAbstractArray* abstractArray =
                    fieldDataPoint->GetAbstractArray(i);
            if (!abstractArray) continue;

            QString arrayName = QString::fromUtf8(abstractArray->GetName());

            // Skip internal VTK arrays, DatasetName (already shown), and
            // MaterialNames (internal use only) MaterialNames is used
            // internally for texture coordinate naming, not displayed in
            // ParaView
            if (arrayName.startsWith("vtk", Qt::CaseInsensitive) ||
                arrayName == "DatasetName" || arrayName == "MaterialNames") {
                continue;
            }

            if (!hasFieldData) {
                tooltip += "\n<hr>";  // Separator line (ParaView style)
                hasFieldData = true;
            }

            // Determine line prefix: first field has no \n, subsequent fields
            // do Use &nbsp; for indent (HTML non-breaking space)
            QString linePrefix =
                    isFirstField ? "&nbsp;&nbsp;" : "\n&nbsp;&nbsp;";

            // Try to cast as string array first (for MaterialLibraries, etc.)
            vtkStringArray* stringArray =
                    vtkStringArray::SafeDownCast(abstractArray);
            if (stringArray && stringArray->GetNumberOfTuples() > 0) {
                // Format string array values
                if (stringArray->GetNumberOfTuples() == 1) {
                    QString value =
                            QString::fromStdString(stringArray->GetValue(0));
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(value);
                } else {
                    // Show all string values separated by commas
                    QStringList values;
                    for (vtkIdType j = 0; j < stringArray->GetNumberOfTuples();
                         ++j) {
                        values << QString::fromStdString(
                                stringArray->GetValue(j));
                    }
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(values.join(", "));
                }
                isFirstField = false;
            } else {
                // Try as numeric data array
                vtkDataArray* array = vtkDataArray::SafeDownCast(abstractArray);
                if (array && array->GetNumberOfTuples() > 0) {
                    // For numeric field data, show first value or array info
                    if (array->GetNumberOfTuples() == 1) {
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr);
                    } else {
                        // Show first value with element count
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3 (array of %4)")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr)
                                           .arg(array->GetNumberOfTuples());
                    }
                    isFirstField = false;
                }
            }
        }
    }

    // Convert \n to <br> for proper HTML rendering in Qt tooltips
    tooltip.replace("\n", "<br>");

    return tooltip;
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatCellTooltip(
        vtkPolyData* polyData, vtkIdType cellId, const QString& datasetName) {
    if (cellId < 0 || cellId >= polyData->GetNumberOfCells()) {
        CVLog::Error("[cvTooltipFormatter] Invalid cell ID: %lld",
                     cellId);
        return QString();
    }

    QString tooltip;

    // ParaView format: Dataset name as first line (no "Block:" prefix)
    if (!datasetName.isEmpty()) {
        tooltip += QString("<b>%1</b>").arg(datasetName);
    }

    // ParaView format: Cell ID (with indent using &nbsp; for HTML)
    tooltip += QString("\n&nbsp;&nbsp;Id: %1").arg(cellId);

    // ParaView format: Cell type (same as ParaView
    // vtkSMCoreUtilities::GetStringForCellType)
    vtkCell* cell = polyData->GetCell(cellId);
    if (cell) {
        QString cellType;
        switch (cell->GetCellType()) {
            case VTK_EMPTY_CELL:
                cellType = "Empty Cell";
                break;
            case VTK_VERTEX:
                cellType = "Vertex";
                break;
            case VTK_POLY_VERTEX:
                cellType = "Poly Vertex";
                break;
            case VTK_LINE:
                cellType = "Line";
                break;
            case VTK_POLY_LINE:
                cellType = "Poly Line";
                break;
            case VTK_TRIANGLE:
                cellType = "Triangle";
                break;
            case VTK_TRIANGLE_STRIP:
                cellType = "Triangle Strip";
                break;
            case VTK_POLYGON:
                cellType = "Polygon";
                break;
            case VTK_PIXEL:
                cellType = "Pixel";
                break;
            case VTK_QUAD:
                cellType = "Quad";
                break;
            case VTK_TETRA:
                cellType = "Tetra";
                break;
            case VTK_VOXEL:
                cellType = "Voxel";
                break;
            case VTK_HEXAHEDRON:
                cellType = "Hexahedron";
                break;
            case VTK_WEDGE:
                cellType = "Wedge";
                break;
            case VTK_PYRAMID:
                cellType = "Pyramid";
                break;
            case VTK_PENTAGONAL_PRISM:
                cellType = "Pentagonal Prism";
                break;
            case VTK_HEXAGONAL_PRISM:
                cellType = "Hexagonal Prism";
                break;
            default:
                cellType = QString("Unknown (%1)").arg(cell->GetCellType());
        }
        tooltip +=
                QString("\n&nbsp;&nbsp;Type: %1").arg(cellType);  // With indent

        // Show number of points in this cell
        vtkIdType npts = cell->GetNumberOfPoints();
        tooltip += QString("\n&nbsp;&nbsp;Number of Points: %1").arg(npts);

        // Show point IDs that make up this cell
        if (npts > 0 && npts <= 10) {  // Only show if reasonable number
            QString pointIds;
            for (vtkIdType i = 0; i < npts; ++i) {
                if (i > 0) pointIds += ", ";
                pointIds += QString::number(cell->GetPointId(i));
            }
            tooltip += QString("\n&nbsp;&nbsp;Point IDs: [%1]").arg(pointIds);
        }

        // Show cell center/centroid
        double center[3] = {0, 0, 0};
        for (vtkIdType i = 0; i < npts; ++i) {
            double pt[3];
            polyData->GetPoint(cell->GetPointId(i), pt);
            center[0] += pt[0];
            center[1] += pt[1];
            center[2] += pt[2];
        }
        if (npts > 0) {
            center[0] /= npts;
            center[1] /= npts;
            center[2] /= npts;
            tooltip += QString("\n&nbsp;&nbsp;Center: (%1, %2, %3)")
                               .arg(center[0], 0, 'g', 6)
                               .arg(center[1], 0, 'g', 6)
                               .arg(center[2], 0, 'g', 6);
        }
    }

    // ParaView format: Cell data arrays
    vtkCellData* cellData = polyData->GetCellData();
    if (cellData) {
        int numArrays = cellData->GetNumberOfArrays();
        int displayedArrays = 0;  // Counter for limiting displayed attributes

        // Show normals if available (with indent, ParaView style)
        if (cellData->GetNormals()) {
            double* normal = cellData->GetNormals()->GetTuple3(cellId);
            tooltip += QString("\n&nbsp;&nbsp;Normals: (%1, %2, %3)")
                               .arg(normal[0], 0, 'f', 4)
                               .arg(normal[1], 0, 'f', 4)
                               .arg(normal[2], 0, 'f', 4);
            displayedArrays++;
        }

        // Show scalars if available (after normals)
        if (cellData->GetScalars() && displayedArrays < m_maxAttributes) {
            QString scalarName =
                    QString::fromUtf8(cellData->GetScalars()->GetName());
            double scalar = cellData->GetScalars()->GetTuple1(cellId);
            tooltip +=
                    QString("\n&nbsp;&nbsp;%1: %2")
                            .arg(scalarName.isEmpty() ? "Scalars" : scalarName)
                            .arg(formatNumber(scalar));
            displayedArrays++;
        }

        // Add additional cell data arrays (custom attributes)
        for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
            if (displayedArrays >= m_maxAttributes) {
                tooltip += QString("\n&nbsp;&nbsp;<i>... (%1 more attributes "
                                   "hidden)</i>")
                                   .arg(cellData->GetNumberOfArrays() - i);
                break;
            }

            vtkDataArray* array = cellData->GetArray(i);
            if (!array) continue;

            QString arrayName = QString::fromUtf8(array->GetName());

            // Skip already displayed arrays
            if (arrayName.isEmpty() || array == cellData->GetNormals() ||
                array == cellData->GetScalars()) {
                continue;
            }

            // Display this custom attribute
            QString valueStr = formatArrayValue(array, cellId);
            if (!valueStr.isEmpty()) {
                tooltip += QString("\n&nbsp;&nbsp;%1: %2")
                                   .arg(arrayName)
                                   .arg(valueStr);
                displayedArrays++;
            }
        }
    }

    // ParaView format: Field data arrays (with horizontal line separator)
    vtkFieldData* fieldDataCell = polyData->GetFieldData();
    if (fieldDataCell && fieldDataCell->GetNumberOfArrays() > 0) {
        bool hasFieldData = false;
        bool isFirstField = true;  // Track if this is the first field data item
        for (int i = 0; i < fieldDataCell->GetNumberOfArrays(); ++i) {
            vtkAbstractArray* abstractArray =
                    fieldDataCell->GetAbstractArray(i);
            if (!abstractArray) continue;

            QString arrayName = QString::fromUtf8(abstractArray->GetName());

            // Skip internal VTK arrays, DatasetName (already shown), and
            // MaterialNames (internal use only) MaterialNames is used
            // internally for texture coordinate naming, not displayed in
            // ParaView
            if (arrayName.startsWith("vtk", Qt::CaseInsensitive) ||
                arrayName == "DatasetName" || arrayName == "MaterialNames") {
                continue;
            }

            if (!hasFieldData) {
                tooltip += "\n<hr>";  // Separator line (ParaView style)
                hasFieldData = true;
            }

            // Determine line prefix: first field has no \n, subsequent fields
            // do Use &nbsp; for indent (HTML non-breaking space)
            QString linePrefix =
                    isFirstField ? "&nbsp;&nbsp;" : "\n&nbsp;&nbsp;";

            // Try to cast as string array first (for MaterialLibraries, etc.)
            vtkStringArray* stringArray =
                    vtkStringArray::SafeDownCast(abstractArray);
            if (stringArray && stringArray->GetNumberOfTuples() > 0) {
                // Format string array values
                if (stringArray->GetNumberOfTuples() == 1) {
                    QString value =
                            QString::fromStdString(stringArray->GetValue(0));
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(value);
                } else {
                    // Show all string values separated by commas
                    QStringList values;
                    for (vtkIdType j = 0; j < stringArray->GetNumberOfTuples();
                         ++j) {
                        values << QString::fromStdString(
                                stringArray->GetValue(j));
                    }
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(values.join(", "));
                }
                isFirstField = false;
            } else {
                // Try as numeric data array
                vtkDataArray* array = vtkDataArray::SafeDownCast(abstractArray);
                if (array && array->GetNumberOfTuples() > 0) {
                    // For numeric field data, show value intelligently
                    if (array->GetNumberOfTuples() == 1) {
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr);
                    } else {
                        // Show first value with element count
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3 (array of %4)")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr)
                                           .arg(array->GetNumberOfTuples());
                    }
                    isFirstField = false;
                }
            }
        }
    }

    // Convert \n to <br> for proper HTML rendering in Qt tooltips
    tooltip.replace("\n", "<br>");

    return tooltip;
}

//-----------------------------------------------------------------------------
void cvTooltipFormatter::addArrayValues(QString& tooltip,
                                              vtkFieldData* fieldData,
                                              vtkIdType tupleIndex) {
    if (!fieldData) {
        return;
    }

    int numArrays = fieldData->GetNumberOfArrays();

    for (int i = 0; i < numArrays; ++i) {
        vtkDataArray* array = fieldData->GetArray(i);
        if (!array) {
            continue;
        }

        // ParaView format: Skip VTK internal arrays and arrays with empty names
        QString arrayName = QString::fromUtf8(array->GetName());
        if (arrayName.isEmpty() ||
            arrayName.startsWith("vtk", Qt::CaseInsensitive) ||
            arrayName == "vtkOriginalPointIds" ||
            arrayName == "vtkOriginalCellIds" ||
            arrayName == "vtkCompositeIndexArray" ||
            arrayName == "vtkGhostType" || arrayName == "vtkValidPointMask") {
            continue;
        }

        // Format array value
        QString valueStr = formatArrayValue(array, tupleIndex);
        if (!valueStr.isEmpty()) {
            // ParaView format: No indent, newline
            tooltip += QString("\n%1: %2").arg(arrayName).arg(valueStr);
        }
    }
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatArrayValue(vtkDataArray* array,
                                                   vtkIdType tupleIndex) {
    if (!array || tupleIndex < 0 || tupleIndex >= array->GetNumberOfTuples()) {
        return QString();
    }

    int numComponents = array->GetNumberOfComponents();
    const int maxDisplayedComp = 9;  // ParaView limit

    if (numComponents == 1) {
        // ParaView format: Scalar value with intelligent formatting
        double value = array->GetTuple1(tupleIndex);
        return formatNumber(value);
    } else {
        // ParaView format: Multi-component with 6 significant digits
        QString result;
        if (numComponents > 1) {
            result = "(";
        }

        for (int i = 0; i < std::min(numComponents, maxDisplayedComp); ++i) {
            double value = array->GetComponent(tupleIndex, i);
            result += formatNumber(value);
            if (i + 1 < numComponents && i < maxDisplayedComp) {
                result += ", ";
            }
        }

        // ParaView format: Show ellipsis if more than maxDisplayedComp
        if (numComponents > maxDisplayedComp) {
            result += ", ...";
        }

        if (numComponents > 1) {
            result += ")";
        }
        return result;
    }
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatNumber(double value) {
    // ParaView intelligent number formatting
    double absValue = qAbs(value);

    // Use scientific notation for very large or very small numbers
    if (absValue > 0 && (absValue < 1e-4 || absValue >= 1e6)) {
        return QString::number(value, 'e', 4);
    }
    // Use 'g' format (auto-select between 'f' and 'e') with 6 significant
    // digits
    else {
        return QString::number(value, 'g', 6);
    }
}
