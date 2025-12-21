// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"

// Qt
#include <QString>

// VTK
#include <vtkSmartPointer.h>

// Forward declarations
class vtkPolyData;
class vtkDataArray;
class vtkFieldData;

/**
 * @brief Helper class for generating tooltip information for selected elements
 *
 * Based on ParaView's vtkSMTooltipSelectionPipeline
 * Generates HTML-formatted tooltip text showing:
 * - Element ID
 * - Coordinates (for points)
 * - Type (for cells)
 * - Data array values (Intensity, RGB, normals, etc.)
 *
 * Reference: ParaView/Remoting/Misc/vtkSMTooltipSelectionPipeline.cxx
 */
class QPCL_ENGINE_LIB_API cvSelectionTooltipHelper {
public:
    /**
     * @brief Element association type
     */
    enum AssociationType { POINTS = 1, CELLS = 0 };

    /**
     * @brief Constructor
     */
    cvSelectionTooltipHelper();

    /**
     * @brief Destructor
     */
    ~cvSelectionTooltipHelper();

    /**
     * @brief Generate tooltip information for a selected element
     * @param polyData The mesh data
     * @param elementId The ID of the selected element (point or cell)
     * @param association Type of element (POINTS or CELLS)
     * @param datasetName Optional name of the dataset
     * @return HTML-formatted tooltip string
     */
    QString getTooltipInfo(vtkPolyData* polyData,
                           vtkIdType elementId,
                           AssociationType association,
                           const QString& datasetName = QString());

    /**
     * @brief Generate plain text tooltip (no HTML formatting)
     * @param polyData The mesh data
     * @param elementId The ID of the selected element
     * @param association Type of element (POINTS or CELLS)
     * @param datasetName Optional name of the dataset
     * @return Plain text tooltip string
     */
    QString getPlainTooltipInfo(vtkPolyData* polyData,
                                vtkIdType elementId,
                                AssociationType association,
                                const QString& datasetName = QString());

    /**
     * @brief Set maximum number of attributes to display
     * @param maxAttribs Maximum number of data arrays to show (default: 15)
     */
    void setMaxAttributes(int maxAttribs);

private:
    /**
     * @brief Format point tooltip information
     */
    QString formatPointTooltip(vtkPolyData* polyData,
                               vtkIdType pointId,
                               const QString& datasetName);

    /**
     * @brief Format cell tooltip information
     */
    QString formatCellTooltip(vtkPolyData* polyData,
                              vtkIdType cellId,
                              const QString& datasetName);

    /**
     * @brief Add data array values to tooltip
     */
    void addArrayValues(QString& tooltip,
                        vtkFieldData* fieldData,
                        vtkIdType tupleIndex);

    /**
     * @brief Format a single data array value
     */
    QString formatArrayValue(vtkDataArray* array, vtkIdType tupleIndex);

    /**
     * @brief Format a number with ParaView-style intelligent formatting
     * Uses scientific notation for very large/small numbers, otherwise 'g'
     * format
     */
    QString formatNumber(double value);

    int m_maxAttributes;  ///< Maximum number of attributes to display
};
