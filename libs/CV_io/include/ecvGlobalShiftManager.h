// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_CORE_LIB
#include <CVGeom.h>

// LOCAL
#include "CV_io.h"

// Qt
#include <QString>

// STL
#include <vector>

class ccHObject;

/**
 * @class ecvGlobalShiftManager
 * @brief Manager for handling large coordinate shifts and scales
 * 
 * Helper class to automatically handle coordinate shift and scale when loading
 * entities with very large or very small coordinates. This is essential for
 * maintaining precision in floating-point calculations.
 * 
 * When point cloud coordinates are very large (e.g., UTM coordinates in meters),
 * floating-point precision can be lost. This manager automatically detects such
 * cases and suggests appropriate coordinate shifts to bring values closer to zero.
 * 
 * Similarly, when dimensions are very large or very small, automatic scaling
 * can be applied to improve numerical stability.
 * 
 * @see CCVector3d
 */
class CV_IO_LIB_API ecvGlobalShiftManager {
public:
    /**
     * @brief Strategy for handling coordinate shift/scale
     */
    enum Mode {
        NO_DIALOG,              ///< No dialog, no automatic shift
        NO_DIALOG_AUTO_SHIFT,   ///< Automatic shift without dialog
        DIALOG_IF_NECESSARY,    ///< Show dialog only if shift needed
        ALWAYS_DISPLAY_DIALOG   ///< Always show dialog
    };

    /**
     * @brief Handle coordinate shift/scale for a 3D point
     * 
     * Main entry point for coordinate transformation management. Analyzes
     * the given point and diagonal length to determine if shift/scale is needed,
     * and handles user interaction according to the specified mode.
     * 
     * @param P Input 3D point to analyze
     * @param diagonal Bounding box diagonal length
     * @param mode Interaction mode
     * @param useInputCoordinatesShiftIfPossible Use provided shift if valid
     * @param coordinatesShift Input/output coordinate shift
     * @param preserveCoordinateShift Output: whether to preserve shift
     * @param coordinatesScale Output: coordinate scale factor
     * @param applyAll Output: whether to apply to all subsequent loads
     * @return true if shift/scale was successfully handled
     */
    static bool Handle(const CCVector3d& P,
                       double diagonal,
                       Mode mode,
                       bool useInputCoordinatesShiftIfPossible,
                       CCVector3d& coordinatesShift,
                       bool* preserveCoordinateShift,
                       double* coordinatesScale,
                       bool* applyAll = 0);

    /**
     * @brief Check if 3D point coordinates need shifting
     * @param P Point to check
     * @return true if coordinates are too large
     */
    static bool NeedShift(const CCVector3d& P);
    
    /**
     * @brief Check if single coordinate needs shifting
     * @param d Coordinate value to check
     * @return true if coordinate is too large (absolute value)
     */
    static bool NeedShift(double d);
    
    /**
     * @brief Check if dimension needs rescaling
     * @param d Dimension value (e.g., diagonal length)
     * @return true if dimension is too large or too small
     */
    static bool NeedRescale(double d);

    /**
     * @brief Suggest optimal shift for a 3D point
     * 
     * Calculates an appropriate coordinate shift to bring point
     * coordinates closer to zero while maintaining precision.
     * @param P Point in global coordinates
     * @return Suggested coordinate shift
     */
    static CCVector3d BestShift(const CCVector3d& P);
    
    /**
     * @brief Suggest optimal scale for a dimension
     * @param d Dimension value in global space
     * @return Suggested scale factor
     */
    static double BestScale(double d);

    /**
     * @brief Get maximum acceptable absolute coordinate value
     * @return Max coordinate threshold
     */
    static double MaxCoordinateAbsValue() { return MAX_COORDINATE_ABS_VALUE; }
    
    /**
     * @brief Set maximum acceptable absolute coordinate value
     * @param value New max coordinate threshold
     */
    static void SetMaxCoordinateAbsValue(double value) {
        MAX_COORDINATE_ABS_VALUE = value;
    }

    /**
     * @brief Get maximum acceptable bounding box diagonal
     * @return Max diagonal length threshold
     */
    static double MaxBoundgBoxDiagonal() { return MAX_DIAGONAL_LENGTH; }
    
    /**
     * @brief Set maximum acceptable bounding box diagonal
     * @param value New max diagonal threshold
     */
    static void SetMaxBoundgBoxDiagonal(double value) {
        MAX_DIAGONAL_LENGTH = value;
    }

    /**
     * @brief Store a shift/scale pair for later reuse
     * @param shift Coordinate shift vector
     * @param scale Scale factor
     * @param preserve Whether to preserve this shift for future loads
     */
    static void StoreShift(const CCVector3d& shift,
                           double scale,
                           bool preserve = true);

public:  // Shift and scale info
    /**
     * @struct ShiftInfo
     * @brief Container for coordinate shift and scale information
     */
    struct ShiftInfo {
        CCVector3d shift;   ///< Coordinate shift vector
        double scale;       ///< Scale factor
        QString name;       ///< Descriptive name
        bool preserve;      ///< Whether to preserve for future use

        /**
         * @brief Default constructor
         * @param str Descriptive name (default: "unnamed")
         */
        ShiftInfo(QString str = QString("unnamed"))
            : shift(0, 0, 0), scale(1.0), name(str), preserve(true) {}
        
        /**
         * @brief Constructor with shift and scale
         * @param str Descriptive name
         * @param T Coordinate shift vector
         * @param s Scale factor (default: 1.0)
         */
        ShiftInfo(QString str, const CCVector3d& T, double s = 1.0)
            : shift(T), scale(s), name(str), preserve(true) {}
    };

    /**
     * @brief Get last stored shift/scale info
     * @param info Output shift information
     * @return true if info was retrieved successfully
     */
    static bool GetLast(ShiftInfo& info);
    
    /**
     * @brief Get all stored shift/scale infos
     * @param infos Output vector of shift information
     * @return true if infos were retrieved successfully
     */
    static bool GetLast(std::vector<ShiftInfo>& infos);

protected:
    /// Max acceptable coordinate absolute value
    static double MAX_COORDINATE_ABS_VALUE;

    /// Max acceptable diagonal length
    static double MAX_DIAGONAL_LENGTH;
};
