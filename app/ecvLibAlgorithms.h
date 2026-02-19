// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <GeometricalAnalysisTools.h>

#include "ecvHObject.h"

class QWidget;

class ccGenericPointCloud;
class ecvProgressDialog;

/**
 * @namespace ccLibAlgorithms
 * @brief High-level wrappers for CloudViewer core algorithms
 * 
 * Provides convenient, application-level interfaces to CloudViewer's core
 * algorithms. These functions handle:
 * - Progress dialog management
 * - Error handling and user feedback
 * - Batch processing of multiple entities
 * - Default parameter calculation
 * - UI integration (parent widgets, etc.)
 * 
 * This namespace acts as a bridge between CloudViewer's low-level core
 * library and the application's user interface.
 * 
 * @see cloudViewer::GeometricalAnalysisTools
 * @see cloudViewer::RegistrationTools
 */
namespace ccLibAlgorithms {

/**
 * @brief Calculate default kernel size for single cloud
 * 
 * Estimates appropriate neighborhood size based on cloud density.
 * Used for local algorithms (normals, curvature, density, etc.).
 * 
 * @param cloud Point cloud to analyze
 * @param knn K-nearest neighbors to consider (default: 12)
 * @return Recommended kernel radius
 */
PointCoordinateType GetDefaultCloudKernelSize(ccGenericPointCloud* cloud,
                                              unsigned knn = 12);

/**
 * @brief Calculate default kernel size for multiple clouds
 * 
 * Estimates appropriate neighborhood size across multiple entities.
 * @param entities Entities to analyze
 * @param knn K-nearest neighbors to consider (default: 12)
 * @return Recommended kernel radius
 */
PointCoordinateType GetDefaultCloudKernelSize(
        const ccHObject::Container& entities, unsigned knn = 12);

/*** CloudViewer standalone algorithms ***/

/**
 * @struct GeomCharacteristic
 * @brief Geometric characteristic with optional sub-parameter
 * 
 * Encapsulates a geometric feature to compute along with
 * algorithm-specific sub-options.
 */
struct GeomCharacteristic {
    /**
     * @brief Constructor
     * @param c Geometric characteristic type
     * @param option Sub-option for characteristic (default: 0)
     */
    GeomCharacteristic(
            cloudViewer::GeometricalAnalysisTools::GeomCharacteristic c,
            int option = 0)
        : charac(c), subOption(option) {}

    cloudViewer::GeometricalAnalysisTools::GeomCharacteristic charac;  ///< Characteristic type
    int subOption;  ///< Algorithm-specific sub-option
};

/// Collection of geometric characteristics
typedef std::vector<GeomCharacteristic> GeomCharacteristicSet;

/**
 * @brief Compute multiple geometric characteristics
 * 
 * Computes several geometric features on entities in a single pass
 * for efficiency. Each result is stored as a scalar field.
 * 
 * @param characteristics Set of characteristics to compute
 * @param radius Neighborhood radius
 * @param entities Entities to process
 * @param roughnessUpDir Up direction for roughness computation (optional)
 * @param parent Parent widget for progress dialogs
 * @return true if all computations succeeded
 */
bool ComputeGeomCharacteristics(const GeomCharacteristicSet& characteristics,
                                PointCoordinateType radius,
                                ccHObject::Container& entities,
                                const CCVector3* roughnessUpDir = nullptr,
                                QWidget* parent = nullptr);

/**
 * @brief Compute single geometric characteristic
 * 
 * Computes one geometric feature on entities. Result stored as scalar field.
 * 
 * @param algo Geometric characteristic to compute
 * @param subOption Algorithm-specific sub-option
 * @param radius Neighborhood radius
 * @param entities Entities to process
 * @param roughnessUpDir Up direction for roughness (optional)
 * @param parent Parent widget for dialogs
 * @param progressDialog Custom progress dialog (optional)
 * @return true if computation succeeded
 * 
 * @see cloudViewer::GeometricalAnalysisTools::GeomCharacteristic
 */
bool ComputeGeomCharacteristic(
        cloudViewer::GeometricalAnalysisTools::GeomCharacteristic algo,
        int subOption,
        PointCoordinateType radius,
        ccHObject::Container& entities,
        const CCVector3* roughnessUpDir = nullptr,
        QWidget* parent = nullptr,
        ecvProgressDialog* progressDialog = nullptr);

/**
 * @brief CloudViewer core library algorithms
 * 
 * Enumeration of algorithms from CloudViewer core library
 * that can be applied via ApplyCCLibAlgorithm().
 */
enum CC_LIB_ALGORITHM {
    CCLIB_ALGO_SF_GRADIENT,  ///< Scalar field gradient computation
};

/**
 * @brief Apply CloudViewer core algorithm
 * 
 * Applies one of the standard CloudViewer library algorithms
 * to a set of entities with progress tracking.
 * 
 * @param algo Algorithm to apply
 * @param entities Entities to process
 * @param parent Parent widget for dialogs
 * @param additionalParameters Algorithm-specific parameters (optional)
 * @return true if algorithm completed successfully
 */
bool ApplyCCLibAlgorithm(CC_LIB_ALGORITHM algo,
                         ccHObject::Container& entities,
                         QWidget* parent = 0,
                         void** additionalParameters = 0);

/**
 * @brief Scale matching algorithms
 * 
 * Methods for aligning entities by matching their scales.
 */
enum ScaleMatchingAlgorithm {
    BB_MAX_DIM,   ///< Match by bounding box maximum dimension
    BB_VOLUME,    ///< Match by bounding box volume
    PCA_MAX_DIM,  ///< Match by PCA maximum dimension
    ICP_SCALE     ///< Match scale using ICP with scale estimation
};

/**
 * @brief Apply scale matching algorithm
 * 
 * Aligns entities by adjusting their scales to match a reference entity.
 * Useful for comparing objects at different scales or from different sources.
 * 
 * @param algo Scale matching algorithm
 * @param entities Entities to align (first is typically reference)
 * @param icpRmsDiff ICP RMS difference threshold (for ICP_SCALE)
 * @param icpFinalOverlap ICP final overlap percentage (for ICP_SCALE)
 * @param refEntityIndex Index of reference entity (default: 0)
 * @param parent Parent widget for dialogs
 * @return true if alignment succeeded
 */
bool ApplyScaleMatchingAlgorithm(ScaleMatchingAlgorithm algo,
                                 ccHObject::Container& entities,
                                 double icpRmsDiff,
                                 int icpFinalOverlap,
                                 unsigned refEntityIndex = 0,
                                 QWidget* parent = 0);
}  // namespace ccLibAlgorithms
