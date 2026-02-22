// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "DgmOctree.h"
#include "Neighbourhood.h"

namespace cloudViewer {

class GenericProgressCallback;
class GenericCloud;
class ScalarField;

/**
 * @class GeometricalAnalysisTools
 * @brief Algorithms for computing point cloud geometric characteristics
 *
 * Provides various algorithms to compute geometric properties of point clouds
 * including curvature, density, roughness, and other local features. These
 * tools are essential for point cloud analysis, segmentation, and
 * classification.
 *
 * @see Neighbourhood
 * @see DgmOctree
 */
class CV_CORE_LIB_API GeometricalAnalysisTools : public CVToolbox {
public:
    /**
     * @brief Geometric characteristics that can be computed
     */
    enum GeomCharacteristic {
        Feature,    ///< Geometric feature (@see Neighbourhood::GeomFeature)
        Curvature,  ///< Surface curvature (@see Neighbourhood::CurvatureType)
        LocalDensity,        ///< Accurate local density (@see Density)
        ApproxLocalDensity,  ///< Approximate local density (@see Density)
        Roughness,           ///< Surface roughness
        MomentOrder1         ///< First order moment
    };

    /**
     * @brief Density measurement methods
     */
    enum Density {
        DENSITY_KNN = 1,  ///< Number of points in neighborhood sphere
        DENSITY_2D,       ///< Points / circle area (2D projection)
        DENSITY_3D,       ///< Points / sphere volume (3D)
    };

    /**
     * @brief Error codes for analysis operations
     */
    enum ErrorCode {
        NoError = 0,                   ///< Success
        InvalidInput = -1,             ///< Invalid input parameters
        NotEnoughPoints = -2,          ///< Insufficient points
        OctreeComputationFailed = -3,  ///< Octree creation failed
        ProcessFailed = -4,            ///< Processing failed
        UnhandledCharacteristic = -5,  ///< Unsupported characteristic
        NotEnoughMemory = -6,          ///< Out of memory
        ProcessCancelledByUser = -7    ///< User cancelled
    };
    /**
     * @brief Compute a geometric characteristic
     *
     * Unified method to compute various geometric properties. Once the main
     * characteristic is chosen, use subOption to specify details (e.g.,
     * specific feature type, curvature type, or density algorithm).
     *
     * @param c Geometric characteristic to compute
     * @param subOption Feature/curvature type or density algorithm (0 if N/A)
     * @param cloud Point cloud to analyze
     * @param kernelRadius Neighborhood sphere radius
     * @param roughnessUpDir Up direction for signed roughness (optional)
     * @param progressCb Progress callback (optional)
     * @param inputOctree Pre-computed octree (optional, computed if nullptr)
     * @return Error code (NoError on success)
     */
    static ErrorCode ComputeCharactersitic(
            GeomCharacteristic c,
            int subOption,
            GenericIndexedCloudPersist* cloud,
            PointCoordinateType kernelRadius,
            const CCVector3* roughnessUpDir = nullptr,
            GenericProgressCallback* progressCb = nullptr,
            DgmOctree* inputOctree = nullptr);

    /**
     * @brief Compute approximate local density
     *
     * Legacy method based only on distance to nearest neighbor.
     * @param cloud Point cloud to analyze
     * @param densityType Density measurement type
     * @param progressCb Progress callback (optional)
     * @param inputOctree Pre-computed octree (optional)
     * @return Error code (NoError on success)
     * @warning DENSITY_KNN corresponds to inverse nearest neighbor distance
     * @warning Assumes input and output scalar fields are different
     */
    static ErrorCode ComputeLocalDensityApprox(
            GenericIndexedCloudPersist* cloud,
            Density densityType,
            GenericProgressCallback* progressCb = nullptr,
            DgmOctree* inputOctree = nullptr);

    /**
     * @brief Compute the gravity center (centroid) of a point cloud
     * @param theCloud Input point cloud
     * @return Gravity center coordinates
     * @warning Uses the cloud's global iterator
     */
    static CCVector3 ComputeGravityCenter(const GenericCloud* theCloud);

    //! Computes the weighted gravity center of a point cloud
    /** \warning this method uses the cloud global iterator
            \param theCloud cloud
            \param weights per point weights (only absolute values are
    considered) \return gravity center
    **/
    static CCVector3 ComputeWeightedGravityCenter(GenericCloud* theCloud,
                                                  ScalarField* weights);

    //! Computes the cross covariance matrix between two clouds (same size)
    /** Used in the ICP algorithm between the cloud to register and the "Closest
    Points Set" determined from the reference cloud. \warning this method uses
    the clouds global iterators \param P the cloud to register \param Q the
    "Closest Point Set" \param pGravityCenter the gravity center of P \param
    qGravityCenter the gravity center of Q \return cross covariance matrix
    **/
    static SquareMatrixd ComputeCrossCovarianceMatrix(
            GenericCloud* P,
            GenericCloud* Q,
            const CCVector3& pGravityCenter,
            const CCVector3& qGravityCenter);

    //! Computes the cross covariance matrix between two clouds (same size) -
    //! weighted version
    /** Used in the ICP algorithm between the cloud to register and the "Closest
    Points Set" determined from the reference cloud. \warning this method uses
    the clouds global iterators \param P the cloud to register \param Q the
    "Closest Point Set" \param pGravityCenter the gravity center of P \param
    qGravityCenter the gravity center of Q \param coupleWeights weights for each
    (Pi,Qi) couple (optional) \return weighted cross covariance matrix
    **/
    static SquareMatrixd ComputeWeightedCrossCovarianceMatrix(
            GenericCloud* P,
            GenericCloud* Q,
            const CCVector3& pGravityCenter,
            const CCVector3& qGravityCenter,
            ScalarField* coupleWeights = nullptr);

    //! Computes the covariance matrix of a clouds
    /** \warning this method uses the cloud global iterator
            \param theCloud point cloud
            \param _gravityCenter if available, its gravity center
            \return covariance matrix
    **/
    static cloudViewer::SquareMatrixd ComputeCovarianceMatrix(
            const GenericCloud* theCloud,
            const PointCoordinateType* _gravityCenter = nullptr);

    //! Flag duplicate points
    /** This method only requires an output scalar field. Duplicate points will
    be associated to scalar value 1 (and 0 for the others). \param theCloud
    processed cloud \param minDistanceBetweenPoints min distance between
    (output) points \param progressCb client application can get some
    notification of the process progress through this callback mechanism (see
    GenericProgressCallback) \param inputOctree if not set as input, octree will
    be automatically computed. \return success (0) or error code (<0)
    **/
    static ErrorCode FlagDuplicatePoints(
            GenericIndexedCloudPersist* theCloud,
            double minDistanceBetweenPoints =
                    std::numeric_limits<double>::epsilon(),
            GenericProgressCallback* progressCb = nullptr,
            DgmOctree* inputOctree = nullptr);

    //! Tries to detect a sphere in a point cloud
    /** Inspired from "Parameter Estimation Techniques: A Tutorial with
    Application to Conic Fitting" by Zhengyou Zhang (Inria Technical Report
    2676). More specifically the section 9.5 about Least Median of Squares.
            \param[in]  cloud input cloud
            \param[in]  outliersRatio proportion of outliers (between 0 and 1)
            \param[out] center center of the detected sphere
            \param[out] radius radius of the detected sphere
            \param[out] rms residuals RMS for the detected sphere
            \param[in] progressCb for progress notification (optional)
            \param[in] confidence probability that the detected sphere is the
    right one (strictly below 1) \param[in] seed if different than 0, this seed
    will be used for random numbers generation (instead of a random one) \result
    success
    **/
    static ErrorCode DetectSphereRobust(
            GenericIndexedCloudPersist* cloud,
            double outliersRatio,
            CCVector3& center,
            PointCoordinateType& radius,
            double& rms,
            GenericProgressCallback* progressCb = nullptr,
            double confidence = 0.99,
            unsigned seed = 0);

    //! Computes the center and radius of a sphere passing through 4 points
    /** \param[in] A first point
            \param[in] B second point
            \param[in] C third point
            \param[in] D fourth point
            \param[out] center center of the sphere
            \param[out] radius radius of the sphere
            \return success
    **/
    static ErrorCode ComputeSphereFrom4(const CCVector3& A,
                                        const CCVector3& B,
                                        const CCVector3& C,
                                        const CCVector3& D,
                                        CCVector3& center,
                                        PointCoordinateType& radius);

    //! Detects a circle from a point cloud
    /** Based on "A Simple approach for the Estimation of Circular Arc Center
    and Its radius" by S. Thomas and Y. Chan \param[in]  cloud		point
    cloud \param[out] center		circle center \param[out] normal
    normal to the plane to which the circle belongs \param[out] radius
    circle radius \param[out] rms			fitting RMS \param[in]
    progressCb	for progress notification (optional) \return success
    **/
    static ErrorCode DetectCircle(
            GenericIndexedCloudPersist* cloud,
            CCVector3& center,
            CCVector3& normal,
            PointCoordinateType& radius,
            double& rms,
            GenericProgressCallback* progressCb = nullptr);

protected:
    //! Computes geom characteristic inside a cell
    /**	\param cell structure describing the cell on which processing is applied
            \param additionalParameters see method description
            \param nProgress optional (normalized) progress notification
    (per-point)
    **/
    static bool ComputeGeomCharacteristicAtLevel(
            const DgmOctree::octreeCell& cell,
            void** additionalParameters,
            NormalizedProgress* nProgress = nullptr);
    //! Computes approximate point density inside a cell
    /**	\param cell structure describing the cell on which processing is applied
            \param additionalParameters see method description
            \param nProgress optional (normalized) progress notification
    (per-point)
    **/
    static bool ComputeApproxPointsDensityInACellAtLevel(
            const DgmOctree::octreeCell& cell,
            void** additionalParameters,
            NormalizedProgress* nProgress = nullptr);

    //! Flags duplicate points inside a cell
    /**	\param cell structure describing the cell on which processing is applied
            \param additionalParameters see method description
            \param nProgress optional (normalized) progress notification
    (per-point)
    **/
    static bool FlagDuplicatePointsInACellAtLevel(
            const DgmOctree::octreeCell& cell,
            void** additionalParameters,
            NormalizedProgress* nProgress = nullptr);

    //! Refines the estimation of a sphere by (iterative) least-squares
    static bool RefineSphereLS(GenericIndexedCloudPersist* cloud,
                               CCVector3& center,
                               PointCoordinateType& radius,
                               double minReltaiveCenterShift = 1.0e-3);
};

}  // namespace cloudViewer
