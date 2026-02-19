// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cloudViewer
#include <RegistrationTools.h>

// CV_DB_LIB
#include <ecvGLMatrix.h>

class QWidget;
#include <QStringList>  // QStringList is a type alias in Qt6, cannot forward declare
class ccHObject;

/**
 * @class ccRegistrationTools
 * @brief High-level wrapper for point cloud registration
 *
 * Provides application-level interface to CloudViewer's registration
 * algorithms. Handles progress dialogs, error reporting, and automatic
 * mesh sampling for registration operations.
 *
 * Main features:
 * - ICP (Iterative Closest Point) registration
 * - Automatic point sampling from meshes
 * - Scalar field-based weighting
 * - Scale estimation support
 * - Progress tracking and cancellation
 *
 * @see cloudViewer::ICPRegistrationTools
 * @see cloudViewer::RegistrationTools
 */
class ccRegistrationTools {
public:
    /**
     * @brief Apply ICP registration between two entities
     *
     * Aligns 'data' entity to 'model' entity using ICP algorithm.
     * Automatically samples points from meshes if needed.
     *
     * @param data Data entity to be aligned
     * @param model Reference model entity
     * @param transMat Output transformation matrix
     * @param finalScale Output scale factor (if scale estimation enabled)
     * @param finalRMS Output RMS error after registration
     * @param finalPointCount Output number of points used in final iteration
     * @param inputParameters ICP algorithm parameters
     * @param useDataSFAsWeights Use data scalar field as weights (default:
     * false)
     * @param useModelSFAsWeights Use model scalar field as weights (default:
     * false)
     * @param parent Parent widget for progress dialogs
     * @return true if registration succeeded
     *
     * @warning Automatically samples points on meshes for ICP
     * @see cloudViewer::ICPRegistrationTools::Parameters
     */
    static bool
    ICP(ccHObject* data,
        ccHObject* model,
        ccGLMatrix& transMat,
        double& finalScale,
        double& finalRMS,
        unsigned& finalPointCount,
        const cloudViewer::ICPRegistrationTools::Parameters& inputParameters,
        bool useDataSFAsWeights = false,
        bool useModelSFAsWeights = false,
        QWidget* parent = nullptr);
};
