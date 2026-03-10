// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file EditCameraTool.h
 * @brief Camera editing tool for loading, saving, and adjusting 3D view.
 */

#include <ecvGenericCameraTool.h>

#include <QObject>

#include "qVTK.h"

/**
 * @class EditCameraTool
 * @brief Tool for editing camera configuration and view direction.
 */
class QVTK_ENGINE_LIB_API EditCameraTool : public ecvGenericCameraTool {
    Q_OBJECT
public:
    /// @param viewer 3D visualizer to attach to.
    EditCameraTool(ecvGenericVisualizer3D* viewer);
    ~EditCameraTool() override;

    static void UpdateCameraInfo();
    static void UpdateCamera();
    /// @param viewer Visualizer to set for static camera operations.
    static void SetVisualizer(ecvGenericVisualizer3D* viewer);

private slots:
    // Description:
    // Choose a file and load/save camera properties.
    virtual void saveCameraConfiguration(const std::string& file) override;
    virtual void loadCameraConfiguration(const std::string& file) override;

    virtual void resetViewDirection(double look_x,
                                    double look_y,
                                    double look_z,
                                    double up_x,
                                    double up_y,
                                    double up_z) override;

    virtual void updateCamera() override;
    virtual void updateCameraParameters() override;

private:
    virtual void adjustCamera(CameraAdjustmentType enType,
                              double value) override;
};
