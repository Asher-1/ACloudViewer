// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvGenericCameraTool.h>

#include <QObject>

#include "qPCL.h"

class QPCL_ENGINE_LIB_API EditCameraTool : public ecvGenericCameraTool {
    Q_OBJECT
public:
    EditCameraTool(ecvGenericVisualizer3D* viewer);
    ~EditCameraTool() override;

    static void UpdateCameraInfo();
    static void UpdateCamera();
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
