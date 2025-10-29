// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVGeom.h>

#include "eCV_db.h"

// QT
#include <QFile>
#include <QObject>

class ecvGenericVisualizer3D;
class ECV_DB_LIB_API ecvGenericCameraTool : public QObject {
    Q_OBJECT
public:
    ecvGenericCameraTool();
    ~ecvGenericCameraTool() override;

    struct ECV_DB_LIB_API CameraInfo {
        CCVector3d position;
        CCVector3d focal;
        CCVector3d viewUp;
        CCVector3d pivot;
        CCVector2d clippRange;
        double rotationFactor;
        double viewAngle;
        double eyeAngle;

        CameraInfo()
            : clippRange(CCVector2d(0.0, 0.0)),
              position(CCVector3d(0.0, 0.0, 0.0)),
              focal(CCVector3d(0.0, 0.0, 0.0)),
              viewUp(CCVector3d(0.0, 0.0, 0.0)),
              pivot(CCVector3d(0.0, 0.0, 0.0)),
              rotationFactor(1.0),
              viewAngle(1.0),
              eyeAngle(1.0) {}

        std::string toString() {
            QStringList cameraInfo;
            const int precision = 4;
            cameraInfo << QString::number(position.x, 'f', precision)
                       << QString::number(position.y, 'f', precision)
                       << QString::number(position.z, 'f', precision)
                       << QString::number(focal.x, 'f', precision)
                       << QString::number(focal.y, 'f', precision)
                       << QString::number(focal.z, 'f', precision)
                       << QString::number(viewUp.x, 'f', precision)
                       << QString::number(viewUp.y, 'f', precision)
                       << QString::number(viewUp.z, 'f', precision)
                       << QString::number(pivot.x, 'f', precision)
                       << QString::number(pivot.y, 'f', precision)
                       << QString::number(pivot.z, 'f', precision)
                       << QString::number(rotationFactor, 'f', precision)
                       << QString::number(viewAngle, 'f', precision)
                       << QString::number(eyeAngle, 'f', precision)
                       << QString::number(clippRange.x, 'f', precision)
                       << QString::number(clippRange.y, 'f', precision);

            return cameraInfo.join(",").toStdString();
        }

        QStringList parseConfig(const QString& info) {
            if (info.isEmpty()) {
                return QStringList();
            }

            return info.split(",");
        }

        void loadConfig(QString& info) {
            QStringList cameraInfo = parseConfig(info);

            assert(cameraInfo.size() == 17);
            position.x = cameraInfo[0].toDouble();
            position.y = cameraInfo[1].toDouble();
            position.z = cameraInfo[2].toDouble();
            focal.x = cameraInfo[3].toDouble();
            focal.y = cameraInfo[4].toDouble();
            focal.z = cameraInfo[5].toDouble();
            viewUp.x = cameraInfo[6].toDouble();
            viewUp.y = cameraInfo[7].toDouble();
            viewUp.z = cameraInfo[8].toDouble();
            pivot.x = cameraInfo[9].toDouble();
            pivot.y = cameraInfo[10].toDouble();
            pivot.z = cameraInfo[11].toDouble();
            rotationFactor = cameraInfo[12].toDouble();
            viewAngle = cameraInfo[13].toDouble();
            eyeAngle = cameraInfo[14].toDouble();
            clippRange.x = cameraInfo[15].toDouble();
            clippRange.y = cameraInfo[16].toDouble();
        }
    };

    static CameraInfo OldCameraParam;
    static CameraInfo CurrentCameraParam;
    static void SaveBuffer() { OldCameraParam = CurrentCameraParam; }

    virtual void saveCameraConfiguration(const std::string& file);
    virtual void loadCameraConfiguration(const std::string& file);

    virtual void resetViewDirection(double look_x,
                                    double look_y,
                                    double look_z,
                                    double up_x,
                                    double up_y,
                                    double up_z) {}

    virtual void setAutoPickPivotAtCenter(bool state);

    enum CameraAdjustmentType { Roll = 0, Elevation, Azimuth, Zoom };

    virtual void adjustCamera(CameraAdjustmentType enType, double value) = 0;
    virtual void updateCamera() = 0;
    virtual void updateCameraParameters() = 0;

public slots:
    void UpdateCamera() { updateCamera(); }
};
