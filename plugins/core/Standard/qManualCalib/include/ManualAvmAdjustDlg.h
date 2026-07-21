// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvOverlayDialog.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <QFutureWatcher>
#include <QTimer>
#include <atomic>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "CalibTypes.h"
#include "CameraModel.h"

namespace mcalib {
class RosBagReader;
}

class ccImage;
class ecvMainAppInterface;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;

class ManualAvmAdjustDlg : public ccOverlayDialog {
    Q_OBJECT

public:
    explicit ManualAvmAdjustDlg(ecvMainAppInterface* app,
                                QWidget* parent = nullptr);
    ~ManualAvmAdjustDlg() override;

    bool linkWith(QWidget* win) override;
    bool start() override;
    void stop(bool accepted) override;

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onLoadConfig();
    void onLoadBag();
    void onLoadImage();
    void onSaveParam();
    void onLoadParam();

    void onCameraChanged(int index);
    void onAvmModeChanged(int index);
    void onBaseRotChanged(int index);
    void onTimeSliderChanged(int value);
    void onTimeStepBack();
    void onTimeStepForward();

    void onVirtualK2Changed(double value);
    void onScaleChanged(double value);
    void onV0OffsetChanged(double value);
    void onImgWidthChanged(double value);
    void onImgHeightChanged(double value);
    void onFocalXChanged(double value);
    void onFocalYChanged(double value);
    void onRotXChanged(double value);
    void onRotYChanged(double value);
    void onRotZChanged(double value);
    void onRectXChanged(double value);
    void onRectYChanged(double value);
    void onRectWidthChanged(double value);
    void onRectHeightChanged(double value);

private:
    void setupUI();
    void updateImage();
    void displayImageInViewer(const cv::Mat& img);
    QWidget* createParamRow(const QString& label,
                            QDoubleSpinBox*& spinBox,
                            double min,
                            double max,
                            double value,
                            double step,
                            int decimals);

    void loadImageFromBag(double percent);
    void processSliderLoad();
    void updateTimeSliderLabel(int sliderValue);
    int bagTimeStepDelta() const;
    void startBackgroundBagIndex();
    std::string getCameraTopic() const;
    void drawRectOverlay(cv::Mat& img) const;

    ecvMainAppInterface* m_app;

    mcalib::VehicleCalibConfig m_calibConfig;
    mcalib::CameraSystem m_cameraSystem;
    std::unique_ptr<mcalib::RosBagReader> m_bagReader;

    std::map<std::string, Eigen::Isometry3d> m_cameraExtrinsics;

    std::string m_currentCamera;
    cv::Mat m_sourceImage;
    QString m_configPath;
    QString m_bagPath;

    std::string m_avmMode = "small_single_view";
    std::string m_baseRot = "rot_front_single";

    std::map<std::string, Eigen::Matrix3d> m_baseRotMap;

    double m_virtualK2 = 0.16;
    double m_targetScale = 1.5;
    double m_v0Offset = 0.03;
    int m_imgWidth = 1014;
    int m_imgHeight = 966;
    int m_focalX = 400;
    int m_focalY = 400;
    double m_rotX = 0;
    double m_rotY = 0;
    double m_rotZ = 0;
    int m_rectX = 0;
    int m_rectY = 0;
    int m_rectWidth = 1014;
    int m_rectHeight = 966;

    bool m_blockUpdate = false;
    bool m_configLoaded = false;

    QPushButton* m_btnLoadBag = nullptr;
    QPushButton* m_btnLoadConfig = nullptr;
    QPushButton* m_btnTimeStepBack = nullptr;
    QPushButton* m_btnTimeStepForward = nullptr;

    QComboBox* m_cmbCamera = nullptr;
    QComboBox* m_cmbAvmMode = nullptr;
    QComboBox* m_cmbBaseRot = nullptr;

    QDoubleSpinBox* m_spinVirtualK2 = nullptr;
    QDoubleSpinBox* m_spinScale = nullptr;
    QDoubleSpinBox* m_spinV0Offset = nullptr;
    QDoubleSpinBox* m_spinImgWidth = nullptr;
    QDoubleSpinBox* m_spinImgHeight = nullptr;
    QDoubleSpinBox* m_spinFocalX = nullptr;
    QDoubleSpinBox* m_spinFocalY = nullptr;
    QDoubleSpinBox* m_spinRotX = nullptr;
    QDoubleSpinBox* m_spinRotY = nullptr;
    QDoubleSpinBox* m_spinRotZ = nullptr;
    QDoubleSpinBox* m_spinRectX = nullptr;
    QDoubleSpinBox* m_spinRectY = nullptr;
    QDoubleSpinBox* m_spinRectWidth = nullptr;
    QDoubleSpinBox* m_spinRectHeight = nullptr;

    QSlider* m_sliderTimePos = nullptr;
    QLabel* m_lblStatus = nullptr;
    QLabel* m_lblTimePos = nullptr;

    QTimer* m_sliderDebounce = nullptr;
    QFutureWatcher<cv::Mat>* m_sliderLoadWatcher = nullptr;
    QFutureWatcher<void>* m_bagIndexWatcher = nullptr;
    int m_pendingSliderValue = 0;
    int m_appliedSliderValue = -1;
    bool m_sliderReloadPending = false;
    std::atomic<bool> m_loadingBag{false};
    std::atomic<int> m_sliderLoadGeneration{0};
    double m_bagDurationSec = 0.0;
    bool m_bagIndexReady = false;

    ccImage* m_vtkImage = nullptr;
};
