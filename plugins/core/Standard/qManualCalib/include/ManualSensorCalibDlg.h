// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvColorTypes.h>
#include <ecvOverlayDialog.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <QFutureWatcher>
#include <QTimer>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "BagAlignment.h"
#include "BatchExportTypes.h"
#include "BevRemapBackend.h"
#include "CalibTypes.h"

namespace mcalib {
class RosBagReader;
class BirdEyeView;
struct BagMessage;
}  // namespace mcalib

class ccImage;
class ccPointCloud;
class ecvMainAppInterface;
class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QProgressDialog;
class QPushButton;
class QSlider;
class QCheckBox;
class QVBoxLayout;

class ManualSensorCalibDlg : public ccOverlayDialog {
    Q_OBJECT

    struct SliderFrameData {
        std::map<std::string, cv::Mat> images;
        std::map<std::string, int64_t> imageStampsUs;
        std::vector<Eigen::Vector3f> pointCloud;
        std::vector<mcalib::PointXYZIRT> pointCloudRaw;
        int64_t cloudStampUs = 0;
        bool isCombinedCloud = true;
        std::string cloudFrameId;
        mcalib::VehicleStateData vehicleState;
        bool attemptedCloudLoad = false;
    };

public:
    explicit ManualSensorCalibDlg(ecvMainAppInterface* app,
                                  QWidget* parent = nullptr);
    ~ManualSensorCalibDlg() override;

    bool linkWith(QWidget* win) override;
    bool start() override;
    void stop(bool accepted) override;

protected:
    void closeEvent(QCloseEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void onLoadConfig();
    void onLoadBag();
    void onSaveConfig();
    void onSavePCD();
    void onExportImage();
    void onBatchExportImages();
    void onBatchExportPCD();
    void onExportBevBatch();
    void onResetParams();

    void onSensorTypeChanged(int index);
    void onSensorNameChanged(const QString& name);
    void onViewModeChanged(int index);
    void onSpeedChanged(int index);
    void onTimeSliderChanged(int value);
    void onTimeStepBack();
    void onTimeStepForward();

    void onRollAdd();
    void onRollSub();
    void onPitchAdd();
    void onPitchSub();
    void onYawAdd();
    void onYawSub();
    void onXAdd();
    void onXSub();
    void onYAdd();
    void onYSub();
    void onZAdd();
    void onZSub();
    void onExtrinsicEditCommitted();

    void onPointSizeChanged(int value);
    void onDistFilterChanged(int value);
    void onDistFilterMinChanged(int value);
    void onGroundFilterChanged(int value);
    void onGroundFilterMaxChanged(int value);
    void onLidarCheckboxToggled();
    void onCalibModeChanged(const QString& mode);
    void onBevRemapBackendChanged(int index);

private:
    void setupUI();
    void updateFusionView();
    void initCameraDeltaExtrinsics();
    void updateDeltaExtrinsic(const mcalib::Vector6d& delta,
                              const std::string& sensor_name);
    void displayDeltaExtrinsic(const std::string& sensor_name);
    void syncCurrentSensorFromCombo();
    void updateSensorList();
    void refreshCalibModeCombo();
    void createLidarCheckboxes();

    void loadImagesFromBag(double percent);
    void loadPointCloudFromBag(double percent);
    void loadAlignedFrameFromBag(double percent);
    void loadInitialBagFrame();
    bool seekFirstValidBagFrame(double& out_percent, double search_step) const;
    void syncBagSliderToPercent(double percent);
    void loadCloudForCurrentTimestamp();
    bool loadCloudAtBagPercent(double percent);
    void applyCloudFrameMetadata(const std::string& frame_id);

    void buildBagTopicLists();
    void startBackgroundBagIndex();
    void updateTimeSliderLabel(int sliderValue);
    void processSliderLoad();
    void reloadCurrentBagFrame();
    int bagTimeStepDelta() const;
    void applySliderFrameData(
            const std::map<std::string, cv::Mat>& images,
            const std::map<std::string, int64_t>& imageStampsUs,
            const std::vector<Eigen::Vector3f>& pointCloud,
            const std::vector<mcalib::PointXYZIRT>& pointCloudRaw,
            int64_t cloudStampUs,
            bool isCombinedCloud = true,
            const std::string& cloudFrameId = {},
            const mcalib::VehicleStateData& vehicleState = {},
            bool attemptedCloudLoad = false);
    bool needsImagesForView() const;
    bool needsCloudForView() const;
    std::vector<std::pair<std::string, std::string>> getCameraTopicsForSlider()
            const;
    std::vector<std::pair<std::string, std::string>>
    getCameraTopicsForAlignment() const;
    void collectBevImageTopicGroups(std::vector<std::string>& svm_topics,
                                    std::vector<std::string>& avm_topics) const;
    void applyAlignedTopicImages(
            const std::map<std::string, cv::Mat>& images_by_topic,
            const std::map<std::string, int64_t>& stamps_ns);
    bool isCombinedCloud() const;
    std::vector<Eigen::Isometry3d> getLidarFinalExtrinsic() const;
    void loadVehicleTrajectory();

    void lidarCamFusion(const std::vector<Eigen::Vector3f>& cloud,
                        cv::Mat& img,
                        const std::string& camera_name,
                        const Eigen::Isometry3d& T_cam_sensing,
                        const cv::Mat& K,
                        const cv::Mat& D,
                        int radius,
                        mcalib::BevRemapMode backend_mode);
    mcalib::BevRemapMode getGpuBackendMode() const;

    void updateBevView();
    void updateLidarProjView();
    void updateLidarSingleFrameView();
    void updateLidarMultiFrameView();

    void enter2DImageView();
    void enter3DPointCloudView();
    void updateViewModeButtonStyle(int activeIndex);

    std::vector<Eigen::Vector3f> getFilteredPointCloud() const;
    std::map<std::string, Eigen::Isometry3d> getFinalExtrinsics() const;

    ccPointCloud* createPointCloudEntity(
            const std::vector<Eigen::Vector3f>& points,
            const QString& name) const;
    void exportPointCloudToDB();
    void exportImageToDB();
    void updateExportButtonStates();
    void updatePointSizeSliderLabel();
    void cleanupPreviewEntities();

    bool prepareBevCameraSubset();
    std::map<std::string, std::string> getBevCameraSlotMap() const;
    void updateCameraConfigForBev();
    void drawBevMeasureOverlay(cv::Mat& bev);
    cv::Point mapMouseToBevDisp(const QPoint& qp) const;
    cv::Point dispToBevUnrotate(const cv::Point& disp_pt) const;
    cv::Point bevToDispRotate(const cv::Point& bev_pt) const;
    void exportBevImages(const std::string& output_dir, int num_samples = 20);
    void startBatchExportJob(
            const QString& title,
            QPushButton* trigger_button,
            const QString& output_dir,
            std::function<mcalib::BatchExportResult(
                    mcalib::BatchExportProgress&)> task,
            const QString& item_label = QStringLiteral("images"));
    cv::Mat renderBevImage();
    cv::Mat renderLidarProjImage(
            std::vector<Eigen::Vector3f>* out_points = nullptr);
    std::vector<Eigen::Vector3f> getDisplayedPointCloudForExport() const;
    ccImage* createImageEntityFromMat(const cv::Mat& bgr,
                                      const QString& name) const;

    void displayImageInViewer(const cv::Mat& img);
    cv::Mat makeBevPlaceholderCanvas(const QString& title,
                                     const QString& subtitle) const;
    void showIdleBevCanvas();
    void displayPointCloudIn3D(const std::vector<Eigen::Vector3f>& points,
                               const std::vector<ecvColor::Rgb>& colors = {});
    void zoomToPreviewCloud();

    int getLidarIndexByName(const std::string& name) const;

    ecvMainAppInterface* m_app;

    mcalib::VehicleCalibConfig m_calibConfig;
    std::unique_ptr<mcalib::RosBagReader> m_bagReader;
    std::unique_ptr<mcalib::BirdEyeView> m_bevViewer;

    std::map<std::string, cv::Mat> m_images;
    std::vector<Eigen::Vector3f> m_pointCloud;
    std::vector<mcalib::PointXYZIRT> m_pointCloudRaw;
    std::deque<std::pair<int64_t, Eigen::Isometry3d>> m_vehiclePoses;
    int64_t m_cloudStampUs = 0;
    bool m_isCombinedCloud = true;
    std::string m_cloudFrameId;
    mcalib::VehicleStateData m_vehicleState;
    std::map<std::string, int64_t> m_imageStampsUs;

    enum CalibMode {
        MODE_SINGLE_CAMERA,
        MODE_ALL_CAMERA,
        MODE_AVM_CAMERA,
        MODE_SVM_CAMERA,
        MODE_SINGLE_LIDAR,
        MODE_ALL_LIDAR
    };

    std::string m_currentSensor;
    std::string m_sensorType = "camera";
    CalibMode m_calibMode = MODE_SINGLE_CAMERA;

    std::map<std::string, mcalib::Vector6d> m_deltaExtrinsics;
    std::vector<mcalib::Vector6d> m_deltaLidarExtrinsics;
    mcalib::Vector6d m_deltaExtrinsic = mcalib::Vector6d::Zero();
    double m_rotResolution = 1.0;
    double m_posResolution = 0.1;
    int m_pointSize = 2;
    int m_viewMode = 0;
    double m_distFilter = 100.0;
    double m_distFilterMin = 0.0;
    double m_groundFilterMin = -5.0;
    double m_groundFilterMax = 30.0;

    std::vector<bool> m_selectedLidars;

    cv::Size m_lastBevDisplaySize;
    cv::Size m_lastBevOriginalSize;
    cv::Size m_lastBevPreRotateSize;
    bool m_lastBevRotatedCw = true;
    cv::Mat m_lastExportImage;

    std::vector<cv::Point3f> m_bevMeasureGroundPts;
    std::vector<cv::Point> m_bevMeasureDispPts;
    double m_bevMeasureBevPxDist = 0.0;

    QComboBox* m_cmbSensorType = nullptr;
    QComboBox* m_cmbSensorName = nullptr;
    QComboBox* m_cmbViewMode = nullptr;
    QComboBox* m_cmbSpeed = nullptr;
    QComboBox* m_cmbCalibMode = nullptr;
    QComboBox* m_cmbBevRemapBackend = nullptr;

    QLineEdit* m_editRoll = nullptr;
    QLineEdit* m_editPitch = nullptr;
    QLineEdit* m_editYaw = nullptr;
    QLineEdit* m_editX = nullptr;
    QLineEdit* m_editY = nullptr;
    QLineEdit* m_editZ = nullptr;

    QSlider* m_sliderPointSize = nullptr;
    QSlider* m_sliderTimePos = nullptr;
    QPushButton* m_btnTimeStepBack = nullptr;
    QPushButton* m_btnTimeStepForward = nullptr;
    QSlider* m_sliderDistFilter = nullptr;
    QSlider* m_sliderDistFilterMin = nullptr;
    QSlider* m_sliderGroundFilter = nullptr;
    QSlider* m_sliderGroundFilterMax = nullptr;
    QLabel* m_lblStatus = nullptr;
    QLabel* m_lblTimePos = nullptr;
    QLabel* m_lblPointSizeTitle = nullptr;
    QLabel* m_lblPointSize = nullptr;
    QLabel* m_lblDistRange = nullptr;
    QLabel* m_lblGroundRange = nullptr;
    QPushButton* m_btnBevView = nullptr;
    QPushButton* m_btnLidarProjView = nullptr;
    QPushButton* m_btnSingleFrameView = nullptr;
    QPushButton* m_btnMultiFrameView = nullptr;
    QPushButton* m_btnExportImage = nullptr;
    QPushButton* m_btnBatchExportImages = nullptr;
    QPushButton* m_btnExportPCD = nullptr;
    QPushButton* m_btnBatchExportPCD = nullptr;
    std::vector<QCheckBox*> m_lidarCheckboxes;
    QGroupBox* m_lidarGroup = nullptr;
    QVBoxLayout* m_lidarGroupLayout = nullptr;

    QString m_configPath;
    QString m_bagPath;
    bool m_configLoaded = false;
    bool m_bagIndexReady = false;
    double m_bagDurationSec = 0.0;
    mcalib::BagImageEncoding m_bagImageEncoding =
            mcalib::BagImageEncoding::Unknown;

    std::vector<std::pair<std::string, std::string>> m_cameraTopics;
    std::vector<std::string> m_cloudTopics;
    QFutureWatcher<void>* m_bagIndexWatcher = nullptr;

    QPushButton* m_btnLoadBag = nullptr;
    QPushButton* m_btnLoadConfig = nullptr;

    QTimer* m_sliderDebounce = nullptr;
    QTimer* m_adjustDebounce = nullptr;
    QFutureWatcher<SliderFrameData>* m_sliderLoadWatcher = nullptr;
    QFutureWatcher<mcalib::BatchExportResult>* m_batchExportWatcher = nullptr;
    QTimer* m_batchExportPollTimer = nullptr;
    int m_pendingSliderValue = 0;
    int m_appliedSliderValue = -1;
    bool m_sliderReloadPending = false;
    std::atomic<bool> m_loadingBag{false};
    std::atomic<int> m_sliderLoadGeneration{0};
    /// Incremented on every calib-mode / view-mode / sensor-name change.
    /// Slider load completions capture this and skip applying stale results
    /// (e.g. SVM images arriving after the user switched to AVM mode).
    std::atomic<int> m_modeGeneration{0};
    bool m_extrinsicDirty = true;

    void scheduleViewUpdate();

    ccImage* m_vtkImage = nullptr;
    ccPointCloud* m_vtkCloud = nullptr;
};
