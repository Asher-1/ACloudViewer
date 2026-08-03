// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFutureWatcher>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QtConcurrent>

#include "aicore/facedetect_capi.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#endif

#include <memory>
#include <vector>

class FaceCaptureWidget : public QWidget {
    Q_OBJECT
public:
    enum class CaptureAngle {
        Front,
        Left45,
        Right45,
        Left90,
        Right90,
        Up15,
        Down15
    };

    struct CapturedFrame {
        QImage image;
        QImage croppedFace;  // 512x512
        CaptureAngle angle;
        QRect faceRect;
        bool valid = false;
    };

    explicit FaceCaptureWidget(QWidget* parent = nullptr);
    ~FaceCaptureWidget() override;

    bool startCamera(int deviceIndex = 0);
    bool startVideoFile(const QString& path);
    void stopCamera();
    void stopCapture();
    bool isCameraActive() const;
    bool isCaptureActive() const;

    enum class InputSource { Camera, VideoFile };
    InputSource inputSource() const { return m_inputSource; }
    int selectedCameraIndex() const;
    QString videoFilePath() const;

    void startGuidedCapture(const std::vector<CaptureAngle>& angles);
    void captureCurrentFrame();
    void resetCapture();

    std::vector<CapturedFrame> capturedFrames() const;
    QStringList exportCapturedImages(const QString& outputDir) const;

    int capturedCount() const;
    int targetCount() const;

    enum class FacePickStrategy {
        LargestFace,
        HighestScore,
        TrackSamePerson,
    };

    float minDetectionScore() const;
    int minCapturesBeforeComplete() const;
    FacePickStrategy facePickStrategy() const;

    /** Stop stream and free GGUF face detector (dialog close / GPU cleanup). */
    void releaseGpuResources();

    void refreshDetectorList();

    static bool isAvailable();

signals:
    void cameraStarted();
    void cameraStopped();
    void faceDetected(const QRect& bbox);
    void faceNotDetected();
    void frameCaptured(int index, int total);
    void captureComplete();
    void cameraError(const QString& error);
    void logMessage(const QString& message);

private slots:
    void processFrame();
    void onDetectorComboChanged(int index);
    void onSourceChanged(int index);
    void onBrowseVideoFile();

private:
    enum class DetectorKind { None, OpenCV, Ggml };

    void setupUi();
    void populateDetectorCombo();
    bool loadCascade();
    bool ensureGgmlModelReady();
    void startModelDownload(const aicore_facedetect_model_entry* model);
    void releaseGgmlModel();
    bool loadGgmlModel(const QString& path);
    void scheduleGgmlModelLoad(const QString& path);
    QString currentGgmlFilename() const;
    bool detectorReady() const;
    QString angleToString(CaptureAngle angle) const;
    void refreshCapturedGallery();
    void updateCaptureProgressUi();
    int currentGuideAngleIndex() const;
    void loadFaceCaptureSettings();
    void saveFaceCaptureSettings();

#ifdef HAS_OPENCV_FACE_CAPTURE
    struct ScoredFace {
        cv::Rect rect;
        float score = 0.f;
    };

    QImage cvMatToQImage(const cv::Mat& mat);
    std::vector<ScoredFace> detectFacesOpenCv(const cv::Mat& frame);
    std::vector<ScoredFace> detectFacesGgml(const cv::Mat& frame);
    std::vector<ScoredFace> detectFaces(const cv::Mat& frame);
    cv::Rect pickFace(const cv::Mat& frame,
                      const std::vector<ScoredFace>& faces);
    bool embedFaceCrop(const cv::Mat& frame,
                       const cv::Rect& rect,
                       std::vector<float>* embedding);
    float embeddingDistance(const std::vector<float>& a,
                            const std::vector<float>& b) const;
    void resetIdentityTrack();
    cv::Rect detectFaceOpenCv(const cv::Mat& frame);
    cv::Rect detectFaceGgml(const cv::Mat& frame);
    cv::Rect detectFace(const cv::Mat& frame);
    QImage cropAndResizeFace(const cv::Mat& frame,
                             const cv::Rect& faceRect,
                             int targetSize = 512);
    void drawOverlay(QImage& image, const cv::Rect& faceRect);
    void drawAngleGuide(QImage& image, CaptureAngle angle);

    cv::VideoCapture m_camera;
    cv::CascadeClassifier m_faceCascade;
    cv::Rect m_lastFaceRect;
    float m_lastFaceScore = 0.f;
#endif

    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QLabel* m_captureProgressLabel = nullptr;
    QProgressBar* m_captureProgress = nullptr;
    QWidget* m_capturedGalleryRow = nullptr;
    QLabel* m_angleLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_downloadProgress = nullptr;
    QPushButton* m_captureBtn = nullptr;
    QComboBox* m_cameraCombo = nullptr;
    QComboBox* m_sourceCombo = nullptr;
    QWidget* m_cameraControlsRow = nullptr;
    QWidget* m_videoFileRow = nullptr;
    QLineEdit* m_videoPathEdit = nullptr;
    QPushButton* m_browseVideoBtn = nullptr;
    QComboBox* m_detectorCombo = nullptr;
    QDoubleSpinBox* m_minScoreSpin = nullptr;
    QSpinBox* m_minCapturesSpin = nullptr;
    QComboBox* m_faceStrategyCombo = nullptr;

    std::vector<float> m_referenceEmbedding;
    bool m_hasReferenceEmbedding = false;
    static constexpr float kSamePersonMaxDistance = 0.55f;

    ecvModelDownloader* m_downloader = nullptr;
    aicore_facedetect_ctx* m_ggmlCtx = nullptr;
    QString m_loadedGgmlPath;

    QTimer* m_frameTimer = nullptr;
    bool m_cameraActive = false;
    InputSource m_inputSource = InputSource::Camera;
    QString m_videoFilePath;
    bool m_cascadeLoaded = false;
    bool m_camerasEnumerated = false;
    bool m_downloadInProgress = false;
    bool m_autoStartAfterDownload = false;
    int m_pendingCameraIndex = 0;
    int m_ggmlFrameSkip = 0;
    bool m_ggmlModelLoading = false;
    QFutureWatcher<aicore_facedetect_ctx*>* m_ggmlLoadWatcher = nullptr;

    DetectorKind m_detectorKind = DetectorKind::OpenCV;

    std::vector<CaptureAngle> m_targetAngles;
    std::vector<CapturedFrame> m_capturedFrames;
    int m_currentAngleIndex = 0;
    bool m_capturingMode = false;
    int m_consecutiveDetections = 0;
    int m_postCaptureCooldown = 0;
    int m_noCascadeCounter = 0;
    static constexpr int kAutoCaptureTrigger = 20;
    static constexpr int kVideoAutoCaptureTrigger = 8;
    static constexpr int kPostCaptureCooldown = 45;
    static constexpr int kNoCascadeAutoInterval = 90;
    static constexpr int kGgmlDetectInterval = 2;
};
