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
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QtConcurrent>

#include "aicore/facedetect_capi.h"
#include "aicore/runtime_capi.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#endif

#include <memory>
#include <vector>

class QResizeEvent;

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

    struct IdentityImageBatch {
        QString id;
        QString name;
        QStringList paths;
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
    std::vector<IdentityImageBatch> exportCapturedIdentityImages(
            const QString& outputDir) const;

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

    /** The capture detector follows the parent reconstruction device choice. */
    void setInferenceDevice(const QString& device);
    QString inferenceDevice() const { return m_inferenceDevice; }
    void requestInferenceCancel();

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
    void onBrowseRegistry();
    void reloadRegistry();
    void filterRegistry(const QString& text);

private:
    void resizeEvent(QResizeEvent* event) override;

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
    void setAngleGuideText(const QString& text);
    int currentGuideAngleIndex() const;
    void loadFaceCaptureSettings();
    void saveFaceCaptureSettings();
    bool configureDetectorForRegistrySelection();
    static std::vector<float> normalizeEmbedding(
            const std::vector<float>& embedding);

    struct RegistryIdentity {
        QString id;
        QString name;
        QString modelFile;
        std::vector<float> embedding;
    };

#ifdef HAS_OPENCV_FACE_CAPTURE
    struct ScoredFace {
        cv::Rect rect;
        float score = 0.f;
        float landmarks[10]{};
        bool hasLandmarks = false;
    };

    struct IdentityTrack {
        RegistryIdentity identity;
        std::vector<CapturedFrame> frames;
        cv::Rect lastRect;
        float lastDistance = 1.f;
        int consecutiveDetections = 0;
        int cooldown = 0;
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
    bool embedScoredFace(const cv::Mat& frame,
                         const ScoredFace& face,
                         std::vector<float>* embedding);
    bool processRegistryIdentities(const cv::Mat& frame,
                                   const std::vector<ScoredFace>& faces,
                                   QImage* preview);
    bool captureIdentityFrame(IdentityTrack* track,
                              const cv::Mat& frame,
                              const cv::Rect& rect);
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
    cv::Mat m_latestFrame;
    cv::Mat m_lastDetectedFrame;
    float m_lastFaceScore = 0.f;
#endif

    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QProgressBar* m_captureProgress = nullptr;
    QScrollArea* m_capturedGalleryScroll = nullptr;
    QWidget* m_capturedGalleryRow = nullptr;
    QLabel* m_angleLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_downloadProgress = nullptr;
    QPushButton* m_captureBtn = nullptr;
    QLabel* m_cameraDeviceLabel = nullptr;
    QComboBox* m_cameraCombo = nullptr;
    QComboBox* m_sourceCombo = nullptr;
    QWidget* m_cameraControlsRow = nullptr;
    QWidget* m_videoFileRow = nullptr;
    QLineEdit* m_videoPathEdit = nullptr;
    QPushButton* m_browseVideoBtn = nullptr;
    QComboBox* m_detectorCombo = nullptr;
    QDoubleSpinBox* m_minScoreSpin = nullptr;
    QSpinBox* m_minCapturesSpin = nullptr;
    QDoubleSpinBox* m_maxDistanceSpin = nullptr;
    QComboBox* m_faceStrategyCombo = nullptr;
    QLineEdit* m_registryPathEdit = nullptr;
    QLineEdit* m_registryFilterEdit = nullptr;
    QListWidget* m_registryList = nullptr;
    QLabel* m_registryStatusLabel = nullptr;

    std::vector<float> m_referenceEmbedding;
    bool m_hasReferenceEmbedding = false;
    static constexpr float kDefaultSamePersonMaxDistance = 0.55f;
    float maxSamePersonDistance() const;
    std::vector<RegistryIdentity> m_registryIdentities;
#ifdef HAS_OPENCV_FACE_CAPTURE
    std::vector<IdentityTrack> m_identityTracks;
#endif

    ecvModelDownloader* m_downloader = nullptr;
    aicore_facedetect_ctx* m_ggmlCtx = nullptr;
    aicore_cancel_token* m_inferenceCancelToken = nullptr;
    QString m_inferenceDevice = QStringLiteral("auto");
    QString m_loadedGgmlPath;
    QString m_pendingGgmlPath;

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

    bool m_registryPathUserChosen = false;
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
