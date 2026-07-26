// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QComboBox>
#include <QImage>
#include <QLabel>
#include <QMutex>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

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
    void stopCamera();
    bool isCameraActive() const;

    void startGuidedCapture(const std::vector<CaptureAngle>& angles);
    void captureCurrentFrame();
    void resetCapture();

    std::vector<CapturedFrame> capturedFrames() const;
    QStringList exportCapturedImages(const QString& outputDir) const;

    int capturedCount() const;
    int targetCount() const;

    static bool isAvailable();

signals:
    void cameraStarted();
    void cameraStopped();
    void faceDetected(const QRect& bbox);
    void faceNotDetected();
    void frameCaptured(int index, int total);
    void captureComplete();
    void cameraError(const QString& error);

private slots:
    void processFrame();

private:
    void setupUi();
    bool loadCascade();
    QString angleToString(CaptureAngle angle) const;

#ifdef HAS_OPENCV_FACE_CAPTURE
    QImage cvMatToQImage(const cv::Mat& mat);
    cv::Rect detectFace(const cv::Mat& frame);
    QImage cropAndResizeFace(const cv::Mat& frame,
                             const cv::Rect& faceRect,
                             int targetSize = 512);
    void drawOverlay(QImage& image, const cv::Rect& faceRect);
    void drawAngleGuide(QImage& image, CaptureAngle angle);

    cv::VideoCapture m_camera;
    cv::CascadeClassifier m_faceCascade;
    cv::Rect m_lastFaceRect;
#endif

    QLabel* m_previewLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QLabel* m_angleLabel = nullptr;
    QPushButton* m_captureBtn = nullptr;
    QComboBox* m_cameraCombo = nullptr;

    QTimer* m_frameTimer = nullptr;
    bool m_cameraActive = false;
    bool m_cascadeLoaded = false;

    std::vector<CaptureAngle> m_targetAngles;
    std::vector<CapturedFrame> m_capturedFrames;
    int m_currentAngleIndex = 0;
    bool m_capturingMode = false;
    int m_consecutiveDetections = 0;
};
