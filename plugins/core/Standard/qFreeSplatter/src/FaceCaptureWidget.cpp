// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceCaptureWidget.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/core/utils/logger.hpp>
#endif

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFont>
#include <QHBoxLayout>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QStandardPaths>
#include <QTemporaryFile>
#include <algorithm>
#include <cstring>

FaceCaptureWidget::FaceCaptureWidget(QWidget* parent) : QWidget(parent) {
    setupUi();
}

FaceCaptureWidget::~FaceCaptureWidget() { stopCamera(); }

bool FaceCaptureWidget::isAvailable() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    return true;
#else
    return false;
#endif
}

void FaceCaptureWidget::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);

    m_previewLabel = new QLabel(this);
    m_previewLabel->setFixedSize(320, 240);
    m_previewLabel->setAlignment(Qt::AlignCenter);
    m_previewLabel->setStyleSheet(
            QStringLiteral("QLabel { background-color: #1a1a1a; "
                           "border: 1px solid #444; border-radius: 4px; }"));
    m_previewLabel->setText(tr("Camera preview"));
    mainLayout->addWidget(m_previewLabel, 0, Qt::AlignCenter);

    m_angleLabel = new QLabel(this);
    m_angleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_angleLabel);

    m_statusLabel = new QLabel(this);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_statusLabel);

#ifdef HAS_OPENCV_FACE_CAPTURE
    auto* controlsLayout = new QHBoxLayout();
    controlsLayout->addWidget(new QLabel(tr("Device:"), this));

    m_cameraCombo = new QComboBox(this);
    m_cameraCombo->addItem(tr("Default (0)"), 0);
    controlsLayout->addWidget(m_cameraCombo, 1);

    m_captureBtn = new QPushButton(tr("Capture"), this);
    m_captureBtn->setEnabled(false);
    controlsLayout->addWidget(m_captureBtn);

    mainLayout->addLayout(controlsLayout);

    m_frameTimer = new QTimer(this);
    m_frameTimer->setInterval(30);

    connect(m_frameTimer, &QTimer::timeout, this,
            &FaceCaptureWidget::processFrame);
    connect(m_captureBtn, &QPushButton::clicked, this,
            &FaceCaptureWidget::captureCurrentFrame);
    connect(m_cameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (!m_cameraActive) return;
                const int idx = m_cameraCombo->currentData().toInt();
                stopCamera();
                startCamera(idx);
            });

    m_statusLabel->setText(
            tr("Ready \u2014 select a camera and start capture"));
#else
    m_statusLabel->setText(
            tr("Face capture unavailable (OpenCV not built with videoio "
               "and objdetect)"));
#endif
}

bool FaceCaptureWidget::loadCascade() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_cascadeLoaded) return true;

    // 1. Try embedded Qt resource (guaranteed to exist if compiled in)
    const QString qrcPath = QStringLiteral(
            ":/CC/plugin/qFreeSplatter/"
            "data/haarcascade_frontalface_alt2.xml");
    if (QFile::exists(qrcPath)) {
        // CascadeClassifier needs a real file path; extract from resource
        const QString tmpDir =
                QStandardPaths::writableLocation(QStandardPaths::TempLocation);
        const QString tmpPath =
                tmpDir + QStringLiteral("/cv_haarcascade_frontalface_alt2.xml");
        if (!QFile::exists(tmpPath)) {
            QFile::copy(qrcPath, tmpPath);
            QFile::setPermissions(
                    tmpPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        }
        if (m_faceCascade.load(tmpPath.toStdString())) {
            m_cascadeLoaded = true;
            return true;
        }
    }

    // 2. Compile-time path from OpenCV build
#ifdef OPENCV_DATA_DIR
    {
        const QString path =
                QString(OPENCV_DATA_DIR) +
                QStringLiteral(
                        "/haarcascades/haarcascade_frontalface_alt2.xml");
        if (QFile::exists(path) && m_faceCascade.load(path.toStdString())) {
            m_cascadeLoaded = true;
            return true;
        }
    }
#endif

    // 3. Common system paths
    const QStringList systemPaths = {
            QCoreApplication::applicationDirPath() +
                    QStringLiteral("/../share/opencv4/haarcascades/"
                                   "haarcascade_frontalface_alt2.xml"),
            QStringLiteral("/usr/share/opencv4/haarcascades/"
                           "haarcascade_frontalface_alt2.xml"),
            QStringLiteral("/usr/local/share/opencv4/haarcascades/"
                           "haarcascade_frontalface_alt2.xml"),
            QStringLiteral("/opt/homebrew/share/opencv4/haarcascades/"
                           "haarcascade_frontalface_alt2.xml"),
    };
    for (const QString& p : systemPaths) {
        if (QFile::exists(p) && m_faceCascade.load(p.toStdString())) {
            m_cascadeLoaded = true;
            return true;
        }
    }

    return false;
#else
    return false;
#endif
}

bool FaceCaptureWidget::startCamera(int deviceIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    stopCamera();

    if (!m_camerasEnumerated && m_cameraCombo) {
        m_camerasEnumerated = true;
        m_cameraCombo->blockSignals(true);
        m_cameraCombo->clear();

        namespace cvlog = cv::utils::logging;
        const auto prevLevel = cvlog::getLogLevel();
        cvlog::setLogLevel(cvlog::LOG_LEVEL_SILENT);

        for (int i = 0; i < 10; ++i) {
            cv::VideoCapture testCap(i, cv::CAP_ANY);
            if (testCap.isOpened()) {
                m_cameraCombo->addItem(tr("Camera %1").arg(i), i);
                testCap.release();
            }
        }

        cvlog::setLogLevel(prevLevel);

        if (m_cameraCombo->count() == 0) {
            m_cameraCombo->addItem(tr("No camera found"), -1);
            m_cameraCombo->blockSignals(false);
            m_statusLabel->setText(tr("No camera devices detected"));
            return false;
        }
        if (deviceIndex == 0 && m_cameraCombo->count() > 0) {
            deviceIndex = m_cameraCombo->itemData(0).toInt();
        }
        m_cameraCombo->blockSignals(false);
    }

    if (!m_camera.open(deviceIndex, cv::CAP_ANY)) {
        m_cameraActive = false;
        const QString error =
                tr("Failed to open camera device %1").arg(deviceIndex);
        m_statusLabel->setText(error);
        emit cameraError(error);
        return false;
    }

    m_camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (!loadCascade()) {
        m_statusLabel->setText(
                tr("Warning: face detection cascade not found \u2014 "
                   "capture without detection"));
    } else {
        m_statusLabel->setText(tr("Camera active \u2014 detecting faces"));
    }

    m_cameraActive = true;
    m_frameTimer->start();
    emit cameraStarted();
    return true;
#else
    Q_UNUSED(deviceIndex);
    return false;
#endif
}

void FaceCaptureWidget::stopCamera() {
    if (m_frameTimer) m_frameTimer->stop();

#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_camera.isOpened()) m_camera.release();
#endif

    if (m_cameraActive) {
        m_cameraActive = false;
        emit cameraStopped();
    }

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_lastFaceRect = cv::Rect();
#endif
    m_consecutiveDetections = 0;
}

bool FaceCaptureWidget::isCameraActive() const { return m_cameraActive; }

void FaceCaptureWidget::startGuidedCapture(
        const std::vector<CaptureAngle>& angles) {
    resetCapture();
    m_targetAngles = angles;
    m_capturingMode = !m_targetAngles.empty();
    m_currentAngleIndex = 0;

    if (m_captureBtn) {
        m_captureBtn->setEnabled(m_capturingMode && m_cameraActive);
    }

    if (m_capturingMode) {
        m_angleLabel->setText(
                tr("Angle: %1 (1/%2)")
                        .arg(angleToString(m_targetAngles.front()))
                        .arg(m_targetAngles.size()));
        m_statusLabel->setText(tr("Position your face and capture each angle"));
    }
}

void FaceCaptureWidget::captureCurrentFrame() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_capturingMode ||
        m_currentAngleIndex >= static_cast<int>(m_targetAngles.size()))
        return;

    if (m_cascadeLoaded && m_consecutiveDetections < 3) {
        m_statusLabel->setText(tr("Hold still \u2014 face not stable yet"));
        return;
    }

    cv::Mat frame;
    if (!m_camera.read(frame) || frame.empty()) {
        emit cameraError(tr("Failed to read frame from camera"));
        return;
    }

    const auto angle = m_targetAngles[static_cast<size_t>(m_currentAngleIndex)];

    CapturedFrame captured;
    captured.image = cvMatToQImage(frame);
    captured.angle = angle;

    if (m_cascadeLoaded && m_lastFaceRect.width > 0) {
        captured.croppedFace = cropAndResizeFace(frame, m_lastFaceRect, 512);
        captured.faceRect = QRect(m_lastFaceRect.x, m_lastFaceRect.y,
                                  m_lastFaceRect.width, m_lastFaceRect.height);
    } else {
        // No face detection -- use full frame resized to 512x512
        cv::Mat resized;
        int side = std::min(frame.cols, frame.rows);
        int x = (frame.cols - side) / 2;
        int y = (frame.rows - side) / 2;
        cv::Mat cropped = frame(cv::Rect(x, y, side, side)).clone();
        cv::resize(cropped, resized, cv::Size(512, 512));
        captured.croppedFace = cvMatToQImage(resized);
    }
    captured.valid = !captured.croppedFace.isNull();

    if (!captured.valid) {
        m_statusLabel->setText(tr("Failed to capture frame"));
        return;
    }

    m_capturedFrames.push_back(captured);

    const int index = static_cast<int>(m_capturedFrames.size());
    const int total = static_cast<int>(m_targetAngles.size());
    emit frameCaptured(index, total);

    m_consecutiveDetections = 0;
    ++m_currentAngleIndex;

    if (m_currentAngleIndex >= static_cast<int>(m_targetAngles.size())) {
        m_capturingMode = false;
        if (m_captureBtn) m_captureBtn->setEnabled(false);
        m_angleLabel->setText(tr("Capture complete"));
        m_statusLabel->setText(
                tr("All angles captured (%1/%2)").arg(index).arg(total));
        emit captureComplete();
        return;
    }

    const auto nextAngle =
            m_targetAngles[static_cast<size_t>(m_currentAngleIndex)];
    m_angleLabel->setText(tr("Angle: %1 (%2/%3)")
                                  .arg(angleToString(nextAngle))
                                  .arg(m_currentAngleIndex + 1)
                                  .arg(total));
    if (m_captureBtn) m_captureBtn->setEnabled(false);
#endif
}

void FaceCaptureWidget::resetCapture() {
    m_targetAngles.clear();
    m_capturedFrames.clear();
    m_currentAngleIndex = 0;
    m_capturingMode = false;
    m_consecutiveDetections = 0;
    m_postCaptureCooldown = 0;
    m_noCascadeCounter = 0;

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_lastFaceRect = cv::Rect();
#endif

    if (m_captureBtn) m_captureBtn->setEnabled(false);
    if (m_angleLabel) m_angleLabel->setText(QString());
    if (m_statusLabel) {
        m_statusLabel->setText(
                m_cameraActive ? tr("Camera active \u2014 detecting faces")
                               : tr("Ready"));
    }
}

std::vector<FaceCaptureWidget::CapturedFrame>
FaceCaptureWidget::capturedFrames() const {
    return m_capturedFrames;
}

QStringList FaceCaptureWidget::exportCapturedImages(
        const QString& outputDir) const {
    QStringList paths;
    QDir dir(outputDir);
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) return paths;

    for (size_t i = 0; i < m_capturedFrames.size(); ++i) {
        const CapturedFrame& f = m_capturedFrames[i];
        if (!f.valid || f.croppedFace.isNull()) continue;

        QString tag = angleToString(f.angle);
        tag.replace(QLatin1Char(' '), QLatin1String("_"));
        tag.replace(QStringLiteral("\u00B0"), QStringLiteral("deg"));

        const QString filename =
                QStringLiteral("face_%1_%2.png")
                        .arg(static_cast<int>(i), 2, 10, QChar('0'))
                        .arg(tag);
        const QString path = dir.filePath(filename);
        if (f.croppedFace.save(path, "PNG")) {
            paths << path;
        }
    }
    return paths;
}

int FaceCaptureWidget::capturedCount() const {
    return static_cast<int>(m_capturedFrames.size());
}

int FaceCaptureWidget::targetCount() const {
    return static_cast<int>(m_targetAngles.size());
}

void FaceCaptureWidget::processFrame() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_cameraActive || !m_camera.isOpened()) return;

    cv::Mat frame;
    if (!m_camera.read(frame) || frame.empty()) return;

    cv::Rect faceRect;
    if (m_cascadeLoaded) {
        faceRect = detectFace(frame);
        if (faceRect.width > 0 && faceRect.height > 0) {
            ++m_consecutiveDetections;
            m_lastFaceRect = faceRect;
            emit faceDetected(QRect(faceRect.x, faceRect.y, faceRect.width,
                                    faceRect.height));
        } else {
            m_consecutiveDetections = 0;
            m_lastFaceRect = cv::Rect();
            emit faceNotDetected();
        }
    }

    QImage preview = cvMatToQImage(frame);
    if (m_cascadeLoaded && faceRect.width > 0) {
        drawOverlay(preview, faceRect);
    }
    if (m_capturingMode &&
        m_currentAngleIndex < static_cast<int>(m_targetAngles.size())) {
        drawAngleGuide(
                preview,
                m_targetAngles[static_cast<size_t>(m_currentAngleIndex)]);
    }

    m_previewLabel->setPixmap(QPixmap::fromImage(preview).scaled(
            m_previewLabel->size(), Qt::KeepAspectRatio,
            Qt::SmoothTransformation));

    if (m_capturingMode) {
        if (m_postCaptureCooldown > 0) {
            --m_postCaptureCooldown;
            m_statusLabel->setText(
                    tr("Repositioning... (%1)")
                            .arg(m_postCaptureCooldown / 30 + 1));
            if (m_captureBtn) m_captureBtn->setEnabled(false);
        } else if (m_cascadeLoaded) {
            const bool stable = m_consecutiveDetections >= kAutoCaptureTrigger;
            if (stable) {
                captureCurrentFrame();
                m_postCaptureCooldown = kPostCaptureCooldown;
                return;
            }
            if (m_captureBtn)
                m_captureBtn->setEnabled(m_consecutiveDetections >= 3);
            if (faceRect.width > 0) {
                int pct = std::min(100, m_consecutiveDetections * 100 /
                                                kAutoCaptureTrigger);
                m_statusLabel->setText(tr("Stabilizing... %1%").arg(pct));
            } else {
                m_statusLabel->setText(tr(
                        "No face detected \u2014 center your face in frame"));
            }
        } else {
            ++m_noCascadeCounter;
            if (m_noCascadeCounter >= kNoCascadeAutoInterval) {
                m_noCascadeCounter = 0;
                captureCurrentFrame();
                m_postCaptureCooldown = kPostCaptureCooldown;
                return;
            }
            if (m_captureBtn) m_captureBtn->setEnabled(true);
            m_statusLabel->setText(
                    tr("Auto-capture in %1s (or click Capture)")
                            .arg((kNoCascadeAutoInterval - m_noCascadeCounter) /
                                         30 +
                                 1));
        }
    }
#endif
}

#ifdef HAS_OPENCV_FACE_CAPTURE

QImage FaceCaptureWidget::cvMatToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();

    cv::Mat rgb;
    if (mat.channels() == 1)
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    else if (mat.channels() == 3)
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 4)
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGBA);
    else
        return QImage();

    QImage image(rgb.cols, rgb.rows, QImage::Format_RGB888);
    for (int y = 0; y < rgb.rows; ++y) {
        std::memcpy(image.scanLine(y), rgb.ptr<uchar>(y),
                    static_cast<size_t>(rgb.cols) * 3);
    }
    return image;
}

cv::Rect FaceCaptureWidget::detectFace(const cv::Mat& frame) {
    if (m_faceCascade.empty() || frame.empty()) return cv::Rect();

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    m_faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(80, 80));
    if (faces.empty()) return cv::Rect();

    return *std::max_element(faces.begin(), faces.end(),
                             [](const cv::Rect& a, const cv::Rect& b) {
                                 return a.area() < b.area();
                             });
}

QImage FaceCaptureWidget::cropAndResizeFace(const cv::Mat& frame,
                                            const cv::Rect& faceRect,
                                            int targetSize) {
    if (frame.empty() || faceRect.width <= 0 || targetSize <= 0)
        return QImage();

    const int expandW = static_cast<int>(faceRect.width * 0.5);
    const int expandH = static_cast<int>(faceRect.height * 0.5);
    int side = std::max(faceRect.width + 2 * expandW,
                        faceRect.height + 2 * expandH);

    int cx = faceRect.x + faceRect.width / 2;
    int cy = faceRect.y + faceRect.height / 2;
    int x = std::max(0, cx - side / 2);
    int y = std::max(0, cy - side / 2);
    if (x + side > frame.cols) side = frame.cols - x;
    if (y + side > frame.rows) side = frame.rows - y;
    if (side <= 0) return QImage();

    cv::Mat cropped = frame(cv::Rect(x, y, side, side)).clone();
    cv::Mat resized;
    int interp =
            (cropped.cols > targetSize) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(cropped, resized, cv::Size(targetSize, targetSize), 0, 0,
               interp);
    return cvMatToQImage(resized);
}

void FaceCaptureWidget::drawOverlay(QImage& image, const cv::Rect& faceRect) {
    if (image.isNull() || faceRect.width <= 0) return;

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);
    QPen pen(QColor(0, 220, 80));
    pen.setWidth(3);
    painter.setPen(pen);
    painter.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
}

void FaceCaptureWidget::drawAngleGuide(QImage& image, CaptureAngle angle) {
    if (image.isNull()) return;

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    QFont font = painter.font();
    font.setPointSize(16);
    font.setBold(true);
    painter.setFont(font);

    const int margin = 12;
    const QFontMetrics fm(font);
    const QRect textRect(margin, margin, image.width() - 2 * margin,
                         fm.height() + 16);
    painter.fillRect(textRect, QColor(0, 0, 0, 160));
    painter.setPen(Qt::white);
    painter.drawText(textRect, Qt::AlignCenter, angleToString(angle));
}

#endif  // HAS_OPENCV_FACE_CAPTURE

QString FaceCaptureWidget::angleToString(CaptureAngle angle) const {
    switch (angle) {
        case CaptureAngle::Front:
            return tr("Look straight ahead");
        case CaptureAngle::Left45:
            return tr("Turn head 45\u00B0 left");
        case CaptureAngle::Right45:
            return tr("Turn head 45\u00B0 right");
        case CaptureAngle::Left90:
            return tr("Turn head 90\u00B0 left");
        case CaptureAngle::Right90:
            return tr("Turn head 90\u00B0 right");
        case CaptureAngle::Up15:
            return tr("Tilt head up ~15\u00B0");
        case CaptureAngle::Down15:
            return tr("Tilt head down ~15\u00B0");
    }
    return tr("Unknown angle");
}
