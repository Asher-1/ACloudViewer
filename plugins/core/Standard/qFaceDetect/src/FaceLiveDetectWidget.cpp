// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceLiveDetectWidget.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QSettings>
#include <QtMath>

#include "FaceDetectEmbedHelpers.h"
#include "FaceDetectTestData.h"
#include "FaceDetectUiHelpers.h"

#ifdef AICore_ENABLED
#include "aicore/facedetect_capi.h"
#endif

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/imgproc.hpp>
#endif

#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

namespace {

#ifdef HAS_OPENCV_FACE_CAPTURE
QImage cvMatToQImage(const cv::Mat& mat) {
    if (mat.empty()) return {};
    cv::Mat rgb;
    if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
    } else {
        return {};
    }
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888)
            .copy();
}
#endif

}  // namespace

bool FaceLiveDetectWidget::isAvailable() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    return true;
#else
    return false;
#endif
}

FaceLiveDetectWidget::FaceLiveDetectWidget(QWidget* parent) : QWidget(parent) {
    setupUi();
    m_inferThread = new QThread(this);
    m_inferWorker = new FaceLiveDetectInferWorker;
    m_inferWorker->moveToThread(m_inferThread);
    connect(m_inferThread, &QThread::finished, m_inferWorker,
            &QObject::deleteLater);
    connect(m_inferWorker, &FaceLiveDetectInferWorker::inferComplete, this,
            &FaceLiveDetectWidget::onInferComplete, Qt::QueuedConnection);
    m_inferThread->start();

    m_frameTimer = new QTimer(this);
    m_frameTimer->setInterval(33);
    connect(m_frameTimer, &QTimer::timeout, this,
            &FaceLiveDetectWidget::processFrame);
    loadSettings();
}

FaceLiveDetectWidget::~FaceLiveDetectWidget() {
    saveSettings();
    stopStream();
    shutdownInferThread();
}

void FaceLiveDetectWidget::shutdownInferThread() {
    if (m_inferWorker && m_inferThread) {
        QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                                  Qt::BlockingQueuedConnection);
        m_inferThread->quit();
        m_inferThread->wait(3000);
        m_inferWorker = nullptr;
    }
}

void FaceLiveDetectWidget::setupUi() {
    auto* main = new QVBoxLayout(this);
    main->setContentsMargins(4, 4, 4, 4);
    main->setSpacing(4);

    m_previewLabel = new ecvClickableImageLabel(this);
    m_previewLabel->setMinimumSize(480, 300);
    m_previewLabel->setSizePolicy(QSizePolicy::Expanding,
                                  QSizePolicy::Expanding);
    m_previewLabel->setStyleSheet(
            "border: 1px solid palette(mid); background: #111; color: #888;");
    m_previewLabel->setText(tr("Live preview"));
    main->addWidget(m_previewLabel, 1);

    auto* settingsGroup = new QGroupBox(tr("Stream settings"), this);
    auto* grid = new QGridLayout(settingsGroup);
    FaceDetectUi::setupTwoColumnFormGrid(grid);
    grid->setContentsMargins(6, 6, 6, 4);
    grid->setHorizontalSpacing(8);
    grid->setVerticalSpacing(3);

    m_modelCombo = new QComboBox(settingsGroup);
    m_deviceCombo = new QComboBox(settingsGroup);
    m_threadsSpin = new QSpinBox(settingsGroup);
    m_threadsSpin->setRange(0, 128);
    m_threadsSpin->setSpecialValueText(tr("Auto"));
    FaceDetectUi::makeCompactSpin(m_threadsSpin);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (m_syncingModelControls) return;
                updateModelPathFromCombo();
                if (m_inferWorker) {
                    QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                                              Qt::QueuedConnection);
                }
                m_config.modelPath = resolveModelPath();
                m_config.device = deviceId();
                m_config.threads = threadCount();
                emit modelSelectionChanged(modelFilename());
                saveSettings();
            });
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (m_syncingModelControls) return;
                m_config.device = deviceId();
                if (m_inferWorker) {
                    QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                                              Qt::QueuedConnection);
                }
                emit deviceSelectionChanged(m_config.device);
                saveSettings();
            });
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
                if (m_syncingModelControls) return;
                m_config.threads = value;
                emit threadCountChanged(value);
                saveSettings();
            });
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Detector GGUF:")), 0, 0);
    grid->addWidget(m_modelCombo, 0, 1);
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Device:")), 0, 2);
    grid->addWidget(m_deviceCombo, 0, 3);
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Threads:")), 1, 0);
    grid->addWidget(m_threadsSpin, 1, 1, Qt::AlignLeft);

    m_modeCombo = new QComboBox(settingsGroup);
    m_modeCombo->addItem(tr("Detect faces only"),
                         static_cast<int>(StreamMode::Detect));
    m_modeCombo->addItem(tr("Recognize (Registry DB)"),
                         static_cast<int>(StreamMode::Recognize));
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceLiveDetectWidget::onStreamModeChanged);
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Mode:")), 1, 2);
    grid->addWidget(m_modeCombo, 1, 3);

    m_sourceCombo = new QComboBox(settingsGroup);
    m_sourceCombo->addItem(tr("Live camera"),
                           static_cast<int>(InputSource::Camera));
    m_sourceCombo->addItem(tr("Video file"),
                           static_cast<int>(InputSource::VideoFile));
    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceLiveDetectWidget::onSourceChanged);
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Source:")), 2, 0);
    grid->addWidget(m_sourceCombo, 2, 1);

    m_minScoreLabel = new QLabel(tr("Min score:"), settingsGroup);
    m_minDetectionScore = FaceDetectUi::makeMinDetectionScoreSpin(
            settingsGroup,
            tr("Faces below this detection score are drawn in red and excluded "
               "from capture/export."));
    grid->addWidget(m_minScoreLabel, 3, 0);
    grid->addWidget(m_minDetectionScore, 3, 1, Qt::AlignLeft);

    m_matchDistLabel = new QLabel(tr("Match dist:"), settingsGroup);
    m_recognizeThreshold = new QDoubleSpinBox(settingsGroup);
    m_recognizeThreshold->setRange(0.05, 1.0);
    m_recognizeThreshold->setSingleStep(0.01);
    m_recognizeThreshold->setValue(0.52);
    FaceDetectUi::makeCompactDoubleSpin(m_recognizeThreshold);
    m_recognizeThreshold->setToolTip(
            tr("Max cosine distance for registry match (lower = stricter)."));
    grid->addWidget(m_matchDistLabel, 3, 2);
    grid->addWidget(m_recognizeThreshold, 3, 3, Qt::AlignLeft);

    m_registryRow = new QWidget(settingsGroup);
    auto* registryLayout = new QHBoxLayout(m_registryRow);
    registryLayout->setContentsMargins(0, 0, 0, 0);
    m_registryPathEdit = new QLineEdit(m_registryRow);
    m_registryPathEdit->setPlaceholderText(
            tr("Face registry .db (required for Recognize mode)"));
    auto* registryBrowse =
            FaceDetectUi::makeBrowseButton(tr("Browse…"), m_registryRow);
    connect(registryBrowse, &QPushButton::clicked, this, [this]() {
        QSettings settings;
        const QString lastDir =
                settings.value(QStringLiteral("qFaceDetect/lastRegistryDir"),
                               FaceDetectEmbed::modelCacheDir())
                        .toString();
        const QString path = QFileDialog::getOpenFileName(
                this, tr("Face registry database"), lastDir,
                tr("SQLite database (*.db);;All files (*.*)"));
        if (path.isEmpty()) return;
        settings.setValue(QStringLiteral("qFaceDetect/lastRegistryDir"),
                          QFileInfo(path).absolutePath());
        setRegistryPath(path, true);
        emit registryPathEdited(path);
    });
    connect(m_registryPathEdit, &QLineEdit::editingFinished, this, [this]() {
        if (!m_registryPathEdit) return;
        const QString path = m_registryPathEdit->text().trimmed();
        if (!path.isEmpty()) {
            m_registryPathUserChosen = true;
            emit registryPathEdited(path);
        }
    });
    registryLayout->addWidget(new QLabel(tr("Registry DB:")));
    registryLayout->addWidget(m_registryPathEdit, 1);
    registryLayout->addWidget(registryBrowse);
    grid->addWidget(m_registryRow, 4, 0, 1, 4);

    connect(m_minDetectionScore,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double value) {
                m_config.minDetectionScore = static_cast<float>(value);
                emit minDetectionScoreChanged(static_cast<float>(value));
                saveSettings();
            });
    connect(m_recognizeThreshold,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double value) {
                m_config.recognizeMaxDistance = static_cast<float>(value);
                emit matchThresholdChanged(static_cast<float>(value));
                saveSettings();
            });
    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { saveSettings(); });

    main->addWidget(settingsGroup);

    m_cameraRow = new QWidget(this);
    auto* camLayout = new QHBoxLayout(m_cameraRow);
    camLayout->setContentsMargins(0, 0, 0, 0);
    m_cameraCombo = new QComboBox(m_cameraRow);
    m_cameraCombo->addItem(tr("Default (0)"), 0);
    camLayout->addWidget(new QLabel(tr("Camera:")));
    camLayout->addWidget(m_cameraCombo, 1);

    m_videoRow = new QWidget(this);
    auto* vidLayout = new QHBoxLayout(m_videoRow);
    vidLayout->setContentsMargins(0, 0, 0, 0);
    m_videoPathEdit = new QLineEdit(m_videoRow);
    auto* browse = FaceDetectUi::makeBrowseButton(tr("Browse…"), m_videoRow);
    m_testDataBtn = new QPushButton(tr("Use test data"), m_videoRow);
    m_testDataBtn->setToolTip(
            tr("Download FriendsFaces sample video and load it here."));
    connect(browse, &QPushButton::clicked, this,
            &FaceLiveDetectWidget::onBrowseVideo);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &FaceLiveDetectWidget::testDataRequested);
    connect(m_videoPathEdit, &QLineEdit::editingFinished, this, [this]() {
        if (!m_videoPathEdit) return;
        const QString path = m_videoPathEdit->text().trimmed();
        if (!path.isEmpty()) {
            m_videoPathUserChosen = true;
            saveSettings();
        }
    });
    vidLayout->addWidget(m_videoPathEdit, 1);
    vidLayout->addWidget(m_testDataBtn);
    vidLayout->addWidget(browse);
    m_videoRow->setVisible(false);

    main->addWidget(m_cameraRow);
    main->addWidget(m_videoRow);

    m_captureBtn = new QPushButton(tr("Capture frame to DB"), this);
    m_captureBtn->setEnabled(false);
    connect(m_captureBtn, &QPushButton::clicked, this,
            &FaceLiveDetectWidget::captureSnapshotToDb);
    main->addWidget(m_captureBtn);

    m_statusLabel = new QLabel(
            isAvailable()
                    ? tr("Configure stream, then press Start in the dialog.")
                    : tr("Live detect unavailable (build with OpenCV "
                         "videoio)."),
            this);
    m_statusLabel->setWordWrap(true);
    main->addWidget(m_statusLabel);

    if (!isAvailable()) setEnabled(false);
    updateThresholdUi();
}

void FaceLiveDetectWidget::setConfig(const Config& config) {
    if (m_config.modelPath != config.modelPath && m_inferWorker) {
        QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                                  Qt::QueuedConnection);
    }
    m_config = config;
    if (m_recognizeThreshold) {
        m_recognizeThreshold->blockSignals(true);
        m_recognizeThreshold->setValue(config.recognizeMaxDistance);
        m_recognizeThreshold->blockSignals(false);
    }
    if (m_minDetectionScore) {
        m_minDetectionScore->blockSignals(true);
        m_minDetectionScore->setValue(config.minDetectionScore);
        m_minDetectionScore->blockSignals(false);
    }
    updateThresholdUi();
    if (m_modeCombo) {
        const int idx =
                m_modeCombo->findData(static_cast<int>(config.streamMode));
        if (idx >= 0) {
            m_modeCombo->blockSignals(true);
            m_modeCombo->setCurrentIndex(idx);
            m_modeCombo->blockSignals(false);
        }
    }
    if (!config.modelPath.isEmpty()) {
        setModelPath(config.modelPath);
    }
    setDevice(config.device);
    if (m_threadsSpin) {
        m_syncingModelControls = true;
        m_threadsSpin->setValue(config.threads);
        m_syncingModelControls = false;
    }
}

void FaceLiveDetectWidget::updateModelPathFromCombo() {
    if (!m_modelCombo) return;
    const QString fn = m_modelCombo->currentData().toString();
    if (fn.isEmpty()) return;
    m_config.modelPath =
            FaceDetectEmbed::modelCacheDir() + QLatin1Char('/') + fn;
}

QString FaceLiveDetectWidget::modelFilename() const {
    return m_modelCombo ? m_modelCombo->currentData().toString() : QString();
}

QString FaceLiveDetectWidget::deviceId() const {
    return m_deviceCombo ? m_deviceCombo->currentData().toString()
                         : m_config.device;
}

int FaceLiveDetectWidget::threadCount() const {
    return m_threadsSpin ? m_threadsSpin->value() : m_config.threads;
}

QString FaceLiveDetectWidget::resolveModelPath() const {
    if (!m_config.modelPath.isEmpty()) return m_config.modelPath;
    const QString fn = modelFilename();
    if (fn.isEmpty()) return {};
    return FaceDetectEmbed::modelCacheDir() + QLatin1Char('/') + fn;
}

void FaceLiveDetectWidget::rebuildModelCombo(const QStringList& labels,
                                             const QStringList& filenames,
                                             const QString& currentFilename) {
    if (!m_modelCombo || labels.size() != filenames.size()) return;
    m_syncingModelControls = true;
    m_modelCombo->clear();
    int selectIndex = 0;
    for (int i = 0; i < labels.size(); ++i) {
        m_modelCombo->addItem(labels.at(i), filenames.at(i));
        if (filenames.at(i) == currentFilename) selectIndex = i;
    }
    m_modelCombo->setCurrentIndex(selectIndex);
    updateModelPathFromCombo();
    m_syncingModelControls = false;
}

void FaceLiveDetectWidget::rebuildDeviceCombo(
        const QComboBox* sourceDeviceCombo) {
    if (!m_deviceCombo || !sourceDeviceCombo) return;
    m_syncingModelControls = true;
    m_deviceCombo->clear();
    for (int i = 0; i < sourceDeviceCombo->count(); ++i) {
        m_deviceCombo->addItem(sourceDeviceCombo->itemText(i),
                               sourceDeviceCombo->itemData(i));
    }
    m_deviceCombo->setCurrentIndex(sourceDeviceCombo->currentIndex());
    m_config.device = deviceId();
    m_syncingModelControls = false;
}

void FaceLiveDetectWidget::syncModelControlsFrom(const QComboBox* modelCombo,
                                                 const QComboBox* deviceCombo,
                                                 const QSpinBox* threadsSpin) {
    if (!modelCombo || !deviceCombo || !threadsSpin) return;
    QStringList labels;
    QStringList filenames;
    for (int i = 0; i < modelCombo->count(); ++i) {
        const QString data = modelCombo->itemData(i).toString();
        if (data == QStringLiteral("CUSTOM")) continue;
        labels.append(modelCombo->itemText(i));
        filenames.append(data);
    }
    rebuildModelCombo(labels, filenames, modelCombo->currentData().toString());
    rebuildDeviceCombo(deviceCombo);
    m_syncingModelControls = true;
    if (m_threadsSpin) m_threadsSpin->setValue(threadsSpin->value());
    m_config.threads = threadsSpin->value();
    m_config.device = deviceId();
    updateModelPathFromCombo();
    m_syncingModelControls = false;
}

void FaceLiveDetectWidget::setModelPath(const QString& path) {
    if (m_config.modelPath != path && m_inferWorker) {
        QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                                  Qt::QueuedConnection);
    }
    m_config.modelPath = path;
    if (!m_modelCombo) return;
    const QString fn = QFileInfo(path).fileName();
    const int idx = m_modelCombo->findData(fn);
    if (idx >= 0) {
        m_syncingModelControls = true;
        m_modelCombo->setCurrentIndex(idx);
        m_syncingModelControls = false;
    }
}

void FaceLiveDetectWidget::setDevice(const QString& device) {
    if (m_config.device != device && m_inferWorker) {
        QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                                  Qt::QueuedConnection);
    }
    m_config.device = device;
    if (!m_deviceCombo) return;
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) {
        m_syncingModelControls = true;
        m_deviceCombo->setCurrentIndex(idx);
        m_syncingModelControls = false;
    }
}

void FaceLiveDetectWidget::setThreads(int threads) {
    m_config.threads = threads;
    if (!m_threadsSpin) return;
    m_syncingModelControls = true;
    m_threadsSpin->setValue(threads);
    m_syncingModelControls = false;
}

void FaceLiveDetectWidget::setMatchThreshold(float value) {
    m_config.recognizeMaxDistance = value;
    if (m_recognizeThreshold) {
        m_recognizeThreshold->blockSignals(true);
        m_recognizeThreshold->setValue(value);
        m_recognizeThreshold->blockSignals(false);
    }
}

void FaceLiveDetectWidget::setMinDetectionScore(float value) {
    m_config.minDetectionScore = value;
    if (m_minDetectionScore) {
        m_minDetectionScore->blockSignals(true);
        m_minDetectionScore->setValue(value);
        m_minDetectionScore->blockSignals(false);
    }
}

void FaceLiveDetectWidget::setRegistryStore(FaceRegistryStore* store) {
    m_config.registry = store;
}

void FaceLiveDetectWidget::updateThresholdUi() {
    const bool recognize = m_config.streamMode == StreamMode::Recognize;
    if (m_matchDistLabel) m_matchDistLabel->setVisible(recognize);
    if (m_recognizeThreshold) m_recognizeThreshold->setVisible(recognize);
    if (m_minScoreLabel) m_minScoreLabel->setVisible(true);
    if (m_minDetectionScore) m_minDetectionScore->setVisible(true);
    updateRegistryUi();
}

void FaceLiveDetectWidget::updateRegistryUi() {
    const bool recognize = m_config.streamMode == StreamMode::Recognize;
    if (m_registryRow) m_registryRow->setVisible(recognize);
}

void FaceLiveDetectWidget::setRegistryPath(const QString& path,
                                           bool userChosen) {
    m_registryPathUserChosen = userChosen;
    if (m_registryPathEdit) {
        m_registryPathEdit->blockSignals(true);
        m_registryPathEdit->setText(path);
        m_registryPathEdit->blockSignals(false);
    }
}

QString FaceLiveDetectWidget::registryPath() const {
    return m_registryPathEdit ? m_registryPathEdit->text().trimmed()
                              : QString();
}

void FaceLiveDetectWidget::setStreamMode(StreamMode mode) {
    m_config.streamMode = mode;
    if (m_modeCombo) {
        const int idx = m_modeCombo->findData(static_cast<int>(mode));
        if (idx >= 0) {
            m_modeCombo->blockSignals(true);
            m_modeCombo->setCurrentIndex(idx);
            m_modeCombo->blockSignals(false);
        }
    }
    updateThresholdUi();
    updateRegistryUi();
    saveSettings();
}

void FaceLiveDetectWidget::loadSettings() {
    QSettings settings;
    const int streamMode =
            settings.value(QStringLiteral("qFaceDetect/liveStreamMode"),
                           static_cast<int>(StreamMode::Detect))
                    .toInt();
    const double minScore =
            settings.value(QStringLiteral("qFaceDetect/minDetectionScore"),
                           settings.value(
                                   QStringLiteral(
                                           "qFaceDetect/liveMinDetectionScore"),
                                   settings.value(
                                           QStringLiteral(
                                                   "qFaceDetect/"
                                                   "registryMinDetectionScore"),
                                           0.5)))
                    .toDouble();
    const double matchDist =
            settings.value(QStringLiteral("qFaceDetect/matchThreshold"),
                           settings.value(
                                   QStringLiteral(
                                           "qFaceDetect/liveMatchDistance"),
                                   0.52))
                    .toDouble();
    const int source = settings.value(QStringLiteral("qFaceDetect/liveSource"),
                                      static_cast<int>(InputSource::Camera))
                               .toInt();
    const QString videoPath =
            settings.value(FaceDetectTestData::manualLiveVideoSettingsKey())
                    .toString();
    if (videoPath.isEmpty()) {
        const QString legacy =
                settings.value(QStringLiteral("qFaceDetect/liveVideoPath"))
                        .toString();
        if (!legacy.isEmpty() &&
            !FaceDetectTestData::isFriendsBundlePath(legacy)) {
            m_videoPathUserChosen = true;
            if (m_videoPathEdit) m_videoPathEdit->setText(legacy);
        }
    } else {
        m_videoPathUserChosen = true;
        if (m_videoPathEdit) m_videoPathEdit->setText(videoPath);
    }

    QString registryPath =
            settings.value(FaceDetectTestData::manualRegistryDbSettingsKey())
                    .toString();
    m_registryPathUserChosen = !registryPath.isEmpty();
    if (registryPath.isEmpty()) {
        const QString legacy =
                settings.value(QStringLiteral("qFaceDetect/liveRegistryDbPath"))
                        .toString();
        if (!legacy.isEmpty() &&
            !FaceDetectTestData::isFriendsBundlePath(legacy)) {
            registryPath = legacy;
            m_registryPathUserChosen = true;
        }
    }
    if (m_registryPathUserChosen && !registryPath.isEmpty()) {
        setRegistryPath(registryPath, true);
    }
    m_config.minDetectionScore = static_cast<float>(minScore);
    m_config.recognizeMaxDistance = static_cast<float>(matchDist);

    if (m_modeCombo) {
        const int idx = m_modeCombo->findData(streamMode);
        if (idx >= 0) {
            m_modeCombo->blockSignals(true);
            m_modeCombo->setCurrentIndex(idx);
            m_modeCombo->blockSignals(false);
        }
    }
    if (m_minDetectionScore) {
        m_minDetectionScore->blockSignals(true);
        m_minDetectionScore->setValue(minScore);
        m_minDetectionScore->blockSignals(false);
    }
    if (m_recognizeThreshold) {
        m_recognizeThreshold->blockSignals(true);
        m_recognizeThreshold->setValue(matchDist);
        m_recognizeThreshold->blockSignals(false);
    }
    m_config.streamMode = static_cast<StreamMode>(streamMode);
    updateRegistryUi();
    if (m_sourceCombo) {
        const int idx = m_sourceCombo->findData(source);
        if (idx >= 0) {
            m_sourceCombo->blockSignals(true);
            m_sourceCombo->setCurrentIndex(idx);
            m_sourceCombo->blockSignals(false);
            onSourceChanged(idx);
        }
    }
    updateThresholdUi();

    if (!settings.contains(QStringLiteral("qFaceDetect/lastVideoFileDir"))) {
        QString dirSource = videoPath;
        if (dirSource.isEmpty()) {
            dirSource =
                    settings.value(QStringLiteral("qFaceDetect/liveVideoPath"))
                            .toString();
        }
        if (!dirSource.isEmpty() &&
            !FaceDetectTestData::isFriendsBundlePath(dirSource)) {
            settings.setValue(QStringLiteral("qFaceDetect/lastVideoFileDir"),
                              QFileInfo(dirSource).absolutePath());
        }
    }
    settings.remove(QStringLiteral("qFaceDetect/liveVideoPath"));
}

void FaceLiveDetectWidget::saveSettings() const {
    QSettings settings;
    settings.setValue(QStringLiteral("qFaceDetect/liveStreamMode"),
                      static_cast<int>(m_config.streamMode));
    settings.setValue(QStringLiteral("qFaceDetect/minDetectionScore"),
                      m_config.minDetectionScore);
    settings.setValue(QStringLiteral("qFaceDetect/liveMatchDistance"),
                      m_config.recognizeMaxDistance);
    settings.setValue(QStringLiteral("qFaceDetect/matchThreshold"),
                      m_config.recognizeMaxDistance);
    if (m_sourceCombo) {
        settings.setValue(QStringLiteral("qFaceDetect/liveSource"),
                          m_sourceCombo->currentData());
    }
    if (m_videoPathUserChosen && m_videoPathEdit) {
        const QString path = m_videoPathEdit->text().trimmed();
        if (!path.isEmpty()) {
            settings.setValue(FaceDetectTestData::manualLiveVideoSettingsKey(),
                              path);
        } else {
            settings.remove(FaceDetectTestData::manualLiveVideoSettingsKey());
        }
    } else {
        settings.remove(FaceDetectTestData::manualLiveVideoSettingsKey());
    }
    settings.remove(QStringLiteral("qFaceDetect/liveVideoPath"));

    if (m_registryPathUserChosen) {
        const QString path = registryPath();
        if (!path.isEmpty()) {
            settings.setValue(FaceDetectTestData::manualRegistryDbSettingsKey(),
                              path);
        } else {
            settings.remove(FaceDetectTestData::manualRegistryDbSettingsKey());
        }
    } else {
        settings.remove(FaceDetectTestData::manualRegistryDbSettingsKey());
    }
    settings.remove(QStringLiteral("qFaceDetect/liveRegistryDbPath"));
}

void FaceLiveDetectWidget::onStreamModeChanged(int index) {
    const auto mode =
            static_cast<StreamMode>(m_modeCombo->itemData(index).toInt());
    m_config.streamMode = mode;
    if (m_recognizeThreshold) {
        m_config.recognizeMaxDistance =
                static_cast<float>(m_recognizeThreshold->value());
    }
    if (m_minDetectionScore) {
        m_config.minDetectionScore =
                static_cast<float>(m_minDetectionScore->value());
    }
    updateThresholdUi();
    saveSettings();
    emit streamModeChanged(mode);
    if (mode == StreamMode::Recognize && m_config.registry &&
        !m_config.registry->isOpen()) {
        m_statusLabel->setText(
                tr("Recognition mode — open Registry tab and ensure DB is "
                   "loaded."));
    }
}

void FaceLiveDetectWidget::onSourceChanged(int index) {
    const int kind = m_sourceCombo->itemData(index).toInt();
    const bool video = kind == static_cast<int>(InputSource::VideoFile);
    if (m_videoRow) m_videoRow->setVisible(video);
    if (m_cameraRow) m_cameraRow->setVisible(!video);
    if (m_streamActive) stopStream();
}

void FaceLiveDetectWidget::onBrowseVideo() {
    QSettings settings;
    QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qFaceDetect"),
                                       QStringLiteral("lastVideoFileDir"),
                                       QDir::homePath());
    if (lastDir.isEmpty() || !QFileInfo(lastDir).exists()) {
        const QString manual =
                settings.value(FaceDetectTestData::manualLiveVideoSettingsKey())
                        .toString();
        if (!manual.isEmpty()) {
            lastDir = QFileInfo(manual).absolutePath();
        }
    }
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select video"), lastDir,
            tr("Video (*.mp4 *.avi *.mkv *.mov *.webm *.m4v)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qFaceDetect"),
                         QStringLiteral("lastVideoFileDir"), path);
    if (m_videoPathEdit) m_videoPathEdit->setText(path);
    m_videoPathUserChosen = true;
    saveSettings();
}

void FaceLiveDetectWidget::setVideoFilePath(const QString& path,
                                            bool userChosen) {
    m_videoPathUserChosen = userChosen;
    if (m_videoPathEdit) m_videoPathEdit->setText(path);
}

void FaceLiveDetectWidget::selectVideoFileSource() {
    if (!m_sourceCombo) return;
    const int idx =
            m_sourceCombo->findData(static_cast<int>(InputSource::VideoFile));
    if (idx >= 0) m_sourceCombo->setCurrentIndex(idx);
}

void FaceLiveDetectWidget::submitInferJob(const QImage& displayRgb,
                                          const QImage& inferRgb,
                                          float inferScale) {
    if (!m_inferWorker || m_inferBusy) return;
    m_inferBusy = true;

    FaceLiveDetectInferWorker::Job job;
    job.displayRgb = displayRgb;
    job.inferRgb = inferRgb;
    job.inferScale = inferScale;
    job.modelPath = m_config.modelPath;
    job.device = m_config.device;
    job.threads = m_config.threads;
    job.minDetectionScore = m_config.minDetectionScore;
    job.matchThreshold =
            m_recognizeThreshold
                    ? static_cast<float>(m_recognizeThreshold->value())
                    : m_config.recognizeMaxDistance;
    job.streamMode = m_config.streamMode == StreamMode::Recognize
                             ? FaceLiveDetectInferWorker::StreamMode::Recognize
                             : FaceLiveDetectInferWorker::StreamMode::Detect;
    job.registry = m_config.registry;

    QMetaObject::invokeMethod(m_inferWorker, "runJob", Qt::QueuedConnection,
                              Q_ARG(FaceLiveDetectInferWorker::Job, job));
}

void FaceLiveDetectWidget::onInferComplete(
        FaceLiveDetectInferWorker::Result result) {
    m_inferBusy = false;
    if (!m_streamActive) return;
    if (!result.ok) {
        m_statusLabel->setText(tr("Inference failed (check model path)."));
        return;
    }

    m_lastSnapshot = result.snapshot;
    m_hasSnapshot = true;
    if (m_captureBtn) m_captureBtn->setEnabled(!result.snapshot.faces.empty());

    QImage display = result.displayImage;
    if (display.isNull() && !result.snapshot.annotatedImage.isNull()) {
        display = result.snapshot.annotatedImage;
    }
    if (!display.isNull()) {
        m_lastSnapshot.annotatedImage = display;
        m_previewLabel->setPreviewImage(display, m_previewLabel->size());
    }

    emit snapshotUpdated(m_lastSnapshot);

    if (m_config.streamMode == StreamMode::Recognize) {
        const int identified = result.identifiedCount;
        const int total = static_cast<int>(result.snapshot.faces.size());
        m_statusLabel->setText(
                tr("Recognize — %1 face(s), %2 identified, %3 unknown, "
                   "match dist ≤ %4")
                        .arg(total)
                        .arg(identified)
                        .arg(total - identified)
                        .arg(m_config.recognizeMaxDistance, 0, 'f', 2));
    } else {
        m_statusLabel->setText(
                tr("Detect — %1 above min score (%2 detected), min score %3")
                        .arg(result.snapshot.faces.size())
                        .arg(result.snapshot.totalDetected)
                        .arg(m_config.minDetectionScore, 0, 'f', 2));
    }
}

void FaceLiveDetectWidget::processFrame() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_streamActive || !m_capture.isOpened()) return;
    cv::Mat frame;
    if (!m_capture.read(frame) || frame.empty()) {
        if (m_sourceCombo && m_sourceCombo->currentData().toInt() ==
                                     static_cast<int>(InputSource::VideoFile)) {
            m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!m_capture.read(frame) || frame.empty()) {
                stopStream();
                m_statusLabel->setText(tr("Video finished"));
                return;
            }
        } else {
            return;
        }
    }

    QImage rgb = cvMatToQImage(frame);
    if (rgb.isNull()) return;

    QImage display = rgb;
    if (m_inferSkip <= 0) {
        if (!m_inferBusy) {
            constexpr int kMaxInferDim = 640;
            QImage inferRgb = rgb;
            float inferScale = 1.f;
            const int maxDim = std::max(rgb.width(), rgb.height());
            if (maxDim > kMaxInferDim) {
                inferScale = static_cast<float>(kMaxInferDim) / maxDim;
                inferRgb = rgb.scaled(
                        static_cast<int>(std::lround(rgb.width() * inferScale)),
                        static_cast<int>(
                                std::lround(rgb.height() * inferScale)),
                        Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            }
            submitInferJob(rgb, inferRgb, inferScale);
        }
        const bool video = m_sourceCombo &&
                           m_sourceCombo->currentData().toInt() ==
                                   static_cast<int>(InputSource::VideoFile);
        m_inferSkip = video ? 4 : 2;
    } else {
        --m_inferSkip;
        if (m_hasSnapshot && !m_lastSnapshot.annotatedImage.isNull()) {
            display = m_lastSnapshot.annotatedImage;
        }
    }

    m_previewLabel->setPreviewImage(display, m_previewLabel->size());
#endif
}

bool FaceLiveDetectWidget::startCamera(int deviceIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    stopStream();
    if (m_config.modelPath.isEmpty() ||
        !QFileInfo::exists(m_config.modelPath)) {
        emit logMessage(
                tr("[Live] Set a detector GGUF on the Image / Batch tab."));
        return false;
    }
    if (!m_camerasEnumerated && m_cameraCombo) {
        m_camerasEnumerated = true;
        m_cameraCombo->clear();
        for (int i = 0; i < 8; ++i) {
            cv::VideoCapture test(i, cv::CAP_ANY);
            if (test.isOpened()) {
                m_cameraCombo->addItem(tr("Camera %1").arg(i), i);
                test.release();
            }
        }
        if (m_cameraCombo->count() == 0) {
            m_cameraCombo->addItem(tr("No camera"), -1);
            return false;
        }
    }
    if (!m_capture.open(deviceIndex, cv::CAP_ANY)) {
        emit logMessage(tr("[Live] Failed to open camera %1").arg(deviceIndex));
        return false;
    }
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    m_streamActive = true;
    m_inferSkip = 0;
    m_frameTimer->setInterval(33);
    m_frameTimer->start();
    m_statusLabel->setText(tr("Camera active"));
    emit streamStarted();
    return true;
#else
    Q_UNUSED(deviceIndex);
    return false;
#endif
}

bool FaceLiveDetectWidget::startVideoFile(const QString& path) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    stopStream();
    if (m_config.modelPath.isEmpty() ||
        !QFileInfo::exists(m_config.modelPath)) {
        emit logMessage(
                tr("[Live] Set a detector GGUF on the Image / Batch tab."));
        return false;
    }
    if (!m_capture.open(path.toStdString(), cv::CAP_FFMPEG) &&
        !m_capture.open(path.toStdString(), cv::CAP_ANY)) {
        const QString err =
                tr("Failed to open video (rebuild OpenCV with FFmpeg): %1")
                        .arg(path);
        emit logMessage(tr("[Live] %1").arg(err));
        m_statusLabel->setText(err);
        return false;
    }
    m_streamActive = true;
    m_inferSkip = 0;
    m_frameTimer->setInterval(66);
    m_frameTimer->start();
    m_statusLabel->setText(tr("Playing video"));
    emit streamStarted();
    return true;
#else
    Q_UNUSED(path);
    return false;
#endif
}

void FaceLiveDetectWidget::stopStream() {
    if (m_frameTimer) m_frameTimer->stop();
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_capture.isOpened()) m_capture.release();
#endif
    if (m_streamActive) {
        m_streamActive = false;
        emit streamStopped();
    }
    m_inferBusy = false;
    m_hasSnapshot = false;
    if (m_captureBtn) m_captureBtn->setEnabled(false);
}

bool FaceLiveDetectWidget::isActive() const { return m_streamActive; }

bool FaceLiveDetectWidget::hasSnapshot() const { return m_hasSnapshot; }

FaceDetectRunResult FaceLiveDetectWidget::lastSnapshot() const {
    return m_lastSnapshot;
}

FaceLiveDetectWidget::InputSource FaceLiveDetectWidget::inputSource() const {
    if (!m_sourceCombo) return InputSource::Camera;
    return static_cast<InputSource>(m_sourceCombo->currentData().toInt());
}

int FaceLiveDetectWidget::selectedCameraIndex() const {
    if (!m_cameraCombo || m_cameraCombo->count() == 0) return 0;
    return m_cameraCombo->currentData().toInt();
}

QString FaceLiveDetectWidget::videoFilePath() const {
    return m_videoPathEdit ? m_videoPathEdit->text().trimmed() : QString();
}

void FaceLiveDetectWidget::captureSnapshotToDb() {
    if (!m_hasSnapshot || m_lastSnapshot.annotatedImage.isNull()) {
        emit logMessage(tr("[Live] No annotated frame to capture."));
        return;
    }
    emit captureToDbRequested(m_lastSnapshot);
}
