// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceLiveDetectWidget.h"

#include <QtCompat.h>
#include <cvFileDialog.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QMutex>
#include <QPainter>
#include <QPixmapCache>
#include <QSettings>
#include <QShowEvent>
#include <QTimer>
#include <QtConcurrent>
#include <QtMath>
#include <atomic>
#include <cmath>

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
// ---------------------------------------------------------------------------
// VideoFrameReader moved to the shared video_base module
// (VideoPlaybackWidget + VideoFrameReader).  See plugins/core/Standard/
// video_base/include/VideoFrameReader.h.
// ---------------------------------------------------------------------------
#endif

}  // namespace

bool FaceLiveDetectWidget::isAvailable() {
    return VideoPlaybackWidget::isAvailable();
}

FaceLiveDetectWidget::FaceLiveDetectWidget(QWidget* parent)
    : VideoPlaybackWidget(parent) {
    // video_base owns the playback panel; cache the base preview/status
    // widgets so existing code keeps working.
    m_previewLabel = previewLabel();
    m_statusLabel = statusLabel();
    setPreviewFixedHeight(300);  // keep the tab geometry stable

    setupUi();
    m_inferThread = new QThread(this);
    m_inferWorker = new FaceLiveDetectInferWorker;
    m_inferWorker->moveToThread(m_inferThread);
    connect(m_inferThread, &QThread::finished, m_inferWorker,
            &QObject::deleteLater);
    connect(m_inferWorker, &FaceLiveDetectInferWorker::inferComplete, this,
            &FaceLiveDetectWidget::onInferComplete, Qt::QueuedConnection);
    connect(
            m_inferWorker, &FaceLiveDetectInferWorker::modelPreloadComplete,
            this,
            [this](bool ok) {
                m_preloadingModel = false;
                if (m_preloadProgress) {
                    m_preloadProgress->setMaximum(100);
                    m_preloadProgress->setValue(0);
                    m_preloadProgress->setTextVisible(false);
                }
                if (!ok || !isActive()) {
                    if (!ok && m_statusLabel) {
                        m_statusLabel->setText(
                                tr("Model loading failed (check model path)."));
                    }
                    return;
                }
            },
            Qt::QueuedConnection);
    m_inferThread->start();

    loadSettings();
}

FaceLiveDetectWidget::~FaceLiveDetectWidget() {
    saveSettings();
    stopStream();  // video_base owns the reader thread teardown
    shutdownInferThread();
}

void FaceLiveDetectWidget::shutdownInferThread() {
    if (!m_inferWorker || !m_inferThread) {
        return;
    }
    // QThread::finished is emitted from the worker thread itself during its
    // teardown. Because m_inferWorker lives on that thread, the queued
    // deleteLater connection is delivered as a DIRECT call before the event
    // loop stops draining — ownership of the worker is then split between
    // this function and the deferred-delete machinery. Drop the connection
    // first so this function is the sole owner of the worker's lifetime (the
    // same contract as the qRMBG/qRFDetr live widgets).
    disconnect(m_inferThread, &QThread::finished, m_inferWorker,
               &QObject::deleteLater);
    // releaseModel runs synchronously on the worker thread, so quit() below is
    // guaranteed to end the event loop; wait() can therefore never time out
    // (its upper bound is the single in-flight inference, which cannot be
    // interrupted). A bounded wait here would instead risk leaking the worker
    // when the wait times out early.
    QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                              Qt::BlockingQueuedConnection);
    m_inferThread->quit();
    m_inferThread->wait();
    delete m_inferWorker;
    m_inferWorker = nullptr;
}

void FaceLiveDetectWidget::setupUi() {
    // The playback panel (preview + source selection + seek/speed controls)
    // is built by the video_base base class; this method only appends the
    // inference-specific controls below the base layout.
    auto* main = mainLayout();

    m_preloadProgress = new QProgressBar(this);
    m_preloadProgress->setFixedHeight(18);
    m_preloadProgress->setTextVisible(false);
    m_preloadProgress->setMaximum(100);
    m_preloadProgress->setValue(0);
    m_preloadProgress->setSizePolicy(QSizePolicy::Expanding,
                                     QSizePolicy::Fixed);
    main->addWidget(m_preloadProgress);

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
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Device:")), 1, 0);
    grid->addWidget(m_deviceCombo, 1, 1);
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Threads:")), 1, 2);
    grid->addWidget(m_threadsSpin, 1, 3, Qt::AlignLeft);

    m_modeCombo = new QComboBox(settingsGroup);
    m_modeCombo->addItem(tr("Detect faces only"),
                         static_cast<int>(StreamMode::Detect));
    m_modeCombo->addItem(tr("Recognize (Registry DB)"),
                         static_cast<int>(StreamMode::Recognize));
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceLiveDetectWidget::onStreamModeChanged);
    grid->addWidget(FaceDetectUi::makeFormLabel(tr("Mode:")), 2, 0);
    grid->addWidget(m_modeCombo, 2, 1);

    m_minScoreLabel = new QLabel(tr("Min score:"), settingsGroup);
    m_minDetectionScore = FaceDetectUi::makeMinDetectionScoreSpin(
            settingsGroup,
            tr("Faces below this detection score are drawn in red and excluded "
               "from capture/export."));
    m_matchDistLabel = new QLabel(tr("Match dist:"), settingsGroup);
    m_recognizeThreshold = new QDoubleSpinBox(settingsGroup);
    m_recognizeThreshold->setRange(0.05, 1.0);
    m_recognizeThreshold->setSingleStep(0.01);
    m_recognizeThreshold->setValue(0.65);
    FaceDetectUi::makeCompactDoubleSpin(m_recognizeThreshold);
    m_recognizeThreshold->setToolTip(
            tr("Max cosine distance for registry match (lower = stricter)."));
    grid->addWidget(m_minScoreLabel, 3, 0);
    grid->addWidget(m_minDetectionScore, 3, 1, Qt::AlignLeft);
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
        const QString path = cvFileDialog::getOpenFileName(
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

    main->addWidget(settingsGroup);

    // Sample-data button (independent row; visible in video-file mode).
    m_testDataBtn = new QPushButton(tr("\U0001f9ea  Try sample data"), this);
    m_testDataBtn->setToolTip(
            tr("Download FriendsFaces sample video and load it here."));
    // Prominent teal accent — consistent with qFreeSplatter / batch tab.
    m_testDataBtn->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold; border: none; border-radius: 4px; padding: 5px 12px; }"
            "QPushButton:hover { background: #00796b; }"
            "QPushButton:pressed { background: #00695c; }"
            "QPushButton:disabled { background: #b2dfdb; color: #e0f2f1; }");
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &FaceLiveDetectWidget::testDataRequested);
    m_testDataBtn->setVisible(false);  // shown when video-file source active
    main->addWidget(m_testDataBtn);

    m_captureBtn = new QPushButton(tr("Capture frame to DB"), this);
    m_captureBtn->setEnabled(false);
    connect(m_captureBtn, &QPushButton::clicked, this,
            &FaceLiveDetectWidget::captureSnapshotToDb);
    main->addWidget(m_captureBtn);

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
                                   0.65))
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
            if (videoPathEdit()) videoPathEdit()->setText(legacy);
        }
    } else {
        m_videoPathUserChosen = true;
        if (videoPathEdit()) videoPathEdit()->setText(videoPath);
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
    if (sourceCombo()) {
        const int idx = sourceCombo()->findData(source);
        if (idx >= 0) {
            sourceCombo()->blockSignals(true);
            sourceCombo()->setCurrentIndex(idx);
            sourceCombo()->blockSignals(false);
            onSourceChanged(static_cast<InputSource>(source));
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
    if (sourceCombo()) {
        settings.setValue(QStringLiteral("qFaceDetect/liveSource"),
                          sourceCombo()->currentData());
    }
    if (m_videoPathUserChosen && videoPathEdit()) {
        const QString path = videoPathEdit()->text().trimmed();
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

void FaceLiveDetectWidget::setVideoFilePath(const QString& path,
                                            bool userChosen) {
    m_videoPathUserChosen = userChosen;
    VideoPlaybackWidget::setVideoFilePath(path);
}

void FaceLiveDetectWidget::submitInferJob(const QImage& inferRgb,
                                          float inferScale) {
    if (!m_inferWorker || m_inferBusy) return;
    m_inferBusy = true;

    FaceLiveDetectInferWorker::Job job;
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
    job.generation = m_streamGeneration;

    QMetaObject::invokeMethod(m_inferWorker, "runJob", Qt::QueuedConnection,
                              Q_ARG(FaceLiveDetectInferWorker::Job, job));
    m_inferSubmitTime.start();
}

void FaceLiveDetectWidget::onInferComplete(
        FaceLiveDetectInferWorker::Result result) {
    // Always reset the busy flag first — even for stale results.
    // If we return early (generation mismatch) without resetting, m_inferBusy
    // stays true forever and no new inference jobs are ever submitted,
    // effectively freezing the detection overlay until a manual stop+start.
    m_inferBusy = false;
    if (result.generation != m_streamGeneration) return;
    if (!isActive()) return;

    // Calculate inference latency.
    m_lastInferLatencyMs =
            m_inferSubmitTime.isValid() ? m_inferSubmitTime.elapsed() : 0;
    m_overlayTimestampMs = QDateTime::currentMSecsSinceEpoch();

    if (!result.ok) {
        m_statusLabel->setText(tr("Inference failed (check model path)."));
        return;
    }

    m_lastSnapshot = result.snapshot;
    m_hasSnapshot = true;
    if (m_captureBtn) m_captureBtn->setEnabled(!result.snapshot.faces.empty());

    // Cache overlay data for persistent display (prevents flicker).
    // snapshot.faces are at display resolution (same as annotatedImage).
    m_overlayFaces = result.snapshot.faces;
    m_overlayInferSize = result.snapshot.annotatedImage.size();
    m_overlayLabels = result.labels;
    // Record which video frame this overlay corresponds to.
    m_overlayFrameNum = m_lastSubmitFrameNum;

    // Keep annotated image in snapshot for batch / export consumers,
    // but do NOT push it to the preview label — drawLiveOverlay handles
    // all live rendering and avoids double-drawing / flicker.
    if (!result.snapshot.annotatedImage.isNull()) {
        m_lastSnapshot.annotatedImage = result.snapshot.annotatedImage;
    }

    emit snapshotUpdated(m_lastSnapshot);

    // Refresh the preview immediately so the overlay (boxes / labels)
    // appears as soon as inference completes, instead of waiting for
    // the next decoded frame.  Without this, slow inference (common on
    // CPU) could delay the first visible result by seconds.
    if (!m_lastDisplayFrame.isNull() && !m_overlayFaces.empty()) {
        QImage refreshed = m_lastDisplayFrame.copy();
        drawLiveOverlay(refreshed);
        if (m_previewLabel) {
            m_previewLabel->setPixmap(QPixmap::fromImage(refreshed));
        }
    }

    if (m_config.streamMode == StreamMode::Recognize) {
        const int identified = result.identifiedCount;
        const int total = static_cast<int>(result.snapshot.faces.size());
        m_statusLabel->setText(
                tr("Recognize \u2014 %1 face(s), %2 identified, %3 unknown, "
                   "match dist \u2264 %4, latency %5 ms")
                        .arg(total)
                        .arg(identified)
                        .arg(total - identified)
                        .arg(m_config.recognizeMaxDistance, 0, 'f', 2)
                        .arg(m_lastInferLatencyMs));
    } else {
        m_statusLabel->setText(
                tr("Detect \u2014 %1 above min score (%2 detected), "
                   "min score %3, latency %4 ms")
                        .arg(result.snapshot.faces.size())
                        .arg(result.snapshot.totalDetected)
                        .arg(m_config.minDetectionScore, 0, 'f', 2)
                        .arg(m_lastInferLatencyMs));
    }
}

void FaceLiveDetectWidget::onFrameDecoded(cv::Mat& frame, int frameIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    // Record the original frame size so onDisplayFrame can scale overlay
    // coordinates from original resolution to the displayed resolution.
    m_lastFrameSize = QSize(frame.cols, frame.rows);

    // Submit inference throttled by video-time, not wall-clock.
    // At any playback speed, submit once per ~2 video frames of content.
    if (!m_inferBusy) {
        const bool isVideoFile = inputSource() == InputSource::VideoFile;
        const bool shouldSubmit = [&]() -> bool {
            if (!isVideoFile || videoFps() <= 0)
                return true;  // camera: every tick
            // Throttle: submit when ≥2 video frames have elapsed since last
            // submit.
            const double videoTimeMs = frameIndex / videoFps() * 1000.0;
            const double lastSubmitMs =
                    m_lastSubmitFrameNum / videoFps() * 1000.0;
            const double thresholdMs = 2.0 / videoFps() * 1000.0;  // 2 frames
            return (videoTimeMs - lastSubmitMs) >= thresholdMs - 1.0;
        }();
        if (shouldSubmit) {
            // Convert only when a job is actually submitted — cvMatToQImage
            // is a full-frame BGR→RGB copy; on busy-inference or throttled
            // frames the conversion is pure waste.
            QImage rgb = VideoPlaybackWidget::cvMatToQImage(frame);
            if (rgb.isNull()) return;
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
                        Qt::IgnoreAspectRatio, Qt::FastTransformation);
            }
            m_lastSubmitFrameNum = frameIndex;
            submitInferJob(inferRgb, inferScale);
        }
    }
#endif
}

void FaceLiveDetectWidget::onDisplayFrame(QImage& display, int frameIndex) {
    Q_UNUSED(frameIndex);
    // Cache the display frame so onInferComplete can redraw the overlay
    // immediately without waiting for the next frame to arrive.
    m_lastDisplayFrame = display;
    drawLiveOverlay(display);
}

void FaceLiveDetectWidget::drawLiveOverlay(QImage& frame) {
    if (m_overlayFaces.empty() || m_overlayInferSize.isEmpty()) return;
    if (frame.isNull()) return;

    const qreal sx = static_cast<qreal>(frame.width()) /
                     static_cast<qreal>(m_overlayInferSize.width());
    const qreal sy = static_cast<qreal>(frame.height()) /
                     static_cast<qreal>(m_overlayInferSize.height());

    // Overlay freshness: opacity decreases as overlay ages.
    // Use speed-adjusted age so the overlay fades proportionally to how much
    // video content has passed, not just wall-clock time.
    // At 4× speed, 200ms wall-clock = 800ms of video content → fade faster.
    // 0–200 ms → full opacity;  200–1000 ms → linear fade to 0.4;  >1 s → 0.4.
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    const qint64 wallClockAgeMs =
            (m_overlayTimestampMs > 0) ? (now - m_overlayTimestampMs) : 0;
    const qint64 ageMs = static_cast<qint64>(wallClockAgeMs *
                                             std::max(1.0, playbackSpeed()));
    qreal overlayAlpha = 1.0;
    if (ageMs > 200) {
        overlayAlpha = qBound(0.4, 1.0 - (ageMs - 200) / 800.0, 1.0);
    }
    const int penAlpha = static_cast<int>(255 * overlayAlpha);
    const int bgAlpha = static_cast<int>(180 * overlayAlpha);

    QPainter painter(&frame);
    painter.setRenderHint(QPainter::Antialiasing);

    const bool isRecognize = (m_config.streamMode == StreamMode::Recognize);
    const int fontSize = std::max(9, frame.height() / 55);
    const int pad = std::max(3, frame.height() / 160);
    QFont font(QStringLiteral("sans-serif"), fontSize);
    font.setBold(true);
    painter.setFont(font);
    const QFontMetrics fm(font);
    const int textHeight = fm.height();
    const int penW = std::max(2, frame.height() / 240);

    for (int i = 0; i < static_cast<int>(m_overlayFaces.size()); ++i) {
        const auto& face = m_overlayFaces[i];
        const QRectF box(face.x1 * sx, face.y1 * sy, (face.x2 - face.x1) * sx,
                         (face.y2 - face.y1) * sy);

        // Face rectangle
        QPen pen(isRecognize ? QColor(0, 200, 255, penAlpha)
                             : QColor(0, 255, 0, penAlpha));
        pen.setWidth(penW);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRoundedRect(box, 2.0, 2.0);

        // Label text
        QString text;
        if (isRecognize && i < m_overlayLabels.size() &&
            !m_overlayLabels[i].isEmpty()) {
            text = m_overlayLabels[i];
        } else if (face.score > 0.f) {
            text = QString::number(face.score, 'f', 2);
        }
        if (text.isEmpty()) continue;

        // Text-width-adaptive background, clamped to image bounds
        const qreal textW = QTCOMPAT_FONTMETRICS_WIDTH(fm, text);
        qreal bgW = textW + 2.0 * pad;
        bgW = qBound(bgW, box.width(), static_cast<qreal>(frame.width()));
        const qreal bgH = textHeight + 2.0 * pad;

        qreal bgX = box.x();
        qreal bgY = box.y() - bgH;
        // Clamp horizontal
        if (bgX + bgW > frame.width()) bgX = frame.width() - bgW;
        if (bgX < 0.0) bgX = 0.0;
        // If not enough room above box, place below top edge
        if (bgY < 0.0) bgY = box.y() + penW;

        const QRectF bgRect(bgX, bgY, bgW, bgH);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, bgAlpha));
        painter.drawRoundedRect(bgRect, 2.0, 2.0);
        painter.setPen(QColor(255, 255, 255, penAlpha));
        painter.drawText(
                QRectF(bgX + pad, bgY + pad, bgW - 2.0 * pad, bgH - 2.0 * pad),
                Qt::AlignLeft | Qt::AlignVCenter, text);
    }
}

bool FaceLiveDetectWidget::onPrepareStream() {
    // The base class opens the capture right after this hook returns true;
    // reject the start when no detector GGUF is configured (mirrors the
    // historical startCamera / startVideoFile validation).
    if (m_config.modelPath.isEmpty() ||
        !QFileInfo::exists(m_config.modelPath)) {
        emit logMessage(
                tr("[Live] Set a detector GGUF on the Image / Batch tab."));
        return false;
    }
    // Warm the inference worker in parallel with the stream start so the
    // first decoded frames already get an overlay.  The worker falls back
    // to lazy loading inside ensureModel() if this never runs.
    if (m_preloadProgress) {
        m_preloadProgress->setMaximum(0);  // indeterminate
        m_preloadProgress->setTextVisible(true);
        m_preloadProgress->setFormat(tr("Loading model…"));
        m_preloadingModel = true;
    }
    QMetaObject::invokeMethod(
            m_inferWorker, "preloadModel", Qt::QueuedConnection,
            Q_ARG(QString, m_config.modelPath), Q_ARG(QString, m_config.device),
            Q_ARG(int, m_config.threads));
    return true;
}

void FaceLiveDetectWidget::onStreamStopping() {
    // Invalidate in-flight inference results and reset the worker-side UI
    // state (mirrors the historical stopStream() inference cleanup).
    ++m_streamGeneration;
    m_preloadingModel = false;
    if (m_preloadProgress) {
        m_preloadProgress->setMaximum(100);
        m_preloadProgress->setValue(0);
        m_preloadProgress->setTextVisible(false);
    }
    m_inferBusy = false;
    m_hasSnapshot = false;
    m_overlayFaces.clear();
    m_overlayLabels.clear();
    m_overlayInferSize = QSize();
    m_overlayFrameNum = 0;
    m_lastSubmitFrameNum = 0;
    m_lastInferLatencyMs = 0;
    m_overlayTimestampMs = 0;
    m_inferSubmitTime = QElapsedTimer();
    if (m_captureBtn) m_captureBtn->setEnabled(false);
}

void FaceLiveDetectWidget::onStreamReset() {
    // Restart resets ALL inference state so the pipeline restarts fresh:
    // - Bump streamGeneration to invalidate stale async inference results
    // - Clear overlays so old boxes/labels disappear immediately
    // - Force m_inferBusy=false so new inference can be submitted
    ++m_streamGeneration;
    m_overlayFaces.clear();
    m_overlayLabels.clear();
    m_overlayInferSize = QSize();
    m_inferBusy = false;
    m_lastSubmitFrameNum = -1;  // -1 forces submission on the very first frame
}

void FaceLiveDetectWidget::onStreamResumed() {
    // Resume from paused state — allow new inference submissions.
    m_inferBusy = false;
}

void FaceLiveDetectWidget::onVideoLooped() {
    // EOF loop: stale overlays belong to the old playback position; force
    // fresh inference from the first frame of the new cycle.
    m_overlayFaces.clear();
    m_overlayLabels.clear();
    m_overlayInferSize = QSize();
    m_inferBusy = false;
    m_lastSubmitFrameNum = 0;
}

void FaceLiveDetectWidget::onSourceChanged(InputSource source) {
    // The sample-data helper only makes sense while a video file is loaded.
    if (m_testDataBtn) {
        m_testDataBtn->setVisible(source == InputSource::VideoFile);
    }
}

bool FaceLiveDetectWidget::hasSnapshot() const { return m_hasSnapshot; }

FaceDetectRunResult FaceLiveDetectWidget::lastSnapshot() const {
    return m_lastSnapshot;
}

void FaceLiveDetectWidget::captureSnapshotToDb() {
    if (!m_hasSnapshot || m_lastSnapshot.annotatedImage.isNull()) {
        emit logMessage(tr("[Live] No annotated frame to capture."));
        return;
    }
    emit captureToDbRequested(m_lastSnapshot);
}
