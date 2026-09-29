// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoTab.h"

#include <CVLog.h>
#include <aicore/backend_capi.h>
#include <aicore/sam3_capi.h>
#include <ecvAICoreUiHelper.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPainter>
#include <QProgressBar>
#include <QProgressDialog>
#include <QPushButton>
#include <QRadioButton>
#include <QSaveFile>
#include <QSlider>
#include <QVBoxLayout>
#include <algorithm>

#include "VideoFrameReader.h"
#include "VideoMaskArchive.h"
#include "VideoPlaybackWidget.h"  // cvMatToQImage
#include "ecvModelDownloader.h"

namespace {
constexpr const char* kTabModelKeys[] = {"sam", "sam-visual", "sam2"};
constexpr int kNumTabModelKeys = 3;

// Per-instance display colors, mirroring upstream examples/main_video.cpp.
const QColor kInstanceColors[] = {
        QColor(255, 51, 51),  QColor(51, 153, 255),  QColor(51, 230, 76),
        QColor(255, 204, 26), QColor(204, 76, 230),  QColor(255, 128, 26),
        QColor(26, 230, 230), QColor(230, 102, 153), QColor(128, 204, 51),
        QColor(76, 76, 255),  QColor(255, 153, 179), QColor(153, 255, 128),
};
constexpr int kNumColors = sizeof(kInstanceColors) / sizeof(kInstanceColors[0]);

const aicore_sam3_model_entry* catalogEntry(const QString& filename) {
    return aicore_sam3_model_by_filename(filename.toUtf8().constData());
}

bool isValidCatalogModel(const QString& path, const QString& filename) {
    const auto* entry = catalogEntry(filename);
    // Presence check: integrity-ledger stat-trust (digest pinned per
    // release asset), falling back to the catalog's exact size for files
    // downloaded before the ledger existed. Never hashes (multi-GB GGUFs
    // on dialog paths).
    return entry &&
           ecvAssetIntegrity::isVerified(
                   path,
                   {QCryptographicHash::Sha256,
                    ecvAssetIntegrity::PinnedDigest(
                            QString::fromUtf8(entry->filename))},
                   64 * 1024, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
                   entry->size_bytes);
}

// Why the selected catalog model cannot be loaded as-is. modelPath()
// falls back to the bare filename when the release check fails, so
// prompts must classify before claiming the GGUF "was not downloaded".
enum class ModelFileStatus { Missing, Unverified, Verified };

QString cachedModelPath(const QString& filename) {
    char* dir = aicore_sam3_model_cache_dir();
    const QString cacheDir = QString::fromUtf8(dir);
    aicore_sam3_free_buffer(dir);
    return cacheDir + QLatin1Char('/') + filename;
}

QString modelDisplayPath(const QString& filename) {
    const QFileInfo selected(filename);
    return selected.isAbsolute() ? filename : cachedModelPath(filename);
}

ModelFileStatus classifyModelFile(const QString& filename) {
    if (filename.isEmpty() || filename == "__browse__") {
        return ModelFileStatus::Missing;
    }
    const QFileInfo selected(filename);
    // An explicitly browsed absolute path bypasses the release check.
    if (selected.isAbsolute()) {
        return selected.isFile() ? ModelFileStatus::Verified
                                 : ModelFileStatus::Missing;
    }
    const QString full = cachedModelPath(filename);
    if (!QFileInfo(full).isFile()) return ModelFileStatus::Missing;
    return isValidCatalogModel(full, filename) ? ModelFileStatus::Verified
                                               : ModelFileStatus::Unverified;
}
}  // namespace

VideoTab::VideoTab(QWidget* parent) : QWidget(parent) {
    setupUi();
    populateModelCombo();

    // Shared model downloader (qDA3-style): catalog GGUFs land in the AICore
    // model cache (sam3_models), are verified once at ingestion (exact size
    // compared at finalize) and stat-trusted afterwards.
    m_modelDownloader = new ecvModelDownloader(this);
    connect(m_modelDownloader, &ecvModelDownloader::logMessage, this,
            &VideoTab::appendLog);
    connect(m_modelDownloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (total > 0 && m_progress) {
                    m_progress->setValue(
                            static_cast<int>(received * 100 / total));
                }
                if (m_downloadLabel) {
                    m_downloadLabel->setText(
                            tr("Downloading %1 — %2")
                                    .arg(m_downloadTargetFilename)
                                    .arg(ecvModelDownloader::
                                                 formatDownloadProgress(
                                                         received, total)));
                }
            });
    connect(m_modelDownloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                const bool thenRun = m_downloadThenRun;
                const QString filename = m_downloadTargetFilename;
                const bool valid = ok && isValidCatalogModel(dest, filename);
                m_downloadInProgress = false;
                m_downloadThenRun = false;
                if (m_downloadLabel) m_downloadLabel->setVisible(false);
                if (m_progress) m_progress->setVisible(false);
                refreshModelCombo();
                updateDownloadButton();
                if (valid) {
                    m_downloadPrompted = false;
                    appendLog(tr("Model downloaded: %1").arg(filename));
                    if (thenRun) ensureModelReady();
                } else {
                    appendLog(tr("Model download failed or did not pass "
                                 "catalog validation: %1")
                                      .arg(filename));
                }
                m_downloadTargetFilename.clear();
            });

    // Shared test data repository (SAM3 dataset: images + videos).
    auto& repo = ecvTestDataRepository::instance();
    connect(&repo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                if (!m_testDataDownloadInProgress) return;
                setStatus(QString("%1 (%2%)").arg(statusText).arg(percent));
                if (m_progress) m_progress->setValue(percent);
            });
    connect(&repo, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_testDataDownloadInProgress) appendLog(message);
            });
    connect(&repo, &ecvTestDataRepository::downloadFinished, this,
            &VideoTab::onTestDataDownloadFinished);
    connect(&repo, &ecvTestDataRepository::extractionProgress, this,
            [this](int current, int total) {
                if (!m_testDataDownloadInProgress || total <= 0) return;
                setStatus(tr("Extracting SAM3 test data... %1/%2")
                                  .arg(current)
                                  .arg(total));
                if (m_progress && total > 0) {
                    m_progress->setValue(current * 100 / total);
                }
            });
    connect(&repo, &ecvTestDataRepository::extractionFinished, this,
            &VideoTab::onTestDataExtractionFinished);

    // Fill the test-video picker from the (possibly cached) SAM3 dataset.
    populateTestVideoCombo();
}

VideoTab::~VideoTab() {
    releaseModel();
    if (m_reader) {
        m_reader->release();
        delete m_reader;
        m_reader = nullptr;
    }
}

void VideoTab::releaseModel() {
    if (!m_videoExportDir.isEmpty())
        finishVideoExport(false, tr("Model released."));
    if (m_exportVideoBtn) m_exportVideoBtn->setEnabled(false);
    m_playTimer.stop();
    m_playing = false;
    if (m_playBtn) m_playBtn->setText(tr("Play"));
    if (m_worker) {
        m_worker->requestCancel();
        // The worker has a persistent request loop, so cancellation is the
        // normal shutdown path. Do not delete a live QThread.
        if (!m_worker->wait(30000)) {
            appendLog(
                    tr("Timed out while releasing the video model; GPU "
                       "resources remain owned by the worker."));
            return;
        }
        delete m_worker;
        m_worker = nullptr;
    }
    m_loadedModelPath.clear();
    m_loadingModelPath.clear();
    m_loadedDevice.clear();
    m_loadingDevice.clear();
    m_activeTextPrompt.clear();
    m_loadingTextPrompt.clear();
    m_trackerActive = false;
    m_visualOnly = false;
    m_promptResetInFlight = false;
    m_trackerPromptDirty = false;
    m_pendingAction = PendingAction::None;
    m_pendingRetrack = false;
    if (m_backendLabel) m_backendLabel->setText(tr("Backend: none"));
    emit backendChanged(QStringLiteral("none"));
    if (m_stepBtn) m_stepBtn->setEnabled(false);
    if (m_resetBtn) m_resetBtn->setEnabled(false);
    if (m_canvas) m_canvas->setInteractive(false);
    clearTrackingVisualization();
}

void VideoTab::setDevice(const QString& device) {
    if (!m_deviceCombo) return;
    for (int i = 0; i < m_deviceCombo->count(); ++i) {
        if (m_deviceCombo->itemText(i).compare(device, Qt::CaseInsensitive) ==
                    0 ||
            m_deviceCombo->itemData(i).toString().compare(
                    device, Qt::CaseInsensitive) == 0) {
            m_deviceCombo->setCurrentIndex(i);
            return;
        }
    }
}

void VideoTab::setStandaloneDeviceControlsVisible(bool visible) {
    if (m_deviceLabel) m_deviceLabel->setVisible(visible);
    if (m_deviceCombo) m_deviceCombo->setVisible(visible);
    if (m_backendLabel) m_backendLabel->setVisible(visible);
}

QString VideoTab::selectedDevice() const {
    return m_deviceCombo->currentData().toString().toLower();
}

void VideoTab::setupUi() {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 6, 4, 4);
    layout->setSpacing(6);

    // ── Row 1: mode + text prompt + playback controls ────────────────────
    // Two compact rows instead of one overloaded row so buttons never get
    // squeezed or overlapped at narrow dialog widths.
    auto* row1 = new QHBoxLayout();
    auto* modeLabel = new QLabel(tr("Mode:"));
    m_modeText = new QRadioButton(tr("Text"));
    m_modeBox = new QRadioButton(tr("Box"));
    m_modePoints = new QRadioButton(tr("Points"));
    m_modeBox->setChecked(true);
    auto* modeGroup = new QButtonGroup(this);
    modeGroup->addButton(m_modeText, 0);
    modeGroup->addButton(m_modeBox, 1);
    modeGroup->addButton(m_modePoints, 2);
    connect(modeGroup, QOverload<int>::of(&QButtonGroup::buttonClicked), this,
            &VideoTab::onModeChanged);

    m_textPrompt = new QLineEdit();
    m_textPrompt->setPlaceholderText(tr("Text prompt (SAM3 only)..."));
    m_textPrompt->setMinimumWidth(ecvAICoreUi::dpiScaled(140));
    // Enabled state is driven by onModeChanged() below; the initial toggle
    // happens at the end of setupUi() so the text box follows the default
    // "Text" mode.

    m_playBtn = new QPushButton(tr("Play"));
    m_stepBtn = new QPushButton(tr("Step >>"));
    m_stepBtn->setEnabled(false);
    m_resetBtn = new QPushButton(tr("Reset"));
    m_resetBtn->setEnabled(false);

    row1->addWidget(modeLabel);
    row1->addWidget(m_modeText);
    row1->addWidget(m_modeBox);
    row1->addWidget(m_modePoints);
    row1->addSpacing(10);
    row1->addWidget(m_textPrompt, 1);
    row1->addSpacing(10);
    row1->addWidget(m_playBtn);
    row1->addWidget(m_stepBtn);
    row1->addWidget(m_resetBtn);
    layout->addLayout(row1);

    // Row 1b: test-video picker + one-click sample data + open file
    auto* row1b = new QHBoxLayout();
    m_testVideoCombo = new QComboBox();
    m_testVideoCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(140));
    m_testVideoCombo->setToolTip(
            tr("Pick which sample video to test (SAM3 test dataset)"));
    // Switching the picker opens the selected sample video right away
    // (cached dataset; no need to press the sample-data button again).
    connect(m_testVideoCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) {
                if (m_busy) {
                    appendLog(
                            tr("[Test data] Wait for the current task to "
                               "finish before switching videos."));
                    return;
                }
                if (!loadRequestedTestVideo()) {
                    appendLog(
                            tr("[Test data] Sample video not found; run "
                               "'Try sample data' once to download the "
                               "dataset."));
                }
            });
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataBtn->setToolTip(
            tr("Download (cached) and open the selected sample video for "
               "one-click testing"));
    m_openBtn = new QPushButton(tr("Open video..."));

    row1b->addWidget(m_testVideoCombo);
    row1b->addWidget(m_testDataBtn);
    row1b->addSpacing(10);
    row1b->addWidget(m_openBtn);
    row1b->addStretch();
    layout->addLayout(row1b);

    // ── Row 2: model + Load ──────────────────────────────────────────────
    // Device is managed by the shared row in SAM3Dialog's top bar, which
    // forwards changes via setDevice(). The VideoTab keeps hidden device
    // widgets so onDeviceChanged() / setDevice() / onLoadModel() continue
    // to work without a visible duplicate.
    auto* row2 = new QHBoxLayout();
    auto* modelLabel = new QLabel(tr("Model:"));
    m_modelCombo = new QComboBox();
    m_modelCombo->setObjectName(QStringLiteral("SAM3VideoModel"));
    m_modelCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(240));
    m_loadBtn = new QPushButton(tr("Load"));
    m_loadBtn->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold;"
            "  border: none; border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background: #00796b; }");
    // No manual Load step: the model loads automatically on first use
    // (Play / Step / Reset / annotate), matching upstream main_video.cpp.
    // The button is kept hidden for the Browse fallback in onLoadModel().
    m_loadBtn->setVisible(false);

    // SAM3Dialog owns the visible selector; standalone hosts show this row.
    // Device IDs are kept as item data for the AICore call and archive
    // metadata.
    m_deviceLabel = new QLabel(tr("Device:"), this);
    m_deviceLabel->setObjectName(QStringLiteral("SAM3VideoDeviceLabel"));
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->setObjectName(QStringLiteral("SAM3VideoDevice"));
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* dev = aicore_device_at(i);
        if (!dev || !dev->id) continue;
        m_deviceCombo->addItem(
                QString::fromUtf8(dev->label ? dev->label : dev->id),
                QString::fromUtf8(dev->id));
        if (dev->is_default)
            m_deviceCombo->setCurrentIndex(m_deviceCombo->count() - 1);
    }
    m_deviceCombo->hide();
    m_backendLabel = new QLabel(tr("Backend: none"), this);
    m_backendLabel->setObjectName(QStringLiteral("SAM3VideoBackend"));
    m_backendLabel->hide();

    row2->addWidget(modelLabel);
    row2->addWidget(m_modelCombo, 1);
    // One-click download of the selected catalog GGUF into the shared
    // AICore model cache (sam3_models), qDA3-style.
    m_downloadBtn = new QPushButton(tr("Download"));
    m_downloadBtn->setToolTip(
            tr("Download the selected model GGUF into the model cache"));
    connect(m_downloadBtn, &QPushButton::clicked, this, [this]() {
        if (!m_downloadInProgress) startDownload(false);
    });
    row2->addWidget(m_downloadBtn);
    row2->addWidget(m_loadBtn);
    row2->addWidget(m_deviceLabel);
    row2->addWidget(m_deviceCombo);
    row2->addWidget(m_backendLabel);
    row2->addStretch();
    layout->addLayout(row2);

    // ── Canvas (fills the remaining tab space; no height cap) ────────────
    m_canvas = new VideoCanvas();
    m_canvas->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    layout->addWidget(m_canvas, 1);

    // ── Timeline ──────────────────────────────────────────────────────────
    m_timeline = new VideoTimeline();
    layout->addWidget(m_timeline);

    // ── Test-data download progress (hidden by default) ───────────────────
    ecvAICoreUi::setupProgressSection(layout, m_downloadLabel, m_progress);

    // ── Bottom rows ─────────────────────────────────────────────────────
    // Two rows instead of one overloaded row: eleven widgets on a single
    // line squeezed the Export buttons until their labels clipped at the
    // minimum dialog width.
    auto* bottom1 = new QHBoxLayout();
    m_showMasks = new QCheckBox(tr("Show masks"));
    m_showMasks->setChecked(true);

    auto* scoreLabel = new QLabel(tr("Score threshold:"));
    m_scoreSpin = new QDoubleSpinBox();
    m_scoreSpin->setRange(0.0, 1.0);
    m_scoreSpin->setDecimals(2);
    m_scoreSpin->setSingleStep(0.05);
    m_scoreSpin->setValue(0.5);
    m_scoreSpin->setToolTip(
            tr("Minimum text-detection score used by the video tracker"));
    ecvAICoreUi::setCompactDoubleSpin(m_scoreSpin);

    m_exportToDbCheckBox = new QCheckBox(tr("Export to DB"));
    m_exportToDbCheckBox->setChecked(false);  // opt-in, not automatic
    m_exportToDbCheckBox->setToolTip(
            tr("Automatically add the segmented result to the DB tree as "
               "an annotated image"));
    m_exportToDbCheckBox->hide();  // replaced by the Export to DB button below

    auto* speedLabel = new QLabel(tr("Speed:"));
    m_speedSlider = new QSlider(Qt::Horizontal);
    m_speedSlider->setRange(1, 40);  // 0.1x..4.0x, 0.1 steps
    m_speedSlider->setValue(10);
    m_speedSlider->setMaximumWidth(120);
    m_speedLabel = new QLabel(tr("1.0x"));
    m_speedLabel->setMinimumWidth(40);

    bottom1->addWidget(m_showMasks);
    bottom1->addWidget(m_exportToDbCheckBox);
    bottom1->addSpacing(12);
    bottom1->addWidget(scoreLabel);
    bottom1->addWidget(m_scoreSpin);
    bottom1->addSpacing(12);
    bottom1->addWidget(speedLabel);
    bottom1->addWidget(m_speedSlider);
    bottom1->addWidget(m_speedLabel);
    bottom1->addStretch();
    layout->addLayout(bottom1);

    auto* bottom2 = new QHBoxLayout();
    m_exportBtn = new QPushButton(tr("Export frame masks"));
    m_exportBtn->setEnabled(false);
    m_exportBtn->setToolTip(
            tr("Export the current frame's per-instance masks as grayscale "
               "images into the DB tree"));

    m_exportFrameToDbBtn = new QPushButton(tr("Export frame to DB"));
    m_exportFrameToDbBtn->setEnabled(false);
    m_exportFrameToDbBtn->setToolTip(
            tr("Export the current annotated frame to the DB tree (one-shot)"));

    m_exportVideoBtn = new QPushButton(tr("Track and export video..."));
    m_exportVideoBtn->setEnabled(false);
    m_exportVideoBtn->setToolTip(
            tr("Track from the current frame through the end; save each "
               "instance mask and frame index to a folder"));

    auto* instLabel = new QLabel(tr("Tracked instances:"));
    m_instanceLabel = new QLabel(tr("(none)"));
    m_instanceLabel->setStyleSheet("color: #999;");

    m_statusLabel = new QLabel(tr("Open a video and load a model to start."));
    m_statusLabel->setStyleSheet("color: #99ccff;");
    // Frame-progress lines are long; wrap instead of clipping.
    m_statusLabel->setWordWrap(true);

    bottom2->addWidget(m_exportBtn);
    bottom2->addWidget(m_exportFrameToDbBtn);
    bottom2->addWidget(m_exportVideoBtn);
    bottom2->addSpacing(12);
    bottom2->addWidget(instLabel);
    bottom2->addWidget(m_instanceLabel);
    bottom2->addSpacing(12);
    bottom2->addWidget(m_statusLabel, 1);
    layout->addLayout(bottom2);

    // ── Connections ───────────────────────────────────────────────────────
    connect(m_openBtn, &QPushButton::clicked, this, &VideoTab::onOpenVideo);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &VideoTab::onUseTestData);
    connect(m_loadBtn, &QPushButton::clicked, this, &VideoTab::onLoadModel);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                m_downloadPrompted = false;
                updateDownloadButton();
            });
    connect(m_playBtn, &QPushButton::clicked, this, &VideoTab::onPlayPause);
    connect(m_stepBtn, &QPushButton::clicked, this, &VideoTab::onStep);
    connect(m_resetBtn, &QPushButton::clicked, this, &VideoTab::onReset);
    connect(m_textPrompt, &QLineEdit::editingFinished, this, [this]() {
        m_trackerPromptDirty = true;
        if (m_worker && m_worker->hasModel() && !m_busy) {
            syncTrackerPrompt(true);
        }
    });
    connect(m_scoreSpin, &QDoubleSpinBox::editingFinished, this, [this]() {
        if (m_worker && m_worker->hasModel() && !m_busy) {
            requestTrackerReset(desiredTextPrompt(), true);
        }
    });
    connect(m_speedSlider, &QSlider::valueChanged, this, [this](int v) {
        m_speedLabel->setText(QString("%1x").arg(v / 10.0, 0, 'f', 1));
    });
    connect(m_showMasks, &QCheckBox::toggled, this,
            [this](bool) { updateCanvasInstances(); });
    connect(m_canvas, &VideoCanvas::boxDrawn, this, &VideoTab::onCanvasBox);
    connect(m_canvas, &VideoCanvas::instanceClicked, this,
            &VideoTab::onCanvasInstanceClicked);
    connect(m_canvas, &VideoCanvas::posPointAdded, this,
            &VideoTab::onCanvasPosPoint);
    connect(m_canvas, &VideoCanvas::negPointAdded, this,
            &VideoTab::onCanvasNegPoint);
    connect(m_timeline, &VideoTimeline::seekRequested, this, &VideoTab::onSeek);
    m_playTimer.setSingleShot(true);
    connect(&m_playTimer, &QTimer::timeout, this, [this]() {
        if (!m_playing || m_currentFrame + 1 >= m_totalFrames) {
            schedulePlayback();
            return;
        }
        ++m_currentFrame;
        trackNextFrame();
    });
    // Hot-swap: re-load the model when the device combo changes.
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &VideoTab::onDeviceChanged);
    // Mask export: PNG files to a directory plus grayscale mask images
    // into the DB tree (see onExportMasks).
    connect(m_exportBtn, &QPushButton::clicked, this, &VideoTab::onExportMasks);
    connect(m_exportVideoBtn, &QPushButton::clicked, this,
            &VideoTab::onExportVideoMasks);
    // One-shot export of the current frame to the DB tree.
    connect(m_exportFrameToDbBtn, &QPushButton::clicked, this,
            &VideoTab::exportCurrentFrameToDb);
    // Receive Space / arrow key shortcuts.
    setFocusPolicy(Qt::StrongFocus);

    // Sync the text-prompt enabled state with the default "Text" mode
    // (m_visualOnly is false at construction, so the prompt starts enabled).
    onModeChanged();
}

void VideoTab::populateModelCombo() {
    m_modelCombo->clear();
    const int n = aicore_sam3_model_count();
    // Cache status suffix mirrors qDA3: "[size] ✓" when the GGUF is cached
    // in the shared AICore model cache, "[download]" otherwise.
    char* cacheDirRaw = aicore_sam3_model_cache_dir();
    const QString cacheDir = QString::fromUtf8(cacheDirRaw);
    aicore_sam3_free_buffer(cacheDirRaw);
    for (int i = 0; i < n; ++i) {
        const auto* entry = aicore_sam3_model_at(i);
        if (!entry || !entry->filename) continue;
        const QString file = cacheDir + QLatin1Char('/') +
                             QString::fromUtf8(entry->filename);
        QFileInfo fi(file);
        QString suffix;
        if (isValidCatalogModel(fi.absoluteFilePath(),
                                QString::fromUtf8(entry->filename))) {
            suffix =
                    QStringLiteral(" [%1] \u2713")
                            .arg(ecvModelDownloader::formatFileSize(fi.size()));
        } else {
            suffix = QStringLiteral(" [download]");
        }
        m_modelCombo->addItem(QString("%1 (%2)%3")
                                      .arg(entry->display_name)
                                      .arg(entry->quant_note)
                                      .arg(suffix),
                              entry->filename);
    }
    m_modelCombo->addItem(tr("Browse..."), QString("__browse__"));
}

void VideoTab::refreshModelCombo() {
    // Save selection before rebuild (mirrors SAM3Dialog::refreshModelCombos).
    const QString keep = m_modelCombo->currentData().toString();
    populateModelCombo();  // clears and rebuilds
    if (!keep.isEmpty()) {
        const int idx = m_modelCombo->findData(keep);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
}

QString VideoTab::modelPath() const {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty() || filename == "__browse__") return QString();
    // Catalog entries contain a basename and must resolve through the shared
    // cache. Only an explicitly browsed absolute path may bypass the catalog.
    const QFileInfo selected(filename);
    if (selected.isAbsolute() && selected.isFile()) {
        return selected.absoluteFilePath();
    }
    // The published GGUFs live in the shared AICore model cache
    // ({CLOUDVIEWER_DATA_ROOT|~}/cloudViewer_data/extract/sam3_models).
    char* dir = aicore_sam3_model_cache_dir();
    const QString cacheDir = QString::fromUtf8(dir);
    aicore_sam3_free_buffer(dir);
    const QString full = cacheDir + "/" + filename;
    if (isValidCatalogModel(full, filename)) return full;
    return filename;
}

void VideoTab::startDownload(bool thenRun) {
    if (m_downloadInProgress || !m_modelDownloader) {
        if (m_downloadInProgress) {
            appendLog(tr("[Warning] A model download is already in progress."));
        }
        return;
    }
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty() || filename == "__browse__") return;
    const auto* entry =
            aicore_sam3_model_by_filename(filename.toUtf8().constData());
    if (!entry || !entry->download_url) return;

    char* dir = aicore_sam3_model_cache_dir();
    const QString cacheDir = QString::fromUtf8(dir);
    aicore_sam3_free_buffer(dir);
    const QString dest = cacheDir + QLatin1Char('/') + filename;
    if (isValidCatalogModel(dest, filename)) {
        appendLog(tr("Model already cached: %1").arg(dest));
        if (thenRun) ensureModelReady();
        return;
    }
    ecvAssetIntegrity::removeIfNotVerified(
            dest, {}, 64 * 1024, true,
            ecvAssetIntegrity::OnMiss::CheapChecksOnly, entry->size_bytes);

    QDir().mkpath(cacheDir);
    m_downloadInProgress = true;
    m_downloadThenRun = thenRun;
    m_downloadTargetFilename = filename;
    if (m_downloadLabel) {
        m_downloadLabel->setText(tr("Downloading %1 ...").arg(filename));
        m_downloadLabel->setVisible(true);
    }
    if (m_progress) {
        m_progress->setVisible(true);
        m_progress->setValue(0);
    }
    updateDownloadButton();
    appendLog(tr("Downloading model %1 (%2) ...")
                      .arg(filename)
                      .arg(ecvModelDownloader::formatFileSize(
                              entry->size_bytes)));

    ecvModelDownloader::Request req;
    req.url = QString::fromUtf8(entry->download_url);
    req.destPath = dest;
    // Content identity from the release digest registry — streamed SHA-256
    // check at ingestion (truncation and corruption both caught, no size
    // guard needed); the verified state is recorded in the ledger.
    req.contentAnchor = {QCryptographicHash::Sha256,
                         ecvAssetIntegrity::PinnedDigest(filename)};
    m_modelDownloader->download(req);
}

void VideoTab::updateDownloadButton() {
    if (!m_downloadBtn) return;
    const QString filename = m_modelCombo->currentData().toString();
    const bool isCatalogModel = !filename.isEmpty() && filename != "__browse__";
    char* dir = aicore_sam3_model_cache_dir();
    const QString cacheDir = QString::fromUtf8(dir);
    aicore_sam3_free_buffer(dir);
    const bool cached =
            isCatalogModel &&
            isValidCatalogModel(cacheDir + QLatin1Char('/') + filename,
                                filename);
    m_downloadBtn->setEnabled(!m_downloadInProgress &&
                              m_videoExportDir.isEmpty() && isCatalogModel &&
                              !cached && !m_busy);
    m_downloadBtn->setText(cached ? tr("Cached") : tr("Download"));
}

// ── Slots ──────────────────────────────────────────────────────────────────

void VideoTab::onOpenVideo() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Open video file"), QDir::homePath(),
            tr("Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)"));
    if (path.isEmpty()) return;
    openVideoFile(path);
}

void VideoTab::openVideoFile(const QString& path) {
    if (!m_videoExportDir.isEmpty())
        finishVideoExport(false, tr("Video changed."));
    // A stale playback timer must never fire into the newly opened video
    // (trackNextFrame does not check m_playing).
    m_playTimer.stop();
    if (!m_reader) {
        m_reader = new VideoFrameReader(this);
        connect(m_reader, &VideoFrameReader::frameReady, this,
                &VideoTab::onFrameReady);
        connect(m_reader, &VideoFrameReader::frameReadFailed, this, [this]() {
            if (!m_videoExportDir.isEmpty()) {
                finishVideoExport(false, tr("Video frame read failed."));
                return;
            }
            m_playing = false;
            m_playBtn->setText(tr("Play"));
            CVLog::Warning(
                    "[qSAM3][VideoTab] frameReadFailed: end of stream / "
                    "read error");
            setStatus(tr("Video read failed / end of stream."));
        });
    }
    m_reader->setConsumerDriven(true);
    m_reader->setPaused(true);
    if (!m_reader->openVideo(path.toStdString())) {
        CVLog::Warning("[qSAM3][VideoTab] openVideo failed: %s",
                       path.toUtf8().constData());
        QMessageBox::warning(this, tr("qSAM3"),
                             tr("Failed to open video:\n%1").arg(path));
        return;
    }
    m_videoPath = path;
    m_totalFrames = static_cast<int>(m_reader->getFrameCount());
    m_fps = m_reader->getFps();
    m_currentFrame = 0;
    m_processedMax = -1;
    m_timelineEntries.clear();
    m_timelineInstanceIds.clear();
    m_lastResult = SAM3WorkerResult{};
    m_lastResultFrame = -1;
    m_timeline->setFrameCount(m_totalFrames);
    m_timeline->setCurrentFrame(0);
    m_timeline->setProcessedMax(-1);
    m_timeline->setTimeline(m_timelineEntries);
    m_timeline->setInstances({}, {});
    m_canvas->clearAll();
    m_canvas->setInteractive(m_worker && m_worker->hasModel());
    m_stepBtn->setEnabled(m_worker && m_worker->hasModel());
    m_resetBtn->setEnabled(true);
    m_playing = false;
    m_playBtn->setText(tr("Play"));

    appendLog(tr("Video: %1 | %2x%3 | %4 frames | %.1f fps")
                      .arg(QFileInfo(path).fileName())
                      .arg(m_reader->getFrameWidth())
                      .arg(m_reader->getFrameHeight())
                      .arg(m_totalFrames)
                      .arg(m_fps));
    setStatus(tr("Pause and annotate to add instances."));

    // Show the first frame right away so the canvas is never left black.
    // The frame is delivered asynchronously via frameReady() and rendered
    // by onFrameReady() even before any model is loaded. The decode is
    // queued onto the event loop instead of running inline: OpenCV read()
    // blocks 50-200 ms, and openVideoFile must return so the dialog repaints
    // and the status line updates first. m_reading in VideoFrameReader
    // guards against re-entrant reads if the user hits Play immediately.
    m_reader->seekToFrame(0);
    QMetaObject::invokeMethod(
            this,
            [this]() {
                if (m_reader && m_reader->isOpened()) {
                    m_reader->readFrame();
                }
            },
            Qt::QueuedConnection);

    // Upstream main_video.cpp auto-creates the tracker (and loads the model
    // on first use) right after opening a video; the first frame is then
    // encoded. Mirror that here so the video is usable without any manual
    // Load step.
    if (m_worker && m_worker->hasModel() && !m_busy) {
        // A tracker owns temporal memory from the previous video. Reusing it
        // for a new file leaks instances/state across unrelated streams.
        requestTrackerReset(desiredTextPrompt(), true);
    } else if (!m_worker || !m_worker->hasModel()) {
        ensureModelReady();
    } else {
        m_trackerPromptDirty = true;
        m_pendingRetrack = true;
    }
}

void VideoTab::onLoadModel() {
    // Lazy loading covers the normal flow (Play / Step / Reset / annotate
    // auto-load the model). This slot remains only for the explicit
    // "Browse..." fallback and the hidden Load button.
    if (m_busy) {
        appendLog(tr("Worker is busy; wait for the current task."));
        return;
    }
    QString path = modelPath();
    const bool isBrowseItem =
            m_modelCombo->currentData().toString() == "__browse__";
    if (path.isEmpty() || isBrowseItem) {
        path = QFileDialog::getOpenFileName(
                this, tr("Select SAM3 GGUF model"), QDir::homePath(),
                tr("GGUF files (*.gguf);;All files (*)"));
        if (path.isEmpty()) return;
    } else if (classifyModelFile(m_modelCombo->currentData().toString()) ==
               ModelFileStatus::Unverified) {
        // A stale generation sits in the cache; say so instead of claiming
        // the GGUF was never downloaded.
        const QString filename = m_modelCombo->currentData().toString();
        const auto* entry = catalogEntry(filename);
        const QFileInfo stale(cachedModelPath(filename));
        const auto answer = QMessageBox::question(
                this, tr("qSAM3"),
                tr("A local copy of %1 exists but does not match the "
                   "published release (size %2, expected %3).\n\n"
                   "Download the verified version instead?")
                        .arg(filename,
                             ecvModelDownloader::formatFileSize(stale.size()),
                             entry ? ecvModelDownloader::formatFileSize(
                                             entry->size_bytes)
                                   : QStringLiteral("unknown")),
                QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
        if (answer != QMessageBox::Yes) return;
        startDownload(false);
        return;
    } else if (!QFileInfo::exists(path)) {
        // The combo lists published models, but the GGUF itself is not
        // downloaded yet. Tell the user instead of failing silently.
        const auto answer = QMessageBox::question(
                this, tr("qSAM3"),
                tr("Model file not found:\n%1\n\n"
                   "It has not been downloaded yet. Browse for a local GGUF "
                   "file instead?")
                        .arg(path),
                QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
        if (answer != QMessageBox::Yes) return;
        path = QFileDialog::getOpenFileName(
                this, tr("Select SAM3 GGUF model"), QDir::homePath(),
                tr("GGUF files (*.gguf);;All files (*)"));
        if (path.isEmpty()) return;
    }
    if (!m_worker) {
        m_worker = new VideoWorker(this);
        connect(m_worker, &VideoWorker::logMessage, this, &VideoTab::onLog);
        connect(m_worker, &VideoWorker::modelReady, this,
                &VideoTab::onModelReady);
        connect(m_worker, &VideoWorker::trackerReset, this,
                &VideoTab::onTrackerReset);
        connect(m_worker, &VideoWorker::frameResultReady, this,
                &VideoTab::onFrameResult);
        connect(m_worker, &VideoWorker::instanceAdded, this,
                &VideoTab::onInstanceAdded);
        connect(m_worker, &VideoWorker::instanceRefined, this,
                &VideoTab::onInstanceRefined);
        connect(m_worker, &VideoWorker::busyChanged, this,
                &VideoTab::onBusyChanged);
    }

    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::LoadModel;
    req.modelPath = path;
    req.device = selectedDevice();
    req.textPrompt = desiredTextPrompt();
    req.scoreThreshold = static_cast<float>(m_scoreSpin->value());
    m_loadingModelPath = QFileInfo(path).absoluteFilePath();
    m_loadingDevice = req.device;
    m_loadingTextPrompt = req.textPrompt;
    CVLog::Print("[qSAM3][VideoTab] loading model: %s (device=%s)",
                 m_loadingModelPath.toUtf8().constData(),
                 req.device.toUtf8().constData());
    m_backendLabel->setText(tr("Backend: loading..."));
    emit backendChanged(QStringLiteral("loading..."));
    setStatus(tr("Loading model..."));
    m_worker->post(req);
}

bool VideoTab::ensureModelReady() {
    const QString path = modelPath();
    const QString requestedPath = QFileInfo(path).absoluteFilePath();
    const QString requestedDevice = selectedDevice();
    // A live worker is only reusable when it owns the model/device selected
    // in the UI. Otherwise the next action must hot-reload first.
    if (m_worker && m_worker->hasModel() &&
        m_loadedModelPath == requestedPath &&
        m_loadedDevice == requestedDevice) {
        return true;
    }
    if (m_busy) {
        appendLog(tr("Worker is busy; wait for the current task to finish."));
        return false;
    }
    // Lazy load: same path as onLoadModel() but with no user interaction
    // (the model file must exist locally; otherwise the load fails with a
    // log message instead of a dialog).
    if (!m_worker) {
        m_worker = new VideoWorker(this);
        connect(m_worker, &VideoWorker::logMessage, this, &VideoTab::onLog);
        connect(m_worker, &VideoWorker::modelReady, this,
                &VideoTab::onModelReady);
        connect(m_worker, &VideoWorker::trackerReset, this,
                &VideoTab::onTrackerReset);
        connect(m_worker, &VideoWorker::frameResultReady, this,
                &VideoTab::onFrameResult);
        connect(m_worker, &VideoWorker::instanceAdded, this,
                &VideoTab::onInstanceAdded);
        connect(m_worker, &VideoWorker::instanceRefined, this,
                &VideoTab::onInstanceRefined);
        connect(m_worker, &VideoWorker::busyChanged, this,
                &VideoTab::onBusyChanged);
    }
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty() || filename == "__browse__") {
        CVLog::Warning("[qSAM3][VideoTab] auto-load: no model selected");
        appendLog(
                tr("No model selected. Pick a catalog model or Browse a "
                   "local GGUF file."));
        return false;
    }
    if (classifyModelFile(filename) == ModelFileStatus::Unverified) {
        // A stale generation sits in the cache; say so instead of claiming
        // the GGUF was never downloaded.
        CVLog::Warning(
                "[qSAM3][VideoTab] auto-load: cached model does not "
                "match the published release: %s",
                modelDisplayPath(filename).toUtf8().constData());
        // Offer to download the verified GGUF once per session (qDA3-style).
        if (!m_downloadPrompted) {
            m_downloadPrompted = true;
            const auto* entry = catalogEntry(filename);
            const QFileInfo stale(cachedModelPath(filename));
            const auto answer = QMessageBox::question(
                    this, tr("qSAM3"),
                    tr("A local copy of %1 exists but does not match the "
                       "published release (size %2, expected %3).\n\n"
                       "Download the verified version now?")
                            .arg(filename,
                                 ecvModelDownloader::formatFileSize(
                                         stale.size()),
                                 entry ? ecvModelDownloader::formatFileSize(
                                                 entry->size_bytes)
                                       : QStringLiteral("unknown")),
                    QMessageBox::Yes | QMessageBox::No);
            if (answer == QMessageBox::Yes) {
                startDownload(true);
                return false;
            }
            appendLog(
                    tr("Cached model does not match the published release. "
                       "Use the Download button or Browse a local GGUF "
                       "file."));
        }
        return false;
    }
    if (!QFileInfo::exists(path)) {
        CVLog::Warning("[qSAM3][VideoTab] auto-load: model file not found: %s",
                       modelDisplayPath(filename).toUtf8().constData());
        // Offer to download the catalog GGUF once per session (qDA3-style).
        if (!m_downloadPrompted) {
            m_downloadPrompted = true;
            const auto answer = QMessageBox::question(
                    this, tr("qSAM3"),
                    tr("Model file not found:\n%1\n\n"
                       "Download it now into the model cache?")
                            .arg(modelDisplayPath(filename)),
                    QMessageBox::Yes | QMessageBox::No);
            if (answer == QMessageBox::Yes) {
                startDownload(true);
                return false;
            }
            appendLog(
                    tr("Model not cached. Use the Download button or "
                       "select a cached model from the combo."));
        }
        return false;
    }
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::LoadModel;
    req.modelPath = path;
    req.device = requestedDevice;
    req.textPrompt = desiredTextPrompt();
    req.scoreThreshold = static_cast<float>(m_scoreSpin->value());
    m_loadingModelPath = requestedPath;
    m_loadingDevice = req.device;
    m_loadingTextPrompt = req.textPrompt;
    CVLog::Print(
            "[qSAM3][VideoTab] auto-loading model on first use: %s "
            "(device=%s)",
            requestedPath.toUtf8().constData(),
            req.device.toUtf8().constData());
    m_backendLabel->setText(tr("Backend: loading..."));
    emit backendChanged(QStringLiteral("loading..."));
    setStatus(tr("Loading model..."));
    m_worker->post(req);
    return false;
}

QString VideoTab::desiredTextPrompt() const {
    if (m_visualOnly || !m_modeText->isChecked() || !m_modeText->isVisible()) {
        return QString();
    }
    return m_textPrompt->text().trimmed();
}

void VideoTab::clearTrackingVisualization() {
    if (!m_videoExportDir.isEmpty())
        finishVideoExport(false, tr("Tracker reset."));
    m_lastResult = SAM3WorkerResult{};
    m_lastResultFrame = -1;
    m_processedMax = -1;
    m_timelineEntries.clear();
    m_timelineInstanceIds.clear();
    m_timeline->setProcessedMax(-1);
    m_timeline->setTimeline(m_timelineEntries);
    m_timeline->setInstances({}, {});
    updateCanvasInstances();
}

void VideoTab::requestTrackerReset(const QString& textPrompt, bool retrack) {
    if (!m_worker || !m_worker->hasModel() || m_busy || m_promptResetInFlight) {
        m_trackerPromptDirty = true;
        m_pendingRetrack = m_pendingRetrack || retrack;
        return;
    }
    clearTrackingVisualization();
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::ResetTracker;
    req.textPrompt = m_visualOnly ? QString() : textPrompt;
    req.scoreThreshold = static_cast<float>(m_scoreSpin->value());
    m_promptResetInFlight = true;
    m_trackerPromptDirty = false;
    m_pendingRetrack = m_pendingRetrack || retrack;
    m_worker->post(req);
    setStatus(req.textPrompt.isEmpty()
                      ? tr("Resetting tracker...")
                      : tr("Applying text prompt: %1").arg(req.textPrompt));
}

bool VideoTab::syncTrackerPrompt(bool retrack) {
    if (!m_worker || !m_worker->hasModel()) return true;
    const QString desired = desiredTextPrompt();
    if (!m_trackerPromptDirty && desired == m_activeTextPrompt) return true;
    if (m_busy || m_promptResetInFlight) {
        m_trackerPromptDirty = true;
        m_pendingRetrack = m_pendingRetrack || retrack;
        return false;
    }
    requestTrackerReset(desired, retrack);
    return false;
}

void VideoTab::onModelReady(const QString& backend, bool visualOnly) {
    m_visualOnly = visualOnly;
    m_loadedModelPath = m_loadingModelPath;
    m_loadedDevice = m_loadingDevice;
    m_activeTextPrompt = visualOnly ? QString() : m_loadingTextPrompt;
    m_loadingModelPath.clear();
    m_loadingDevice.clear();
    m_loadingTextPrompt.clear();
    m_trackerPromptDirty = false;
    m_promptResetInFlight = false;
    m_backendLabel->setText(QString("Backend: %1").arg(backend));
    emit backendChanged(backend);
    // Text mode is only available on full SAM3 models.
    m_modeText->setVisible(!visualOnly);
    m_textPrompt->setEnabled(!visualOnly && m_modeText->isChecked());
    if (visualOnly && m_modeText->isChecked()) {
        m_modeBox->setChecked(true);
    }
    m_canvas->setInteractive(true);
    m_stepBtn->setEnabled(true);
    m_exportVideoBtn->setEnabled(!m_videoPath.isEmpty());
    m_trackerActive = true;
    setStatus(tr("Tracker created. Press Play or add instances."));
    // Re-run the operation that triggered the lazy load (Play / Step /
    // Reset / annotate), mirroring upstream main_video.cpp where the model
    // loads automatically on first use. If nothing was pending, just
    // process the current frame so a freshly opened video is never black.
    const PendingAction pending = m_pendingAction;
    m_pendingAction = PendingAction::None;
    if (pending == PendingAction::Play) {
        m_playing = true;
        m_playBtn->setText(tr("Pause"));
        trackNextFrame();
        return;
    }
    if (pending == PendingAction::Step) {
        onStep();
        return;
    }
    if (pending == PendingAction::Seek) {
        trackNextFrame();
        return;
    }
    if (pending == PendingAction::Reset) {
        onReset();
        return;
    }
    if (pending == PendingAction::Annotate) {
        // The user already drew a box / point while the model was loading;
        // add the instance now that the tracker exists.
        addInstanceFromPrompts();
        return;
    }
    // If a video is already open, process the current frame right away.
    // trackNextFrame() itself defers to onBusyChanged(false) when the
    // worker is still busy (modelReady arrives before busyChanged(false)).
    if (!m_videoPath.isEmpty()) {
        trackNextFrame();
    }
}

void VideoTab::onTrackerReset(const QString& textPrompt, bool ok) {
    m_promptResetInFlight = false;
    if (ok) {
        m_activeTextPrompt = m_visualOnly ? QString() : textPrompt;
        m_trackerPromptDirty = desiredTextPrompt() != m_activeTextPrompt;
        m_trackerActive = true;
    } else {
        m_trackerPromptDirty = true;
        m_trackerActive = false;
    }
}

void VideoTab::trackNextFrame() {
    // Decode and display the next frame. Tracking (segment) needs a loaded
    // model; without one, the video still plays — the user sees raw frames
    // and can load a model later without restarting.
    if (m_busy || !m_reader) {
        // The worker is busy (model load / ResetTracker); modelReady and
        // the reset completion arrive before busyChanged(false), so a
        // direct call here would be dropped. Remember the request and
        // re-run it from onBusyChanged(false).
        if (m_busy && m_reader) m_pendingRetrack = true;
        return;
    }
    if (m_worker && m_worker->hasModel() && !syncTrackerPrompt(true)) {
        return;
    }
    if (m_currentFrame >= m_totalFrames) {
        m_playing = false;
        m_playBtn->setText(tr("Play"));
        setStatus(tr("End of video."));
        return;
    }
    m_reader->seekToFrame(m_currentFrame);
    m_reader->readFrame();  // async: frameReady() delivers the decoded frame
}

void VideoTab::onFrameReady(const cv::Mat& rgbFrame, int frameIndex) {
    if (!m_videoExportDir.isEmpty() &&
        frameIndex != m_videoExportExpectedFrame) {
        finishVideoExport(false, tr("Unexpected decoded frame %1; expected %2.")
                                         .arg(frameIndex)
                                         .arg(m_videoExportExpectedFrame));
        return;
    }
    QImage img = VideoPlaybackWidget::cvMatToQImage(rgbFrame);
    if (img.isNull()) {
        CVLog::Warning("[qSAM3][VideoTab] frame %d decoded to a null QImage",
                       frameIndex);
        appendLog(tr("Frame %1 has no data.").arg(frameIndex));
        if (!m_videoExportDir.isEmpty()) {
            finishVideoExport(false,
                              tr("Frame %1 decode failed.").arg(frameIndex));
        } else {
            m_playing = false;
            m_playTimer.stop();
            m_playBtn->setText(tr("Play"));
        }
        return;
    }
    m_currentFrameImage = img;
    m_currentFrame = frameIndex;
    // Keep the previous frame's instance masks visible on the newly decoded
    // frame until this frame's result arrives. Clearing here made the
    // overlay blink off for the whole decode+inference window of every
    // frame ("one frame with masks, one frame without"): the playback
    // period equals decode+inference, so the blank gap dominated. A mask
    // that is one frame old is imperceptible at playback rates. No
    // setInstances() call here: VideoCanvas::setFrame → redraw() composites
    // the retained boxes / masks onto the new frame, and onFrameResult()
    // → updateCanvasInstances() then swaps in the fresh result.
    m_canvas->setFrame(img);
    m_timeline->setCurrentFrame(frameIndex);

    if (m_worker && m_worker->hasModel()) {
        if (!m_videoExportDir.isEmpty()) m_videoExportAwaitedFrame = frameIndex;
        VideoWorker::TrackRequest req;
        req.action = VideoWorker::Action::TrackFrame;
        req.frameIndex = frameIndex;
        req.frame = img;
        m_worker->post(req);
    } else {
        setStatus(tr("Frame %1/%2 — no tracker active")
                          .arg(frameIndex)
                          .arg(m_totalFrames));
        schedulePlayback();
    }
}

void VideoTab::onFrameResult(const SAM3WorkerResult& result, int frameIndex) {
    // A latest-wins worker can finish an older request after a newer frame was
    // decoded. During an archive run, accepting that result would append a
    // valid-looking mask under the wrong frame and move the UI backwards.
    if (!m_videoExportDir.isEmpty() &&
        frameIndex != m_videoExportExpectedFrame) {
        finishVideoExport(
                false,
                tr("Unexpected tracking result for frame %1; expected %2.")
                        .arg(frameIndex)
                        .arg(m_videoExportExpectedFrame));
        return;
    }
    if (frameIndex != m_currentFrame) {
        appendLog(tr("Ignored stale tracking result for frame %1.")
                          .arg(frameIndex));
        return;
    }

    m_lastResult = result;
    m_lastResultFrame = frameIndex;
    m_currentFrame = frameIndex;
    if (frameIndex > m_processedMax) m_processedMax = frameIndex;
    updateCanvasInstances();
    updateTimeline(frameIndex, result);

    if (!m_videoExportDir.isEmpty()) {
        if (!result.valid || !result.errorMsg.isEmpty() ||
            !appendVideoMasks(result, frameIndex)) {
            finishVideoExport(
                    false,
                    result.errorMsg.isEmpty()
                            ? tr("Frame %1 export failed.").arg(frameIndex)
                            : result.errorMsg);
            return;
        } else if (m_videoExportProgress) {
            m_videoExportAwaitedFrame = -1;
            m_videoExportExpectedFrame = frameIndex + 1;
            m_videoExportProgress->setValue(frameIndex + 1);
        }
    }

    // Auto-export to DB tree if enabled.
    if (m_exportToDbCheckBox->isChecked() && m_app) {
        exportCurrentFrameToDb();
    }

    if (!result.errorMsg.isEmpty()) {
        m_playing = false;
        m_playTimer.stop();
        m_playBtn->setText(tr("Play"));
        setStatus(result.errorMsg);
        return;
    } else if (result.valid) {
        setStatus(tr("Frame %1/%2 — %3 objects tracked")
                          .arg(frameIndex)
                          .arg(m_totalFrames)
                          .arg(result.detCount));
    }
    schedulePlayback();
}

void VideoTab::schedulePlayback() {
    if (!m_playing || m_totalFrames <= 0) return;
    if (m_currentFrame + 1 >= m_totalFrames) {
        if (!m_videoExportDir.isEmpty()) finishVideoExport(true);
        m_playing = false;
        m_playBtn->setText(tr("Play"));
        setStatus(tr("End of video."));
        return;
    }
    const double speed = m_speedSlider->value() / 10.0;
    const int intervalMs =
            !m_videoExportDir.isEmpty()
                    ? 0
                    : (m_fps > 0.0 ? static_cast<int>(1000.0 / (m_fps * speed))
                                   : 40);
    m_playTimer.start(std::max(1, intervalMs));
}

void VideoTab::onPlayPause() {
    if (!m_videoExportDir.isEmpty()) {
        finishVideoExport(false, tr("Canceled."));
        return;
    }
    if (m_videoPath.isEmpty()) {
        CVLog::Warning("[qSAM3][VideoTab] Play pressed with no video open");
        appendLog(tr("Open a video first."));
        return;
    }
    if (!ensureModelReady()) {
        // Model loads lazily on first use; the play request re-runs from
        // onModelReady once the load finishes.
        m_pendingAction = PendingAction::Play;
        return;
    }
    m_playing = !m_playing;
    m_playBtn->setText(m_playing ? tr("Pause") : tr("Play"));
    if (m_playing) {
        m_playTimer.stop();
        // If we are at the end, restart from the beginning.
        if (m_currentFrame + 1 >= m_totalFrames) {
            m_currentFrame = 0;
        }
        trackNextFrame();
    } else {
        m_playTimer.stop();
    }
}

void VideoTab::onStep() {
    if (!ensureModelReady()) {
        m_pendingAction = PendingAction::Step;
        return;
    }
    if (m_currentFrame + 1 < m_totalFrames) {
        m_playing = false;
        m_playBtn->setText(tr("Play"));
        m_playTimer.stop();
        ++m_currentFrame;
        trackNextFrame();
    }
}

void VideoTab::onReset() {
    if (!m_videoExportDir.isEmpty())
        finishVideoExport(false, tr("Tracker reset."));
    m_playing = false;
    m_playBtn->setText(tr("Play"));
    m_playTimer.stop();
    resetPrompts();
    m_lastResult = SAM3WorkerResult{};
    m_lastResultFrame = -1;
    m_timelineEntries.clear();
    m_timelineInstanceIds.clear();
    m_timeline->setTimeline(m_timelineEntries);
    m_timeline->setInstances({}, {});
    m_canvas->clearAll();
    m_currentFrame = 0;
    m_processedMax = -1;
    m_timeline->setCurrentFrame(0);
    m_timeline->setProcessedMax(-1);

    if (m_videoPath.isEmpty()) {
        setStatus(tr("Reset. Ready to annotate."));
        return;
    }
    // The model loads lazily on first use (upstream main_video.cpp Reset
    // re-creates the tracker and re-encodes frame 0; without a model it
    // auto-loads one first).
    if (!ensureModelReady()) {
        m_pendingAction = PendingAction::Reset;
        return;
    }

    m_trackerActive = true;
    m_canvas->setInteractive(true);
    m_currentFrame = 0;
    requestTrackerReset(desiredTextPrompt(), true);
    setStatus(tr("Reset. Ready to annotate."));
}

void VideoTab::onCanvasBox() {
    if (!ensureModelReady()) {
        m_pendingAction = PendingAction::Annotate;
        return;
    }
    addInstanceFromPrompts();
}

void VideoTab::onCanvasInstanceClicked(int id, const QPointF& pos) {
    if (!ensureModelReady()) {
        m_pendingAction = PendingAction::Annotate;
        return;
    }
    if (!m_trackerActive) {
        setStatus(tr("Tracker not ready; press Play or Reset first."));
        return;
    }
    // Upstream main_video.cpp: clicking an existing mask refines it with a
    // positive point at the click position.
    refineInstance(id, {pos}, {});
    setStatus(tr("Refined instance #%1").arg(id));
}

void VideoTab::onCanvasPosPoint(const QPointF& p) {
    if (!ensureModelReady()) {
        m_pendingAction = PendingAction::Annotate;
        return;
    }
    if (!m_trackerActive) {
        setStatus(tr("Tracker not ready; press Play or Reset first."));
        return;
    }
    // Upstream main_video.cpp Points mode: a click on empty canvas adds a
    // new tracked instance from the positive point right away.
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::AddInstance;
    req.prompt.posPoints.append(
            {static_cast<float>(p.x()), static_cast<float>(p.y())});
    req.frame = m_currentFrameImage;
    req.frameIndex = m_currentFrame;
    m_worker->post(req);
}

void VideoTab::onCanvasNegPoint(const QPointF& p) {
    if (!ensureModelReady()) {
        m_pendingAction = PendingAction::Annotate;
        return;
    }
    // Negative point: refine the instance under the cursor, or queue it for
    // the next AddInstance.
    const int hit = m_canvas->hitTestInstance(p);
    if (hit >= 0 && m_trackerActive && m_worker && m_worker->hasModel()) {
        // Upstream main_video.cpp: right-click on a tracked mask refines it
        // with the mask centroid as positive point + the click as negative.
        QPointF centroid;
        bool hasCentroid = false;
        const int idx = m_lastResult.instanceIds.indexOf(hit);
        if (idx >= 0 && idx < m_lastResult.instanceMasks.size()) {
            const QImage& mask = m_lastResult.instanceMasks[idx];
            double cx = 0.0, cy = 0.0;
            qint64 n = 0;
            for (int y = 0; y < mask.height(); ++y) {
                const uchar* row = mask.constScanLine(y);
                for (int x = 0; x < mask.width(); ++x) {
                    if (row[x] > 127) {
                        cx += x;
                        cy += y;
                        ++n;
                    }
                }
            }
            if (n > 0) {
                centroid = QPointF(cx / n, cy / n);
                hasCentroid = true;
            }
        }
        refineInstance(
                hit,
                hasCentroid ? QVector<QPointF>{centroid} : QVector<QPointF>(),
                {p});
        return;
    }
    if (!m_trackerActive || !m_worker || !m_worker->hasModel()) {
        setStatus(tr("Load a model first to enable tracking."));
        return;
    }
    // Upstream main_video.cpp Points mode: a negative click on empty canvas
    // is queued on the canvas and included in the next instance creation.
    m_canvas->addNegPoint(p);
    appendLog(
            tr("Negative point queued; draw a box or click a positive "
               "point to add an instance."));
}

void VideoTab::addInstanceFromPrompts() {
    if (!m_trackerActive || !m_worker || !m_worker->hasModel()) {
        CVLog::Warning(
                "[qSAM3][VideoTab] addInstanceFromPrompts skipped: no model "
                "loaded (trackerActive=%d hasModel=%d)",
                m_trackerActive ? 1 : 0,
                (m_worker && m_worker->hasModel()) ? 1 : 0);
        setStatus(tr("Load a model first to enable tracking."));
        return;
    }
    if (!m_canvas) return;
    if (m_currentFrameImage.isNull()) {
        CVLog::Warning(
                "[qSAM3][VideoTab] addInstanceFromPrompts skipped: no current "
                "frame image (video not displaying)");
        appendLog(tr("No video frame available; open a video first."));
        return;
    }
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::AddInstance;
    if (m_canvas->hasBox()) {
        const QRectF b = m_canvas->box();
        req.prompt.usePvsBox = true;
        req.prompt.pvsBox = {
                static_cast<float>(b.left()), static_cast<float>(b.top()),
                static_cast<float>(b.right()), static_cast<float>(b.bottom())};
    }
    for (const auto& pt : m_canvas->posPoints()) {
        req.prompt.posPoints.append(
                {static_cast<float>(pt.x()), static_cast<float>(pt.y())});
    }
    for (const auto& pt : m_canvas->negPoints()) {
        req.prompt.negPoints.append(
                {static_cast<float>(pt.x()), static_cast<float>(pt.y())});
    }
    if (req.prompt.posPoints.isEmpty() && !req.prompt.usePvsBox) {
        appendLog(tr("Click a positive point or drag a box first."));
        return;
    }
    // Carry the current frame so the worker can run the display PVS
    // (upstream main_video.cpp shows the new instance's mask right away).
    req.frame = m_currentFrameImage;
    req.frameIndex = m_currentFrame;
    m_worker->post(req);
    resetPrompts();
}

void VideoTab::refineInstance(int id,
                              const QVector<QPointF>& pos,
                              const QVector<QPointF>& neg) {
    if (!m_trackerActive || !m_worker || !m_worker->hasModel()) {
        CVLog::Warning(
                "[qSAM3][VideoTab] refineInstance skipped: no model loaded");
        setStatus(tr("Load a model first to enable tracking."));
        return;
    }
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::RefineInstance;
    req.instanceId = id;
    // Carry the current frame so the worker can (re-)encode it when the
    // tracker state is stale (refine runs PVS internally and needs the
    // current frame's encoded features).
    req.frame = m_currentFrameImage;
    req.frameIndex = m_currentFrame;
    for (const auto& pt : pos) {
        req.prompt.posPoints.append(
                {static_cast<float>(pt.x()), static_cast<float>(pt.y())});
    }
    for (const auto& pt : neg) {
        req.prompt.negPoints.append(
                {static_cast<float>(pt.x()), static_cast<float>(pt.y())});
    }
    m_worker->post(req);
}

void VideoTab::onInstanceAdded(int instanceId) {
    appendLog(tr("Added instance #%1").arg(instanceId));
    setStatus(tr("Added instance #%1").arg(instanceId));
}

void VideoTab::onInstanceRefined(int instanceId, bool ok) {
    setStatus(ok ? tr("Refined instance #%1").arg(instanceId)
                 : tr("Failed to refine instance #%1").arg(instanceId));
}

void VideoTab::onBusyChanged(bool busy) {
    m_busy = busy;
    const bool canEditModel = !busy && m_videoExportDir.isEmpty();
    m_loadBtn->setEnabled(canEditModel);
    m_modelCombo->setEnabled(canEditModel);
    m_deviceCombo->setEnabled(canEditModel);
    m_canvas->setInteractive(!busy && m_videoExportDir.isEmpty());
    m_stepBtn->setEnabled(canEditModel && m_worker && m_worker->hasModel());
    m_exportVideoBtn->setEnabled(!busy && m_videoExportDir.isEmpty() &&
                                 m_worker && m_worker->hasModel() &&
                                 !m_videoPath.isEmpty());
    updateDownloadButton();
    if (!busy && !m_videoExportDir.isEmpty() &&
        m_videoExportAwaitedFrame >= 0) {
        finishVideoExport(false, tr("Tracking failed on frame %1.")
                                         .arg(m_videoExportAwaitedFrame));
    }
    if (busy) {
        m_playTimer.stop();
    } else if (m_worker && !m_worker->hasModel() &&
               !m_loadingModelPath.isEmpty()) {
        // A failed load has no modelReady signal, so clear the transient
        // loading identity here when the worker becomes idle.
        m_loadingModelPath.clear();
        m_loadingDevice.clear();
        m_loadingTextPrompt.clear();
        m_backendLabel->setText(tr("Backend: none"));
        emit backendChanged(QStringLiteral("none"));
    } else if (m_reloadWhenIdle) {
        m_reloadWhenIdle = false;
        onLoadModel();
    } else if (m_trackerPromptDirty && m_worker && m_worker->hasModel()) {
        // Text/mode may have changed while inference was in flight. Apply it
        // before releasing the pending frame to the tracker.
        syncTrackerPrompt(m_pendingRetrack);
    } else if (m_pendingRetrack) {
        // A frame request was queued while the worker was busy (model
        // load / ResetTracker); deliver it now that the worker is free.
        m_pendingRetrack = false;
        if (!m_videoPath.isEmpty()) {
            trackNextFrame();
        }
    }
}

void VideoTab::onLog(const QString& msg) { setStatus(msg); }

void VideoTab::onSeek(int frame) {
    if (!m_videoExportDir.isEmpty())
        finishVideoExport(false, tr("Seek interrupted export."));
    if (m_videoPath.isEmpty()) return;
    if (!ensureModelReady()) {
        // Track the requested frame once the lazy model load finishes
        // (onModelReady re-runs trackNextFrame for the current frame).
        m_pendingAction = PendingAction::Seek;
        m_currentFrame = qBound(0, frame, m_totalFrames - 1);
        return;
    }
    m_playing = false;
    m_playBtn->setText(tr("Play"));
    m_playTimer.stop();
    m_currentFrame = qBound(0, frame, m_totalFrames - 1);
    // A seek lands far from the frame the retained masks belong to; they
    // would highlight the wrong objects. Drop them — the first tracked
    // frame after the seek repopulates the overlay.
    m_lastResult = SAM3WorkerResult{};
    m_lastResultFrame = -1;
    updateCanvasInstances();
    trackNextFrame();
}

void VideoTab::onModeChanged() {
    if (m_modeText->isChecked() && m_visualOnly) {
        // Visual-only model: fall back to box mode.
        m_modeBox->setChecked(true);
    }
    const bool textMode = m_modeText->isChecked() && !m_visualOnly;
    m_textPrompt->setEnabled(textMode);
    resetPrompts();
    m_trackerPromptDirty = desiredTextPrompt() != m_activeTextPrompt;
    if (m_trackerPromptDirty && m_worker && m_worker->hasModel() && !m_busy) {
        // Switching Text <-> Box/Points changes whether automatic text
        // detections belong in the tracker; rebuild immediately.
        syncTrackerPrompt(true);
    }
    appendLog(tr("Mode: %1").arg(textMode ? tr("Text") : tr("Box / Points")));
}

void VideoTab::onDeviceChanged() {
    // Hot-swap: with a model loaded, re-load it on the newly selected
    // backend right away (upstream main_video.cpp Devices combo triggers
    // reload_model() on the fly).
    if (!m_worker || !m_worker->hasModel()) return;
    if (m_busy) {
        appendLog(
                tr("Device changed; re-loading when the current task "
                   "finishes."));
        m_reloadWhenIdle = true;
        return;
    }
    if (m_modelCombo->currentData().toString() == "__browse__") {
        appendLog(tr("Device changed; re-load the model to apply."));
        return;
    }
    appendLog(tr("Device changed to %1 - reloading model...")
                      .arg(m_deviceCombo->currentText()));
    onLoadModel();
}

void VideoTab::keyPressEvent(QKeyEvent* e) {
    // Keyboard shortcuts mirroring upstream main_video.cpp: Space toggles
    // play/pause, Right steps forward, Left steps back. Widgets that
    // consume keys (e.g. the text-prompt QLineEdit) handle them first, so
    // typing in the prompt never triggers playback.
    if (e->key() == Qt::Key_Space && !e->isAutoRepeat()) {
        onPlayPause();
        e->accept();
        return;
    }
    if (e->key() == Qt::Key_Right && !e->isAutoRepeat()) {
        onStep();
        e->accept();
        return;
    }
    if (e->key() == Qt::Key_Left && !e->isAutoRepeat()) {
        if (m_currentFrame > 0) {
            m_playing = false;
            m_playBtn->setText(tr("Play"));
            m_playTimer.stop();
            --m_currentFrame;
            // Backwards tracking restarts from the seek target; the retained
            // masks belong to a later frame. Drop them (see onSeek).
            m_lastResult = SAM3WorkerResult{};
            updateCanvasInstances();
            trackNextFrame();
        }
        e->accept();
        return;
    }
    QWidget::keyPressEvent(e);
}

void VideoTab::onExportMasks() {
    if (!m_lastResult.valid || m_lastResult.instanceMasks.isEmpty()) {
        appendLog(tr("No masks on the current frame."));
        return;
    }
    // One-shot DB export: each per-instance mask lands as its own grayscale
    // ccImage in the DB tree — no filesystem round-trip. The annotated frame
    // belongs to the Export-frame-to-DB button, not to this one.
    const int dbCount = exportMasksToDb();
    if (dbCount > 0) {
        appendLog(tr("Added %1 mask image(s) to the DB tree.").arg(dbCount));
    } else {
        appendLog(tr("No DB interface available; masks were not exported."));
    }
}

void VideoTab::onExportVideoMasks() {
    if (!m_videoExportDir.isEmpty() || m_busy || !m_worker ||
        !m_worker->hasModel() || m_videoPath.isEmpty() || m_totalFrames <= 0) {
        setStatus(tr("Open a video and wait for the tracker to be ready."));
        return;
    }
    if (m_currentFrame > 0) {
        QMessageBox scope(this);
        scope.setWindowTitle(tr("Video tracking range"));
        scope.setText(tr("Tracking will start at frame %1 of %2.")
                              .arg(m_currentFrame)
                              .arg(m_totalFrames));
        scope.setInformativeText(
                tr("Export the remaining frames, or reset to frame 0 for a "
                   "complete video. Reset clears manually added instances; "
                   "add their prompts on frame 0, then click export again."));
        auto* remaining = scope.addButton(tr("Export remaining frames"),
                                          QMessageBox::AcceptRole);
        auto* reset = scope.addButton(tr("Reset to frame 0"),
                                      QMessageBox::ActionRole);
        scope.addButton(QMessageBox::Cancel);
        scope.exec();
        if (scope.clickedButton() == reset) {
            onReset();
            setStatus(
                    tr("Reset to frame 0. Add prompts on the first frame, "
                       "then choose Track and export video again."));
            return;
        }
        if (scope.clickedButton() != remaining) return;
    }
    const QString parent = QFileDialog::getExistingDirectory(
            this, tr("Choose tracking output folder"),
            QFileInfo(m_videoPath).absolutePath());
    if (parent.isEmpty()) return;
    startVideoExport(parent);
}

void VideoTab::startVideoExport(const QString& parent) {
    if (parent.isEmpty() || !m_videoExportDir.isEmpty() ||
        m_videoPath.isEmpty() || m_totalFrames <= 0 || !m_worker ||
        !m_worker->hasModel()) {
        setStatus(tr("Open a video and wait for the tracker to be ready."));
        return;
    }
    QString digestError;
    const QString sourceDigest =
            videoMaskArchiveSha256(m_videoPath, &digestError);
    if (sourceDigest.isEmpty()) {
        setStatus(tr("Could not read source video: %1").arg(digestError));
        return;
    }
    const QString folder = QFileInfo(m_videoPath).completeBaseName() +
                           QStringLiteral("_sam3_") +
                           QDateTime::currentDateTimeUtc().toString(
                                   QStringLiteral("yyyyMMddTHHmmsszzz"));
    QDir directory(parent);
    if (!directory.mkdir(folder)) {
        QMessageBox::warning(
                this, tr("Video tracking"),
                tr("Cannot create output folder in %1.").arg(parent));
        return;
    }
    m_videoExportDir = directory.filePath(folder);
    m_videoExportSourceDigest = sourceDigest;
    m_videoExportFrames = QJsonArray();
    m_videoExportStart = m_currentFrame;
    m_videoExportExpectedFrame = m_currentFrame;
    m_videoExportAwaitedFrame = -1;
    m_videoExportProgress =
            new QProgressDialog(tr("Tracking frames %1 to %2...")
                                        .arg(m_videoExportStart)
                                        .arg(m_totalFrames - 1),
                                tr("Cancel"), 0, m_totalFrames, this);
    m_videoExportProgress->setWindowModality(Qt::WindowModal);
    m_videoExportProgress->setAutoClose(false);
    m_videoExportProgress->setAutoReset(false);
    m_videoExportProgress->setMinimumDuration(0);
    m_videoExportProgress->setValue(m_currentFrame);
    connect(m_videoExportProgress, &QProgressDialog::canceled, this,
            [this]() { finishVideoExport(false, tr("Canceled.")); });
    m_videoExportProgress->show();
    m_openBtn->setEnabled(false);
    m_testVideoCombo->setEnabled(false);
    m_testDataBtn->setEnabled(false);
    m_resetBtn->setEnabled(false);
    m_stepBtn->setEnabled(false);
    m_exportVideoBtn->setEnabled(false);
    m_modelCombo->setEnabled(false);
    m_deviceCombo->setEnabled(false);
    m_downloadBtn->setEnabled(false);
    m_canvas->setInteractive(false);
    m_timeline->setEnabled(false);
    m_modeText->setEnabled(false);
    m_modeBox->setEnabled(false);
    m_modePoints->setEnabled(false);
    m_textPrompt->setEnabled(false);
    m_scoreSpin->setEnabled(false);
    m_playBtn->setText(tr("Stop export"));
    m_playing = true;
    m_playTimer.stop();
    if (m_lastResult.valid && m_lastResultFrame == m_currentFrame) {
        if (!appendVideoMasks(m_lastResult, m_currentFrame)) {
            finishVideoExport(false, tr("Could not write the current frame."));
            return;
        }
        m_videoExportProgress->setValue(m_currentFrame + 1);
        m_videoExportExpectedFrame = m_currentFrame + 1;
        schedulePlayback();
    } else {
        trackNextFrame();
    }
}

bool VideoTab::appendVideoMasks(const SAM3WorkerResult& result,
                                int frameIndex) {
    QJsonObject entry;
    if (!writeVideoMaskFrame(m_videoExportDir, frameIndex, result, &entry))
        return false;
    m_videoExportFrames.append(entry);
    return true;
}

void VideoTab::finishVideoExport(bool complete, const QString& error) {
    if (m_videoExportDir.isEmpty()) return;
    complete = complete && m_videoExportExpectedFrame == m_totalFrames &&
               m_videoExportFrames.size() == m_totalFrames - m_videoExportStart;
    QString archiveError = error;
    m_playing = false;
    m_playTimer.stop();
    m_playBtn->setText(tr("Play"));
    QJsonObject manifest{
            {QStringLiteral("format"),
             QStringLiteral("acloudviewer_sam3_tracking_v1")},
            {QStringLiteral("source"), m_videoPath},
            {QStringLiteral("model"), QFileInfo(m_loadedModelPath).fileName()},
            {QStringLiteral("device"), m_loadedDevice},
            {QStringLiteral("fps"), m_fps},
            {QStringLiteral("width"), m_currentFrameImage.width()},
            {QStringLiteral("height"), m_currentFrameImage.height()},
            {QStringLiteral("total_frames"), m_totalFrames},
            {QStringLiteral("start_frame"), m_videoExportStart},
            {QStringLiteral("complete"), complete},
            {QStringLiteral("full_video"), false},
            {QStringLiteral("frames"), m_videoExportFrames}};
    QString digestError;
    const QString sourceDigest =
            videoMaskArchiveSha256(m_videoPath, &digestError);
    if (!sourceDigest.isEmpty() && sourceDigest == m_videoExportSourceDigest) {
        manifest.insert(QStringLiteral("source_sha256"),
                        m_videoExportSourceDigest);
    } else {
        complete = false;
        const QString sourceError =
                sourceDigest.isEmpty()
                        ? tr("Could not read source video: %1").arg(digestError)
                        : tr("Source video changed during tracking.");
        archiveError =
                archiveError.isEmpty()
                        ? sourceError
                        : archiveError + QStringLiteral("; ") + sourceError;
        appendLog(sourceError);
    }
    manifest.insert(QStringLiteral("complete"), complete);
    manifest.insert(QStringLiteral("full_video"),
                    complete && m_videoExportStart == 0);
    if (!archiveError.isEmpty())
        manifest.insert(QStringLiteral("error"), archiveError);
    const QString manifestPath =
            QDir(m_videoExportDir).filePath(QStringLiteral("manifest.json"));
    const bool saved = writeVideoMaskManifest(m_videoExportDir, manifest);
    if (m_videoExportProgress) {
        disconnect(m_videoExportProgress, nullptr, this, nullptr);
        m_videoExportProgress->close();
        m_videoExportProgress->deleteLater();
        m_videoExportProgress = nullptr;
    }
    m_openBtn->setEnabled(true);
    m_testVideoCombo->setEnabled(true);
    m_testDataBtn->setEnabled(true);
    m_resetBtn->setEnabled(true);
    m_stepBtn->setEnabled(!m_busy && m_worker && m_worker->hasModel());
    m_exportVideoBtn->setEnabled(m_worker && m_worker->hasModel());
    m_modelCombo->setEnabled(!m_busy);
    m_deviceCombo->setEnabled(!m_busy);
    updateDownloadButton();
    m_canvas->setInteractive(!m_busy && m_worker && m_worker->hasModel());
    m_timeline->setEnabled(true);
    m_modeText->setEnabled(!m_visualOnly);
    m_modeBox->setEnabled(true);
    m_modePoints->setEnabled(true);
    m_textPrompt->setEnabled(!m_visualOnly && m_modeText->isChecked());
    m_scoreSpin->setEnabled(true);
    setStatus(saved ? (complete ? tr("Tracking saved: %1").arg(m_videoExportDir)
                                : tr("Partial tracking saved: %1 (%2)")
                                          .arg(m_videoExportDir, archiveError))
                    : tr("Could not write tracking manifest: %1")
                              .arg(manifestPath));
    m_videoExportDir.clear();
    m_videoExportSourceDigest.clear();
    m_videoExportFrames = QJsonArray();
    m_videoExportStart = -1;
    m_videoExportExpectedFrame = -1;
    m_videoExportAwaitedFrame = -1;
}

int VideoTab::exportMasksToDb() {
    if (!m_lastResult.valid || !m_app || m_lastResult.instanceMasks.isEmpty()) {
        return 0;
    }
    const QString deviceTag =
            ecvPluginDbNaming::deviceTagFromName(selectedDevice());
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(modelPath());
    const QString baseName = QFileInfo(m_videoPath).completeBaseName();
    int added = 0;
    for (int i = 0; i < m_lastResult.instanceMasks.size(); ++i) {
        const QImage mask = m_lastResult.instanceMasks.value(i);
        if (mask.isNull()) continue;
        const int id = m_lastResult.instanceIds.value(i, i + 1);
        const QString name = ecvPluginDbNaming::makeUnique(
                QStringLiteral("SAM3_Video_%1_%2_%3_mask_obj%4_frame%5")
                        .arg(modelTag, baseName, deviceTag)
                        .arg(id)
                        .arg(m_currentFrame, 4, 10, QLatin1Char('0')),
                m_app);
        auto* img = new ccImage(mask, name);
        img->setMetaData(QStringLiteral("SAM3"), true);
        img->setMetaData(QStringLiteral("SAM3/Kind"), QStringLiteral("mask"));
        img->setMetaData(QStringLiteral("SAM3/InstanceId"),
                         static_cast<qlonglong>(id));
        img->setMetaData(QStringLiteral("SAM3/Score"),
                         static_cast<double>(m_lastResult.scores.value(i)));
        img->setMetaData(QStringLiteral("SAM3/Frame"),
                         static_cast<qlonglong>(m_currentFrame));
        img->setMetaData(QStringLiteral("SAM3/Device"), selectedDevice());
        img->setMetaData(QStringLiteral("SAM3/Model"),
                         QFileInfo(m_loadedModelPath).fileName());
        if (!m_videoPath.isEmpty()) {
            img->setMetaData(QStringLiteral("Source"), m_videoPath);
        }
        m_app->addToDB(img, /*updateZoom=*/false, /*autoExpandDBTree=*/true,
                       /*checkDimensions=*/false, /*autoRedraw=*/true);
        ++added;
    }
    return added;
}

// ── Private helpers ────────────────────────────────────────────────────────

void VideoTab::updateCanvasInstances() {
    QVector<VideoInstanceBox> boxes;
    if (m_showMasks->isChecked() && m_lastResult.valid) {
        for (int i = 0; i < m_lastResult.detCount; ++i) {
            VideoInstanceBox b;
            b.id = m_lastResult.instanceIds.value(i);
            const aicore_sam3_box box = m_lastResult.boxes.value(i);
            b.box = QRectF(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
            b.color = instanceColor(b.id);
            b.score = m_lastResult.scores.value(i);
            boxes.append(b);
        }
    }
    m_canvas->setInstances(boxes, m_showMasks->isChecked()
                                          ? m_lastResult.instanceMasks
                                          : QVector<QImage>());

    // Instance list label.
    if (!m_lastResult.valid || m_lastResult.detCount <= 0) {
        m_instanceLabel->setText(tr("(none)"));
        m_instanceLabel->setStyleSheet("color: #999;");
    } else {
        QString html;
        for (int i = 0; i < m_lastResult.detCount; ++i) {
            const QColor c = instanceColor(m_lastResult.instanceIds.value(i));
            html += QString("<span style='color:%1; font-weight:bold;'>"
                            "object #%2 — score %3</span>&nbsp; ")
                            .arg(c.name())
                            .arg(m_lastResult.instanceIds.value(i))
                            .arg(m_lastResult.scores.value(i), 0, 'f', 2);
        }
        m_instanceLabel->setText(html);
        m_instanceLabel->setStyleSheet(QString());
    }
    m_exportBtn->setEnabled(m_lastResult.valid &&
                            !m_lastResult.instanceMasks.isEmpty());
    m_exportFrameToDbBtn->setEnabled(m_lastResult.valid && m_app);
}

void VideoTab::updateTimeline(int frameIndex, const SAM3WorkerResult& result) {
    if (frameIndex >= m_timelineEntries.size()) {
        m_timelineEntries.resize(frameIndex + 1);
    }
    m_timelineEntries[frameIndex].instances.clear();
    for (int i = 0; i < result.detCount; ++i) {
        m_timelineEntries[frameIndex].instances.append(
                {result.instanceIds.value(i), result.scores.value(i)});
        const int id = result.instanceIds.value(i);
        if (!m_timelineInstanceIds.contains(id)) {
            m_timelineInstanceIds.append(id);
        }
    }
    std::sort(m_timelineInstanceIds.begin(), m_timelineInstanceIds.end());

    QVector<QColor> colors;
    for (const int id : m_timelineInstanceIds) {
        colors.append(instanceColor(id));
    }
    m_timeline->setCurrentFrame(frameIndex);
    m_timeline->setProcessedMax(m_processedMax);
    m_timeline->setTimeline(m_timelineEntries);
    m_timeline->setInstances(m_timelineInstanceIds, colors);
}

QColor VideoTab::instanceColor(int id) const {
    const int ci = id > 0 ? (id - 1) % kNumColors : 0;
    return kInstanceColors[ci];
}

void VideoTab::resetPrompts() {
    m_canvas->setPromptPoints({}, {});
    m_canvas->clearBox();
}

void VideoTab::appendLog(const QString& msg) { setStatus(msg); }

void VideoTab::setStatus(const QString& msg) { m_statusLabel->setText(msg); }

void VideoTab::exportCurrentFrameToDb() {
    if (!m_lastResult.valid || !m_app || m_currentFrameImage.isNull()) return;

    // Build the annotated frame: composite masks atop the original frame.
    // The mask compositor below writes QRgb pixels. Decoded video frames are
    // commonly RGB888, so normalize the destination before 32-bit writes.
    QImage annotated =
            m_currentFrameImage.convertToFormat(QImage::Format_RGB32);
    if (!m_lastResult.instanceMasks.isEmpty()) {
        QPainter p(&annotated);
        for (int i = 0; i < m_lastResult.instanceMasks.size(); ++i) {
            const QColor tint =
                    instanceColor(m_lastResult.instanceIds.value(i));
            const QImage mask = m_lastResult.instanceMasks.value(i);
            // Blend the mask with its instance colour.
            for (int y = 0; y < mask.height() && y < annotated.height(); ++y) {
                const uchar* src = mask.scanLine(y);
                QRgb* dst = reinterpret_cast<QRgb*>(annotated.scanLine(y));
                for (int x = 0; x < mask.width() && x < annotated.width();
                     ++x) {
                    if (src[x] > 0) {
                        const float a = 0.45f;
                        const QRgb bg = dst[x];
                        const QRgb fg = tint.rgb();
                        dst[x] = qRgb(static_cast<int>(a * qRed(fg) +
                                                       (1 - a) * qRed(bg)),
                                      static_cast<int>(a * qGreen(fg) +
                                                       (1 - a) * qGreen(bg)),
                                      static_cast<int>(a * qBlue(fg) +
                                                       (1 - a) * qBlue(bg)));
                    }
                }
            }
        }
        p.end();
    }

    const QString deviceTag =
            ecvPluginDbNaming::deviceTagFromName(selectedDevice());
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(modelPath());
    const QString baseName = QFileInfo(m_videoPath).completeBaseName();
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3_Video_%1_%2_%3_frame%4")
                    .arg(modelTag, baseName, deviceTag)
                    .arg(m_currentFrame, 4, 10, QLatin1Char('0')),
            m_app);

    auto* img = new ccImage(annotated, name);
    img->setMetaData(QStringLiteral("SAM3"), true);
    img->setMetaData(QStringLiteral("SAM3/DetectionCount"),
                     static_cast<qlonglong>(m_lastResult.detCount));
    img->setMetaData(QStringLiteral("SAM3/Device"), selectedDevice());
    img->setMetaData(QStringLiteral("SAM3/Frame"),
                     static_cast<qlonglong>(m_currentFrame));
    img->setMetaData(QStringLiteral("Runtime (ms)"),
                     m_lastResult.timings.e2e_ms);
    if (!m_videoPath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), m_videoPath);
    }

    for (int i = 0; i < m_lastResult.detCount; ++i) {
        const QString p = QStringLiteral("SAM3/Det%1/").arg(i + 1);
        const aicore_sam3_box& b = m_lastResult.boxes.value(i);
        img->setMetaData(
                p + QStringLiteral("InstanceId"),
                static_cast<qlonglong>(m_lastResult.instanceIds.value(i)));
        img->setMetaData(p + QStringLiteral("Score"),
                         static_cast<double>(m_lastResult.scores.value(i)));
        img->setMetaData(p + QStringLiteral("Box"),
                         QStringLiteral("[%1,%2,%3,%4]")
                                 .arg(b.x0, 0, 'f', 1)
                                 .arg(b.y0, 0, 'f', 1)
                                 .arg(b.x1, 0, 'f', 1)
                                 .arg(b.y1, 0, 'f', 1));
    }

    m_app->addToDB(img, /*updateZoom=*/false, /*autoExpandDBTree=*/true,
                   /*checkDimensions=*/false, /*autoRedraw=*/true);
    m_app->setSelectedInDB(img, true);
}

// ── Test data (shared ecvTestDataRepository, SAM3 dataset) ─────────────────

void VideoTab::populateTestVideoCombo() {
    if (!m_testVideoCombo) return;
    const QStringList videos = ecvTestDataRepository::getSamVideos(
            ecvTestDataRepository::extractPath(
                    ecvTestDataRepository::Dataset::SAM3));
    m_testVideoCombo->blockSignals(true);
    m_testVideoCombo->clear();
    if (videos.isEmpty()) {
        m_testVideoCombo->addItem(tr("(no test video)"), QString());
    } else {
        for (const QString& path : videos) {
            const QString name = QFileInfo(path).fileName();
            m_testVideoCombo->addItem(name, name);
        }
    }
    m_testVideoCombo->blockSignals(false);
}

bool VideoTab::loadRequestedTestVideo() {
    if (!m_testVideoCombo) return false;
    const QString fileName = m_testVideoCombo->currentData().toString();
    if (fileName.isEmpty()) return false;

    const QString path = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::SAM3, fileName);
    if (path.isEmpty()) return false;

    openVideoFile(path);
    appendLog(tr("[Test data] Loaded video: %1").arg(path));
    appendLog(tr("Load a model, then pause and annotate to add instances."));
    return true;
}

void VideoTab::onUseTestData() {
    if (m_testDataDownloadInProgress) {
        appendLog(tr("[Test data] Download already in progress."));
        return;
    }
    if (m_busy) {
        appendLog(tr("[Test data] Wait for the current task to finish."));
        return;
    }
    if (loadRequestedTestVideo()) return;

    auto& repo = ecvTestDataRepository::instance();
    if (repo.isDownloadInProgress()) {
        appendLog(tr("[Test data] Another test-data download is running."));
        return;
    }

    const auto kind = ecvTestDataRepository::Dataset::SAM3;
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    m_testDataDownloadInProgress = true;
    setTestDataControlsEnabled(false);
    if (m_progress) {
        m_progress->setVisible(true);
        m_progress->setValue(0);
    }
    if (m_downloadLabel) {
        m_downloadLabel->setVisible(true);
    }
    if (ecvAssetIntegrity::isVerified(ecvTestDataRepository::zipPath(kind),
                                      info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        appendLog(tr("[Test data] Extracting cached archive..."));
        setStatus(tr("Extracting SAM3 test data..."));
        repo.extractDataset(kind);
        return;
    }
    appendLog(tr("[Test data] Downloading SAM3 test data..."));
    setStatus(tr("Downloading SAM3 test data..."));
    repo.startDownload(kind);
}

void VideoTab::onTestDataDownloadFinished(bool success,
                                          ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::SAM3) {
        return;
    }
    if (!success) {
        appendLog(tr("[Test data] Download failed."));
        m_testDataDownloadInProgress = false;
        setTestDataControlsEnabled(true);
        if (m_progress) m_progress->setVisible(false);
        if (m_downloadLabel) m_downloadLabel->setVisible(false);
        setStatus(tr("Ready."));
        return;
    }
    appendLog(tr("[Test data] Extracting..."));
    setStatus(tr("Extracting SAM3 test data..."));
    if (m_progress) m_progress->setValue(0);
    ecvTestDataRepository::instance().extractDataset(kind);
}

void VideoTab::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::SAM3) {
        return;
    }
    m_testDataDownloadInProgress = false;
    setTestDataControlsEnabled(true);

    if (m_progress) m_progress->setVisible(false);
    if (m_downloadLabel) m_downloadLabel->setVisible(false);

    if (!success) {
        appendLog(tr("[Test data] Failed to extract zip archive."));
        setStatus(tr("Ready."));
        return;
    }
    // The picker was empty before the first extract; fill it and open the
    // (first) video so the one-click flow completes automatically.
    populateTestVideoCombo();
    if (!loadRequestedTestVideo()) {
        appendLog(tr("[Test data] Sample video not found in the archive."));
    }
    setStatus(tr("Ready."));
}

void VideoTab::setTestDataControlsEnabled(bool enabled) {
    if (m_testDataBtn) m_testDataBtn->setEnabled(enabled);
    if (m_testVideoCombo) m_testVideoCombo->setEnabled(enabled);
}
