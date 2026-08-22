// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceDetectDialog.h"

#include <CVLog.h>
#include <cvFileDialog.h>

#include <QCloseEvent>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QGridLayout>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QScreen>
#include <QSet>
#include <QSettings>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>

#include "FaceDetectEmbedHelpers.h"
#include "FaceDetectTestData.h"
#include "FaceDetectTestDataWorker.h"
#include "FaceDetectUiHelpers.h"
#include "ecvAICoreUiHelper.h"
#include "FaceLiveDetectWidget.h"
#include "FaceRegistryWidget.h"
#include "aicore/backend_capi.h"
#include "aicore/facedetect_capi.h"
#include "aicore/inference_log.h"
#include "ecvModelDownloader.h"

namespace {

constexpr auto kFriends = ecvTestDataRepository::Dataset::FriendsFaces;

const int kThumbSize = ecvAICoreUi::previewSize();
constexpr int kTabViewportMinHeight = 280;
// QListWidgetItem data role carrying the full-resolution ccImage for the
// click-to-enlarge preview (the 24 px icon is only for list display).
constexpr int kDbFullImageRole = Qt::UserRole + 1;

// Qt logical coordinates scale with the active style/font metrics, but
// plain integer clamps do NOT.  On Windows 150%% scaling the logical DPI
// is 144, so a 280 px minimum must become 420 px.  Scale every hardcoded
// pixel so tab viewport math stays correct across resolutions and
// per-monitor DPI (1080p@100%% vs 4K@150%%).
// Shared DPI-aware scaling from the AICore UI kit (same semantics as the
// former local static function).
using ecvAICoreUi::dpiScaled;

bool isSupportedImageFile(const QString& filePath) {
    static const QStringList extensions = {
            QStringLiteral("png"),  QStringLiteral("jpg"),
            QStringLiteral("jpeg"), QStringLiteral("bmp"),
            QStringLiteral("tif"),  QStringLiteral("tiff"),
            QStringLiteral("webp"),
    };
    return extensions.contains(QFileInfo(filePath).suffix(),
                               Qt::CaseInsensitive);
}

bool isValidCachedGguf(const QFileInfo& fi) {
    return ecvModelDownloader::isValidCachedFile(fi.absoluteFilePath());
}

}  // namespace

QString FaceDetectDialog::modelCacheDir() {
    return FaceDetectEmbed::modelCacheDir();
}

QString FaceDetectDialog::registryPath() {
    return modelCacheDir() + QStringLiteral("/face_registry.db");
}

FaceDetectDialog::FaceDetectDialog(QWidget* parent) : QDialog(parent) {
    FaceDetectTestData::purgeFriendsPathsFromSettings();

    setWindowTitle(tr("Face Detect"));
    setMinimumWidth(ecvAICoreUi::dpiScaled(760));
    setMinimumHeight(ecvAICoreUi::dpiScaled(520));
    setSizeGripEnabled(false);

    QSettings settings;
    const QString savedManualVideo =
            settings.value(FaceDetectTestData::manualLiveVideoSettingsKey())
                    .toString();
    QString savedVideo = savedManualVideo;
    if (savedVideo.isEmpty()) {
        const QString legacy =
                settings.value(QStringLiteral("qFaceDetect/liveVideoPath"))
                        .toString();
        if (!legacy.isEmpty() &&
            !FaceDetectTestData::isFriendsBundlePath(legacy)) {
            savedVideo = legacy;
        }
    }
    const bool savedVideoIsManual = !savedVideo.isEmpty();

    auto* outer = new QVBoxLayout(this);
    m_tabWidget = new QTabWidget(this);
    // Expanding lets the active page (and its video preview) grow with the
    // dialog; the per-tab minimum height is managed by
    // updateActiveTabViewportHeight.
    m_tabWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    ecvAICoreUi::styleTabWidget(m_tabWidget);

    m_batchTab = new QWidget;
    setupBatchTab(m_batchTab);
    m_tabWidget->addTab(m_batchTab, tr("Image / Batch"));

    m_registryWidget = new FaceRegistryWidget;
    connect(m_registryWidget, &FaceRegistryWidget::logMessage, this,
            &FaceDetectDialog::appendLog);
    connect(m_registryWidget, &FaceRegistryWidget::registryChanged, this,
            [this]() {
                if (m_liveWidget && m_registryWidget) {
                    m_liveWidget->setRegistryStore(m_registryWidget->store());
                }
            });
    connect(m_registryWidget, &FaceRegistryWidget::testDataRequested, this,
            [this]() { ensureFriendsTestData(true, true, true); });
    connect(m_registryWidget, &FaceRegistryWidget::modelSelectionChanged, this,
            [this](const QString& modelFilename) {
                if (!m_modelCombo || modelFilename.isEmpty()) return;
                const int idx = m_modelCombo->findData(modelFilename);
                if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
                syncLiveConfig();
            });
    connect(m_registryWidget, &FaceRegistryWidget::deviceSelectionChanged, this,
            [this](const QString& deviceId) {
                if (!m_deviceCombo) return;
                const int idx = m_deviceCombo->findData(deviceId);
                if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
                syncLiveConfig();
            });
    connect(m_registryWidget, &FaceRegistryWidget::threadCountChanged, this,
            [this](int threads) {
                if (m_threads) m_threads->setValue(threads);
                syncLiveConfig();
            });
    connect(m_registryWidget, &FaceRegistryWidget::authResultImageReady, this,
            &FaceDetectDialog::onAuthResultImageReady);
    m_tabWidget->addTab(m_registryWidget, tr("Registry / Auth"));

    auto* liveTab = new QWidget;
    auto* liveLayout = new QVBoxLayout(liveTab);
    if (FaceLiveDetectWidget::isAvailable()) {
        m_liveWidget = new FaceLiveDetectWidget(liveTab);
        if (savedVideoIsManual && !savedVideo.isEmpty()) {
            m_liveWidget->setVideoFilePath(savedVideo, true);
        }
        liveLayout->addWidget(m_liveWidget, 1);
        // The playback controls row appears only once a video is loaded;
        // re-measure the tab viewport so the new row is never squeezed into
        // the preview area (the "controls overlapping the video" bug).
        connect(m_liveWidget,
                &FaceLiveDetectWidget::videoControlsVisibilityChanged, this,
                [this](bool) {
                    QTimer::singleShot(0, this, [this]() {
                        cacheTabViewportHeights();
                        updateActiveTabViewportHeight();
                    });
                });
        auto* liveBtnRow = new QHBoxLayout;
        m_liveStartBtn = new QPushButton(tr("Start"));
        m_liveStopBtn = new QPushButton(tr("Stop"));
        m_liveRestartBtn = new QPushButton(tr("Restart"));
        m_liveStopBtn->setEnabled(false);
        m_liveRestartBtn->setEnabled(false);
        connect(m_liveStartBtn, &QPushButton::clicked, this,
                &FaceDetectDialog::onLiveStart);
        connect(m_liveStopBtn, &QPushButton::clicked, this,
                &FaceDetectDialog::onLiveStop);
        connect(m_liveRestartBtn, &QPushButton::clicked, this,
                &FaceDetectDialog::onLiveRestart);
        connect(m_liveWidget, &FaceLiveDetectWidget::logMessage, this,
                &FaceDetectDialog::appendLog);
        connect(m_liveWidget, &FaceLiveDetectWidget::captureToDbRequested, this,
                &FaceDetectDialog::onLiveCapture);
        connect(m_liveWidget, &FaceLiveDetectWidget::streamStarted, this,
                [this]() {
                    if (m_liveStartBtn) m_liveStartBtn->setEnabled(false);
                    if (m_liveStopBtn) m_liveStopBtn->setEnabled(true);
                    if (m_liveRestartBtn && m_liveWidget &&
                        m_liveWidget->inputSource() ==
                                FaceLiveDetectWidget::InputSource::VideoFile) {
                        m_liveRestartBtn->setEnabled(true);
                    }
                });
        connect(m_liveWidget, &FaceLiveDetectWidget::streamStopped, this,
                [this]() {
                    if (m_liveStartBtn) m_liveStartBtn->setEnabled(true);
                    if (m_liveStopBtn) m_liveStopBtn->setEnabled(false);
                    // Restart stays enabled if video is paused (can restart)
                    if (m_liveRestartBtn &&
                        (!m_liveWidget ||
                         m_liveWidget->inputSource() !=
                                 FaceLiveDetectWidget::InputSource::
                                         VideoFile)) {
                        m_liveRestartBtn->setEnabled(false);
                    }
                });
        auto* liveTestDataBtn = m_liveWidget->testDataButton();
        liveBtnRow->addWidget(liveTestDataBtn);
        liveBtnRow->addWidget(m_liveStartBtn);
        liveBtnRow->addWidget(m_liveStopBtn);
        liveBtnRow->addWidget(m_liveRestartBtn);
        liveBtnRow->addStretch();
        liveLayout->addLayout(liveBtnRow);
        connect(m_liveWidget, &FaceLiveDetectWidget::streamModeChanged, this,
                [this](FaceLiveDetectWidget::StreamMode mode) {
                    onLiveStreamModeChanged(static_cast<int>(mode), true);
                });
        connect(m_liveWidget, &FaceLiveDetectWidget::testDataRequested, this,
                [this]() { ensureFriendsTestData(false, true, false); });
        connect(m_liveWidget, &FaceLiveDetectWidget::registryPathEdited, this,
                [this](const QString& path) {
                    if (m_registryWidget) {
                        m_registryWidget->setRegistryPath(path, true);
                    }
                });
        connect(m_liveWidget, &FaceLiveDetectWidget::modelSelectionChanged,
                this, [this](const QString& modelFilename) {
                    if (!m_modelCombo || modelFilename.isEmpty()) return;
                    const int idx = m_modelCombo->findData(modelFilename);
                    if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
                    syncLiveConfig();
                });
        connect(m_liveWidget, &FaceLiveDetectWidget::deviceSelectionChanged,
                this, [this](const QString& deviceId) {
                    if (!m_deviceCombo) return;
                    const int idx = m_deviceCombo->findData(deviceId);
                    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
                    syncLiveConfig();
                });
        connect(m_liveWidget, &FaceLiveDetectWidget::threadCountChanged, this,
                [this](int threads) {
                    if (m_threads) m_threads->setValue(threads);
                    syncLiveConfig();
                });
    } else {
        liveLayout->addWidget(new QLabel(tr(
                "Live detect requires OpenCV videoio (enable BUILD_OPENCV).")));
    }
    m_tabWidget->addTab(liveTab, tr("Live (camera / video)"));

    if (m_registryWidget && m_liveWidget) {
        connect(m_registryWidget, &FaceRegistryWidget::registryPathChanged,
                this, [this](const QString& path) {
                    if (m_liveWidget) {
                        m_liveWidget->setRegistryPath(
                                path,
                                m_registryWidget->isRegistryPathUserChosen());
                    }
                });
    }

    connect(m_tabWidget, &QTabWidget::currentChanged, this, [this](int) {
        // The stacked page changes after this signal. Defer the measurement so
        // a short tab never inherits the former page's height.  The deferred
        // resize uses the minimumSizeHint-derived chrome, which is stable
        // regardless of the current geometry.
        QTimer::singleShot(0, this,
                           [this]() { updateActiveTabViewportHeight(); });
    });
    outer->addWidget(m_tabWidget, 0);

    ecvAICoreUi::setupProgressSection(outer, m_downloadLabel, m_progress);
    m_progress->setFixedHeight(ecvAICoreUi::dpiScaled(16));
    m_progress->setTextVisible(false);
    // The progress bar stays visible as reserved space (the shared helper
    // hides it by default); only the label toggles during transfers.
    m_progress->setVisible(true);
    m_progress->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    m_testDataWorker = new FaceDetectTestDataWorker(this);
    connect(m_testDataWorker, &FaceDetectTestDataWorker::phaseProgress, this,
            &FaceDetectDialog::updateTestDataProgress);
    connect(m_testDataWorker, &FaceDetectTestDataWorker::logMessage, this,
            &FaceDetectDialog::appendLog);
    connect(m_testDataWorker, &FaceDetectTestDataWorker::finished, this,
            [this](bool ok, int registered, int authFaces, int authMatched) {
                const bool wasRegistryJob = m_testPostFillRegistry;
                m_testDataProcessing = false;
                setTestDataBusy(false);
                if (m_progress) {
                    m_progress->setValue(m_progress->maximum());
                }
                if (m_downloadLabel) {
                    if (ok) {
                        m_downloadLabel->setText(tr("Test data ready."));
                    } else {
                        m_downloadLabel->setText(tr("Test data setup failed."));
                    }
                }
                if (!ok) {
                    if (wasRegistryJob) {
                        QMessageBox::warning(this, tr("Test data setup failed"),
                                             tr("FriendsFaces registration did "
                                                "not complete.\n\n"
                                                "Check the console log for "
                                                "details (registry path, "
                                                "model availability)."));
                    } else {
                        QMessageBox::warning(
                                this, tr("Test data setup failed"),
                                tr("Could not extract FriendsFaces sample "
                                   "data.\n\n"
                                   "Check the console log for details."));
                    }
                } else {
                    if (wasRegistryJob && m_registryWidget) {
                        m_registryWidget->setRegistryPath(
                                m_registryWidget->registryPath(), false);
                        m_registryWidget->refreshList();
                        if (authFaces > 0) {
                            m_registryWidget->showVerifySummary(
                                    authFaces, authMatched,
                                    m_registryWidget->authThreshold());
                        }
                    }
                    if (wasRegistryJob && m_liveWidget && m_registryWidget) {
                        m_liveWidget->setRegistryStore(
                                m_registryWidget->store());
                    }
                    FaceDetectFriendsBundle bundle;
                    if (tryResolveFriendsTestBundle(&bundle)) {
                        if (wasRegistryJob && m_registryWidget) {
                            m_registryWidget->fillFriendsTestBundleFields(
                                    bundle);
                        }
                        if (wasRegistryJob && m_liveWidget &&
                            m_registryWidget) {
                            m_liveWidget->setRegistryPath(
                                    m_registryWidget->registryPath(),
                                    m_registryWidget
                                            ->isRegistryPathUserChosen());
                        }
                        if (m_testPostFillLiveVideo ||
                            m_testPostFillBatchImage) {
                            applyFriendsTestDataPaths(bundle,
                                                      m_testPostFillLiveVideo,
                                                      m_testPostFillBatchImage);
                        }
                    }
                }
                m_testPostFillRegistry = false;
                m_testPostFillLiveVideo = false;
                m_testPostFillBatchImage = false;
                if (wasRegistryJob) {
                    appendLog(tr("[Test data] Done — registered %1, verify "
                                 "%2/%3 face(s).")
                                      .arg(registered)
                                      .arg(authMatched)
                                      .arg(authFaces));
                } else {
                    appendLog(tr("[Test data] Done — sample paths ready."));
                }
                QTimer::singleShot(1500, this, [this]() {
                    if (!m_testDataDownloadInProgress &&
                        !m_downloadInProgress && !m_testDataProcessing &&
                        m_downloadLabel) {
                        m_downloadLabel->setVisible(false);
                    }
                    if (m_progress && !m_testDataDownloadInProgress &&
                        !m_downloadInProgress && !m_testDataProcessing) {
                        m_progress->setValue(0);
                        m_progress->setTextVisible(false);
                        m_progress->setMaximum(100);
                    }
                });
            });

    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &FaceDetectDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (total > 0 && m_progress) {
                    m_progress->setValue(
                            static_cast<int>(received * 100 / total));
                }
                if (m_downloadLabel) {
                    m_downloadLabel->setText(
                            tr("Downloading %1 — %2")
                                    .arg(m_modelCombo->currentData().toString())
                                    .arg(ecvModelDownloader::
                                                 formatDownloadProgress(
                                                         received, total)));
                }
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                const QString finishedFilename = QFileInfo(dest).fileName();
                m_downloadInProgress = false;
                m_downloadLabel->setVisible(false);
                if (ok) {
                    appendLog(tr("[OK] Downloaded model: %1").arg(dest));
                    populateModelCombo(finishedFilename);
                    populateLandmarkModelCombo();
                    if (m_autoRunAfterDownload) {
                        m_autoRunAfterDownload = false;
                        onRun();
                    }
                } else {
                    populateModelCombo(finishedFilename);
                    m_autoRunAfterDownload = false;
                }
            });
    m_testDataDownloader = new ecvModelDownloader(this);
    connect(m_testDataDownloader, &ecvModelDownloader::logMessage, this,
            &FaceDetectDialog::appendLog);
    connect(m_testDataDownloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (!m_testDataDownloadInProgress || total <= 0 ||
                    !m_progress) {
                    return;
                }
                // m_progress is always visible (reserved space)
                if (m_downloadLabel) m_downloadLabel->setVisible(true);
                m_progress->setMaximum(kTestDataOverallMax);
                m_progress->setTextVisible(true);
                m_progress->setFormat(tr("%p%"));
                m_progress->setValue(static_cast<int>(
                        received * kTestDataDownloadShare / total));
                if (m_downloadLabel) {
                    m_downloadLabel->setText(
                            tr("Downloading FriendsFaces test data — %1")
                                    .arg(ecvModelDownloader::
                                                 formatDownloadProgress(
                                                         received, total)));
                }
            });
    connect(m_testDataDownloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                m_testDataDownloadInProgress = false;
                const bool fillRegistry = m_testFillRegistry;
                const bool fillLive = m_testFillLiveVideo;
                const bool fillBatch = m_testFillBatchImage;
                m_testFillRegistry = false;
                m_testFillLiveVideo = false;
                m_testFillBatchImage = false;
                if (!ok) {
                    setTestDataBusy(false);
                    if (m_downloadLabel) m_downloadLabel->setVisible(false);
                    appendLog(tr("[Test data] Download failed."));
                    return;
                }
                if (!ecvTestDataRepository::verifyZipIntegrity(
                            dest,
                            ecvTestDataRepository::getDatasetInfo(kFriends)
                                    .expectedMd5,
                            ecvTestDataRepository::getDatasetInfo(kFriends)
                                    .expectedSize)) {
                    QFile::remove(dest);
                    setTestDataBusy(false);
                    if (m_downloadLabel) m_downloadLabel->setVisible(false);
                    appendLog(tr(
                            "[Test data] Download rejected: friends_faces.zip "
                            "failed MD5 integrity check (incomplete or "
                            "corrupted)."));
                    QMessageBox::warning(this, tr("Test data download"),
                                         tr("The downloaded friends_faces.zip "
                                            "failed integrity "
                                            "verification and was "
                                            "removed.\n\nPlease retry the "
                                            "download."));
                    return;
                }
                appendLog(tr("[Test data] Downloaded %1 (MD5 OK)").arg(dest));
                QDir().mkpath(ecvTestDataRepository::extractDir());
                FaceDetectFriendsBundle bundle;
                m_testDataPostProgressBase = kTestDataDownloadShare;
                const bool clearExisting = m_testClearExistingEntries;
                startTestDataPostProcess(bundle, fillRegistry, fillLive,
                                         fillBatch, true, dest, clearExisting);
            });
    CVLog::Print(QString("[FaceDetect] Model cache: %1").arg(modelCacheDir()));
    aicore_inference_log::log_backend_probe(QStringLiteral("FaceDetect"));
    populateModelCombo();
    populateLandmarkModelCombo();
    if (m_registryWidget) {
        m_registryWidget->loadSettings();
    }
    syncLiveConfig();
    tryAutoDiscoverRegistryDb();
    if (m_liveWidget && m_registryWidget) {
        m_liveWidget->setRegistryStore(m_registryWidget->store());
    }
    setupMatchThresholdLinks();
    setupMinScoreLinks();
    loadBatchSettings();
    validateLiveRecognizeModeFromSettings();
    cacheTabViewportHeights();
    updateActiveTabViewportHeight();
}

void FaceDetectDialog::cacheTabViewportHeights() {
    if (!m_tabWidget) return;
    for (int i = 0; i < m_tabWidget->count(); ++i) {
        QWidget* content = m_tabWidget->widget(i);
        if (!content) continue;
        // Use sizeHint (not minimumSizeHint): a form tab's minimum is
        // often far below what its content actually needs (e.g. the batch
        // tab min 380 vs real 750), and sizing the viewport to the minimum
        // inflated the dialog on every tab switch.  Captured before a live
        // preview receives pixmaps; later content growth belongs to the
        // page's own scroll area.
        m_tabViewportHeights.insert(content,
                                    content->sizeHint().height());
    }
}

void FaceDetectDialog::updateActiveTabViewportHeight() {
    if (!m_tabWidget) return;
    QWidget* content = m_tabWidget->currentWidget();
    if (!content) return;

    const int tabChrome =
            m_tabWidget->tabBar()->sizeHint().height() +
            2 * m_tabWidget->style()->pixelMetric(QStyle::PM_DefaultFrameWidth);
    const int cachedHeight = m_tabViewportHeights.value(
            content, content->minimumSizeHint().height());
    // Use the construction-time layout size, not a live preview's
    // pixmap-driven size hint. The active page owns its overflow.
    const int contentHeight =
            std::max(dpiScaled(kTabViewportMinHeight), cachedHeight);
    const int targetHeight = tabChrome + contentHeight;

    // Minimum (not fixed) height: the active page may grow taller when the
    // user enlarges the dialog (video preview scales with the window);
    // content is never compressed below its minimum, so controls cannot be
    // pushed over the video area.
    m_tabWidget->setMinimumHeight(targetHeight);
    m_tabWidget->updateGeometry();
    if (isVisible() && m_baseChrome >= 0 &&
        targetHeight != m_activeTabHeight) {
        // Dialog height = stable non-tab chrome + the incoming tab's
        // content.  The chrome is derived from minimumSizeHint deltas (not
        // the current geometry) so it stays constant no matter how the
        // user resized the window — the old delta formula mixed
        // minimum-based and sizeHint-based numbers, inflating the dialog
        // by hundreds of pixels on every tab switch, and the video preview
        // then absorbed the surplus as a huge empty area.
        const QScreen* screen =
                QGuiApplication::screenAt(frameGeometry().center());
        const int available =
                screen ? screen->availableGeometry().height() : 800;
        resize(width(),
               qBound(dpiScaled(420), m_baseChrome + targetHeight,
                      available - dpiScaled(20)));
    }
    // Tab content is measured at construction and when the video controls row
    // appears; after that, each tab's own scroll area handles overflow.  The
    // dialog itself has no permanent minimum clamp so switching to a shorter
    // tab works correctly (setMinimumHeight is updated per-tab below).
    m_activeTabHeight = targetHeight;
}

void FaceDetectDialog::changeEvent(QEvent* event) {
    QDialog::changeEvent(event);
    // Windows per-monitor DPI: moving to a differently-scaled display (or
    // changing the scale factor) invalidates BOTH the cached
    // minimumSizeHint values and the hardcoded clamps.  Re-cache and
    // re-measure so the active tab never shows clipped content or stale
    // height from the previous monitor.
    if (event->type() == QEvent::ScreenChangeInternal) {
        QTimer::singleShot(0, this, [this]() {
            cacheTabViewportHeights();
            // Re-measure the chrome on the new monitor: Qt already
            // resized the window to the new DPI before this deferred
            // callback runs, so the height() - tab height delta reflects
            // the fresh decorations and style metrics.
            if (m_tabWidget) {
                m_baseChrome = std::max(0, height() - m_tabWidget->height());
            }
            updateActiveTabViewportHeight();
            update();
        });
    }
}

void FaceDetectDialog::setupBatchTab(QWidget* batchTab) {
    batchTab->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto* main = new QVBoxLayout(batchTab);
    FaceDetectUi::setupCompactMainLayout(main);
    ecvAICoreUi::setupTabLayout(main);

    auto* testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    testDataBtn->setToolTip(tr(
            "Download FriendsFaces sample pack and fill batch image path "
            "(does not register identities — use Registry / Auth tab for "
            "that).\n\n"
            "Downloads the FriendsFaces sample pack and fills the batch image "
            "path (group photo). Does not register identities — use "
            "Registry / Auth for enrollment and authentication."));

    connect(testDataBtn, &QPushButton::clicked, this,
            [this]() { ensureFriendsTestData(false, false, true); });

    auto* modelGroup = new QGroupBox(tr("Model"));
    auto* modelLayout = new QGridLayout(modelGroup);
    FaceDetectUi::setupTwoColumnFormGrid(modelLayout);
    FaceDetectUi::tightenGroupBox(modelGroup);

    auto* pipelineHint = new QLabel(
            tr("<b>Pipeline:</b> Detect → Analyze → Dense Landmarks → Verify"));
    pipelineHint->setWordWrap(true);
    pipelineHint->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    pipelineHint->setStyleSheet(
            "color: #334155; background: #f8fafc; border: 1px solid #cbd5e1; "
            "padding: 2px 6px; border-radius: 3px; font-size: 11px;");
    modelLayout->addWidget(pipelineHint, 0, 0, 1, 4);

    m_modelCombo = new QComboBox;
    modelLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Detector GGUF:")), 1,
                           0);
    modelLayout->addWidget(m_modelCombo, 1, 1);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceDetectDialog::onModelComboChanged);

    m_modeCombo = new QComboBox;
    m_modeCombo->addItem(tr("Detect (boxes + 5 landmarks)"),
                         static_cast<int>(Mode::Detect));
    m_modeCombo->addItem(tr("Analyze (age + gender)"),
                         static_cast<int>(Mode::Analyze));
    m_modeCombo->addItem(tr("Dense Landmarks (106 2D + 68 3D)"),
                         static_cast<int>(Mode::DenseLandmarks));
    m_modeCombo->addItem(tr("Verify (identity match)"),
                         static_cast<int>(Mode::Verify));
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceDetectDialog::onModeChanged);
    modelLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Mode:")), 1, 2);
    modelLayout->addWidget(m_modeCombo, 1, 3);

    m_variantHintLabel = new QLabel;
    m_variantHintLabel->setWordWrap(true);
    m_variantHintLabel->setSizePolicy(QSizePolicy::Preferred,
                                      QSizePolicy::Maximum);
    m_variantHintLabel->setStyleSheet(
            "color: #333; background: #f3e8ff; border: 1px solid #c4b5fd; "
            "padding: 2px 6px; border-radius: 4px; font-size: 11px;");
    modelLayout->addWidget(m_variantHintLabel, 2, 0, 1, 4);

    m_customModelRow = new QWidget;
    auto* customLayout = new QHBoxLayout(m_customModelRow);
    m_customModelPath = new QLineEdit;
    auto* browseModel =
            FaceDetectUi::makeBrowseButton(tr("Browse…"), m_customModelRow);
    connect(browseModel, &QPushButton::clicked, this,
            &FaceDetectDialog::onBrowseCustomModel);
    customLayout->addWidget(m_customModelPath, 1);
    customLayout->addWidget(browseModel);
    m_customModelRow->setVisible(false);
    modelLayout->addWidget(m_customModelRow, 3, 0, 1, 4);

    m_landmarkModelRow = new QWidget;
    auto* landmarkLayout = new QGridLayout(m_landmarkModelRow);
    landmarkLayout->setContentsMargins(0, 0, 0, 0);
    FaceDetectUi::tightenFormGrid(landmarkLayout);
    m_landmarkModelCombo = new QComboBox;
    landmarkLayout->addWidget(new QLabel(tr("Landmark GGUF:")), 0, 0);
    landmarkLayout->addWidget(m_landmarkModelCombo, 0, 1, 1, 3);
    connect(m_landmarkModelCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &FaceDetectDialog::onLandmarkModelComboChanged);

    m_customLandmarkModelPath = new QLineEdit;
    m_customLandmarkModelPath->setPlaceholderText(
            tr("Path to landmark .gguf (auto-filled from selection above)"));
    auto* browseLandmarkModel =
            FaceDetectUi::makeBrowseButton(tr("Browse…"), m_landmarkModelRow);
    connect(browseLandmarkModel, &QPushButton::clicked, this,
            &FaceDetectDialog::onBrowseCustomLandmarkModel);
    landmarkLayout->addWidget(new QLabel(tr("Model path:")), 1, 0);
    landmarkLayout->addWidget(m_customLandmarkModelPath, 1, 1, 1, 2);
    landmarkLayout->addWidget(browseLandmarkModel, 1, 3);
    m_customLandmarkModelRow = nullptr;
    m_landmarkModelRow->setVisible(false);
    modelLayout->addWidget(m_landmarkModelRow, 4, 0, 1, 4);

    m_deviceCombo = new QComboBox;
    for (int i = 0; i < aicore_device_count(); ++i) {
        if (const aicore_device_info* d = aicore_device_at(i)) {
            m_deviceCombo->addItem(tr(d->label), QString::fromUtf8(d->id));
            if (d->is_default) m_deviceCombo->setCurrentIndex(i);
        }
    }
    m_threads = new QSpinBox;
    m_threads->setRange(0, 128);
    m_threads->setSpecialValueText(tr("Auto"));
    FaceDetectUi::makeCompactSpin(m_threads);
    modelLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Device:")), 5, 0);
    modelLayout->addWidget(m_deviceCombo, 5, 1);
    modelLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Threads:")), 5, 2);
    modelLayout->addWidget(m_threads, 5, 3, Qt::AlignLeft);
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                syncRegistryModelControlsFromBatch();
                syncLiveConfig();
            });
    connect(m_threads, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) {
                syncRegistryModelControlsFromBatch();
                syncLiveConfig();
            });

    m_minScoreLabel = FaceDetectUi::makeFormLabel(tr("Min score:"));
    m_minDetectionScore = FaceDetectUi::makeMinDetectionScoreSpin(
            modelGroup,
            tr("Detector confidence in [0, 1] — higher means a stronger face "
               "detection. Faces below this value are discarded (Detect / "
               "Analyze only). This is not the Verify cosine distance."));
    modelLayout->addWidget(m_minScoreLabel, 6, 0);
    modelLayout->addWidget(m_minDetectionScore, 6, 1, Qt::AlignLeft);
    m_linkMatchThresholdsCheck =
            new QCheckBox(tr("Link match threshold across tabs"), modelGroup);
    m_linkMatchThresholdsCheck->setChecked(true);
    m_linkMatchThresholdsCheck->setToolTip(tr(
            "When enabled, changing match threshold on any tab updates Batch "
            "Verify, Registry Auth, and Live Recognize."));
    m_applyMatchThresholdBtn = new QPushButton(
            tr("Apply match threshold to all tabs"), modelGroup);
    auto* matchThreshRow = new QHBoxLayout;
    matchThreshRow->setContentsMargins(0, 0, 0, 0);
    matchThreshRow->addWidget(m_linkMatchThresholdsCheck);
    matchThreshRow->addWidget(m_applyMatchThresholdBtn);
    matchThreshRow->addStretch();
    modelLayout->addLayout(matchThreshRow, 7, 0, 1, 4);
    m_batchMinScoreRow = nullptr;
    main->addWidget(modelGroup);

    auto* ioGroup = new QGroupBox(tr("Input"));
    FaceDetectUi::tightenGroupBox(ioGroup);
    auto* ioLayout = new QVBoxLayout(ioGroup);
    ioLayout->setContentsMargins(6, 4, 6, 4);
    ioLayout->setSpacing(3);
    auto* pathRow = new QHBoxLayout;
    pathRow->setContentsMargins(0, 0, 0, 0);
    pathRow->setSpacing(6);
    m_imagePath = new QLineEdit;
    m_imagePath->setPlaceholderText(
            tr("Image file path or db://EntityName from DB tree"));
    // Long db:// entity names get truncated inside the line edit; the
    // tooltip always shows the complete path.
    m_imagePath->setToolTip(m_imagePath->placeholderText());
    connect(m_imagePath, &QLineEdit::textChanged, this,
            [this](const QString& text) {
                if (m_imagePath) m_imagePath->setToolTip(text);
                updateImagePreview();
            });
    connect(m_imagePath, &QLineEdit::editingFinished, this, [this]() {
        if (!m_imagePath) return;
        const QString path = m_imagePath->text().trimmed();
        if (!path.isEmpty()) {
            m_batchImagePathUserChosen = true;
        }
    });
    // Focus tracking: DB double-click assigns to the slot the user is
    // currently editing (Image A by default, Image B when its field has
    // focus in Verify mode).  The tracking itself lives in the
    // QApplication::focusChanged handler installed below, next to the DB
    // image list.
    auto* browseImg = FaceDetectUi::makeBrowseButton(tr("Browse…"));
    connect(browseImg, &QPushButton::clicked, this,
            &FaceDetectDialog::onBrowseImage);
    m_previewLabel = new ecvClickableImageLabel;
    m_previewLabel->setFixedSize(kThumbSize, kThumbSize);
    m_previewLabel->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_previewLabel->setText(tr("A"));
    auto* labelA = new QLabel(tr("Image A:"));
    labelA->setFixedWidth(60);
    labelA->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    pathRow->addWidget(labelA);
    pathRow->addWidget(m_imagePath, 1);
    pathRow->addWidget(browseImg);
    pathRow->addWidget(
            ecvClickableImageLabel::wrapWithTapToPreviewHint(m_previewLabel));
    ioLayout->addLayout(pathRow);

    m_secondImageRow = new QWidget;
    auto* secondLayout = new QHBoxLayout(m_secondImageRow);
    secondLayout->setContentsMargins(0, 0, 0, 0);
    secondLayout->setSpacing(6);
    m_secondImagePath = new QLineEdit;
    m_secondImagePath->setPlaceholderText(tr("Second image for Verify mode"));
    connect(m_secondImagePath, &QLineEdit::textChanged, this,
            [this](const QString&) { updateSecondImagePreview(); });
    auto* browseSecond =
            FaceDetectUi::makeBrowseButton(tr("Browse…"), m_secondImageRow);
    connect(browseSecond, &QPushButton::clicked, this,
            &FaceDetectDialog::onBrowseSecondImage);
    auto* labelB = new QLabel(tr("Image B:"));
    labelB->setFixedWidth(60);
    labelB->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    secondLayout->addWidget(labelB);
    secondLayout->addWidget(m_secondImagePath, 1);
    secondLayout->addWidget(browseSecond);
    m_previewLabelB = new ecvClickableImageLabel;
    m_previewLabelB->setFixedSize(kThumbSize, kThumbSize);
    m_previewLabelB->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_previewLabelB->setText(tr("B"));
    auto* previewBWrap =
            ecvClickableImageLabel::wrapWithTapToPreviewHint(m_previewLabelB);
    previewBWrap->setVisible(false);
    secondLayout->addWidget(previewBWrap);
    m_secondImageRow->setVisible(false);
    ioLayout->addWidget(m_secondImageRow);

    m_verifyOptionsRow = new QWidget;
    auto* verifyLayout = new QGridLayout(m_verifyOptionsRow);
    verifyLayout->setContentsMargins(0, 0, 0, 0);
    FaceDetectUi::setupTwoColumnFormGrid(verifyLayout);
    m_verifyThreshold = new QDoubleSpinBox;
    m_verifyThreshold->setRange(0.01, 1.0);
    m_verifyThreshold->setSingleStep(0.01);
    m_verifyThreshold->setValue(0.65);
    FaceDetectUi::makeCompactDoubleSpin(m_verifyThreshold);
    m_verifyThreshold->setToolTip(
            tr("Maximum cosine distance for a Verify match (lower = stricter). "
               "Typical ArcFace: 0.25–0.45. This is not the detection score."));
    m_verifyMinDetectionScore = FaceDetectUi::makeMinDetectionScoreSpin(
            m_verifyOptionsRow, m_minDetectionScore->toolTip());
    m_antiSpoofCheck = new QCheckBox(tr("Anti-spoof veto (MiniFASNet)"));
    verifyLayout->addWidget(FaceDetectUi::makeFormLabel(
                                    tr("Match threshold (max cosine dist):")),
                            0, 0);
    verifyLayout->addWidget(m_verifyThreshold, 0, 1);
    verifyLayout->addWidget(
            FaceDetectUi::makeFormLabel(tr("Min detection score:")), 0, 2);
    verifyLayout->addWidget(m_verifyMinDetectionScore, 0, 3);
    verifyLayout->addWidget(m_antiSpoofCheck, 1, 0, 1, 4);
    verifyLayout->setColumnStretch(1, 1);
    verifyLayout->setColumnStretch(3, 1);
    m_verifyOptionsRow->setVisible(false);
    ioLayout->addWidget(m_verifyOptionsRow);

    m_previewRow = nullptr;

    auto* dbHeader = new QHBoxLayout;
    m_dbToggleBtn = new QToolButton;
    m_dbToggleBtn->setArrowType(Qt::RightArrow);
    m_dbToggleBtn->setCheckable(true);
    m_dbToggleBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_dbToggleBtn->setText(tr("DB Source Images (optional)"));
    connect(m_dbToggleBtn, &QToolButton::toggled, this, [this](bool checked) {
        m_dbToggleBtn->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        m_dbContentWidget->setVisible(checked);
    });
    dbHeader->addWidget(m_dbToggleBtn);
    dbHeader->addStretch();
    ioLayout->addLayout(dbHeader);

    m_dbContentWidget = new QWidget;
    m_dbContentWidget->setSizePolicy(QSizePolicy::Expanding,
                                     QSizePolicy::Preferred);
    auto* dbLayout = new QVBoxLayout(m_dbContentWidget);
    m_dbImageList = new QListWidget;
    // Layout mirrors qDeepLSD's DB Source Images: fixed single-line rows
    // (no word wrap) so row heights stay stable inside the tab viewport.
    // Word-wrapped rows inflate the list's height requirement, and since
    // the tab viewport height is measured before images arrive, the list
    // ends up clipped mid-row — the "squeezed" bug reported.
    m_dbImageList->setSizePolicy(QSizePolicy::Expanding,
                                 QSizePolicy::Expanding);
    m_dbImageList->setAlternatingRowColors(true);
    m_dbImageList->setWordWrap(false);
    m_dbImageList->setUniformItemSizes(true);
    // Long db:// entity names elide in the middle; hover shows the full
    // name via the per-item tooltip.
    m_dbImageList->setTextElideMode(Qt::ElideMiddle);
    // Height follows content (clamped below): an empty list collapses,
    // a full list scrolls instead of being squeezed.
    m_dbImageList->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    m_dbImageList->setMinimumHeight(80);
    m_dbImageList->setMaximumHeight(ecvAICoreUi::dbListMaxHeight());
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &FaceDetectDialog::onDbListActivated);

    // DB-list double-click target follows the input slot the user is
    // editing: focus in Image B (or moving straight from Image B into the
    // list) targets B; any other focus change — including entering the
    // list from anywhere else, clicking the mode combo, a button, another
    // tab or the main window — falls back to the default Image A.  This
    // replaces the old FocusIn-only tracking whose stale "B" state made
    // every subsequent DB double-click overwrite Image B.
    connect(qApp, &QApplication::focusChanged, this,
            [this](QWidget* old, QWidget* now) {
                if (now == m_secondImagePath) {
                    m_dbAssignToSecondImage = true;
                } else if (now == m_imagePath) {
                    m_dbAssignToSecondImage = false;
                } else if (now == m_dbImageList) {
                    // Direct hand-off Image B → list: keep the B target
                    // for the double-click the user is about to perform.
                    if (old != m_secondImagePath) {
                        m_dbAssignToSecondImage = false;
                    }
                } else {
                    // Focus left both inputs (mode combo, buttons, other
                    // tabs, main window…): default back to Image A.
                    m_dbAssignToSecondImage = false;
                }
                updateDbAssignHint();
            });

    // Verify-mode hint making the focus-driven assignment discoverable
    // instead of silently overwriting one of the two inputs.
    m_dbAssignHintLabel = new QLabel;
    m_dbAssignHintLabel->setStyleSheet(
            "color: #6b7280; font-size: 11px;");
    updateDbAssignHint();
    m_dbAssignHintLabel->setVisible(false);
    dbLayout->addWidget(m_dbAssignHintLabel);
    dbLayout->addWidget(m_dbImageList, 1);
    auto* refreshBtn = new QPushButton(tr("Refresh DB Images"));
    connect(refreshBtn, &QPushButton::clicked, this,
            &FaceDetectDialog::refreshDbImagesRequested);
    dbLayout->addWidget(refreshBtn);
    m_dbContentWidget->setVisible(false);
    ioLayout->addWidget(m_dbContentWidget);

    m_addAnnotatedCheck = new QCheckBox(
            tr("Add annotated ccImage to DB tree after Detect/Analyze/Dense"));
    m_addAnnotatedCheck->setChecked(true);
    ioLayout->addWidget(m_addAnnotatedCheck);
    main->addWidget(ioGroup);

    auto* btnRow = new QHBoxLayout;
    m_runBtn = new QPushButton(tr("Run"));
    m_cancelBtn = new QPushButton(tr("Cancel"));
    m_cancelBtn->setEnabled(false);
    connect(m_runBtn, &QPushButton::clicked, this, &FaceDetectDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this,
            &FaceDetectDialog::onCancel);
    btnRow->addStretch();
    btnRow->addWidget(testDataBtn);
    btnRow->addWidget(m_runBtn);
    btnRow->addWidget(m_cancelBtn);
    main->addLayout(btnRow);
}

void FaceDetectDialog::populateModelCombo(const QString& keepFilename) {
    const QString cache = modelCacheDir();
    QString selected = keepFilename;
    if (selected.isEmpty() && m_modelCombo && m_modelCombo->count() > 0) {
        selected = m_modelCombo->currentData().toString();
    }

    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (int i = 0; i < aicore_facedetect_detector_model_count(); ++i) {
        const aicore_facedetect_model_entry* m =
                aicore_facedetect_detector_model_at(i);
        if (!m) continue;
        const QFileInfo fi(cache + QLatin1Char('/') +
                           QString::fromUtf8(m->filename));
        const QString suffix =
                isValidCachedGguf(fi)
                        ? QString(" [%1] \u2713")
                                  .arg(ecvModelDownloader::formatFileSize(
                                          fi.size()))
                        : QString(" [download]");
        m_modelCombo->addItem(QCoreApplication::translate("FaceDetectModels",
                                                          m->display_name) +
                                      suffix,
                              QString::fromUtf8(m->filename));
    }
    m_modelCombo->addItem(tr("Custom..."), "CUSTOM");
    selectModelByFilename(m_modelCombo, selected);
    m_modelCombo->blockSignals(false);
    onModelComboChanged(m_modelCombo->currentIndex());
    syncRegistryModelControlsFromBatch();
}

bool FaceDetectDialog::selectModelByFilename(QComboBox* combo,
                                             const QString& filename) {
    if (!combo || filename.isEmpty()) return false;
    for (int i = 0; i < combo->count(); ++i) {
        if (combo->itemData(i).toString() == filename) {
            combo->setCurrentIndex(i);
            return true;
        }
    }
    return false;
}

void FaceDetectDialog::syncRegistryModelControlsFromBatch() {
    if (!m_registryWidget || !m_modelCombo || !m_deviceCombo || !m_threads)
        return;
    m_registryWidget->syncModelControlsFrom(m_modelCombo, m_deviceCombo,
                                            m_threads);
    m_registryWidget->setModelPath(resolveModelPath());
    if (m_liveWidget) {
        m_liveWidget->syncModelControlsFrom(m_modelCombo, m_deviceCombo,
                                            m_threads);
    }
}

void FaceDetectDialog::populateLandmarkModelCombo(const QString& keepFilename) {
    if (!m_landmarkModelCombo) return;
    QString selected = keepFilename;
    if (selected.isEmpty() && m_landmarkModelCombo->count() > 0) {
        selected = m_landmarkModelCombo->currentData().toString();
    }

    m_landmarkModelCombo->blockSignals(true);
    m_landmarkModelCombo->clear();
    const QString cache = modelCacheDir();
    QSet<QString> added;
    for (int i = 0; i < aicore_facedetect_landmark_model_count(); ++i) {
        const aicore_facedetect_model_entry* m =
                aicore_facedetect_landmark_model_at(i);
        if (!m) continue;
        const QString fn = QString::fromUtf8(m->filename);
        added.insert(fn);
        const QFileInfo fi(cache + QLatin1Char('/') + fn);
        const QString suffix =
                isValidCachedGguf(fi)
                        ? QString(" [%1] \u2713")
                                  .arg(ecvModelDownloader::formatFileSize(
                                          fi.size()))
                        : QString(" [download]");
        m_landmarkModelCombo->addItem(
                QCoreApplication::translate("FaceDetectModels",
                                            m->display_name) +
                        suffix,
                fn);
    }
    // Catalog may be empty in older AICore builds — still list GGUFs found on
    // disk.
    const QString diskDefault = defaultLandmarkModelPathOnDisk();
    if (!diskDefault.isEmpty()) {
        const QString fn = QFileInfo(diskDefault).fileName();
        if (!added.contains(fn)) {
            const QFileInfo fi(diskDefault);
            m_landmarkModelCombo->addItem(
                    tr("Dense landmarks (%1) [%2] \u2713")
                            .arg(fn)
                            .arg(ecvModelDownloader::formatFileSize(fi.size())),
                    fn);
            added.insert(fn);
        }
    }
    m_landmarkModelCombo->addItem(tr("Custom..."), QStringLiteral("CUSTOM"));
    selectModelByFilename(m_landmarkModelCombo, selected);
    m_landmarkModelCombo->blockSignals(false);
    ensureLandmarkModelPathFilled();
}

void FaceDetectDialog::syncLandmarkPathFromCombo() {
    if (!m_landmarkModelCombo || !m_customLandmarkModelPath) return;
    const QString data = m_landmarkModelCombo->currentData().toString();
    if (data.isEmpty() || data == QStringLiteral("CUSTOM")) return;
    m_customLandmarkModelPath->setText(modelCacheDir() + QLatin1Char('/') +
                                       data);
}

void FaceDetectDialog::selectDefaultLandmarkModel() {
    ensureLandmarkModelPathFilled();
}

void FaceDetectDialog::onLandmarkModelComboChanged(int index) {
    Q_UNUSED(index);
    if (!m_landmarkModelCombo || !m_customLandmarkModelPath) return;
    const QString data = m_landmarkModelCombo->currentData().toString();
    if (data == QStringLiteral("CUSTOM")) {
        const Mode mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
        if (mode == Mode::DenseLandmarks) {
            ensureLandmarkModelPathFilled();
            if (!m_customLandmarkModelPath->text().trimmed().isEmpty()) {
                return;
            }
        }
        m_customLandmarkModelPath->clear();
        m_customLandmarkModelPath->setFocus();
        return;
    }
    syncLandmarkPathFromCombo();
}

void FaceDetectDialog::refreshModelList() {
    populateModelCombo();
    populateLandmarkModelCombo();
}

FaceDetectDialog::Settings FaceDetectDialog::getSettings() const {
    Settings s;
    s.modelPath = resolveModelPath();
    s.landmarkModelPath = resolveLandmarkModelPath();
    s.inputPath = m_imagePath->text().trimmed();
    s.secondInputPath = m_secondImagePath->text().trimmed();
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    s.verifyThreshold = static_cast<float>(m_verifyThreshold->value());
    if (m_verifyMinDetectionScore && s.mode == Mode::Verify) {
        s.minDetectionScore =
                static_cast<float>(m_verifyMinDetectionScore->value());
    } else if (m_minDetectionScore) {
        s.minDetectionScore = static_cast<float>(m_minDetectionScore->value());
    }
    s.antiSpoof = m_antiSpoofCheck->isChecked();
    s.addAnnotatedImageToDb = m_addAnnotatedCheck->isChecked();
    return s;
}

QString FaceDetectDialog::resolveModelPath() const {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return m_customModelPath->text().trimmed();
    return modelCacheDir() + "/" + data;
}

QString FaceDetectDialog::defaultLandmarkModelFilename() const {
    if (aicore_facedetect_landmark_model_count() <= 0) return {};
    if (const aicore_facedetect_model_entry* m =
                aicore_facedetect_landmark_model_at(0)) {
        return QString::fromUtf8(m->filename);
    }
    return {};
}

QString FaceDetectDialog::defaultLandmarkModelPathOnDisk() const {
    const QString cache = modelCacheDir();
    const QStringList knownNames = {
            QStringLiteral("landmarks-2d106-1k3d68.gguf"),
    };
    for (const QString& fn : knownNames) {
        const QString full = cache + QLatin1Char('/') + fn;
        if (QFileInfo::exists(full)) return full;
    }
    for (int i = 0; i < aicore_facedetect_landmark_model_count(); ++i) {
        const aicore_facedetect_model_entry* m =
                aicore_facedetect_landmark_model_at(i);
        if (!m) continue;
        const QString full =
                cache + QLatin1Char('/') + QString::fromUtf8(m->filename);
        if (QFileInfo::exists(full)) return full;
    }
    QDir dir(cache);
    const QStringList ggufs =
            dir.entryList({QStringLiteral("*.gguf")}, QDir::Files, QDir::Name);
    for (const QString& fn : ggufs) {
        if (fn.contains(QStringLiteral("landmark"), Qt::CaseInsensitive)) {
            return dir.filePath(fn);
        }
    }
    return {};
}

void FaceDetectDialog::ensureLandmarkModelPathFilled() {
    if (!m_customLandmarkModelPath) return;
    const QString current = m_customLandmarkModelPath->text().trimmed();
    if (!current.isEmpty() && QFileInfo::exists(current)) return;

    const QString path = defaultLandmarkModelPathOnDisk();
    if (path.isEmpty()) return;

    m_customLandmarkModelPath->setText(path);
    if (m_landmarkModelCombo) {
        const QString fn = QFileInfo(path).fileName();
        const int idx = m_landmarkModelCombo->findData(fn);
        if (idx >= 0) {
            m_landmarkModelCombo->blockSignals(true);
            m_landmarkModelCombo->setCurrentIndex(idx);
            m_landmarkModelCombo->blockSignals(false);
        }
    }
}

QString FaceDetectDialog::resolveLandmarkModelFilename() const {
    if (m_landmarkModelCombo) {
        const QString data = m_landmarkModelCombo->currentData().toString();
        if (!data.isEmpty() && data != QStringLiteral("CUSTOM")) {
            return data;
        }
    }
    const QString def = defaultLandmarkModelFilename();
    if (!def.isEmpty()) return def;
    return QStringLiteral("landmarks-2d106-1k3d68.gguf");
}

QString FaceDetectDialog::resolveLandmarkModelPath() const {
    if (m_customLandmarkModelPath) {
        const QString typed = m_customLandmarkModelPath->text().trimmed();
        if (!typed.isEmpty()) return typed;
    }
    const QString disk = defaultLandmarkModelPathOnDisk();
    if (!disk.isEmpty()) return disk;

    const QString fn = resolveLandmarkModelFilename();
    if (!fn.isEmpty()) return modelCacheDir() + QLatin1Char('/') + fn;
    return {};
}

void FaceDetectDialog::appendLog(const QString& msg) {
    aicore_inference_log::log(msg);
}

void FaceDetectDialog::setProgress(int current, int total) {
    if (!m_progress) return;
    m_progress->setMaximum(total);
    m_progress->setValue(current);
    m_progress->setTextVisible(true);
    m_progress->setFormat(tr("%p%"));
}

void FaceDetectDialog::setRunning(bool running) {
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
    if (m_progress) {
        if (running) {
            m_progress->setMaximum(100);
            m_progress->setValue(0);
            m_progress->setTextVisible(true);
            m_progress->setFormat(tr("%p%"));
        } else {
            // Keep showing 100 % briefly, then reset to idle.
            QTimer::singleShot(1200, this, [this]() {
                if (m_progress && m_runBtn->isEnabled() &&
                    !m_cancelBtn->isEnabled()) {
                    m_progress->setValue(0);
                    m_progress->setTextVisible(false);
                    m_progress->setMaximum(100);
                }
            });
        }
    }
}

void FaceDetectDialog::setDbImages(const QList<DbImageEntry>& images) {
    m_dbImageList->clear();
    if (images.isEmpty()) {
        m_dbToggleBtn->setText(tr("DB Source Images (optional)"));
        m_dbToggleBtn->setChecked(false);
        return;
    }
    for (const auto& entry : images) {
        auto* item = new QListWidgetItem(entry.name);
        item->setToolTip(entry.name);
        if (!entry.preview.isNull()) {
            // Small 24px thumbnail — a 48px icon would squeeze the text out
            // of view, which is the "severely compressed" bug reported.
            item->setIcon(QIcon(QPixmap::fromImage(entry.preview)
                                        .scaled(24, 24, Qt::KeepAspectRatio,
                                                Qt::SmoothTransformation)));
            // Keep the FULL-RESOLUTION image for the preview component so
            // clicking the thumbnail enlarges the original, not a 24 px
            // upscaled blur.
            item->setData(kDbFullImageRole, entry.preview);
        }
        m_dbImageList->addItem(item);
    }
    m_dbToggleBtn->setText(tr("DB Source Images (%1)").arg(images.size()));
    // Auto-expand: without setChecked(true) the panel stays collapsed and
    // the loaded images are invisible — the "can't see contents" bug.
    m_dbToggleBtn->setChecked(true);
    // The tab viewport height was measured while the list was still empty;
    // re-measure now that items exist, or the list is clipped to the
    // empty-list height (the "squeezed" bug).
    QTimer::singleShot(0, this, [this]() {
        cacheTabViewportHeights();
        updateActiveTabViewportHeight();
    });
}

void FaceDetectDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    const QString name = imageNames.first();
    for (int i = 0; i < m_dbImageList->count(); ++i) {
        if (m_dbImageList->item(i)->text() == name) {
            m_dbImageList->setCurrentRow(i);
            break;
        }
    }
    m_imagePath->setText(QStringLiteral("db://") + name);
    appendLog(tr("[Info] Assigned DB image '%1'.").arg(name));
}

void FaceDetectDialog::updateImagePreview() {
    const QString path = m_imagePath->text().trimmed();
    if (path.startsWith(QStringLiteral("db://"))) {
        for (int i = 0; i < m_dbImageList->count(); ++i) {
            if (m_dbImageList->item(i)->text() == path.mid(5)) {
                // Use the stored full-resolution image so the enlarged
                // preview shows the original pixels (not the 24 px icon).
                const QVariant full =
                        m_dbImageList->item(i)->data(kDbFullImageRole);
                if (full.canConvert<QImage>()) {
                    const QImage fullImg = full.value<QImage>();
                    if (!fullImg.isNull()) {
                        m_previewLabel->setPreviewImage(fullImg, kThumbSize);
                        return;
                    }
                }
                const QIcon icon = m_dbImageList->item(i)->icon();
                if (!icon.isNull()) {
                    m_previewLabel->setPreviewPixmap(
                            icon.pixmap(kThumbSize, kThumbSize), kThumbSize);
                    return;
                }
            }
        }
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("DB"));
        return;
    }
    if (path.isEmpty() || !isSupportedImageFile(path)) {
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("Preview"));
        return;
    }
    QImage img(path);
    if (img.isNull()) {
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("?"));
        return;
    }
    m_previewLabel->setPreviewImage(img, kThumbSize);
}

void FaceDetectDialog::updateSecondImagePreview() {
    if (!m_previewLabelB) return;
    const QString path =
            m_secondImagePath ? m_secondImagePath->text().trimmed() : QString();
    if (path.isEmpty()) {
        m_previewLabelB->clearPreview();
        m_previewLabelB->setText(tr("Image B"));
        return;
    }
    QImage img;
    if (path.startsWith(QStringLiteral("db://"))) {
        // DB-tree entity: look up the stored full-resolution image
        const QString name = path.mid(5);
        for (int i = 0; i < m_dbImageList->count(); ++i) {
            if (m_dbImageList->item(i)->text() == name) {
                const QVariant full =
                        m_dbImageList->item(i)->data(kDbFullImageRole);
                if (full.canConvert<QImage>()) {
                    img = full.value<QImage>();
                }
                break;
            }
        }
    } else if (isSupportedImageFile(path)) {
        img = QImage(path);
    }
    if (img.isNull()) {
        m_previewLabelB->clearPreview();
        m_previewLabelB->setText(tr("?"));
        return;
    }
    m_previewLabelB->setPreviewImage(img, kThumbSize);
}

void FaceDetectDialog::updateDbAssignHint() {
    if (!m_dbAssignHintLabel) return;
    m_dbAssignHintLabel->setText(
            m_dbAssignToSecondImage
                    ? tr("Double-click a DB image fills Image B (focus in "
                         "Image B)")
                    : tr("Double-click a DB image fills Image A"));
}

void FaceDetectDialog::updateModeUi() {
    const Mode mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    const bool verify = mode == Mode::Verify;
    const bool dense = mode == Mode::DenseLandmarks;
    m_secondImageRow->setVisible(verify);
    m_verifyOptionsRow->setVisible(verify);
    if (m_dbAssignHintLabel) {
        // The DB double-click target hint is only meaningful in Verify
        // mode; elsewhere the DB list always fills Image A.
        m_dbAssignHintLabel->setVisible(verify);
        updateDbAssignHint();
    }
    if (m_previewLabelB) {
        if (QWidget* wrap = m_previewLabelB->parentWidget()) {
            wrap->setVisible(verify);
        } else {
            m_previewLabelB->setVisible(verify);
        }
    }
    if (m_previewLabel) {
        m_previewLabel->setText(verify ? tr("A") : tr("Preview"));
    }
    if (verify) updateSecondImagePreview();
    if (m_landmarkModelRow) {
        m_landmarkModelRow->setVisible(dense);
    }
    if (dense) {
        ensureLandmarkModelPathFilled();
    }
    if (m_batchMinScoreRow) {
        m_batchMinScoreRow->setVisible(!verify);
    } else if (m_minScoreLabel && m_minDetectionScore) {
        m_minScoreLabel->setVisible(!verify);
        m_minDetectionScore->setVisible(!verify);
    }
    m_addAnnotatedCheck->setEnabled(!verify);
}

void FaceDetectDialog::onBrowseImage() {
    QSettings settings;
    const QString lastDir =
            settings.value("qFaceDetect/lastImageFileDir", QDir::homePath())
                    .toString();
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"));
    if (path.isEmpty()) return;
    settings.setValue("qFaceDetect/lastImageFileDir",
                      QFileInfo(path).absolutePath());
    m_imagePath->setText(path);
    m_batchImagePathUserChosen = true;
}

void FaceDetectDialog::onBrowseSecondImage() {
    QSettings settings;
    const QString lastDir =
            settings.value("qFaceDetect/lastImageFileDir", QDir::homePath())
                    .toString();
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select second image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"));
    if (path.isEmpty()) return;
    settings.setValue("qFaceDetect/lastImageFileDir",
                      QFileInfo(path).absolutePath());
    m_secondImagePath->setText(path);
}

void FaceDetectDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir =
            settings.value("qFaceDetect/lastModelDir", modelCacheDir())
                    .toString();
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select GGUF"), lastDir, tr("GGUF (*.gguf)"));
    if (path.isEmpty()) return;
    settings.setValue("qFaceDetect/lastModelDir",
                      QFileInfo(path).absolutePath());
    m_customModelPath->setText(path);
    onModelComboChanged(m_modelCombo->currentIndex());
}

void FaceDetectDialog::onBrowseCustomLandmarkModel() {
    QSettings settings;
    const QString lastDir =
            settings.value("qFaceDetect/lastModelDir", modelCacheDir())
                    .toString();
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select landmark GGUF"), lastDir, tr("GGUF (*.gguf)"));
    if (path.isEmpty()) return;
    settings.setValue("qFaceDetect/lastModelDir",
                      QFileInfo(path).absolutePath());
    if (m_customLandmarkModelPath) {
        m_customLandmarkModelPath->setText(path);
    }
}

void FaceDetectDialog::syncRegistryConfig() {
    if (!m_registryWidget) return;
    if (m_threads) m_registryWidget->setThreads(m_threads->value());
}

void FaceDetectDialog::syncLiveConfig() {
    if (!m_liveWidget) return;
    syncRegistryConfig();

    FaceLiveDetectWidget::Config cfg = m_liveWidget->config();
    cfg.modelPath = resolveModelPath();
    cfg.device = m_deviceCombo ? m_deviceCombo->currentData().toString()
                               : QStringLiteral("auto");
    cfg.threads = m_threads ? m_threads->value() : 0;
    if (m_liveWidget && m_modelCombo) {
        m_liveWidget->syncModelControlsFrom(m_modelCombo, m_deviceCombo,
                                            m_threads);
        cfg.modelPath = m_liveWidget->resolveModelPath();
        cfg.device = m_liveWidget->deviceId();
        cfg.threads = m_liveWidget->threadCount();
    }
    if (m_registryWidget) {
        cfg.registry = m_registryWidget->store();
    }
    if (m_minDetectionScore) {
        cfg.minDetectionScore =
                static_cast<float>(m_minDetectionScore->value());
    }

    m_liveWidget->setConfig(cfg);
    cfg = m_liveWidget->config();

    if (m_registryWidget) {
        m_registryWidget->setModelPath(cfg.modelPath);
        m_registryWidget->setDevice(cfg.device);
    }
}

void FaceDetectDialog::validateLiveRecognizeModeFromSettings() {
    if (!m_liveWidget) return;
    tryAutoDiscoverRegistryDb();
    syncLiveConfig();
    if (m_liveWidget->config().streamMode ==
        FaceLiveDetectWidget::StreamMode::Recognize) {
        onLiveStreamModeChanged(
                static_cast<int>(FaceLiveDetectWidget::StreamMode::Recognize),
                false);
    }
}

void FaceDetectDialog::onLiveStreamModeChanged(int streamMode,
                                               bool showUserPrompt) {
    if (static_cast<FaceLiveDetectWidget::StreamMode>(streamMode) !=
        FaceLiveDetectWidget::StreamMode::Recognize) {
        return;
    }

    tryAutoDiscoverRegistryDb();
    syncLiveConfig();
    const FaceRegistryStore* store =
            m_registryWidget ? m_registryWidget->store() : nullptr;
    const bool valid = store && store->isOpen() && !store->entries().empty();
    if (valid) return;

    appendLog(
            tr("[Live] Registry DB invalid or empty — Recognize mode requires "
               "registered identities. Open Registry / Auth tab and register "
               "faces (Use test data or manual)."));

    if (showUserPrompt) {
        QMessageBox::information(
                this, tr("Recognize unavailable"),
                tr("Recognize mode requires a face registry database with at "
                   "least one enrolled identity (this is separate from the "
                   "detector GGUF models in the model cache).\n\n"
                   "Open the Registry / Auth tab, click Use test data or "
                   "register faces manually, then try Recognize again.\n\n"
                   "Mode will switch back to Detect faces only."));
    }

    if (m_liveWidget) {
        m_liveWidget->setStreamMode(FaceLiveDetectWidget::StreamMode::Detect);
    }
}

FaceDetectDialog::~FaceDetectDialog() {
    saveBatchSettings();
    if (m_registryWidget) m_registryWidget->saveSettings();
    if (m_liveWidget) m_liveWidget->saveSettings();
    if (m_testDataWorker && m_testDataWorker->isRunning()) {
        m_testDataWorker->wait(5000);
    }
}

void FaceDetectDialog::onLiveStart() {
    if (!m_liveWidget) return;
    // Live tab only supports Detect / Recognize — never DenseLandmarks — so
    // only verify the face detector model, not the landmark model.
    if (!ensureDetectorAvailable()) return;
    syncLiveConfig();

    const FaceLiveDetectWidget::Config liveCfg = m_liveWidget->config();
    if (liveCfg.streamMode == FaceLiveDetectWidget::StreamMode::Recognize) {
        const FaceRegistryStore* store =
                m_registryWidget ? m_registryWidget->store() : nullptr;
        if (!store || !store->isOpen() || store->entries().empty()) {
            QMessageBox::information(
                    this, tr("Recognize unavailable"),
                    tr("Recognize mode is unavailable: the face registry "
                       "database is "
                       "missing, could not be opened, or has no enrolled "
                       "identities.\n\n"
                       "Open the Registry / Auth tab, click Use test data or "
                       "register "
                       "faces manually, then try Recognize again."));
            if (m_tabWidget && m_registryWidget) {
                m_tabWidget->setCurrentIndex(1);
            }
            return;
        }
    }

    if (m_liveWidget->inputSource() ==
        FaceLiveDetectWidget::InputSource::VideoFile) {
        const QString path = m_liveWidget->videoFilePath();
        if (path.isEmpty() || !QFile::exists(path)) {
            appendLog(tr("[Live] Select a valid video file first."));
            return;
        }
        if (!m_liveWidget->startVideoFile(path)) {
            appendLog(tr("[Live] Failed to start video."));
        }
        return;
    }
    // Fallback: if the user set a video path via test data but the source
    // combo hasn't caught up (async test-data download / race), use it.
    {
        const QString path = m_liveWidget->videoFilePath();
        if (!path.isEmpty() && QFile::exists(path)) {
            m_liveWidget->selectVideoFileSource();
            if (!m_liveWidget->startVideoFile(path)) {
                appendLog(tr("[Live] Failed to start video."));
            }
            return;
        }
    }
    const int camIdx = m_liveWidget->selectedCameraIndex();
    if (camIdx < 0) {
        appendLog(tr("[Live] No camera available."));
        return;
    }
    if (!m_liveWidget->startCamera(camIdx)) {
        appendLog(tr("[Live] Failed to start camera %1.").arg(camIdx));
    }
}

void FaceDetectDialog::onLiveStop() {
    if (m_liveWidget) m_liveWidget->stopStream();
}

void FaceDetectDialog::onLiveRestart() {
    if (m_liveWidget) m_liveWidget->restartVideoFile();
}

void FaceDetectDialog::onLiveCapture(const FaceDetectRunResult& result) {
    emit liveCaptureReady(result);
}

void FaceDetectDialog::onModelComboChanged(int index) {
    const QString data = m_modelCombo->itemData(index).toString();
    m_customModelRow->setVisible(data == "CUSTOM");

    QString hint;
    if (const aicore_facedetect_model_entry* m =
                aicore_facedetect_model_by_filename(
                        data.toUtf8().constData())) {
        hint = QCoreApplication::translate("FaceDetectModels", m->quant_note) +
               QStringLiteral(" — ") +
               QCoreApplication::translate("FaceDetectModels", m->license_note);
    }
    if (data == "CUSTOM") {
        hint = tr("Custom GGUF from disk (see MODEL_CARD.md).");
    }
    m_variantHintLabel->setText(hint);
    syncRegistryModelControlsFromBatch();
    syncLiveConfig();
}

void FaceDetectDialog::onAuthResultImageReady(const QImage& annotated,
                                              const QString& summary) {
    if (annotated.isNull()) return;
    appendLog(tr(
            "[Registry] Exporting authentication visualization to DB tree."));
    emit authVisualizationReady(annotated, summary);
}

void FaceDetectDialog::onModeChanged(int) {
    updateModeUi();
    // Mode changes toggle visibility of extra rows (Verify: second image +
    // options; DenseLandmarks: landmark model).  Re-measure content so the
    // tab viewport expands instead of compressing the newly visible widgets.
    //
    // Defer measurement via singleShot(0) so the layout engine fully
    // processes visibility propagation across all platforms/styles before
    // we read minimumSizeHint().  This mirrors the existing deferred
    // measurement used for tab switching (see currentChanged handler).
    if (m_tabWidget && m_batchTab) {
        // Invalidate the cached height so updateActiveTabViewportHeight
        // falls back to a fresh sizeHint() if called before the deferred
        // re-measurement completes.
        m_tabViewportHeights.remove(m_batchTab);
        QTimer::singleShot(0, this, [this]() {
            if (!m_batchTab || !m_tabWidget) return;
            const int freshHeight = m_batchTab->sizeHint().height();
            m_tabViewportHeights.insert(m_batchTab, freshHeight);
            updateActiveTabViewportHeight();
        });
    }
}

bool FaceDetectDialog::ensureModelAvailable() {
    const QString path = resolveModelPath();
    if (path.isEmpty()) return false;
    if (!QFile::exists(path) || !isValidCachedGguf(QFileInfo(path))) {
        const QString data = m_modelCombo->currentData().toString();
        if (data == "CUSTOM") {
            QMessageBox::warning(
                    this, tr("Model missing"),
                    tr("Custom detector model not found:\n%1").arg(path));
            return false;
        }
        if (const aicore_facedetect_model_entry* m =
                    aicore_facedetect_model_by_filename(
                            data.toUtf8().constData())) {
            startDownload(m);
            return false;
        }
        return false;
    }

    const Mode mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    if (mode != Mode::DenseLandmarks) return true;

    const QString landmarkPath = resolveLandmarkModelPath();
    if (!landmarkPath.isEmpty() && QFile::exists(landmarkPath) &&
        isValidCachedGguf(QFileInfo(landmarkPath))) {
        return true;
    }

    // Landmark model missing (or path empty) — resolve filename from combo /
    // catalog default and trigger auto-download.
    const QString lmFilename = resolveLandmarkModelFilename();
    const aicore_facedetect_model_entry* m =
            aicore_facedetect_model_by_filename(
                    lmFilename.toUtf8().constData());
    if (!m) {
        appendLog(tr("[Error] No landmark model catalog entry for %1.")
                          .arg(lmFilename));
        return false;
    }

    QMessageBox::information(
            this, tr("Landmark model required"),
            tr("Dense Landmarks runs a two-stage pipeline: the detector finds "
               "faces, then a separate landmark GGUF refines 106 2D + 68 3D "
               "points.\n\n"
               "Downloading %1 now\u2026")
                    .arg(lmFilename));
    startDownload(m);
    return false;
}

bool FaceDetectDialog::ensureDetectorAvailable() {
    const QString path = resolveModelPath();
    if (path.isEmpty()) return false;
    if (!QFile::exists(path) || !isValidCachedGguf(QFileInfo(path))) {
        const QString data = m_modelCombo->currentData().toString();
        if (data == "CUSTOM") {
            QMessageBox::warning(
                    this, tr("Model missing"),
                    tr("Custom detector model not found:\n%1").arg(path));
            return false;
        }
        if (const aicore_facedetect_model_entry* m =
                    aicore_facedetect_model_by_filename(
                            data.toUtf8().constData())) {
            startDownload(m);
            return false;
        }
        return false;
    }
    return true;
}

void FaceDetectDialog::startDownload(
        const aicore_facedetect_model_entry* model) {
    if (!model || m_downloadInProgress) return;
    QDir().mkpath(modelCacheDir());
    const QString dest = modelCacheDir() + QLatin1Char('/') +
                         QString::fromUtf8(model->filename);
    m_downloadInProgress = true;
    m_autoRunAfterDownload = true;
    m_downloadLabel->setVisible(true);
    m_downloadLabel->setText(
            tr("Downloading %1 ...").arg(QString::fromUtf8(model->filename)));
    m_progress->setValue(0);
    appendLog(tr("[FaceDetect] Downloading %1 ...")
                      .arg(QString::fromUtf8(model->filename)));

    ecvModelDownloader::Request req;
    req.url = QString::fromUtf8(model->download_url);
    req.destPath = dest;
    m_downloader->download(req);
}

void FaceDetectDialog::cancelDownload() {
    if (m_downloader) m_downloader->cancel();
    m_downloadInProgress = false;
    m_autoRunAfterDownload = false;
    m_downloadLabel->setVisible(false);
}

void FaceDetectDialog::onRun() {
    if (m_imagePath->text().trimmed().isEmpty()) {
        appendLog(tr("[Error] Input image required."));
        return;
    }
    if (!ensureModelAvailable()) return;

    const Mode mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    if (mode == Mode::DenseLandmarks) {
        if (m_customLandmarkModelPath &&
            m_customLandmarkModelPath->text().trimmed().isEmpty()) {
            selectDefaultLandmarkModel();
        }
        const QString landmarkPath = resolveLandmarkModelPath();
        if (landmarkPath.isEmpty() || !QFileInfo::exists(landmarkPath)) {
            appendLog(
                    tr("[Error] Landmark model required for Dense Landmarks "
                       "mode."));
            QMessageBox::warning(
                    this, tr("Landmark model required"),
                    tr("Select or download a landmark GGUF model.\n\n"
                       "Expected default:\n%1")
                            .arg(modelCacheDir() +
                                 QStringLiteral(
                                         "/landmarks-2d106-1k3d68.gguf")));
            return;
        }
    }

    emit runRequested(getSettings());
}

void FaceDetectDialog::onCancel() {
    cancelDownload();
    emit cancelRequested();
}

void FaceDetectDialog::closeEvent(QCloseEvent* event) {
    onCancel();
    onLiveStop();
    saveBatchSettings();
    if (m_liveWidget) m_liveWidget->saveSettings();
    if (m_registryWidget) m_registryWidget->saveSettings();
    QDialog::closeEvent(event);
}

void FaceDetectDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    if (m_baseChrome >= 0) {
        return;  // measured once; DPI changes re-measure in changeEvent
    }
    // At show time the layout is already settled (Qt applies sizeHint
    // geometry before sending Show), so the initial dialog height minus
    // the tab height is pure chrome, never polluted by user resizing.
    // Measuring and resizing synchronously here applies the geometry
    // BEFORE the window is mapped — deferring it into the event loop
    // loses the race against the platform's async window map and the
    // sizeHint-sized initial frame is shown first.
    if (!m_tabWidget) return;
    m_baseChrome = std::max(0, height() - m_tabWidget->height());
    // First display: snap the dialog to exactly chrome + active tab
    // so the initial size never carries layout slack from sizeHint
    // inflation.
    updateActiveTabViewportHeight();
    if (m_activeTabHeight >= 0) {
        resize(width(),
               std::max(dpiScaled(420), m_baseChrome + m_activeTabHeight));
    }
}

void FaceDetectDialog::loadBatchSettings() {
    QSettings settings;
    QString imagePath =
            settings.value(FaceDetectTestData::manualBatchImageSettingsKey())
                    .toString();
    m_batchImagePathUserChosen = !imagePath.isEmpty();
    if (imagePath.isEmpty()) {
        const QString legacy =
                settings.value(QStringLiteral("qFaceDetect/batchImagePath"))
                        .toString();
        if (!legacy.isEmpty() &&
            !FaceDetectTestData::isFriendsBundlePath(legacy)) {
            imagePath = legacy;
            m_batchImagePathUserChosen = true;
        }
    }
    const int mode = settings.value(QStringLiteral("qFaceDetect/mode"),
                                    static_cast<int>(Mode::Detect))
                             .toInt();
    const QString modelFn =
            settings.value(QStringLiteral("qFaceDetect/model")).toString();
    const QString device =
            settings.value(QStringLiteral("qFaceDetect/device")).toString();
    const int threads =
            settings.value(QStringLiteral("qFaceDetect/threads"), 0).toInt();
    const double minScore =
            settings.value(QStringLiteral("qFaceDetect/minDetectionScore"), 0.5)
                    .toDouble();
    const double verifyThresh =
            settings.value(QStringLiteral("qFaceDetect/matchThreshold"),
                           settings.value(
                                   QStringLiteral(
                                           "qFaceDetect/verifyThreshold"),
                                   0.65))
                    .toDouble();
    const bool antiSpoof =
            settings.value(QStringLiteral("qFaceDetect/antiSpoof"), false)
                    .toBool();
    const bool addAnnotated =
            settings.value(QStringLiteral("qFaceDetect/addAnnotated"), true)
                    .toBool();

    if (m_modeCombo) {
        const int idx = m_modeCombo->findData(mode);
        if (idx >= 0) m_modeCombo->setCurrentIndex(idx);
    }
    if (m_modelCombo && !modelFn.isEmpty()) {
        const int idx = m_modelCombo->findData(modelFn);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
    if (m_deviceCombo && !device.isEmpty()) {
        const int idx = m_deviceCombo->findData(device);
        if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    }
    if (m_threads) m_threads->setValue(threads);
    if (m_minDetectionScore) m_minDetectionScore->setValue(minScore);
    if (m_verifyMinDetectionScore)
        m_verifyMinDetectionScore->setValue(minScore);
    if (m_verifyThreshold) m_verifyThreshold->setValue(verifyThresh);
    if (m_antiSpoofCheck) m_antiSpoofCheck->setChecked(antiSpoof);
    if (m_addAnnotatedCheck) m_addAnnotatedCheck->setChecked(addAnnotated);
    if (m_imagePath && !imagePath.isEmpty()) {
        m_imagePath->setText(imagePath);
        m_batchImagePathUserChosen = true;
    }
    const QString landmarkFn =
            settings.value(QStringLiteral("qFaceDetect/landmarkModel"))
                    .toString();
    const QString landmarkPath =
            settings.value(QStringLiteral("qFaceDetect/landmarkModelPath"))
                    .toString();
    if (m_landmarkModelCombo && !landmarkFn.isEmpty()) {
        const int idx = m_landmarkModelCombo->findData(landmarkFn);
        if (idx >= 0) {
            m_landmarkModelCombo->blockSignals(true);
            m_landmarkModelCombo->setCurrentIndex(idx);
            m_landmarkModelCombo->blockSignals(false);
        }
    }
    if (m_customLandmarkModelPath && !landmarkPath.isEmpty() &&
        QFileInfo::exists(landmarkPath)) {
        m_customLandmarkModelPath->setText(landmarkPath);
    }
    if (m_registryWidget) {
        m_registryWidget->setAuthThreshold(static_cast<float>(verifyThresh));
    }
    if (m_liveWidget) {
        m_liveWidget->setMatchThreshold(static_cast<float>(verifyThresh));
    }
    if (m_minDetectionScore) {
        applyMinDetectionScoreToAllTabs(m_minDetectionScore->value());
    }
    updateModeUi();
    updateImagePreview();
}

void FaceDetectDialog::saveBatchSettings() const {
    QSettings settings;
    if (m_imagePath) {
        const QString path = m_imagePath->text().trimmed();
        if (m_batchImagePathUserChosen && !path.isEmpty()) {
            settings.setValue(FaceDetectTestData::manualBatchImageSettingsKey(),
                              path);
        } else {
            settings.remove(FaceDetectTestData::manualBatchImageSettingsKey());
        }
        settings.remove(QStringLiteral("qFaceDetect/batchImagePath"));
    }
    if (m_modeCombo) {
        settings.setValue(QStringLiteral("qFaceDetect/mode"),
                          m_modeCombo->currentData());
    }
    if (m_modelCombo) {
        settings.setValue(QStringLiteral("qFaceDetect/model"),
                          m_modelCombo->currentData());
    }
    if (m_deviceCombo) {
        settings.setValue(QStringLiteral("qFaceDetect/device"),
                          m_deviceCombo->currentData());
    }
    if (m_threads) {
        settings.setValue(QStringLiteral("qFaceDetect/threads"),
                          m_threads->value());
    }
    if (m_minDetectionScore) {
        settings.setValue(QStringLiteral("qFaceDetect/minDetectionScore"),
                          m_minDetectionScore->value());
    }
    if (m_verifyThreshold) {
        settings.setValue(QStringLiteral("qFaceDetect/verifyThreshold"),
                          m_verifyThreshold->value());
        settings.setValue(QStringLiteral("qFaceDetect/matchThreshold"),
                          m_verifyThreshold->value());
    }
    if (m_landmarkModelCombo) {
        settings.setValue(QStringLiteral("qFaceDetect/landmarkModel"),
                          m_landmarkModelCombo->currentData());
    }
    if (m_customLandmarkModelPath) {
        const QString lmPath = m_customLandmarkModelPath->text().trimmed();
        if (!lmPath.isEmpty() && QFileInfo::exists(lmPath)) {
            settings.setValue(QStringLiteral("qFaceDetect/landmarkModelPath"),
                              lmPath);
        }
    }
    if (m_antiSpoofCheck) {
        settings.setValue(QStringLiteral("qFaceDetect/antiSpoof"),
                          m_antiSpoofCheck->isChecked());
    }
    if (m_addAnnotatedCheck) {
        settings.setValue(QStringLiteral("qFaceDetect/addAnnotated"),
                          m_addAnnotatedCheck->isChecked());
    }
}

bool FaceDetectDialog::tryResolveFriendsTestBundle(
        FaceDetectFriendsBundle* out) {
    return out != nullptr && FaceDetectTestData::resolveBundle(out);
}

void FaceDetectDialog::applyFriendsTestDataPaths(
        const FaceDetectFriendsBundle& bundle,
        bool fillLiveVideo,
        bool fillBatchImage) {
    if (fillLiveVideo && m_liveWidget && !bundle.videoPath.isEmpty()) {
        m_liveWidget->selectVideoFileSource();
        m_liveWidget->setVideoFilePath(bundle.videoPath, false);
        appendLog(tr("[Test data] Live video: %1").arg(bundle.videoPath));
    }
    if (!fillBatchImage || !m_imagePath) return;

    m_batchImagePathUserChosen = false;
    const Mode mode =
            m_modeCombo ? static_cast<Mode>(m_modeCombo->currentData().toInt())
                        : Mode::Detect;

    if (mode == Mode::Verify) {
        QString imageA;
        QString imageB;
        if (FaceDetectTestData::verifyTestImagePair(bundle, &imageA, &imageB)) {
            m_imagePath->setText(imageA);
            if (m_secondImagePath) m_secondImagePath->setText(imageB);
            updateImagePreview();
            updateSecondImagePreview();
            appendLog(tr("[Test data] Verify Image A: %1").arg(imageA));
            appendLog(tr("[Test data] Verify Image B: %1").arg(imageB));
        } else {
            appendLog(
                    tr("[Test data] Could not resolve verify portrait pair."));
        }
        return;
    }

    const QString groupPhoto = FaceDetectTestData::groupPhotoPath(bundle);
    if (!groupPhoto.isEmpty()) {
        m_imagePath->setText(groupPhoto);
        updateImagePreview();
        appendLog(tr("[Test data] Batch image (group photo): %1")
                          .arg(groupPhoto));
    } else if (!bundle.batchImage.isEmpty()) {
        m_imagePath->setText(bundle.batchImage);
        updateImagePreview();
        appendLog(tr("[Test data] Batch image: %1").arg(bundle.batchImage));
    }
}

void FaceDetectDialog::applyFriendsTestBundle(
        const FaceDetectFriendsBundle& bundle,
        bool fillRegistry,
        bool fillLiveVideo,
        bool fillBatchImage) {
    if (!fillRegistry) {
        applyFriendsTestDataPaths(bundle, fillLiveVideo, fillBatchImage);
        return;
    }

    QString modelFilename;
    if (m_modelCombo) {
        modelFilename = m_modelCombo->currentData().toString();
    }
    if (modelFilename.isEmpty() || modelFilename == QStringLiteral("CUSTOM")) {
        modelFilename = QStringLiteral("buffalo_l.gguf");
    }

    if (fillRegistry && m_registryWidget) {
        QString registryPath = bundle.registryDbPath;
        if (!bundle.extractRoot.isEmpty()) {
            registryPath = FaceDetectTestData::registryPathForModel(
                    bundle.extractRoot, modelFilename);
        }
        if (!registryPath.isEmpty()) {
            m_registryWidget->setRegistryPath(registryPath, false);
        }

        const bool needsGallery = !bundle.galleryEntries.isEmpty() ||
                                  !bundle.extractRoot.isEmpty();
        if (needsGallery) {
            if (!ensureModelAvailable()) {
                appendLog(
                        tr("[Test data] Face model required before gallery "
                           "registration."));
            } else {
                startTestDataPostProcess(bundle, true, fillLiveVideo,
                                         fillBatchImage, false, QString(),
                                         m_testClearExistingEntries);
            }
        } else {
            appendLog(
                    tr("[Test data] Registry path ready — no gallery to "
                       "register."));
        }
    }
}

void FaceDetectDialog::setTestDataBusy(bool busy) {
    if (m_runBtn) m_runBtn->setEnabled(!busy);
    if (m_liveStartBtn) m_liveStartBtn->setEnabled(!busy);
    if (m_registryWidget) m_registryWidget->setEnabled(!busy);
    if (m_liveWidget) m_liveWidget->setEnabled(!busy);
}

void FaceDetectDialog::updateTestDataProgress(int current,
                                              int total,
                                              const QString& label) {
    if (m_downloadLabel) {
        m_downloadLabel->setVisible(true);
        m_downloadLabel->setText(label);
    }
    if (m_progress) {
        m_progress->setTextVisible(true);
        m_progress->setFormat(tr("%p%"));
        m_progress->setMaximum(kTestDataOverallMax);
        if (total > 0) {
            const int span = kTestDataOverallMax - m_testDataPostProgressBase;
            const int value = m_testDataPostProgressBase +
                              static_cast<int>(static_cast<qint64>(current) *
                                               span / total);
            m_progress->setValue(std::min(value, kTestDataOverallMax));
            m_progress->setFormat(tr("%p%"));
        } else {
            m_progress->setValue(m_testDataPostProgressBase);
            m_progress->setFormat(tr("%p%"));
        }
    }
}

void FaceDetectDialog::startTestDataPostProcess(
        const FaceDetectFriendsBundle& bundle,
        bool fillRegistry,
        bool fillLiveVideo,
        bool fillBatchImage,
        bool extractZipFirst,
        const QString& zipPath,
        bool clearExistingEntries) {
    if (m_testDataProcessing) {
        appendLog(tr("[Test data] Setup already in progress."));
        return;
    }
    if (m_testDataWorker && m_testDataWorker->isRunning()) {
        appendLog(tr("[Test data] Wait for the current setup to finish."));
        return;
    }

    FaceDetectTestDataWorker::Job job;
    job.bundle = bundle;
    job.extractZipFirst = extractZipFirst;
    job.zipPath = zipPath;
    job.extractParentDir = ecvTestDataRepository::extractDir();
    job.registerGallery = fillRegistry;
    job.runVerify = fillRegistry && fillBatchImage;
    job.clearExistingEntries = clearExistingEntries;

    QString modelFilename;
    if (m_modelCombo) {
        modelFilename = m_modelCombo->currentData().toString();
    }
    if (modelFilename.isEmpty() || modelFilename == QStringLiteral("CUSTOM")) {
        modelFilename = QStringLiteral("buffalo_l.gguf");
    }

    if (fillRegistry) {
        if (!ensureModelAvailable()) {
            appendLog(
                    tr("[Test data] Face model required before gallery "
                       "registration."));
            return;
        }
        job.modelPath = resolveModelPath();
        job.device = m_deviceCombo ? m_deviceCombo->currentData().toString()
                                   : QStringLiteral("auto");
        job.threads = m_threads ? m_threads->value() : 0;

        if (m_registryWidget) {
            job.registryPath = m_registryWidget->registryPath();
            job.minDetectionScore = m_registryWidget->minDetectionScore();
            job.authThreshold = m_registryWidget->authThreshold();
        }

        if (job.registryPath.isEmpty() && !bundle.extractRoot.isEmpty()) {
            job.registryPath = FaceDetectTestData::registryPathForModel(
                    bundle.extractRoot, modelFilename);
        }
    }

    m_testDataProcessing = true;
    m_testPostFillRegistry = fillRegistry;
    m_testPostFillLiveVideo = fillLiveVideo;
    m_testPostFillBatchImage = fillBatchImage;
    setTestDataBusy(true);
    if (fillRegistry && m_registryWidget) {
        m_registryWidget->releaseStoreConnection();
    }
    if (extractZipFirst && m_testDataPostProgressBase == 0 && m_progress &&
        m_progress->value() >= kTestDataDownloadShare) {
        m_testDataPostProgressBase = kTestDataDownloadShare;
    } else if (!extractZipFirst) {
        m_testDataPostProgressBase = 0;
    }
    updateTestDataProgress(0, 1, tr("Preparing FriendsFaces test data…"));

    m_testDataWorker->setJob(std::move(job));
    m_testDataWorker->start();
}

void FaceDetectDialog::startFriendsTestDataDownload(bool fillRegistry,
                                                    bool fillLiveVideo,
                                                    bool fillBatchImage) {
    if (m_testDataDownloadInProgress) {
        appendLog(tr("[Test data] Download already in progress."));
        return;
    }
    if (m_testDataProcessing) {
        appendLog(tr("[Test data] Wait for setup to finish first."));
        return;
    }
    if (m_downloadInProgress) {
        appendLog(tr("[Test data] Wait for model download to finish first."));
        return;
    }
    m_testFillRegistry = fillRegistry;
    m_testFillLiveVideo = fillLiveVideo;
    m_testFillBatchImage = fillBatchImage;
    QDir().mkpath(ecvTestDataRepository::downloadDir());
    const QString zipPath = ecvTestDataRepository::zipPath(kFriends);
    if (QFileInfo::exists(zipPath) &&
        !ecvTestDataRepository::verifyZipIntegrity(
                zipPath,
                ecvTestDataRepository::getDatasetInfo(kFriends).expectedMd5,
                ecvTestDataRepository::getDatasetInfo(kFriends).expectedSize)) {
        QFile::remove(zipPath);
        appendLog(
                tr("[Test data] Removed cached friends_faces.zip (MD5 "
                   "mismatch)."));
    }
    ecvModelDownloader::removeInvalidCacheFile(
            zipPath, 30 * 1024 * 1024,
            false /* GGUF magic not relevant for zip */);
    m_testDataDownloadInProgress = true;
    setTestDataBusy(true);
    if (m_downloadLabel) {
        m_downloadLabel->setVisible(true);
        m_downloadLabel->setText(tr("Downloading FriendsFaces test data …"));
    }
    if (m_progress) {
        m_progress->setMaximum(kTestDataOverallMax);
        m_progress->setTextVisible(true);
        m_progress->setFormat(tr("%p%"));
        m_progress->setValue(0);
    }
    appendLog(tr("[Test data] Downloading friends_faces.zip ..."));

    ecvModelDownloader::Request req;
    req.url = ecvTestDataRepository::getDatasetInfo(kFriends).downloadUrl;
    req.destPath = ecvTestDataRepository::zipPath(kFriends);
    req.minBytes = 30 * 1024 * 1024;
    req.requireGgufMagic = false; /* zip archive, not a GGUF */
    m_testDataDownloader->download(req);
}

void FaceDetectDialog::ensureFriendsTestData(bool fillRegistry,
                                             bool fillLiveVideo,
                                             bool fillBatchImage) {
    if (m_testDataDownloadInProgress || m_testDataProcessing) {
        appendLog(tr("[Test data] Already in progress."));
        return;
    }

    m_testClearExistingEntries = true;
    if (fillRegistry) {
        const TestDataRegistryMode mode = confirmTestDataRegistryMode();
        if (mode == TestDataRegistryMode::Cancel) {
            return;
        }
        m_testClearExistingEntries = (mode == TestDataRegistryMode::Overwrite);
    }

    FaceDetectFriendsBundle bundle;
    if (tryResolveFriendsTestBundle(&bundle)) {
        if (fillRegistry) {
            applyFriendsTestBundle(bundle, true, fillLiveVideo, fillBatchImage);
        } else {
            applyFriendsTestDataPaths(bundle, fillLiveVideo, fillBatchImage);
        }
        return;
    }
    if (ecvTestDataRepository::instance().isDatasetAvailable(kFriends)) {
        setTestDataBusy(true);
        if (m_downloadLabel) {
            m_downloadLabel->setVisible(true);
            m_downloadLabel->setText(
                    tr("Extracting cached FriendsFaces archive…"));
        }
        if (m_progress) {
            m_progress->setMaximum(0);
            m_progress->setTextVisible(true);
            m_progress->setFormat(tr("Loading…"));
        }
        startTestDataPostProcess(bundle, fillRegistry, fillLiveVideo,
                                 fillBatchImage, true,
                                 ecvTestDataRepository::zipPath(kFriends),
                                 m_testClearExistingEntries);
        return;
    }
    startFriendsTestDataDownload(fillRegistry, fillLiveVideo, fillBatchImage);
}

void FaceDetectDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    m_batchImagePathUserChosen = true;
    const QString dbPath = QStringLiteral("db://") + item->text();
    const Mode mode =
            m_modeCombo ? static_cast<Mode>(m_modeCombo->currentData().toInt())
                        : Mode::Detect;
    if (mode == Mode::Verify && m_secondImagePath &&
        m_dbAssignToSecondImage) {
        // Verify mode with the Image B field focused: assign there.
        m_secondImagePath->setText(dbPath);
    } else {
        m_imagePath->setText(dbPath);
    }
}

void FaceDetectDialog::tryAutoDiscoverRegistryDb() {
    if (!m_registryWidget) return;

    const FaceRegistryStore* store = m_registryWidget->store();
    if (store && store->isOpen() && !store->entries().empty()) {
        if (m_liveWidget) {
            m_liveWidget->setRegistryPath(
                    m_registryWidget->registryPath(),
                    m_registryWidget->isRegistryPathUserChosen());
        }
        return;
    }

    QString modelFilename;
    if (m_modelCombo) {
        modelFilename = m_modelCombo->currentData().toString();
    }
    if (modelFilename.isEmpty() || modelFilename == QStringLiteral("CUSTOM")) {
        modelFilename = QStringLiteral("buffalo_l.gguf");
    }

    const QString discovered =
            FaceDetectTestData::discoverRegistryDbPath(modelFilename);
    if (discovered.isEmpty()) return;

    m_registryWidget->setRegistryPath(discovered, false);
    if (m_liveWidget) {
        m_liveWidget->setRegistryPath(discovered, false);
        m_liveWidget->setRegistryStore(m_registryWidget->store());
    }
    appendLog(tr("[Registry] Auto-loaded %1 enrolled identit(ies) from %2")
                      .arg(m_registryWidget->store()->entries().size())
                      .arg(discovered));
}

FaceDetectDialog::TestDataRegistryMode
FaceDetectDialog::confirmTestDataRegistryMode() const {
    if (!m_registryWidget) return TestDataRegistryMode::Overwrite;
    const FaceRegistryStore* store = m_registryWidget->store();
    if (!store || !store->isOpen() || store->entries().empty()) {
        return TestDataRegistryMode::Overwrite;
    }

    QMessageBox box(QMessageBox::Question, tr("Registry not empty"),
                    tr("The face registry already contains %1 enrolled "
                       "identit(ies).\n\n"
                       "Append — register only new names from test data.\n"
                       "Overwrite all — clear the database and register the "
                       "FriendsFaces gallery.\n"
                       "Cancel — abort.")
                            .arg(store->entries().size()),
                    QMessageBox::NoButton, const_cast<FaceDetectDialog*>(this));
    auto* appendBtn = box.addButton(tr("Append"), QMessageBox::AcceptRole);
    auto* overwriteBtn =
            box.addButton(tr("Overwrite all"), QMessageBox::DestructiveRole);
    box.addButton(QMessageBox::Cancel);
    box.setDefaultButton(appendBtn);
    box.exec();

    if (box.clickedButton() == appendBtn) {
        return TestDataRegistryMode::Append;
    }
    if (box.clickedButton() == overwriteBtn) {
        return TestDataRegistryMode::Overwrite;
    }
    return TestDataRegistryMode::Cancel;
}

void FaceDetectDialog::applyMatchThresholdToAllTabs(double value) {
    m_syncingMatchThresholds = true;
    if (m_verifyThreshold) {
        m_verifyThreshold->blockSignals(true);
        m_verifyThreshold->setValue(value);
        m_verifyThreshold->blockSignals(false);
    }
    if (m_registryWidget) {
        m_registryWidget->setAuthThreshold(static_cast<float>(value));
    }
    if (m_liveWidget) {
        m_liveWidget->setMatchThreshold(static_cast<float>(value));
    }
    QSettings settings;
    settings.setValue(QStringLiteral("qFaceDetect/matchThreshold"), value);
    settings.setValue(QStringLiteral("qFaceDetect/verifyThreshold"), value);
    m_syncingMatchThresholds = false;
}

void FaceDetectDialog::setupMatchThresholdLinks() {
    if (m_verifyThreshold) {
        connect(m_verifyThreshold,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this](double value) {
                    if (m_syncingMatchThresholds) return;
                    if (m_linkMatchThresholdsCheck &&
                        m_linkMatchThresholdsCheck->isChecked()) {
                        applyMatchThresholdToAllTabs(value);
                    }
                });
    }
    if (m_registryWidget) {
        connect(m_registryWidget, &FaceRegistryWidget::authThresholdChanged,
                this, [this](float value) {
                    if (m_syncingMatchThresholds) return;
                    if (m_linkMatchThresholdsCheck &&
                        m_linkMatchThresholdsCheck->isChecked()) {
                        applyMatchThresholdToAllTabs(value);
                    }
                });
    }
    if (m_liveWidget) {
        connect(m_liveWidget, &FaceLiveDetectWidget::matchThresholdChanged,
                this, [this](float value) {
                    if (m_syncingMatchThresholds) return;
                    if (m_linkMatchThresholdsCheck &&
                        m_linkMatchThresholdsCheck->isChecked()) {
                        applyMatchThresholdToAllTabs(value);
                    }
                });
    }
    if (m_applyMatchThresholdBtn && m_verifyThreshold) {
        connect(m_applyMatchThresholdBtn, &QPushButton::clicked, this,
                [this]() {
                    if (m_verifyThreshold) {
                        applyMatchThresholdToAllTabs(
                                m_verifyThreshold->value());
                    }
                });
    }
}

void FaceDetectDialog::applyMinDetectionScoreToAllTabs(double value) {
    if (m_syncingMinScores) return;
    m_syncingMinScores = true;
    if (m_minDetectionScore) {
        m_minDetectionScore->blockSignals(true);
        m_minDetectionScore->setValue(value);
        m_minDetectionScore->blockSignals(false);
    }
    if (m_verifyMinDetectionScore) {
        m_verifyMinDetectionScore->blockSignals(true);
        m_verifyMinDetectionScore->setValue(value);
        m_verifyMinDetectionScore->blockSignals(false);
    }
    if (m_registryWidget) {
        m_registryWidget->setMinDetectionScore(static_cast<float>(value));
    }
    if (m_liveWidget) {
        m_liveWidget->setMinDetectionScore(static_cast<float>(value));
    }
    QSettings settings;
    settings.setValue(QStringLiteral("qFaceDetect/minDetectionScore"), value);
    m_syncingMinScores = false;
}

void FaceDetectDialog::setupMinScoreLinks() {
    if (m_minDetectionScore) {
        connect(m_minDetectionScore,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this](double value) {
                    if (m_syncingMinScores) return;
                    applyMinDetectionScoreToAllTabs(value);
                });
    }
    if (m_verifyMinDetectionScore) {
        connect(m_verifyMinDetectionScore,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this](double value) {
                    if (m_syncingMinScores) return;
                    applyMinDetectionScoreToAllTabs(value);
                });
    }
    if (m_liveWidget) {
        connect(m_liveWidget, &FaceLiveDetectWidget::minDetectionScoreChanged,
                this, [this](float value) {
                    if (m_syncingMinScores) return;
                    applyMinDetectionScoreToAllTabs(value);
                });
    }
    if (m_registryWidget) {
        connect(m_registryWidget, &FaceRegistryWidget::minDetectionScoreChanged,
                this, [this](float value) {
                    if (m_syncingMinScores) return;
                    applyMinDetectionScoreToAllTabs(value);
                });
    }
}
