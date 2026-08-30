// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SAM3Dialog.h"

#include <CVLog.h>
#include <aicore/sam3_capi.h>

#ifdef HAS_OPENCV_FACE_CAPTURE
#include "VideoTab.h"
#endif

#include <ecvAICoreUiHelper.h>
#include <ecvClickableImageLabel.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QSettings>
#include <QShowEvent>
#include <QSplitter>
#include <QTabWidget>
#include <QVBoxLayout>
#include <cstring>

#include "ecvModelDownloader.h"

namespace {

// SAM3 test data lives in the shared ecvTestDataRepository SAM3 dataset
// (images/ + videos/ under ~/cloudViewer_data/extract/sam_test_data).

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

}  // namespace

// ---------------------------------------------------------------------------
// SAM3Canvas implementation
// ---------------------------------------------------------------------------

SAM3Canvas::SAM3Canvas(QWidget* parent) : QLabel(parent) {
    setAcceptDrops(true);
    setMinimumSize(ecvAICoreUi::dpiScaled(320), ecvAICoreUi::dpiScaled(240));
    setAlignment(Qt::AlignCenter);
    setStyleSheet(
            "QLabel { background: #1a1a26; border: 1px solid #333;"
            " border-radius: 4px; color: #666; font-size: 13px; }");
    updatePlaceholderText();
}

void SAM3Canvas::updatePlaceholderText() {
    setText(
            tr("Step 1: drop an image here (or load test data)\n"
               "Step 2: type a text prompt, or click on the object /\n"
               "        drag a box (the model loads automatically)\n"
               "(left-click +point · right-click -point · drag box)"));
}

void SAM3Canvas::setInteractive(bool on) {
    m_interactive = on;
    if (!on)
        setCursor(Qt::ArrowCursor);
    else
        setCursor(Qt::CrossCursor);
}

void SAM3Canvas::setImage(const QImage& img) {
    m_original = img;
    m_overlay = img.copy();
    m_maskOverlay = QImage();
    clearPoints();  // also resets box state
    m_dragging = false;
    setPixmap(QPixmap::fromImage(m_overlay.scaled(size(), Qt::KeepAspectRatio,
                                                  Qt::SmoothTransformation)));
    setText(QString());
}

void SAM3Canvas::setMaskOverlay(const QImage& mask) { m_maskOverlay = mask; }

void SAM3Canvas::clearMaskOverlay() { m_maskOverlay = QImage(); }

void SAM3Canvas::setDetections(const QVector<SAM3DetBox>& detections) {
    m_detections = detections;
}

void SAM3Canvas::clearDetections() { m_detections.clear(); }

void SAM3Canvas::setExemplars(const QVector<QRectF>& exemplars) {
    m_exemplars = exemplars;
}

void SAM3Canvas::clearExemplars() { m_exemplars.clear(); }

void SAM3Canvas::setExemplarMode(bool on) { m_exemplarMode = on; }

bool SAM3Canvas::isInsideImage(const QPointF& p) const {
    return !m_original.isNull() && p.x() >= 0 && p.y() >= 0 &&
           p.x() < static_cast<double>(m_original.width()) &&
           p.y() < static_cast<double>(m_original.height());
}

void SAM3Canvas::updateOverlay() {
    if (m_original.isNull()) return;
    QImage display = m_original.copy();
    QPainter p(&display);
    // Layer order: image → mask → point/box annotations
    if (!m_maskOverlay.isNull()) {
        p.drawImage(0, 0, m_maskOverlay);
    }
    drawAnnotations(p, display);
    p.end();
    m_overlay = display;
    setPixmap(QPixmap::fromImage(display.scaled(size(), Qt::KeepAspectRatio,
                                                Qt::SmoothTransformation)));
}

void SAM3Canvas::resizeEvent(QResizeEvent* e) {
    QLabel::resizeEvent(e);
    // Keep the displayed pixmap in sync with the widget size
    if (!m_overlay.isNull()) {
        setPixmap(QPixmap::fromImage(m_overlay.scaled(
                size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    }
}

void SAM3Canvas::clearOverlay() {
    m_original = QImage();
    m_overlay = QImage();
    m_maskOverlay = QImage();
    clearPoints();
    m_hasBox = false;
    m_box = QRectF();
    updatePlaceholderText();
    setPixmap(QPixmap());
}

void SAM3Canvas::addPosPoint(const QPointF& p) {
    m_posPoints.append(p);
    updateOverlay();
}

void SAM3Canvas::addNegPoint(const QPointF& p) {
    m_negPoints.append(p);
    updateOverlay();
}

void SAM3Canvas::clearPoints() {
    m_posPoints.clear();
    m_negPoints.clear();
    m_hasBox = false;
    m_box = QRectF();
    updateOverlay();
}

void SAM3Canvas::clearPointsKeepBox() {
    m_posPoints.clear();
    m_negPoints.clear();
    updateOverlay();
}

QPointF SAM3Canvas::screenToImage(const QPointF& screen) const {
    const QPixmap* pm = pixmap();
    if (!pm || pm->isNull() || m_original.isNull()) return QPointF(-1, -1);
    // The pixmap is centered inside the QLabel (Qt::AlignCenter); its top-left
    // corner is offset from the widget origin by half the letterbox margin.
    const double offX = (width() - pm->width()) / 2.0;
    const double offY = (height() - pm->height()) / 2.0;
    const double ix = (screen.x() - offX) / pm->width() * m_original.width();
    const double iy = (screen.y() - offY) / pm->height() * m_original.height();
    return QPointF(ix, iy);
}

void SAM3Canvas::mouseDoubleClickEvent(QMouseEvent* e) {
    if (!m_original.isNull()) {
        ecvClickableImageLabel::showEnlargedImage(
                this, m_original, tr("SAM3 — full image preview"));
        return;
    }
    QLabel::mouseDoubleClickEvent(e);
}

void SAM3Canvas::mousePressEvent(QMouseEvent* e) {
    if (!m_interactive || m_original.isNull()) {
        QLabel::mousePressEvent(e);
        return;
    }
    if (e->button() == Qt::LeftButton) {
        const QPointF ip = screenToImage(e->localPos());
        if (isInsideImage(ip)) {
            m_dragging = true;
            m_dragStart = ip;
            m_dragRect = QRectF(ip, QSizeF(0, 0));
        }
    } else if (e->button() == Qt::RightButton) {
        const QPointF ip = screenToImage(e->localPos());
        if (isInsideImage(ip)) {
            addNegPoint(ip);
            emit pointAdded(1);
        }
    }
}

void SAM3Canvas::mouseMoveEvent(QMouseEvent* e) {
    if (m_dragging) {
        const QPointF ip = screenToImage(e->localPos());
        m_dragRect = QRectF(m_dragStart, ip).normalized();
        // Show drag rectangle on overlay
        QImage display = m_original.copy();
        QPainter p(&display);
        if (!m_maskOverlay.isNull()) {
            p.drawImage(0, 0, m_maskOverlay);
        }
        drawAnnotations(p, display);
        // Outline only: drawAnnotations may leave a filled brush active
        // (positive / negative point ellipses); drawing the drag rect with
        // that leftover brush filled its interior and hid the image under
        // it, making the box impossible to position.
        p.setBrush(Qt::NoBrush);
        p.setPen(QPen(QColor(255, 255, 0, 180), 2));
        p.drawRect(QRectF(
                m_dragRect.left() / m_original.width() * display.width(),
                m_dragRect.top() / m_original.height() * display.height(),
                m_dragRect.width() / m_original.width() * display.width(),
                m_dragRect.height() / m_original.height() * display.height()));
        p.end();
        m_overlay = display;
        setPixmap(QPixmap::fromImage(display.scaled(size(), Qt::KeepAspectRatio,
                                                    Qt::SmoothTransformation)));
    }
}

void SAM3Canvas::mouseReleaseEvent(QMouseEvent* e) {
    if (!m_dragging) {
        QLabel::mouseReleaseEvent(e);
        return;
    }
    m_dragging = false;

    QPointF ip = screenToImage(e->localPos());
    // Clamp so drags that leave the widget stay valid image coordinates
    ip.setX(qBound(0.0, ip.x(), m_original.width() - 1.0));
    ip.setY(qBound(0.0, ip.y(), m_original.height() - 1.0));

    const double dx = ip.x() - m_dragStart.x();
    const double dy = ip.y() - m_dragStart.y();

    if (dx * dx + dy * dy > 25.0) {
        // Drag → bounding box
        const QRectF drawn = QRectF(m_dragStart, ip).normalized();
        if (m_exemplarMode) {
            // Exemplar (PCS): the box joins the green exemplar list via the
            // boxDrawn(QRectF) payload; the PVS prompt-box slots stay
            // untouched and existing detections stay visible until the next
            // run — upstream main_image.cpp accumulates pos_exemplars next
            // to the live result.
            updateOverlay();
        } else {
            // Points / Box (PVS): a fresh prompt supersedes the previous
            // result — drop the stale detection boxes so the new prompt box
            // is visible immediately (drawAnnotations hides the prompt box
            // while detection boxes exist, to keep one box per object).
            m_detections.clear();
            m_box = drawn;
            m_hasBox = true;
            updateOverlay();
        }
        emit boxDrawn(drawn);
    } else {
        // Click → positive point
        addPosPoint(m_dragStart);
        emit pointAdded(0);
    }
}

void SAM3Canvas::paintEvent(QPaintEvent* e) {
    QLabel::paintEvent(e);
    // Extra overlay handled in mouseMoveEvent; paintEvent draws the pixmap
}

void SAM3Canvas::drawAnnotations(QPainter& p, const QImage& display) {
    const double scaleX =
            static_cast<double>(display.width()) / m_original.width();
    const double scaleY =
            static_cast<double>(display.height()) / m_original.height();

    // Positive points (green)
    p.setPen(QPen(Qt::white, 2));
    p.setBrush(QColor(0, 255, 0, 220));
    for (const auto& pt : m_posPoints) {
        p.drawEllipse(QPointF(pt.x() * scaleX, pt.y() * scaleY), 6, 6);
    }

    // Negative points (red)
    p.setBrush(QColor(255, 0, 0, 220));
    for (const auto& pt : m_negPoints) {
        p.drawEllipse(QPointF(pt.x() * scaleX, pt.y() * scaleY), 6, 6);
    }

    // Exemplar boxes (green), upstream main_image.cpp: pos_exemplars drawn
    // as 2px green rects.
    for (const auto& ex : m_exemplars) {
        p.setPen(QPen(QColor(0, 255, 0, 200), 2));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(ex.left() * scaleX, ex.top() * scaleY,
                          ex.width() * scaleX, ex.height() * scaleY));
    }

    // Confirmed box (cyan) — hidden in Exemplar (PCS) mode where the green
    // exemplar rects above are the only boxes (upstream main_image.cpp).
    // Also hidden while detection boxes exist: a PVS box prompt's result
    // covers the same region, and stacking both rectangles on one object
    // reads as two overlapping results. The prompt box returns as soon as
    // the detections are cleared (Clear button / a freshly dragged box).
    if (m_hasBox && !m_exemplarMode && m_detections.isEmpty()) {
        p.setPen(QPen(QColor(0, 255, 255, 220), 3));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(m_box.left() * scaleX, m_box.top() * scaleY,
                          m_box.width() * scaleX, m_box.height() * scaleY));
    }

    // Detection boxes + labels (upstream main_image.cpp: rect + "#id score")
    for (const auto& det : m_detections) {
        p.setPen(QPen(det.color, 2));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(det.box.left() * scaleX, det.box.top() * scaleY,
                          det.box.width() * scaleX, det.box.height() * scaleY));
        p.drawText(QPointF(det.box.left() * scaleX + 2,
                           det.box.top() * scaleY + 2),
                   QString("#%1 %2")
                           .arg(det.instanceId)
                           .arg(det.score, 0, 'f', 2));
    }
}

// ── Drag & drop ──────────────────────────────────────────────────────────

void SAM3Canvas::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) {
        e->acceptProposedAction();
    }
}

void SAM3Canvas::dropEvent(QDropEvent* e) {
    const auto urls = e->mimeData()->urls();
    if (urls.isEmpty()) return;
    const QString path = urls.first().toLocalFile();
    if (path.isEmpty()) return;

    QImage img(path);
    if (img.isNull()) return;
    setImage(img);
    emit imageDropped(path);
}

// ---------------------------------------------------------------------------
// SAM3Dialog implementation
// ---------------------------------------------------------------------------

SAM3Dialog::SAM3Dialog(QWidget* parent) : QDialog(parent) {
    // Cross-thread queue connections (SAM3Worker / VideoWorker emit
    // resultReady / frameResultReady from their worker threads) require the
    // custom result type to be registered with Qt's meta-type system;
    // without this the queued signal is silently dropped and the UI never
    // updates (mirrors the qYOLO/qRFDetr/qDeepLSD qRegisterMetaType calls).
    qRegisterMetaType<SAM3WorkerResult>();
    setWindowTitle(tr("SAM3 Image & Video Segmentation"));
    setMinimumSize(ecvAICoreUi::dpiScaled(900), ecvAICoreUi::dpiScaled(700));
    setupUi();

    // Shared model downloader (qDA3-style): catalog GGUFs land in the AICore
    // model cache (sam3_models) and are verified by size + GGUF magic.
    m_modelDownloader = new ecvModelDownloader(this);
    connect(m_modelDownloader, &ecvModelDownloader::logMessage, this,
            &SAM3Dialog::appendLog);
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
                const int tab = m_downloadTab;
                const QString filename = m_downloadTargetFilename;
                const bool valid = ok && isValidCatalogModel(dest, filename);
                m_downloadInProgress = false;
                m_downloadThenRun = false;
                if (m_downloadLabel) m_downloadLabel->setVisible(false);
                if (m_progress) m_progress->setVisible(false);
                refreshModelCombos();
                updateDownloadButtons();
                setBusy(false);
                if (valid) {
                    m_downloadPrompted = false;
                    appendLog(tr("Model downloaded: %1").arg(filename));
                    if (thenRun && static_cast<int>(currentTab()) == tab) {
                        // Re-run the pending operation now that the GGUF
                        // exists (lazy model load on first use).
                        m_retryAfterModelLoad = true;
                        autoLoadModelIfAvailable();
                    }
                } else {
                    appendLog(tr("Model download failed or did not pass "
                                 "catalog validation: %1")
                                      .arg(filename));
                }
                m_downloadTargetFilename.clear();
            });

    loadSettings();
}

void SAM3Dialog::showEvent(QShowEvent* e) {
    QDialog::showEvent(e);
    if (m_firstShow) {
        m_firstShow = false;
        adjustSize();
        // No eager model load on dialog open: the initial model (or a model
        // switched later) loads lazily on the first Segment / click / box.
        updateStatus(
                tr("Ready. The selected model loads automatically on "
                   "the first Segment / click / box."));
    }
    if (m_busyOverlay) {
        m_busyOverlay->setGeometry(rect());
        m_busyOverlay->raise();
    }
}

void SAM3Dialog::resizeEvent(QResizeEvent* e) {
    QDialog::resizeEvent(e);
    if (m_busyOverlay) {
        m_busyOverlay->setGeometry(rect());
    }
}

SAM3Dialog::~SAM3Dialog() {
    saveSettings();
    stopWorker();
    // Teardown guard: QObject destroys children (QTabWidget -> VideoTab)
    // AFTER this class's members (m_tabs, m_backendLabel) are gone. The
    // VideoTab destructor calls releaseModel(), which emits backendChanged;
    // the setupUi() lambda connected with `this` context would then run on
    // destroyed members (SIGSEGV in QFunctorSlotObject). Disconnect before
    // the deleteChildren cascade.
    if (m_videoTab) {
        disconnect(m_videoTab, &VideoTab::backendChanged, this, nullptr);
    }
}

void SAM3Dialog::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(8, 8, 8, 8);
    mainLayout->setSpacing(6);

    // ── Device row (shared across the image tabs, compact, on top) ───────
    auto* deviceRow = new QHBoxLayout();

    auto* deviceLabel = new QLabel(tr("Device:"));
    m_deviceCombo = new QComboBox();
    m_deviceCombo->addItems({"Auto", "CPU", "CUDA", "Vulkan"});
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SAM3Dialog::onDeviceChanged);

    m_backendLabel = new QLabel(tr("Backend: none"));
    m_backendLabel->setStyleSheet("color: #99b4d1;");

    deviceRow->addWidget(deviceLabel);
    deviceRow->addWidget(m_deviceCombo);
    deviceRow->addSpacing(20);
    deviceRow->addWidget(m_backendLabel);
    deviceRow->addStretch();
    mainLayout->addLayout(deviceRow);

    // ── Tabs: one self-contained page per model family ────────────────────
    m_tabs = new QTabWidget(this);
    ecvAICoreUi::styleTabWidget(m_tabs);

    // Model-switch hint: the new model loads lazily on the next Segment /
    // click / box (no reload while the dialog is idle).
    const auto onModelComboChanged = [this]() {
        m_downloadPrompted = false;
        if (m_worker && m_worker->context() && modelSelectionChanged()) {
            appendLog(
                    tr("Model changed - the new model will load on the "
                       "next Segment / click / box."));
        }
        updateDownloadButtons();
    };

    // Each image tab mirrors the upstream main_image.cpp single panel:
    // two compact control rows, the canvas expanding to fill all remaining
    // space and a compact bottom bar. Nothing is shared with the Video tab,
    // so the image tabs are never stretched by the video layout.
    auto buildImageTab = [this, &onModelComboChanged](Sam3Tab which,
                                                      const QString& title,
                                                      bool isFull) {
        ImageTabUi& u = m_tabsUi[static_cast<int>(which)];
        u.tab = new QWidget();
        auto* layout = new QVBoxLayout(u.tab);
        ecvAICoreUi::setupTabLayout(layout);

        // Row 1: mode + (Full: text prompt + Segment) + test-image picker
        auto* row1 = new QHBoxLayout();
        auto* modeLabel = new QLabel(tr("Mode:"));
        u.modePoints = new QRadioButton(tr("Points"));
        u.modeBox = new QRadioButton(tr("Box (PVS)"));
        u.modeBox->setChecked(true);
        u.modeExemplar =
                isFull ? new QRadioButton(tr("Exemplar (PCS)")) : nullptr;
        auto* modeGroup = new QButtonGroup(this);
        modeGroup->addButton(u.modePoints, 0);
        modeGroup->addButton(u.modeBox, 1);
        if (u.modeExemplar) modeGroup->addButton(u.modeExemplar, 2);
        connect(modeGroup, QOverload<int>::of(&QButtonGroup::buttonClicked),
                this, &SAM3Dialog::onModeChanged);

        if (isFull) {
            u.textPrompt = new QLineEdit();
            u.textPrompt->setPlaceholderText(
                    tr("Describe the object to "
                       "segment (e.g. person, "
                       "car)..."));
            u.textPrompt->setMinimumWidth(ecvAICoreUi::dpiScaled(240));
            connect(u.textPrompt, &QLineEdit::returnPressed, this,
                    &SAM3Dialog::onRunSegment);
            u.segmentBtn = new QPushButton(tr("Segment"));
            u.segmentBtn->setEnabled(false);
            u.segmentBtn->setStyleSheet(
                    "QPushButton { background: #00897b; color: white; "
                    "font-weight: bold;"
                    "  border: none; border-radius: 4px; padding: 5px 14px; }"
                    "QPushButton:hover { background: #00796b; }"
                    "QPushButton:disabled { background: #555; color: #999; }");
            connect(u.segmentBtn, &QPushButton::clicked, this,
                    &SAM3Dialog::onRunSegment);
        }

        u.testDataCombo = new QComboBox();
        u.testDataCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(140));
        u.testDataCombo->setToolTip(
                tr("Pick which sample image to load (SAM3 test dataset)"));

        u.testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
        u.testDataBtn->setToolTip(
                tr("Download (cached) and load the selected sample image for "
                   "one-click testing"));
        connect(u.testDataBtn, &QPushButton::clicked, this,
                &SAM3Dialog::requestTestData);
        // Switching the picker auto-loads that sample into this tab's
        // canvas immediately (no need to press the sample-data button
        // again once the dataset is cached).
        connect(u.testDataCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this, &u](int) {
                    if (u.currentImage.isNull() || isRunning()) {
                        // Nothing to replace yet, or a task is using the
                        // current image; the next click re-runs anyway.
                        return;
                    }
                    loadTestImageInto(u);
                    appendLog(
                            tr("Sample switched - the selected model "
                               "re-encodes the new image on the next "
                               "Segment / click / box."));
                });

        row1->addWidget(modeLabel);
        row1->addWidget(u.modePoints);
        row1->addWidget(u.modeBox);
        if (u.modeExemplar) row1->addWidget(u.modeExemplar);
        if (isFull) {
            row1->addSpacing(10);
            row1->addWidget(u.textPrompt, 1);
            row1->addWidget(u.segmentBtn);
        }
        row1->addSpacing(10);
        row1->addWidget(u.testDataCombo);
        row1->addWidget(u.testDataBtn);
        layout->addLayout(row1);

        // Row 2: model + Load
        auto* row2 = new QHBoxLayout();
        auto* modelLabel = new QLabel(tr("Model:"));
        u.modelCombo = new QComboBox();
        u.modelCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(240));
        u.loadBtn = new QPushButton(tr("Load"));
        u.loadBtn->setStyleSheet(
                "QPushButton { background: #00897b; color: white; "
                "font-weight: bold;"
                "  border: none; border-radius: 4px; padding: 5px 14px; }"
                "QPushButton:hover { background: #00796b; }");
        // No manual Load step: the model loads lazily on the first Segment /
        // click / box. The button stays wired for the Browse fallback but
        // is hidden so the UI matches the upstream flow.
        u.loadBtn->setVisible(false);
        connect(u.loadBtn, &QPushButton::clicked, this,
                &SAM3Dialog::onLoadModel);
        connect(u.modelCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                onModelComboChanged);
        row2->addWidget(modelLabel);
        row2->addWidget(u.modelCombo, 1);
        // One-click download of the selected catalog GGUF into the shared
        // AICore model cache (sam3_models), qDA3-style.
        u.downloadBtn = new QPushButton(tr("Download"));
        u.downloadBtn->setToolTip(
                tr("Download the selected model GGUF into the model cache"));
        connect(u.downloadBtn, &QPushButton::clicked, this, [this]() {
            if (!m_downloadInProgress) startDownload(false);
        });
        row2->addWidget(u.downloadBtn);
        row2->addWidget(u.loadBtn);
        layout->addLayout(row2);

        // Canvas: fills all remaining tab space (upstream: the image region
        // is the whole panel below the compact controls).
        u.canvas = new SAM3Canvas();
        u.canvas->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        connect(u.canvas, &SAM3Canvas::pointAdded, this,
                &SAM3Dialog::onCanvasPoint);
        connect(u.canvas, &SAM3Canvas::boxDrawn, this,
                &SAM3Dialog::onCanvasBox);
        connect(u.canvas, &SAM3Canvas::imageDropped, this,
                [this, &u](const QString& path) {
                    QImage img(path);
                    if (img.isNull()) return;
                    u.currentImage = img;
                    u.currentImagePath = path;
                    // The canvas already displayed the image in dropEvent()
                    // (setImage → clearPoints); calling it again here would
                    // re-decode/rescale the full image and reset the prompt
                    // state a second time.
                    appendLog(tr("Loaded image: %1")
                                      .arg(QFileInfo(path).fileName()));
                    if (m_worker && m_worker->context()) {
                        appendLog(
                                tr("Model ready — click on an object or "
                                   "drag a box to segment it."));
                    } else {
                        appendLog(
                                tr("The selected model loads automatically "
                                   "on the first Segment / click / box."));
                    }
                    updateSegmentButtonState();
                });
        layout->addWidget(u.canvas, 1);

        // Bottom bar (compact, mirroring the upstream bottom panel)
        auto* bottom = new QHBoxLayout();
        u.detLabel = new QLabel(tr("Detections: 0 instances"));
        u.scoreSpin = new QDoubleSpinBox();
        u.scoreSpin->setRange(0.0, 1.0);
        u.scoreSpin->setDecimals(2);
        u.scoreSpin->setSingleStep(0.05);
        u.scoreSpin->setValue(0.5);
        ecvAICoreUi::setCompactDoubleSpin(u.scoreSpin);
        auto* scoreLabel = new QLabel(tr("Score threshold:"));

        u.showMasks = new QCheckBox(tr("Show masks"));
        u.showMasks->setChecked(true);
        connect(u.showMasks, &QCheckBox::toggled, this, [this](bool) {
            if (currentUi().lastResult.valid) {
                updateCanvasFromResult();
            }
        });

        u.exportToDbCheckBox = new QCheckBox(tr("Export to DB"));
        u.exportToDbCheckBox->setChecked(false);  // opt-in, not automatic
        u.exportToDbCheckBox->setToolTip(
                tr("Automatically add the segmented result to the DB tree "
                   "as an annotated image"));

        u.multimask = new QCheckBox(tr("Multi-mask (PVS)"));
        u.multimask->setToolTip(
                tr("PVS mask-decoder output mode.\n"
                   "ON: run the decoder with 3 candidate masks (whole "
                   "object / part / subpart) and return the one with the "
                   "highest predicted IoU — helps point prompts on objects "
                   "with an ambiguous extent (planes, crowds, herds...).\n"
                   "OFF: return the single default mask. Recommended for box "
                   "prompts, where the box already defines the extent."));

        u.clearBtn = new QPushButton(tr("Clear"));
        connect(u.clearBtn, &QPushButton::clicked, this, &SAM3Dialog::onClear);
        u.exportBtn = new QPushButton(tr("Export masks"));
        u.exportBtn->setToolTip(
                tr("Export the current detections' per-instance masks as "
                   "grayscale images into the DB tree"));
        connect(u.exportBtn, &QPushButton::clicked, this,
                &SAM3Dialog::onExportMasks);

        u.statusLabel = new QLabel(tr("Ready."));
        u.statusLabel->setStyleSheet("color: #99ccff;");
        // Timing / log lines are long; wrap instead of clipping at the
        // right dialog edge.
        u.statusLabel->setWordWrap(true);

        bottom->addWidget(u.detLabel);
        bottom->addWidget(scoreLabel);
        bottom->addWidget(u.scoreSpin);
        bottom->addWidget(u.showMasks);
        bottom->addWidget(u.exportToDbCheckBox);
        bottom->addWidget(u.multimask);
        bottom->addSpacing(16);
        bottom->addWidget(u.clearBtn);
        bottom->addWidget(u.exportBtn);
        bottom->addWidget(u.statusLabel, 1);
        layout->addLayout(bottom);

        // Detection list: wraps to multiple lines (10+ instances overflow
        // a single row) and stays selectable for copying scores out.
        u.detectionLabel = new QLabel();
        u.detectionLabel->setWordWrap(true);
        u.detectionLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
        layout->addWidget(u.detectionLabel);

        m_tabs->addTab(u.tab, title);
    };

    buildImageTab(Sam3Tab::Full, tr("SAM 3 Full"), /*isFull=*/true);
    buildImageTab(Sam3Tab::Visual, tr("SAM 3 Visual"), /*isFull=*/false);
    buildImageTab(Sam3Tab::Sam2, tr("SAM 2 / 2.1"), /*isFull=*/false);

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Video segmentation & tracking (upstream examples/main_video.cpp):
    // fully self-contained page with its own canvas, timeline, device combo
    // and bottom bar — it never shares layout state with the image tabs.
    m_videoTab = new VideoTab();
    if (m_app) {
        m_videoTab->setAppInterface(m_app);
    }
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (m_videoTab) {
                    m_videoTab->setDevice(m_deviceCombo->currentText());
                }
            });
    connect(m_videoTab, &VideoTab::backendChanged, this,
            [this](const QString& backend) {
                if (m_tabs && m_videoTab &&
                    m_tabs->currentWidget() == m_videoTab) {
                    m_backendLabel->setText(tr("Backend: %1").arg(backend));
                }
            });
    m_tabs->addTab(m_videoTab, tr("Video"));
#endif

    // The tab area holds the self-contained pages; it expands to fill the
    // dialog so each page's canvas gets all the remaining space.
    mainLayout->addWidget(m_tabs, 1);

    // ── Progress section (shared helper, hidden by default) ────────────────
    ecvAICoreUi::setupProgressSection(mainLayout, m_downloadLabel, m_progress);

    // ── Busy overlay (spinner + dim, upstream main_image.cpp overlay) ────
    m_busyOverlay = new QLabel(this);
    m_busyOverlay->setAlignment(Qt::AlignCenter);
    m_busyOverlay->setStyleSheet(
            "QLabel { background: rgba(10,10,16,150); color: white;"
            " font-size: 15px; font-weight: bold; border: none; }");
    m_busyOverlay->setVisible(false);
    m_busyTimer = new QTimer(this);
    m_busyTimer->setInterval(100);
    connect(m_busyTimer, &QTimer::timeout, this, [this]() {
        static const QString kFrames = QString::fromUtf8("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        const QChar c = kFrames.at(m_busyFrame % kFrames.size());
        m_busyOverlay->setText(QString("%1  Working...").arg(c));
        ++m_busyFrame;
    });

    // ── Connections ─────────────────────────────────────────────────────────
    connect(m_tabs, &QTabWidget::currentChanged, this,
            &SAM3Dialog::onTabChanged);

    // Shared test data repository (SAM3 dataset).
    auto& repo = ecvTestDataRepository::instance();
    connect(&repo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                if (!m_testDataDownloadInProgress) return;
                updateStatus(QString("%1 (%2%)").arg(statusText).arg(percent));
                if (m_progress) {
                    m_progress->setValue(percent);
                }
            });
    connect(&repo, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_testDataDownloadInProgress) appendLog(message);
            });
    connect(&repo, &ecvTestDataRepository::downloadFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                onTestDataDownloadFinished(success, kind);
            });
    connect(&repo, &ecvTestDataRepository::extractionProgress, this,
            [this](int current, int total) {
                if (!m_testDataDownloadInProgress || total <= 0) return;
                updateStatus(tr("Extracting test data... %1/%2")
                                     .arg(current)
                                     .arg(total));
                if (m_progress && total > 0) {
                    m_progress->setValue(current * 100 / total);
                }
            });
    connect(&repo, &ecvTestDataRepository::extractionFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                onTestDataExtractionFinished(success, kind);
            });

    populateModelCombo(m_tabsUi[0].modelCombo, Sam3Tab::Full);
    populateModelCombo(m_tabsUi[1].modelCombo, Sam3Tab::Visual);
    populateModelCombo(m_tabsUi[2].modelCombo, Sam3Tab::Sam2);

    // Fill the test-image pickers from the (possibly cached) SAM3 dataset.
    populateTestDataCombos();
}

// Model family filter per tab. SAM 2 / 2.1 share one tab (visual-only Hiera).
bool SAM3Dialog::familyMatches(const char* family, Sam3Tab tab) {
    if (!family || !*family) return false;
    switch (tab) {
        case Sam3Tab::Full:
            return std::strcmp(family, "sam3") == 0;
        case Sam3Tab::Visual:
            return std::strcmp(family, "sam3-visual") == 0;
        case Sam3Tab::Sam2:
            return std::strncmp(family, "sam2", 4) == 0;  // sam2 / sam2.1
    }
    return false;
}

void SAM3Dialog::populateModelCombo(QComboBox* combo, Sam3Tab tab) {
    combo->clear();
    const int n = aicore_sam3_model_count();
    if (n <= 0) {
        combo->addItem(tr("(no models)"));
        return;
    }
    // Cache status suffix mirrors qDA3: "[size] ✓" when the GGUF is cached
    // in the shared AICore model cache, "[download]" otherwise.
    char* cacheDirRaw = aicore_sam3_model_cache_dir();
    const QString cacheDir = QString::fromUtf8(cacheDirRaw);
    aicore_sam3_free_buffer(cacheDirRaw);
    // Add all catalog entries for this tab's model family. The catalog already
    // excludes the unpublishable sam3-f32, so every entry (including the
    // sam2/sam2.1 f32 variants) is selectable.
    for (int i = 0; i < n; ++i) {
        const auto* entry = aicore_sam3_model_at(i);
        if (!entry) continue;
        if (!familyMatches(entry->model_family, tab)) continue;
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
        combo->addItem(QString("%1 (%2)%3")
                               .arg(entry->display_name)
                               .arg(entry->quant_note)
                               .arg(suffix),
                       entry->filename);
    }
    combo->addItem(tr("Browse..."), QString("__browse__"));
}

void SAM3Dialog::selectModelByFilename(QComboBox* combo,
                                       const QString& filename) {
    if (filename.isEmpty() || !combo) return;
    const int idx = combo->findData(filename);
    if (idx >= 0) combo->setCurrentIndex(idx);
}

void SAM3Dialog::loadSettings() {
    QSettings settings("qSAM3");
    m_settings.device = settings.value("device", "auto").toString();
    m_settings.threads = settings.value("threads", 4).toInt();
    m_settings.scoreThreshold =
            settings.value("scoreThreshold", 0.5).toDouble();
    m_settings.nmsThreshold = settings.value("nmsThreshold", 0.1).toDouble();
    m_settings.modelFull = settings.value("modelFull").toString();
    m_settings.modelVisual = settings.value("modelVisual").toString();
    m_settings.modelSam2 = settings.value("modelSam2").toString();

    for (int i = 0; i < m_deviceCombo->count(); ++i) {
        if (m_deviceCombo->itemText(i).compare(m_settings.device,
                                               Qt::CaseInsensitive) == 0) {
            m_deviceCombo->setCurrentIndex(i);
            break;
        }
    }
    // Export-to-DB is intentionally NOT persisted: it must be off by
    // default on every start (the user opts in per session; all previews
    // live in the plugin UI). Restoring a previously saved "on" state
    // would silently re-enable DB-tree exports.
    for (ImageTabUi& u : m_tabsUi) {
        if (u.scoreSpin) u.scoreSpin->setValue(m_settings.scoreThreshold);
    }

    selectModelByFilename(m_tabsUi[0].modelCombo, m_settings.modelFull);
    selectModelByFilename(m_tabsUi[1].modelCombo, m_settings.modelVisual);
    selectModelByFilename(m_tabsUi[2].modelCombo, m_settings.modelSam2);
}

void SAM3Dialog::saveSettings() {
    QSettings settings("qSAM3");
    settings.setValue("device", m_settings.device);
    settings.setValue("threads", m_settings.threads);
    settings.setValue("scoreThreshold", m_tabsUi[0].scoreSpin
                                                ? m_tabsUi[0].scoreSpin->value()
                                                : m_settings.scoreThreshold);
    settings.setValue("nmsThreshold", m_settings.nmsThreshold);
    settings.setValue("modelFull",
                      m_tabsUi[0].modelCombo->currentData().toString());
    settings.setValue("modelVisual",
                      m_tabsUi[1].modelCombo->currentData().toString());
    settings.setValue("modelSam2",
                      m_tabsUi[2].modelCombo->currentData().toString());
    // Export-to-DB is deliberately not saved (see loadSettings).
}

QString SAM3Dialog::modelPath() const {
    const QString filename = currentModelCombo()->currentData().toString();
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

void SAM3Dialog::startDownload(bool thenRun) {
    if (m_downloadInProgress || !m_modelDownloader) {
        if (m_downloadInProgress) {
            appendLog(tr("[Warning] A model download is already in progress."));
        }
        return;
    }
    const QString filename = currentModelCombo()->currentData().toString();
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
        if (thenRun) {
            m_retryAfterModelLoad = true;
            autoLoadModelIfAvailable();
        }
        return;
    }
    ecvAssetIntegrity::removeIfNotVerified(
            dest, {}, 64 * 1024, true,
            ecvAssetIntegrity::OnMiss::CheapChecksOnly, entry->size_bytes);

    QDir().mkpath(cacheDir);
    m_downloadInProgress = true;
    m_downloadThenRun = thenRun;
    m_downloadTab = static_cast<int>(currentTab());
    m_downloadTargetFilename = filename;
    if (m_downloadLabel) {
        m_downloadLabel->setText(tr("Downloading %1 ...").arg(filename));
        m_downloadLabel->setVisible(true);
    }
    if (m_progress) {
        m_progress->setVisible(true);
        m_progress->setValue(0);
    }
    setBusy(true);
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

void SAM3Dialog::updateDownloadButtons() {
    if (!m_modelDownloader) return;
    char* dir = aicore_sam3_model_cache_dir();
    const QString cacheDir = QString::fromUtf8(dir);
    aicore_sam3_free_buffer(dir);
    for (ImageTabUi& u : m_tabsUi) {
        if (!u.downloadBtn) continue;
        const QString filename = u.modelCombo->currentData().toString();
        const bool isCatalogModel =
                !filename.isEmpty() && filename != "__browse__";
        const bool cached =
                isCatalogModel &&
                isValidCatalogModel(cacheDir + QLatin1Char('/') + filename,
                                    filename);
        u.downloadBtn->setEnabled(!m_downloadInProgress && isCatalogModel &&
                                  !cached && !m_busy);
        u.downloadBtn->setText(cached ? tr("Cached") : tr("Download"));
    }
}

void SAM3Dialog::refreshModelCombos() {
    const QString keep[3] = {
            m_tabsUi[0].modelCombo->currentData().toString(),
            m_tabsUi[1].modelCombo->currentData().toString(),
            m_tabsUi[2].modelCombo->currentData().toString(),
    };
    for (int i = 0; i < 3; ++i) {
        populateModelCombo(m_tabsUi[i].modelCombo, static_cast<Sam3Tab>(i));
        selectModelByFilename(m_tabsUi[i].modelCombo, keep[i]);
    }
}

void SAM3Dialog::applyDbTreeSelection(const QStringList& names) {
    if (names.isEmpty()) return;
    // For now we just log the selection; image loading is drag-and-drop only
    appendLog(tr("DB selection: %1").arg(names.join(", ")));
}

void SAM3Dialog::setAppInterface(ecvMainAppInterface* app) {
    m_app = app;
    // The Video tab needs the interface too; qSAM3 calls this after the
    // dialog is constructed, so forward it here (setupUi's own forwarding
    // runs before m_app is set and is a no-op).
    if (m_videoTab) m_videoTab->setAppInterface(app);
}

void SAM3Dialog::appendLog(const QString& msg) { updateStatus(msg); }

// ── Slots ──────────────────────────────────────────────────────────────────

void SAM3Dialog::onLoadModel() {
    if (isRunning()) {
        appendLog(tr("Worker is busy; wait for the current task to finish."));
        return;
    }

    QString path = modelPath();
    const bool isBrowseItem =
            currentModelCombo()->currentData().toString() == "__browse__";
    if (path.isEmpty() || isBrowseItem) {
        path = QFileDialog::getOpenFileName(
                this, tr("Select SAM3 GGUF model"), QDir::homePath(),
                tr("GGUF files (*.gguf);;All files (*)"));
        if (path.isEmpty()) return;
    } else if (!QFileInfo::exists(path)) {
        // The combo lists published models, but the GGUF itself is not
        // downloaded yet. Tell the user instead of failing silently and
        // leaving the Segment button grey with no explanation.
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

    m_modelPath = path;
    m_settings.device = m_deviceCombo->currentText().toLower();
    ImageTabUi& u = currentUi();
    if (u.segmentBtn) u.segmentBtn->setEnabled(false);
    setBusy(true);

    startWorker(SAM3WorkerAction::LoadModel);
}

void SAM3Dialog::onRunSegment() { runSegmentation(false); }

void SAM3Dialog::runSegmentation(bool canvasPrompt) {
    ImageTabUi& u = currentUi();
    if (u.currentImage.isNull()) {
        CVLog::Warning("[qSAM3] onRunSegment: no image loaded (tab %d)",
                       static_cast<int>(currentTab()));
        appendLog(tr("No image loaded. Drag and drop an image on the canvas."));
        return;
    }
    if (isRunning()) {
        CVLog::Warning("[qSAM3] onRunSegment: worker busy, Segment ignored");
        appendLog(tr("Worker is busy; wait for the current task to finish."));
        return;
    }
    m_retryCanvasPrompt = canvasPrompt;
    if (!ensureModelReady()) {
        // Model not ready: a lazy load is running (the operation re-runs
        // once it finishes) or the GGUF file is missing (already prompted).
        return;
    }

    SAM3Worker::Prompt prompt;

    // A non-empty text prompt runs PCS (text + tracking) in every mode of
    // the Full tab: typing "person" and pressing Segment must segment the
    // person without first switching to Exemplar mode.
    const QString text =
            u.textPrompt ? u.textPrompt->text().trimmed() : QString();
    const bool haveText = !canvasPrompt && !text.isEmpty() &&
                          currentTab() == Sam3Tab::Full && !m_visualOnly;

    if (m_visualOnly && !text.isEmpty()) {
        // The C-API rejects PCS on a visual-only model with no feedback;
        // surface the mismatch instead of silently doing nothing.
        QMessageBox::information(
                this, tr("qSAM3"),
                tr("The loaded model is visual-only and cannot read text "
                   "prompts.\n\n"
                   "Click positive points on the object or drag a bounding "
                   "box around it instead."));
        return;
    }

    if (!canvasPrompt && (currentPcsMode() || haveText)) {
        // PCS mode — mirror upstream examples/main_image.cpp: the text
        // prompt is optional; exemplar boxes alone are enough to run PCS
        // (text just adds a guide).
        qstrncpy(prompt.text, text.toUtf8().constData(), sizeof(prompt.text));
        // Exemplar boxes drawn on the canvas (upstream main_image.cpp
        // pos_exemplars): every box is a positive exemplar for PCS.
        for (const auto& b : u.posExemplars) {
            prompt.posExemplars.push_back({static_cast<float>(b.left()),
                                           static_cast<float>(b.top()),
                                           static_cast<float>(b.right()),
                                           static_cast<float>(b.bottom())});
        }
        // A box drawn in Points / Box mode while a text prompt is present
        // acts as an exemplar (text + region hint) instead of being dropped.
        if (prompt.posExemplars.empty() && u.canvas->hasBox()) {
            const QRectF b = u.canvas->box();
            prompt.posExemplars.push_back({static_cast<float>(b.left()),
                                           static_cast<float>(b.top()),
                                           static_cast<float>(b.right()),
                                           static_cast<float>(b.bottom())});
        }
        if (!prompt.text[0] && prompt.posExemplars.empty()) {
            QMessageBox::information(
                    this, tr("qSAM3"),
                    tr("Nothing to segment yet.\n\n"
                       "Type a text prompt, click positive points on the "
                       "object, or drag a bounding box around it."));
            return;
        }
        prompt.scoreThreshold = static_cast<float>(
                u.scoreSpin ? u.scoreSpin->value() : m_settings.scoreThreshold);
        prompt.nmsThreshold = m_settings.nmsThreshold;
        m_worker->setAction(SAM3WorkerAction::EncodeAndSegmentPCS);
    } else {
        // PVS mode: points or box
        const QVector<QPointF>& posPts = u.canvas->posPoints();
        const QVector<QPointF>& negPts = u.canvas->negPoints();
        if (posPts.isEmpty() && !u.canvas->hasBox()) {
            // The C-API requires at least one positive point or a box;
            // negative-only clicks must not start a pointless run. Give
            // visible feedback instead of a silent status-line update.
            QMessageBox::information(
                    this, tr("qSAM3"),
                    tr("Nothing to segment yet.\n\n"
                       "Click positive points on the object, drag a bounding "
                       "box around it, or type a text prompt."));
            return;
        }
        for (const auto& pt : posPts) {
            prompt.posPoints.append(
                    {static_cast<float>(pt.x()), static_cast<float>(pt.y())});
        }
        for (const auto& pt : negPts) {
            prompt.negPoints.append(
                    {static_cast<float>(pt.x()), static_cast<float>(pt.y())});
        }
        if (u.canvas->hasBox()) {
            const QRectF b = u.canvas->box();
            prompt.pvsBox = {static_cast<float>(b.left()),
                             static_cast<float>(b.top()),
                             static_cast<float>(b.right()),
                             static_cast<float>(b.bottom())};
            prompt.usePvsBox = true;
        }
        prompt.multimask = u.multimask->isChecked();
        m_worker->setAction(SAM3WorkerAction::EncodeAndSegmentPVS);
    }

    setBusy(true);
    m_worker->setImage(u.currentImage);
    m_worker->setPrompt(prompt);
    m_worker->start();
}

void SAM3Dialog::onClear() {
    // Drop any pending lazy-load retry so a stale operation never fires
    // after the canvas / prompt have been cleared.
    m_retryAfterModelLoad = false;
    ImageTabUi& u = currentUi();
    u.canvas->clearPoints();
    u.canvas->clearMaskOverlay();
    u.canvas->clearDetections();
    u.canvas->clearExemplars();
    u.canvas->updateOverlay();
    u.lastResult = SAM3WorkerResult{};
    if (u.textPrompt) u.textPrompt->clear();
    u.posExemplars.clear();
    u.detectionLabel->clear();
    updateStatus(tr("Cleared."));
}

void SAM3Dialog::onExportMasks() {
    // One-shot DB export: each per-instance mask lands as its own grayscale
    // ccImage in the DB tree — no filesystem round-trip. The annotated
    // composite belongs to the Export-to-DB checkbox, not to this button.
    ImageTabUi& u = currentUi();
    if (!u.lastResult.valid || u.lastResult.instanceMasks.isEmpty()) {
        appendLog(tr("No masks to export."));
        return;
    }
    const int dbCount = exportMasksToDb(&u);
    if (dbCount > 0) {
        appendLog(tr("Added %1 mask image(s) to the DB tree.").arg(dbCount));
    } else {
        appendLog(tr("No DB interface available; masks were not exported."));
    }
}

int SAM3Dialog::exportMasksToDb(ImageTabUi* target) {
    ImageTabUi& u = target ? *target : currentUi();
    if (!u.lastResult.valid || !m_app || u.lastResult.instanceMasks.isEmpty()) {
        return 0;
    }
    const QString deviceTag =
            ecvPluginDbNaming::deviceTagFromName(m_settings.device);
    // Model tag keeps runs from different checkpoint variants separable
    // in the DB tree (source + device alone collide across variants).
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(modelPath());
    const QString sourceLabel =
            u.currentImagePath.isEmpty()
                    ? QStringLiteral("canvas")
                    : QFileInfo(u.currentImagePath).completeBaseName();
    int added = 0;
    for (int i = 0; i < u.lastResult.instanceMasks.size(); ++i) {
        const QImage mask = u.lastResult.instanceMasks.value(i);
        if (mask.isNull()) continue;
        const int id = u.lastResult.instanceIds.value(i, i + 1);
        const QString name = ecvPluginDbNaming::makeUnique(
                QStringLiteral("SAM3_%1_%2_%3_mask_obj%4")
                        .arg(modelTag, sourceLabel, deviceTag)
                        .arg(id),
                m_app);
        auto* img = new ccImage(mask, name);
        img->setMetaData(QStringLiteral("SAM3"), true);
        img->setMetaData(QStringLiteral("SAM3/Kind"), QStringLiteral("mask"));
        img->setMetaData(QStringLiteral("SAM3/InstanceId"),
                         static_cast<qlonglong>(id));
        img->setMetaData(QStringLiteral("SAM3/Score"),
                         static_cast<double>(u.lastResult.scores.value(i)));
        const aicore_sam3_box& b = u.lastResult.boxes.value(i);
        img->setMetaData(QStringLiteral("SAM3/Box"),
                         QStringLiteral("[%1,%2,%3,%4]")
                                 .arg(b.x0, 0, 'f', 1)
                                 .arg(b.y0, 0, 'f', 1)
                                 .arg(b.x1, 0, 'f', 1)
                                 .arg(b.y1, 0, 'f', 1));
        img->setMetaData(QStringLiteral("SAM3/Device"), m_settings.device);
        img->setMetaData(QStringLiteral("SAM3/Model"),
                         QFileInfo(m_modelPath).fileName());
        if (!u.currentImagePath.isEmpty()) {
            img->setMetaData(QStringLiteral("Source"), u.currentImagePath);
        }
        m_app->addToDB(img, /*updateZoom=*/false, /*autoExpandDBTree=*/true,
                       /*checkDimensions=*/false, /*autoRedraw=*/true);
        ++added;
    }
    return added;
}

void SAM3Dialog::onWorkerFinished(bool ok) {
    setBusy(false);
    if (!ok) {
        m_tabsUi[m_taskTab].statusLabel->setText(
                tr("Task failed or "
                   "cancelled."));
    }
}

void SAM3Dialog::onWorkerProgress(int current, int total) {
    // Not used for now
}

void SAM3Dialog::onWorkerLog(const QString& msg) {
    if (m_tabsUi[m_taskTab].statusLabel) {
        m_tabsUi[m_taskTab].statusLabel->setText(msg);
    }
}

void SAM3Dialog::onWorkerResult(const SAM3WorkerResult& result) {
    // Always land on the tab that started the task, even if the user
    // switched tabs while it was running.
    ImageTabUi& u = m_tabsUi[m_taskTab];
    if (!result.valid) {
        u.statusLabel->setText(result.errorMsg.isEmpty() ? tr("No "
                                                              "detections.")
                                                         : result.errorMsg);
        return;
    }
    u.lastResult = result;
    updateCanvasFromResult(&u);
    updateDetectionList(&u);

    // Auto-export to DB tree if enabled (opt-in) and we have an app
    // interface; all previews always stay in the plugin UI.
    if (u.exportToDbCheckBox->isChecked() && m_app) {
        exportToDb(&u);
    }

    const auto& t = result.timings;
    u.statusLabel->setText(
            tr("Done — preprocess %1 ms, inference %2 ms, total %3 ms — "
               "%4 detections")
                    .arg(QString::number(t.preprocess_ms, 'f', 0))
                    .arg(QString::number(t.inference_ms, 'f', 0))
                    .arg(QString::number(t.e2e_ms, 'f', 0))
                    .arg(result.detCount));
}

void SAM3Dialog::onCanvasPoint(int type) {
    ImageTabUi& u = currentUi();
    // Auto-segment on point addition. A negative-only click has no meaning
    // for the C-API (it needs a positive point or a box), so wait until the
    // user has placed a positive point/box.
    if (u.currentImage.isNull()) {
        CVLog::Warning("[qSAM3] canvas point on tab with no image");
        appendLog(tr("Drop an image on the canvas first."));
        return;
    }
    if (type == 1 && u.canvas->posPoints().isEmpty() && !u.canvas->hasBox()) {
        return;
    }
    // The first click lazily loads the model; the segmentation re-runs once
    // the load finishes.
    runSegmentation(true);
}

void SAM3Dialog::onCanvasBox(QRectF box) {
    ImageTabUi& u = currentUi();
    // Auto-segment on box drawn
    if (u.currentImage.isNull()) {
        CVLog::Warning("[qSAM3] canvas box on tab with no image");
        appendLog(tr("Drop an image on the canvas first."));
        return;
    }
    if (currentPcsMode()) {
        // Exemplar (PCS): the drawn box becomes a positive exemplar; the
        // actual search runs when the user presses Segment, matching the
        // upstream main_image.cpp behavior (drag: exemplar box, then
        // Segment runs PCS with text + exemplars). No model is needed to
        // collect the exemplar.
        u.posExemplars.append(box);
        u.canvas->setExemplars(u.posExemplars);
        // Drop pending PVS point prompts, but keep the box prompt slots:
        // upstream stores pos_exemplars and pvs_box independently, so a box
        // drawn in Box (PVS) mode survives exemplar collection and still
        // serves as the fallback exemplar for the next Segment.
        u.canvas->clearPointsKeepBox();
        appendLog(tr("Added positive exemplar box (%1). Press Segment to "
                     "run detection (text is optional).")
                          .arg(u.posExemplars.size()));
        return;
    }
    // The first box lazily loads the model; the segmentation re-runs once
    // the load finishes.
    runSegmentation(true);
}

void SAM3Dialog::onTabChanged(int index) {
    if (index >= 3) {
        // Image and video contexts can each consume multiple GiB. Keep only
        // the active domain resident so opening Video cannot fail while an
        // image model and its cached graph still occupy the same GPU.
        stopWorker();
        m_modelPath.clear();
        m_backendLabel->setText(tr("Backend: none"));
        return;
    }
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_videoTab) m_videoTab->releaseModel();
#endif
    // The active tab defines the interaction mode; refresh the hint text.
    updateStatus(tr("Mode: %1")
                         .arg(currentPcsMode() ? tr("Exemplar (PCS)")
                                               : tr("Points / Box (PVS)")));
    // Keep the current tab's canvas / buttons consistent with its image
    // state (each tab owns its own image, so the segment button state and
    // interactivity must follow the tab switch).
    updateSegmentButtonState();
    if (m_worker && m_worker->context() && modelSelectionChanged()) {
        appendLog(tr("Switched to %1 model family. The new model will load "
                     "automatically on the next Segment / click / box.")
                          .arg(m_tabs->tabText(m_tabs->currentIndex())));
    }
}

void SAM3Dialog::onModeChanged() {
    ImageTabUi& u = currentUi();
    // The text prompt works in every mode of the Full tab (text + tracking):
    // typing a prompt and pressing Segment runs PCS even in Points / Box
    // mode. Only visual-only models (no text encoder) disable the field.
    const bool textEnabled = (currentTab() == Sam3Tab::Full) && !m_visualOnly;
    if (u.textPrompt) u.textPrompt->setEnabled(textEnabled);
    // Keep the Segment button visible in every mode: in Points / Box it runs
    // the current prompt, in Exemplar it triggers the text-based search.
    if (u.segmentBtn) u.segmentBtn->setVisible(true);
    // Update help text
    if (currentPointsRadio()->isChecked())
        updateStatus(
                tr("Left-click: +point | Right-click: -point | Drag: bounding "
                   "box"));
    else if (currentBoxRadio()->isChecked())
        updateStatus(
                tr("Drag: bounding box (PVS) | Left-click: +point | "
                   "Right-click: -point"));
    else
        updateStatus(tr("Drag: exemplar box | Type text and press Segment"));

    // Exemplar mode draws green exemplar rects and hides the cyan confirmed
    // box, mirroring upstream main_image.cpp pos_exemplars.
    u.canvas->setExemplarMode(currentPcsMode());
    u.canvas->setExemplars(u.posExemplars);
    u.canvas->updateOverlay();
}

void SAM3Dialog::onDeviceChanged(int idx) {
    const QStringList devNames = {"auto", "cpu", "cuda", "vulkan"};
    m_settings.device =
            (idx >= 0 && idx < devNames.size()) ? devNames[idx] : "auto";
    // Hot-swap: with a model loaded, re-load it on the newly selected
    // backend right away (upstream main_image.cpp Devices combo triggers
    // reload_model() on the fly). loadSettings() also fires this slot
    // during construction, when m_worker is still null.
    if (m_worker && m_worker->context() && !m_modelPath.isEmpty()) {
        if (isRunning()) {
            m_reloadAfterCurrentTask = true;
            appendLog(tr("Device changed to %1; re-loading when the "
                         "current task finishes.")
                              .arg(m_settings.device));
            return;
        }
        appendLog(tr("Device changed to %1 - reloading model...")
                          .arg(m_settings.device));
        setBusy(true);
        startWorker(SAM3WorkerAction::LoadModel);
    } else {
        appendLog(tr("Device set to %1 (load a model to apply).")
                          .arg(m_settings.device));
    }
}

// ── Private helpers ────────────────────────────────────────────────────────

void SAM3Dialog::startWorker(SAM3WorkerAction action) {
    // The worker is shared, but every callback must land on the tab that
    // started the task (a Ctrl+Tab during a busy run must not write a
    // sibling tab's UI/state).
    m_taskTab = m_tabs ? qBound(0, m_tabs->currentIndex(), 2) : 0;
    stopWorker();
    ImageTabUi& u = m_tabsUi[m_taskTab];
    SAM3Worker::Settings s;
    s.modelPath = m_modelPath;
    s.device = m_settings.device;
    s.threads = m_settings.threads;
    s.encodeImgSize = m_settings.encodeImgSize;
    s.scoreThreshold = u.scoreSpin ? static_cast<float>(u.scoreSpin->value())
                                   : m_settings.scoreThreshold;
    s.nmsThreshold = m_settings.nmsThreshold;

    m_worker = new SAM3Worker(s, this);
    if (action == SAM3WorkerAction::LoadModel) {
        m_backendLabel->setText(tr("Backend: loading..."));
    }
    m_worker->setAction(action);
    m_worker->setImage(u.currentImage);
    connect(m_worker, &SAM3Worker::logMessage, this, &SAM3Dialog::onWorkerLog);
    connect(m_worker, &SAM3Worker::resultReady, this,
            &SAM3Dialog::onWorkerResult);
    connect(m_worker, &SAM3Worker::modelReady, this,
            [this](const QString& backend, int, bool vis) {
                m_backendLabel->setText(QString("Backend: %1").arg(backend));
                m_visualOnly = vis;
                // Text prompt / Exemplar only make sense on the Full tab and
                // only for models that carry the text encoder.
                ImageTabUi& u = m_tabsUi[m_taskTab];
                const bool showText = (m_taskTab == 0) && !vis;
                if (u.modeExemplar) u.modeExemplar->setVisible(showText);
                if (u.textPrompt) u.textPrompt->setVisible(showText);
                u.canvas->setInteractive(!u.currentImage.isNull());
                appendLog(tr("Model loaded successfully on %1.").arg(backend));
                if (vis) {
                    appendLog(
                            tr("Model is visual-only; use point / box "
                               "interaction."));
                }
                if (u.currentImage.isNull()) {
                    appendLog(
                            tr("Now drop an image on the canvas (or load "
                               "test data) to start."));
                } else {
                    appendLog(
                            tr("Click on an object or drag a box to "
                               "segment it."));
                }
                updateSegmentButtonState();
            });
    connect(m_worker, &SAM3Worker::finished, this, [this]() {
        setBusy(false);
        if (m_reloadAfterCurrentTask && !m_modelPath.isEmpty()) {
            m_reloadAfterCurrentTask = false;
            appendLog(tr("Re-loading model on %1...").arg(m_settings.device));
            setBusy(true);
            startWorker(SAM3WorkerAction::LoadModel);
            return;
        }
        // A load task that never built a context means the model
        // file failed to load — surface it prominently instead of
        // leaving the Segment button silently disabled.
        if (m_worker && !m_worker->context() &&
            m_worker->action() == SAM3WorkerAction::LoadModel) {
            m_retryAfterModelLoad = false;  // drop pending operation
            m_backendLabel->setText(tr("Backend: none"));
            QMessageBox::warning(
                    this, tr("qSAM3"),
                    tr("Failed to load the model.\n\n"
                       "Check that the GGUF file exists and is "
                       "valid, or pick another model from the list "
                       "(the log above shows the details)."));
        } else if (m_retryAfterModelLoad && m_worker && m_worker->context()) {
            // Lazy load finished: re-run the segmentation the user
            // triggered before the model was ready. Switch back to
            // the tab that started the task so the operation runs
            // against the right image / prompt.
            const bool canvasPrompt = m_retryCanvasPrompt;
            m_retryAfterModelLoad = false;
            m_retryCanvasPrompt = false;
            if (m_tabs && m_tabs->currentIndex() != m_taskTab) {
                m_tabs->setCurrentIndex(m_taskTab);
            }
            runSegmentation(canvasPrompt);
        }
    });
    m_worker->start();
}

void SAM3Dialog::stopWorker() {
    if (m_worker) {
        m_worker->requestCancel();
        // Segmentation on CPU can take well over 5 s; give the worker a
        // generous window before we fall back to deleting it (deleting a
        // still-running QThread aborts the whole process).
        m_worker->wait(30000);
        delete m_worker;
        m_worker = nullptr;
    }
}

bool SAM3Dialog::autoLoadModelIfAvailable() {
    if (isRunning()) return false;  // a task is running
    const QString path = modelPath();
    if (path.isEmpty() || path == "__browse__" || !QFileInfo::exists(path)) {
        return false;
    }
    // The same model is already loaded — nothing to do.
    if (m_worker && m_worker->context() &&
        QFileInfo(path).fileName() == QFileInfo(m_modelPath).fileName()) {
        return false;
    }
    m_modelPath = path;
    m_settings.device = m_deviceCombo->currentText().toLower();
    appendLog(tr("Auto-loading model: %1 ...").arg(QFileInfo(path).fileName()));
    setBusy(true);
    startWorker(SAM3WorkerAction::LoadModel);
    return true;
}

bool SAM3Dialog::ensureModelReady() {
    // Already loaded and matches the current combo selection.
    if (m_worker && m_worker->context() && !modelSelectionChanged()) {
        return true;
    }
    if (isRunning()) {
        appendLog(tr("Worker is busy; wait for the current task to finish."));
        return false;
    }
    if (autoLoadModelIfAvailable()) {
        // The lazy load runs in the worker; the finished handler re-runs
        // the pending operation once the model is ready.
        m_retryAfterModelLoad = true;
        return false;
    }
    // The combo's GGUF file does not exist locally; offer to download it
    // once per session (qDA3-style).
    const QString path = modelPath();
    if (!m_downloadPrompted) {
        const auto answer = QMessageBox::question(
                this, tr("qSAM3"),
                tr("Model file not found:\n%1\n\n"
                   "Download it now into the model cache?")
                        .arg(path),
                QMessageBox::Yes | QMessageBox::No);
        m_downloadPrompted = true;
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

bool SAM3Dialog::modelSelectionChanged() const {
    if (!m_worker || !m_worker->context()) return true;
    const QString path = modelPath();
    if (path.isEmpty() || path == "__browse__") return false;
    return QFileInfo(path).absoluteFilePath() !=
           QFileInfo(m_modelPath).absoluteFilePath();
}

void SAM3Dialog::setBusy(bool busy) {
    m_busy = busy;
    for (ImageTabUi& u : m_tabsUi) {
        if (u.loadBtn) u.loadBtn->setEnabled(!busy);
        if (u.modelCombo) u.modelCombo->setEnabled(!busy);
        if (u.testDataBtn) u.testDataBtn->setEnabled(!busy);
        if (u.testDataCombo) u.testDataCombo->setEnabled(!busy);
    }
    m_deviceCombo->setEnabled(!busy);
    updateDownloadButtons();
    // Canvas interactivity and the Segment button are kept in sync by
    // updateSegmentButtonState(); the model loads lazily on first use.
    updateSegmentButtonState();
    if (busy) {
        updateStatus(
                tr("Working... please wait (segmentation may take "
                   "several seconds)"));
    }
    // Spinner overlay covering the whole dialog.
    if (m_busyOverlay) {
        m_busyOverlay->setVisible(busy);
        m_busyOverlay->raise();
        if (busy) {
            m_busyFrame = 0;
            m_busyOverlay->setText(tr("Working..."));
            m_busyTimer->start();
        } else {
            m_busyTimer->stop();
        }
    }
}

void SAM3Dialog::updateSegmentButtonState() {
    ImageTabUi& u = currentUi();
    // The model loads lazily on first use, so the Segment button only needs
    // an image and a non-busy worker; the first click auto-loads the model.
    const bool canRun = !m_busy && !u.currentImage.isNull();
    if (u.segmentBtn) u.segmentBtn->setEnabled(canRun);
    if (u.canvas) {
        u.canvas->setInteractive(canRun);
    }
    if (!canRun && !m_busy && u.currentImage.isNull()) {
        updateStatus(tr("Drop an image on the canvas to start."));
    }
}

void SAM3Dialog::updateCanvasFromResult(ImageTabUi* target) {
    ImageTabUi& u = target ? *target : currentUi();
    // The mask overlay is a layer owned by the canvas; updateOverlay()
    // composites image → mask → annotations in one pass so the mask is
    // never clobbered by the annotation redraw.
    if (u.lastResult.valid && !u.lastResult.maskComposite.isNull() &&
        u.showMasks->isChecked()) {
        u.canvas->setMaskOverlay(u.lastResult.maskComposite);
    } else {
        u.canvas->clearMaskOverlay();
    }
    // Detection boxes + labels, mirroring upstream main_image.cpp.
    static const char* kColors[] = {
            "#ff3333", "#3399ff", "#33e64c", "#ffcc1a", "#cc4ce6",
            "#ff801a", "#1ae6e6", "#e66699", "#80cc33", "#4c4cff",
    };
    const int nColors = sizeof(kColors) / sizeof(kColors[0]);
    QVector<SAM3DetBox> dets;
    if (u.lastResult.valid) {
        for (int i = 0; i < u.lastResult.detCount; ++i) {
            SAM3DetBox det;
            const aicore_sam3_box& b = u.lastResult.boxes.value(i);
            det.box = QRectF(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
            det.instanceId = u.lastResult.instanceIds.value(i);
            det.score = u.lastResult.scores.value(i);
            det.color = QColor(kColors[i % nColors]);
            dets.append(det);
        }
    }
    u.canvas->setDetections(dets);
    u.canvas->updateOverlay();
}

void SAM3Dialog::updateDetectionList(ImageTabUi* target) {
    ImageTabUi& u = target ? *target : currentUi();
    // Keep the bottom-left counter in sync with the current result.
    u.detLabel->setText(
            tr("Detections: %1 %2")
                    .arg(u.lastResult.valid ? u.lastResult.detCount : 0)
                    .arg(u.lastResult.detCount == 1 ? tr("instance")
                                                    : tr("instances")));
    if (!u.lastResult.valid || u.lastResult.detCount <= 0) {
        u.detectionLabel->clear();
        return;
    }
    // Spell out what the numbers mean: id = detected object index,
    // score = confidence.
    QString html =
            tr("<span style='color:#888;'>Detected objects "
               "(id — confidence):</span> ");
    static const char* kColors[] = {
            "#ff3333", "#3399ff", "#33e64c", "#ffcc1a", "#cc4ce6",
            "#ff801a", "#1ae6e6", "#e66699", "#80cc33", "#4c4cff",
    };
    const int nColors = sizeof(kColors) / sizeof(kColors[0]);

    for (int i = 0; i < u.lastResult.detCount; ++i) {
        const QString color = kColors[i % nColors];
        html += QString("<span style='color:%1; font-weight:bold;'>"
                        "object #%2 — score %3</span>&nbsp; ")
                        .arg(color)
                        .arg(u.lastResult.instanceIds.value(i))
                        .arg(u.lastResult.scores.value(i), 0, 'f', 2);
    }
    u.detectionLabel->setText(html);
}

void SAM3Dialog::updateStatus(const QString& msg) {
    ImageTabUi& u = currentUi();
    if (u.statusLabel) u.statusLabel->setText(msg);
}

void SAM3Dialog::exportToDb(ImageTabUi* target) {
    ImageTabUi& u = target ? *target : currentUi();
    // Build the annotated image: current image with mask overlay.
    if (!u.lastResult.valid || !m_app) return;

    QImage annotated = u.currentImage;
    if (!u.lastResult.maskComposite.isNull()) {
        // Composite the mask onto the image for a visually useful result.
        QPainter p(&annotated);
        // The mask is a single-channel rgba overlay; we draw it atop.
        const QImage maskRgba = u.lastResult.maskComposite.convertToFormat(
                QImage::Format_RGBA8888_Premultiplied);
        p.drawImage(0, 0, maskRgba);
        p.end();
    }

    const QString deviceTag =
            ecvPluginDbNaming::deviceTagFromName(m_settings.device);
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(modelPath());
    const QString sourceLabel =
            u.currentImagePath.isEmpty()
                    ? QStringLiteral("canvas")
                    : QFileInfo(u.currentImagePath).completeBaseName();
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3_%1_%2_%3")
                    .arg(modelTag, sourceLabel, deviceTag),
            m_app);

    auto* img = new ccImage(annotated, name);
    img->setMetaData(QStringLiteral("SAM3"), true);
    img->setMetaData(QStringLiteral("SAM3/DetectionCount"),
                     static_cast<qlonglong>(u.lastResult.detCount));
    img->setMetaData(QStringLiteral("SAM3/Device"), m_settings.device);
    img->setMetaData(QStringLiteral("SAM3/Model"),
                     QFileInfo(m_modelPath).fileName());
    img->setMetaData(QStringLiteral("Runtime (ms)"),
                     u.lastResult.timings.e2e_ms);
    if (!u.currentImagePath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), u.currentImagePath);
    }

    // Serialise per-instance metadata.
    for (int i = 0; i < u.lastResult.detCount; ++i) {
        const QString p = QStringLiteral("SAM3/Det%1/").arg(i + 1);
        const aicore_sam3_box& b = u.lastResult.boxes.value(i);
        img->setMetaData(
                p + QStringLiteral("InstanceId"),
                static_cast<qlonglong>(u.lastResult.instanceIds.value(i)));
        img->setMetaData(p + QStringLiteral("Score"),
                         static_cast<double>(u.lastResult.scores.value(i)));
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
    appendLog(tr("Added '%1' to DB tree.").arg(name));
}

// ── Tab helpers ────────────────────────────────────────────────────────────

SAM3Dialog::Sam3Tab SAM3Dialog::currentTab() const {
    const int idx = m_tabs ? m_tabs->currentIndex() : 0;
    if (idx <= 0) return Sam3Tab::Full;
    if (idx == 1) return Sam3Tab::Visual;
    return Sam3Tab::Sam2;
}

QComboBox* SAM3Dialog::currentModelCombo() const {
    return currentUi().modelCombo;
}

QComboBox* SAM3Dialog::currentTestDataCombo() const {
    return currentUi().testDataCombo;
}

bool SAM3Dialog::currentPcsMode() const {
    // PCS (text / exemplar) only exists on the Full tab.
    const ImageTabUi& u = currentUi();
    return currentTab() == Sam3Tab::Full && u.modeExemplar &&
           u.modeExemplar->isChecked();
}

QRadioButton* SAM3Dialog::currentPointsRadio() const {
    return currentUi().modePoints;
}

QRadioButton* SAM3Dialog::currentBoxRadio() const {
    return currentUi().modeBox;
}

// ── Test data (shared ecvTestDataRepository, SAM3 dataset) ──────────────

void SAM3Dialog::requestTestData() {
    if (m_testDataDownloadInProgress) {
        appendLog(tr("[Test data] Download already in progress."));
        return;
    }
    if (isRunning()) {
        appendLog(tr("[Test data] Wait for the current task to finish."));
        return;
    }
    // Remember which tab asked for the data; the auto-load after extraction
    // lands there even if the user switches tabs while downloading.
    if (m_tabs) m_testDataTab = qBound(0, m_tabs->currentIndex(), 2);
    if (loadRequestedTestData()) return;

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
        updateStatus(tr("Extracting SAM3 test data..."));
        repo.extractDataset(kind);
        return;
    }
    appendLog(tr("[Test data] Downloading SAM3 test data..."));
    updateStatus(tr("Downloading SAM3 test data..."));
    repo.startDownload(kind);
}

bool SAM3Dialog::loadRequestedTestData() {
    return loadTestImageInto(currentUi());
}

bool SAM3Dialog::loadTestImageInto(ImageTabUi& u) {
    const auto kind = ecvTestDataRepository::Dataset::SAM3;
    if (!u.testDataCombo) return false;
    const QString fileName = u.testDataCombo->currentData().toString();
    if (fileName.isEmpty()) return false;

    const QString path = ecvTestDataRepository::findDatasetFile(kind, fileName);
    if (path.isEmpty()) {
        CVLog::Warning("[qSAM3] test image not found in dataset: %s",
                       fileName.toUtf8().constData());
        return false;
    }

    QImage img(path);
    if (img.isNull()) {
        CVLog::Warning("[qSAM3] test image decode failed: %s",
                       path.toUtf8().constData());
        appendLog(
                tr("[Test data] Failed to decode sample image: %1").arg(path));
        return true;  // cached file exists but is unusable; don't re-download
    }
    u.currentImage = img;
    u.currentImagePath = path;
    u.canvas->setImage(img);
    u.canvas->clearPoints();
    u.canvas->clearExemplars();
    u.posExemplars.clear();
    u.lastResult = SAM3WorkerResult{};
    u.detectionLabel->clear();
    u.detLabel->setText(tr("Detections: 0 instances"));
    appendLog(tr("[Test data] Loaded sample image: %1").arg(path));
    appendLog(
            tr("Click on an object to segment it, or drag a box around "
               "it."));
    if (!m_worker || !m_worker->context()) {
        appendLog(
                tr("The selected model loads automatically on the first "
                   "Segment / click / box."));
    }
    updateSegmentButtonState();
    return true;
}

void SAM3Dialog::populateTestDataCombos() {
    const QStringList images = ecvTestDataRepository::getSamImages(
            ecvTestDataRepository::extractPath(
                    ecvTestDataRepository::Dataset::SAM3));
    for (ImageTabUi& u : m_tabsUi) {
        if (!u.testDataCombo) continue;
        u.testDataCombo->blockSignals(true);
        u.testDataCombo->clear();
        if (images.isEmpty()) {
            u.testDataCombo->addItem(tr("(no test data)"), QString());
        } else {
            for (const QString& path : images) {
                const QString name = QFileInfo(path).fileName();
                u.testDataCombo->addItem(name, name);
            }
        }
        u.testDataCombo->blockSignals(false);
    }
}

void SAM3Dialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
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
        updateStatus(tr("Ready."));
        return;
    }
    appendLog(tr("[Test data] Extracting..."));
    updateStatus(tr("Extracting SAM3 test data..."));
    if (m_progress) m_progress->setValue(0);
    ecvTestDataRepository::instance().extractDataset(kind);
}

void SAM3Dialog::onTestDataExtractionFinished(
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
        updateStatus(tr("Ready."));
        return;
    }
    // The pickers were empty before the first extract; fill them and load
    // the (first) sample so the one-click flow completes automatically on
    // the tab that requested the download.
    populateTestDataCombos();
    if (m_tabs && m_tabs->currentIndex() != m_testDataTab) {
        m_tabs->setCurrentIndex(m_testDataTab);
    }
    if (!loadRequestedTestData()) {
        appendLog(tr("[Test data] Sample image not found in the archive."));
    }
    updateStatus(tr("Ready."));
}

void SAM3Dialog::setTestDataControlsEnabled(bool enabled) {
    for (ImageTabUi& u : m_tabsUi) {
        if (u.testDataBtn) u.testDataBtn->setEnabled(enabled);
        if (u.testDataCombo) u.testDataCombo->setEnabled(enabled);
    }
}
