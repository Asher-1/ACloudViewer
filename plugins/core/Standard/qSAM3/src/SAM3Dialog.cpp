// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SAM3Dialog.h"

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

namespace {

// SAM3 test data lives in the shared ecvTestDataRepository SAM3 dataset
// (images/ + videos/ under ~/cloudViewer_data/extract/sam_test_data).

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
    setText(tr("Step 1: drop an image here (or load test data)\n"
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

void SAM3Canvas::setBox(const QRectF& box) {
    m_box = box;
    m_hasBox = true;
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
        m_box = QRectF(m_dragStart, ip).normalized();
        m_hasBox = true;
        updateOverlay();
        emit boxDrawn();
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

    // Confirmed box (cyan)
    if (m_hasBox) {
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
                          det.box.width() * scaleX,
                          det.box.height() * scaleY));
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
    setWindowTitle(tr("SAM3 Image & Video Segmentation"));
    setMinimumSize(ecvAICoreUi::dpiScaled(900), ecvAICoreUi::dpiScaled(700));
    setupUi();
    loadSettings();
}

void SAM3Dialog::showEvent(QShowEvent* e) {
    QDialog::showEvent(e);
    if (m_firstShow) {
        m_firstShow = false;
        adjustSize();
        // No eager model load on dialog open: the initial model (or a model
        // switched later) loads lazily on the first Segment / click / box.
        updateStatus(tr("Ready. The selected model loads automatically on "
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
}

void SAM3Dialog::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSizeConstraint(QLayout::SetNoConstraint);
    ecvAICoreUi::setupTabLayout(mainLayout);

    // ── Tabs: one per model family ────────────────────────────────────────
    m_tabs = new QTabWidget(this);
    ecvAICoreUi::styleTabWidget(m_tabs);

    // ── Tab 1: SAM 3 Full (ViT + text detector) ──────────────────────────
    auto* fullTab = new QWidget();
    auto* fullLayout = new QVBoxLayout(fullTab);
    ecvAICoreUi::setupTabLayout(fullLayout);

    auto* fullRow1 = new QHBoxLayout();
    auto* promptLabel = new QLabel(tr("Text prompt:"));
    m_textPrompt = new QLineEdit();
    m_textPrompt->setPlaceholderText(
            tr("Describe the object to segment (e.g. person, car)..."));
    m_textPrompt->setMinimumWidth(ecvAICoreUi::dpiScaled(240));

    m_segmentBtn = new QPushButton(tr("Segment"));
    m_segmentBtn->setEnabled(false);
    m_segmentBtn->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold;"
            "  border: none; border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background: #00796b; }"
            "QPushButton:disabled { background: #555; color: #999; }");

    m_testDataCombo = new QComboBox();
    m_testDataCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(140));
    m_testDataCombo->setToolTip(
            tr("Pick which sample image to load (SAM3 test dataset)"));

    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataBtn->setToolTip(
            tr("Download (cached) and load the selected sample image for "
               "one-click testing"));

    fullRow1->addWidget(promptLabel);
    fullRow1->addWidget(m_textPrompt, 1);
    fullRow1->addWidget(m_segmentBtn);
    fullRow1->addWidget(m_testDataCombo);
    fullRow1->addWidget(m_testDataBtn);
    fullLayout->addLayout(fullRow1);

    auto* fullRow2 = new QHBoxLayout();
    auto* modeLabel = new QLabel(tr("Mode:"));
    m_modePoints = new QRadioButton(tr("Points"));
    m_modePoints->setChecked(true);
    m_modeBox = new QRadioButton(tr("Box (PVS)"));
    m_modeExemplar = new QRadioButton(tr("Exemplar (PCS)"));

    auto* fullModeGroup = new QButtonGroup(this);
    fullModeGroup->addButton(m_modePoints, 0);
    fullModeGroup->addButton(m_modeBox, 1);
    fullModeGroup->addButton(m_modeExemplar, 2);
    connect(fullModeGroup, QOverload<int>::of(&QButtonGroup::buttonClicked),
            this, &SAM3Dialog::onModeChanged);

    fullRow2->addWidget(modeLabel);
    fullRow2->addWidget(m_modePoints);
    fullRow2->addWidget(m_modeBox);
    fullRow2->addWidget(m_modeExemplar);
    fullRow2->addSpacing(20);

    auto* modelLabel = new QLabel(tr("Model:"));
    m_modelCombo = new QComboBox();
    m_modelCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(240));
    m_loadBtn = new QPushButton(tr("Load"));
    m_loadBtn->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold;"
            "  border: none; border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background: #00796b; }");

    fullRow2->addWidget(modelLabel);
    fullRow2->addWidget(m_modelCombo, 1);
    fullRow2->addWidget(m_loadBtn);
    fullLayout->addLayout(fullRow2);
    m_tabs->addTab(fullTab, tr("SAM 3 Full"));

    // ── Tab 2 & 3: visual-only families (no text encoder) ────────────────
    auto makeVisualTab = [this](const QString& title, QRadioButton*& points,
                                QRadioButton*& box, QComboBox*& combo,
                                QPushButton*& loadBtn,
                                QComboBox*& testDataCombo,
                                QPushButton*& testDataBtn) {
        auto* tab = new QWidget();
        auto* layout = new QVBoxLayout(tab);
        ecvAICoreUi::setupTabLayout(layout);

        auto* row1 = new QHBoxLayout();
        auto* modeLabel = new QLabel(tr("Mode:"));
        points = new QRadioButton(tr("Points"));
        points->setChecked(true);
        box = new QRadioButton(tr("Box (PVS)"));
        auto* group = new QButtonGroup(this);
        group->addButton(points, 0);
        group->addButton(box, 1);
        connect(group, QOverload<int>::of(&QButtonGroup::buttonClicked), this,
                &SAM3Dialog::onModeChanged);

        testDataCombo = new QComboBox();
        testDataCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(140));
        testDataCombo->setToolTip(
                tr("Pick which sample image to load (SAM3 test dataset)"));

        testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
        testDataBtn->setToolTip(
                tr("Download (cached) and load the selected sample image for "
                   "one-click testing"));

        row1->addWidget(modeLabel);
        row1->addWidget(points);
        row1->addWidget(box);
        row1->addStretch();
        row1->addWidget(testDataCombo);
        row1->addWidget(testDataBtn);
        layout->addLayout(row1);

        auto* row2 = new QHBoxLayout();
        auto* modelLabel = new QLabel(tr("Model:"));
        combo = new QComboBox();
        combo->setMinimumWidth(ecvAICoreUi::dpiScaled(240));
        loadBtn = new QPushButton(tr("Load"));
        loadBtn->setStyleSheet(
                "QPushButton { background: #00897b; color: white; font-weight: "
                "bold;"
                "  border: none; border-radius: 4px; padding: 5px 14px; }"
                "QPushButton:hover { background: #00796b; }");
        row2->addWidget(modelLabel);
        row2->addWidget(combo, 1);
        row2->addWidget(loadBtn);
        layout->addLayout(row2);

        m_tabs->addTab(tab, title);
    };
    makeVisualTab(tr("SAM 3 Visual"), m_modePointsV, m_modeBoxV, m_modelComboV,
                  m_loadBtnV, m_testDataComboV, m_testDataBtnV);
    makeVisualTab(tr("SAM 2 / 2.1"), m_modePointsS, m_modeBoxS, m_modelComboS,
                  m_loadBtnS, m_testDataComboS, m_testDataBtnS);

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Video segmentation & tracking (upstream examples/main_video.cpp).
    m_videoTab = new VideoTab();
    if (m_app) {
        m_videoTab->setAppInterface(m_app);
    }
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (m_videoTab) {
                    m_videoTab->setDevice(
                            m_deviceCombo->currentText().toLower());
                }
            });
    m_tabs->addTab(m_videoTab, tr("Video"));
#endif

    mainLayout->addWidget(m_tabs);

    // ── Device row (shared across tabs) ──────────────────────────────────
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

    // ── Canvas (shared across tabs) ────────────────────────────────────────
    m_canvas = new SAM3Canvas();
    m_canvas->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(m_canvas, &SAM3Canvas::pointAdded, this,
            &SAM3Dialog::onCanvasPoint);
    connect(m_canvas, &SAM3Canvas::boxDrawn, this, &SAM3Dialog::onCanvasBox);
    connect(m_canvas, &SAM3Canvas::imageDropped, this,
            [this](const QString& path) {
                QImage img(path);
                if (img.isNull()) return;
                m_currentImage = img;
                m_currentImagePath = path;
                m_encoded = false;
                m_canvas->setImage(img);
                appendLog(
                        tr("Loaded image: %1").arg(QFileInfo(path).fileName()));
                if (m_worker && m_worker->context()) {
                    appendLog(tr("Model ready — click on an object or drag a "
                                 "box to segment it."));
                } else {
                    appendLog(tr("The selected model loads automatically on "
                                 "the first Segment / click / box."));
                }
                updateSegmentButtonState();
            });
    mainLayout->addWidget(m_canvas, 1);

    // ── Bottom panel ───────────────────────────────────────────────────────
    auto* bottomLayout = new QHBoxLayout();

    auto* detLabel = new QLabel(tr("Detections: 0 instances"));
    m_scoreSpin = new QDoubleSpinBox();
    m_scoreSpin->setRange(0.0, 1.0);
    m_scoreSpin->setSingleStep(0.05);
    m_scoreSpin->setValue(0.5);
    m_scoreSpin->setPrefix(tr("Score: "));
    ecvAICoreUi::setCompactDoubleSpin(m_scoreSpin);

    m_showMasks = new QCheckBox(tr("Show masks"));
    m_showMasks->setChecked(true);
    connect(m_showMasks, &QCheckBox::toggled, this, [this](bool) {
        if (m_lastResult.valid) {
            updateCanvasFromResult();
        }
    });

    m_exportToDbCheckBox = new QCheckBox(tr("Export to DB"));
    m_exportToDbCheckBox->setChecked(true);
    m_exportToDbCheckBox->setToolTip(
            tr("Automatically add the segmented result to the DB tree as "
               "an annotated image"));

    m_multimask = new QCheckBox(tr("Multi-mask (PVS)"));

    m_clearBtn = new QPushButton(tr("Clear"));
    m_exportBtn = new QPushButton(tr("Export masks"));

    bottomLayout->addWidget(detLabel);
    bottomLayout->addWidget(m_scoreSpin);
    bottomLayout->addWidget(m_showMasks);
    bottomLayout->addWidget(m_exportToDbCheckBox);
    bottomLayout->addWidget(m_multimask);
    bottomLayout->addSpacing(16);
    bottomLayout->addWidget(m_clearBtn);
    bottomLayout->addWidget(m_exportBtn);

    m_statusLabel = new QLabel(tr("Ready."));
    m_statusLabel->setStyleSheet("color: #99ccff;");
    bottomLayout->addWidget(m_statusLabel, 1);

    mainLayout->addLayout(bottomLayout);

    // ── Detection list ──────────────────────────────────────────────────────
    m_detectionLabel = new QLabel();
    m_detectionLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    mainLayout->addWidget(m_detectionLabel);

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
    connect(m_loadBtn, &QPushButton::clicked, this, &SAM3Dialog::onLoadModel);
    connect(m_loadBtnV, &QPushButton::clicked, this, &SAM3Dialog::onLoadModel);
    connect(m_loadBtnS, &QPushButton::clicked, this, &SAM3Dialog::onLoadModel);
    connect(m_segmentBtn, &QPushButton::clicked, this,
            &SAM3Dialog::onRunSegment);
    // Enter in the text prompt submits, like upstream main_image.cpp
    // (ImGuiInputTextFlags_EnterReturnsTrue).
    connect(m_textPrompt, &QLineEdit::returnPressed, this,
            &SAM3Dialog::onRunSegment);
    connect(m_clearBtn, &QPushButton::clicked, this, &SAM3Dialog::onClear);
    connect(m_exportBtn, &QPushButton::clicked, this,
            &SAM3Dialog::onExportMasks);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &SAM3Dialog::requestTestData);
    connect(m_testDataBtnV, &QPushButton::clicked, this,
            &SAM3Dialog::requestTestData);
    connect(m_testDataBtnS, &QPushButton::clicked, this,
            &SAM3Dialog::requestTestData);
    connect(m_tabs, &QTabWidget::currentChanged, this,
            &SAM3Dialog::onTabChanged);

    // Switching the model does not reload immediately: the new model loads
    // lazily on the next Segment / click / box (user-facing feedback below).
    const auto onModelComboChanged = [this]() {
        if (m_worker && m_worker->context() && modelSelectionChanged()) {
            appendLog(tr("Model changed - the new model will load on the "
                         "next Segment / click / box."));
        }
    };
    connect(m_modelCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            onModelComboChanged);
    connect(m_modelComboV,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            onModelComboChanged);
    connect(m_modelComboS,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            onModelComboChanged);

    // Shared test data repository (ObjectsDetection dataset).
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

    populateModelCombo(m_modelCombo, Sam3Tab::Full);
    populateModelCombo(m_modelComboV, Sam3Tab::Visual);
    populateModelCombo(m_modelComboS, Sam3Tab::Sam2);

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
    // Add all catalog entries for this tab's model family. The catalog already
    // excludes the unpublishable sam3-f32, so every entry (including the
    // sam2/sam2.1 f32 variants) is selectable.
    for (int i = 0; i < n; ++i) {
        const auto* entry = aicore_sam3_model_at(i);
        if (!entry) continue;
        if (!familyMatches(entry->model_family, tab)) continue;
        combo->addItem(QString("%1 (%2)")
                               .arg(entry->display_name)
                               .arg(entry->quant_note),
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
    m_settings.exportToDb = settings.value("exportToDb", true).toBool();

    const int devIdx =
            m_deviceCombo->findText(m_settings.device, Qt::MatchStartsWith);
    if (devIdx >= 0) m_deviceCombo->setCurrentIndex(devIdx);
    m_scoreSpin->setValue(m_settings.scoreThreshold);
    m_exportToDbCheckBox->setChecked(m_settings.exportToDb);

    selectModelByFilename(m_modelCombo, m_settings.modelFull);
    selectModelByFilename(m_modelComboV, m_settings.modelVisual);
    selectModelByFilename(m_modelComboS, m_settings.modelSam2);
}

void SAM3Dialog::saveSettings() {
    QSettings settings("qSAM3");
    settings.setValue("device", m_settings.device);
    settings.setValue("threads", m_settings.threads);
    settings.setValue("scoreThreshold", m_scoreSpin->value());
    settings.setValue("nmsThreshold", m_settings.nmsThreshold);
    settings.setValue("modelFull", m_modelCombo->currentData().toString());
    settings.setValue("modelVisual", m_modelComboV->currentData().toString());
    settings.setValue("modelSam2", m_modelComboS->currentData().toString());
    settings.setValue("exportToDb", m_exportToDbCheckBox->isChecked());
}

QString SAM3Dialog::modelPath() const {
    const QString filename = currentModelCombo()->currentData().toString();
    if (filename.isEmpty() || filename == "__browse__") return QString();
    if (QFileInfo::exists(filename)) return filename;
    // Search common model directories
    const QStringList searchDirs = {
            QDir::homePath() + "/.cache/cloudViewer/models/sam",
            QDir::homePath() + "/develop/code/github/dl/sam3-ggml/models",
    };
    for (const auto& dir : searchDirs) {
        const QString full = dir + "/" + filename;
        if (QFileInfo::exists(full)) return full;
    }
    return filename;
}

void SAM3Dialog::applyDbTreeSelection(const QStringList& names) {
    if (names.isEmpty()) return;
    // For now we just log the selection; image loading is drag-and-drop only
    appendLog(tr("DB selection: %1").arg(names.join(", ")));
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
    m_segmentBtn->setEnabled(false);
    setBusy(true);

    SAM3Worker::Settings workerSettings;
    workerSettings.modelPath = m_modelPath;
    workerSettings.device = m_settings.device;
    workerSettings.threads = m_settings.threads;
    workerSettings.encodeImgSize = m_settings.encodeImgSize;
    workerSettings.scoreThreshold = static_cast<float>(m_scoreSpin->value());
    workerSettings.nmsThreshold = m_settings.nmsThreshold;

    startWorker(SAM3WorkerAction::LoadModel);
}

void SAM3Dialog::onRunSegment() {
    if (m_currentImage.isNull()) {
        appendLog(tr("No image loaded. Drag and drop an image on the canvas."));
        return;
    }
    if (isRunning()) {
        appendLog(tr("Worker is busy; wait for the current task to finish."));
        return;
    }
    if (!ensureModelReady()) {
        // Model not ready: a lazy load is running (the operation re-runs
        // once it finishes) or the GGUF file is missing (already prompted).
        return;
    }

    SAM3Worker::Prompt prompt;

    // A non-empty text prompt runs PCS (text + tracking) in every mode of
    // the Full tab: typing "person" and pressing Segment must segment the
    // person without first switching to Exemplar mode.
    const QString text = m_textPrompt->text().trimmed();
    const bool haveText =
            !text.isEmpty() && currentTab() == Sam3Tab::Full && !m_visualOnly;

    if (currentPcsMode() || haveText) {
        // PCS mode — mirror upstream examples/main_image.cpp: the text
        // prompt is optional; exemplar boxes alone are enough to run PCS
        // (text just adds a guide).
        qstrncpy(prompt.text, text.toUtf8().constData(), sizeof(prompt.text));
        // Exemplar boxes drawn on the canvas (upstream main_image.cpp
        // pos_exemplars): every box is a positive exemplar for PCS.
        for (const auto& b : m_posExemplars) {
            prompt.posExemplars.push_back(
                    {static_cast<float>(b.left()),
                     static_cast<float>(b.top()),
                     static_cast<float>(b.right()),
                     static_cast<float>(b.bottom())});
        }
        // A box drawn in Points / Box mode while a text prompt is present
        // acts as an exemplar (text + region hint) instead of being dropped.
        if (prompt.posExemplars.empty() && m_canvas->hasBox()) {
            const QRectF b = m_canvas->box();
            prompt.posExemplars.push_back(
                    {static_cast<float>(b.left()),
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
        prompt.scoreThreshold = m_settings.scoreThreshold;
        prompt.nmsThreshold = m_settings.nmsThreshold;
        m_worker->setAction(SAM3WorkerAction::EncodeAndSegmentPCS);
    } else {
        // PVS mode: points or box
        const QVector<QPointF>& posPts = m_canvas->posPoints();
        const QVector<QPointF>& negPts = m_canvas->negPoints();
        if (posPts.isEmpty() && !m_canvas->hasBox()) {
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
        if (m_canvas->hasBox()) {
            const QRectF b = m_canvas->box();
            prompt.pvsBox = {static_cast<float>(b.left()),
                             static_cast<float>(b.top()),
                             static_cast<float>(b.right()),
                             static_cast<float>(b.bottom())};
            prompt.usePvsBox = true;
        }
        prompt.multimask = m_multimask->isChecked();
        m_worker->setAction(SAM3WorkerAction::EncodeAndSegmentPVS);
    }

    setBusy(true);
    m_worker->setImage(m_currentImage);
    m_worker->setPrompt(prompt);
    m_worker->start();
}

void SAM3Dialog::onClear() {
    // Drop any pending lazy-load retry so a stale operation never fires
    // after the canvas / prompt have been cleared.
    m_retryAfterModelLoad = false;
    m_canvas->clearPoints();
    m_canvas->clearMaskOverlay();
    m_canvas->clearDetections();
    m_canvas->updateOverlay();
    m_lastResult = SAM3WorkerResult{};
    m_textPrompt->clear();
    m_posExemplars.clear();
    m_detectionLabel->clear();
    updateStatus(tr("Cleared."));
}

void SAM3Dialog::onExportMasks() {
    // Upstream examples/main_image.cpp exports one PNG per detection
    // (mask_%02d.png); we additionally write the composite overlay.
    if (!m_lastResult.valid || m_lastResult.instanceMasks.isEmpty()) {
        appendLog(tr("No masks to export."));
        return;
    }
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Export masks to directory"), QDir::homePath());
    if (dir.isEmpty()) return;
    int exported = 0;
    for (int i = 0; i < m_lastResult.instanceMasks.size(); ++i) {
        const QString path =
                QStringLiteral("%1/mask_%2.png")
                        .arg(dir)
                        .arg(i, 2, 10, QLatin1Char('0'));
        if (m_lastResult.instanceMasks[i].save(path)) ++exported;
    }
    if (!m_lastResult.maskComposite.isNull() &&
        m_lastResult.maskComposite.save(dir + "/mask_composite.png")) {
        ++exported;
    }
    appendLog(tr("Exported %1 mask(s) to %2").arg(exported).arg(dir));
}

void SAM3Dialog::onWorkerFinished(bool ok) {
    setBusy(false);
    if (!ok) {
        updateStatus(tr("Task failed or cancelled."));
    }
}

void SAM3Dialog::onWorkerProgress(int current, int total) {
    // Not used for now
}

void SAM3Dialog::onWorkerLog(const QString& msg) { updateStatus(msg); }

void SAM3Dialog::onWorkerResult(const SAM3WorkerResult& result) {
    if (!result.valid) {
        updateStatus(result.errorMsg.isEmpty() ? tr("No detections.")
                                               : result.errorMsg);
        return;
    }
    m_lastResult = result;
    updateCanvasFromResult();
    updateDetectionList();

    // Auto-export to DB tree if enabled and we have a valid app interface.
    if (m_exportToDbCheckBox->isChecked() && m_app) {
        exportToDb();
    }

    const auto& t = result.timings;
    updateStatus(QString("Done | pre=%.0f inf=%.0f e2e=%.0f ms | %1 detections")
                         .arg(t.preprocess_ms)
                         .arg(t.inference_ms)
                         .arg(t.e2e_ms)
                         .arg(result.detCount));
}

void SAM3Dialog::onCanvasPoint(int type) {
    // Auto-segment on point addition. A negative-only click has no meaning
    // for the C-API (it needs a positive point or a box), so wait until the
    // user has placed a positive point/box.
    if (m_currentImage.isNull()) {
        appendLog(tr("Drop an image on the canvas first."));
        return;
    }
    if (type == 1 && m_canvas->posPoints().isEmpty() &&
        !m_canvas->hasBox()) {
        return;
    }
    // The first click lazily loads the model; the segmentation re-runs once
    // the load finishes.
    if (!ensureModelReady()) return;
    onRunSegment();
}

void SAM3Dialog::onCanvasBox() {
    // Auto-segment on box drawn
    if (m_currentImage.isNull()) {
        appendLog(tr("Drop an image on the canvas first."));
        return;
    }
    if (currentPcsMode()) {
        // Exemplar (PCS): the drawn box becomes a positive exemplar; the
        // actual search runs when the user presses Segment, matching the
        // upstream main_image.cpp behavior (drag: exemplar box, then
        // Segment runs PCS with text + exemplars). No model is needed to
        // collect the exemplar.
        m_posExemplars.append(m_canvas->box());
        m_canvas->clearPoints();
        m_canvas->setBox(m_posExemplars.last());
        appendLog(tr("Added positive exemplar box (%1). Press Segment to "
                     "run detection (text is optional).")
                          .arg(m_posExemplars.size()));
        return;
    }
    // The first box lazily loads the model; the segmentation re-runs once
    // the load finishes.
    if (!ensureModelReady()) return;
    onRunSegment();
}

void SAM3Dialog::onTabChanged(int index) {
    Q_UNUSED(index);
    // The active tab defines the interaction mode; refresh the hint text.
    updateStatus(tr("Mode: %1")
                         .arg(currentPcsMode() ? tr("Exemplar (PCS)")
                                               : tr("Points / Box (PVS)")));
    if (m_worker && m_worker->context() && modelSelectionChanged()) {
        appendLog(tr("Switched to %1 model family. The new model will load "
                     "automatically on the next Segment / click / box.")
                          .arg(m_tabs->tabText(m_tabs->currentIndex())));
    }
}

void SAM3Dialog::onModeChanged() {
    // The text prompt works in every mode of the Full tab (text + tracking):
    // typing a prompt and pressing Segment runs PCS even in Points / Box
    // mode. Only visual-only models (no text encoder) disable the field.
    const bool textEnabled = (currentTab() == Sam3Tab::Full) && !m_visualOnly;
    m_textPrompt->setEnabled(textEnabled);
    // Keep the Segment button visible in every mode: in Points / Box it runs
    // the current prompt, in Exemplar it triggers the text-based search.
    m_segmentBtn->setVisible(true);
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
    stopWorker();
    SAM3Worker::Settings s;
    s.modelPath = m_modelPath;
    s.device = m_settings.device;
    s.threads = m_settings.threads;
    s.encodeImgSize = m_settings.encodeImgSize;
    s.scoreThreshold = m_settings.scoreThreshold;
    s.nmsThreshold = m_settings.nmsThreshold;

    m_worker = new SAM3Worker(s, this);
    m_worker->setAction(action);
    m_worker->setImage(m_currentImage);
    connect(m_worker, &SAM3Worker::logMessage, this, &SAM3Dialog::onWorkerLog);
    connect(m_worker, &SAM3Worker::resultReady, this,
            &SAM3Dialog::onWorkerResult);
    connect(m_worker, &SAM3Worker::modelReady, this,
            [this](const QString& backend, int, bool vis) {
                m_backendLabel->setText(QString("Backend: %1").arg(backend));
                m_visualOnly = vis;
                // Text prompt / Exemplar only make sense on the Full tab and
                // only for models that carry the text encoder.
                const bool showText = (currentTab() == Sam3Tab::Full) && !vis;
                m_modeExemplar->setVisible(showText);
                m_textPrompt->setVisible(showText);
                m_canvas->setInteractive(!m_currentImage.isNull());
                appendLog(tr("Model loaded successfully on %1.").arg(backend));
                if (vis) {
                    appendLog(
                            tr("Model is visual-only; use point / box "
                               "interaction."));
                }
                if (m_currentImage.isNull()) {
                    appendLog(tr("Now drop an image on the canvas (or load "
                                 "test data) to start."));
                } else {
                    appendLog(tr("Click on an object or drag a box to "
                                 "segment it."));
                }
                updateSegmentButtonState();
            });
    connect(m_worker, &SAM3Worker::finished, this,
            [this]() {
                setBusy(false);
                // A load task that never built a context means the model
                // file failed to load — surface it prominently instead of
                // leaving the Segment button silently disabled.
                if (m_worker && !m_worker->context() &&
                    m_worker->action() == SAM3WorkerAction::LoadModel) {
                    m_retryAfterModelLoad = false;  // drop pending operation
                    QMessageBox::warning(
                            this, tr("qSAM3"),
                            tr("Failed to load the model.\n\n"
                               "Check that the GGUF file exists and is "
                               "valid, or pick another model from the list "
                               "(the log above shows the details)."));
                } else if (m_retryAfterModelLoad && m_worker &&
                           m_worker->context()) {
                    // Lazy load finished: re-run the segmentation the user
                    // triggered before the model was ready.
                    m_retryAfterModelLoad = false;
                    onRunSegment();
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
    appendLog(tr("Auto-loading model: %1 ...")
                      .arg(QFileInfo(path).fileName()));
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
    // The combo's GGUF file does not exist locally; nothing to load. Give
    // visible feedback so the click never appears to be ignored.
    QMessageBox::information(
            this, tr("qSAM3"),
            tr("No model loaded yet.\n\n"
               "Pick a model from the list above and click Load (its GGUF "
               "file must exist locally), or use Browse to select a file."));
    return false;
}

bool SAM3Dialog::modelSelectionChanged() const {
    if (!m_worker || !m_worker->context()) return true;
    const QString path = modelPath();
    if (path.isEmpty() || path == "__browse__") return false;
    return QFileInfo(path).fileName() != QFileInfo(m_modelPath).fileName();
}

void SAM3Dialog::setBusy(bool busy) {
    m_busy = busy;
    m_loadBtn->setEnabled(!busy);
    m_loadBtnV->setEnabled(!busy);
    m_loadBtnS->setEnabled(!busy);
    m_modelCombo->setEnabled(!busy);
    m_modelComboV->setEnabled(!busy);
    m_modelComboS->setEnabled(!busy);
    m_deviceCombo->setEnabled(!busy);
    // Canvas interactivity and the Segment button are kept in sync by
    // updateSegmentButtonState(); the model loads lazily on first use.
    updateSegmentButtonState();
    if (busy) {
        updateStatus(tr("Working... please wait (segmentation may take "
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
    if (!m_segmentBtn) return;
    // The model loads lazily on first use, so the Segment button only needs
    // an image and a non-busy worker; the first click auto-loads the model.
    const bool canRun = !m_busy && !m_currentImage.isNull();
    m_segmentBtn->setEnabled(canRun);
    if (m_canvas) {
        m_canvas->setInteractive(canRun);
    }
    if (!canRun && !m_busy && m_currentImage.isNull()) {
        updateStatus(tr("Drop an image on the canvas to start."));
    }
}

void SAM3Dialog::updateCanvasFromResult() {
    // The mask overlay is a layer owned by the canvas; updateOverlay()
    // composites image → mask → annotations in one pass so the mask is
    // never clobbered by the annotation redraw.
    if (m_lastResult.valid && !m_lastResult.maskComposite.isNull() &&
        m_showMasks->isChecked()) {
        m_canvas->setMaskOverlay(m_lastResult.maskComposite);
    } else {
        m_canvas->clearMaskOverlay();
    }
    // Detection boxes + labels, mirroring upstream main_image.cpp.
    static const char* kColors[] = {
            "#ff3333", "#3399ff", "#33e64c", "#ffcc1a", "#cc4ce6",
            "#ff801a", "#1ae6e6", "#e66699", "#80cc33", "#4c4cff",
    };
    const int nColors = sizeof(kColors) / sizeof(kColors[0]);
    QVector<SAM3DetBox> dets;
    if (m_lastResult.valid) {
        for (int i = 0; i < m_lastResult.detCount; ++i) {
            SAM3DetBox det;
            const aicore_sam3_box& b = m_lastResult.boxes.value(i);
            det.box = QRectF(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
            det.instanceId = m_lastResult.instanceIds.value(i);
            det.score = m_lastResult.scores.value(i);
            det.color = QColor(kColors[i % nColors]);
            dets.append(det);
        }
    }
    m_canvas->setDetections(dets);
    m_canvas->updateOverlay();
}

void SAM3Dialog::updateDetectionList() {
    if (!m_lastResult.valid || m_lastResult.detCount <= 0) {
        m_detectionLabel->clear();
        return;
    }
    QString html;
    static const char* kColors[] = {
            "#ff3333", "#3399ff", "#33e64c", "#ffcc1a", "#cc4ce6",
            "#ff801a", "#1ae6e6", "#e66699", "#80cc33", "#4c4cff",
    };
    const int nColors = sizeof(kColors) / sizeof(kColors[0]);

    for (int i = 0; i < m_lastResult.detCount; ++i) {
        const QString color = kColors[i % nColors];
        html += QString("<span style='color:%1; font-weight:bold;'>"
                        "#%2: %3</span> ")
                        .arg(color)
                        .arg(m_lastResult.instanceIds.value(i))
                        .arg(m_lastResult.scores.value(i), 0, 'f', 2);
    }
    m_detectionLabel->setText(html);
}

void SAM3Dialog::updateStatus(const QString& msg) {
    m_statusLabel->setText(msg);
}

void SAM3Dialog::exportToDb() {
    // Build the annotated image: current image with mask overlay.
    if (!m_lastResult.valid || !m_app) return;

    QImage annotated = m_currentImage;
    if (!m_lastResult.maskComposite.isNull()) {
        // Composite the mask onto the image for a visually useful result.
        QPainter p(&annotated);
        // The mask is a single-channel rgba overlay; we draw it atop.
        const QImage maskRgba = m_lastResult.maskComposite.convertToFormat(
                QImage::Format_RGBA8888_Premultiplied);
        p.drawImage(0, 0, maskRgba);
        p.end();
    }

    const QString deviceTag =
            ecvPluginDbNaming::deviceTagFromName(m_settings.device);
    const QString sourceLabel =
            m_currentImagePath.isEmpty()
                    ? QStringLiteral("canvas")
                    : QFileInfo(m_currentImagePath).completeBaseName();
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3_%1_%2").arg(sourceLabel, deviceTag), m_app);

    auto* img = new ccImage(annotated, name);
    img->setMetaData(QStringLiteral("SAM3"), true);
    img->setMetaData(QStringLiteral("SAM3/DetectionCount"),
                     static_cast<qlonglong>(m_lastResult.detCount));
    img->setMetaData(QStringLiteral("SAM3/Device"), m_settings.device);
    img->setMetaData(QStringLiteral("SAM3/Model"),
                     QFileInfo(m_modelPath).fileName());
    img->setMetaData(QStringLiteral("Runtime (ms)"),
                     m_lastResult.timings.e2e_ms);
    if (!m_currentImagePath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), m_currentImagePath);
    }

    // Serialise per-instance metadata.
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
    switch (currentTab()) {
        case Sam3Tab::Full:
            return m_modelCombo;
        case Sam3Tab::Visual:
            return m_modelComboV;
        case Sam3Tab::Sam2:
            return m_modelComboS;
    }
    return m_modelCombo;
}

QComboBox* SAM3Dialog::currentTestDataCombo() const {
    switch (currentTab()) {
        case Sam3Tab::Full:
            return m_testDataCombo;
        case Sam3Tab::Visual:
            return m_testDataComboV;
        case Sam3Tab::Sam2:
            return m_testDataComboS;
    }
    return m_testDataCombo;
}

bool SAM3Dialog::currentPcsMode() const {
    // PCS (text / exemplar) only exists on the Full tab.
    return currentTab() == Sam3Tab::Full && m_modeExemplar &&
           m_modeExemplar->isChecked();
}

QRadioButton* SAM3Dialog::currentPointsRadio() const {
    switch (currentTab()) {
        case Sam3Tab::Full:
            return m_modePoints;
        case Sam3Tab::Visual:
            return m_modePointsV;
        case Sam3Tab::Sam2:
            return m_modePointsS;
    }
    return m_modePoints;
}

QRadioButton* SAM3Dialog::currentBoxRadio() const {
    switch (currentTab()) {
        case Sam3Tab::Full:
            return m_modeBox;
        case Sam3Tab::Visual:
            return m_modeBoxV;
        case Sam3Tab::Sam2:
            return m_modeBoxS;
    }
    return m_modeBox;
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
    if (ecvTestDataRepository::verifyZipIntegrity(
                ecvTestDataRepository::zipPath(kind), info.expectedMd5,
                info.expectedSize)) {
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
    const auto kind = ecvTestDataRepository::Dataset::SAM3;
    QComboBox* combo = currentTestDataCombo();
    if (!combo) return false;
    const QString fileName = combo->currentData().toString();
    if (fileName.isEmpty()) return false;

    const QString path =
            ecvTestDataRepository::findDatasetFile(kind, fileName);
    if (path.isEmpty()) return false;

    QImage img(path);
    if (img.isNull()) {
        appendLog(
                tr("[Test data] Failed to decode sample image: %1").arg(path));
        return true;  // cached file exists but is unusable; don't re-download
    }
    m_currentImage = img;
    m_currentImagePath = path;
    m_encoded = false;
    m_canvas->setImage(img);
    m_canvas->clearPoints();
    m_lastResult = SAM3WorkerResult{};
    m_detectionLabel->clear();
    appendLog(tr("[Test data] Loaded sample image: %1").arg(path));
    appendLog(tr("Click on an object to segment it, or drag a box around "
                 "it."));
    if (!m_worker || !m_worker->context()) {
        appendLog(tr("The selected model loads automatically on the first "
                     "Segment / click / box."));
    }
    updateSegmentButtonState();
    return true;
}

void SAM3Dialog::populateTestDataCombos() {
    const QStringList images = ecvTestDataRepository::getSamImages(
            ecvTestDataRepository::extractPath(
                    ecvTestDataRepository::Dataset::SAM3));
    for (QComboBox* combo : {m_testDataCombo, m_testDataComboV,
                             m_testDataComboS}) {
        if (!combo) continue;
        combo->blockSignals(true);
        combo->clear();
        if (images.isEmpty()) {
            combo->addItem(tr("(no test data)"), QString());
        } else {
            for (const QString& path : images) {
                const QString name = QFileInfo(path).fileName();
                combo->addItem(name, name);
            }
        }
        combo->blockSignals(false);
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
    // the (first) sample so the one-click flow completes automatically.
    populateTestDataCombos();
    if (!loadRequestedTestData()) {
        appendLog(tr("[Test data] Sample image not found in the archive."));
    }
    updateStatus(tr("Ready."));
}

void SAM3Dialog::setTestDataControlsEnabled(bool enabled) {
    m_testDataBtn->setEnabled(enabled);
    m_testDataBtnV->setEnabled(enabled);
    m_testDataBtnS->setEnabled(enabled);
    m_testDataCombo->setEnabled(enabled);
    m_testDataComboV->setEnabled(enabled);
    m_testDataComboS->setEnabled(enabled);
}