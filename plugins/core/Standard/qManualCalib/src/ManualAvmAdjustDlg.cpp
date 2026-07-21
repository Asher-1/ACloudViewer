// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ManualAvmAdjustDlg.h"

#include <CVLog.h>
#include <ecvDisplayTools.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSlider>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "BagLoadHelper.h"
#include "CalibConfigParser.h"
#include "CalibTypes.h"
#include "CameraModel.h"
#include "ProtoDecoder.h"
#include "RosBagReader.h"
#include "mcalib_portability.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <fstream>
#include <sstream>

static inline QWidget* asWidget(QMainWindow* w) { return w; }

static inline float GetRoot(float p, float q) {
    float q2 = q / 2.f;
    float p3 = p / 3.f;
    float delta = std::sqrt(q2 * q2 + p3 * p3 * p3);
    return std::pow(-q2 + delta, 1.f / 3.f) - std::pow(q2 + delta, 1.f / 3.f);
}

ManualAvmAdjustDlg::ManualAvmAdjustDlg(ecvMainAppInterface* app,
                                       QWidget* parent)
    : ccOverlayDialog(
              parent,
              Qt::Tool | Qt::CustomizeWindowHint | Qt::WindowCloseButtonHint),
      m_app(app) {
    setWindowTitle(tr("AVM View Adjustment"));

    m_baseRotMap["rot_front_single"] =
            (Eigen::Matrix3d() << 0, -1, 0, 0, 0, -1, 1, 0, 0).finished();
    m_baseRotMap["rot_back_single"] =
            (Eigen::Matrix3d() << 0, 1, 0, 0, 0, -1, -1, 0, 0).finished();
    m_baseRotMap["rot_down_single"] =
            (Eigen::Matrix3d() << 0, -1, 0, -1, 0, 0, 0, 0, -1).finished();

    setupUI();
}

ManualAvmAdjustDlg::~ManualAvmAdjustDlg() {
    if (m_sliderLoadWatcher) {
        ++m_sliderLoadGeneration;
        m_sliderLoadWatcher->disconnect();
        m_sliderLoadWatcher->waitForFinished();
        delete m_sliderLoadWatcher;
        m_sliderLoadWatcher = nullptr;
    }
    if (m_bagIndexWatcher) {
        m_bagIndexWatcher->cancel();
        m_bagIndexWatcher->waitForFinished();
        delete m_bagIndexWatcher;
        m_bagIndexWatcher = nullptr;
    }
    m_bagReader.reset();
    if (m_vtkImage && m_app) {
        ccHObject* obj = m_vtkImage;
        ecvDisplayTools::RemoveEntities(obj);
        obj->removeFromRenderScreen(true);
        m_app->removeFromDB(obj, true);
        m_vtkImage = nullptr;
    }
}

bool ManualAvmAdjustDlg::linkWith(QWidget* win) {
    return ccOverlayDialog::linkWith(win);
}

bool ManualAvmAdjustDlg::start() { return ccOverlayDialog::start(); }

void ManualAvmAdjustDlg::stop(bool accepted) {
    auto* app = m_app;

    if (m_sliderLoadWatcher) {
        ++m_sliderLoadGeneration;
        m_sliderLoadWatcher->disconnect();
        m_sliderLoadWatcher->waitForFinished();
        delete m_sliderLoadWatcher;
        m_sliderLoadWatcher = nullptr;
    }
    if (m_bagIndexWatcher) {
        m_bagIndexWatcher->cancel();
        m_bagIndexWatcher->waitForFinished();
        delete m_bagIndexWatcher;
        m_bagIndexWatcher = nullptr;
    }
    m_loadingBag.store(false);
    m_sliderReloadPending = false;
    m_bagReader.reset();

    if (m_vtkImage) {
        ccHObject* obj = m_vtkImage;
        ecvDisplayTools::RemoveEntities(obj);
        obj->removeFromRenderScreen(true);
        m_app->removeFromDB(obj, true);
        m_vtkImage = nullptr;
    }

    if (app) {
        app->toggle3DView(true);
        app->refreshAll(true, true);
        app->redrawAll(true, true);
    }

    ccOverlayDialog::stop(accepted);

    QTimer::singleShot(50, [app]() {
        if (!app) return;
        app->toggle3DView(true);
        app->redrawAll(true);
        app->refreshAll(true);
        app->setGlobalZoom();
    });
}

void ManualAvmAdjustDlg::closeEvent(QCloseEvent* event) {
    stop(false);
    event->accept();
}

QWidget* ManualAvmAdjustDlg::createParamRow(const QString& label,
                                            QDoubleSpinBox*& spinBox,
                                            double min,
                                            double max,
                                            double value,
                                            double step,
                                            int decimals) {
    auto* widget = new QWidget;
    auto* layout = new QHBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(3);

    auto* lbl = new QLabel(label);
    lbl->setFixedWidth(70);
    layout->addWidget(lbl);

    auto* slider = new QSlider(Qt::Horizontal);
    int sliderScale =
            (decimals > 0) ? static_cast<int>(std::pow(10, decimals)) : 1;
    slider->setRange(static_cast<int>(min * sliderScale),
                     static_cast<int>(max * sliderScale));
    slider->setValue(static_cast<int>(value * sliderScale));
    slider->setMinimumWidth(100);
    layout->addWidget(slider);

    spinBox = new QDoubleSpinBox;
    spinBox->setRange(min, max);
    spinBox->setValue(value);
    spinBox->setSingleStep(step);
    spinBox->setDecimals(decimals);
    spinBox->setFixedWidth(65);
    layout->addWidget(spinBox);

    connect(slider, &QSlider::valueChanged, [spinBox, sliderScale](int v) {
        spinBox->setValue(static_cast<double>(v) / sliderScale);
    });
    connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [slider, sliderScale](double v) {
                slider->blockSignals(true);
                slider->setValue(static_cast<int>(v * sliderScale));
                slider->blockSignals(false);
            });

    return widget;
}

void ManualAvmAdjustDlg::setupUI() {
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(4, 4, 4, 4);
    rootLayout->setSpacing(3);

    auto* controlScroll = new QScrollArea;
    controlScroll->setWidgetResizable(true);
    controlScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    controlScroll->setFixedWidth(320);
    auto* controlWidget = new QWidget;
    auto* controlLayout = new QVBoxLayout(controlWidget);
    controlLayout->setContentsMargins(2, 2, 2, 2);
    controlLayout->setSpacing(3);
    controlScroll->setWidget(controlWidget);

    // -- Load cfg then bag --
    auto* loadRow = new QHBoxLayout;
    loadRow->setSpacing(2);
    m_btnLoadConfig = new QPushButton(tr("1. load cfg"));
    m_btnLoadBag = new QPushButton(tr("2. load bag"));
    m_btnLoadConfig->setFixedHeight(26);
    m_btnLoadBag->setFixedHeight(26);
    m_btnLoadBag->setEnabled(false);
    m_btnLoadBag->setToolTip(tr("Load cameras.cfg directory first"));
    loadRow->addWidget(m_btnLoadConfig);
    loadRow->addWidget(m_btnLoadBag);
    controlLayout->addLayout(loadRow);

    connect(m_btnLoadConfig, &QPushButton::clicked, this,
            &ManualAvmAdjustDlg::onLoadConfig);
    connect(m_btnLoadBag, &QPushButton::clicked, this,
            &ManualAvmAdjustDlg::onLoadBag);

    // -- Bag time slider (duration on title row, full-width slider below) --
    auto* bagTimeHeader = new QHBoxLayout;
    auto* bagTimeTitle = new QLabel(tr("Bag Time"));
    bagTimeTitle->setStyleSheet("font-weight: bold;");
    m_lblTimePos = new QLabel("0.0%");
    m_lblTimePos->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    bagTimeHeader->addWidget(bagTimeTitle);
    bagTimeHeader->addWidget(m_lblTimePos, 1);
    controlLayout->addLayout(bagTimeHeader);

    auto* timeRow = new QHBoxLayout;
    m_btnTimeStepBack = new QPushButton(QStringLiteral("◀"));
    m_btnTimeStepForward = new QPushButton(QStringLiteral("▶"));
    m_btnTimeStepBack->setFixedSize(28, 24);
    m_btnTimeStepForward->setFixedSize(28, 24);
    m_btnTimeStepBack->setToolTip(tr("Previous frame (~10Hz camera step)"));
    m_btnTimeStepForward->setToolTip(tr("Next frame (~10Hz camera step)"));
    m_sliderTimePos = new QSlider(Qt::Horizontal);
    m_sliderTimePos->setRange(0, 1000);
    m_sliderTimePos->setValue(0);
    m_sliderTimePos->setToolTip(tr("Scrub through bag recording timeline"));
    timeRow->addWidget(m_btnTimeStepBack);
    timeRow->addWidget(m_sliderTimePos, 1);
    timeRow->addWidget(m_btnTimeStepForward);
    controlLayout->addLayout(timeRow);
    connect(m_btnTimeStepBack, &QPushButton::clicked, this,
            &ManualAvmAdjustDlg::onTimeStepBack);
    connect(m_btnTimeStepForward, &QPushButton::clicked, this,
            &ManualAvmAdjustDlg::onTimeStepForward);
    connect(m_sliderTimePos, &QSlider::sliderMoved, this,
            &ManualAvmAdjustDlg::updateTimeSliderLabel);
    connect(m_sliderTimePos, &QSlider::valueChanged, this,
            &ManualAvmAdjustDlg::onTimeSliderChanged);

    // -- Camera + AVM mode --
    auto* modeRow = new QHBoxLayout;
    m_cmbCamera = new QComboBox;
    m_cmbCamera->addItems({mcalib::kPanoramic1, mcalib::kPanoramic2,
                           mcalib::kPanoramic3, mcalib::kPanoramic4});
    m_cmbAvmMode = new QComboBox;
    m_cmbAvmMode->addItems(
            {"small_single_view", "large_single_view", "wheel_hub_view"});
    modeRow->addWidget(m_cmbCamera);
    modeRow->addWidget(m_cmbAvmMode);
    controlLayout->addLayout(modeRow);

    m_cmbBaseRot = new QComboBox;
    m_cmbBaseRot->addItems(
            {"rot_front_single", "rot_back_single", "rot_down_single"});
    controlLayout->addWidget(m_cmbBaseRot);

    connect(m_cmbCamera, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ManualAvmAdjustDlg::onCameraChanged);
    connect(m_cmbAvmMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ManualAvmAdjustDlg::onAvmModeChanged);
    connect(m_cmbBaseRot, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ManualAvmAdjustDlg::onBaseRotChanged);

    // -- Parameters --
    controlLayout->addWidget(createParamRow("virtual_k2", m_spinVirtualK2, -0.5,
                                            0.5, 0.16, 0.01, 2));
    controlLayout->addWidget(createParamRow("v0_offset", m_spinV0Offset, -0.5,
                                            0.5, 0.03, 0.01, 2));
    controlLayout->addWidget(
            createParamRow("scale", m_spinScale, 0.5, 2.0, 1.5, 0.01, 2));
    controlLayout->addWidget(
            createParamRow("img_width", m_spinImgWidth, 500, 2000, 1014, 1, 0));
    controlLayout->addWidget(createParamRow("img_height", m_spinImgHeight, 466,
                                            1466, 966, 1, 0));
    controlLayout->addWidget(
            createParamRow("focal_x", m_spinFocalX, 100, 1500, 400, 1, 0));
    controlLayout->addWidget(
            createParamRow("focal_y", m_spinFocalY, 100, 1500, 400, 1, 0));
    controlLayout->addWidget(
            createParamRow("rot_x", m_spinRotX, -90, 90, 0, 0.1, 1));
    controlLayout->addWidget(
            createParamRow("rot_y", m_spinRotY, -90, 90, 0, 0.1, 1));
    controlLayout->addWidget(
            createParamRow("rot_z", m_spinRotZ, -90, 90, 0, 0.1, 1));
    controlLayout->addWidget(
            createParamRow("rect_x", m_spinRectX, 0, 2000, 0, 1, 0));
    controlLayout->addWidget(
            createParamRow("rect_y", m_spinRectY, 0, 1466, 0, 1, 0));
    controlLayout->addWidget(
            createParamRow("rect_width", m_spinRectWidth, 0, 2000, 1014, 1, 0));
    controlLayout->addWidget(createParamRow("rect_height", m_spinRectHeight, 0,
                                            1466, 966, 1, 0));

    connect(m_spinVirtualK2,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ManualAvmAdjustDlg::onVirtualK2Changed);
    connect(m_spinScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onScaleChanged);
    connect(m_spinV0Offset,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ManualAvmAdjustDlg::onV0OffsetChanged);
    connect(m_spinImgWidth,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ManualAvmAdjustDlg::onImgWidthChanged);
    connect(m_spinImgHeight,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ManualAvmAdjustDlg::onImgHeightChanged);
    connect(m_spinFocalX, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onFocalXChanged);
    connect(m_spinFocalY, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onFocalYChanged);
    connect(m_spinRotX, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onRotXChanged);
    connect(m_spinRotY, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onRotYChanged);
    connect(m_spinRotZ, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onRotZChanged);
    connect(m_spinRectX, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onRectXChanged);
    connect(m_spinRectY, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ManualAvmAdjustDlg::onRectYChanged);
    connect(m_spinRectWidth,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ManualAvmAdjustDlg::onRectWidthChanged);
    connect(m_spinRectHeight,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ManualAvmAdjustDlg::onRectHeightChanged);

    // -- Save / Load / Close --
    auto* actionRow = new QHBoxLayout;
    actionRow->setSpacing(2);
    auto* btnSave = new QPushButton("save_param");
    auto* btnLoad = new QPushButton("load_param");
    btnSave->setFixedHeight(26);
    btnLoad->setFixedHeight(26);
    actionRow->addWidget(btnSave);
    actionRow->addWidget(btnLoad);
    controlLayout->addLayout(actionRow);

    connect(btnSave, &QPushButton::clicked, this,
            &ManualAvmAdjustDlg::onSaveParam);
    connect(btnLoad, &QPushButton::clicked, this,
            &ManualAvmAdjustDlg::onLoadParam);

    m_lblStatus = new QLabel(tr("Step 1: Load cfg (cameras.cfg directory)"));
    m_lblStatus->setStyleSheet("color: #999; font-size: 11px;");
    controlLayout->addWidget(m_lblStatus);
    controlLayout->addStretch();

    rootLayout->addWidget(controlScroll);
}

void ManualAvmAdjustDlg::onLoadConfig() {
    QSettings settings;
    QString lastCfg =
            settings.value("qManualCalib/lastAvmConfigPath", m_configPath)
                    .toString();
    QString dir = QFileDialog::getExistingDirectory(
            asWidget(m_app->getMainWindow()), tr("Select Config Directory"),
            lastCfg);
    if (dir.isEmpty()) return;

    settings.setValue("qManualCalib/lastAvmConfigPath", dir);
    m_configPath = dir;

    m_calibConfig = mcalib::VehicleCalibConfig();

    const std::string camera_cfg = mcalib::pathFromQString(
            QDir(dir).filePath(QStringLiteral("cameras.cfg")));
    if (!mcalib::CalibConfigParser::loadCameraConfig(camera_cfg,
                                                     m_calibConfig)) {
        CVLog::Warning("[AvmAdjust] Failed to load cameras.cfg: %s",
                       camera_cfg.c_str());
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Failed to load config: %1")
                                     .arg(QString::fromStdString(camera_cfg)));
        return;
    }

    const std::string ground_cfg = mcalib::pathFromQString(
            QDir(dir).filePath(QStringLiteral("ground.cfg")));
    mcalib::CalibConfigParser::loadGroundConfig(ground_cfg, m_calibConfig);

    m_cameraExtrinsics.clear();
    mcalib::CalibConfigParser::getCameraExtrinsics(m_calibConfig,
                                                   m_cameraExtrinsics);

    m_cameraSystem.loadFromConfig(m_calibConfig);

    m_currentCamera = m_cmbCamera->currentText().toStdString();
    m_configLoaded = true;
    if (m_btnLoadBag) {
        m_btnLoadBag->setEnabled(true);
        m_btnLoadBag->setToolTip(QString());
    }
    CVLog::Print("[AvmAdjust] Config loaded: %zu cameras, extrinsics for: %zu",
                 m_calibConfig.cameras.size(), m_cameraExtrinsics.size());
    m_lblStatus->setText(tr("Config loaded: %1 cameras — now load bag")
                                 .arg(m_calibConfig.cameras.size()));
    updateImage();
}

void ManualAvmAdjustDlg::onLoadBag() {
    if (!m_configLoaded || m_calibConfig.cameras.empty()) {
        QMessageBox::information(
                asWidget(m_app->getMainWindow()), tr("Load Config First"),
                tr("Please load the config directory (cameras.cfg / "
                   "ground.cfg) before loading bag data."));
        return;
    }

    QSettings settings;
    QString lastBag =
            settings.value("qManualCalib/lastAvmBagPath", m_bagPath).toString();
    const QString selected = mcalib::ui::pickBagInputPath(
            asWidget(m_app->getMainWindow()), lastBag);
    if (selected.isEmpty()) return;

    settings.setValue("qManualCalib/lastAvmBagPath", selected);
    m_bagPath = selected;

    const std::string input_path = mcalib::pathFromQString(selected);
    const mcalib::BagDiscoveryResult discovery =
            mcalib::discoverBagLayout(input_path);
    if (discovery.layout == mcalib::BagLayoutType::Unknown ||
        discovery.sessions.empty()) {
        QMessageBox::warning(
                asWidget(m_app->getMainWindow()), tr("Error"),
                tr("Failed to discover bag layout under:\n%1").arg(selected));
        return;
    }

    const int session_index = mcalib::ui::pickBagSessionIndex(
            asWidget(m_app->getMainWindow()), discovery,
            mcalib::ui::preferredSessionKeyFromInput(input_path));
    if (session_index < 0) return;

    auto* progress =
            new QProgressDialog(tr("Resolving bag input..."), tr("Cancel"), 0,
                                100, asWidget(m_app->getMainWindow()));
    progress->setWindowModality(Qt::WindowModal);
    progress->setMinimumDuration(0);
    progress->setValue(10);
    QCoreApplication::processEvents();

    const mcalib::BagResolveResult resolved =
            mcalib::resolveBagInput(input_path, session_index);
    if (!resolved.ok) {
        progress->close();
        delete progress;
        CVLog::Error("[AvmAdjust] Failed to resolve bag input: %s",
                     resolved.error_message.c_str());
        QMessageBox::warning(
                asWidget(m_app->getMainWindow()), tr("Error"),
                tr("Failed to resolve bag input:\n%1")
                        .arg(QString::fromStdString(resolved.error_message)));
        return;
    }

    m_bagReader = std::make_unique<mcalib::RosBagReader>();

    progress->setLabelText(tr("Opening bag file..."));
    progress->setValue(10);
    QCoreApplication::processEvents();

    if (!mcalib::ui::openResolvedBag(*m_bagReader, resolved)) {
        progress->close();
        delete progress;
        CVLog::Error("[AvmAdjust] Failed to open bag session: %s",
                     resolved.session_key.c_str());
        QMessageBox::warning(
                asWidget(m_app->getMainWindow()), tr("Error"),
                tr("Failed to open bag session: %1")
                        .arg(QString::fromStdString(resolved.session_key)));
        m_bagReader.reset();
        return;
    }

    CVLog::Print("[AvmAdjust] Bag layout: %s, session: %s, source bags: %zu",
                 mcalib::pathFromQString(
                         mcalib::ui::layoutDescription(resolved.layout))
                         .c_str(),
                 resolved.session_key.c_str(), resolved.source_bags.size());

    progress->setLabelText(tr("Scanning topics..."));
    progress->setValue(30);
    QCoreApplication::processEvents();

    m_bagReader->refinePlaybackTimeRange(getCameraTopic());

    double dur_s = m_bagReader->getDuration() / 1e9;
    m_bagDurationSec = dur_s;
    m_bagIndexReady = false;
    updateTimeSliderLabel(m_sliderTimePos->value());
    auto topicTypes = m_bagReader->getTopicTypes();
    int cam_count = 0;
    for (const auto& [topic, type] : topicTypes) {
        if (topic.find("camera") != std::string::npos ||
            topic.find("image") != std::string::npos) {
            CVLog::Print("[AvmAdjust]   camera topic: '%s' [%s]", topic.c_str(),
                         type.c_str());
            ++cam_count;
        }
    }
    CVLog::Print(
            "[AvmAdjust] Bag loaded: %.1fs, %u messages, %zu topics (%d "
            "camera)",
            dur_s, m_bagReader->getMessageCount(), topicTypes.size(),
            cam_count);

    progress->setLabelText(tr("Loading initial image..."));
    progress->setValue(50);
    QCoreApplication::processEvents();

    loadImageFromBag(0.0);

    startBackgroundBagIndex();

    progress->setLabelText(tr("Preparing view..."));
    progress->setValue(90);
    QCoreApplication::processEvents();

    progress->setValue(100);
    progress->close();
    delete progress;

    m_lblStatus->setText(tr("Bag loaded: %1s, %2 msgs")
                                 .arg(dur_s, 0, 'f', 1)
                                 .arg(m_bagReader->getMessageCount()));
    m_appliedSliderValue = 0;
    updateImage();
    QTimer::singleShot(0, this, [this]() {
        updateImage();
        if (m_app) {
            m_app->redrawAll(false, true);
            m_app->refreshAll(false, true);
        }
    });
}

std::string ManualAvmAdjustDlg::getCameraTopic() const {
    return "/sensors/camera/" + m_currentCamera + "_raw_data/compressed_proto";
}

void ManualAvmAdjustDlg::startBackgroundBagIndex() {
    if (!m_bagReader || !m_bagReader->isOpen()) return;

    if (m_bagIndexWatcher) {
        m_bagIndexWatcher->cancel();
        m_bagIndexWatcher->waitForFinished();
        delete m_bagIndexWatcher;
        m_bagIndexWatcher = nullptr;
    }

    const std::string topic = getCameraTopic();
    m_bagIndexReady = false;
    m_bagIndexWatcher = new QFutureWatcher<void>(this);
    connect(m_bagIndexWatcher, &QFutureWatcher<void>::finished, this, [this]() {
        m_bagIndexReady = m_bagReader && m_bagReader->hasTopicTimeIndex();
        CVLog::Print("[AvmAdjust] Background bag time index %s",
                     m_bagIndexReady ? "ready" : "unavailable");
    });

    mcalib::RosBagReader* reader = m_bagReader.get();
    m_bagIndexWatcher->setFuture(QtConcurrent::run([reader, topic]() {
        if (!reader) return;
        reader->buildTopicTimeIndex({topic}, nullptr, 0);
    }));
}

void ManualAvmAdjustDlg::updateTimeSliderLabel(int sliderValue) {
    if (!m_lblTimePos) return;
    const double percent = sliderValue / 1000.0;
    if (m_bagDurationSec > 0.0) {
        const double current_s = m_bagDurationSec * percent;
        m_lblTimePos->setText(QString("%1s / %2s")
                                      .arg(current_s, 0, 'f', 1)
                                      .arg(m_bagDurationSec, 0, 'f', 1));
    } else {
        m_lblTimePos->setText(QString("%1%").arg(percent * 100.0, 0, 'f', 1));
    }
}

void ManualAvmAdjustDlg::loadImageFromBag(double percent) {
    if (!m_bagReader || !m_bagReader->isOpen()) {
        CVLog::Warning("[AvmAdjust] loadImage: bagReader not ready");
        return;
    }

    std::string camera_topic = getCameraTopic();
    CVLog::Print("[AvmAdjust] loadImage: querying '%s' at %.1f%%",
                 camera_topic.c_str(), percent * 100.0);

    auto msg = m_bagReader->readMessageAtPercent(camera_topic, percent);
    if (msg.data.empty()) {
        CVLog::Warning(
                "[AvmAdjust] loadImage: primary topic returned no data, "
                "scanning...");
        auto topics = m_bagReader->getTopics();
        for (const auto& topic : topics) {
            if (topic.find(m_currentCamera) != std::string::npos &&
                topic.find("camera") != std::string::npos) {
                CVLog::Print("[AvmAdjust]   trying fallback: '%s'",
                             topic.c_str());
                msg = m_bagReader->readMessageAtPercent(topic, percent);
                if (!msg.data.empty()) {
                    CVLog::Print("[AvmAdjust]   fallback hit: %zu bytes",
                                 msg.data.size());
                    break;
                }
            }
        }
    } else {
        CVLog::Print("[AvmAdjust] loadImage: got %zu bytes", msg.data.size());
    }

    if (!msg.data.empty()) {
        double ts = 0;
        if (mcalib::ProtoDecoder::decodeCompressedImageFromBag(
                    msg.data, m_sourceImage, ts)) {
            CVLog::Print("[AvmAdjust] loadImage: decoded %dx%d, ts=%.3f",
                         m_sourceImage.cols, m_sourceImage.rows, ts);
        } else {
            CVLog::Warning("[AvmAdjust] loadImage: decode FAILED (%zu bytes)",
                           msg.data.size());
        }
    } else {
        CVLog::Warning("[AvmAdjust] loadImage: no data found for camera '%s'",
                       m_currentCamera.c_str());
    }
}

void ManualAvmAdjustDlg::onLoadImage() {
    QString filename = QFileDialog::getOpenFileName(
            asWidget(m_app->getMainWindow()), tr("Open Image"), QString(),
            tr("Images (*.jpg *.png *.bmp *.tiff)"));
    if (filename.isEmpty()) return;

    m_sourceImage = cv::imread(mcalib::pathFromQString(filename));
    if (m_sourceImage.empty()) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Failed to load image: %1").arg(filename));
        return;
    }
    m_lblStatus->setText(tr("Image loaded: %1x%2")
                                 .arg(m_sourceImage.cols)
                                 .arg(m_sourceImage.rows));
    updateImage();
}

void ManualAvmAdjustDlg::updateImage() {
    if (m_blockUpdate || m_sourceImage.empty()) return;

    cv::Mat img = m_sourceImage.clone();

    auto it_cam = m_calibConfig.cameras.find(m_currentCamera);

    cv::Mat camera_matrix, distor_coeffs;
    cv::Size image_size;
    if (it_cam != m_calibConfig.cameras.end()) {
        camera_matrix = it_cam->second.intrinsic.getCameraMatrix();
        distor_coeffs = it_cam->second.intrinsic.getDistCoeffs();
        image_size = it_cam->second.intrinsic.getImageSize();
    } else {
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        camera_matrix.at<double>(0, 0) = img.cols * 0.5;
        camera_matrix.at<double>(1, 1) = img.rows * 0.5;
        camera_matrix.at<double>(0, 2) = img.cols * 0.5;
        camera_matrix.at<double>(1, 2) = img.rows * 0.5;
        distor_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        image_size = cv::Size(img.cols, img.rows);
    }

    cv::Size target_image_size(m_imgWidth, m_imgHeight);
    float scale = static_cast<float>(m_targetScale) *
                  (target_image_size.width + target_image_size.height) /
                  static_cast<float>(image_size.width + image_size.height);
    const float fx = static_cast<float>(camera_matrix.at<double>(0, 0));
    const float fy = static_cast<float>(camera_matrix.at<double>(1, 1));
    const float cx = static_cast<float>(camera_matrix.at<double>(0, 2));
    const float cy = static_cast<float>(camera_matrix.at<double>(1, 2));

    float target_fx = fx * scale;
    float target_fy = fy * scale;
    float target_cx = target_image_size.width / 2.f;
    float target_cy =
            target_image_size.height * (0.5f + static_cast<float>(m_v0Offset));

    if (m_currentCamera == mcalib::kPanoramic3) {
        target_fx = -target_fx;
    }

    cv::Mat mapxy = cv::Mat::zeros(target_image_size, CV_32FC2);

    if (m_avmMode == "wheel_hub_view") {
        auto cam_model = m_cameraSystem.getCamera(m_currentCamera);
        if (cam_model) {
            Eigen::AngleAxisd R_X(m_rotX * M_PI / 180.0,
                                  Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd R_Y(m_rotY * M_PI / 180.0,
                                  Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd R_Z(m_rotZ * M_PI / 180.0,
                                  Eigen::Vector3d::UnitZ());
            Eigen::Matrix3d r_tc_sensing = R_X.matrix() * R_Y.matrix() *
                                           R_Z.matrix() *
                                           m_baseRotMap[m_baseRot];

            auto it_ext = m_cameraExtrinsics.find(m_currentCamera);
            if (it_ext != m_cameraExtrinsics.end()) {
                Eigen::Matrix3d rot_result =
                        r_tc_sensing * it_ext->second.inverse().linear();
                cv::Mat cv_r_cam_target;
                cv::eigen2cv(rot_result, cv_r_cam_target);
                cv_r_cam_target.convertTo(cv_r_cam_target, CV_32F);

                cv::Mat map1, map2;
                cam_model->initUndistortRectifyMap(
                        map1, map2, static_cast<float>(m_focalX),
                        static_cast<float>(m_focalY), target_image_size,
                        target_image_size.width / 2.f,
                        target_image_size.height / 2.f, cv_r_cam_target);

                if (!map1.empty()) {
                    if (map1.channels() == 2) {
                        mapxy = map1;
                    } else if (!map2.empty()) {
                        std::vector<cv::Mat> map_list = {map1, map2};
                        cv::merge(map_list, mapxy);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < target_image_size.height; i++) {
            for (int j = 0; j < target_image_size.width; j++) {
                if (m_avmMode == "small_single_view") {
                    float y_hat = (i - target_cy) / target_fy;
                    float x_hat = (j - target_cx) / target_fx;
                    float theta =
                            GetRoot(1.0f / static_cast<float>(m_virtualK2),
                                    -cv::sqrt(y_hat * y_hat + x_hat * x_hat) /
                                            static_cast<float>(m_virtualK2));
                    float theta2 = theta * theta;
                    float theta4 = theta2 * theta2;
                    float theta6 = theta4 * theta2;
                    float theta8 = theta6 * theta2;
                    float coeff =
                            (1.f +
                             static_cast<float>(distor_coeffs.at<double>(0)) *
                                     theta2 +
                             static_cast<float>(distor_coeffs.at<double>(1)) *
                                     theta4 +
                             static_cast<float>(distor_coeffs.at<double>(2)) *
                                     theta6 +
                             static_cast<float>(distor_coeffs.at<double>(3)) *
                                     theta8) /
                            (1.f + static_cast<float>(m_virtualK2) * theta2);
                    float tx = fx * (x_hat * coeff) + cx;
                    float ty = fy * (y_hat * coeff) + cy;
                    mapxy.at<cv::Vec2f>(i, j) = cv::Vec2f(tx, ty);
                } else if (m_avmMode == "large_single_view") {
                    float scale_x = scale;
                    float scale_y = scale;
                    if (m_currentCamera == mcalib::kPanoramic3) {
                        scale_x = -scale_x;
                    }
                    float x_hat = (j - target_cx) / scale_x;
                    float y_hat = (i - target_cy) / scale_y;
                    float tx = x_hat + cx;
                    float ty = y_hat + cy;
                    mapxy.at<cv::Vec2f>(i, j) = cv::Vec2f(tx, ty);
                }
            }
        }
    }

    cv::remap(img, img, mapxy, cv::Mat(), cv::INTER_LINEAR);

    if (img.cols > 3600) {
        cv::resize(img, img, cv::Size(), 0.5, 0.5);
    } else if (img.cols > 2400) {
        cv::resize(img, img, cv::Size(), 0.8, 0.8);
    }

    drawRectOverlay(img);
    displayImageInViewer(img);
}

void ManualAvmAdjustDlg::drawRectOverlay(cv::Mat& img) const {
    if (m_avmMode != "wheel_hub_view" || img.empty()) return;

    cv::Rect rect(m_rectX, m_rectY, m_rectWidth, m_rectHeight);
    rect &= cv::Rect(0, 0, img.cols, img.rows);
    if (rect.width <= 0 || rect.height <= 0) return;

    cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

void ManualAvmAdjustDlg::displayImageInViewer(const cv::Mat& img) {
    if (!m_app) return;

    m_app->toggle3DView(false);

    const cv::Mat& display = img;

    cv::Mat rgb;
    if (display.channels() == 1) {
        cv::cvtColor(display, rgb, cv::COLOR_GRAY2RGB);
    } else if (display.channels() == 4) {
        cv::cvtColor(display, rgb, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(display, rgb, cv::COLOR_BGR2RGB);
    }
    QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                QImage::Format_RGB888);
    QImage copied = qimg.copy();

    if (!m_vtkImage) {
        const QString configTag = ecvPluginDbNaming::sanitizeSegment(
                QFileInfo(m_configPath).completeBaseName(), 20);
        const QString baseName =
                QStringLiteral("MCalib_AVM_%1_Preview")
                        .arg(configTag.isEmpty() ? QStringLiteral("NoCfg")
                                                 : configTag);
        const QString previewName =
                ecvPluginDbNaming::makeUnique(baseName, m_app);
        m_vtkImage = new ccImage(copied, previewName);
        m_vtkImage->setEnabled(true);
        m_vtkImage->setVisible(true);
        m_app->addToDB(m_vtkImage, false, false, false);
    } else {
        m_vtkImage->setData(copied);
        m_vtkImage->setVisible(true);
        m_vtkImage->setEnabled(true);
    }
    m_vtkImage->setRedrawFlagRecursive(true);
    m_app->redrawAll(false, true);
    m_app->refreshAll(false, true);

    auto* app = m_app;
    QTimer::singleShot(50, [app]() {
        if (!app) return;
        app->toggle3DView(false);
        app->redrawAll(false, true);
        app->refreshAll(false, true);
    });
}

void ManualAvmAdjustDlg::onCameraChanged(int) {
    m_currentCamera = m_cmbCamera->currentText().toStdString();
    if (m_bagReader && m_bagReader->isOpen()) {
        m_appliedSliderValue = -1;
        startBackgroundBagIndex();
        processSliderLoad();
    } else {
        updateImage();
    }
}

void ManualAvmAdjustDlg::onAvmModeChanged(int) {
    m_avmMode = m_cmbAvmMode->currentText().toStdString();

    m_blockUpdate = true;
    if (m_avmMode == "small_single_view") {
        m_virtualK2 = 0.16;
        m_targetScale = 1.5;
        m_v0Offset = 0.03;
        m_imgWidth = 1014;
        m_imgHeight = 966;
        m_spinVirtualK2->setValue(m_virtualK2);
        m_spinScale->setValue(m_targetScale);
        m_spinV0Offset->setValue(m_v0Offset);
        m_spinImgWidth->setValue(m_imgWidth);
        m_spinImgHeight->setValue(m_imgHeight);
    } else {
        m_targetScale = 1.4;
        m_v0Offset = -0.01;
        m_imgWidth = 1656;
        m_imgHeight = 966;
        m_rectWidth = m_imgWidth;
        m_rectHeight = m_imgHeight;
        m_spinScale->setValue(m_targetScale);
        m_spinV0Offset->setValue(m_v0Offset);
        m_spinImgWidth->setValue(m_imgWidth);
        m_spinImgHeight->setValue(m_imgHeight);
        m_spinRectWidth->setValue(m_rectWidth);
        m_spinRectHeight->setValue(m_rectHeight);
    }
    m_blockUpdate = false;

    if (m_bagReader && m_bagReader->isOpen()) {
        m_appliedSliderValue = -1;
        processSliderLoad();
    } else {
        updateImage();
    }
}

void ManualAvmAdjustDlg::onBaseRotChanged(int) {
    m_baseRot = m_cmbBaseRot->currentText().toStdString();
    updateImage();
}

void ManualAvmAdjustDlg::onTimeSliderChanged(int value) {
    updateTimeSliderLabel(value);
    m_pendingSliderValue = value;

    if (!m_sliderDebounce) {
        m_sliderDebounce = new QTimer(this);
        m_sliderDebounce->setSingleShot(true);
        connect(m_sliderDebounce, &QTimer::timeout, this,
                &ManualAvmAdjustDlg::processSliderLoad);
    }

    const int debounceMs =
            (m_bagReader && m_bagReader->hasTopicTimeIndex()) ? 12 : 24;
    m_sliderDebounce->start(debounceMs);
}

int ManualAvmAdjustDlg::bagTimeStepDelta() const {
    if (m_bagDurationSec <= 0.01) return 1;
    constexpr double kCameraFrameIntervalSec = 0.1;  // ~10Hz
    const double percent = kCameraFrameIntervalSec / m_bagDurationSec;
    return std::max(1, static_cast<int>(percent * 1000.0 + 0.5));
}

void ManualAvmAdjustDlg::onTimeStepBack() {
    if (!m_sliderTimePos) return;
    const int step = bagTimeStepDelta();
    const int next = std::max(0, m_sliderTimePos->value() - step);
    m_sliderTimePos->setValue(next);
}

void ManualAvmAdjustDlg::onTimeStepForward() {
    if (!m_sliderTimePos) return;
    const int step = bagTimeStepDelta();
    const int next = std::min(1000, m_sliderTimePos->value() + step);
    m_sliderTimePos->setValue(next);
}

void ManualAvmAdjustDlg::processSliderLoad() {
    if (!m_bagReader || !m_bagReader->isOpen()) return;

    if (m_loadingBag.load()) {
        m_sliderReloadPending = true;
        return;
    }
    if (m_pendingSliderValue == m_appliedSliderValue &&
        !m_sliderReloadPending) {
        return;
    }

    m_loadingBag.store(true);
    const int generation = ++m_sliderLoadGeneration;

    const double percent = m_pendingSliderValue / 1000.0;
    const std::string camera = m_currentCamera;
    mcalib::RosBagReader* reader = m_bagReader.get();

    QFuture<cv::Mat> future = QtConcurrent::run([reader, percent,
                                                 camera]() -> cv::Mat {
        cv::Mat image;
        if (!reader || !reader->isOpen()) return image;

        auto tryDecode = [&](const std::string& topic) -> bool {
            auto msgs = reader->readMessagesAtPercentParallel({topic}, percent);
            if (msgs.empty() || msgs[0].data.empty()) return false;
            double ts = 0;
            return mcalib::ProtoDecoder::decodeCompressedImageFromBag(
                    msgs[0].data, image, ts);
        };

        const std::string primary =
                "/sensors/camera/" + camera + "_raw_data/compressed_proto";
        if (tryDecode(primary)) return image;

        for (const auto& topic : reader->getTopics()) {
            if (topic.find(camera) != std::string::npos &&
                topic.find("camera") != std::string::npos) {
                if (tryDecode(topic)) return image;
            }
        }
        return image;
    });

    if (!m_sliderLoadWatcher) {
        m_sliderLoadWatcher = new QFutureWatcher<cv::Mat>(this);
        connect(m_sliderLoadWatcher, &QFutureWatcher<cv::Mat>::finished, this,
                [this]() {
                    const int generation =
                            m_sliderLoadWatcher->property("generation").toInt();
                    if (generation != m_sliderLoadGeneration.load()) {
                        m_loadingBag.store(false);
                        if (m_sliderReloadPending) {
                            m_sliderReloadPending = false;
                            processSliderLoad();
                        }
                        return;
                    }

                    cv::Mat image = m_sliderLoadWatcher->result();
                    if (!image.empty()) {
                        m_sourceImage = std::move(image);
                    }
                    m_appliedSliderValue = m_pendingSliderValue;
                    updateImage();
                    if (m_app) {
                        m_app->redrawAll(false, true);
                        m_app->refreshAll(false, true);
                    }

                    m_loadingBag.store(false);
                    if (m_sliderReloadPending) {
                        m_sliderReloadPending = false;
                        processSliderLoad();
                    }
                });
    }

    m_sliderLoadWatcher->setProperty("generation", generation);
    m_sliderLoadWatcher->setFuture(future);
}

#define IMPL_PARAM_CHANGED(member, type) \
    m_blockUpdate = false;               \
    member = static_cast<type>(value);   \
    updateImage();

void ManualAvmAdjustDlg::onVirtualK2Changed(double value) {
    IMPL_PARAM_CHANGED(m_virtualK2, double)
}
void ManualAvmAdjustDlg::onScaleChanged(double value) {
    IMPL_PARAM_CHANGED(m_targetScale, double)
}
void ManualAvmAdjustDlg::onV0OffsetChanged(double value) {
    IMPL_PARAM_CHANGED(m_v0Offset, double)
}
void ManualAvmAdjustDlg::onImgWidthChanged(double value) {
    IMPL_PARAM_CHANGED(m_imgWidth, int)
}
void ManualAvmAdjustDlg::onImgHeightChanged(double value) {
    IMPL_PARAM_CHANGED(m_imgHeight, int)
}
void ManualAvmAdjustDlg::onFocalXChanged(double value) {
    IMPL_PARAM_CHANGED(m_focalX, int)
}
void ManualAvmAdjustDlg::onFocalYChanged(double value) {
    IMPL_PARAM_CHANGED(m_focalY, int)
}
void ManualAvmAdjustDlg::onRotXChanged(double value) {
    IMPL_PARAM_CHANGED(m_rotX, double)
}
void ManualAvmAdjustDlg::onRotYChanged(double value) {
    IMPL_PARAM_CHANGED(m_rotY, double)
}
void ManualAvmAdjustDlg::onRotZChanged(double value) {
    IMPL_PARAM_CHANGED(m_rotZ, double)
}
void ManualAvmAdjustDlg::onRectXChanged(double value) {
    IMPL_PARAM_CHANGED(m_rectX, int)
}
void ManualAvmAdjustDlg::onRectYChanged(double value) {
    IMPL_PARAM_CHANGED(m_rectY, int)
}
void ManualAvmAdjustDlg::onRectWidthChanged(double value) {
    IMPL_PARAM_CHANGED(m_rectWidth, int)
}
void ManualAvmAdjustDlg::onRectHeightChanged(double value) {
    IMPL_PARAM_CHANGED(m_rectHeight, int)
}
#undef IMPL_PARAM_CHANGED

void ManualAvmAdjustDlg::onSaveParam() {
    QString filename = QFileDialog::getSaveFileName(
            asWidget(m_app->getMainWindow()), tr("Save Parameters"),
            m_configPath, tr("Config (*.txt *.cfg)"));
    if (filename.isEmpty()) return;

    std::ofstream ofs;
    if (!mcalib::openOutputFile(ofs, mcalib::pathFromQString(filename))) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Cannot write: %1").arg(filename));
        return;
    }

    ofs << "camera_name: " << m_currentCamera << "\n";
    ofs << "avm_mode: " << m_avmMode << "\n";
    ofs << "base_rot: " << m_baseRot << "\n";
    ofs << "virtual_k2: " << m_virtualK2 << "\n";
    ofs << "target_scale: " << m_targetScale << "\n";
    ofs << "v0_offset: " << m_v0Offset << "\n";
    ofs << "img_width: " << m_imgWidth << "\n";
    ofs << "img_height: " << m_imgHeight << "\n";
    ofs << "focal_x: " << m_focalX << "\n";
    ofs << "focal_y: " << m_focalY << "\n";
    ofs << "rot_x: " << m_rotX << "\n";
    ofs << "rot_y: " << m_rotY << "\n";
    ofs << "rot_z: " << m_rotZ << "\n";
    ofs << "rect_x: " << m_rectX << "\n";
    ofs << "rect_y: " << m_rectY << "\n";
    ofs << "rect_width: " << m_rectWidth << "\n";
    ofs << "rect_height: " << m_rectHeight << "\n";

    m_lblStatus->setText(tr("Parameters saved to: %1").arg(filename));
}

void ManualAvmAdjustDlg::onLoadParam() {
    QString filename = QFileDialog::getOpenFileName(
            asWidget(m_app->getMainWindow()), tr("Load Parameters"),
            m_configPath, tr("Config (*.txt *.cfg)"));
    if (filename.isEmpty()) return;

    std::ifstream ifs;
    if (!mcalib::openInputFile(ifs, mcalib::pathFromQString(filename))) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Cannot read: %1").arg(filename));
        return;
    }

    m_blockUpdate = true;
    std::string line;
    while (std::getline(ifs, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 2);

        if (key == "virtual_k2") {
            m_virtualK2 = std::stod(val);
            m_spinVirtualK2->setValue(m_virtualK2);
        } else if (key == "target_scale") {
            m_targetScale = std::stod(val);
            m_spinScale->setValue(m_targetScale);
        } else if (key == "v0_offset") {
            m_v0Offset = std::stod(val);
            m_spinV0Offset->setValue(m_v0Offset);
        } else if (key == "img_width") {
            m_imgWidth = std::stoi(val);
            m_spinImgWidth->setValue(m_imgWidth);
        } else if (key == "img_height") {
            m_imgHeight = std::stoi(val);
            m_spinImgHeight->setValue(m_imgHeight);
        } else if (key == "focal_x") {
            m_focalX = std::stoi(val);
            m_spinFocalX->setValue(m_focalX);
        } else if (key == "focal_y") {
            m_focalY = std::stoi(val);
            m_spinFocalY->setValue(m_focalY);
        } else if (key == "rot_x") {
            m_rotX = std::stod(val);
            m_spinRotX->setValue(m_rotX);
        } else if (key == "rot_y") {
            m_rotY = std::stod(val);
            m_spinRotY->setValue(m_rotY);
        } else if (key == "rot_z") {
            m_rotZ = std::stod(val);
            m_spinRotZ->setValue(m_rotZ);
        } else if (key == "rect_x") {
            m_rectX = std::stoi(val);
            m_spinRectX->setValue(m_rectX);
        } else if (key == "rect_y") {
            m_rectY = std::stoi(val);
            m_spinRectY->setValue(m_rectY);
        } else if (key == "rect_width") {
            m_rectWidth = std::stoi(val);
            m_spinRectWidth->setValue(m_rectWidth);
        } else if (key == "rect_height") {
            m_rectHeight = std::stoi(val);
            m_spinRectHeight->setValue(m_rectHeight);
        } else if (key == "avm_mode") {
            m_avmMode = val;
            int idx = m_cmbAvmMode->findText(QString::fromStdString(val));
            if (idx >= 0) m_cmbAvmMode->setCurrentIndex(idx);
        } else if (key == "base_rot") {
            m_baseRot = val;
            int idx = m_cmbBaseRot->findText(QString::fromStdString(val));
            if (idx >= 0) m_cmbBaseRot->setCurrentIndex(idx);
        }
    }
    m_blockUpdate = false;
    updateImage();
    m_lblStatus->setText(tr("Parameters loaded from: %1").arg(filename));
}
