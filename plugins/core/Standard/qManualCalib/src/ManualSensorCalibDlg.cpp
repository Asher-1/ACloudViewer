// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ManualSensorCalibDlg.h"

#include <CVLog.h>
#include <ecvDisplayTools.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDoubleValidator>
#include <QFileDialog>
#include <QFuture>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMouseEvent>
#include <QMutex>
#include <QMutexLocker>
#include <QPointer>
#include <QProgressDialog>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <algorithm>
#include <future>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "BagAlignment.h"
#include "BagLoadHelper.h"
#include "BevBatchExport.h"
#include "BevRemapBackend.h"
#include "BirdEyeView.h"
#include "CalibConfigParser.h"
#include "CalibTypes.h"
#include "ImageBatchExport.h"
#include "LidarProjBackend.h"
#include "PcdBatchExport.h"
#include "ProtoDecoder.h"
#include "RosBagReader.h"
#include "mcalib_portability.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

static inline QWidget* asWidget(QMainWindow* w) { return w; }

ManualSensorCalibDlg::ManualSensorCalibDlg(ecvMainAppInterface* app,
                                           QWidget* parent)
    : ccOverlayDialog(
              parent,
              Qt::Tool | Qt::CustomizeWindowHint | Qt::WindowCloseButtonHint),
      m_app(app) {
    setWindowTitle(tr("Manual Sensor Calibration"));
    setupUI();
}

ManualSensorCalibDlg::~ManualSensorCalibDlg() {
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
    cleanupPreviewEntities();
}

bool ManualSensorCalibDlg::linkWith(QWidget* win) {
    return ccOverlayDialog::linkWith(win);
}

bool ManualSensorCalibDlg::start() {
    const bool ok = ccOverlayDialog::start();
    if (ok) {
        QTimer::singleShot(0, this, [this]() { showIdleBevCanvas(); });
    }
    return ok;
}

void ManualSensorCalibDlg::stop(bool accepted) {
    auto* app = m_app;

    if (m_sliderLoadWatcher) {
        ++m_sliderLoadGeneration;
        m_sliderLoadWatcher->disconnect();
        m_sliderLoadWatcher->waitForFinished();
        delete m_sliderLoadWatcher;
        m_sliderLoadWatcher = nullptr;
    }
    m_loadingBag.store(false);
    m_sliderReloadPending = false;
    m_bagReader.reset();

    cleanupPreviewEntities();

    if (app) {
        app->toggle3DView(true);
        app->redrawAll(true);
        app->refreshAll(true);
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

void ManualSensorCalibDlg::cleanupPreviewEntities() {
    if (!m_app) return;

    if (m_vtkImage) {
        ccHObject* obj = m_vtkImage;
        ecvDisplayTools::RemoveEntities(obj);
        obj->removeFromRenderScreen(true);
        m_app->removeFromDB(obj, true);
        m_vtkImage = nullptr;
    }
    if (m_vtkCloud) {
        ccHObject* obj = m_vtkCloud;
        ecvDisplayTools::RemoveEntities(obj);
        obj->removeFromRenderScreen(true);
        m_app->removeFromDB(obj, true);
        m_vtkCloud = nullptr;
    }

    m_app->toggle3DView(true);
    m_app->refreshAll(true, true);
    m_app->redrawAll(true, true);
}

void ManualSensorCalibDlg::closeEvent(QCloseEvent* event) {
    stop(false);
    event->accept();
}

void ManualSensorCalibDlg::setupUI() {
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(4, 4, 4, 4);
    rootLayout->setSpacing(3);

    auto* controlScroll = new QScrollArea;
    controlScroll->setWidgetResizable(true);
    controlScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    controlScroll->setFixedWidth(360);
    auto* controlWidget = new QWidget;
    auto* controlLayout = new QVBoxLayout(controlWidget);
    controlLayout->setContentsMargins(2, 2, 2, 2);
    controlLayout->setSpacing(3);
    controlScroll->setWidget(controlWidget);

    // -- View mode buttons (compact 2x2 grid) --
    m_cmbViewMode = new QComboBox;
    m_cmbViewMode->addItems(
            {"BEV", "Lidar Projection", "Single Frame", "Multi Frame"});
    m_cmbViewMode->setVisible(false);

    auto* viewGrid = new QGridLayout;
    viewGrid->setSpacing(2);
    auto* btnBev = new QPushButton(tr("BEV View"));
    auto* btnLidarProj = new QPushButton(tr("LiDAR Proj"));
    auto* btnSingleFrame = new QPushButton(tr("Single Frame"));
    auto* btnMultiFrame = new QPushButton(tr("Multi Frame"));
    m_btnBevView = btnBev;
    m_btnLidarProjView = btnLidarProj;
    m_btnSingleFrameView = btnSingleFrame;
    m_btnMultiFrameView = btnMultiFrame;
    btnBev->setFixedHeight(26);
    btnLidarProj->setFixedHeight(26);
    btnSingleFrame->setFixedHeight(26);
    btnMultiFrame->setFixedHeight(26);
    viewGrid->addWidget(btnBev, 0, 0);
    viewGrid->addWidget(btnLidarProj, 0, 1);
    viewGrid->addWidget(btnSingleFrame, 1, 0);
    viewGrid->addWidget(btnMultiFrame, 1, 1);

    connect(btnBev, &QPushButton::clicked,
            [this] { m_cmbViewMode->setCurrentIndex(0); });
    connect(btnLidarProj, &QPushButton::clicked,
            [this] { m_cmbViewMode->setCurrentIndex(1); });
    connect(btnSingleFrame, &QPushButton::clicked,
            [this] { m_cmbViewMode->setCurrentIndex(2); });
    connect(btnMultiFrame, &QPushButton::clicked,
            [this] { m_cmbViewMode->setCurrentIndex(3); });
    connect(m_cmbViewMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ManualSensorCalibDlg::onViewModeChanged);
    updateViewModeButtonStyle(0);
    controlLayout->addLayout(viewGrid);

    // -- Load cfg then bag (cfg must be loaded first) --
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
            &ManualSensorCalibDlg::onLoadConfig);
    connect(m_btnLoadBag, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onLoadBag);

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
            &ManualSensorCalibDlg::onTimeStepBack);
    connect(m_btnTimeStepForward, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onTimeStepForward);
    connect(m_sliderTimePos, &QSlider::sliderMoved, this,
            &ManualSensorCalibDlg::updateTimeSliderLabel);
    connect(m_sliderTimePos, &QSlider::valueChanged, this,
            &ManualSensorCalibDlg::onTimeSliderChanged);

    // -- Point size (LiDAR proj / 3D) or focal scale (BEV) slider --
    auto* ptSizeRow = new QHBoxLayout;
    m_lblPointSizeTitle = new QLabel(tr("Point size"));
    m_sliderPointSize = new QSlider(Qt::Horizontal);
    m_sliderPointSize->setRange(0, 6);
    m_sliderPointSize->setValue(2);
    m_lblPointSize = new QLabel("2");
    m_lblPointSize->setMinimumWidth(28);
    ptSizeRow->addWidget(m_lblPointSizeTitle);
    ptSizeRow->addWidget(m_sliderPointSize);
    ptSizeRow->addWidget(m_lblPointSize);
    controlLayout->addLayout(ptSizeRow);
    connect(m_sliderPointSize, &QSlider::valueChanged, this,
            &ManualSensorCalibDlg::onPointSizeChanged);
    connect(m_sliderPointSize, &QSlider::valueChanged, this,
            [this](int) { updatePointSizeSliderLabel(); });
    updatePointSizeSliderLabel();

    // -- GPU backend (Auto / CUDA / OpenCL / CPU) shared by BEV + LiDAR
    // projection --
    auto* bevBackendRow = new QHBoxLayout;
    bevBackendRow->addWidget(new QLabel(tr("GPU backend")));
    m_cmbBevRemapBackend = new QComboBox;
    m_cmbBevRemapBackend->addItem(tr("Auto"),
                                  static_cast<int>(mcalib::BevRemapMode::Auto));
    m_cmbBevRemapBackend->addItem(tr("CUDA"),
                                  static_cast<int>(mcalib::BevRemapMode::CUDA));
    m_cmbBevRemapBackend->addItem(
            tr("OpenCL"), static_cast<int>(mcalib::BevRemapMode::OpenCL));
    m_cmbBevRemapBackend->addItem(tr("CPU"),
                                  static_cast<int>(mcalib::BevRemapMode::CPU));

    QString cudaHint =
            mcalib::BevRemapper::cudaAvailable() ? QString() : tr(" (N/A)");
    QString oclHint =
            mcalib::BevRemapper::openclAvailable() ? QString() : tr(" (N/A)");
    m_cmbBevRemapBackend->setItemData(
            0, tr("CUDA if available, else OpenCL (macOS / no-CUDA), else CPU"),
            Qt::ToolTipRole);
    const QString cudaTip =
            QString(tr("NVIDIA GPU remap + alpha fusion")) + cudaHint;
    const QString oclTip =
            QString(tr("OpenCL GPU remap + alpha fusion (macOS default)")) +
            oclHint;
    m_cmbBevRemapBackend->setItemData(1, cudaTip, Qt::ToolTipRole);
    m_cmbBevRemapBackend->setItemData(2, oclTip, Qt::ToolTipRole);
    m_cmbBevRemapBackend->setItemData(3, tr("Force CPU parallel remap + blend"),
                                      Qt::ToolTipRole);

    if (!mcalib::BevRemapper::cudaAvailable()) {
        m_cmbBevRemapBackend->setItemData(1, 0, Qt::UserRole - 1);
    }
    if (!mcalib::BevRemapper::openclAvailable()) {
        m_cmbBevRemapBackend->setItemData(2, 0, Qt::UserRole - 1);
    }

    QSettings settings;
    int savedBackend =
            settings.value("qManualCalib/bevRemapBackend",
                           static_cast<int>(mcalib::BevRemapMode::Auto))
                    .toInt();
    if (savedBackend == static_cast<int>(mcalib::BevRemapMode::CUDA) &&
        !mcalib::BevRemapper::cudaAvailable()) {
        savedBackend = mcalib::BevRemapper::openclAvailable()
                               ? static_cast<int>(mcalib::BevRemapMode::OpenCL)
                               : static_cast<int>(mcalib::BevRemapMode::Auto);
    } else if (savedBackend == static_cast<int>(mcalib::BevRemapMode::OpenCL) &&
               !mcalib::BevRemapper::openclAvailable()) {
        savedBackend = static_cast<int>(mcalib::BevRemapMode::Auto);
    }
    const int backendIdx = m_cmbBevRemapBackend->findData(savedBackend);
    m_cmbBevRemapBackend->setCurrentIndex(backendIdx >= 0 ? backendIdx : 0);
    bevBackendRow->addWidget(m_cmbBevRemapBackend);
    controlLayout->addLayout(bevBackendRow);
    connect(m_cmbBevRemapBackend,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ManualSensorCalibDlg::onBevRemapBackendChanged);

    // -- Distance filter --
    auto* distRow = new QHBoxLayout;
    distRow->addWidget(new QLabel(tr("Dist(m)")));
    m_sliderDistFilterMin = new QSlider(Qt::Horizontal);
    m_sliderDistFilterMin->setRange(0, 50);
    m_sliderDistFilterMin->setValue(0);
    m_sliderDistFilter = new QSlider(Qt::Horizontal);
    m_sliderDistFilter->setRange(1, 200);
    m_sliderDistFilter->setValue(100);
    m_lblDistRange = new QLabel("0.5~100m");
    m_lblDistRange->setMinimumWidth(60);
    distRow->addWidget(m_sliderDistFilterMin);
    distRow->addWidget(m_sliderDistFilter);
    distRow->addWidget(m_lblDistRange);
    controlLayout->addLayout(distRow);
    connect(m_sliderDistFilter, &QSlider::valueChanged, this,
            &ManualSensorCalibDlg::onDistFilterChanged);
    connect(m_sliderDistFilterMin, &QSlider::valueChanged, this,
            &ManualSensorCalibDlg::onDistFilterMinChanged);
    onDistFilterMinChanged(m_sliderDistFilterMin->value());
    onDistFilterChanged(m_sliderDistFilter->value());

    // -- Ground filter --
    auto* gndRow = new QHBoxLayout;
    gndRow->addWidget(new QLabel(tr("Gnd(m)")));
    m_sliderGroundFilter = new QSlider(Qt::Horizontal);
    m_sliderGroundFilter->setRange(-100, 0);
    m_sliderGroundFilter->setValue(-50);
    m_sliderGroundFilterMax = new QSlider(Qt::Horizontal);
    m_sliderGroundFilterMax->setRange(0, 300);
    m_sliderGroundFilterMax->setValue(300);
    m_lblGroundRange = new QLabel("-5.0~5.0m");
    m_lblGroundRange->setMinimumWidth(60);
    gndRow->addWidget(m_sliderGroundFilter);
    gndRow->addWidget(m_sliderGroundFilterMax);
    gndRow->addWidget(m_lblGroundRange);
    controlLayout->addLayout(gndRow);
    connect(m_sliderGroundFilter, &QSlider::valueChanged, this,
            &ManualSensorCalibDlg::onGroundFilterChanged);
    connect(m_sliderGroundFilterMax, &QSlider::valueChanged, this,
            &ManualSensorCalibDlg::onGroundFilterMaxChanged);
    onGroundFilterChanged(m_sliderGroundFilter->value());
    onGroundFilterMaxChanged(m_sliderGroundFilterMax->value());

    // -- Parameter adjustment area --
    auto* paramTitle = new QLabel(tr("Parameter Adjustment"));
    paramTitle->setStyleSheet("font-weight: bold;");
    controlLayout->addWidget(paramTitle);

    // Sensor Type + target sensor side by side
    auto* sensorPickRow = new QHBoxLayout;
    sensorPickRow->setSpacing(4);
    auto* sensorTypeCol = new QVBoxLayout;
    sensorTypeCol->setSpacing(2);
    sensorTypeCol->addWidget(new QLabel(tr("Sensor Type")));
    m_cmbSensorType = new QComboBox;
    m_cmbSensorType->addItems({tr("Camera"), tr("Lidar"), tr("Radar")});
    sensorTypeCol->addWidget(m_cmbSensorType);
    sensorPickRow->addLayout(sensorTypeCol, 1);

    auto* sensorNameCol = new QVBoxLayout;
    sensorNameCol->setSpacing(2);
    sensorNameCol->addWidget(new QLabel(tr("Sensor")));
    m_cmbSensorName = new QComboBox;
    sensorNameCol->addWidget(m_cmbSensorName);
    sensorPickRow->addLayout(sensorNameCol, 1);
    controlLayout->addLayout(sensorPickRow);

    connect(m_cmbSensorType,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ManualSensorCalibDlg::onSensorTypeChanged);
    connect(m_cmbSensorName, &QComboBox::currentTextChanged, this,
            &ManualSensorCalibDlg::onSensorNameChanged);

    // Adjust Range + calib mode side by side, directly above 6-DOF controls
    auto* rangeModeRow = new QHBoxLayout;
    rangeModeRow->setSpacing(4);
    auto* rangeCol = new QVBoxLayout;
    rangeCol->setSpacing(2);
    rangeCol->addWidget(new QLabel(tr("Adjust Range")));
    m_cmbSpeed = new QComboBox;
    m_cmbSpeed->addItems({"1x", "5x", "10x", "50x", "100x", "1000x"});
    m_cmbSpeed->setCurrentIndex(4);
    rangeCol->addWidget(m_cmbSpeed);
    rangeModeRow->addLayout(rangeCol, 1);

    auto* calibModeCol = new QVBoxLayout;
    calibModeCol->setSpacing(2);
    calibModeCol->addWidget(new QLabel(tr("Calib Mode")));
    m_cmbCalibMode = new QComboBox;
    calibModeCol->addWidget(m_cmbCalibMode);
    rangeModeRow->addLayout(calibModeCol, 1);
    controlLayout->addLayout(rangeModeRow);

    connect(m_cmbSpeed, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ManualSensorCalibDlg::onSpeedChanged);
    onSpeedChanged(m_cmbSpeed->currentIndex());
    connect(m_cmbCalibMode, &QComboBox::currentTextChanged, this,
            &ManualSensorCalibDlg::onCalibModeChanged);

    // -- 6-DOF controls --
    auto* adjustLayout = new QGridLayout;
    adjustLayout->setSpacing(3);

    auto createAdjustRow = [&](const QString& label, QLineEdit*& edit, int row,
                               auto addSlot, auto subSlot) {
        adjustLayout->addWidget(new QLabel(label), row, 0);
        edit = new QLineEdit("0.0000");
        edit->setMaximumWidth(70);
        edit->setAlignment(Qt::AlignRight);
        auto* validator = new QDoubleValidator(-9999.0, 9999.0, 6, edit);
        validator->setNotation(QDoubleValidator::StandardNotation);
        edit->setValidator(validator);
        adjustLayout->addWidget(edit, row, 1);
        auto* btnAdd = new QPushButton("+");
        auto* btnSub = new QPushButton("-");
        btnAdd->setFixedSize(28, 24);
        btnSub->setFixedSize(28, 24);
        adjustLayout->addWidget(btnAdd, row, 2);
        adjustLayout->addWidget(btnSub, row, 3);
        connect(btnAdd, &QPushButton::clicked, this, addSlot);
        connect(btnSub, &QPushButton::clicked, this, subSlot);
        connect(edit, &QLineEdit::editingFinished, this,
                &ManualSensorCalibDlg::onExtrinsicEditCommitted);
    };

    createAdjustRow(tr("Roll"), m_editRoll, 0, &ManualSensorCalibDlg::onRollAdd,
                    &ManualSensorCalibDlg::onRollSub);
    createAdjustRow(tr("Pitch"), m_editPitch, 1,
                    &ManualSensorCalibDlg::onPitchAdd,
                    &ManualSensorCalibDlg::onPitchSub);
    createAdjustRow(tr("Yaw"), m_editYaw, 2, &ManualSensorCalibDlg::onYawAdd,
                    &ManualSensorCalibDlg::onYawSub);
    createAdjustRow("x", m_editX, 3, &ManualSensorCalibDlg::onXAdd,
                    &ManualSensorCalibDlg::onXSub);
    createAdjustRow("y", m_editY, 4, &ManualSensorCalibDlg::onYAdd,
                    &ManualSensorCalibDlg::onYSub);
    createAdjustRow("z", m_editZ, 5, &ManualSensorCalibDlg::onZAdd,
                    &ManualSensorCalibDlg::onZSub);
    controlLayout->addLayout(adjustLayout);

    // -- Reset / Save / Export buttons --
    auto* actionRow = new QHBoxLayout;
    auto* btnReset = new QPushButton(tr("Reset"));
    auto* btnSave = new QPushButton(tr("Save"));
    btnReset->setFixedHeight(28);
    btnSave->setFixedHeight(28);
    actionRow->addWidget(btnReset);
    actionRow->addWidget(btnSave);
    controlLayout->addLayout(actionRow);

    auto* exportPcdRow = new QHBoxLayout;
    exportPcdRow->setSpacing(2);
    m_btnExportPCD = new QPushButton(tr("Export PCD"));
    m_btnExportPCD->setFixedHeight(28);
    m_btnExportPCD->setToolTip(tr("Export displayed point cloud to DB tree"));
    m_btnBatchExportPCD = new QPushButton(tr("Batch Export PCD"));
    m_btnBatchExportPCD->setFixedHeight(28);
    m_btnBatchExportPCD->setToolTip(
            tr("Batch export point clouds from bag to PCD files"));
    exportPcdRow->addWidget(m_btnExportPCD);
    exportPcdRow->addWidget(m_btnBatchExportPCD);
    controlLayout->addLayout(exportPcdRow);

    auto* exportImageRow = new QHBoxLayout;
    exportImageRow->setSpacing(2);
    m_btnExportImage = new QPushButton(tr("Export Image"));
    m_btnExportImage->setFixedHeight(28);
    m_btnExportImage->setToolTip(
            tr("Export current BEV or LiDAR projection image to DB tree"));
    m_btnBatchExportImages = new QPushButton(tr("Batch Export Images"));
    m_btnBatchExportImages->setFixedHeight(28);
    m_btnBatchExportImages->setToolTip(
            tr("Batch export BEV or LiDAR projection images (based on current "
               "view mode)"));
    exportImageRow->addWidget(m_btnExportImage);
    exportImageRow->addWidget(m_btnBatchExportImages);
    controlLayout->addLayout(exportImageRow);

    connect(btnReset, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onResetParams);
    connect(btnSave, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onSaveConfig);
    connect(m_btnExportPCD, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onSavePCD);
    connect(m_btnBatchExportPCD, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onBatchExportPCD);
    connect(m_btnExportImage, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onExportImage);
    connect(m_btnBatchExportImages, &QPushButton::clicked, this,
            &ManualSensorCalibDlg::onBatchExportImages);
    updateExportButtonStates();

    // -- LiDAR selection (populated dynamically) --
    m_lidarGroup = new QGroupBox(tr("LiDAR"));
    m_lidarGroupLayout = new QVBoxLayout(m_lidarGroup);
    m_lidarGroup->setVisible(false);
    controlLayout->addWidget(m_lidarGroup);

    m_lblStatus = new QLabel(tr("Step 1: Load cfg (cameras.cfg directory)"));
    m_lblStatus->setStyleSheet("color: #999; font-size: 11px;");
    m_lblStatus->setWordWrap(true);
    m_lblStatus->setMaximumWidth(350);
    m_lblStatus->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    controlLayout->addWidget(m_lblStatus);

    controlWidget->setMaximumWidth(360);
    rootLayout->addWidget(controlScroll);
    setFixedWidth(372);

    onSensorTypeChanged(m_cmbSensorType->currentIndex());
}

void ManualSensorCalibDlg::onLoadConfig() {
    QSettings settings;
    QString lastCfg = settings.value("qManualCalib/lastConfigDir", m_configPath)
                              .toString();
    QString dir = QFileDialog::getExistingDirectory(
            asWidget(m_app->getMainWindow()), tr("Select Config Directory"),
            lastCfg);
    if (dir.isEmpty()) return;

    settings.setValue("qManualCalib/lastConfigDir", dir);
    m_configPath = dir;
    const std::string config_dir = mcalib::pathFromQString(dir);

    m_calibConfig = mcalib::VehicleCalibConfig();
    if (!mcalib::CalibConfigParser::loadConfigDirectory(config_dir,
                                                        m_calibConfig)) {
        CVLog::Warning("[SensorCalib] Failed to load config directory: %s",
                       config_dir.c_str());
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Config directory must contain cameras.cfg, "
                                "lidars.cfg and ground.cfg:\n%1")
                                     .arg(dir));
        return;
    }

    mcalib::CalibConfigParser::alignCameraSizes(m_calibConfig);

    m_selectedLidars.resize(m_calibConfig.lidars.size(), true);
    m_deltaLidarExtrinsics.resize(m_calibConfig.lidars.size(),
                                  mcalib::Vector6d::Zero());
    initCameraDeltaExtrinsics();

    createLidarCheckboxes();

    m_bevViewer = std::make_unique<mcalib::BirdEyeView>();
    mcalib::BirdEyeView::Config bev_cfg;
    if (m_cmbBevRemapBackend) {
        bev_cfg.remap_mode = static_cast<mcalib::BevRemapMode>(
                m_cmbBevRemapBackend->currentData().toInt());
    }
    m_bevViewer->init(m_calibConfig, bev_cfg);
    m_extrinsicDirty = true;
    m_configLoaded = true;

    if (m_btnLoadBag) {
        m_btnLoadBag->setEnabled(true);
        m_btnLoadBag->setToolTip(QString());
    }

    updateSensorList();
    syncCurrentSensorFromCombo();
    CVLog::Print("[SensorCalib] Config loaded: %zu cameras, %zu lidars",
                 m_calibConfig.cameras.size(), m_calibConfig.lidars.size());
    m_lblStatus->setText(
            tr("Config loaded: %1 cameras, %2 lidars — now load bag")
                    .arg(m_calibConfig.cameras.size())
                    .arg(m_calibConfig.lidars.size()));

    if (m_bagReader && m_bagReader->isOpen()) {
        CVLog::Print(
                "[SensorCalib] Config reloaded, refreshing view with existing "
                "bag data");
        updateFusionView();
    } else if (m_viewMode == 0) {
        updateFusionView();
    }
}

void ManualSensorCalibDlg::onLoadBag() {
    if (!m_configLoaded || !m_bevViewer) {
        QMessageBox::information(
                asWidget(m_app->getMainWindow()), tr("Load Config First"),
                tr("Please load the config directory (cameras.cfg / "
                   "ground.cfg) before loading bag data."));
        return;
    }

    QSettings settings;
    QString lastBag =
            settings.value("qManualCalib/lastBagPath", m_bagPath).toString();
    const QString selected = mcalib::ui::pickBagInputPath(
            asWidget(m_app->getMainWindow()), lastBag);
    if (selected.isEmpty()) return;

    settings.setValue("qManualCalib/lastBagPath", selected);
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
        CVLog::Error("[SensorCalib] Failed to resolve bag input: %s",
                     resolved.error_message.c_str());
        QMessageBox::warning(
                asWidget(m_app->getMainWindow()), tr("Error"),
                tr("Failed to resolve bag input:\n%1")
                        .arg(QString::fromStdString(resolved.error_message)));
        return;
    }

    m_bagReader = std::make_unique<mcalib::RosBagReader>();
    m_bagIndexReady = false;
    m_bagDurationSec = 0.0;
    m_cameraTopics.clear();
    m_cloudTopics.clear();
    m_images.clear();
    m_imageStampsUs.clear();
    m_pointCloud.clear();
    m_pointCloudRaw.clear();

    progress->setLabelText(tr("Opening bag file..."));
    progress->setValue(5);
    QCoreApplication::processEvents();

    if (!mcalib::ui::openResolvedBag(*m_bagReader, resolved)) {
        progress->close();
        delete progress;
        CVLog::Error("[SensorCalib] Failed to open bag session: %s",
                     resolved.session_key.c_str());
        QMessageBox::warning(
                asWidget(m_app->getMainWindow()), tr("Error"),
                tr("Failed to open bag session: %1")
                        .arg(QString::fromStdString(resolved.session_key)));
        m_bagReader.reset();
        return;
    }

    CVLog::Print("[SensorCalib] Bag layout: %s, session: %s, source bags: %zu",
                 mcalib::pathFromQString(
                         mcalib::ui::layoutDescription(resolved.layout))
                         .c_str(),
                 resolved.session_key.c_str(), resolved.source_bags.size());

    progress->setLabelText(tr("Scanning topics..."));
    progress->setValue(20);
    QCoreApplication::processEvents();

    auto topics = m_bagReader->getTopicTypes();
    int img_topics = 0, cloud_topics = 0;
    for (const auto& [topic, type] : topics) {
        if (topic.find("/sensors/camera/") != std::string::npos ||
            topic.find("image") != std::string::npos) {
            CVLog::Print("[SensorCalib]   camera topic: '%s' [%s]",
                         topic.c_str(), type.c_str());
            ++img_topics;
        } else if (topic.find("lidar") != std::string::npos ||
                   topic.find("point_cloud") != std::string::npos ||
                   topic.find("cloud") != std::string::npos) {
            CVLog::Print("[SensorCalib]   lidar topic: '%s' [%s]",
                         topic.c_str(), type.c_str());
            ++cloud_topics;
        }
    }

    buildBagTopicLists();

    m_bagReader->refinePlaybackTimeRange(
            "/sensors/camera/camera_1_raw_data/compressed_proto");

    std::vector<std::string> probe_topics;
    probe_topics.reserve(m_cameraTopics.size());
    for (const auto& [topic, _] : m_cameraTopics) {
        probe_topics.push_back(topic);
    }
    m_bagImageEncoding =
            mcalib::probeBagImageEncoding(*m_bagReader, probe_topics);
    const char* encoding_name = "unknown";
    switch (m_bagImageEncoding) {
        case mcalib::BagImageEncoding::Jpeg:
            encoding_name = "jpeg";
            break;
        case mcalib::BagImageEncoding::H264:
            encoding_name = "h264";
            break;
        case mcalib::BagImageEncoding::Hevc:
            encoding_name = "hevc";
            break;
        case mcalib::BagImageEncoding::Mixed:
            encoding_name = "mixed";
            break;
        default:
            break;
    }
    CVLog::Print("[SensorCalib] Bag image encoding: %s", encoding_name);

    double duration_s = m_bagReader->getDuration() / 1e9;
    m_bagDurationSec = duration_s;
    updateTimeSliderLabel(m_sliderTimePos->value());
    CVLog::Print("[SensorCalib] Bag: %.1fs, %zu total topics, %d cam, %d lidar",
                 duration_s, topics.size(), img_topics, cloud_topics);

    std::set<std::string> index_topics;
    for (const auto& [topic, _] : m_cameraTopics) {
        index_topics.insert(topic);
    }
    for (const auto& topic : m_cloudTopics) {
        index_topics.insert(topic);
    }
    index_topics.insert("/canbus/car_state");

    if (mcalib::bagUsesVideoCodec(m_bagImageEncoding)) {
        progress->setLabelText(tr("Building time index (video)..."));
        progress->setValue(45);
        QCoreApplication::processEvents();
        m_bagIndexReady =
                m_bagReader->buildTopicTimeIndex(index_topics, nullptr, 0) &&
                m_bagReader->hasTopicTimeIndex();
        CVLog::Print("[SensorCalib] Video bag index: %s",
                     m_bagIndexReady ? "ready" : "unavailable");
    } else {
        progress->setValue(45);
        QCoreApplication::processEvents();
    }

    progress->setLabelText(tr("Loading vehicle trajectory..."));
    progress->setValue(30);
    QCoreApplication::processEvents();
    {
        const std::string pose_topic = "/localization/pose";
        m_vehiclePoses.clear();
        m_bagReader->readMessages(
                [&](const mcalib::BagMessage& msg) {
                    mcalib::ProtoDecoder::InsPose ins;
                    if (mcalib::ProtoDecoder::decodeInsPoseFromBag(msg.data,
                                                                   ins)) {
                        mcalib::Vector6d v;
                        v << ins.pos_x, ins.pos_y, ins.pos_z, ins.euler_x,
                                ins.euler_y, ins.euler_z;
                        Eigen::Isometry3d iso;
                        mcalib::xyzeuler2isometry(v, iso);
                        m_vehiclePoses.emplace_back(ins.measurement_time_us,
                                                    iso);
                    }
                    return true;
                },
                {pose_topic});
        CVLog::Print("[SensorCalib] Vehicle trajectory: %zu poses loaded",
                     m_vehiclePoses.size());
    }

    progress->setLabelText(tr("Loading initial frame..."));
    progress->setValue(70);
    QCoreApplication::processEvents();
    loadInitialBagFrame();

    if (!m_bagIndexReady) {
        startBackgroundBagIndex();
    }

    progress->setLabelText(tr("Preparing view..."));
    progress->setValue(95);
    QCoreApplication::processEvents();

    progress->setValue(100);
    progress->close();
    delete progress;

    m_lblStatus->setText(tr("Bag loaded: %1s (%2 cam, %3 lidar)")
                                 .arg(duration_s, 0, 'f', 1)
                                 .arg(img_topics)
                                 .arg(cloud_topics));
    m_extrinsicDirty = true;
    m_appliedSliderValue = m_sliderTimePos ? m_sliderTimePos->value() : 0;
    m_pendingSliderValue = m_appliedSliderValue;
    updateFusionView();
    updateExportButtonStates();
    QTimer::singleShot(0, this, [this]() {
        updateFusionView();
        if (m_app) {
            m_app->redrawAll(false, true);
            m_app->refreshAll(false, true);
        }
    });
}

void ManualSensorCalibDlg::buildBagTopicLists() {
    m_cameraTopics.clear();
    for (const auto& [name, _] : m_calibConfig.cameras) {
        std::string topic =
                "/sensors/camera/" + name + "_raw_data/compressed_proto";
        m_cameraTopics.emplace_back(topic, name);
    }

    if (m_cameraTopics.empty() && m_bagReader && m_bagReader->isOpen()) {
        CVLog::Print(
                "[SensorCalib] buildBagTopicLists: no cameras in config, "
                "scanning bag topics...");
        auto topics = m_bagReader->getTopics();
        for (const auto& topic : topics) {
            if (topic.find("/sensors/camera/") != std::string::npos &&
                topic.find("compressed_proto") != std::string::npos) {
                std::string cam_name = topic.substr(17);
                auto pos = cam_name.find("_raw_data");
                if (pos != std::string::npos)
                    cam_name = cam_name.substr(0, pos);
                m_cameraTopics.emplace_back(topic, cam_name);
            }
        }
    }

    m_cloudTopics = {"/sensors/lidar/combined_point_cloud_with_64_proto",
                     "/sensors/lidar/combined_point_cloud_proto",
                     "/sensors/lidar/combined_point_cloud_downsample_proto"};
}

void ManualSensorCalibDlg::startBackgroundBagIndex() {
    if (!m_bagReader || !m_bagReader->isOpen()) return;

    if (m_bagIndexWatcher) {
        m_bagIndexWatcher->cancel();
        m_bagIndexWatcher->waitForFinished();
        delete m_bagIndexWatcher;
        m_bagIndexWatcher = nullptr;
    }

    std::set<std::string> topics;
    for (const auto& [topic, _] : m_cameraTopics) {
        topics.insert(topic);
    }
    for (const auto& topic : m_cloudTopics) {
        topics.insert(topic);
    }
    topics.insert("/canbus/car_state");
    if (topics.empty()) return;

    m_bagIndexReady = false;
    m_bagIndexWatcher = new QFutureWatcher<void>(this);
    connect(m_bagIndexWatcher, &QFutureWatcher<void>::finished, this, [this]() {
        m_bagIndexReady = m_bagReader && m_bagReader->hasTopicTimeIndex();
        CVLog::Print("[SensorCalib] Background bag time index %s",
                     m_bagIndexReady ? "ready" : "unavailable");
        if (m_bagIndexReady && m_bagReader && m_bagReader->isOpen()) {
            if (needsImagesForView() && m_images.empty()) {
                loadInitialBagFrame();
                updateFusionView();
            }
            m_appliedSliderValue = -1;
            processSliderLoad();
        }
    });

    mcalib::RosBagReader* reader = m_bagReader.get();
    auto topics_copy = topics;
    m_bagIndexWatcher->setFuture(QtConcurrent::run([reader, topics_copy]() {
        if (!reader) return;
        reader->buildTopicTimeIndex(topics_copy, nullptr, 0);
    }));
}

bool ManualSensorCalibDlg::needsImagesForView() const {
    return m_viewMode == 0 || m_viewMode == 1;
}

bool ManualSensorCalibDlg::needsCloudForView() const { return m_viewMode >= 1; }

std::vector<std::pair<std::string, std::string>>
ManualSensorCalibDlg::getCameraTopicsForAlignment() const {
    std::vector<std::pair<std::string, std::string>> result;
    const bool tuning_is_pan = (m_currentSensor.compare(0, 3, "pan") == 0);

    for (const auto& [topic, name] : m_cameraTopics) {
        bool include = false;

        if (m_sensorType == "lidar") {
            include = true;
        } else if (m_calibMode == MODE_SINGLE_CAMERA) {
            include = (name == m_currentSensor);
        } else if (m_calibMode == MODE_AVM_CAMERA) {
            include = (name.compare(0, 3, "pan") == 0);
        } else if (m_calibMode == MODE_SVM_CAMERA) {
            include = (name.compare(0, 7, "camera_") == 0) ||
                      (name.compare(0, 3, "tra") == 0);
        } else if (m_calibMode == MODE_ALL_CAMERA) {
            if (tuning_is_pan) {
                include = (name.compare(0, 3, "pan") == 0);
            } else {
                include = (name.compare(0, 3, "tra") == 0) ||
                          (name.compare(0, 7, "camera_") == 0) ||
                          (name.compare(0, 3, "pan") == 0);
            }
        }

        if (m_viewMode == 0 && m_sensorType == "camera" &&
            m_calibMode != MODE_ALL_CAMERA) {
            const auto slot_map = getBevCameraSlotMap();
            include = false;
            for (const auto& [_, source] : slot_map) {
                if (source == name) {
                    include = true;
                    break;
                }
            }
        }

        if (m_viewMode == 1) {
            std::string proj_cam = m_currentSensor;
            if (m_sensorType == "lidar") {
                if (!m_calibConfig.cameras.count(proj_cam) &&
                    !m_calibConfig.cameras.empty()) {
                    proj_cam = m_calibConfig.cameras.begin()->first;
                }
            }
            include = (name == proj_cam);
        }

        if (m_viewMode == 3) {
            include = false;
        }

        if (include) {
            result.emplace_back(topic, name);
        }
    }
    return result;
}

void ManualSensorCalibDlg::collectBevImageTopicGroups(
        std::vector<std::string>& svm_topics,
        std::vector<std::string>& avm_topics) const {
    svm_topics.clear();
    avm_topics.clear();
    for (const auto& [topic, name] : m_cameraTopics) {
        if (name.compare(0, 3, "pan") == 0) {
            avm_topics.push_back(topic);
        } else {
            svm_topics.push_back(topic);
        }
    }
    std::sort(svm_topics.begin(), svm_topics.end());
    std::sort(avm_topics.begin(), avm_topics.end());
}

void ManualSensorCalibDlg::syncBagSliderToPercent(double percent) {
    if (!m_sliderTimePos) return;
    const int val = std::clamp(static_cast<int>(std::lround(percent * 1000.0)),
                               0, 1000);
    const QSignalBlocker blocker(m_sliderTimePos);
    m_sliderTimePos->setValue(val);
    updateTimeSliderLabel(val);
    m_pendingSliderValue = val;
    m_appliedSliderValue = val;
}

bool ManualSensorCalibDlg::seekFirstValidBagFrame(double& out_percent,
                                                  double search_step) const {
    out_percent = 0.0;
    if (!m_bagReader || !m_bagReader->isOpen()) return false;

    const auto align_topics = getCameraTopicsForAlignment();
    if (align_topics.empty()) return false;

    std::vector<std::string> image_topics;
    image_topics.reserve(align_topics.size());
    for (const auto& [topic, _] : align_topics) {
        image_topics.push_back(topic);
    }

    std::vector<std::string> svm_topics;
    std::vector<std::string> avm_topics;
    mcalib::partitionCameraImageTopics(image_topics, svm_topics, avm_topics);
    const bool use_bev_groups =
            !svm_topics.empty() && !avm_topics.empty() &&
            svm_topics.size() + avm_topics.size() == image_topics.size();

    int64_t ref_stamp_ns = 0;
    if (use_bev_groups) {
        return mcalib::findBestAlignedPercentBevGroups(
                *m_bagReader, svm_topics, avm_topics, 0.0, 1.0, out_percent,
                ref_stamp_ns, search_step);
    }
    return mcalib::findBestAlignedPercent(*m_bagReader, image_topics, 0.0, 1.0,
                                          out_percent, ref_stamp_ns,
                                          search_step);
}

void ManualSensorCalibDlg::loadInitialBagFrame() {
    if (!m_bagReader || !m_bagReader->isOpen()) return;

    double percent = 0.0;
    loadAlignedFrameFromBag(percent);

    if (m_images.empty() && needsImagesForView()) {
        constexpr double kInitialSeekStep = 0.05;
        if (seekFirstValidBagFrame(percent, kInitialSeekStep)) {
            CVLog::Print(
                    "[SensorCalib] Auto-seek first valid frame @%.1f%% "
                    "(initial 0%% had no synced images)",
                    percent * 100.0);
            loadAlignedFrameFromBag(percent);
        } else {
            CVLog::Warning(
                    "[SensorCalib] No synced camera frame found in bag; "
                    "scrub timeline to locate valid data");
        }
    }

    syncBagSliderToPercent(percent);
    CVLog::Print(
            "[SensorCalib] loadAlignedFrame: images=%zu, raw_cloud=%zu, "
            "air_susp=%d @%.1f%%",
            m_images.size(), m_pointCloudRaw.size(),
            m_vehicleState.has_air_susp_report ? 1 : 0, percent * 100.0);
}

void ManualSensorCalibDlg::applyAlignedTopicImages(
        const std::map<std::string, cv::Mat>& images_by_topic,
        const std::map<std::string, int64_t>& stamps_ns) {
    for (const auto& [topic, cam_name] : m_cameraTopics) {
        auto it_img = images_by_topic.find(topic);
        auto it_stamp = stamps_ns.find(topic);
        if (it_img == images_by_topic.end() || it_stamp == stamps_ns.end()) {
            continue;
        }

        cv::Mat img = it_img->second;
        auto it_cam = m_calibConfig.cameras.find(cam_name);
        if (it_cam != m_calibConfig.cameras.end()) {
            const int w = it_cam->second.intrinsic.width;
            const int h = it_cam->second.intrinsic.height;
            if (w > 0 && h > 0 && !img.empty() &&
                (img.cols != w || img.rows != h)) {
                cv::resize(img, img, cv::Size(w, h));
            }
        }

        m_images[cam_name] = std::move(img);
        m_imageStampsUs[cam_name] = it_stamp->second / 1000;
    }
}

bool ManualSensorCalibDlg::isCombinedCloud() const { return m_isCombinedCloud; }

std::vector<Eigen::Isometry3d> ManualSensorCalibDlg::getLidarFinalExtrinsic()
        const {
    std::vector<Eigen::Isometry3d> extrinsic_gnss_lidar(
            m_calibConfig.lidars.size());
    const bool is_combined = isCombinedCloud();

    for (size_t i = 0; i < m_calibConfig.lidars.size(); ++i) {
        mcalib::Vector6d delta = mcalib::Vector6d::Zero();
        if (i < m_deltaLidarExtrinsics.size()) {
            delta = m_deltaLidarExtrinsics[i];
        }
        mcalib::Vector6d delta_rad = delta;
        delta_rad.segment(0, 3) *= (M_PI / 180.0);

        Eigen::Isometry3d iso_tune;
        mcalib::Vec2Isometry(delta_rad, iso_tune);

        if (!is_combined) {
            extrinsic_gnss_lidar[i] =
                    m_calibConfig.iso_sensing_vehicle.inverse() * iso_tune *
                    m_calibConfig.lidars[i].extrinsic;
        } else {
            extrinsic_gnss_lidar[i] =
                    m_calibConfig.iso_sensing_vehicle.inverse() * iso_tune;
        }
    }
    return extrinsic_gnss_lidar;
}

void ManualSensorCalibDlg::applyCloudFrameMetadata(
        const std::string& frame_id) {
    m_cloudFrameId = frame_id;
    m_isCombinedCloud = frame_id.empty() || frame_id == "lidar";
    if (!frame_id.empty() && frame_id != "lidar" &&
        frame_id != "lidar_uncalibrated") {
        CVLog::Warning("[SensorCalib] cloud frame_id: %s", frame_id.c_str());
    }
}

void ManualSensorCalibDlg::loadCloudForCurrentTimestamp() {
    if (!m_bagReader || !m_bagReader->isOpen() || m_images.empty()) {
        CVLog::Warning(
                "[SensorCalib] loadCloudForCurrentTimestamp: bag/images "
                "unavailable");
        return;
    }

    std::string ref_cam = m_currentSensor;
    if (m_sensorType == "lidar" && !m_calibConfig.cameras.empty()) {
        if (!m_calibConfig.cameras.count(ref_cam)) {
            ref_cam = m_calibConfig.cameras.begin()->first;
        }
    }

    int64_t ref_stamp_ns = 0;
    auto it_ts = m_imageStampsUs.find(ref_cam);
    if (it_ts != m_imageStampsUs.end()) {
        ref_stamp_ns = it_ts->second * 1000;
    } else {
        for (const auto& [_, stamp_us] : m_imageStampsUs) {
            ref_stamp_ns = stamp_us * 1000;
            break;
        }
    }
    if (ref_stamp_ns == 0) {
        CVLog::Warning(
                "[SensorCalib] loadCloudForCurrentTimestamp: no image stamp");
        return;
    }

    std::string cam_topic;
    for (const auto& [topic, name] : m_cameraTopics) {
        if (name == ref_cam) {
            cam_topic = topic;
            break;
        }
    }

    int64_t ref_bag_stamp_ns = 0;
    if (!cam_topic.empty() && ref_stamp_ns > 0) {
        const auto msg = m_bagReader->readMessageNearestTime(
                cam_topic, static_cast<uint64_t>(ref_stamp_ns),
                static_cast<uint64_t>(mcalib::kAlignSearchRangeNs));
        if (!msg.data.empty()) {
            ref_bag_stamp_ns = static_cast<int64_t>(msg.timestamp_ns);
        }
    }
    if (ref_bag_stamp_ns <= 0 && !cam_topic.empty() && m_sliderTimePos) {
        const double percent = m_sliderTimePos->value() / 1000.0;
        const auto msg = m_bagReader->readMessageAtPercent(cam_topic, percent);
        if (!msg.data.empty()) {
            ref_bag_stamp_ns = static_cast<int64_t>(msg.timestamp_ns);
        }
    }

    std::vector<mcalib::PointXYZIRT> cloud_raw;
    int64_t cloud_stamp_us = 0;
    std::string frame_id;
    if (!mcalib::loadCloudNearImageStamp(
                *m_bagReader, m_cloudTopics, ref_stamp_ns, cloud_raw,
                cloud_stamp_us, &frame_id, ref_bag_stamp_ns, &m_calibConfig)) {
        return;
    }

    m_pointCloudRaw = std::move(cloud_raw);
    m_pointCloud.clear();
    m_cloudStampUs = cloud_stamp_us;
    applyCloudFrameMetadata(frame_id);
}

bool ManualSensorCalibDlg::loadCloudAtBagPercent(double percent) {
    if (!m_bagReader || !m_bagReader->isOpen()) return false;
    if (m_cloudTopics.empty()) buildBagTopicLists();

    std::string ref_topic;
    for (const auto& [topic, name] : m_cameraTopics) {
        if (name == mcalib::kCamera1) {
            ref_topic = topic;
            break;
        }
    }
    if (ref_topic.empty()) {
        for (const auto& [topic, name] : m_cameraTopics) {
            if (name.compare(0, 7, "camera_") == 0) {
                ref_topic = topic;
                break;
            }
        }
    }
    if (ref_topic.empty() && !m_cameraTopics.empty()) {
        ref_topic = m_cameraTopics.front().first;
    }

    int64_t ref_stamp_ns = 0;
    int64_t ref_bag_stamp_ns = 0;
    if (!ref_topic.empty()) {
        const auto msg = m_bagReader->readMessageAtPercent(ref_topic, percent);
        if (!msg.data.empty()) {
            ref_bag_stamp_ns = static_cast<int64_t>(msg.timestamp_ns);
            ref_stamp_ns = ref_bag_stamp_ns;
            double ts_sec = 0;
            if (mcalib::ProtoDecoder::extractCompressedImageTimestampFromBag(
                        msg.data, ts_sec) &&
                ts_sec > 0) {
                const int64_t proto_ns = static_cast<int64_t>(ts_sec * 1e9);
                const int64_t diff = proto_ns > ref_bag_stamp_ns
                                             ? proto_ns - ref_bag_stamp_ns
                                             : ref_bag_stamp_ns - proto_ns;
                if (diff <= 60LL * 1000000000LL) {
                    ref_stamp_ns = proto_ns;
                }
            }
        }
    }

    std::vector<mcalib::PointXYZIRT> cloud_raw;
    int64_t cloud_stamp_us = 0;
    std::string frame_id;
    if (!mcalib::getAlignedCloud(*m_bagReader, m_cloudTopics, ref_stamp_ns,
                                 false, cloud_raw, cloud_stamp_us, &frame_id,
                                 ref_bag_stamp_ns, &m_calibConfig)) {
        return false;
    }

    m_pointCloudRaw = std::move(cloud_raw);
    m_pointCloud.clear();
    m_cloudStampUs = cloud_stamp_us;
    applyCloudFrameMetadata(frame_id);
    return !m_pointCloudRaw.empty();
}

void ManualSensorCalibDlg::loadAlignedFrameFromBag(double percent) {
    if (!m_bagReader || !m_bagReader->isOpen()) return;
    if (m_cameraTopics.empty() || m_cloudTopics.empty()) {
        buildBagTopicLists();
    }

    m_pointCloud.clear();
    m_pointCloudRaw.clear();
    m_images.clear();
    m_imageStampsUs.clear();

    const auto align_topics = getCameraTopicsForAlignment();
    std::vector<std::string> image_topics;
    image_topics.reserve(align_topics.size());
    for (const auto& [topic, _] : align_topics) {
        image_topics.push_back(topic);
    }

    std::map<std::string, cv::Mat> images_by_topic;
    std::map<std::string, int64_t> stamps_ns;
    std::vector<mcalib::PointXYZIRT> cloud_raw;
    int64_t cloud_stamp_us = 0;
    std::string frame_id;

    const bool bev_mode = (m_viewMode == 0);
    const bool allow_cloud_ref = (m_viewMode == 2);

    if (bev_mode) {
        int64_t cloud_stamp_ns = 0;
        if (!image_topics.empty() &&
            mcalib::getAlignedImagesForBev(*m_bagReader, image_topics, percent,
                                           images_by_topic, stamps_ns,
                                           cloud_stamp_ns, &m_vehicleState)) {
            applyAlignedTopicImages(images_by_topic, stamps_ns);
            m_cloudStampUs = cloud_stamp_ns / 1000;
        }
    } else if (!image_topics.empty() &&
               mcalib::getAlignedImagesCloud(
                       *m_bagReader, image_topics, m_cloudTopics, percent,
                       allow_cloud_ref, images_by_topic, stamps_ns, cloud_raw,
                       cloud_stamp_us, &frame_id, &m_vehicleState,
                       &m_calibConfig)) {
        applyAlignedTopicImages(images_by_topic, stamps_ns);
        if (!cloud_raw.empty()) {
            m_pointCloudRaw = std::move(cloud_raw);
            m_cloudStampUs = cloud_stamp_us;
            applyCloudFrameMetadata(frame_id);
        }
    }

    CVLog::Print(
            "[SensorCalib] loadAlignedFrame: images=%zu, raw_cloud=%zu, "
            "air_susp=%d",
            m_images.size(), m_pointCloudRaw.size(),
            m_vehicleState.has_air_susp_report ? 1 : 0);
}

std::vector<std::pair<std::string, std::string>>
ManualSensorCalibDlg::getCameraTopicsForSlider() const {
    if (m_viewMode == 1) {
        std::string cam_name = m_currentSensor;
        if (m_sensorType == "lidar" && !m_calibConfig.cameras.empty()) {
            cam_name = m_calibConfig.cameras.begin()->first;
        }
        for (const auto& topic_pair : m_cameraTopics) {
            if (topic_pair.second == cam_name) {
                return {topic_pair};
            }
        }
    }
    return m_cameraTopics;
}

void ManualSensorCalibDlg::updateTimeSliderLabel(int sliderValue) {
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

void ManualSensorCalibDlg::processSliderLoad() {
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
    const int modeGeneration = m_modeGeneration.load();

    const double percent = m_pendingSliderValue / 1000.0;
    const bool needImages = needsImagesForView();
    const bool needCloud = needsCloudForView();
    const auto cameraTopics = getCameraTopicsForAlignment();
    const auto allCameraTopics = m_cameraTopics;
    const auto cloudTopics = m_cloudTopics;
    const int viewMode = m_viewMode;
    const bool bev_mode = (viewMode == 0);
    const bool allow_cloud_ref = (viewMode == 2);
    mcalib::RosBagReader* reader = m_bagReader.get();
    const auto calibConfig = m_calibConfig;

    QFuture<SliderFrameData> future =
            QtConcurrent::run([=]() -> SliderFrameData {
                SliderFrameData data;

                if (!reader || !reader->isOpen()) return data;

                std::vector<std::string> image_topics;
                image_topics.reserve(cameraTopics.size());
                for (const auto& [topic, _] : cameraTopics) {
                    image_topics.push_back(topic);
                }

                std::map<std::string, cv::Mat> images_by_topic;
                std::map<std::string, int64_t> stamps_ns;
                std::string frame_id;
                mcalib::VehicleStateData vehicle_state;

                if (bev_mode) {
                    int64_t cloud_stamp_ns = 0;
                    if (image_topics.empty() ||
                        !mcalib::getAlignedImagesForBev(
                                *reader, image_topics, percent, images_by_topic,
                                stamps_ns, cloud_stamp_ns, &vehicle_state)) {
                        return data;
                    }
                    data.cloudStampUs = cloud_stamp_ns / 1000;
                    data.vehicleState = vehicle_state;
                } else if (needCloud && viewMode == 3) {
                    data.attemptedCloudLoad = true;
                    std::string ref_topic;
                    for (const auto& [topic, name] : allCameraTopics) {
                        if (name == mcalib::kCamera1) {
                            ref_topic = topic;
                            break;
                        }
                    }
                    if (ref_topic.empty() && !allCameraTopics.empty()) {
                        ref_topic = allCameraTopics.front().first;
                    }
                    int64_t ref_stamp_ns = 0;
                    int64_t ref_bag_stamp_ns = 0;
                    if (!ref_topic.empty()) {
                        const auto msg = reader->readMessageAtPercent(ref_topic,
                                                                      percent);
                        if (!msg.data.empty()) {
                            ref_bag_stamp_ns =
                                    static_cast<int64_t>(msg.timestamp_ns);
                            ref_stamp_ns = ref_bag_stamp_ns;
                            double ts_sec = 0;
                            if (mcalib::ProtoDecoder::
                                        extractCompressedImageTimestampFromBag(
                                                msg.data, ts_sec) &&
                                ts_sec > 0) {
                                const int64_t proto_ns =
                                        static_cast<int64_t>(ts_sec * 1e9);
                                const int64_t diff =
                                        proto_ns > ref_bag_stamp_ns
                                                ? proto_ns - ref_bag_stamp_ns
                                                : ref_bag_stamp_ns - proto_ns;
                                if (diff <= 60LL * 1000000000LL) {
                                    ref_stamp_ns = proto_ns;
                                }
                            }
                        }
                    }
                    if (!mcalib::getAlignedCloud(
                                *reader, cloudTopics, ref_stamp_ns, false,
                                data.pointCloudRaw, data.cloudStampUs,
                                &frame_id, ref_bag_stamp_ns, &calibConfig)) {
                        return data;
                    }
                    data.cloudFrameId = frame_id;
                    data.isCombinedCloud =
                            frame_id.empty() || frame_id == "lidar";
                } else if (needCloud) {
                    data.attemptedCloudLoad = true;
                    if (!mcalib::getAlignedImagesCloud(
                                *reader, image_topics, cloudTopics, percent,
                                allow_cloud_ref, images_by_topic, stamps_ns,
                                data.pointCloudRaw, data.cloudStampUs,
                                &frame_id, &vehicle_state, &calibConfig)) {
                        return data;
                    }
                    data.cloudFrameId = frame_id;
                    data.isCombinedCloud =
                            frame_id.empty() || frame_id == "lidar";
                    data.vehicleState = vehicle_state;
                } else if (needImages) {
                    if (!mcalib::getAlignedImages(*reader, image_topics,
                                                  percent, images_by_topic,
                                                  stamps_ns, &vehicle_state)) {
                        return data;
                    }
                    data.vehicleState = vehicle_state;
                }

                for (const auto& [topic, cam_name] : allCameraTopics) {
                    auto it_img = images_by_topic.find(topic);
                    auto it_stamp = stamps_ns.find(topic);
                    if (it_img == images_by_topic.end() ||
                        it_stamp == stamps_ns.end()) {
                        continue;
                    }

                    cv::Mat img = it_img->second;
                    auto it_cam = calibConfig.cameras.find(cam_name);
                    if (it_cam != calibConfig.cameras.end()) {
                        const int w = it_cam->second.intrinsic.width;
                        const int h = it_cam->second.intrinsic.height;
                        if (w > 0 && h > 0 && !img.empty() &&
                            (img.cols != w || img.rows != h)) {
                            cv::resize(img, img, cv::Size(w, h));
                        }
                    }

                    data.images[cam_name] = std::move(img);
                    data.imageStampsUs[cam_name] = it_stamp->second / 1000;
                }

                return data;
            });

    if (!m_sliderLoadWatcher) {
        m_sliderLoadWatcher = new QFutureWatcher<SliderFrameData>(this);
        connect(m_sliderLoadWatcher, &QFutureWatcher<SliderFrameData>::finished,
                this, [this]() {
                    const int generation =
                            m_sliderLoadWatcher->property("generation").toInt();
                    const int modeGen =
                            m_sliderLoadWatcher->property("modeGeneration")
                                    .toInt();
                    if (generation != m_sliderLoadGeneration.load()) {
                        m_loadingBag.store(false);
                        if (m_sliderReloadPending) {
                            m_sliderReloadPending = false;
                            processSliderLoad();
                        }
                        return;
                    }

                    auto data = m_sliderLoadWatcher->result();
                    // If calib-mode / view-mode / sensor-name changed while
                    // this load was in flight, the loaded camera set may not
                    // match the current slot map. We still merge the images
                    // (cheap) but skip the immediate updateFusionView() — the
                    // pending reloadCurrentBagFrame() will refresh and render.
                    const bool modeChanged =
                            (modeGen != m_modeGeneration.load());
                    applySliderFrameData(
                            data.images, data.imageStampsUs, data.pointCloud,
                            data.pointCloudRaw, data.cloudStampUs,
                            data.isCombinedCloud, data.cloudFrameId,
                            data.vehicleState, data.attemptedCloudLoad);
                    m_appliedSliderValue = m_pendingSliderValue;
                    if (!modeChanged) {
                        updateFusionView();
                        if (m_app) {
                            m_app->redrawAll(false, true);
                            m_app->refreshAll(false, true);
                        }
                    }

                    m_loadingBag.store(false);
                    if (m_sliderReloadPending || modeChanged) {
                        m_sliderReloadPending = false;
                        // Re-capture current mode's topics for the reload.
                        m_appliedSliderValue = -1;
                        processSliderLoad();
                    }
                });
    }

    m_sliderLoadWatcher->setProperty("generation", generation);
    m_sliderLoadWatcher->setProperty("modeGeneration", modeGeneration);
    m_sliderLoadWatcher->setFuture(future);
}

void ManualSensorCalibDlg::applySliderFrameData(
        const std::map<std::string, cv::Mat>& images,
        const std::map<std::string, int64_t>& imageStampsUs,
        const std::vector<Eigen::Vector3f>& pointCloud,
        const std::vector<mcalib::PointXYZIRT>& pointCloudRaw,
        int64_t cloudStampUs,
        bool isCombinedCloud,
        const std::string& cloudFrameId,
        const mcalib::VehicleStateData& vehicleState,
        bool attemptedCloudLoad) {
    if (!images.empty()) {
        for (const auto& [name, img] : images) {
            if (!img.empty()) {
                m_images[name] = img;
            }
        }
        for (const auto& [name, stamp] : imageStampsUs) {
            m_imageStampsUs[name] = stamp;
        }
    }

    if (!pointCloudRaw.empty()) {
        m_cloudStampUs = cloudStampUs;
        m_pointCloudRaw = pointCloudRaw;
        m_pointCloud.clear();
        applyCloudFrameMetadata(cloudFrameId);
    } else if (!pointCloud.empty()) {
        m_cloudStampUs = cloudStampUs;
        m_pointCloud = pointCloud;
        applyCloudFrameMetadata(cloudFrameId);
    } else if (attemptedCloudLoad) {
        m_cloudStampUs = 0;
        m_pointCloudRaw.clear();
        m_pointCloud.clear();
    }

    if (vehicleState.has_air_susp_report || vehicleState.timestamp_us > 0) {
        m_vehicleState = vehicleState;
    }
}

void ManualSensorCalibDlg::loadImagesFromBag(double percent) {
    loadAlignedFrameFromBag(percent);
}

void ManualSensorCalibDlg::loadPointCloudFromBag(double percent) {
    loadAlignedFrameFromBag(percent);
}

void ManualSensorCalibDlg::loadVehicleTrajectory() {
    if (!m_bagReader || !m_bagReader->isOpen()) return;

    m_vehiclePoses.clear();
    const std::string pose_topic = "/localization/pose";

    m_bagReader->readMessages(
            [&](const mcalib::BagMessage& msg) {
                mcalib::ProtoDecoder::InsPose ins;
                if (mcalib::ProtoDecoder::decodeInsPoseFromBag(msg.data, ins)) {
                    mcalib::Vector6d v;
                    v << ins.pos_x, ins.pos_y, ins.pos_z, ins.euler_x,
                            ins.euler_y, ins.euler_z;
                    Eigen::Isometry3d iso;
                    mcalib::xyzeuler2isometry(v, iso);
                    m_vehiclePoses.emplace_back(ins.measurement_time_us, iso);
                }
                return true;
            },
            {pose_topic});

    CVLog::Print("[SensorCalib] Vehicle trajectory: %zu poses loaded",
                 m_vehiclePoses.size());
}

void ManualSensorCalibDlg::onSaveConfig() {
    if (!m_configLoaded || m_configPath.isEmpty()) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Load config directory first"));
        return;
    }

    if (m_sensorType == "camera") {
        std::map<std::string, Eigen::Isometry3d> final_extrinsics_cam_sensing;
        auto extrinsics = getFinalExtrinsics();
        for (const auto& [name, ext] : extrinsics) {
            auto it = m_deltaExtrinsics.find(name);
            if (it == m_deltaExtrinsics.end() || it->second.isZero(1e-12)) {
                continue;
            }
            final_extrinsics_cam_sensing[name] = ext.inverse();
        }
        if (final_extrinsics_cam_sensing.empty()) {
            QMessageBox::information(
                    asWidget(m_app->getMainWindow()), tr("Save"),
                    tr("No camera extrinsic adjustments to save."));
            return;
        }
        const std::string output_file =
                m_calibConfig.camera_cfg_path + "_fix.cfg";
        if (mcalib::CalibConfigParser::saveMultiCameraExtrinsic(
                    output_file, final_extrinsics_cam_sensing, m_calibConfig)) {
            CVLog::Print("[SensorCalib] Camera config saved: %s",
                         output_file.c_str());
            m_lblStatus->setText(
                    tr("Camera config saved: %1")
                            .arg(QString::fromStdString(output_file)));
        } else {
            QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                                 tr("Failed to save camera config"));
        }
    } else if (m_sensorType == "lidar") {
        bool has_delta = false;
        for (const auto& delta : m_deltaLidarExtrinsics) {
            if (!delta.isZero(1e-12)) {
                has_delta = true;
                break;
            }
        }
        if (!has_delta) {
            QMessageBox::information(
                    asWidget(m_app->getMainWindow()), tr("Save"),
                    tr("No lidar extrinsic adjustments to save."));
            return;
        }

        std::vector<Eigen::Isometry3d> extrinsics_gnss_lidar =
                getLidarFinalExtrinsic();
        if (isCombinedCloud()) {
            for (size_t i = 0; i < m_calibConfig.lidars.size(); ++i) {
                if (i >= extrinsics_gnss_lidar.size() ||
                    i >= m_deltaLidarExtrinsics.size()) {
                    break;
                }
                mcalib::Vector6d delta_rad = m_deltaLidarExtrinsics[i];
                delta_rad.segment(0, 3) *= (M_PI / 180.0);
                Eigen::Isometry3d iso_tune;
                mcalib::Vec2Isometry(delta_rad, iso_tune);
                extrinsics_gnss_lidar[i] =
                        m_calibConfig.iso_sensing_vehicle.inverse() * iso_tune *
                        m_calibConfig.lidars[i].extrinsic;
            }
        }
        std::string output_file = m_calibConfig.lidar_cfg_path + "_fix.cfg";
        if (mcalib::CalibConfigParser::saveGnssMultiLidarExtrinsic(
                    output_file, extrinsics_gnss_lidar, m_calibConfig,
                    "manual_calib")) {
            CVLog::Print("[SensorCalib] GNSS lidar config saved: %s",
                         output_file.c_str());
            m_lblStatus->setText(tr("GNSS lidar config saved"));
        } else {
            QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                                 tr("Failed to save lidar config"));
        }
    }
}

void ManualSensorCalibDlg::onSavePCD() { exportPointCloudToDB(); }

std::map<std::string, std::string> ManualSensorCalibDlg::getBevCameraSlotMap()
        const {
    bool use_avm = (m_calibMode == MODE_AVM_CAMERA);
    if (m_calibMode != MODE_AVM_CAMERA && m_calibMode != MODE_SVM_CAMERA) {
        use_avm = (m_currentSensor.compare(0, 3, "pan") == 0);
    }

    const std::map<std::string, std::string>& base_map =
            use_avm ? mcalib::kAvmCameraMap
                    : ((m_calibMode != MODE_AVM_CAMERA &&
                        m_currentSensor == mcalib::kTraffic2)
                               ? mcalib::kSvmT2CameraMap
                               : mcalib::kSvmCameraMap);

    std::map<std::string, std::string> filtered;
    for (const auto& [slot, source] : base_map) {
        if (m_calibConfig.cameras.count(source) > 0) {
            filtered[slot] = source;
        }
    }
    return filtered;
}

bool ManualSensorCalibDlg::prepareBevCameraSubset() {
    if (!m_bevViewer || m_images.empty()) return false;

    const auto slot_map = getBevCameraSlotMap();
    if (slot_map.empty()) return false;

    // Defensive: ensure at least one slot's source has an image in m_images.
    // After a mode/sensor switch, m_images may still hold the previous mode's
    // camera set (e.g. SVM camera_1..6 while the new slot map expects
    // panoramic_1..4). Rendering in that state produces a "0 cameras matched"
    // BEV and the misleading "GPU alpha fusion failed" warning. Skip the render
    // — the pending reloadCurrentBagFrame() will refresh m_images and trigger
    // updateFusionView() once the new frame arrives.
    bool any_source_loaded = false;
    for (const auto& [slot, source] : slot_map) {
        (void)slot;
        auto it = m_images.find(source);
        if (it != m_images.end() && !it->second.empty()) {
            any_source_loaded = true;
            break;
        }
    }
    if (!any_source_loaded) {
        CVLog::Print(
                "[SensorCalib] prepareBevCameraSubset: slot map sources not "
                "in m_images (stale after mode switch?), skipping BEV render");
        return false;
    }

    const bool subsetChanged = m_bevViewer->setCameraSlotMap(slot_map);

    if (subsetChanged || m_extrinsicDirty) {
        auto extrinsics = getFinalExtrinsics();
        m_bevViewer->updateExtrinsics(extrinsics);
        m_extrinsicDirty = false;
    }
    return true;
}

cv::Mat ManualSensorCalibDlg::renderBevImage() {
    if (!prepareBevCameraSubset()) return {};
    m_bevViewer->setFocalScale(0.3 * (m_pointSize + 1));
    return m_bevViewer->generate(m_images);
}

cv::Mat ManualSensorCalibDlg::renderLidarProjImage(
        std::vector<Eigen::Vector3f>* out_points) {
    std::string cam_name = m_currentSensor;
    if (m_sensorType == "lidar" && !m_calibConfig.cameras.empty()) {
        cam_name = m_calibConfig.cameras.begin()->first;
    }
    auto it_cam = m_calibConfig.cameras.find(cam_name);
    if (it_cam == m_calibConfig.cameras.end()) return {};

    auto it_img = m_images.find(cam_name);
    cv::Mat display;
    if (it_img != m_images.end() && !it_img->second.empty()) {
        display = it_img->second.clone();
    } else {
        return {};
    }

    const int target_w = it_cam->second.intrinsic.width;
    const int target_h = it_cam->second.intrinsic.height;
    if (target_w > 0 && target_h > 0 &&
        (display.cols != target_w || display.rows != target_h)) {
        cv::resize(display, display, cv::Size(target_w, target_h));
    }

    std::vector<Eigen::Vector3f> merged;
    if (!m_pointCloudRaw.empty()) {
        std::vector<mcalib::PointXYZIRT> source_raw;
        if (!isCombinedCloud()) {
            for (size_t j = 0; j < m_calibConfig.lidars.size(); ++j) {
                if (j >= m_selectedLidars.size() || !m_selectedLidars[j])
                    continue;
                const auto& lc = m_calibConfig.lidars[j];
                const Eigen::Matrix4f T = lc.extrinsic.matrix().cast<float>();
                for (const auto& pt : m_pointCloudRaw) {
                    if (pt.ring < lc.ring_start || pt.ring > lc.ring_end)
                        continue;
                    mcalib::PointXYZIRT tpt = pt;
                    const Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
                    const Eigen::Vector4f tp = T * p;
                    tpt.x = tp.x();
                    tpt.y = tp.y();
                    tpt.z = tp.z();
                    source_raw.push_back(tpt);
                }
            }
        } else {
            source_raw = m_pointCloudRaw;
        }

        std::vector<mcalib::PointXYZIRT> tuned_raw;
        for (size_t j = 0; j < m_calibConfig.lidars.size(); ++j) {
            if (j >= m_selectedLidars.size() || !m_selectedLidars[j]) continue;
            const auto& lc = m_calibConfig.lidars[j];
            mcalib::Vector6d delta = mcalib::Vector6d::Zero();
            if (j < m_deltaLidarExtrinsics.size())
                delta = m_deltaLidarExtrinsics[j];
            mcalib::Vector6d delta_rad = delta;
            delta_rad.segment(0, 3) *= (M_PI / 180.0);
            Eigen::Isometry3d iso_tune;
            mcalib::Vec2Isometry(delta_rad, iso_tune);
            Eigen::Matrix4f T = iso_tune.matrix().cast<float>();

            for (const auto& pt : source_raw) {
                if (pt.ring < lc.ring_start || pt.ring > lc.ring_end) continue;
                Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
                Eigen::Vector4f tp = T * p;
                mcalib::PointXYZIRT tpt = pt;
                tpt.x = tp.x();
                tpt.y = tp.y();
                tpt.z = tp.z();
                tuned_raw.push_back(tpt);
            }
        }
        if (m_calibConfig.lidars.empty()) {
            tuned_raw = source_raw;
        }

        int64_t target_stamp = m_cloudStampUs;
        if (!m_imageStampsUs.empty()) {
            auto it_ts = m_imageStampsUs.find(cam_name);
            if (it_ts != m_imageStampsUs.end())
                target_stamp = it_ts->second;
            else
                target_stamp = m_imageStampsUs.begin()->second;
        }

        if (!m_vehiclePoses.empty() && !tuned_raw.empty()) {
            std::vector<Eigen::Vector3f> undistorted;
            const Eigen::Isometry3d iso_vehicle_sensing =
                    m_calibConfig.iso_sensing_vehicle.inverse();
            if (mcalib::undistortPointCloud(tuned_raw, m_cloudStampUs,
                                            target_stamp, m_vehiclePoses,
                                            iso_vehicle_sensing, undistorted)) {
                for (const auto& pt : undistorted) {
                    float dist = pt.norm();
                    if (dist >= m_distFilterMin && dist <= m_distFilter)
                        merged.push_back(pt);
                }
            } else {
                for (const auto& pt : tuned_raw) {
                    float dist =
                            std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
                    if (dist >= m_distFilterMin && dist <= m_distFilter)
                        merged.emplace_back(pt.x, pt.y, pt.z);
                }
            }
        } else {
            for (const auto& pt : tuned_raw) {
                float dist = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
                if (dist >= m_distFilterMin && dist <= m_distFilter)
                    merged.emplace_back(pt.x, pt.y, pt.z);
            }
        }
    } else {
        merged = getFilteredPointCloud();
    }

    if (!merged.empty()) {
        auto extrinsics = getFinalExtrinsics();
        auto it_ext = extrinsics.find(cam_name);
        if (it_ext != extrinsics.end()) {
            cv::Mat K = it_cam->second.intrinsic.getCameraMatrix();
            cv::Mat D = it_cam->second.intrinsic.getDistCoeffs();
            lidarCamFusion(merged, display, cam_name, it_ext->second.inverse(),
                           K, D, m_pointSize - 1, getGpuBackendMode());
        }
    }

    if (out_points) *out_points = std::move(merged);
    return display;
}

std::vector<Eigen::Vector3f>
ManualSensorCalibDlg::getDisplayedPointCloudForExport() const {
    if (m_viewMode == 1) {
        ManualSensorCalibDlg* self = const_cast<ManualSensorCalibDlg*>(this);
        std::vector<Eigen::Vector3f> points;
        self->renderLidarProjImage(&points);
        return points;
    }
    if (m_viewMode == 3) {
        std::vector<Eigen::Vector3f> points;
        points.reserve(m_pointCloudRaw.size());
        for (const auto& pt : m_pointCloudRaw) {
            points.emplace_back(pt.x, pt.y, pt.z);
        }
        return points;
    }
    if (m_viewMode == 2) {
        if (!m_pointCloudRaw.empty() && !m_calibConfig.lidars.empty()) {
            const auto extrinsic_gnss_lidar = getLidarFinalExtrinsic();
            std::vector<Eigen::Vector3f> merged;
            for (size_t j = 0; j < m_calibConfig.lidars.size(); ++j) {
                if (j >= m_selectedLidars.size() || !m_selectedLidars[j])
                    continue;
                const auto& lc = m_calibConfig.lidars[j];
                if (j >= extrinsic_gnss_lidar.size()) continue;

                const Eigen::Matrix4f T =
                        extrinsic_gnss_lidar[j].matrix().cast<float>();
                std::vector<mcalib::PointXYZIRT> single_raw;
                for (const auto& pt : m_pointCloudRaw) {
                    if (pt.ring < lc.ring_start || pt.ring > lc.ring_end)
                        continue;
                    mcalib::PointXYZIRT tpt = pt;
                    const Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
                    const Eigen::Vector4f tp = T * p;
                    tpt.x = tp.x();
                    tpt.y = tp.y();
                    tpt.z = tp.z();
                    single_raw.push_back(tpt);
                }

                std::vector<Eigen::Vector3f> undistorted;
                bool did_undistort = false;
                if (!m_vehiclePoses.empty()) {
                    did_undistort = mcalib::undistortPointCloud(
                            single_raw, m_cloudStampUs, m_cloudStampUs,
                            m_vehiclePoses, Eigen::Isometry3d::Identity(),
                            undistorted);
                }
                if (did_undistort) {
                    for (const auto& pt : undistorted) {
                        if (pt.z() < m_groundFilterMin ||
                            pt.z() > m_groundFilterMax)
                            continue;
                        merged.push_back(pt);
                    }
                } else {
                    for (const auto& pt : single_raw) {
                        if (pt.z < m_groundFilterMin ||
                            pt.z > m_groundFilterMax)
                            continue;
                        merged.emplace_back(pt.x, pt.y, pt.z);
                    }
                }
            }
            if (!merged.empty()) return merged;
        }
        return getFilteredPointCloud();
    }
    return {};
}

ccImage* ManualSensorCalibDlg::createImageEntityFromMat(
        const cv::Mat& bgr, const QString& name) const {
    if (bgr.empty()) return nullptr;

    cv::Mat rgb;
    if (bgr.channels() == 1) {
        cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
    } else if (bgr.channels() == 4) {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    }

    QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                QImage::Format_RGB888);
    auto* exported = new ccImage(qimg.copy(), name);
    exported->setVisible(true);
    exported->setEnabled(false);
    return exported;
}

ccPointCloud* ManualSensorCalibDlg::createPointCloudEntity(
        const std::vector<Eigen::Vector3f>& points, const QString& name) const {
    if (points.empty()) return nullptr;

    auto* cloud = new ccPointCloud(name);
    const unsigned n = static_cast<unsigned>(points.size());
    if (!cloud->reserve(n)) {
        delete cloud;
        return nullptr;
    }

    auto* sf = new ccScalarField("Depth");
    if (!sf->reserveSafe(n)) {
        sf->release();
        delete cloud;
        return nullptr;
    }

    for (unsigned i = 0; i < n; ++i) {
        cloud->addPoint(
                CCVector3(static_cast<PointCoordinateType>(points[i].x()),
                          static_cast<PointCoordinateType>(points[i].y()),
                          static_cast<PointCoordinateType>(points[i].z())));
        sf->addElement(static_cast<ScalarType>(points[i].norm()));
    }
    sf->computeMinAndMax();
    int sfIdx = cloud->addScalarField(sf);
    cloud->setCurrentDisplayedScalarField(sfIdx);
    cloud->showSF(true);
    cloud->setPointSize(static_cast<unsigned char>(std::max(0, m_pointSize)));
    cloud->setVisible(true);
    cloud->setEnabled(false);
    return cloud;
}

namespace {

QString manualCalibSensorTag(const std::string& sensorType) {
    if (sensorType == "lidar") return QStringLiteral("Lidar");
    if (sensorType == "radar") return QStringLiteral("Radar");
    return QStringLiteral("Camera");
}

QString manualCalibViewTag(int viewMode) {
    switch (viewMode) {
        case 0:
            return QStringLiteral("BEV");
        case 1:
            return QStringLiteral("LidarProj");
        case 2:
            return QStringLiteral("SingleFrame");
        case 3:
            return QStringLiteral("MultiFrame");
        default:
            return QStringLiteral("View");
    }
}

QString manualCalibConfigTag(const QString& configPath) {
    const QString tag = ecvPluginDbNaming::sanitizeSegment(
            QFileInfo(configPath).completeBaseName(), 20);
    return tag.isEmpty() ? QStringLiteral("NoCfg") : tag;
}

QString buildManualCalibDbBaseName(const std::string& sensorType,
                                   int viewMode,
                                   const QString& configPath,
                                   const QString& entityKind) {
    return QStringLiteral("MCalib_%1_%2_%3_%4")
            .arg(manualCalibSensorTag(sensorType), manualCalibViewTag(viewMode),
                 manualCalibConfigTag(configPath), entityKind);
}

}  // namespace

void ManualSensorCalibDlg::exportPointCloudToDB() {
    if (!m_app || m_viewMode == 0) return;

    auto points = getDisplayedPointCloudForExport();
    if (points.empty()) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("No point cloud to export"));
        return;
    }

    const QString baseName = buildManualCalibDbBaseName(
            m_sensorType, m_viewMode, m_configPath, QStringLiteral("PCD"));
    const QString name = ecvPluginDbNaming::makeUnique(baseName, m_app);
    ccPointCloud* cloud = createPointCloudEntity(points, name);
    if (!cloud) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Failed to create point cloud entity"));
        return;
    }

    m_app->addToDB(cloud, false, true, false, true);
    m_lblStatus->setText(
            tr("Exported %1 points to DB: %2").arg(points.size()).arg(name));
    CVLog::Print("[SensorCalib] Exported %zu points to DB as %s", points.size(),
                 name.toStdString().c_str());
}

void ManualSensorCalibDlg::exportImageToDB() {
    if (!m_app) return;
    if (m_viewMode != 0 && m_viewMode != 1) return;

    cv::Mat image;
    QString entityKind;
    if (m_viewMode == 0) {
        image = m_lastExportImage.empty() ? renderBevImage()
                                          : m_lastExportImage.clone();
        entityKind = QStringLiteral("BEV");
    } else {
        image = m_lastExportImage.empty() ? renderLidarProjImage()
                                          : m_lastExportImage.clone();
        entityKind = QStringLiteral("ProjImg");
    }

    if (image.empty()) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("No image available to export"));
        return;
    }

    const QString baseName = buildManualCalibDbBaseName(
            m_sensorType, m_viewMode, m_configPath, entityKind);
    const QString name = ecvPluginDbNaming::makeUnique(baseName, m_app);
    ccImage* exported = createImageEntityFromMat(image, name);
    if (!exported) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Failed to create image entity"));
        return;
    }

    m_app->addToDB(exported, false, true, false, true);
    m_app->redrawAll(true);
    m_app->refreshAll(true);
    m_lblStatus->setText(tr("Exported image to DB: %1").arg(name));
    CVLog::Print("[SensorCalib] Exported image %dx%d to DB as %s", image.cols,
                 image.rows, name.toStdString().c_str());
}

void ManualSensorCalibDlg::onResetParams() {
    for (auto& [name, delta] : m_deltaExtrinsics) {
        (void)name;
        delta.setZero();
    }
    for (auto& delta : m_deltaLidarExtrinsics) {
        delta.setZero();
    }
    m_deltaExtrinsic.setZero();
    m_extrinsicDirty = true;
    displayDeltaExtrinsic(m_currentSensor);
    updateFusionView();
}

void ManualSensorCalibDlg::initCameraDeltaExtrinsics() {
    m_deltaExtrinsics.clear();
    for (const auto& [name, _] : m_calibConfig.cameras) {
        m_deltaExtrinsics[name] = mcalib::Vector6d::Zero();
    }
    m_deltaExtrinsic.setZero();
}

void ManualSensorCalibDlg::updateSensorList() {
    const QSignalBlocker blockSensorCombo(m_cmbSensorName);
    m_cmbSensorName->clear();
    if (m_sensorType == "camera") {
        for (const auto& [name, _] : m_calibConfig.cameras) {
            if (m_calibMode == MODE_AVM_CAMERA &&
                name.compare(0, 3, "pan") != 0)
                continue;
            if (m_calibMode == MODE_SVM_CAMERA &&
                name.compare(0, 3, "pan") == 0)
                continue;
            m_cmbSensorName->addItem(QString::fromStdString(name));
        }
    } else if (m_sensorType == "lidar") {
        if (m_deltaLidarExtrinsics.size() != m_calibConfig.lidars.size()) {
            m_deltaLidarExtrinsics.resize(m_calibConfig.lidars.size(),
                                          mcalib::Vector6d::Zero());
        }
        for (size_t i = 0; i < m_calibConfig.lidars.size(); i++) {
            m_cmbSensorName->addItem(
                    tr("lidar_%1").arg(m_calibConfig.lidars[i].lidar_idx));
        }
    }
}

void ManualSensorCalibDlg::refreshCalibModeCombo() {
    if (!m_cmbCalibMode) return;

    m_cmbCalibMode->blockSignals(true);
    m_cmbCalibMode->clear();
    m_cmbCalibMode->setEnabled(m_sensorType != "radar");

    if (m_sensorType == "camera") {
        m_cmbCalibMode->addItems(
                {"single_camera", "avm_camera", "svm_camera", "all_camera"});
        m_cmbCalibMode->setCurrentIndex(0);
        m_calibMode = MODE_SINGLE_CAMERA;
    } else if (m_sensorType == "lidar") {
        m_cmbCalibMode->addItems({"single_lidar", "all_lidar"});
        m_cmbCalibMode->setCurrentIndex(0);
        m_calibMode = MODE_SINGLE_LIDAR;
    }
    m_cmbCalibMode->blockSignals(false);
}

void ManualSensorCalibDlg::onSensorTypeChanged(int idx) {
    m_sensorType = (idx == 1) ? "lidar" : (idx == 2 ? "radar" : "camera");

    refreshCalibModeCombo();
    updateSensorList();
    syncCurrentSensorFromCombo();
    reloadCurrentBagFrame();
}

void ManualSensorCalibDlg::onCalibModeChanged(const QString& mode) {
    if (mode.isEmpty()) return;

    std::string m = mode.toStdString();
    if (m == "single_camera")
        m_calibMode = MODE_SINGLE_CAMERA;
    else if (m == "avm_camera")
        m_calibMode = MODE_AVM_CAMERA;
    else if (m == "svm_camera")
        m_calibMode = MODE_SVM_CAMERA;
    else if (m == "all_camera")
        m_calibMode = MODE_ALL_CAMERA;
    else if (m == "single_lidar")
        m_calibMode = MODE_SINGLE_LIDAR;
    else if (m == "all_lidar")
        m_calibMode = MODE_ALL_LIDAR;
    else
        return;

    // Invalidate any in-flight slider load: its captured topic set belongs to
    // the previous calib mode and won't match the new slot map. The pending
    // reloadCurrentBagFrame() below will re-capture topics for the new mode.
    ++m_modeGeneration;
    m_extrinsicDirty = true;
    updateSensorList();
    syncCurrentSensorFromCombo();
    if (m_viewMode == 0) {
        scheduleViewUpdate();
    }
    reloadCurrentBagFrame();
}

void ManualSensorCalibDlg::onBevRemapBackendChanged(int index) {
    if (!m_cmbBevRemapBackend || index < 0) return;

    const auto mode = static_cast<mcalib::BevRemapMode>(
            m_cmbBevRemapBackend->currentData().toInt());

    QSettings settings;
    settings.setValue("qManualCalib/bevRemapBackend", static_cast<int>(mode));

    if (!m_bevViewer) return;

    m_bevViewer->setRemapMode(mode);
    m_extrinsicDirty = true;

    const auto active = m_bevViewer->getActiveRemapMode();
    CVLog::Print("[SensorCalib] BEV remap backend: requested=%s active=%s",
                 mcalib::BevRemapper::modeName(mode),
                 mcalib::BevRemapper::modeName(active));

    if (m_viewMode == 0) {
        scheduleViewUpdate();
    } else if (m_viewMode == 1) {
        scheduleViewUpdate();
    }
}

mcalib::BevRemapMode ManualSensorCalibDlg::getGpuBackendMode() const {
    if (!m_cmbBevRemapBackend) return mcalib::BevRemapMode::Auto;
    return static_cast<mcalib::BevRemapMode>(
            m_cmbBevRemapBackend->currentData().toInt());
}

int ManualSensorCalibDlg::getLidarIndexByName(const std::string& name) const {
    auto pos = name.find_last_of('_');
    if (pos == std::string::npos) return -1;
    try {
        int idx = std::stoi(name.substr(pos + 1));
        for (size_t i = 0; i < m_calibConfig.lidars.size(); ++i) {
            if (m_calibConfig.lidars[i].lidar_idx == idx)
                return static_cast<int>(i);
        }
    } catch (...) {
    }
    return -1;
}

void ManualSensorCalibDlg::onSensorNameChanged(const QString& name) {
    m_currentSensor = name.toStdString();
    if (m_viewMode == 1) {
        m_pointCloudRaw.clear();
        m_pointCloud.clear();
        m_cloudStampUs = 0;
    }
    // Invalidate any in-flight slider load: the captured topic filter (e.g.
    // single-camera projection) depends on the previous sensor name.
    ++m_modeGeneration;
    m_extrinsicDirty = true;
    displayDeltaExtrinsic(m_currentSensor);
    if (m_viewMode == 0) {
        scheduleViewUpdate();
    }
    reloadCurrentBagFrame();
}

void ManualSensorCalibDlg::syncCurrentSensorFromCombo() {
    if (!m_cmbSensorName || m_cmbSensorName->count() == 0) {
        m_currentSensor.clear();
        m_deltaExtrinsic.setZero();
        displayDeltaExtrinsic(m_currentSensor);
        return;
    }
    if (m_cmbSensorName->currentText().isEmpty()) {
        m_cmbSensorName->setCurrentIndex(0);
    }
    m_currentSensor = m_cmbSensorName->currentText().toStdString();
    displayDeltaExtrinsic(m_currentSensor);
}

void ManualSensorCalibDlg::onViewModeChanged(int index) {
    m_viewMode = index;
    updateViewModeButtonStyle(index);

    bool showDist = (index == 1);
    bool showGround = (index == 2);
    bool showLidar = (index == 1 || index == 2);

    m_sliderDistFilter->setVisible(showDist);
    m_sliderDistFilterMin->setVisible(showDist);
    m_lblDistRange->setVisible(showDist);

    m_sliderGroundFilter->setVisible(showGround);
    m_sliderGroundFilterMax->setVisible(showGround);
    m_lblGroundRange->setVisible(showGround);

    m_lidarGroup->setVisible(showLidar && !m_calibConfig.lidars.empty());

    updatePointSizeSliderLabel();
    // Invalidate any in-flight slider load: its captured topic set / cloud
    // selection belongs to the previous view mode. The processSliderLoad()
    // below will re-capture for the new view mode.
    ++m_modeGeneration;
    updateFusionView();
    updateExportButtonStates();

    if (m_bagReader && m_bagReader->isOpen()) {
        m_appliedSliderValue = -1;
        processSliderLoad();
    }
}

void ManualSensorCalibDlg::updateExportButtonStates() {
    if (m_btnExportImage) {
        const bool canExportImage = (m_viewMode == 0 || m_viewMode == 1) &&
                                    m_configLoaded && m_bagReader &&
                                    m_bagReader->isOpen();
        m_btnExportImage->setEnabled(canExportImage);
    }
    if (m_btnBatchExportImages) {
        const bool export_running =
                m_batchExportWatcher && m_batchExportWatcher->isRunning();
        const bool canBatchExport =
                !export_running && (m_viewMode == 0 || m_viewMode == 1) &&
                m_configLoaded && m_bagReader && m_bagReader->isOpen();
        m_btnBatchExportImages->setEnabled(canBatchExport);
    }
    if (m_btnExportPCD) {
        const bool canExportPcd = m_viewMode != 0 && m_configLoaded &&
                                  m_bagReader && m_bagReader->isOpen();
        m_btnExportPCD->setEnabled(canExportPcd);
    }
    if (m_btnBatchExportPCD) {
        const bool export_running =
                m_batchExportWatcher && m_batchExportWatcher->isRunning();
        const bool canBatchExportPcd = !export_running && m_configLoaded &&
                                       m_bagReader && m_bagReader->isOpen();
        m_btnBatchExportPCD->setEnabled(canBatchExportPcd);
    }
}

void ManualSensorCalibDlg::updateViewModeButtonStyle(int activeIndex) {
    static const char* kActive =
            "QPushButton { background-color: #3d7ec9; color: white; "
            "font-weight: bold; }";
    static const char* kNormal = "";

    if (m_btnBevView) {
        m_btnBevView->setStyleSheet(activeIndex == 0 ? kActive : kNormal);
    }
    if (m_btnLidarProjView) {
        m_btnLidarProjView->setStyleSheet(activeIndex == 1 ? kActive : kNormal);
    }
    if (m_btnSingleFrameView) {
        m_btnSingleFrameView->setStyleSheet(activeIndex == 2 ? kActive
                                                             : kNormal);
    }
    if (m_btnMultiFrameView) {
        m_btnMultiFrameView->setStyleSheet(activeIndex == 3 ? kActive
                                                            : kNormal);
    }
}

void ManualSensorCalibDlg::enter2DImageView() {
    if (!m_app) return;
    m_app->toggle3DView(false);
    if (m_vtkCloud) {
        m_vtkCloud->setVisible(false);
        m_vtkCloud->setRedrawFlagRecursive(true);
    }
}

void ManualSensorCalibDlg::enter3DPointCloudView() {
    if (!m_app) return;
    m_app->toggle3DView(true);
    if (m_vtkImage) {
        m_vtkImage->setVisible(false);
        m_vtkImage->setRedrawFlagRecursive(true);
    }
}

void ManualSensorCalibDlg::onSpeedChanged(int index) {
    static const double speedFactors[] = {1.0, 5.0, 10.0, 50.0, 100.0, 1000.0};
    constexpr double kBaseRotRes = 0.01;
    constexpr double kBasePosRes = 0.001;
    if (index >= 0 && index < 6) {
        m_rotResolution = kBaseRotRes * speedFactors[index];
        m_posResolution = kBasePosRes * speedFactors[index];
    }
}

void ManualSensorCalibDlg::onTimeSliderChanged(int value) {
    updateTimeSliderLabel(value);
    m_pendingSliderValue = value;

    if (!m_sliderDebounce) {
        m_sliderDebounce = new QTimer(this);
        m_sliderDebounce->setSingleShot(true);
        connect(m_sliderDebounce, &QTimer::timeout, this,
                &ManualSensorCalibDlg::processSliderLoad);
    }

    const int debounceMs = [&]() {
        if (mcalib::bagUsesVideoCodec(m_bagImageEncoding)) {
            return (m_bagReader && m_bagReader->hasTopicTimeIndex()) ? 40 : 80;
        }
        return (m_bagReader && m_bagReader->hasTopicTimeIndex()) ? 12 : 24;
    }();
    m_sliderDebounce->start(debounceMs);
}

int ManualSensorCalibDlg::bagTimeStepDelta() const {
    if (m_bagDurationSec <= 0.01) return 1;
    constexpr double kCameraFrameIntervalSec = 0.1;  // ~10Hz
    const double percent = kCameraFrameIntervalSec / m_bagDurationSec;
    return std::max(1, static_cast<int>(percent * 1000.0 + 0.5));
}

void ManualSensorCalibDlg::reloadCurrentBagFrame() {
    if (!m_bagReader || !m_bagReader->isOpen()) {
        updateFusionView();
        return;
    }
    m_appliedSliderValue = -1;
    m_pendingSliderValue = m_sliderTimePos ? m_sliderTimePos->value() : 0;
    processSliderLoad();
}

void ManualSensorCalibDlg::onTimeStepBack() {
    if (!m_sliderTimePos) return;
    const int step = bagTimeStepDelta();
    const int next = std::max(0, m_sliderTimePos->value() - step);
    m_sliderTimePos->setValue(next);
}

void ManualSensorCalibDlg::onTimeStepForward() {
    if (!m_sliderTimePos) return;
    const int step = bagTimeStepDelta();
    const int next = std::min(1000, m_sliderTimePos->value() + step);
    m_sliderTimePos->setValue(next);
}

void ManualSensorCalibDlg::scheduleViewUpdate() {
    if (!m_adjustDebounce) {
        m_adjustDebounce = new QTimer(this);
        m_adjustDebounce->setSingleShot(true);
        connect(m_adjustDebounce, &QTimer::timeout, this,
                &ManualSensorCalibDlg::updateFusionView);
    }
    m_adjustDebounce->start(80);
}

#define IMPL_ADJUST(idx, sign)                                    \
    if (m_currentSensor.empty()) return;                          \
    m_deltaExtrinsic(idx) +=                                      \
            sign * (idx < 3 ? m_rotResolution : m_posResolution); \
    updateDeltaExtrinsic(m_deltaExtrinsic, m_currentSensor);      \
    displayDeltaExtrinsic(m_currentSensor);                       \
    scheduleViewUpdate();

void ManualSensorCalibDlg::updateDeltaExtrinsic(
        const mcalib::Vector6d& delta, const std::string& sensor_name) {
    if (sensor_name.empty()) return;

    m_extrinsicDirty = true;
    if (m_sensorType == "camera") {
        if (m_calibMode == MODE_SINGLE_CAMERA) {
            auto it = m_deltaExtrinsics.find(sensor_name);
            if (it != m_deltaExtrinsics.end()) {
                it->second = delta;
            }
        } else if (m_calibMode == MODE_ALL_CAMERA) {
            for (auto& [k, v] : m_deltaExtrinsics) {
                (void)k;
                v = delta;
            }
        } else if (m_calibMode == MODE_AVM_CAMERA) {
            for (auto& [k, v] : m_deltaExtrinsics) {
                if (k.compare(0, 3, "pan") == 0) v = delta;
            }
        } else if (m_calibMode == MODE_SVM_CAMERA) {
            for (auto& [k, v] : m_deltaExtrinsics) {
                if (k.compare(0, 3, "pan") != 0) v = delta;
            }
        }
    } else if (m_sensorType == "lidar") {
        if (m_calibMode == MODE_SINGLE_LIDAR) {
            const int li = getLidarIndexByName(sensor_name);
            if (li >= 0 &&
                li < static_cast<int>(m_deltaLidarExtrinsics.size())) {
                m_deltaLidarExtrinsics[li] = delta;
            }
        } else if (m_calibMode == MODE_ALL_LIDAR) {
            for (auto& v : m_deltaLidarExtrinsics) {
                v = delta;
            }
        }
    }
}

void ManualSensorCalibDlg::onExtrinsicEditCommitted() {
    if (m_currentSensor.empty()) return;

    QLineEdit* edits[] = {m_editRoll, m_editPitch, m_editYaw,
                          m_editX,    m_editY,     m_editZ};
    for (int i = 0; i < 6; ++i) {
        if (!edits[i]) return;
        bool ok = false;
        const double value = edits[i]->text().toDouble(&ok);
        if (!ok) {
            displayDeltaExtrinsic(m_currentSensor);
            return;
        }
        m_deltaExtrinsic(i) = value;
    }

    updateDeltaExtrinsic(m_deltaExtrinsic, m_currentSensor);
    displayDeltaExtrinsic(m_currentSensor);
    scheduleViewUpdate();
}

void ManualSensorCalibDlg::onRollAdd() { IMPL_ADJUST(0, 1) }
void ManualSensorCalibDlg::onRollSub() { IMPL_ADJUST(0, -1) }
void ManualSensorCalibDlg::onPitchAdd() { IMPL_ADJUST(1, 1) }
void ManualSensorCalibDlg::onPitchSub() { IMPL_ADJUST(1, -1) }
void ManualSensorCalibDlg::onYawAdd() { IMPL_ADJUST(2, 1) }
void ManualSensorCalibDlg::onYawSub() { IMPL_ADJUST(2, -1) }
void ManualSensorCalibDlg::onXAdd() { IMPL_ADJUST(3, 1) }
void ManualSensorCalibDlg::onXSub() { IMPL_ADJUST(3, -1) }
void ManualSensorCalibDlg::onYAdd() { IMPL_ADJUST(4, 1) }
void ManualSensorCalibDlg::onYSub() { IMPL_ADJUST(4, -1) }
void ManualSensorCalibDlg::onZAdd() { IMPL_ADJUST(5, 1) }
void ManualSensorCalibDlg::onZSub() { IMPL_ADJUST(5, -1) }
#undef IMPL_ADJUST

void ManualSensorCalibDlg::updatePointSizeSliderLabel() {
    if (!m_lblPointSizeTitle || !m_lblPointSize || !m_sliderPointSize) return;

    const int value = m_sliderPointSize->value();
    if (m_viewMode == 0) {
        m_lblPointSizeTitle->setText(tr("Focal scale"));
        const double focal = 0.3 * (value + 1);
        m_lblPointSize->setText(QString::number(focal, 'f', 1));
        m_sliderPointSize->setToolTip(
                tr("BEV virtual camera focal scale (0.3 × (value + 1))"));
    } else {
        m_lblPointSizeTitle->setText(tr("Point size"));
        m_lblPointSize->setText(QString::number(value));
        m_sliderPointSize->setToolTip(
                tr("Point cloud / projection dot radius in pixels"));
    }
}

void ManualSensorCalibDlg::onPointSizeChanged(int value) {
    m_pointSize = value;
    if (m_viewMode == 0 && m_bevViewer) {
        m_bevViewer->setFocalScale(0.3 * (value + 1));
    }
    updateFusionView();
}

void ManualSensorCalibDlg::onDistFilterChanged(int value) {
    m_distFilter = value;
    m_lblDistRange->setText(QString("%1~%2m")
                                    .arg(m_distFilterMin, 0, 'f', 1)
                                    .arg(m_distFilter, 0, 'f', 1));
    updateFusionView();
}

void ManualSensorCalibDlg::onDistFilterMinChanged(int value) {
    m_distFilterMin = value / 10.0;
    m_lblDistRange->setText(QString("%1~%2m")
                                    .arg(m_distFilterMin, 0, 'f', 1)
                                    .arg(m_distFilter, 0, 'f', 1));
    updateFusionView();
}

void ManualSensorCalibDlg::onGroundFilterChanged(int value) {
    m_groundFilterMin = value / 10.0;
    m_lblGroundRange->setText(QString("%1~%2m")
                                      .arg(m_groundFilterMin, 0, 'f', 1)
                                      .arg(m_groundFilterMax, 0, 'f', 1));
    updateFusionView();
}

void ManualSensorCalibDlg::onGroundFilterMaxChanged(int value) {
    m_groundFilterMax = value / 10.0;
    m_lblGroundRange->setText(QString("%1~%2m")
                                      .arg(m_groundFilterMin, 0, 'f', 1)
                                      .arg(m_groundFilterMax, 0, 'f', 1));
    updateFusionView();
}

void ManualSensorCalibDlg::onLidarCheckboxToggled() {
    for (size_t i = 0; i < m_lidarCheckboxes.size(); i++) {
        m_selectedLidars[i] = m_lidarCheckboxes[i]->isChecked();
    }
    updateFusionView();
}

std::vector<Eigen::Vector3f> ManualSensorCalibDlg::getFilteredPointCloud()
        const {
    std::vector<Eigen::Vector3f> filtered;
    if (!m_pointCloud.empty()) {
        filtered.reserve(m_pointCloud.size());
        for (const auto& pt : m_pointCloud) {
            float dist = pt.norm();
            if (dist < m_distFilterMin || dist > m_distFilter) continue;
            if (pt.z() < m_groundFilterMin || pt.z() > m_groundFilterMax)
                continue;
            filtered.push_back(pt);
        }
        return filtered;
    }

    filtered.reserve(m_pointCloudRaw.size());
    for (const auto& pt : m_pointCloudRaw) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        float dist = p.norm();
        if (dist < m_distFilterMin || dist > m_distFilter) continue;
        if (p.z() < m_groundFilterMin || p.z() > m_groundFilterMax) continue;
        filtered.push_back(p);
    }
    return filtered;
}

std::map<std::string, Eigen::Isometry3d>
ManualSensorCalibDlg::getFinalExtrinsics() const {
    std::map<std::string, Eigen::Isometry3d> result;
    for (const auto& [name, cam] : m_calibConfig.cameras) {
        const Eigen::Isometry3d& iso_sensing_cam = cam.extrinsic;
        mcalib::Vector6d tune = mcalib::Vector6d::Zero();
        auto it = m_deltaExtrinsics.find(name);
        if (it != m_deltaExtrinsics.end()) {
            tune = it->second;
        }

        mcalib::Vector6d tune_rad = tune;
        tune_rad.segment(0, 3) *= (M_PI / 180.0);

        Eigen::Isometry3d iso_tune;
        mcalib::Vec2Isometry(tune_rad, iso_tune);
        Eigen::Isometry3d iso_result;
        iso_result.linear() = iso_tune.linear() * iso_sensing_cam.linear();
        iso_result.translation() =
                iso_sensing_cam.translation() + iso_tune.translation();
        result[name] = iso_result;
    }
    return result;
}

void ManualSensorCalibDlg::displayDeltaExtrinsic(
        const std::string& sensor_name) {
    m_deltaExtrinsic = mcalib::Vector6d::Zero();
    if (sensor_name.empty()) {
        const QSignalBlocker blockRoll(m_editRoll);
        const QSignalBlocker blockPitch(m_editPitch);
        const QSignalBlocker blockYaw(m_editYaw);
        const QSignalBlocker blockX(m_editX);
        const QSignalBlocker blockY(m_editY);
        const QSignalBlocker blockZ(m_editZ);
        if (m_editRoll) m_editRoll->setText(QStringLiteral("0.0000"));
        if (m_editPitch) m_editPitch->setText(QStringLiteral("0.0000"));
        if (m_editYaw) m_editYaw->setText(QStringLiteral("0.0000"));
        if (m_editX) m_editX->setText(QStringLiteral("0.0000"));
        if (m_editY) m_editY->setText(QStringLiteral("0.0000"));
        if (m_editZ) m_editZ->setText(QStringLiteral("0.0000"));
        return;
    }

    if (m_sensorType == "camera") {
        auto it = m_deltaExtrinsics.find(sensor_name);
        if (it != m_deltaExtrinsics.end()) {
            m_deltaExtrinsic = it->second;
        }
    } else if (m_sensorType == "lidar") {
        const int li = getLidarIndexByName(sensor_name);
        if (li >= 0 && li < static_cast<int>(m_deltaLidarExtrinsics.size())) {
            m_deltaExtrinsic = m_deltaLidarExtrinsics[li];
        }
    }

    const QSignalBlocker blockRoll(m_editRoll);
    const QSignalBlocker blockPitch(m_editPitch);
    const QSignalBlocker blockYaw(m_editYaw);
    const QSignalBlocker blockX(m_editX);
    const QSignalBlocker blockY(m_editY);
    const QSignalBlocker blockZ(m_editZ);
    if (!m_editRoll || !m_editPitch || !m_editYaw || !m_editX || !m_editY ||
        !m_editZ) {
        return;
    }
    m_editRoll->setText(QString::number(m_deltaExtrinsic(0), 'f', 4));
    m_editPitch->setText(QString::number(m_deltaExtrinsic(1), 'f', 4));
    m_editYaw->setText(QString::number(m_deltaExtrinsic(2), 'f', 4));
    m_editX->setText(QString::number(m_deltaExtrinsic(3), 'f', 4));
    m_editY->setText(QString::number(m_deltaExtrinsic(4), 'f', 4));
    m_editZ->setText(QString::number(m_deltaExtrinsic(5), 'f', 4));
}

void ManualSensorCalibDlg::updateFusionView() {
    if (!m_configLoaded) {
        if (m_viewMode == 0) {
            showIdleBevCanvas();
        }
        return;
    }
    if (m_sensorType == "radar") {
        m_lastExportImage.release();
        m_lblStatus->setText(
                tr("Radar calibration view is not available in this build"));
        return;
    }
    if (m_viewMode != 0 && m_viewMode != 1) {
        m_lastExportImage.release();
    }
    switch (m_viewMode) {
        case 0:
            updateBevView();
            break;
        case 1:
            updateLidarProjView();
            break;
        case 2:
            updateLidarSingleFrameView();
            break;
        case 3:
            updateLidarMultiFrameView();
            break;
        default:
            updateLidarProjView();
            break;
    }
}

void ManualSensorCalibDlg::updateBevView() {
    if (!m_bevViewer || m_images.empty()) {
        const bool bagOpen = m_bagReader && m_bagReader->isOpen();
        if (!m_bevViewer) {
            if (m_viewMode == 0) {
                showIdleBevCanvas();
            }
            return;
        }
        if (bagOpen) {
            CVLog::Print("[SensorCalib] updateBevView: no frame images (%zu)",
                         m_images.size());
        }
        m_lastExportImage.release();
        const QString subtitle =
                bagOpen ? tr("Waiting for synced camera frame...")
                        : tr("Step 2: Load bag");
        displayImageInViewer(makeBevPlaceholderCanvas(
                tr("Manual Sensor Calibration - BEV"), subtitle));
        return;
    }

    cv::Mat bev = renderBevImage();
    if (bev.empty()) {
        // renderBevImage returns empty when the slot map and m_images are out
        // of sync (e.g. right after a mode switch, before the async bag reload
        // has finished). Show a "loading" placeholder instead of a misleading
        // "generation failed" error. The pending processSliderLoad() will
        // trigger updateFusionView() once the new frame is ready.
        CVLog::Print(
                "[SensorCalib] updateBevView: BEV not ready (stale images "
                "or empty render result)");
        m_lastExportImage.release();
        displayImageInViewer(
                makeBevPlaceholderCanvas(tr("Manual Sensor Calibration - BEV"),
                                         tr("Loading BEV frame...")));
        return;
    }
    CVLog::Print("[SensorCalib] updateBevView: BEV generated %dx%d ch=%d",
                 bev.cols, bev.rows, bev.channels());

    m_lastExportImage = bev.clone();
    if (m_bevViewer) {
        m_lastBevPreRotateSize = bev.size();
        m_lastBevRotatedCw = true;
        cv::rotate(bev, bev, cv::ROTATE_90_CLOCKWISE);
    }
    drawBevMeasureOverlay(bev);
    mcalib::drawVehicleStateOverlay(bev, m_vehicleState, cv::Point(20, 20), 0.5,
                                    cv::Scalar(0, 0, 255), 1);
    const int hint_y = std::max(20, bev.rows - 12);
    cv::putText(bev, "BEV Mode (L:measure R:clear)", cv::Point(10, hint_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    displayImageInViewer(bev);
}

void ManualSensorCalibDlg::updateLidarProjView() {
    if (m_pointCloudRaw.empty()) {
        loadCloudForCurrentTimestamp();
    }

    cv::Mat display = renderLidarProjImage();
    if (display.empty()) {
        m_lastExportImage.release();
        if (m_bagReader && m_bagReader->isOpen()) {
            CVLog::Print(
                    "[SensorCalib] updateLidarProjView: no projection image");
        }
        cv::Mat placeholder(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        const char* hint = (m_bagReader && m_bagReader->isOpen())
                                   ? "No projection for this frame"
                                   : "Load bag for LiDAR projection";
        cv::putText(placeholder, hint, cv::Point(100, 240),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255),
                    2);
        displayImageInViewer(placeholder);
        return;
    }

    m_lastExportImage = display.clone();
    std::string cam_name = m_currentSensor;
    if (m_sensorType == "lidar" && !m_calibConfig.cameras.empty()) {
        cam_name = m_calibConfig.cameras.begin()->first;
    }
    std::string info = "Sensor: " + cam_name;
    cv::putText(display, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    std::string stamp_text = "cloud_stamp " + std::to_string(m_cloudStampUs);
    cv::putText(display, stamp_text, cv::Point(60, 60),
                cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 255), 1,
                cv::LINE_AA);
    if (display.cols > 3600) {
        cv::resize(display, display, cv::Size(), 0.4, 0.4);
    } else if (display.cols > 1800) {
        cv::resize(display, display, cv::Size(), 0.8, 0.8);
    }
    displayImageInViewer(display);
}

void ManualSensorCalibDlg::updateLidarSingleFrameView() {
    if (m_pointCloudRaw.empty()) {
        loadCloudForCurrentTimestamp();
    }

    enter3DPointCloudView();

    auto filtered = getFilteredPointCloud();
    if (filtered.empty() && m_pointCloudRaw.empty()) {
        CVLog::Warning(
                "[SensorCalib] updateSingleFrameView: no point cloud data");
        m_lblStatus->setText(tr("No point cloud data"));
        return;
    }
    CVLog::Print("[SensorCalib] updateSingleFrameView: filtered=%zu, raw=%zu",
                 filtered.size(), m_pointCloudRaw.size());

    std::vector<Eigen::Vector3f> merged;
    std::vector<ecvColor::Rgb> mergedColors;

    if (!m_pointCloudRaw.empty() && !m_calibConfig.lidars.empty()) {
        const auto extrinsic_gnss_lidar = getLidarFinalExtrinsic();
        for (size_t j = 0; j < m_calibConfig.lidars.size(); ++j) {
            if (j >= m_selectedLidars.size() || !m_selectedLidars[j]) continue;
            const auto& lc = m_calibConfig.lidars[j];
            if (j >= extrinsic_gnss_lidar.size()) continue;

            const Eigen::Matrix4f T =
                    extrinsic_gnss_lidar[j].matrix().cast<float>();
            std::vector<mcalib::PointXYZIRT> single_raw;
            for (const auto& pt : m_pointCloudRaw) {
                if (pt.ring < lc.ring_start || pt.ring > lc.ring_end) continue;
                mcalib::PointXYZIRT tpt = pt;
                const Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
                const Eigen::Vector4f tp = T * p;
                tpt.x = tp.x();
                tpt.y = tp.y();
                tpt.z = tp.z();
                single_raw.push_back(tpt);
            }

            std::vector<Eigen::Vector3f> undistorted;
            bool did_undistort = false;
            if (!m_vehiclePoses.empty() && !single_raw.empty()) {
                did_undistort = mcalib::undistortPointCloud(
                        single_raw, m_cloudStampUs, m_cloudStampUs,
                        m_vehiclePoses, Eigen::Isometry3d::Identity(),
                        undistorted);
            }

            uint8_t cr, cg, cb;
            mcalib::highContrastLidarColor(static_cast<int>(j), cr, cg, cb);
            ecvColor::Rgb color(cr, cg, cb);

            if (did_undistort) {
                std::vector<Eigen::Vector3f> filtered_ring;
                mcalib::filterGroundReflectionPoints(undistorted, filtered_ring,
                                                     m_groundFilterMin,
                                                     m_groundFilterMax);
                for (const auto& pt : filtered_ring) {
                    merged.push_back(pt);
                    mergedColors.push_back(color);
                }
            } else {
                std::vector<Eigen::Vector3f> raw_xyz;
                raw_xyz.reserve(single_raw.size());
                for (const auto& pt : single_raw) {
                    raw_xyz.emplace_back(pt.x, pt.y, pt.z);
                }
                std::vector<Eigen::Vector3f> filtered_ring;
                mcalib::filterGroundReflectionPoints(raw_xyz, filtered_ring,
                                                     m_groundFilterMin,
                                                     m_groundFilterMax);
                for (const auto& pt : filtered_ring) {
                    merged.push_back(pt);
                    mergedColors.push_back(color);
                }
            }
        }
    } else {
        merged = filtered;
    }

    displayPointCloudIn3D(merged, mergedColors);
    m_lblStatus->setText(tr("3D View: %1 points").arg(merged.size()));
}

void ManualSensorCalibDlg::updateLidarMultiFrameView() {
    if (!m_bagReader || !m_bagReader->isOpen()) {
        m_lblStatus->setText(tr("Multi-frame: load bag first"));
        return;
    }

    std::vector<Eigen::Vector3f> merged;
    std::vector<ecvColor::Rgb> mergedColors;
    const int center = m_sliderTimePos ? m_sliderTimePos->value() : 0;
    const int offsets[] = {-20, -10, 0, 10, 20};
    int loaded_frames = 0;

    for (size_t fi = 0; fi < sizeof(offsets) / sizeof(offsets[0]); ++fi) {
        const int slider_val = std::clamp(center + offsets[fi], 0, 1000);
        const double percent = slider_val / 1000.0;
        if (!loadCloudAtBagPercent(percent)) continue;

        auto pts = getFilteredPointCloud();
        if (pts.empty()) continue;
        ++loaded_frames;

        const ecvColor::Rgb color(static_cast<ColorCompType>((fi * 60) % 255),
                                  static_cast<ColorCompType>((fi * 110) % 255),
                                  static_cast<ColorCompType>((fi * 170) % 255));
        for (const auto& pt : pts) {
            merged.push_back(pt);
            mergedColors.push_back(color);
        }
    }

    if (merged.empty()) {
        m_lblStatus->setText(tr("Multi-frame: no point cloud data"));
        return;
    }

    displayPointCloudIn3D(merged, mergedColors);
    m_lblStatus->setText(tr("Multi-frame: %1 points from %2 samples")
                                 .arg(merged.size())
                                 .arg(loaded_frames));
}

void ManualSensorCalibDlg::lidarCamFusion(
        const std::vector<Eigen::Vector3f>& cloud,
        cv::Mat& img,
        const std::string& camera_name,
        const Eigen::Isometry3d& T_cam_sensing,
        const cv::Mat& K,
        const cv::Mat& D,
        int radius,
        mcalib::BevRemapMode backend_mode) {
    if (cloud.empty() || img.empty()) return;

    const Eigen::Matrix3d rot = T_cam_sensing.linear();
    const Eigen::Vector3d trans = T_cam_sensing.translation();

    auto cam_model =
            m_bevViewer ? m_bevViewer->getCameraSystem().getCamera(camera_name)
                        : nullptr;

    auto it_cam = m_calibConfig.cameras.find(camera_name);
    const bool use_gpu_proj = it_cam != m_calibConfig.cameras.end() &&
                              (it_cam->second.intrinsic.model_type ==
                                       mcalib::CameraIntrinsic::PINHOLE ||
                               it_cam->second.intrinsic.model_type ==
                                       mcalib::CameraIntrinsic::KANNALA_BRANDT);

    if (use_gpu_proj) {
        const auto& intr = it_cam->second.intrinsic;
        mcalib::LidarProjResult proj;
        bool projected = false;
        if (intr.model_type == mcalib::CameraIntrinsic::PINHOLE) {
            projected = mcalib::LidarProjBackend::projectPoints(
                    backend_mode, cloud, rot, trans, intr.fx, intr.fy, intr.cx,
                    intr.cy, proj);
        } else {
            const mcalib::KannalaBrandtCoeffs kb{intr.k1, intr.k2, intr.k3,
                                                 intr.k4};
            projected = mcalib::LidarProjBackend::projectPointsKb(
                    backend_mode, cloud, rot, trans, intr.fx, intr.fy, intr.cx,
                    intr.cy, kb, proj);
        }
        if (projected) {
            for (size_t i = 0; i < proj.image_points.size(); ++i) {
                const cv::Scalar color =
                        mcalib::colorFromDepth(proj.depths[i], 2.0f);
                if (radius > 0) {
                    cv::circle(img, proj.image_points[i], radius, color, -1);
                } else {
                    const int row =
                            static_cast<int>(proj.image_points[i].y + 0.5f);
                    const int col =
                            static_cast<int>(proj.image_points[i].x + 0.5f);
                    if (row >= 0 && row < img.rows && col >= 0 &&
                        col < img.cols) {
                        img.at<cv::Vec3b>(row, col) =
                                cv::Vec3b(static_cast<uchar>(color[0]),
                                          static_cast<uchar>(color[1]),
                                          static_cast<uchar>(color[2]));
                    }
                }
            }
            return;
        }
    }

    std::vector<cv::Point3f> cloud_cam;
    cloud_cam.reserve(cloud.size());
    for (const auto& pt : cloud) {
        const Eigen::Vector3d p(pt.x(), pt.y(), pt.z());
        const Eigen::Vector3d pc = rot * p + trans;
        if (pc.z() > 0) {
            cloud_cam.emplace_back(static_cast<float>(pc.x()),
                                   static_cast<float>(pc.y()),
                                   static_cast<float>(pc.z()));
        }
    }
    if (cloud_cam.empty()) return;

    std::vector<cv::Point2f> image_points;
    image_points.resize(cloud_cam.size());
    if (cam_model) {
        const size_t n = cloud_cam.size();
        const size_t chunk = 8192;
        if (n > chunk * 2) {
            std::vector<std::future<void>> futures;
            for (size_t start = 0; start < n; start += chunk) {
                const size_t end = std::min(start + chunk, n);
                futures.push_back(
                        std::async(std::launch::async, [&, start, end]() {
                            for (size_t i = start; i < end; ++i) {
                                const auto& p3 = cloud_cam[i];
                                Eigen::Vector2d px;
                                cam_model->spaceToPlane(
                                        Eigen::Vector3d(p3.x, p3.y, p3.z), px);
                                image_points[i] =
                                        cv::Point2f(static_cast<float>(px.x()),
                                                    static_cast<float>(px.y()));
                            }
                        }));
            }
            for (auto& fut : futures) fut.get();
        } else {
            for (size_t i = 0; i < n; ++i) {
                const auto& p3 = cloud_cam[i];
                Eigen::Vector2d px;
                cam_model->spaceToPlane(Eigen::Vector3d(p3.x, p3.y, p3.z), px);
                image_points[i] = cv::Point2f(static_cast<float>(px.x()),
                                              static_cast<float>(px.y()));
            }
        }
    } else {
        const double fx = K.at<double>(0, 0);
        const double fy = K.at<double>(1, 1);
        const double cx = K.at<double>(0, 2);
        const double cy = K.at<double>(1, 2);
        for (const auto& p3 : cloud_cam) {
            image_points.emplace_back(
                    static_cast<float>(fx * p3.x / p3.z + cx),
                    static_cast<float>(fy * p3.y / p3.z + cy));
        }
    }

    for (size_t i = 0; i < image_points.size(); ++i) {
        const float depth = cloud_cam[i].z;
        const cv::Scalar color = mcalib::colorFromDepth(depth, 2.0f);
        const cv::Point2f& uv = image_points[i];
        if (radius > 0) {
            cv::circle(img, uv, radius, color, -1);
        } else {
            const int row = static_cast<int>(uv.y + 0.5f);
            const int col = static_cast<int>(uv.x + 0.5f);
            if (row >= 0 && row < img.rows && col >= 0 && col < img.cols) {
                img.at<cv::Vec3b>(row, col) =
                        cv::Vec3b(static_cast<uchar>(color[0]),
                                  static_cast<uchar>(color[1]),
                                  static_cast<uchar>(color[2]));
            }
        }
    }
}

void ManualSensorCalibDlg::createLidarCheckboxes() {
    for (auto* cb : m_lidarCheckboxes) {
        m_lidarGroupLayout->removeWidget(cb);
        delete cb;
    }
    m_lidarCheckboxes.clear();

    if (m_calibConfig.lidars.empty()) {
        m_lidarGroup->setVisible(false);
        return;
    }

    for (size_t i = 0; i < m_calibConfig.lidars.size(); i++) {
        auto* cb =
                new QCheckBox(tr("LiDAR %1 (rings %2-%3)")
                                      .arg(m_calibConfig.lidars[i].lidar_idx)
                                      .arg(m_calibConfig.lidars[i].ring_start)
                                      .arg(m_calibConfig.lidars[i].ring_end));
        cb->setChecked(true);
        connect(cb, &QCheckBox::toggled, this,
                &ManualSensorCalibDlg::onLidarCheckboxToggled);
        m_lidarGroupLayout->addWidget(cb);
        m_lidarCheckboxes.push_back(cb);
    }
    m_lidarGroup->setVisible(true);
}

void ManualSensorCalibDlg::updateCameraConfigForBev() {
    // Original update_camera_config writes tuned extrinsics into
    // cur_camera_config. Here the same result is applied through
    // getFinalExtrinsics() + updateExtrinsics().
}

cv::Point ManualSensorCalibDlg::dispToBevUnrotate(
        const cv::Point& disp_pt) const {
    if (m_lastBevRotatedCw && m_lastBevPreRotateSize.height > 0) {
        const int bev_x = disp_pt.y;
        const int bev_y = m_lastBevPreRotateSize.height - 1 - disp_pt.x;
        return cv::Point(bev_x, bev_y);
    }
    return disp_pt;
}

cv::Point ManualSensorCalibDlg::bevToDispRotate(const cv::Point& bev_pt) const {
    if (m_lastBevRotatedCw && m_lastBevPreRotateSize.height > 0) {
        const int disp_x = m_lastBevPreRotateSize.height - 1 - bev_pt.y;
        const int disp_y = bev_pt.x;
        return cv::Point(disp_x, disp_y);
    }
    return bev_pt;
}

cv::Point ManualSensorCalibDlg::mapMouseToBevDisp(const QPoint& qp) const {
    QWidget* win = getAssociatedWindow();
    if (!win || m_lastBevDisplaySize.width <= 0 ||
        m_lastBevDisplaySize.height <= 0) {
        return cv::Point(-1, -1);
    }
    const double sx = static_cast<double>(m_lastBevDisplaySize.width) /
                      std::max(1, win->width());
    const double sy = static_cast<double>(m_lastBevDisplaySize.height) /
                      std::max(1, win->height());
    return cv::Point(static_cast<int>(qp.x() * sx),
                     static_cast<int>(qp.y() * sy));
}

void ManualSensorCalibDlg::drawBevMeasureOverlay(cv::Mat& bev) {
    if (!m_bevViewer) return;

    m_bevMeasureDispPts.clear();
    for (const auto& gpt : m_bevMeasureGroundPts) {
        cv::Point bev_px;
        Eigen::Vector3d ground(gpt.x, gpt.y, gpt.z);
        if (!m_bevViewer->projectGroundToBevPixel(ground, bev_px)) continue;
        cv::Point disp = bevToDispRotate(bev_px);
        m_bevMeasureDispPts.push_back(disp);
        cv::circle(bev, disp, 3, cv::Scalar(0, 255, 0), -1);
    }

    if (m_bevMeasureDispPts.size() >= 2) {
        const cv::Point& p0 = m_bevMeasureDispPts[0];
        const cv::Point& p1 = m_bevMeasureDispPts[1];
        cv::line(bev, p0, p1, cv::Scalar(0, 255, 255), 1);

        if (m_bevMeasureGroundPts.size() >= 2) {
            const auto& g0 = m_bevMeasureGroundPts[0];
            const auto& g1 = m_bevMeasureGroundPts[1];
            const cv::Point2f q0 = m_bevViewer->groundToBevPixel(
                    Eigen::Vector3d(g0.x, g0.y, g0.z));
            const cv::Point2f q1 = m_bevViewer->groundToBevPixel(
                    Eigen::Vector3d(g1.x, g1.y, g1.z));
            m_bevMeasureBevPxDist =
                    std::hypot(static_cast<double>(q0.x - q1.x),
                               static_cast<double>(q0.y - q1.y));
        } else {
            const double dx = static_cast<double>(p0.x - p1.x);
            const double dy = static_cast<double>(p0.y - p1.y);
            m_bevMeasureBevPxDist = std::sqrt(dx * dx + dy * dy);
        }

        char text[256];
        std::snprintf(text, sizeof(text), "BEV(abs): %.1f px",
                      m_bevMeasureBevPxDist);
        if (m_bevMeasureGroundPts.size() >= 2) {
            const auto& g0 = m_bevMeasureGroundPts[0];
            const auto& g1 = m_bevMeasureGroundPts[1];
            const double dx = g1.x - g0.x;
            const double dy = g1.y - g0.y;
            const double dist_m = std::sqrt(dx * dx + dy * dy);
            char extra[64];
            std::snprintf(extra, sizeof(extra), "  %.2fm", dist_m);
            std::strncat(text, extra, sizeof(text) - std::strlen(text) - 1);
        }

        constexpr double font_scale = 0.6;
        constexpr int thickness = 2;
        int baseline = 0;
        const cv::Size text_size =
                cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                                thickness, &baseline);
        const cv::Point org(bev.cols - 80 - text_size.width,
                            20 + text_size.height);
        cv::putText(bev, text, org, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(0, 0, 255), thickness);
    }
}

void ManualSensorCalibDlg::startBatchExportJob(
        const QString& title,
        QPushButton* trigger_button,
        const QString& output_dir,
        std::function<mcalib::BatchExportResult(mcalib::BatchExportProgress&)>
                task,
        const QString& item_label) {
    if (m_batchExportWatcher && m_batchExportWatcher->isRunning()) {
        QMessageBox::information(asWidget(m_app->getMainWindow()), title,
                                 tr("A batch export task is already running."));
        return;
    }

    struct SharedExportState {
        std::atomic<bool> cancelled{false};
        std::atomic<int> completed{0};
        std::atomic<int> total{1};
        QString label;
        QMutex label_mutex;
    };
    auto state = std::make_shared<SharedExportState>();

    auto* progress =
            new QProgressDialog(tr("Preparing export..."), tr("Cancel"), 0, 100,
                                asWidget(m_app->getMainWindow()));
    progress->setWindowTitle(title);
    progress->setWindowModality(Qt::WindowModal);
    progress->setMinimumDuration(0);
    progress->setAutoClose(false);
    progress->setAutoReset(false);
    progress->setMinimumWidth(480);
    QPointer<QProgressDialog> progress_guard(progress);

    connect(progress, &QProgressDialog::canceled, this,
            [state]() { state->cancelled.store(true); });

    if (trigger_button) {
        trigger_button->setEnabled(false);
    }

    if (!m_batchExportPollTimer) {
        m_batchExportPollTimer = new QTimer(this);
        m_batchExportPollTimer->setInterval(50);
    }
    m_batchExportPollTimer->disconnect();
    connect(m_batchExportPollTimer, &QTimer::timeout, this,
            [progress_guard, state, output_dir]() {
                if (!progress_guard) return;
                const int total = std::max(1, state->total.load());
                const int completed = std::min(state->completed.load(), total);
                progress_guard->setMaximum(total);
                progress_guard->setValue(completed);
                QString detail;
                {
                    QMutexLocker lock(&state->label_mutex);
                    detail = state->label;
                }
                const int percent = (completed * 100) / total;
                progress_guard->setLabelText(
                        tr("Exporting to:\n%1\n\n%2\nProgress: %3 / %4 (%5%)")
                                .arg(output_dir)
                                .arg(detail)
                                .arg(completed)
                                .arg(total)
                                .arg(percent));
                if (progress_guard->wasCanceled()) {
                    state->cancelled.store(true);
                }
            });

    if (!m_batchExportWatcher) {
        m_batchExportWatcher =
                new QFutureWatcher<mcalib::BatchExportResult>(this);
    }
    m_batchExportWatcher->disconnect();
    connect(m_batchExportWatcher,
            &QFutureWatcher<mcalib::BatchExportResult>::finished, this,
            [this, progress_guard, trigger_button, output_dir, title,
             item_label]() {
                if (m_batchExportPollTimer) {
                    m_batchExportPollTimer->stop();
                }
                if (progress_guard) {
                    progress_guard->close();
                    progress_guard->deleteLater();
                }
                if (trigger_button) {
                    trigger_button->setEnabled(true);
                }
                updateExportButtonStates();

                const mcalib::BatchExportResult result =
                        m_batchExportWatcher->result();
                if (result.cancelled) {
                    if (result.exported > 0) {
                        m_lblStatus->setText(
                                tr("Export cancelled: %1/%2 %3 saved to %4")
                                        .arg(result.exported)
                                        .arg(result.total)
                                        .arg(item_label)
                                        .arg(output_dir));
                        QMessageBox::information(
                                asWidget(m_app->getMainWindow()), title,
                                tr("Export cancelled.\n%1 of %2 %3 were saved "
                                   "to:\n%4")
                                        .arg(result.exported)
                                        .arg(result.total)
                                        .arg(item_label)
                                        .arg(output_dir));
                    } else {
                        m_lblStatus->setText(tr("Export cancelled"));
                        QMessageBox::information(
                                asWidget(m_app->getMainWindow()), title,
                                tr("Export cancelled before any %1 was "
                                   "written.")
                                        .arg(item_label));
                    }
                    return;
                }

                if (result.ok()) {
                    m_lblStatus->setText(
                            tr("Batch export finished: %1 %2 -> %3")
                                    .arg(result.exported)
                                    .arg(item_label)
                                    .arg(output_dir));
                } else {
                    QMessageBox::warning(
                            asWidget(m_app->getMainWindow()), tr("Error"),
                            tr("Batch export failed or produced no %1.")
                                    .arg(item_label));
                }
            });

    m_batchExportWatcher->setFuture(
            QtConcurrent::run([task, state]() -> mcalib::BatchExportResult {
                mcalib::BatchExportProgress progress;
                progress.cancel_flag = &state->cancelled;
                progress.report = [state](int completed, int total,
                                          const std::string& label) {
                    state->completed.store(completed);
                    state->total.store(std::max(1, total));
                    {
                        QMutexLocker lock(&state->label_mutex);
                        state->label = QString::fromStdString(label);
                    }
                    return !state->cancelled.load(std::memory_order_relaxed);
                };
                return task(progress);
            }));

    m_batchExportPollTimer->start();
    progress->show();
    QCoreApplication::processEvents();
}

void ManualSensorCalibDlg::exportBevImages(const std::string& output_dir,
                                           int num_samples) {
    if (!m_bagReader || !m_bagReader->isOpen()) {
        CVLog::Warning("[SensorCalib] exportBevImages: bag not loaded");
        return;
    }

    mcalib::BevBatchExportContext ctx;
    ctx.config = m_calibConfig;
    ctx.delta_extrinsics = m_deltaExtrinsics;
    ctx.camera_topics = m_cameraTopics;
    ctx.cloud_topics = m_cloudTopics;

    mcalib::BevBatchExportOptions options;
    options.num_samples = num_samples;
    options.use_parallel_remap = true;

    const QString output_qdir = QString::fromStdString(output_dir);
    startBatchExportJob(tr("BEV Batch Export"), nullptr, output_qdir,
                        [this, ctx, output_dir, options](
                                mcalib::BatchExportProgress& progress) mutable {
                            mcalib::BevBatchExportOptions opts = options;
                            opts.progress = progress;
                            return mcalib::exportBevImagesBatch(
                                    *m_bagReader, ctx, output_dir, opts);
                        });
}

void ManualSensorCalibDlg::onExportBevBatch() {
    if (!m_configLoaded || !m_bagReader || !m_bagReader->isOpen()) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Load config and bag first"));
        return;
    }

    QString dir = QFileDialog::getExistingDirectory(
            asWidget(m_app->getMainWindow()),
            tr("Select BEV Export Directory"));
    if (dir.isEmpty()) return;

    exportBevImages(mcalib::pathFromQString(dir), 20);
}

void ManualSensorCalibDlg::onBatchExportImages() {
    if (!m_configLoaded || !m_bagReader || !m_bagReader->isOpen()) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Load config and bag first"));
        return;
    }
    if (m_viewMode != 0 && m_viewMode != 1) {
        QMessageBox::information(
                asWidget(m_app->getMainWindow()), tr("Batch Export"),
                tr("Batch export supports BEV (view mode 0) and LiDAR "
                   "projection (view mode 1) only."));
        return;
    }

    QString dir = QFileDialog::getExistingDirectory(
            asWidget(m_app->getMainWindow()),
            tr("Select Image Export Directory"));
    if (dir.isEmpty()) return;

    mcalib::ImageBatchExportContext ctx;
    ctx.config = m_calibConfig;
    ctx.delta_extrinsics = m_deltaExtrinsics;
    ctx.delta_lidar_extrinsics = m_deltaLidarExtrinsics;
    ctx.camera_topics = m_cameraTopics;
    ctx.cloud_topics = m_cloudTopics;
    ctx.bev_slot_map = getBevCameraSlotMap();
    ctx.projection_camera = m_currentSensor;
    if (m_sensorType == "lidar" && !m_calibConfig.cameras.empty() &&
        !m_calibConfig.cameras.count(ctx.projection_camera)) {
        ctx.projection_camera = m_calibConfig.cameras.begin()->first;
    }

    mcalib::ImageBatchExportOptions options;
    options.view_mode = m_viewMode;
    options.num_samples = 20;
    options.remap_mode = getGpuBackendMode();
    options.point_size = m_pointSize;

    const std::string output_dir = mcalib::pathFromQString(dir);
    const QString title = (m_viewMode == 0)
                                  ? tr("BEV Image Batch Export")
                                  : tr("Projection Image Batch Export");

    startBatchExportJob(title, m_btnBatchExportImages, dir,
                        [this, ctx, output_dir, options](
                                mcalib::BatchExportProgress& progress) mutable {
                            mcalib::ImageBatchExportOptions opts = options;
                            opts.progress = progress;
                            return mcalib::exportImagesBatch(*m_bagReader, ctx,
                                                             output_dir, opts);
                        });
}

void ManualSensorCalibDlg::onBatchExportPCD() {
    if (!m_configLoaded || !m_bagReader || !m_bagReader->isOpen()) {
        QMessageBox::warning(asWidget(m_app->getMainWindow()), tr("Error"),
                             tr("Load config and bag first"));
        return;
    }

    QString dir = QFileDialog::getExistingDirectory(
            asWidget(m_app->getMainWindow()),
            tr("Select PCD Export Directory"));
    if (dir.isEmpty()) return;

    mcalib::PcdBatchExportContext ctx;
    ctx.config = m_calibConfig;
    ctx.delta_lidar_extrinsics = m_deltaLidarExtrinsics;
    ctx.selected_lidars = m_selectedLidars;
    ctx.camera_topics = m_cameraTopics;
    ctx.cloud_topics = m_cloudTopics;
    ctx.vehicle_poses = m_vehiclePoses;
    ctx.ground_filter_min = m_groundFilterMin;
    ctx.ground_filter_max = m_groundFilterMax;
    ctx.dist_filter_min = m_distFilterMin;
    ctx.dist_filter = m_distFilter;

    mcalib::PcdBatchExportOptions options;
    options.num_samples = 20;

    const std::string output_dir = mcalib::pathFromQString(dir);
    startBatchExportJob(
            tr("PCD Batch Export"), m_btnBatchExportPCD, dir,
            [this, ctx, output_dir,
             options](mcalib::BatchExportProgress& progress) mutable {
                mcalib::PcdBatchExportOptions opts = options;
                opts.progress = progress;
                return mcalib::exportPcdsBatch(*m_bagReader, ctx, output_dir,
                                               opts);
            },
            tr("PCD files"));
}

cv::Mat ManualSensorCalibDlg::makeBevPlaceholderCanvas(
        const QString& title, const QString& subtitle) const {
    cv::Mat canvas(960, 1600, CV_8UC3, cv::Scalar(28, 28, 28));
    const cv::Scalar grid(52, 52, 52);
    for (int x = 0; x < canvas.cols; x += 80) {
        cv::line(canvas, cv::Point(x, 0), cv::Point(x, canvas.rows - 1), grid,
                 1);
    }
    for (int y = 0; y < canvas.rows; y += 80) {
        cv::line(canvas, cv::Point(0, y), cv::Point(canvas.cols - 1, y), grid,
                 1);
    }
    cv::drawMarker(canvas, cv::Point(canvas.cols / 2, canvas.rows / 2),
                   cv::Scalar(70, 70, 70), cv::MARKER_CROSS, 48, 1);

    const auto put = [&](const QString& text, int y, double scale,
                         const cv::Scalar& color) {
        cv::putText(canvas, text.toStdString(), cv::Point(40, y),
                    cv::FONT_HERSHEY_SIMPLEX, scale, color, 2, cv::LINE_AA);
    };
    put(title, 70, 0.9, cv::Scalar(210, 210, 210));
    put(subtitle, canvas.rows / 2, 0.85, cv::Scalar(180, 180, 180));
    if (!m_configLoaded) {
        put(tr("Step 1: Load cfg"), canvas.rows / 2 + 50, 0.75,
            cv::Scalar(140, 140, 140));
    }

    // Match post-rotate BEV display size (1600x960 landscape); real BEV is
    // generated portrait then rotated in updateBevView().
    return canvas;
}

void ManualSensorCalibDlg::showIdleBevCanvas() {
    if (!m_app || m_viewMode != 0) return;
    displayImageInViewer(makeBevPlaceholderCanvas(
            tr("Manual Sensor Calibration - BEV"),
            m_configLoaded ? tr("Step 2: Load bag")
                           : tr("Step 1: Load cfg, then load bag")));
}

void ManualSensorCalibDlg::displayImageInViewer(const cv::Mat& img) {
    if (!m_app) return;

    enter2DImageView();

    cv::Mat display = img;
    if (display.cols > 1920) {
        double scale = 1920.0 / display.cols;
        cv::resize(display, display, cv::Size(), scale, scale);
    }

    m_lastBevOriginalSize = img.size();
    m_lastBevDisplaySize = display.size();

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
        const QString previewName = ecvPluginDbNaming::makeUnique(
                buildManualCalibDbBaseName(m_sensorType, m_viewMode,
                                           m_configPath,
                                           QStringLiteral("Preview")),
                m_app);
        m_vtkImage = new ccImage(copied, previewName);
        m_vtkImage->setEnabled(true);
        m_vtkImage->setVisible(true);
        m_app->addToDB(m_vtkImage, false, false, false);
        CVLog::Print("[SensorCalib] displayImageInViewer: created %dx%d",
                     copied.width(), copied.height());
    } else {
        m_vtkImage->setData(copied);
        m_vtkImage->setVisible(true);
        m_vtkImage->setEnabled(true);
    }

    m_vtkImage->setRedrawFlagRecursive(true);

    if (m_vtkCloud) {
        m_vtkCloud->setVisible(false);
        m_vtkCloud->setRedrawFlagRecursive(true);
    }

    m_app->redrawAll(false, true);
    m_app->refreshAll(false, true);

    auto* app = m_app;
    QTimer::singleShot(50, [app]() {
        if (!app) return;
        app->toggle3DView(false);
        app->redrawAll(false, true);
        app->refreshAll(false, true);
    });

    CVLog::Print("[SensorCalib] displayImageInViewer: updated %dx%d",
                 copied.width(), copied.height());
}

bool ManualSensorCalibDlg::eventFilter(QObject* obj, QEvent* event) {
    QWidget* win = getAssociatedWindow();
    if (m_viewMode == 0 && win && (obj == win || obj == parent())) {
        if (event->type() == QEvent::MouseButtonPress) {
            auto* me = static_cast<QMouseEvent*>(event);
            const cv::Point p_disp = mapMouseToBevDisp(me->pos());
            if (p_disp.x >= 0 && p_disp.y >= 0 &&
                p_disp.x < m_lastBevDisplaySize.width &&
                p_disp.y < m_lastBevDisplaySize.height) {
                if (me->button() == Qt::LeftButton) {
                    if (m_bevMeasureGroundPts.size() >= 2) {
                        m_bevMeasureGroundPts.clear();
                        m_bevMeasureDispPts.clear();
                    }
                    const cv::Point p_bev = dispToBevUnrotate(p_disp);
                    Eigen::Vector3d ground_pt;
                    if (m_bevViewer && m_bevViewer->unprojectBevPixelToGround(
                                               p_bev, ground_pt)) {
                        m_bevMeasureGroundPts.emplace_back(
                                static_cast<float>(ground_pt.x()),
                                static_cast<float>(ground_pt.y()),
                                static_cast<float>(ground_pt.z()));
                    }
                    updateBevView();
                    return true;
                }
                if (me->button() == Qt::RightButton) {
                    m_bevMeasureGroundPts.clear();
                    m_bevMeasureDispPts.clear();
                    m_bevMeasureBevPxDist = 0.0;
                    updateBevView();
                    return true;
                }
            }
        }
    }

    if (event->type() == QEvent::KeyPress) {
        auto* ke = static_cast<QKeyEvent*>(event);
        if (ke->key() >= Qt::Key_1 && ke->key() <= Qt::Key_9) {
            int idx = ke->key() - Qt::Key_1;
            if (idx < static_cast<int>(m_selectedLidars.size())) {
                m_selectedLidars[idx] = !m_selectedLidars[idx];
                if (idx < static_cast<int>(m_lidarCheckboxes.size()))
                    m_lidarCheckboxes[idx]->setChecked(m_selectedLidars[idx]);
                updateFusionView();
                return true;
            }
        }
    }
    return ccOverlayDialog::eventFilter(obj, event);
}

void ManualSensorCalibDlg::displayPointCloudIn3D(
        const std::vector<Eigen::Vector3f>& points,
        const std::vector<ecvColor::Rgb>& colors) {
    if (!m_app || points.empty()) {
        CVLog::Warning(
                "[SensorCalib] displayPointCloudIn3D: no points to display");
        return;
    }
    enter3DPointCloudView();
    CVLog::Print("[SensorCalib] displayPointCloudIn3D: %zu points",
                 points.size());

    unsigned n = static_cast<unsigned>(points.size());

    if (!m_vtkCloud) {
        const QString cloudName = ecvPluginDbNaming::makeUnique(
                buildManualCalibDbBaseName(m_sensorType, m_viewMode,
                                           m_configPath,
                                           QStringLiteral("Cloud3D")),
                m_app);
        m_vtkCloud = new ccPointCloud(cloudName);
        if (!m_vtkCloud->reserve(n)) {
            delete m_vtkCloud;
            m_vtkCloud = nullptr;
            return;
        }
        bool hasCol = !colors.empty() && colors.size() == points.size();
        if (hasCol) m_vtkCloud->reserveTheRGBTable();

        for (unsigned i = 0; i < n; ++i) {
            m_vtkCloud->addPoint(
                    CCVector3(static_cast<PointCoordinateType>(points[i].x()),
                              static_cast<PointCoordinateType>(points[i].y()),
                              static_cast<PointCoordinateType>(points[i].z())));
            if (hasCol) m_vtkCloud->addRGBColor(colors[i]);
        }
        if (hasCol) {
            m_vtkCloud->showColors(true);
        } else {
            auto* sf = new ccScalarField("Depth");
            if (sf->reserveSafe(n)) {
                for (unsigned i = 0; i < n; ++i)
                    sf->addElement(static_cast<ScalarType>(points[i].norm()));
                sf->computeMinAndMax();
                int sfIdx = m_vtkCloud->addScalarField(sf);
                m_vtkCloud->setCurrentDisplayedScalarField(sfIdx);
                m_vtkCloud->showSF(true);
            } else {
                sf->release();
            }
        }
        m_vtkCloud->setVisible(true);
        m_vtkCloud->setEnabled(true);
        m_vtkCloud->setPointSize(static_cast<unsigned char>(m_pointSize));
        m_app->addToDB(m_vtkCloud, false, false, false);
    } else {
        m_vtkCloud->clear();
        if (!m_vtkCloud->reserve(n)) return;

        bool hasCol = !colors.empty() && colors.size() == points.size();
        if (hasCol) m_vtkCloud->reserveTheRGBTable();

        for (unsigned i = 0; i < n; ++i) {
            m_vtkCloud->addPoint(
                    CCVector3(static_cast<PointCoordinateType>(points[i].x()),
                              static_cast<PointCoordinateType>(points[i].y()),
                              static_cast<PointCoordinateType>(points[i].z())));
            if (hasCol) m_vtkCloud->addRGBColor(colors[i]);
        }
        if (hasCol) {
            m_vtkCloud->showColors(true);
        } else {
            m_vtkCloud->deleteAllScalarFields();
            auto* sf = new ccScalarField("Depth");
            if (sf->reserveSafe(n)) {
                for (unsigned i = 0; i < n; ++i)
                    sf->addElement(static_cast<ScalarType>(points[i].norm()));
                sf->computeMinAndMax();
                int sfIdx = m_vtkCloud->addScalarField(sf);
                m_vtkCloud->setCurrentDisplayedScalarField(sfIdx);
                m_vtkCloud->showSF(true);
            } else {
                sf->release();
            }
        }
        m_vtkCloud->setPointSize(static_cast<unsigned char>(m_pointSize));
        m_vtkCloud->setVisible(true);
        m_vtkCloud->setRedrawFlagRecursive(true);
    }

    zoomToPreviewCloud();

    QPointer<ManualSensorCalibDlg> self(this);
    QTimer::singleShot(50, [self]() {
        if (!self) return;
        self->enter3DPointCloudView();
        self->zoomToPreviewCloud();
    });
}

void ManualSensorCalibDlg::zoomToPreviewCloud() {
    if (!m_app || !m_vtkCloud || m_vtkCloud->size() == 0) return;

    m_vtkCloud->refreshBB();
    m_vtkCloud->setRedrawFlagRecursive(true);
    m_app->zoomOnEntities(m_vtkCloud);
    m_app->redrawAll(true);
    m_app->refreshAll(true);
}

void ManualSensorCalibDlg::onExportImage() { exportImageToDB(); }
