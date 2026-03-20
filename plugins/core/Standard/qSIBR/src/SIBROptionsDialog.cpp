// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SIBROptionsDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QVBoxLayout>

namespace {
QString settingsKey(int type, const QString& field) {
    return QString("qSIBR/viewer%1/%2").arg(type).arg(field);
}
}  // namespace

SIBROptionsDialog::SIBROptionsDialog(ViewerType type, QWidget* parent)
    : QDialog(parent), m_type(type) {
    setupUI();
}

void SIBROptionsDialog::setupUI() {
    QString title;
    switch (m_type) {
        case ULR:
            title = tr("ULR Viewer Options");
            break;
        case ULRv2:
            title = tr("ULR v2/v3 Viewer Options");
            break;
        case TexturedMesh:
            title = tr("Textured Mesh Viewer Options");
            break;
        case PointBased:
            title = tr("Point-Based Viewer Options");
            break;
        case GaussianSplatting:
            title = tr("3D Gaussian Splatting Viewer Options");
            break;
        case RemoteGaussian:
            title = tr("Remote Gaussian Viewer Options");
            break;
    }
    setWindowTitle(title);
    setMinimumWidth(500);

    auto* mainLayout = new QVBoxLayout(this);

    auto* pathGroup = new QGroupBox(tr("Dataset"));
    auto* pathLayout = new QFormLayout(pathGroup);

    m_selectionLabel = new QLabel();
    m_selectionLabel->setStyleSheet(
            "QLabel { color: #1a73e8; font-style: italic; padding: 2px; }");
    m_selectionLabel->setVisible(false);
    pathLayout->addRow("", m_selectionLabel);

    m_datasetPath = new QLineEdit();
    if (m_type == RemoteGaussian) {
        m_datasetPath->setPlaceholderText(
                tr("(optional) local scene for camera overlay"));
    }
    auto* browseBtn = new QPushButton(tr("Browse..."));
    connect(browseBtn, &QPushButton::clicked, this,
            &SIBROptionsDialog::browseDatasetPath);
    auto* pathRow = new QHBoxLayout();
    pathRow->addWidget(m_datasetPath, 1);
    pathRow->addWidget(browseBtn);
    QString pathLabel = (m_type == RemoteGaussian)
                                ? tr("Dataset Path (optional):")
                                : tr("Dataset Path:");
    pathLayout->addRow(pathLabel, pathRow);

    if (m_type == GaussianSplatting) {
        setupGaussianFields();
        m_modelPath = new QLineEdit();
        auto* modelBtn = new QPushButton(tr("Browse..."));
        connect(modelBtn, &QPushButton::clicked, this,
                &SIBROptionsDialog::browseModelPath);
        auto* modelRow = new QHBoxLayout();
        modelRow->addWidget(m_modelPath, 1);
        modelRow->addWidget(modelBtn);
        pathLayout->addRow(tr("Model Path:"), modelRow);

        m_iteration = new QLineEdit();
        m_iteration->setPlaceholderText(tr("auto (latest)"));
        pathLayout->addRow(tr("Iteration:"), m_iteration);
    }
    mainLayout->addWidget(pathGroup);

    auto* renderGroup = new QGroupBox(tr("Rendering"));
    auto* renderLayout = new QFormLayout(renderGroup);

    m_width = new QSpinBox();
    m_width->setRange(320, 7680);
    m_width->setValue(1280);
    m_width->setSuffix(" px");
    renderLayout->addRow(tr("Width:"), m_width);

    m_height = new QSpinBox();
    m_height->setRange(240, 4320);
    m_height->setValue(720);
    m_height->setSuffix(" px");
    renderLayout->addRow(tr("Height:"), m_height);

    m_forceAspectRatio = new QCheckBox(tr("Force aspect ratio"));
    renderLayout->addRow("", m_forceAspectRatio);

    m_loadImages = new QCheckBox(tr("Load input images (scene overview)"));
    renderLayout->addRow("", m_loadImages);

    mainLayout->addWidget(renderGroup);

    if (m_type == GaussianSplatting) {
        auto* cudaGroup = new QGroupBox(tr("CUDA"));
        auto* cudaLayout = new QFormLayout(cudaGroup);

        m_device = new QSpinBox();
        m_device->setRange(0, 15);
        m_device->setValue(0);
        cudaLayout->addRow(tr("CUDA Device:"), m_device);

        m_noInterop = new QCheckBox(tr("Disable CUDA-GL Interop (WSL/Mesa)"));
        cudaLayout->addRow("", m_noInterop);

        mainLayout->addWidget(cudaGroup);
    }

    if (m_type == RemoteGaussian) {
        setupRemoteFields();
        auto* netGroup = new QGroupBox(tr("Network"));
        auto* netLayout = new QFormLayout(netGroup);

        m_remoteIP = new QLineEdit("127.0.0.1");
        netLayout->addRow(tr("IP Address:"), m_remoteIP);

        m_remotePort = new QSpinBox();
        m_remotePort->setRange(1, 65535);
        m_remotePort->setValue(6009);
        netLayout->addRow(tr("Port:"), m_remotePort);

        mainLayout->addWidget(netGroup);
    }

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok |
                                         QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(buttons);

    QSettings s;
    if (m_datasetPath && m_datasetPath->text().isEmpty())
        m_datasetPath->setText(
                s.value(settingsKey(m_type, "dataset")).toString());
    if (m_modelPath && m_modelPath->text().isEmpty())
        m_modelPath->setText(s.value(settingsKey(m_type, "model")).toString());
    if (m_iteration && m_iteration->text().isEmpty())
        m_iteration->setText(
                s.value(settingsKey(m_type, "iteration")).toString());
    if (m_remoteIP && m_remoteIP->text() == "127.0.0.1") {
        QString savedIP = s.value(settingsKey(m_type, "ip")).toString();
        if (!savedIP.isEmpty()) m_remoteIP->setText(savedIP);
    }
    if (m_remotePort) {
        int savedPort = s.value(settingsKey(m_type, "port"), 0).toInt();
        if (savedPort > 0) m_remotePort->setValue(savedPort);
    }
}

void SIBROptionsDialog::setupGaussianFields() {}
void SIBROptionsDialog::setupRemoteFields() {}

void SIBROptionsDialog::setInitialPaths(const QString& datasetPath,
                                        const QString& modelPath) {
    if (m_datasetPath && !datasetPath.isEmpty())
        m_datasetPath->setText(datasetPath);
    if (m_modelPath && !modelPath.isEmpty()) m_modelPath->setText(modelPath);
}

void SIBROptionsDialog::setSelectedEntityInfo(const QString& entityName,
                                              const QString& entityType) {
    if (!m_selectionLabel) return;
    if (entityName.isEmpty()) {
        m_selectionLabel->setVisible(false);
        return;
    }
    m_selectionLabel->setText(
            tr("From selected %1: %2").arg(entityType, entityName));
    m_selectionLabel->setVisible(true);
}

SIBRViewerOptions SIBROptionsDialog::getOptions() const {
    SIBRViewerOptions opts;
    opts.width = m_width->value();
    opts.height = m_height->value();
    opts.forceAspectRatio =
            m_forceAspectRatio ? m_forceAspectRatio->isChecked() : false;
    opts.loadImages = m_loadImages ? m_loadImages->isChecked() : false;
    opts.datasetPath = m_datasetPath->text();

    if (m_modelPath) opts.modelPath = m_modelPath->text();
    if (m_iteration) opts.iteration = m_iteration->text();
    if (m_device) opts.device = m_device->value();
    if (m_noInterop) opts.noInterop = m_noInterop->isChecked();
    if (m_remoteIP) opts.remoteIP = m_remoteIP->text();
    if (m_remotePort) opts.remotePort = m_remotePort->value();

    QSettings s;
    if (!opts.datasetPath.isEmpty())
        s.setValue(settingsKey(m_type, "dataset"), opts.datasetPath);
    if (!opts.modelPath.isEmpty())
        s.setValue(settingsKey(m_type, "model"), opts.modelPath);
    if (!opts.iteration.isEmpty())
        s.setValue(settingsKey(m_type, "iteration"), opts.iteration);
    if (m_remoteIP) s.setValue(settingsKey(m_type, "ip"), opts.remoteIP);
    if (m_remotePort) s.setValue(settingsKey(m_type, "port"), opts.remotePort);

    return opts;
}

void SIBROptionsDialog::browseDatasetPath() {
    QString dir = QFileDialog::getExistingDirectory(
            this, tr("Select Dataset Directory"), m_datasetPath->text(),
            QFileDialog::ShowDirsOnly);
    if (!dir.isEmpty()) m_datasetPath->setText(dir);
}

void SIBROptionsDialog::browseModelPath() {
    QString dir = QFileDialog::getExistingDirectory(
            this, tr("Select Model Directory"),
            m_modelPath ? m_modelPath->text() : "", QFileDialog::ShowDirsOnly);
    if (!dir.isEmpty() && m_modelPath) m_modelPath->setText(dir);
}
