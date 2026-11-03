// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VLXSeekDialog.h"

#ifdef AICore_ENABLED
#include <aicore/backend_capi.h>
#include <aicore/vlx_seek_capi.h>
#endif

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QVBoxLayout>

#include "ecvAICoreUiHelper.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

namespace {

constexpr int kRoleWedetect = 0;
constexpr int kRoleVision = 1;
constexpr int kRoleLlm = 2;

}  // namespace

VLXSeekDialog::VLXSeekDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("VLX-Seek Fine-Grained Grounding"));
    resize(560, 480);

    m_catalog = VLXSeekHelpers::catalogModels();

    // ---- fixed top block: image input + actions + progress (never scroll)
    // ----
    m_inputLabel = new QLabel(tr("Input image:"), this);
    m_inputEdit = new QLineEdit(this);
    m_inputEdit->setPlaceholderText(
            tr("Image path (DB selection, test data "
               "or file)"));
    m_browseButton = ecvAICoreUi::makeBrowseBtn(tr("Browse..."), this);
    m_testDataButton = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataButton->setToolTip(
            tr("Downloads the official VLX-Seek test images once into the "
               "shared data root and fills the demo image"));

    m_runButton = new QPushButton(tr("Run"), this);
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setEnabled(false);
    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_statusLabel = new QLabel(this);
    m_statusLabel->setWordWrap(true);

    auto* inputRow = new QHBoxLayout();
    inputRow->addWidget(m_inputEdit, 1);
    inputRow->addWidget(m_testDataButton);
    inputRow->addWidget(m_browseButton);

    auto* actionRow = new QHBoxLayout();
    actionRow->setSpacing(ecvAICoreUi::hSpacing());
    m_exportDb = new QCheckBox(tr("Export result to DB tree"), this);
    m_exportDb->setChecked(true);
    m_exportDb->setToolTip(
            tr("Add the annotated result image to the DB tree after every "
               "run"));
    actionRow->addWidget(m_exportDb);
    actionRow->addStretch();
    actionRow->addWidget(m_runButton);
    actionRow->addWidget(m_cancelButton);
    actionRow->addWidget(m_progress, 1);

    // ---- preview row (fixed top block): input + result thumbnails with
    // click-to-enlarge, following the shared qDA3/qRFDetr pattern.
    auto* previewRow = new QHBoxLayout();
    previewRow->setSpacing(ecvAICoreUi::hSpacing());
    m_inputPreview = new ecvClickableImageLabel(this);
    m_resultPreview = new ecvClickableImageLabel(this);
    const int ps = ecvAICoreUi::previewSize();
    for (ecvClickableImageLabel* label : {m_inputPreview, m_resultPreview}) {
        label->setFixedSize(ps, ps);
        label->setAlignment(Qt::AlignCenter);
        label->setStyleSheet(QStringLiteral(
                "border: 1px solid palette(mid); background: palette(base);"));
    }
    m_inputPreview->setText(tr("Input"));
    m_resultPreview->setText(tr("Result"));
    auto* inputPreviewBox = new QVBoxLayout();
    inputPreviewBox->setContentsMargins(0, 0, 0, 0);
    inputPreviewBox->addWidget(m_inputPreview, 0, Qt::AlignHCenter);
    inputPreviewBox->addWidget(new QLabel(tr("Input"), this), 0,
                               Qt::AlignHCenter);
    auto* resultPreviewBox = new QVBoxLayout();
    resultPreviewBox->setContentsMargins(0, 0, 0, 0);
    resultPreviewBox->addWidget(m_resultPreview, 0, Qt::AlignHCenter);
    resultPreviewBox->addWidget(new QLabel(tr("Result"), this), 0,
                                Qt::AlignHCenter);
    previewRow->addLayout(inputPreviewBox);
    previewRow->addStretch();
    previewRow->addLayout(resultPreviewBox);

    auto* topLayout = new QVBoxLayout();
    topLayout->addWidget(m_inputLabel);
    topLayout->addLayout(inputRow);
    topLayout->addLayout(previewRow);
    topLayout->addLayout(actionRow);
    topLayout->addWidget(m_statusLabel);

    // ---- scrollable configuration area ----
    auto* form = new QFormLayout();
    m_taskCombo = new QComboBox(this);
    m_taskCombo->addItems(
            {QStringLiteral("detection"), QStringLiteral("grounding_single"),
             QStringLiteral("counting"), QStringLiteral("reasoning_detection"),
             QStringLiteral("brief_region_caption"),
             QStringLiteral("region_ocr"), QStringLiteral("vqa")});
    m_langCombo = new QComboBox(this);
    m_langCombo->addItems({QStringLiteral("en"), QStringLiteral("zh")});
    m_textEdit = new QLineEdit(this);
    m_textEdit->setPlaceholderText(
            tr("Target description, e.g. person wearing red"));
    m_targetsEdit = new QLineEdit(this);
    m_targetsEdit->setPlaceholderText(tr("e.g. 2, 3"));
    m_targetsEdit->setToolTip(
            tr("Zero-based detector proposal indexes to describe or read "
               "text from (required for region caption / OCR)"));
    m_targetsRow = new QWidget(this);
    auto* targetsLayout = new QHBoxLayout(m_targetsRow);
    targetsLayout->setContentsMargins(0, 0, 0, 0);
    targetsLayout->addWidget(new QLabel(tr("Target indexes:"), this));
    targetsLayout->addWidget(m_targetsEdit, 1);
    m_targetsRow->setVisible(false);
    m_precisionCombo = new QComboBox(this);
    m_deviceCombo = new QComboBox(this);
    m_threadsSpin = new QSpinBox(this);
    m_threadsSpin->setRange(0, 256);
    m_threadsSpin->setSpecialValueText(tr("auto"));
    m_tokensSpin = new QSpinBox(this);
    m_tokensSpin->setRange(16, 4096);
    m_tokensSpin->setValue(1024);

    form->addRow(tr("Task:"), m_taskCombo);
    form->addRow(tr("Language:"), m_langCombo);
    form->addRow(tr("Text:"), m_textEdit);
    form->addRow(m_targetsRow);
    form->addRow(tr("Precision:"), m_precisionCombo);
    form->addRow(tr("Device:"), m_deviceCombo);
    form->addRow(tr("CPU threads:"), m_threadsSpin);
    form->addRow(tr("Max new tokens:"), m_tokensSpin);

    auto* formHost = new QWidget(this);
    formHost->setLayout(form);
    m_configScroll = new QScrollArea(this);
    m_configScroll->setWidget(formHost);
    m_configScroll->setWidgetResizable(true);
    m_configScroll->setFrameShape(QFrame::NoFrame);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(topLayout);
    mainLayout->addWidget(m_configScroll, 1);

    populateCatalogCombos();

    connect(m_browseButton, &QPushButton::clicked, this,
            &VLXSeekDialog::onBrowse);
    connect(m_testDataButton, &QPushButton::clicked, this,
            &VLXSeekDialog::onUseTestData);
    connect(m_runButton, &QPushButton::clicked, this, &VLXSeekDialog::onRun);
    connect(m_cancelButton, &QPushButton::clicked, this,
            &VLXSeekDialog::onCancel);
    connect(m_inputEdit, &QLineEdit::textChanged, this,
            [this](const QString&) { updateInputPreview(); });
    connect(m_taskCombo, &QComboBox::currentTextChanged, this,
            [this](const QString&) { updateTargetsRowVisibility(); });

    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            &VLXSeekDialog::onDownloadProgress);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            &VLXSeekDialog::onDownloadFinished);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            [this](const QString& message) {
                m_statusLabel->setText(message);
            });

    updateRunState();
}

VLXSeekDialog::~VLXSeekDialog() {
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->wait(15000);
        m_worker->deleteLater();
        m_worker = nullptr;
    }
}

void VLXSeekDialog::populateCatalogCombos() {
    // One precision row per vision entry; the wedetect f32 detector and the
    // matching LLM row are derived from the same family.
    m_precisionCombo->clear();
    for (const VLXSeekModelEntry& entry : m_catalog) {
        if (entry.role != kRoleVision) continue;
        m_precisionCombo->addItem(entry.displayName, entry.filename);
    }
    const int def = VLXSeekHelpers::catalogDefaultIndex();
    if (def >= 0 && def < m_catalog.size() &&
        m_catalog[def].role == kRoleVision) {
        const int idx = m_precisionCombo->findData(m_catalog[def].filename);
        if (idx >= 0) m_precisionCombo->setCurrentIndex(idx);
    }
    m_deviceCombo->clear();
    m_deviceCombo->addItem(tr("auto"));
#ifdef AICore_ENABLED
    const int devices = aicore_device_count();
    for (int i = 0; i < devices; ++i) {
        if (const aicore_device_info* info = aicore_device_at(i))
            m_deviceCombo->addItem(QString::fromUtf8(info->id));
    }
#endif
}

void VLXSeekDialog::setInputPath(const QString& path) {
    if (path.isEmpty()) return;
    m_inputEdit->setText(path);
}

bool VLXSeekDialog::exportToDbRequested() const {
    return m_exportDb && m_exportDb->isChecked();
}

void VLXSeekDialog::updateInputPreview() {
    if (!m_inputPreview) return;
    const QString path = m_inputEdit->text().trimmed();
    QImage img;
    if (!path.isEmpty() && QFileInfo(path).isFile()) img = QImage(path);
    if (img.isNull()) {
        m_inputPreview->clearPreview();
        m_inputPreview->setText(tr("Input"));
        return;
    }
    m_inputPreview->setPreviewImage(img, ecvAICoreUi::previewSize());
}

void VLXSeekDialog::updateTargetsRowVisibility() {
    if (!m_targetsRow) return;
    const QString task = m_taskCombo->currentText();
    // The runtime hard-rejects region tasks without target indexes; only
    // those two tasks consume the proposal-index list.
    m_targetsRow->setVisible(task == QStringLiteral("brief_region_caption") ||
                             task == QStringLiteral("region_ocr"));
}

void VLXSeekDialog::onBrowse() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select input image"), QString(),
            tr("Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files "
               "(*)"));
    if (!path.isEmpty()) m_inputEdit->setText(path);
}

void VLXSeekDialog::onUseTestData() {
    // Absent cache is a recoverable preparation, not an error: the shared
    // repository downloads, verifies and extracts once, then this slot
    // refills the input from the pinned demo member.
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    if (repo.isDatasetAvailable(ecvTestDataRepository::Dataset::VlxSeek)) {
        const QString demo = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::VlxSeek,
                QStringLiteral("demo_image.jpg"));
        if (!demo.isEmpty()) {
            setInputPath(demo);
            m_statusLabel->setText(tr("Test data ready: %1").arg(demo));
            return;
        }
    }
    m_testDataButton->setEnabled(false);
    m_statusLabel->setText(tr("Preparing official test images..."));
    connect(
            &repo, &ecvTestDataRepository::datasetReady, this,
            [this](bool ok, ecvTestDataRepository::Dataset kind) {
                if (kind != ecvTestDataRepository::Dataset::VlxSeek) return;
                m_testDataButton->setEnabled(true);
                if (!ok) {
                    m_statusLabel->setText(
                            tr("Test data preparation failed (see console "
                               "for details)"));
                    return;
                }
                const QString demo = ecvTestDataRepository::findDatasetFile(
                        ecvTestDataRepository::Dataset::VlxSeek,
                        QStringLiteral("demo_image.jpg"));
                setInputPath(demo);
                m_statusLabel->setText(tr("Test data ready: %1").arg(demo));
            },
            Qt::UniqueConnection);
    repo.ensureDatasetReady(ecvTestDataRepository::Dataset::VlxSeek);
}

void VLXSeekDialog::onDownloadProgress(qint64 received, qint64 total) {
    if (total > 0) {
        m_progress->setValue(int(received * 100 / total));
        m_statusLabel->setText(
                tr("Downloading %1...").arg(m_downloadTargetFilename));
    }
}

void VLXSeekDialog::onDownloadFinished(bool ok, const QString& path) {
    m_downloadInProgress = false;
    m_progress->setValue(ok ? 100 : 0);
    if (!ok) {
        m_statusLabel->setText(
                tr("Download failed for %1 (see console)").arg(path));
        m_pendingAction = PendingAction::None;
        updateRunState();
        return;
    }
    m_statusLabel->setText(tr("Model ready: %1").arg(path));
    if (m_pendingAction == PendingAction::Run) onRun();
}

void VLXSeekDialog::startDownload(const VLXSeekModelEntry& entry) {
    m_downloadInProgress = true;
    m_downloadTargetFilename = entry.filename;
    m_progress->setValue(0);
    ecvModelDownloader::Request request;
    request.url = entry.downloadUrl;
    request.destPath =
            QDir(VLXSeekHelpers::modelCacheDir()).filePath(entry.filename);
    request.contentAnchor = {QCryptographicHash::Sha256, entry.sha256.toUtf8()};
    m_downloader->download(request);
}

bool VLXSeekDialog::ensureModelsAvailable() {
    const QString precisionRow = m_precisionCombo->currentData().toString();
    VLXSeekModelEntry vision;
    if (!VLXSeekHelpers::findModelByFilename(precisionRow, &vision))
        return false;
    VLXSeekModelEntry wedetect, llm;
    for (const VLXSeekModelEntry& entry : m_catalog) {
        if (entry.role == kRoleWedetect && entry.quantization == "f32")
            wedetect = entry;
        if (entry.role == kRoleLlm && entry.quantization == vision.quantization)
            llm = entry;
    }
    const QString cache = VLXSeekHelpers::modelCacheDir();
    struct Need {
        VLXSeekModelEntry entry;
        QString role;
    };
    const QVector<Need> needs = {{wedetect, QStringLiteral("detector")},
                                 {vision, QStringLiteral("vision")},
                                 {llm, QStringLiteral("LLM")}};
    for (const Need& need : needs) {
        if (need.entry.filename.isEmpty()) return false;
        const QString path = QDir(cache).filePath(need.entry.filename);
        if (QFileInfo::exists(path)) continue;
        if (m_downloadInProgress) return false;  // one download at a time
        m_statusLabel->setText(tr("Downloading %1 (%2)...")
                                       .arg(need.entry.filename, need.role));
        startDownload(need.entry);
        return false;
    }
    m_settings.wedetectPath = QDir(cache).filePath(wedetect.filename);
    m_settings.visionPath = QDir(cache).filePath(vision.filename);
    m_settings.llmPath = QDir(cache).filePath(llm.filename);
    return true;
}

void VLXSeekDialog::onRun() {
    if (m_worker) return;  // one running frame at a time
    if (m_inputEdit->text().isEmpty() ||
        !QFileInfo::exists(m_inputEdit->text())) {
        QMessageBox::information(this, tr("VLX-Seek"),
                                 tr("Select an input image first."));
        return;
    }
    m_pendingAction = PendingAction::Run;
    if (!ensureModelsAvailable()) {
        updateRunState();
        return;  // download in flight; onDownloadFinished re-enters onRun
    }

    VLXSeekWorker::Settings settings;
    settings.wedetectPath = m_settings.wedetectPath;
    settings.visionPath = m_settings.visionPath;
    settings.llmPath = m_settings.llmPath;
    {
        // Quantization of the selected catalog row (reported in model_info).
        VLXSeekModelEntry vision;
        if (VLXSeekHelpers::findModelByFilename(
                    m_precisionCombo->currentData().toString(), &vision))
            settings.precision = vision.quantization;
    }
    settings.device = m_deviceCombo->currentText().toLower();
    settings.threads = m_threadsSpin->value();
    settings.maxNewTokens = m_tokensSpin->value();
    settings.inputPath = m_inputEdit->text();
    settings.task = m_taskCombo->currentText();
    settings.lang = m_langCombo->currentText();
    settings.text = m_textEdit->text();
    settings.targetIndexes = m_targetsEdit->text();

    m_worker = new VLXSeekWorker(settings, this);
    connect(m_worker, &VLXSeekWorker::logMessage, this,
            [this](const QString& message) {
                m_statusLabel->setText(message);
            });
    connect(m_worker, &VLXSeekWorker::resultReady, this,
            &VLXSeekDialog::onResultPreview);
    connect(m_worker, &VLXSeekWorker::taskFinished, this,
            &VLXSeekDialog::onWorkerFinished);
    m_progress->setRange(0, 0);  // indeterminate while inference runs
    m_statusLabel->setText(tr("Running %1...").arg(settings.task));
    updateRunState();
    m_worker->start();
    emit runRequested(settings);
}

void VLXSeekDialog::onCancel() {
    if (m_downloadInProgress) {
        m_downloader->cancel();
        return;
    }
    // The vlx task decode is a single greedy call per request; a cancel
    // takes effect after the current request settles (worker completion).
    m_statusLabel->setText(
            tr("Cancel requested; waiting for the current "
               "request to settle..."));
}

void VLXSeekDialog::onResultPreview(const VLXSeekRunResult& result) {
    if (m_resultPreview) {
        if (result.annotatedImage.isNull()) {
            m_resultPreview->clearPreview();
            m_resultPreview->setText(tr("Result"));
        } else {
            m_resultPreview->setPreviewImage(result.annotatedImage,
                                             ecvAICoreUi::previewSize());
        }
    }
    m_lastSummary = tr("%1 box(es) — %2")
                            .arg(result.boxes.size())
                            .arg(result.answer.left(120));
    emit resultReady(result);
}

void VLXSeekDialog::onWorkerFinished(bool success) {
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->wait(15000);
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    m_progress->setRange(0, 100);
    m_progress->setValue(success ? 100 : 0);
    m_pendingAction = PendingAction::None;
    if (success && !m_lastSummary.isEmpty()) {
        m_statusLabel->setText(tr("Done: %1").arg(m_lastSummary));
    } else {
        m_statusLabel->setText(success ? tr("Done.")
                                       : tr("Failed (see console)."));
    }
    updateRunState();
}

void VLXSeekDialog::updateRunState() {
    const bool busy = m_worker != nullptr || m_downloadInProgress;
    m_runButton->setEnabled(!busy);
    m_testDataButton->setEnabled(!busy && m_testDataButton->isEnabled());
    m_cancelButton->setEnabled(busy);
}
