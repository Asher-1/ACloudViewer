// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "WeVisOneDialog.h"

#ifdef AICore_ENABLED
#include <aicore/backend_capi.h>
#include <aicore/wevisone_capi.h>
#endif

#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QVBoxLayout>

#include "ecvAICoreDeviceUiHelper.h"
#include "ecvAICoreUiHelper.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

#ifdef AICore_ENABLED
#include <ecvAssetIntegrity.h>
#endif

namespace {

constexpr int kRoleVlm = 0;
constexpr int kRoleMmproj = 1;
constexpr int kRoleDecoder = 2;

}  // namespace

WeVisOneDialog::WeVisOneDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("WeVisOne-VL Unified Visual Perception"));
    // Geometry conservation (qQuerySplat rule): the fixed top block plus a
    // minimum visible config-scroll height must fit the minimum window.
    resize(680, 780);
    setMinimumSize(620, 660);

    m_catalog = WeVisOneHelpers::catalogModels();

    // ---- fixed top block: image input + actions + progress (never scroll)
    m_inputLabel = new QLabel(tr("Input image:"), this);
    m_inputEdit = new QLineEdit(this);
    m_inputEdit->setObjectName(QStringLiteral("inputImage"));
    m_inputEdit->setPlaceholderText(
            tr("Image path (DB selection, test data "
               "or file)"));
    m_browseButton = ecvAICoreUi::makeBrowseBtn(tr("Browse..."), this);
    m_testDataButton = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataButton->setObjectName(QStringLiteral("sampleData"));
    m_testDataButton->setToolTip(
            tr("Downloads the official WeVisOne-VL test images once into the "
               "shared data root and fills the demo image plus the "
               "task-specific query from the upstream test-data manifest"));

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

    // ---- preview row (fixed top block): the input thumbnail with
    // click-to-enlarge (the shared qDA3/qRFDetr pattern). The annotated
    // result is visualized through the DB tree ccImage export instead of a
    // second dialog panel, matching qMapAnything/qQuerySplat.
    auto* previewRow = new QHBoxLayout();
    previewRow->setSpacing(ecvAICoreUi::hSpacing());
    m_inputPreview = new ecvClickableImageLabel(this);
    const int ps = ecvAICoreUi::previewSize();
    m_inputPreview->setFixedSize(ps, ps);
    m_inputPreview->setAlignment(Qt::AlignCenter);
    m_inputPreview->setStyleSheet(QStringLiteral(
            "border: 1px solid palette(mid); background: palette(base);"));
    m_inputPreview->setText(tr("Input"));
    auto* inputPreviewBox = new QVBoxLayout();
    inputPreviewBox->setContentsMargins(0, 0, 0, 0);
    inputPreviewBox->addWidget(m_inputPreview, 0, Qt::AlignHCenter);
    inputPreviewBox->addWidget(new QLabel(tr("Input"), this), 0,
                               Qt::AlignHCenter);
    previewRow->addStretch();
    previewRow->addLayout(inputPreviewBox);
    previewRow->addStretch();

    auto* topLayout = new QVBoxLayout();
    topLayout->addWidget(m_inputLabel);
    topLayout->addLayout(inputRow);
    topLayout->addLayout(previewRow);
    topLayout->addLayout(actionRow);
    topLayout->addWidget(m_statusLabel);

    // ---- scrollable configuration area ----
    auto* form = new QFormLayout();
    m_taskCombo = new QComboBox(this);
    m_taskCombo->setObjectName(QStringLiteral("taskMode"));
    m_taskCombo->addItems(WeVisOneHelpers::taskNames());
    // Multi-line query: the detection task consumes a newline-separated
    // category list (the upstream prompts.py join), so a single-line edit
    // cannot express the multi-category demo.
    m_queryEdit = new QPlainTextEdit(this);
    m_queryEdit->setObjectName(QStringLiteral("queryText"));
    m_queryEdit->setFixedHeight(56);
    m_queryEdit->setPlaceholderText(
            tr("Categories (one per line), referring phrase or freeform "
               "instruction"));
    m_refBoxesEdit = new QLineEdit(this);
    m_refBoxesEdit->setObjectName(QStringLiteral("refBoxes"));
    m_refBoxesEdit->setPlaceholderText(
            tr("x1,y1,x2,y2[;x1,y1,x2,y2...] in image pixels"));
    m_refBoxesEdit->setToolTip(
            tr("Reference boxes for the visual_prompt task (upstream "
               "--ref-boxes format): the model locates all objects similar "
               "to the referenced regions."));
    m_refBoxesRow = new QWidget(this);
    auto* refBoxesLayout = new QHBoxLayout(m_refBoxesRow);
    refBoxesLayout->setContentsMargins(0, 0, 0, 0);
    refBoxesLayout->addWidget(new QLabel(tr("Reference boxes:"), this));
    refBoxesLayout->addWidget(m_refBoxesEdit, 1);
    m_refBoxesRow->setVisible(false);
    m_sizeCombo = new QComboBox(this);
    m_precisionCombo = new QComboBox(this);
    m_decoderPrecisionCombo = new QComboBox(this);
    m_decodeMasks = new QCheckBox(tr("Decode masks / mattes"), this);
    m_decodeMasks->setChecked(true);
    m_decodeMasks->setToolTip(
            tr("Run the pixel decoder when the answer carries <OBJ> tokens "
               "(segmentation / matting). Uncheck for a pure object-level "
               "detector (boxes / points / counts / OCR)."));
    m_deviceCombo = new QComboBox(this);
    m_threadsSpin = new QSpinBox(this);
    m_threadsSpin->setRange(0, 256);
    m_threadsSpin->setSpecialValueText(tr("auto"));
    m_tokensSpin = new QSpinBox(this);
    m_tokensSpin->setRange(16, 8192);
    m_tokensSpin->setValue(4096);

    form->addRow(tr("Task:"), m_taskCombo);
    form->addRow(tr("Query:"), m_queryEdit);
    form->addRow(m_refBoxesRow);
    form->addRow(tr("Model size:"), m_sizeCombo);
    form->addRow(tr("VLM precision:"), m_precisionCombo);
    form->addRow(tr("Decoder precision:"), m_decoderPrecisionCombo);
    form->addRow(QString(), m_decodeMasks);
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
    updateRefBoxesVisibility();

    connect(m_browseButton, &QPushButton::clicked, this,
            &WeVisOneDialog::onBrowse);
    connect(m_testDataButton, &QPushButton::clicked, this,
            &WeVisOneDialog::onUseTestData);
    connect(m_runButton, &QPushButton::clicked, this, &WeVisOneDialog::onRun);
    connect(m_cancelButton, &QPushButton::clicked, this,
            &WeVisOneDialog::onCancel);
    connect(m_inputEdit, &QLineEdit::textChanged, this,
            [this](const QString&) { updateInputPreview(); });
    connect(m_taskCombo, &QComboBox::currentTextChanged, this,
            [this](const QString& task) {
                updateRefBoxesVisibility();
                // The ocr task runs a fixed prompt; visual_prompt grounds
                // from the reference boxes — the query field is unused.
                if (task == QStringLiteral("ocr")) {
                    m_queryEdit->setPlaceholderText(
                            tr("(fixed OCR prompt — query unused)"));
                } else if (task == QStringLiteral("visual_prompt")) {
                    m_queryEdit->setPlaceholderText(
                            tr("(unused — grounding comes from the "
                               "reference boxes)"));
                } else if (task == QStringLiteral("freeform")) {
                    m_queryEdit->setPlaceholderText(
                            tr("Free instruction, e.g. describe the image "
                               "with bounding boxes in detail"));
                } else {
                    m_queryEdit->setPlaceholderText(
                            tr("Categories (one per line), referring phrase "
                               "or freeform instruction"));
                }
            });
    connect(m_sizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { repopulatePrecisionCombo(); });

    m_downloader = new ecvModelDownloader(this);

    connect(this, &QDialog::finished, m_downloader,
            &ecvModelDownloader::cancel);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            &WeVisOneDialog::onDownloadProgress);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            &WeVisOneDialog::onDownloadFinished);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            [this](const QString& message) {
                m_statusLabel->setText(message);
            });

    connect(this, &QDialog::finished, this, [this] {
        m_testDataBusy = false;
        m_testDataButton->setEnabled(true);
        m_pendingAction = PendingAction::None;
    });
    auto& repository = ecvTestDataRepository::instance();
    connect(&repository, &ecvTestDataRepository::datasetReady, this,
            [this](bool ok, ecvTestDataRepository::Dataset kind) {
                if (!m_testDataBusy ||
                    kind != ecvTestDataRepository::Dataset::WeVisOne)
                    return;
                m_testDataBusy = false;
                m_testDataButton->setEnabled(true);
                if (!ok) {
                    m_statusLabel->setText(
                            tr("Test data preparation failed (see console "
                               "for details)"));
                    return;
                }
                fillDemoForCurrentTask();
            });
    // Surface the shared preparation's transport detail (endpoint rotation,
    // resume, percent) while this dialog owns a test-data request, so a slow
    // first download reads as progress rather than a dead button. Guarded by
    // m_testDataBusy because the repository is shared across plugins.
    connect(&repository, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_testDataBusy) m_statusLabel->setText(message);
            });
    connect(&repository, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                if (m_testDataBusy)
                    m_statusLabel->setText(tr("Preparing test images: %1% %2")
                                                   .arg(percent)
                                                   .arg(statusText));
            });

    updateRunState();
}

WeVisOneDialog::~WeVisOneDialog() {
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->wait(15000);
        m_worker->deleteLater();
        m_worker = nullptr;
    }
}

void WeVisOneDialog::populateCatalogCombos() {
    m_sizeCombo->clear();
    const QStringList families = WeVisOneHelpers::catalogFamilies();
    for (const QString& family : families) {
        m_sizeCombo->addItem(family, family);
    }
    // The catalog default row (2b q8_0 "(recommended)") selects both the
    // family and the precision so first Run is the recommended tier.
    const int def = WeVisOneHelpers::catalogDefaultIndex();
    if (def >= 0 && def < m_catalog.size()) {
        const int idx = m_sizeCombo->findData(m_catalog[def].family);
        if (idx >= 0) m_sizeCombo->setCurrentIndex(idx);
    }
    repopulatePrecisionCombo();
    if (def >= 0 && def < m_catalog.size() && m_catalog[def].role == kRoleVlm) {
        const int idx = m_precisionCombo->findData(m_catalog[def].filename);
        if (idx >= 0) m_precisionCombo->setCurrentIndex(idx);
    }
    m_deviceCombo->clear();
#ifdef AICore_ENABLED
    // Runtime backend registry projection — never a device the core cannot
    // resolve; item data carries the id passed to the C API (a translated
    // display text must never reach the runtime device request).
    ecvAICoreUi::populateDeviceCombo(m_deviceCombo);
#else
    m_deviceCombo->addItem(tr("auto"), QStringLiteral("auto"));
#endif
}

void WeVisOneDialog::repopulatePrecisionCombo() {
    const QString family = m_sizeCombo->currentData().toString();
    // Keep the quantization (not the filename) across a family switch.
    QString quant;
    WeVisOneModelEntry current;
    if (WeVisOneHelpers::findModelByFilename(
                m_precisionCombo->currentData().toString(), &current)) {
        quant = current.quantization;
    }
    m_precisionCombo->clear();
    m_decoderPrecisionCombo->clear();
    for (const WeVisOneModelEntry& entry : m_catalog) {
        if (entry.family != family) continue;
        if (entry.role == kRoleVlm)
            m_precisionCombo->addItem(entry.displayName, entry.filename);
        if (entry.role == kRoleDecoder)
            m_decoderPrecisionCombo->addItem(entry.displayName,
                                             entry.quantization);
    }
    if (!quant.isEmpty()) {
        for (int i = 0; i < m_precisionCombo->count(); ++i) {
            WeVisOneModelEntry row;
            if (WeVisOneHelpers::findModelByFilename(
                        m_precisionCombo->itemData(i).toString(), &row) &&
                row.quantization == quant) {
                m_precisionCombo->setCurrentIndex(i);
                break;
            }
        }
    }
    // The parity-gated decoder tier is f32 (the checkpoint dtype).
    const int f32Idx = m_decoderPrecisionCombo->findData(QStringLiteral("f32"));
    if (f32Idx >= 0) m_decoderPrecisionCombo->setCurrentIndex(f32Idx);
}

void WeVisOneDialog::setInputPath(const QString& path) {
    if (path.isEmpty()) return;
    m_inputEdit->setText(path);
}

bool WeVisOneDialog::exportToDbRequested() const {
    return m_exportDb && m_exportDb->isChecked();
}

void WeVisOneDialog::updateInputPreview() {
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

void WeVisOneDialog::updateRefBoxesVisibility() {
    if (!m_refBoxesRow) return;
    m_refBoxesRow->setVisible(m_taskCombo->currentText() ==
                              QStringLiteral("visual_prompt"));
}

void WeVisOneDialog::onBrowse() {
    // Start in the directory of the filled input (test-data auto-fill or a
    // previous pick) so browsing continues where the user already is.
    const QString path = ecvAICoreUi::browseInputImage(
            this, m_inputEdit->text(), tr("Select input image"));
    if (!path.isEmpty()) m_inputEdit->setText(path);
}

void WeVisOneDialog::onUseTestData() {
    if (m_testDataBusy) return;
    // Absent cache is a recoverable preparation, not an error: the shared
    // repository downloads, verifies and extracts once, then this slot
    // refills the input from the task-optimal demo member.
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    if (repo.isDatasetAvailable(ecvTestDataRepository::Dataset::WeVisOne) &&
        fillDemoForCurrentTask()) {
        return;
    }
    m_testDataBusy = true;
    m_testDataButton->setEnabled(false);
    m_statusLabel->setText(tr("Preparing official test images..."));
    repo.ensureDatasetReady(ecvTestDataRepository::Dataset::WeVisOne);
}

// Offline fallback demo table, used only when the extracted manifest.json
// is absent (e.g. a synthetic offline cache). The primary path reads the
// dataset's own manifest.json (applyManifestDemo) and prefers WeVisOne
// case-study originals; this table mirrors the manifest's first samples so
// an offline cache still yields a runnable per-task demo.
//
// | Task          | Image           | Query (manifest-verbatim)
// |---------------|-----------------|--------------------------------------
// | detection     | livingroom.jpg  | sofa\npillow\nlamp
// | referring     | dogs.jpg        | the dog closest to the camera
// | point         | parasol.jpg     | umbrella
// | count_box     | boys.jpg        | people
// | count_point   | chess.jpg       | white chess pieces
// | ocr           | coffee_sign.jpg | (fixed prompt)
// | ocr_grounding | coffee_sign.jpg | COFFEE
// | segmentation  | parasol.jpg     | parasol with alternating rings ...
// | matting       | seabreeze.jpg   | person
// | visual_prompt | seabreeze.jpg   | the person + box 880,120,1500,1070
// | freeform      | livingroom.jpg  | describe the image with bounding ...
QString WeVisOneDialog::demoImageForCurrentTask() const {
    const QString task = m_taskCombo->currentText();
    if (task == QStringLiteral("referring")) return QStringLiteral("dogs.jpg");
    if (task == QStringLiteral("point") ||
        task == QStringLiteral("segmentation"))
        return QStringLiteral("parasol.jpg");
    if (task == QStringLiteral("count_box")) return QStringLiteral("boys.jpg");
    if (task == QStringLiteral("count_point"))
        return QStringLiteral("chess.jpg");
    if (task == QStringLiteral("ocr") ||
        task == QStringLiteral("ocr_grounding"))
        return QStringLiteral("coffee_sign.jpg");
    if (task == QStringLiteral("matting") ||
        task == QStringLiteral("visual_prompt"))
        return QStringLiteral("seabreeze.jpg");
    // detection, freeform and fallback: the multi-object living room.
    return QStringLiteral("livingroom.jpg");
}

QString WeVisOneDialog::fallbackQueryForCurrentTask() const {
    const QString task = m_taskCombo->currentText();
    if (task == QStringLiteral("detection"))
        return QStringLiteral("sofa\npillow\nlamp");
    if (task == QStringLiteral("referring"))
        return QStringLiteral("the dog closest to the camera");
    if (task == QStringLiteral("point")) return QStringLiteral("umbrella");
    if (task == QStringLiteral("count_box")) return QStringLiteral("people");
    if (task == QStringLiteral("count_point"))
        return QStringLiteral("white chess pieces");
    if (task == QStringLiteral("ocr")) return QString();  // fixed OCR prompt
    if (task == QStringLiteral("ocr_grounding"))
        return QStringLiteral("COFFEE");
    if (task == QStringLiteral("segmentation"))
        return QStringLiteral(
                "parasol with alternating rings of light green "
                "and dark green with flowers");
    if (task == QStringLiteral("matting")) return QStringLiteral("person");
    if (task == QStringLiteral("visual_prompt"))
        return QStringLiteral("the person");
    return QStringLiteral("describe the image with bounding boxes in detail");
}

void WeVisOneDialog::applyDemoScene(const QString& demo,
                                    const QString& query,
                                    const QString& refBoxes) {
    if (demo.isEmpty()) return;
    setInputPath(demo);
    const QString task = m_taskCombo->currentText();
    // "Use test data" is an explicit user action: always populate the
    // task-specific query (never leave a previous task's text behind).
    if (task == QStringLiteral("ocr")) {
        m_queryEdit->clear();  // fixed OCR prompt
    } else {
        m_queryEdit->setPlainText(query);
    }
    // Reference boxes: only visual_prompt consumes them; every other demo
    // clears the field so a previous task's boxes never leak into the run.
    if (m_refBoxesEdit) {
        if (task == QStringLiteral("visual_prompt")) {
            m_refBoxesEdit->setText(refBoxes);
        } else {
            m_refBoxesEdit->clear();
        }
    }
    updateRefBoxesVisibility();
    m_statusLabel->setText(tr("Test data ready [%1]: %2")
                                   .arg(task, QFileInfo(demo).fileName()));
}

bool WeVisOneDialog::applyManifestDemo() {
    const QString manifest = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::WeVisOne,
            QStringLiteral("manifest.json"));
    if (manifest.isEmpty()) return false;
    WeVisOneHelpers::WeVisOneDemoSample sample;
    if (!WeVisOneHelpers::readDemoSample(manifest, m_taskCombo->currentText(),
                                         &sample))
        return false;
    const QString demo = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::WeVisOne, sample.image);
    if (demo.isEmpty()) return false;
    applyDemoScene(demo, sample.query, sample.refBoxes);
    return true;
}

bool WeVisOneDialog::applyFallbackDemo() {
    const QString demo = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::WeVisOne,
            demoImageForCurrentTask());
    if (demo.isEmpty()) return false;
    const QString task = m_taskCombo->currentText();
    applyDemoScene(demo, fallbackQueryForCurrentTask(),
                   task == QStringLiteral("visual_prompt")
                           ? QStringLiteral("880,120,1500,1070")
                           : QString());
    return true;
}

bool WeVisOneDialog::fillDemoForCurrentTask() {
    if (applyManifestDemo()) return true;
    return applyFallbackDemo();
}

void WeVisOneDialog::onDownloadProgress(qint64 received, qint64 total) {
    if (total > 0) {
        m_progress->setValue(int(received * 100 / total));
        m_statusLabel->setText(
                tr("Downloading %1...").arg(m_downloadTargetFilename));
    }
}

void WeVisOneDialog::onDownloadFinished(bool ok, const QString& path) {
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

void WeVisOneDialog::startDownload(const WeVisOneModelEntry& entry) {
    m_downloadInProgress = true;
    m_downloadTargetFilename = entry.filename;
    m_progress->setValue(0);
    ecvModelDownloader::Request request;
    request.url = entry.downloadUrl;
    request.destPath =
            QDir(WeVisOneHelpers::modelCacheDir()).filePath(entry.filename);
    request.contentAnchor = {QCryptographicHash::Sha256, entry.sha256.toUtf8()};
    m_downloader->download(request);
}

bool WeVisOneDialog::ensureModelsAvailable() {
    const QString precisionRow = m_precisionCombo->currentData().toString();
    WeVisOneModelEntry vlm;
    if (!WeVisOneHelpers::findModelByFilename(precisionRow, &vlm)) {
        // Hard failure: with no staged download the pending Run would never
        // re-enter, so clear the pending action and surface the reason.
        m_pendingAction = PendingAction::None;
        m_statusLabel->setText(
                tr("Catalog has no VLM row for the selected precision."));
        return false;
    }
    const QString decoderQuant =
            m_decoderPrecisionCombo->currentData().toString();
    WeVisOneModelEntry mmproj, decoder;
    for (const WeVisOneModelEntry& entry : m_catalog) {
        if (entry.family != vlm.family) continue;
        // The mmproj is always f16 (the only published projector tier).
        if (entry.role == kRoleMmproj) mmproj = entry;
        if (entry.role == kRoleDecoder && entry.quantization == decoderQuant)
            decoder = entry;
    }
    const QString cache = WeVisOneHelpers::modelCacheDir();
    struct Need {
        WeVisOneModelEntry entry;
        QString role;
    };
    const QVector<Need> needs = {{vlm, QStringLiteral("VLM")},
                                 {mmproj, QStringLiteral("mmproj")},
                                 {decoder, QStringLiteral("decoder")}};
    for (const Need& need : needs) {
        if (need.entry.filename.isEmpty()) {
            m_pendingAction = PendingAction::None;
            m_statusLabel->setText(
                    tr("Catalog has no %1 row for the selected family and "
                       "precision.")
                            .arg(need.role));
            return false;
        }
        const QString path = QDir(cache).filePath(need.entry.filename);
        if (ecvAssetIntegrity::isModelVerified(path)) continue;
        if (m_downloadInProgress) return false;  // one download at a time
        m_statusLabel->setText(tr("Downloading %1 (%2)...")
                                       .arg(need.entry.filename, need.role));
        startDownload(need.entry);
        return false;
    }
    m_settings.vlmPath = QDir(cache).filePath(vlm.filename);
    m_settings.mmprojPath = QDir(cache).filePath(mmproj.filename);
    m_settings.decoderPath = QDir(cache).filePath(decoder.filename);
    return true;
}

void WeVisOneDialog::onRun() {
    if (m_worker) return;  // one running frame at a time
    if (m_inputEdit->text().isEmpty() ||
        !QFileInfo::exists(m_inputEdit->text())) {
        QMessageBox::information(this, tr("WeVisOne-VL"),
                                 tr("Select an input image first."));
        return;
    }
    const QString task = m_taskCombo->currentText();
    const QString query = m_queryEdit->toPlainText().trimmed();
    // Mirror the prompt templates up front: ocr runs a fixed prompt and
    // visual_prompt grounds from the reference boxes; every other task
    // consumes the query (freeform passes it through verbatim).
    if (task != QStringLiteral("ocr") &&
        task != QStringLiteral("visual_prompt") && query.isEmpty()) {
        QMessageBox::information(
                this, tr("WeVisOne-VL"),
                tr("Task '%1' requires a query (categories, a referring "
                   "phrase or a freeform instruction).")
                        .arg(task));
        return;
    }
    QString refBoxesSpec;
    if (task == QStringLiteral("visual_prompt")) {
        refBoxesSpec = m_refBoxesEdit->text().trimmed();
        QString boxesError;
        if (WeVisOneHelpers::parseRefBoxes(refBoxesSpec, &boxesError)
                    .isEmpty()) {
            QMessageBox::information(
                    this, tr("WeVisOne-VL"),
                    boxesError.isEmpty()
                            ? tr("Task visual_prompt requires reference "
                                 "boxes (x1,y1,x2,y2 in image pixels, "
                                 "semicolon-separated for several).")
                            : boxesError);
            return;
        }
    }
    m_pendingAction = PendingAction::Run;
    if (!ensureModelsAvailable()) {
        updateRunState();
        return;  // download in flight; onDownloadFinished re-enters onRun
    }

    WeVisOneWorker::Settings settings;
    settings.vlmPath = m_settings.vlmPath;
    settings.mmprojPath = m_settings.mmprojPath;
    settings.decoderPath = m_settings.decoderPath;
    settings.size = m_sizeCombo->currentData().toString();
    {
        // Quantization of the selected catalog row (reported in model_info).
        WeVisOneModelEntry vlm;
        if (WeVisOneHelpers::findModelByFilename(
                    m_precisionCombo->currentData().toString(), &vlm))
            settings.precision = vlm.quantization;
    }
    settings.decoderPrecision =
            m_decoderPrecisionCombo->currentData().toString();
    settings.decodeMasks = m_decodeMasks->isChecked();
    settings.device = ecvAICoreUi::currentDeviceId(m_deviceCombo);
    settings.threads = m_threadsSpin->value();
    settings.maxNewTokens = m_tokensSpin->value();
    settings.inputPath = m_inputEdit->text();
    settings.task = task;
    settings.query = m_queryEdit->toPlainText();
    settings.refBoxes = refBoxesSpec;

    m_worker = new WeVisOneWorker(settings, this);
    m_runningTask = settings.task;
    connect(m_worker, &WeVisOneWorker::logMessage, this,
            [this](const QString& message) {
                m_statusLabel->setText(message);
            });
    connect(m_worker, &WeVisOneWorker::resultReady, this,
            &WeVisOneDialog::onWorkerResult);
    connect(m_worker, &WeVisOneWorker::taskFinished, this,
            &WeVisOneDialog::onWorkerFinished);
    m_progress->setRange(0, 0);  // indeterminate while inference runs
    m_statusLabel->setText(tr("Running task %1...").arg(settings.task));
    updateRunState();
    m_worker->start();
    emit runRequested(settings);
}

void WeVisOneDialog::onCancel() {
    m_testDataBusy = false;
    m_testDataButton->setEnabled(true);
    m_pendingAction = PendingAction::None;
    if (m_downloadInProgress) {
        m_downloader->cancel();
        return;
    }
    // The wevisone decode is a single greedy call per request; a cancel
    // takes effect after the current request settles (worker completion).
    m_statusLabel->setText(
            tr("Cancel requested; waiting for the current "
               "request to settle..."));
}

void WeVisOneDialog::onWorkerResult(const WeVisOneRunResult& result) {
    // The annotated image is visualized by the DB tree export (ccImage);
    // here only a concise headline is staged for the worker-settled status
    // line. The task mode prefixes the summary so the user sees which of the
    // eleven modes produced this result. The parsed per-label boxes are
    // written to the console log (qWeVisOne::onResultReady), not the UI — the
    // status line stays a short count headline, never the raw <ref>/<box>
    // token string.
    m_lastSummary =
            tr("[%1] %2 box(es)%3")
                    .arg(m_runningTask)
                    .arg(result.boxes.size())
                    .arg(result.hasMasks
                                 ? tr(", %1 matte(s)").arg(result.matteCount)
                                 : QString());
    // Counting tasks: the model's stated "Total number: N." is the count
    // (upstream demo/app.py semantics); the box list is localization
    // evidence and may be shorter than the stated total.
    const int stated = WeVisOneHelpers::statedTotalCount(result.answer);
    if ((m_runningTask == QStringLiteral("count_box") ||
         m_runningTask == QStringLiteral("count_point")) &&
        stated >= 0) {
        m_lastSummary =
                tr("[%1] count %2 (%3 box(es))%4")
                        .arg(m_runningTask)
                        .arg(stated)
                        .arg(result.boxes.size())
                        .arg(result.hasMasks ? tr(", %1 matte(s)")
                                                       .arg(result.matteCount)
                                             : QString());
    }
    emit resultReady(result);
}

void WeVisOneDialog::onWorkerFinished(bool success) {
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

void WeVisOneDialog::updateRunState() {
    const bool busy = m_worker != nullptr || m_downloadInProgress;
    m_runButton->setEnabled(!busy);
    // The testDataButton must track busy directly; reading its own isEnabled()
    // inside setEnabled creates a state-lock (once disabled by busy=true it
    // can never re-enable because `true && false == false`).
    m_testDataButton->setEnabled(!busy);
    m_cancelButton->setEnabled(busy);
}