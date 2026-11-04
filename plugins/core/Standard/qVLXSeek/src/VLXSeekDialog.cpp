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
#include <QCryptographicHash>
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
#include <QRegExp>
#include <QScrollArea>
#include <QSpinBox>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QVBoxLayout>

#include "ecvAICoreUiHelper.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

#ifdef AICore_ENABLED
#include <ecvAssetIntegrity.h>
#endif

namespace {

constexpr int kRoleWedetect = 0;
constexpr int kRoleVision = 1;
constexpr int kRoleLlm = 2;

// Strip VLX-Seek internal XML tags from the answer for human-readable
// display. The raw model output uses:
//   <ground>text</ground><objects><obj1><obj2></objects>  (detection)
//   <think>...</think><answer>text</answer>               (reasoning)
//   plain text                                             (vqa)
// This extracts the human-readable content while preserving the object
// references as [obj1, obj2] for context.
QString stripAnswerTags(const QString& answer) {
    if (answer.isEmpty()) return answer;
    QString result = answer;
    // Remove <think>...</think> block entirely (reasoning steps)
    result.remove(QRegExp(QStringLiteral("<think>.*</think>"),
                          Qt::CaseSensitive, QRegExp::RegExp2));
    // Extract text from <answer>...</answer> if present
    QRegExp answerRe(QStringLiteral("<answer>\\s*(.*)\\s*</answer>"),
                     Qt::CaseSensitive, QRegExp::RegExp2);
    if (answerRe.indexIn(result) >= 0) {
        result = answerRe.cap(1);
    }
    // Replace <ground>text</ground> with just the text
    result.remove(QRegExp(QStringLiteral("</?ground>")));
    // Replace <objects>...</objects> with [obj1, obj2, ...]
    QRegExp objectsRe(QStringLiteral("<objects>(.*)</objects>"),
                      Qt::CaseSensitive, QRegExp::RegExp2);
    if (objectsRe.indexIn(result) >= 0) {
        QString objs = objectsRe.cap(1);
        QStringList objList;
        QRegExp objRe(QStringLiteral("<obj(\\d+)>"), Qt::CaseSensitive,
                      QRegExp::RegExp2);
        int pos = 0;
        while ((pos = objRe.indexIn(objs, pos)) != -1) {
            objList << QStringLiteral("obj%1").arg(objRe.cap(1));
            pos += objRe.matchedLength();
        }
        if (objList.count() > 0) {
            result = result.replace(objectsRe.cap(0),
                                    QStringLiteral(" [%1]").arg(objList.join(
                                            QStringLiteral(", "))));
        } else {
            result.remove(objectsRe.cap(0));
        }
    }
    // Clean up any remaining tags
    result.remove(QRegExp(QStringLiteral("</?[a-z]+>")));
    return result.trimmed();
}

}  // namespace

VLXSeekDialog::VLXSeekDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("VLX-Seek Fine-Grained Grounding"));
    // Geometry conservation (qQuerySplat rule): the fixed top block plus a
    // minimum visible config-scroll height must fit the minimum window.
    resize(680, 780);
    setMinimumSize(620, 660);

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
               "shared data root and fills the demo image plus a showcase "
               "prompt"));

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
            tr("Zero-based indexes into the proposal boxes generated by "
               "WeDetect-Uni (obj0, obj1, ..., obj99). For example, '1, 2' "
               "selects obj1 and obj2 for the model to describe or OCR. "
               "Required for brief_region_caption and region_ocr tasks."));
    m_targetsRow = new QWidget(this);
    auto* targetsLayout = new QHBoxLayout(m_targetsRow);
    targetsLayout->setContentsMargins(0, 0, 0, 0);
    targetsLayout->addWidget(new QLabel(tr("Target indexes:"), this));
    targetsLayout->addWidget(m_targetsEdit, 1);
    m_targetsRow->setVisible(false);
    // Proposal source: the built-in WeDetect-Uni proposer (100 regions),
    // an external boxes JSON (upstream --bbox-list parity) for users who
    // already have candidate regions from another detector, or the OCR
    // text detector (region_ocr only) whose text boxes make the target
    // indexes address actual text lines.
    m_proposalSourceCombo = new QComboBox(this);
    m_proposalSourceCombo->addItem(tr("WeDetect-Uni (built-in)"),
                                   QStringLiteral("builtin"));
    m_proposalSourceCombo->addItem(tr("External boxes (JSON file)"),
                                   QStringLiteral("external"));
    m_proposalSourceCombo->addItem(tr("OCR text detector (PP-OCR)"),
                                   QStringLiteral("ocr_det"));
    m_boxesEdit = new QLineEdit(this);
    m_boxesEdit->setPlaceholderText(
            tr("JSON file with [[x1,y1,x2,y2],...] in image pixels"));
    m_boxesBrowseButton = ecvAICoreUi::makeBrowseBtn(tr("Browse..."), this);
    m_boxesRow = new QWidget(this);
    auto* boxesLayout = new QHBoxLayout(m_boxesRow);
    boxesLayout->setContentsMargins(0, 0, 0, 0);
    boxesLayout->addWidget(m_boxesEdit, 1);
    boxesLayout->addWidget(m_boxesBrowseButton);
    m_boxesRow->setVisible(false);
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
    form->addRow(tr("Proposal source:"), m_proposalSourceCombo);
    form->addRow(m_boxesRow);
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
    updateProposalSourceAvailability();

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
            [this](const QString& task) {
                updateTargetsRowVisibility();
                updateProposalSourceAvailability();
                // Update Text field placeholder: region tasks auto-generate
                // the prompt from the template, so the field should be empty.
                if (task == QStringLiteral("brief_region_caption")) {
                    m_textEdit->setPlaceholderText(
                            tr("(auto-generated from region template)"));
                } else if (task == QStringLiteral("region_ocr")) {
                    m_textEdit->setPlaceholderText(
                            tr("(auto-generated OCR prompt from region)"));
                } else {
                    m_textEdit->setPlaceholderText(
                            tr("Target description, e.g. person wearing red"));
                }
            });
    connect(m_proposalSourceCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) {
                m_boxesRow->setVisible(
                        m_proposalSourceCombo->currentData().toString() ==
                        QStringLiteral("external"));
            });
    connect(m_boxesBrowseButton, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
                this, tr("Select external boxes JSON"), QString(),
                tr("JSON (*.json);;All files (*)"));
        if (!path.isEmpty()) m_boxesEdit->setText(path);
    });

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

void VLXSeekDialog::updateProposalSourceAvailability() {
    if (!m_proposalSourceCombo) return;
    const int idx = m_proposalSourceCombo->findData(QStringLiteral("ocr_det"));
    if (idx < 0) return;
    // The OCR text detector composes region_ocr proposals only: the other
    // tasks ground their own regions (or ignore proposals entirely), and a
    // source left selected from a previous region_ocr run must not leak.
    const bool enabled =
            m_taskCombo->currentText() == QStringLiteral("region_ocr");
    auto* model =
            qobject_cast<QStandardItemModel*>(m_proposalSourceCombo->model());
    if (model) {
        if (QStandardItem* item = model->item(idx)) item->setEnabled(enabled);
    }
    if (!enabled && m_proposalSourceCombo->currentIndex() == idx)
        m_proposalSourceCombo->setCurrentIndex(0);
}

void VLXSeekDialog::onBrowse() {
    // Start in the directory of the filled input (test-data auto-fill or a
    // previous pick) so browsing continues where the user already is.
    const QString path = ecvAICoreUi::browseInputImage(
            this, m_inputEdit->text(), tr("Select input image"));
    if (!path.isEmpty()) m_inputEdit->setText(path);
}

void VLXSeekDialog::onUseTestData() {
    // Absent cache is a recoverable preparation, not an error: the shared
    // repository downloads, verifies and extracts once, then this slot
    // refills the input from the task-optimal demo member.
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    const QString demoFilename = demoImageForCurrentTask();
    if (repo.isDatasetAvailable(ecvTestDataRepository::Dataset::VlxSeek)) {
        const QString demo = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::VlxSeek, demoFilename);
        if (!demo.isEmpty()) {
            applyDemoScene(demo);
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
                        demoImageForCurrentTask());
                applyDemoScene(demo);
            },
            Qt::UniqueConnection);
    repo.ensureDatasetReady(ecvTestDataRepository::Dataset::VlxSeek);
}

// Each task mode has a different optimal demo image and prompt from the
// upstream VLX-Seek inference guide (github.com/om-ai-lab/VLX-Seek docs/
// inference.md). The mapping ensures "Use test data" produces a runnable,
// task-specific demo rather than the same image for every mode.
//
// | Task                  | Image                    | Prompt (upstream-exact)
// | Lang | Targets |
// |-----------------------|--------------------------|--------------------------------|------|----------|
// | detection             | ebike_riders.jpg         | people riding electric
// bikes   | en   |          | | grounding_single      | demo_image2.jpg | the
// person in a white shirt    | en   |          | | counting              |
// water_bottles_table.jpg  | water bottles                  | en   |          |
// | reasoning_detection   | demo_image2.jpg          | the woman wearing red
// clothes  | en   |          | | brief_region_caption  | kitchen_modern.jpg |
// (auto from region template)    | en   | 1        | | region_ocr            |
// demo_image2.jpg          | (auto from region template)    | en   | 0        |
// | vqa                   | ebike_riders.jpg         | What are people doing
// here?    | en   |          |
//
// detection uses ebike_riders.jpg (our richer multi-instance showcase)
// instead of the upstream demo_image.jpg fruits; the prompt stays in the
// same open-vocabulary plural form that exercises multi-box enumeration.
// grounding_single / reasoning_detection / region_ocr use demo_image2.jpg
// (upstream-exact; has visible text watermarks ideal for OCR demo).
// brief_region_caption uses kitchen_modern.jpg (indoor scene for captioning).
// region_ocr additionally switches the proposal source to the OCR text
// detector: WeDetect-Uni proposes object regions, while the OCR det model
// proposes text lines so the demo's target index addresses actual text and
// the exported boxes are text bboxes.
QString VLXSeekDialog::demoImageForCurrentTask() const {
    const QString task = m_taskCombo->currentText();
    if (task == QStringLiteral("grounding_single") ||
        task == QStringLiteral("reasoning_detection") ||
        task == QStringLiteral("region_ocr")) {
        return QStringLiteral("demo_image2.jpg");
    }
    if (task == QStringLiteral("counting")) {
        return QStringLiteral("water_bottles_table.jpg");
    }
    if (task == QStringLiteral("brief_region_caption")) {
        return QStringLiteral("kitchen_modern.jpg");
    }
    // detection, vqa, and fallback: the multi-instance street scene.
    return QStringLiteral("ebike_riders.jpg");
}

void VLXSeekDialog::applyDemoScene(const QString& demo) {
    if (demo.isEmpty()) return;
    setInputPath(demo);
    const QString task = m_taskCombo->currentText();
    // "Try sample data" is an explicit user action: always populate the
    // task-specific prompt (never leave a previous task's text behind —
    // e.g. switching detection → grounding_single must replace the text).
    // Each prompt matches the upstream inference guide example.
    // Region tasks clear Text (the runtime template fills <objN>
    // references from target indexes); instead they need target indexes.
    if (task == QStringLiteral("grounding_single")) {
        m_textEdit->setText(QStringLiteral("the person in a white shirt"));
    } else if (task == QStringLiteral("counting")) {
        m_textEdit->setText(QStringLiteral("water bottles"));
    } else if (task == QStringLiteral("reasoning_detection")) {
        m_textEdit->setText(QStringLiteral("the woman wearing red clothes"));
    } else if (task == QStringLiteral("vqa")) {
        m_textEdit->setText(
                QStringLiteral("What are the people doing in this image?"));
    } else if (task == QStringLiteral("detection")) {
        m_textEdit->setText(QStringLiteral("people riding electric bikes"));
    } else {
        // Region tasks (brief_region_caption, region_ocr): clear text so
        // the runtime template can fill <objN> references from targets.
        m_textEdit->clear();
    }
    // Lang: upstream default is "en" for every task; reset the combo to
    // match so the demo is a one-click reproducible upstream example.
    if (m_langCombo) {
        const int enIdx = m_langCombo->findText(QStringLiteral("en"));
        if (enIdx >= 0) m_langCombo->setCurrentIndex(enIdx);
    }
    // Proposal source: region_ocr demos run on OCR text-detector proposals
    // (text bboxes); every other demo resets to the built-in proposer so a
    // previous task's source never leaks into the next run.
    if (m_proposalSourceCombo) {
        const int ocrIdx =
                m_proposalSourceCombo->findData(QStringLiteral("ocr_det"));
        if (task == QStringLiteral("region_ocr") && ocrIdx >= 0) {
            m_proposalSourceCombo->setCurrentIndex(ocrIdx);
        } else {
            m_proposalSourceCombo->setCurrentIndex(0);
        }
    }
    // Region tasks need target indexes; prefill per the upstream example.
    // brief_region_caption upstream uses [1]; region_ocr uses [0].
    if (m_targetsEdit) {
        if (task == QStringLiteral("brief_region_caption")) {
            m_targetsEdit->setText(QStringLiteral("1"));
        } else if (task == QStringLiteral("region_ocr")) {
            m_targetsEdit->setText(QStringLiteral("0"));
        } else {
            m_targetsEdit->clear();
        }
    }
    // Ensure the targets row is visible for region tasks (the task combo
    // visibility signal may not have fired if the task was already selected).
    updateTargetsRowVisibility();
    m_statusLabel->setText(tr("Test data ready [%1]: %2")
                                   .arg(task, QFileInfo(demo).fileName()));
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

void VLXSeekDialog::startDownload(const QString& url,
                                  const QString& destPath,
                                  const QString& pinnedFilename) {
    m_downloadInProgress = true;
    m_downloadTargetFilename = pinnedFilename;
    m_progress->setValue(0);
    ecvModelDownloader::Request request;
    request.url = url;
    request.destPath = destPath;
    request.minBytes = 1024 * 1024;
    request.requireGgufMagic = true;
    // Content identity from the pinned digest registry (asset_digests.h);
    // the OCR catalog entry struct does not carry the digest itself.
    request.contentAnchor = {QCryptographicHash::Sha256,
                             ecvAssetIntegrity::PinnedDigest(pinnedFilename)};
    m_downloader->download(request);
}

bool VLXSeekDialog::ensureModelsAvailable() {
    const QString precisionRow = m_precisionCombo->currentData().toString();
    VLXSeekModelEntry vision;
    if (!VLXSeekHelpers::findModelByFilename(precisionRow, &vision)) {
        // Hard failure: with no staged download the pending Run would never
        // re-enter, so clear the pending action and surface the reason.
        m_pendingAction = PendingAction::None;
        m_statusLabel->setText(tr("Catalog has no vision row for precision %1.")
                                       .arg(precisionRow));
        return false;
    }
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
        if (need.entry.filename.isEmpty()) {
            m_pendingAction = PendingAction::None;
            m_statusLabel->setText(
                    tr("Catalog has no %1 row for the selected precision.")
                            .arg(need.role));
            return false;
        }
        const QString path = QDir(cache).filePath(need.entry.filename);
        if (QFileInfo::exists(path)) continue;
        if (m_downloadInProgress) return false;  // one download at a time
        m_statusLabel->setText(tr("Downloading %1 (%2)...")
                                       .arg(need.entry.filename, need.role));
        startDownload(need.entry);
        return false;
    }
    // Region-OCR text-detector proposals add the PP-OCR det/rec pair from
    // the public OCR catalog: same shared downloader, pinned digest anchor
    // and cache root as the qOCR plugin (no plugin-private model state).
    m_settings.ocrDetPath.clear();
    m_settings.ocrRecPath.clear();
    if (m_taskCombo->currentText() == QStringLiteral("region_ocr") &&
        m_proposalSourceCombo->currentData().toString() ==
                QStringLiteral("ocr_det")) {
        const VLXSeekHelpers::OcrProposerModels ocr =
                VLXSeekHelpers::ocrProposerModels();
        if (ocr.cacheDir.isEmpty() || ocr.detFileName.isEmpty() ||
            ocr.recFileName.isEmpty()) {
            m_pendingAction = PendingAction::None;
            m_statusLabel->setText(
                    tr("OCR text-detector proposer unavailable (public OCR "
                       "catalog missing)."));
            return false;
        }
        QDir(ocr.cacheDir).mkpath(QStringLiteral("."));
        struct OcrNeed {
            QString fileName;
        };
        const QVector<OcrNeed> ocrNeeds = {{ocr.detFileName},
                                           {ocr.recFileName}};
        for (const OcrNeed& need : ocrNeeds) {
            const QString path = QDir(ocr.cacheDir).filePath(need.fileName);
            const ecvAssetIntegrity::Anchor anchor{
                    QCryptographicHash::Sha256,
                    ecvAssetIntegrity::PinnedDigest(need.fileName)};
            if (ecvAssetIntegrity::isVerified(path, anchor, 1024 * 1024, true))
                continue;
            if (m_downloadInProgress) return false;  // one download at a time
            const QString url =
                    VLXSeekHelpers::ocrModelDownloadUrl(need.fileName);
            if (url.isEmpty()) {
                m_pendingAction = PendingAction::None;
                m_statusLabel->setText(
                        tr("%1 is not a published OCR catalog asset.")
                                .arg(need.fileName));
                return false;
            }
            m_statusLabel->setText(tr("Downloading %1 (OCR text detector)...")
                                           .arg(need.fileName));
            startDownload(url, path, need.fileName);
            return false;
        }
        m_settings.ocrDetPath = QDir(ocr.cacheDir).filePath(ocr.detFileName);
        m_settings.ocrRecPath = QDir(ocr.cacheDir).filePath(ocr.recFileName);
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
    settings.ocrDetPath = m_settings.ocrDetPath;
    settings.ocrRecPath = m_settings.ocrRecPath;
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
    // Proposal source: "ocr_det" is only selectable for region_ocr (the
    // combo item is disabled otherwise); the external-JSON branch below
    // keeps the upstream --bbox-list parity.
    settings.proposalSource = m_proposalSourceCombo->currentData().toString();
    // Region tasks (brief_region_caption / region_ocr) consume zero-based
    // proposal indexes; the AICore runtime hard-rejects them without targets,
    // so validate early on the GUI thread with a clear user message instead
    // of letting the worker fail silently after model load.
    const bool isRegionTask =
            settings.task == QStringLiteral("brief_region_caption") ||
            settings.task == QStringLiteral("region_ocr");
    if (isRegionTask && settings.targetIndexes.trimmed().isEmpty()) {
        QMessageBox::information(
                this, tr("VLX-Seek"),
                tr("Task '%1' requires target indexes (zero-based proposal "
                   "indexes, e.g. '0' or '0,1,2'). Fill in the Target "
                   "indexes field first.")
                        .arg(settings.task));
        return;
    }
    if (m_proposalSourceCombo->currentData().toString() ==
        QStringLiteral("external")) {
        settings.externalBoxesPath = m_boxesEdit->text().trimmed();
        if (settings.externalBoxesPath.isEmpty()) {
            QMessageBox::information(
                    this, tr("VLX-Seek"),
                    tr("Select an external boxes JSON file first."));
            return;
        }
    }

    m_worker = new VLXSeekWorker(settings, this);
    connect(m_worker, &VLXSeekWorker::logMessage, this,
            [this](const QString& message) {
                m_statusLabel->setText(message);
            });
    connect(m_worker, &VLXSeekWorker::resultReady, this,
            &VLXSeekDialog::onWorkerResult);
    connect(m_worker, &VLXSeekWorker::taskFinished, this,
            &VLXSeekDialog::onWorkerFinished);
    m_progress->setRange(0, 0);  // indeterminate while inference runs
    m_statusLabel->setText(tr("Running task %1...").arg(settings.task));
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

void VLXSeekDialog::onWorkerResult(const VLXSeekRunResult& result) {
    // The annotated image is visualized by the DB tree export (ccImage);
    // here only the headline is staged for the worker-settled status line.
    // The task mode prefixes the summary so the user sees which of the seven
    // modes produced this result (vqa / region_ocr / counting / ...).
    // The raw model output contains internal XML tags; stripAnswerTags
    // extracts the human-readable content for the status bar.
    const QString readableAnswer = stripAnswerTags(result.answer);
    m_lastSummary = tr("[%1] %2 box(es) — %3")
                            .arg(m_taskCombo->currentText())
                            .arg(result.boxes.size())
                            .arg(readableAnswer.left(300));
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
    // The testDataButton must track busy directly; reading its own isEnabled()
    // inside setEnabled creates a state-lock (once disabled by busy=true it
    // can never re-enable because `true && false == false`).
    m_testDataButton->setEnabled(!busy);
    m_cancelButton->setEnabled(busy);
}
