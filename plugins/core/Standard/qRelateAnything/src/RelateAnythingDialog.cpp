// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RelateAnythingDialog.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSaveFile>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QTabWidget>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <memory>
#include <utility>

#include "CVLog.h"
#include "RelateAnythingPresets.h"
#include "RelateAnythingVideoWidget.h"
#include "aicore/backend_capi.h"
#include "aicore/yolo_capi.h"
#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"
#include "ecvTestDataRepository.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

namespace {
// Open-vocabulary segmenter used by the upstream demo (boxes + masks).
constexpr const char* kDefaultDetector = "yoloe-26s-seg-pf-f16.gguf";
// One catalog row of any detector family offered in the combo.
struct DetectorInfo {
    QString filename, url, dir;
    bool rfdetr = false;
    // Open-vocabulary model that needs the RA-4M class list + a text encoder
    // (YOLO-World, and YOLOE checkpoints that are not prompt-free).
    bool vocabulary = false;
    bool world = false;
    bool valid() const { return !filename.isEmpty(); }
};
QString cacheDir(char* raw, void (*release)(void*)) {
    const QString dir = raw ? QString::fromUtf8(raw) : QString();
    if (raw) release(raw);
    return dir;
}
DetectorInfo detectorInfo(const QString& filename) {
    DetectorInfo info;
    if (filename.startsWith(QStringLiteral("rfdetr-"))) {
        for (int i = 0; i < aicore_rfdetr_model_count(); ++i) {
            const auto* e = aicore_rfdetr_model_at(i);
            if (e && filename == QString::fromUtf8(e->filename)) {
                info.filename = filename;
                info.url = QString::fromUtf8(e->download_url);
                info.dir = cacheDir(aicore_rfdetr_model_cache_dir(),
                                    aicore_rfdetr_free_buffer);
                info.rfdetr = true;
            }
        }
        return info;
    }
    for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY); ++i) {
        const auto* e = aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
        if (!e || filename != QString::fromUtf8(e->filename)) continue;
        info.filename = filename;
        info.url = QString::fromUtf8(e->download_url);
        info.dir = cacheDir(aicore_yolo_model_cache_dir(),
                            aicore_yolo_free_buffer);
        info.world = filename.contains(QStringLiteral("-world"));
        info.vocabulary =
                e->text_input &&
                QString::fromUtf8(e->task) != QStringLiteral("text") &&
                !filename.contains(QStringLiteral("-pf-"));
    }
    return info;
}
QString textModelFile(const DetectorInfo& d) {
    return d.world ? QStringLiteral("clip-ViT-B-32-f16.gguf")
                   : QStringLiteral("mobileclip2_b-f16.gguf");
}
void addDetectors(QComboBox* combo,
                  const QString& group,
                  const QStringList& filenames,
                  const QString& note = {}) {
    for (const auto& file : filenames) {
        const auto info = detectorInfo(file);
        if (!info.valid() || ecvAssetIntegrity::PinnedDigest(file).isEmpty() ||
            !(file.endsWith(QStringLiteral("-f16.gguf")) ||
              file.endsWith(QStringLiteral("-q8_0.gguf"))))
            continue;
        QString name;
        if (info.rfdetr) {
            for (int i = 0; i < aicore_rfdetr_model_count(); ++i)
                if (file ==
                    QString::fromUtf8(aicore_rfdetr_model_at(i)->filename))
                    name = QString::fromUtf8(
                            aicore_rfdetr_model_at(i)->display_name);
        } else {
            for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY);
                 ++i) {
                const auto* e = aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
                if (file == QString::fromUtf8(e->filename))
                    name = QString::fromUtf8(e->display_name) + " — " +
                           QString::fromUtf8(e->quant_note);
            }
        }
        name.remove(QStringLiteral(" (recommended)"));
        if (file == QLatin1String(kDefaultDetector))
            name += QStringLiteral(" (default)");
        combo->addItem(group + QStringLiteral(" · ") + name + note, file);
    }
}
QStringList catalogFiles(enum aicore_yolo_model_role role) {
    QStringList files;
    for (int i = 0; i < aicore_yolo_model_count(role); ++i)
        files << QString::fromUtf8(aicore_yolo_model_at(i, role)->filename);
    return files;
}
}  // namespace
RelateAnythingDialog::RelateAnythingDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("RelateAnything · Scene relations"));
    resize(1000, 700);
    ecvPS::registerSettingsGroup(QStringLiteral("qRelateAnything"));
    qRegisterMetaType<RelateAnythingResult>();
    m_worker = new RelateAnythingWorker;
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(m_worker, &RelateAnythingWorker::completed, this,
            &RelateAnythingDialog::completed);
    m_thread.start();
    auto* root = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(root);
    auto* top = new QHBoxLayout;
    m_model = new QComboBox;
    m_model->setObjectName("relateModel");
    m_model->setSizeAdjustPolicy(
            QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_model->setMinimumContentsLength(18);
    for (int i = 0; i < aicore_relation_model_count(); ++i) {
        auto* e = aicore_relation_model_at(i);
        if (e->role == AICORE_RELATION_MODEL_RELATIONS)
            m_model->addItem(QString::fromUtf8(e->display_name), i);
    }
    m_model->setCurrentIndex(
            m_model->findData(aicore_relation_model_default_index()));
    m_device = new QComboBox;
    m_device->setObjectName("relateDevice");
    m_device->setSizeAdjustPolicy(
            QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_device->setMinimumContentsLength(22);
    for (int i = 0; i < aicore_device_count(); ++i) {
        auto* d = aicore_device_at(i);
        const QString id = QString::fromUtf8(d->id);
        if (id != "auto" && id != "cpu" && !id.startsWith("cuda") &&
            !id.startsWith("vulkan"))
            continue;
        m_device->addItem(QString::fromUtf8(d->label), id);
    }
    m_download = new QPushButton(tr("Download model"));
    m_sample = ecvAICoreUi::makeSampleDataBtn(this);
    m_sample->setObjectName("relateUseTestData");
    top->addWidget(ecvAICoreUi::makeLabel(tr("Model")));
    top->addWidget(m_model, 2);
    top->addWidget(ecvAICoreUi::makeLabel(tr("Device")));
    top->addWidget(m_device, 1);
    top->addWidget(m_download);
    top->addWidget(m_sample);
    root->addLayout(top);
    auto* quickStart = new QLabel(tr(
            "Click Try sample data to fill image and video inputs, then Run. "
            "Objects are detected first; RelateAnything predicts "
            "subject–predicate–object relations. "
            "Leave custom predicates empty to use the model vocabulary."));
    quickStart->setObjectName("relateQuickStart");
    quickStart->setWordWrap(true);
    root->addWidget(quickStart);
    m_tabs = new QTabWidget;
    m_tabs->setObjectName("relateTabs");
    root->addWidget(m_tabs);
    auto* inputs = new QWidget;
    auto* il = new QVBoxLayout(inputs);
    ecvAICoreUi::setupTabLayout(il);
    auto* buttons = new QHBoxLayout;
    for (auto action : {tr("Add images…"), tr("Add folder…"),
                        tr("Selected DB images"), tr("Clear")}) {
        auto* b = new QPushButton(action);
        buttons->addWidget(b);
        connect(b, &QPushButton::clicked, this, [this, action]() {
            if (action == tr("Add images…"))
                addFiles();
            else if (action == tr("Add folder…")) {
                auto dir = QFileDialog::getExistingDirectory(
                        this, tr("Image folder"));
                for (const auto& f : QDir(dir).entryInfoList(
                             {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"},
                             QDir::Files, QDir::Name)) {
                    if (!dir.isEmpty()) addInput(f.absoluteFilePath());
                }
            } else if (action == tr("Clear")) {
                cancel();
                m_inputs->clear();
                resetResults();
            } else
                for (const auto& pair : m_db) {
                    addInput("db://" + pair.first, pair.second);
                }
        });
    }
    il->addLayout(buttons);
    auto* inputSplit = new QSplitter;
    m_inputs = new QListWidget;
    m_inputs->setObjectName("relateInputs");
    m_inputs->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_preview = new ecvClickableImageLabel;
    m_preview->setMinimumSize(300, 180);
    connect(m_inputs, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem* item) {
                if (!item) return;
                QImage image = qvariant_cast<QImage>(item->data(Qt::UserRole));
                if (image.isNull()) image.load(item->text());
                m_preview->setPreviewImage(image, QSize(480, 280));
            });
    inputSplit->addWidget(m_inputs);
    inputSplit->addWidget(m_preview);
    il->addWidget(inputSplit, 1);
    auto* selection = new QHBoxLayout;
    selection->addWidget(new QLabel(
            tr("Check the images to process; click an image to preview.")));
    selection->addStretch();
    for (bool checked : {true, false}) {
        auto* button =
                new QPushButton(checked ? tr("Check all") : tr("Uncheck all"));
        button->setObjectName(checked ? "relateCheckAll" : "relateUncheckAll");
        selection->addWidget(button);
        connect(button, &QPushButton::clicked, this, [this, checked]() {
            for (int i = 0; i < m_inputs->count(); ++i)
                m_inputs->item(i)->setCheckState(checked ? Qt::Checked
                                                         : Qt::Unchecked);
        });
    }
    il->addLayout(selection);
    m_tabs->addTab(inputs, tr("Images / batch"));
    m_video = new RelateAnythingVideoWidget;
    m_video->setEnabled(VideoPlaybackWidget::isAvailable());
    connect(m_video, &RelateAnythingVideoWidget::frameReady, this,
            [this](const QImage& image, int frame) {
                if (m_closed) {
                    m_video->acknowledge();
                    return;
                }
                const QString source =
                        tr("%1#frame=%2")
                                .arg(m_video->inputSource() ==
                                                     VideoPlaybackWidget::
                                                             InputSource::Camera
                                             ? "camera"
                                             : m_video->videoFilePath())
                                .arg(frame);
                if (m_busy) {
                    // A restarted stream can deliver its first frame while
                    // the previous generation is cancelling. Hold the shared
                    // consumer gate until that job completes.
                    m_heldFrame = image;
                    m_heldFrameSource = source;
                    return;
                }
                submit(image, source, true);
            });
    connect(m_video, &RelateAnythingVideoWidget::invalidated, this, [this]() {
        ++m_generation;
        m_framePending = false;
        m_heldFrame = {};
        m_heldFrameSource.clear();
        m_queue.clear();
        if (m_batch && m_progress) m_progress->hide();
        m_batch = false;
        // L2 fix: guard worker access after shutdown.
        if (m_worker) m_worker->requestCancel();
        // New video, seek, stop or source change: forget tracks and beliefs.
        // Queued, so it runs before any frame submitted afterwards.
        if (m_worker)
            QMetaObject::invokeMethod(
                    m_worker, [w = m_worker]() { w->resetTemporal(); },
                    Qt::QueuedConnection);
    });
    connect(m_video, &VideoPlaybackWidget::streamError, this,
            &RelateAnythingDialog::log);
    connect(m_video, &VideoPlaybackWidget::logMessage, this,
            &RelateAnythingDialog::log);
    auto* resultPage = new QWidget;
    auto* rl = new QVBoxLayout(resultPage);
    ecvAICoreUi::setupTabLayout(rl);
    m_summary = new QLabel;
    m_summary->setWordWrap(true);
    m_summary->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    rl->addWidget(m_summary);
    auto* filterRow = new QHBoxLayout;
    m_filter = new QLineEdit;
    m_filter->setPlaceholderText(tr("Filter relations…"));
    m_filter->setObjectName("relateFilter");
    filterRow->addWidget(m_filter, 1);
    rl->addLayout(filterRow);
    m_table = new QTableWidget(0, 4);
    m_table->setObjectName("relateRelations");
    m_table->setHorizontalHeaderLabels(
            {tr("Subject"), tr("Predicate"), tr("Object"), tr("Score")});
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_table->setMinimumHeight(220);
    rl->addWidget(m_table, 1);
    auto* exports = new QHBoxLayout;
    auto* exportButton = new QPushButton(tr("Export JSON / CSV / text…"));
    auto* dbButton = new QPushButton(tr("Add result image to DB"));
    exportButton->setObjectName("relateExport");
    dbButton->setObjectName("relateAddToDb");
    exports->addWidget(exportButton);
    exports->addWidget(dbButton);
    exports->addStretch();
    rl->addLayout(exports);
    connect(exportButton, &QPushButton::clicked, this,
            &RelateAnythingDialog::exportResults);
    connect(dbButton, &QPushButton::clicked, this, [this]() {
        if (!m_last.image.isNull()) emit addToDbRequested(m_last);
    });
    connect(m_filter, &QLineEdit::textChanged, this,
            &RelateAnythingDialog::showResult);
    m_tabs->addTab(resultPage, tr("Results"));
    auto* options = new QWidget;
    auto* optionsLayout = new QVBoxLayout(options);
    optionsLayout->setContentsMargins(0, 0, 0, 0);
    auto* optionsTop = new QFormLayout;
    optionsLayout->addLayout(optionsTop);
    // Collapsible group: the common settings stay open, the rarely touched
    // ones (video filter noise, file inputs) are folded away.
    const auto section = [&](const QString& title, bool expanded,
                             const char* name) -> QFormLayout* {
        auto* box = new QGroupBox(title);
        box->setObjectName(name);
        box->setCheckable(true);
        box->setChecked(expanded);
        auto* boxLayout = new QVBoxLayout(box);
        boxLayout->setContentsMargins(8, 2, 8, 6);
        auto* content = new QWidget;
        auto* form = new QFormLayout(content);
        form->setContentsMargins(0, 0, 0, 0);
        boxLayout->addWidget(content);
        content->setVisible(expanded);
        connect(box, &QGroupBox::toggled, content, &QWidget::setVisible);
        optionsLayout->addWidget(box);
        return form;
    };
    QFormLayout* ol = optionsTop;
    m_objectSource = new QComboBox;
    m_objectSource->setObjectName("relateObjectSource");
    m_objectSource->addItems({tr("Detect objects automatically"),
                              tr("Use supplied object boxes")});
    ol->addRow(tr("Object source"), m_objectSource);
    auto fileInput = [&](QLineEdit*& edit, const char* name,
                         const QString& label, const QString& hint) {
        auto* row = new QWidget;
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        edit = new QLineEdit;
        edit->setObjectName(name);
        edit->setPlaceholderText(hint);
        edit->setToolTip(hint);
        auto* browse = new QPushButton(tr("Browse…"));
        layout->addWidget(edit, 1);
        layout->addWidget(browse);
        connect(browse, &QPushButton::clicked, this, [this, edit, label]() {
            const auto file = QFileDialog::getOpenFileName(this, label);
            if (!file.isEmpty()) edit->setText(file);
        });
        ol->addRow(label, row);
    };
    m_threads = new QSpinBox;
    m_threads->setObjectName("relateThreads");
    m_threads->setRange(1, 256);
    m_threads->setValue(8);
    m_detector = new QComboBox;
    m_detector->setObjectName("relateDetector");
    // RelateAnything consumes object regions, so any detector or segmenter
    // works. Segmenters also draw masks. Only rows with a pinned digest in
    // F16 / Q8_0 are offered; groups follow upstream's three input modes.
    QStringList promptFree, yoloeVocab, world, segment, detect, rfdetr;
    for (const auto& f : catalogFiles(AICORE_YOLO_ROLE_YOLOE))
        (f.contains(QStringLiteral("-pf-")) ? promptFree : yoloeVocab) << f;
    for (const auto& f : catalogFiles(AICORE_YOLO_ROLE_WORLD)) world << f;
    segment = catalogFiles(AICORE_YOLO_ROLE_SEGMENT);
    detect = catalogFiles(AICORE_YOLO_ROLE_DETECTION);
    for (int i = 0; i < aicore_rfdetr_model_count(); ++i)
        rfdetr << QString::fromUtf8(aicore_rfdetr_model_at(i)->filename);
    const QString ra4m = tr(" · RA-4M 497 classes");
    addDetectors(m_detector, tr("YOLOE prompt-free (masks)"), promptFree);
    addDetectors(m_detector, tr("YOLOE (masks)"), yoloeVocab, ra4m);
    addDetectors(m_detector, tr("YOLO-World"), world, ra4m);
    addDetectors(m_detector, tr("YOLO segment"), segment);
    addDetectors(m_detector, tr("RF-DETR"), rfdetr);
    addDetectors(m_detector, tr("YOLO detect (boxes)"), detect);
    m_detector->setCurrentIndex(qMax(
            0, m_detector->findData(QString::fromLatin1(kDefaultDetector))));
    m_detector->setToolTip(
            tr("Proposes the objects whose relations are predicted. "
               "Segmentation models (recommended) also give object masks, "
               "as in the official demo."));
    m_threshold = new QDoubleSpinBox;
    m_threshold->setRange(.01, 1);
    m_threshold->setSingleStep(.05);
    m_threshold->setValue(.20);
    m_threshold->setObjectName("relateDetectionThreshold");
    m_maxBoxes = new QSpinBox;
    m_maxBoxes->setRange(2, 60);
    m_maxBoxes->setValue(24);
    m_relThreshold = new QDoubleSpinBox;
    m_relThreshold->setRange(0, 1);
    m_relThreshold->setSingleStep(.05);
    m_relThreshold->setValue(.45);
    m_relThreshold->setObjectName("relateRelationThreshold");
    m_relThreshold->setToolTip(
            tr("Minimum calibrated score for interaction relations (riding, "
               "wearing, holding...). Upstream video default 0.45."));
    m_spatialThreshold = new QDoubleSpinBox;
    m_spatialThreshold->setRange(0, 1);
    m_spatialThreshold->setSingleStep(.05);
    m_spatialThreshold->setValue(.42);
    m_spatialThreshold->setObjectName("relateSpatialThreshold");
    m_spatialThreshold->setToolTip(
            tr("Minimum calibrated score for spatial relations (beside, on "
               "top of...); at most three are shown."));
    m_maxRelations = new QSpinBox;
    m_maxRelations->setRange(1, 128);
    m_maxRelations->setValue(8);
    m_maxRelations->setObjectName("relateMaxRelations");
    m_decompose = new QCheckBox(tr("Separate spatial and semantic relations"));
    m_decompose->setObjectName("relateDecompose");
    m_decompose->setChecked(true);
    m_detIou = new QDoubleSpinBox;
    m_detIou->setRange(.05, 1);
    m_detIou->setSingleStep(.05);
    m_detIou->setValue(.5);
    m_detIou->setObjectName("relateDetectionIou");
    m_detIou->setToolTip(
            tr("NMS IoU of the YOLO detectors (upstream 0.5). "
               "RF-DETR is NMS-free and ignores it."));
    m_maxBoxFraction = new QDoubleSpinBox;
    m_maxBoxFraction->setRange(.1, 1);
    m_maxBoxFraction->setSingleStep(.05);
    m_maxBoxFraction->setValue(.85);
    m_maxBoxFraction->setObjectName("relateMaxBoxFraction");
    m_maxBoxFraction->setToolTip(
            tr("Drop detections covering more than this fraction of the "
               "image (upstream 0.85). 1.0 keeps everything."));
    m_maxSpatial = new QSpinBox;
    m_maxSpatial->setRange(0, 128);
    m_maxSpatial->setValue(3);
    m_maxSpatial->setObjectName("relateMaxSpatial");
    m_maxSpatial->setToolTip(
            tr("How many of the shown relations may be "
               "spatial (beside, behind...). 0 = "
               "interactions only."));
    m_profile = new QComboBox;
    m_profile->setObjectName("relateProfile");
    m_profile->addItem(tr("Official video reel (default)"),
                       int(RelatePresets::Profile::VideoReel));
    m_profile->addItem(tr("Official image demo"),
                       int(RelatePresets::Profile::ImageDemo));
    m_profile->addItem(tr("Custom"), int(RelatePresets::Profile::Custom));
    m_profile->setToolTip(
            tr("Operating points taken from upstream: the video reel "
               "(render_video.py) separates spatial and interaction "
               "relations with floors 0.45 / 0.42; the image demo "
               "(demo_webcam.py) uses one graph with floor 0.5. Editing any "
               "threshold switches this to Custom."));
    m_predicatePreset = new QComboBox;
    m_predicatePreset->setObjectName("relatePredicatePreset");
    m_predicatePreset->addItem(tr("Official video vocabulary (65)"), 0);
    m_predicatePreset->addItem(tr("Official image demo vocabulary (35)"), 1);
    m_predicatePreset->addItem(tr("Full model vocabulary"), 2);
    m_predicatePreset->addItem(tr("Custom text below"), 3);
    m_predicatePreset->setToolTip(
            tr("Which relations the model may name. The official demos use "
               "curated lists; the full model vocabulary also contains "
               "unusual phrases such as \"has holding\"."));
    m_boxScoreWeight =
            new QCheckBox(tr("Rank relations by detector confidence"));
    m_boxScoreWeight->setObjectName("relateBoxScoreWeight");
    m_boxScoreWeight->setChecked(true);
    m_boxScoreWeight->setToolTip(
            tr("Order candidate relations by score x confidence(subject) x "
               "confidence(object), upstream's box_score_weight (default on). "
               "Ranking only: shown scores and floors are unchanged."));
    m_arithmetic = new QComboBox;
    m_arithmetic->addItem(tr("Automatic (recommended)"), -1);
    m_arithmetic->addItem(tr("Precise"), 1);
    m_arithmetic->addItem(tr("Optimized"), 0);
    m_arithmetic->setToolTip(
            tr("Automatic follows the model's recommended arithmetic: precise "
               "for CUDA F32 models, optimized for other models. "
               "Choose Precise or Optimized to override this policy."));
    m_arithmetic->setObjectName("relateArithmetic");
    m_predicates = new QLineEdit;
    m_predicates->setObjectName("relatePredicates");
    m_predicates->setPlaceholderText(
            tr("Optional predicates separated by |, for example riding | "
               "holding | next to"));
    ol->addRow(tr("Operating point"), m_profile);
    ol = section(tr("Objects"), true, "relateSectionObjects");
    ol->addRow(tr("Object detector"), m_detector);
    ol->addRow(tr("Detection confidence"), m_threshold);
    ol->addRow(tr("Detection NMS IoU"), m_detIou);
    ol->addRow(tr("Maximum objects"), m_maxBoxes);
    ol->addRow(tr("Largest object (fraction of image)"), m_maxBoxFraction);
    fileInput(m_regions, "relateRegions", tr("Object boxes"),
              tr("JSON list of pixel boxes, or an object with boxes, labels, "
                 "scores and image_size"));
    fileInput(m_masks, "relateMasks", tr("Object masks (optional)"),
              tr("Binary mask file, one full image mask per object; only for "
                 "supplied boxes"));
    ol = section(tr("Relations"), true, "relateSectionRelations");
    ol->addRow(tr("Relation score threshold"), m_relThreshold);
    ol->addRow(tr("Spatial relation threshold"), m_spatialThreshold);
    ol->addRow(tr("Maximum relations shown"), m_maxRelations);
    ol->addRow(tr("Maximum spatial relations"), m_maxSpatial);
    ol->addRow(m_decompose);
    ol->addRow(m_boxScoreWeight);
    ol->addRow(tr("Relation vocabulary"), m_predicatePreset);
    ol->addRow(tr("Custom predicates"), m_predicates);
    ol = section(tr("Video tracking and smoothing"), false,
                 "relateSectionTracking");
    buildTrackingUi(ol);
    ol = section(tr("Advanced"), false, "relateSectionAdvanced");
    ol->addRow(tr("CPU threads"), m_threads);
    ol->addRow(tr("Model arithmetic"), m_arithmetic);
    // A profile fills every threshold; touching one flips it to Custom.
    for (auto* spin : std::initializer_list<QDoubleSpinBox*>{
                 m_threshold, m_detIou, m_maxBoxFraction, m_relThreshold,
                 m_spatialThreshold})
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
                [this]() { markCustomProfile(); });
    for (auto* spin : {m_maxBoxes, m_maxRelations, m_maxSpatial})
        connect(spin, qOverload<int>(&QSpinBox::valueChanged), this,
                [this]() { markCustomProfile(); });
    connect(m_decompose, &QCheckBox::toggled, this,
            [this]() { markCustomProfile(); });
    connect(m_profile, qOverload<int>(&QComboBox::activated), this,
            [this](int) { applyProfile(m_profile->currentData().toInt()); });
    // Presets write the predicate text; typing flips the preset to Custom.
    connect(m_predicatePreset, qOverload<int>(&QComboBox::activated), this,
            [this](int) {
                const int p = m_predicatePreset->currentData().toInt();
                m_applyingPreset = true;
                if (p == 0)
                    m_predicates->setText(RelatePresets::join(
                            RelatePresets::videoPredicates()));
                else if (p == 1)
                    m_predicates->setText(RelatePresets::join(
                            RelatePresets::webcamPredicates()));
                else if (p == 2)
                    m_predicates->clear();
                m_applyingPreset = false;
            });
    connect(m_predicates, &QLineEdit::textEdited, this, [this]() {
        const int custom = m_predicatePreset->findData(3);
        if (!m_applyingPreset && custom >= 0)
            m_predicatePreset->setCurrentIndex(custom);
    });
    m_applyingPreset = true;
    m_predicates->setText(
            RelatePresets::join(RelatePresets::videoPredicates()));
    m_threshold->setValue(
            defaultConfidence(m_detector->currentData().toString(),
                              int(RelatePresets::Profile::VideoReel)));
    m_applyingPreset = false;
    // Each detector family has its own official confidence (make_reel.py
    // MODES): follow it unless the user has chosen their own values.
    connect(m_detector, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this]() {
                const int profile = m_profile->currentData().toInt();
                if (m_applyingPreset ||
                    profile == int(RelatePresets::Profile::Custom))
                    return;
                m_applyingPreset = true;
                m_threshold->setValue(defaultConfidence(
                        m_detector->currentData().toString(), profile));
                m_applyingPreset = false;
            });
    fileInput(m_vocabulary, "relateVocabulary",
              tr("Vocabulary vectors (optional)"),
              tr("Use an existing vocabulary vector file instead of custom "
                 "predicate text"));
    fileInput(m_vocabularyNames, "relateVocabularyNames",
              tr("Vocabulary names"),
              tr("JSON list of predicate names in the same order as the "
                 "vectors"));
    fileInput(
            m_predicateTypes, "relatePredicateTypes",
            tr("Predicate types (optional)"),
            tr("JSON predicate type map for spatial and semantic separation"));
    connect(m_objectSource, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int index) {
                m_detector->setEnabled(index == 0);
                m_threshold->setEnabled(index == 0);
                m_regions->setEnabled(index == 1);
                m_masks->setEnabled(index == 1);
            });
    m_regions->setEnabled(false);
    m_masks->setEnabled(false);
    connect(m_vocabulary, &QLineEdit::textChanged, this, [this]() {
        m_predicates->setEnabled(m_vocabulary->text().trimmed().isEmpty());
    });
    auto* optionsScroll = new QScrollArea;
    optionsScroll->setWidgetResizable(true);
    optionsScroll->setFrameShape(QFrame::NoFrame);
    optionsScroll->setWidget(options);
    m_tabs->addTab(optionsScroll, tr("Options"));
    auto* videoPage = new QWidget;
    auto* videoLayout = new QVBoxLayout(videoPage);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    auto* videoGuide =
            new QLabel(tr("Click <i>Try sample data</i> to fill the "
                          "recommended video and the image list, or choose a "
                          "video file or camera, then Run. Each frame advances "
                          "when inference completes."));
    videoGuide->setWordWrap(true);
    videoLayout->addWidget(videoGuide);
    videoLayout->addWidget(m_video, 1);
    m_tabs->addTab(videoPage, tr("Video / camera"));
    connect(m_model, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this]() { cancel(); });
    connect(m_device, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this]() { cancel(); });
    connect(m_detector, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this]() { cancel(); });
    m_timing = new QLabel;
    m_timing->setWordWrap(true);
    root->addWidget(m_timing);
    m_progress = new QProgressBar;
    m_progress->hide();
    root->addWidget(m_progress);
    auto* controls = new QHBoxLayout;
    m_run = new QPushButton(tr("Run"));
    m_run->setObjectName("relateRun");
    m_cancelButton = new QPushButton(tr("Cancel / stop"));
    auto* close = new QPushButton(tr("Close"));
    m_cancelButton->setObjectName("relateCancel");
    controls->addStretch();
    controls->addWidget(m_run);
    controls->addWidget(m_cancelButton);
    controls->addWidget(close);
    root->addLayout(controls);
    connect(m_run, &QPushButton::clicked, this, &RelateAnythingDialog::run);
    connect(m_cancelButton, &QPushButton::clicked, this,
            &RelateAnythingDialog::cancel);
    connect(close, &QPushButton::clicked, this, &RelateAnythingDialog::close);
    connect(m_sample, &QPushButton::clicked, this,
            &RelateAnythingDialog::useTestData);
    connect(m_download, &QPushButton::clicked, this,
            [this]() { ensureModel(false); });
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &RelateAnythingDialog::log);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 n, qint64 total) {
                m_progress->setRange(0, total > 0 ? 100 : 0);
                if (total > 0) m_progress->setValue(int(100 * n / total));
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString&) {
                m_progress->hide();
                // H2 fix: re-enable all controls that were disabled on start,
                // including device and detector.
                m_download->setEnabled(true);
                m_model->setEnabled(true);
                m_device->setEnabled(true);
                const bool useSupplied = m_objectSource->currentIndex() == 1;
                m_detector->setEnabled(!useSupplied);
                m_run->setEnabled(!m_busy);
                bool again = m_runAfterDownload;
                m_runAfterDownload = false;
                if (ok && ensureModel(again) && again) run();
            });
    auto& repository = ecvTestDataRepository::instance();
    connect(&repository, &ecvTestDataRepository::downloadFinished, this,
            [this](bool ok, ecvTestDataRepository::Dataset kind) {
                if (kind != ecvTestDataRepository::Dataset::RelateAnything ||
                    !m_waitingForSample)
                    return;
                m_waitingForSample = false;
                m_sample->setEnabled(true);
                if (ok)
                    useTestData();
                else
                    log(tr("Sample data download failed."));
            });
    connect(&repository, &ecvTestDataRepository::downloadLogMessage, this,
            &RelateAnythingDialog::log);
    QSettings s;
    s.beginGroup("qRelateAnything");
    m_model->setCurrentIndex(qBound(
            0,
            s.value("model",
                    m_model->findData(aicore_relation_model_default_index()))
                    .toInt(),
            m_model->count() - 1));
    int index = m_device->findData(s.value("device", "auto"));
    if (index >= 0) m_device->setCurrentIndex(index);
    m_threads->setValue(s.value("threads", 8).toInt());
    // Older schemas stored a closed-set detector index and different relation
    // defaults (floor 0.5, ten shown, decomposition off); drop them so the
    // upstream defaults apply.
    // Schema 4 adds the operating point, NMS IoU, spatial cap, largest-object
    // filter and the relation vocabulary preset; older files keep upstream
    // defaults.
    const bool current = s.value("schema", 1).toInt() >= 4;
    if (current) {
        m_applyingPreset = true;
        m_threshold->setValue(s.value("confidence", .20).toDouble());
        m_detIou->setValue(s.value("detectionIou", .5).toDouble());
        m_maxBoxFraction->setValue(s.value("maxBoxFraction", .85).toDouble());
        m_maxBoxes->setValue(s.value("maxBoxes", 24).toInt());
        m_relThreshold->setValue(s.value("relationThreshold", .45).toDouble());
        m_spatialThreshold->setValue(
                s.value("spatialThreshold", .42).toDouble());
        m_maxRelations->setValue(s.value("maxRelations", 8).toInt());
        m_maxSpatial->setValue(s.value("maxSpatial", 3).toInt());
        m_applyingPreset = false;
        const int profile = m_profile->findData(s.value("profile", 0));
        if (profile >= 0) m_profile->setCurrentIndex(profile);
        const int preset =
                m_predicatePreset->findData(s.value("predicatePreset", 0));
        if (preset >= 0) m_predicatePreset->setCurrentIndex(preset);
        m_applyingPreset = true;
        m_predicates->setText(s.value("predicates").toString());
        m_applyingPreset = false;
    }
    m_regions->setText(s.value("regions").toString());
    m_masks->setText(s.value("masks").toString());
    m_vocabulary->setText(s.value("vocabulary").toString());
    m_vocabularyNames->setText(s.value("vocabularyNames").toString());
    m_predicateTypes->setText(s.value("predicateTypes").toString());
    m_objectSource->setCurrentIndex(
            s.value("objectSource", 0).toInt() == 1 ? 1 : 0);
    const int arithmetic = s.value("arithmetic", -1).toInt();
    m_arithmetic->setCurrentIndex(qMax(0, m_arithmetic->findData(arithmetic)));
    // M3 fix: restore decompose inside m_applyingPreset guard so it doesn't
    // flip the profile combo to Custom when the saved value differs from
    // the constructed default (true) while restoring a preset.
    if (current) {
        m_applyingPreset = true;
        m_decompose->setChecked(s.value("decompose", true).toBool());
        m_applyingPreset = false;
    }
    // Schema 5 adds the video tracking controls; older files keep upstream
    // defaults (the widgets were constructed with them).
    if (s.value("schema", 1).toInt() >= 5) {
        m_applyingPreset = true;
        for (auto& w : m_trackD)
            w.second->setValue(
                    s.value(w.second->objectName(), w.second->value())
                            .toDouble());
        for (auto& w : m_trackI)
            w.second->setValue(
                    s.value(w.second->objectName(), w.second->value()).toInt());
        m_belief->setChecked(s.value("relateBelief", true).toBool());
        m_stabilize->setChecked(s.value("relateStabilize", true).toBool());
        m_boxScoreWeight->setChecked(s.value("boxScoreWeight", true).toBool());
        m_applyingPreset = false;
    }
    if (current) {
        const int detector =
                m_detector->findData(s.value("detectorModel").toString());
        if (detector >= 0) {
            m_applyingPreset = true;  // keep the stored confidence
            m_detector->setCurrentIndex(detector);
            m_applyingPreset = false;
        }
    }
}
// Video tracking and smoothing controls: one row per upstream render_video.py
// flag. Relation floors and edge caps are shared with the image path and live
// in the widgets above, so they are not repeated here.
void RelateAnythingDialog::buildTrackingUi(QFormLayout* form) {
    using P = RelateTemporal::Params;
    m_stabilize =
            new QCheckBox(tr("Stabilise video (track objects, smooth "
                             "relations over time)"));
    m_stabilize->setObjectName("relateStabilize");
    m_stabilize->setChecked(true);
    m_stabilize->setToolTip(
            tr("Video and camera frames keep object identities and relation "
               "beliefs between frames, as upstream render_video.py does. "
               "Single images and batches are never smoothed."));
    form->addRow(m_stabilize);
    m_belief = new QCheckBox(tr("Kalman relation belief (off = EMA)"));
    m_belief->setObjectName("relateBelief");
    m_belief->setChecked(true);
    m_belief->setToolTip(
            tr("Upstream default. Off falls back to the older "
               "EMA + hysteresis scoring (--no_belief)."));
    form->addRow(m_belief);
    const auto dbl = [&](const QString& label, const QString& key,
                         float P::*field, double lo, double hi, double step,
                         int decimals, const QString& tip) {
        auto* w = new QDoubleSpinBox;
        w->setObjectName(key);
        w->setRange(lo, hi);
        w->setSingleStep(step);
        w->setDecimals(decimals);
        w->setValue(P().*field);
        w->setToolTip(tip);
        form->addRow(label, w);
        m_trackD.emplace_back(field, w);
    };
    const auto integer = [&](const QString& label, const QString& key,
                             int P::*field, int lo, int hi,
                             const QString& tip) {
        auto* w = new QSpinBox;
        w->setObjectName(key);
        w->setRange(lo, hi);
        w->setValue(P().*field);
        w->setToolTip(tip);
        form->addRow(label, w);
        m_trackI.emplace_back(field, w);
    };
    dbl(tr("Track confidence (--track_high)"), "relateTrackHigh", &P::trackHigh,
        0, 1, .05, 2,
        tr("Detections at or above this start new tracks; lower ones only "
           "extend existing tracks (ByteTrack's second stage)."));
    dbl(tr("Track IoU gate (--iou_thr)"), "relateIouThr", &P::iouThr, 0, 1, .05,
        2, tr("Minimum IoU to match a confident detection to a track."));
    dbl(tr("Duplicate-region IoU (--merge_iou)"), "relateMergeIou",
        &P::mergeIou, 0, 1, .05, 2,
        tr("Regions overlapping more than this are one object under two "
           "names (man / person)."));
    integer(tr("Track max age (--max_age)"), "relateMaxAge", &P::maxAge, 0, 300,
            tr("Frames a track survives without a detection."));
    integer(tr("Track min hits (--min_hits)"), "relateMinHits", &P::minHits, 1,
            30, tr("Detections needed before a track is drawn."));
    dbl(tr("Box process noise (--kf_q)"), "relateKfQ", &P::kfQ, 0, 100, .1, 2,
        tr("How fast an object really moves. Small = steadier boxes."));
    dbl(tr("Box measurement noise (--kf_r)"), "relateKfR", &P::kfR, 0, 1000, 1,
        1, tr("How much the detector's box wobbles. Large = smoother."));
    dbl(tr("Relation process noise (--kf_rel_q)"), "relateKfRelQ", &P::relQ, 0,
        10, .01, 3, tr("How fast a relation may genuinely change."));
    dbl(tr("Relation measurement noise (--kf_rel_r)"), "relateKfRelR", &P::relR,
        .01, 100, .1, 2, tr("How much one frame's opinion is worth."));
    dbl(tr("Relation EMA (--score_ema)"), "relateScoreEma", &P::scoreEma, 0, 1,
        .05, 2, tr("Smoothing factor when the Kalman belief is off."));
    dbl(tr("Relation off threshold (--off_thr)"), "relateOffThr", &P::offThr, 0,
        1, .02, 2,
        tr("A shown relation falls under this to start fading out; capped at "
           "the interaction floor - 0.10."));
    integer(tr("Frames to show a relation (--on_frames)"), "relateOnFrames",
            &P::onFrames, 1, 60,
            tr("Observations above the floor before a relation appears."));
    integer(tr("Frames to hide a relation (--off_frames)"), "relateOffFrames",
            &P::offFrames, 1, 120,
            tr("Observations below the off threshold before it leaves."));
    integer(tr("Predicates per pair (--max_per_pair)"), "relateMaxPerPair",
            &P::maxPerPair, 1, 4,
            tr("1 keeps only each pair's best predicate; 2 (upstream) lets "
               "holding and playing coexist."));
    integer(tr("Hypotheses per frame (--topk)"), "relateBirthTopk",
            &P::birthTopk, 1, 512,
            tr("How many top-ranked relations are promoted each frame."));
    connect(m_belief, &QCheckBox::toggled, this,
            [this]() { markCustomProfile(); });
    for (auto& w : m_trackD)
        connect(w.second, qOverload<double>(&QDoubleSpinBox::valueChanged),
                this, [this]() { markCustomProfile(); });
    for (auto& w : m_trackI)
        connect(w.second, qOverload<int>(&QSpinBox::valueChanged), this,
                [this]() { markCustomProfile(); });
}
RelateTemporal::Params RelateAnythingDialog::trackingParams() const {
    RelateTemporal::Params p;
    for (const auto& w : m_trackD) p.*(w.first) = float(w.second->value());
    for (const auto& w : m_trackI) p.*(w.first) = w.second->value();
    p.belief = m_belief->isChecked();
    // Shared with the image path: relation floors and edge caps.
    p.onThr = float(m_relThreshold->value());
    p.spatialOnThr = float(m_spatialThreshold->value());
    p.maxEdges = m_maxRelations->value();
    p.maxSpatial = m_maxSpatial->value();
    return p;
}
void RelateAnythingDialog::resetTrackingDefaults() {
    const RelateTemporal::Params d;
    for (auto& w : m_trackD) w.second->setValue(d.*(w.first));
    for (auto& w : m_trackI) w.second->setValue(d.*(w.first));
    m_belief->setChecked(d.belief);
}
// make_reel.py MODES: YOLO-World boxes 0.15, YOLOE prompt-free masks 0.25;
// render_video.py's own default 0.20 elsewhere. The image demo is 0.25.
double RelateAnythingDialog::defaultConfidence(const QString& detector,
                                               int profile) const {
    if (profile == int(RelatePresets::Profile::ImageDemo)) return .25;
    if (detector.contains(QStringLiteral("-pf-"))) return .25;
    if (detector.contains(QStringLiteral("-world"))) return .15;
    return RelatePresets::thresholds(RelatePresets::Profile::VideoReel)
            .detConfidence;
}
void RelateAnythingDialog::applyProfile(int profile) {
    if (profile == int(RelatePresets::Profile::Custom)) return;
    const auto t = RelatePresets::thresholds(RelatePresets::Profile(profile));
    m_applyingPreset = true;
    m_threshold->setValue(
            defaultConfidence(m_detector->currentData().toString(), profile));
    m_detIou->setValue(t.detIou);
    m_maxBoxFraction->setValue(t.maxBoxFraction);
    m_relThreshold->setValue(t.relation);
    m_spatialThreshold->setValue(t.spatial);
    m_maxBoxes->setValue(t.maxBoxes);
    m_maxRelations->setValue(t.maxRelations);
    m_maxSpatial->setValue(t.maxSpatial);
    m_decompose->setChecked(t.decompose);
    resetTrackingDefaults();  // upstream defaults for every tracking flag
    m_applyingPreset = false;
    const int index = m_profile->findData(profile);
    if (index >= 0) m_profile->setCurrentIndex(index);
}
void RelateAnythingDialog::markCustomProfile() {
    if (m_applyingPreset || !m_profile) return;
    const int custom = m_profile->findData(int(RelatePresets::Profile::Custom));
    if (custom >= 0) m_profile->setCurrentIndex(custom);
}
RelateAnythingDialog::~RelateAnythingDialog() { shutdown(); }
void RelateAnythingDialog::shutdown() {
    if (!m_worker) return;
    cancel();
    saveSettings();
    QMetaObject::invokeMethod(
            m_worker, [worker = m_worker]() { worker->release(); },
            Qt::BlockingQueuedConnection);
    m_thread.quit();
    m_thread.wait();
    m_worker = nullptr;
}
void RelateAnythingDialog::closeEvent(QCloseEvent* event) {
    cancel();
    saveSettings();
    m_closed = true;
    if (m_worker)
        QMetaObject::invokeMethod(
                m_worker, [worker = m_worker]() { worker->release(); },
                Qt::QueuedConnection);
    event->accept();
}

void RelateAnythingDialog::reject() {
    // H1 fix: Esc must trigger the same cleanup as the close button. Qt's
    // default reject() only hides the dialog without calling closeEvent, so
    // streams and downloads stay running with the window invisible.
    close();
}

void RelateAnythingDialog::showEvent(QShowEvent* event) {
    m_closed = false;
    QDialog::showEvent(event);
}
void RelateAnythingDialog::saveSettings() {
    QSettings s;
    s.beginGroup("qRelateAnything");
    s.setValue("model", m_model->currentIndex());
    s.setValue("device", m_device->currentData());
    s.setValue("threads", m_threads->value());
    s.setValue("confidence", m_threshold->value());
    s.setValue("schema", 5);
    s.setValue("profile", m_profile->currentData());
    s.setValue("predicatePreset", m_predicatePreset->currentData());
    s.setValue("detectionIou", m_detIou->value());
    s.setValue("maxBoxFraction", m_maxBoxFraction->value());
    s.setValue("maxSpatial", m_maxSpatial->value());
    s.setValue("maxBoxes", m_maxBoxes->value());
    s.setValue("relationThreshold", m_relThreshold->value());
    s.setValue("spatialThreshold", m_spatialThreshold->value());
    s.setValue("maxRelations", m_maxRelations->value());
    s.setValue("predicates", m_predicates->text());
    s.setValue("objectSource", m_objectSource->currentIndex());
    s.setValue("regions", m_regions->text());
    s.setValue("masks", m_masks->text());
    s.setValue("vocabulary", m_vocabulary->text());
    s.setValue("vocabularyNames", m_vocabularyNames->text());
    s.setValue("predicateTypes", m_predicateTypes->text());
    s.setValue("arithmetic", m_arithmetic->currentData().toInt());
    s.setValue("decompose", m_decompose->isChecked());
    for (const auto& w : m_trackD)
        s.setValue(w.second->objectName(), w.second->value());
    for (const auto& w : m_trackI)
        s.setValue(w.second->objectName(), w.second->value());
    s.setValue("relateBelief", m_belief->isChecked());
    s.setValue("relateStabilize", m_stabilize->isChecked());
    s.setValue("boxScoreWeight", m_boxScoreWeight->isChecked());
    s.setValue("detectorModel", m_detector->currentData());
}
void RelateAnythingDialog::setDbImages(
        const QList<QPair<QString, QImage>>& images) {
    m_db = images;
}
void RelateAnythingDialog::addInput(const QString& path,
                                    const QImage& image,
                                    bool checked) {
    auto* item = new QListWidgetItem(path, m_inputs);
    item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
    item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
    item->setData(Qt::UserRole, image);
    item->setToolTip(path);
}
void RelateAnythingDialog::addFiles() {
    const auto files = QFileDialog::getOpenFileNames(
            this, tr("Input images"), {},
            tr("Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)"));
    for (const auto& path : files) addInput(path);
}
void RelateAnythingDialog::useTestData() {
    m_closed = false;
    auto kind = ecvTestDataRepository::Dataset::RelateAnything;
    auto& repo = ecvTestDataRepository::instance();
    const QStringList names{"skateboard.jpg", "horse.jpg",     "frisbee.jpg",
                            "tennis.jpg",     "catlaptop.jpg", "bicycle.jpg"};
    auto find = [&]() {
        QStringList paths;
        for (const auto& n : names) paths << repo.findDatasetFile(kind, n);
        return paths;
    };
    auto demos = find();
    QString video = repo.findDatasetFile(kind, "r3c_adoption_54729571599.webm");
    if ((demos.contains(QString()) || video.isEmpty()) &&
        repo.isDatasetAvailable(kind)) {
        repo.extractDataset(kind);
        demos = find();
        video = repo.findDatasetFile(kind, "r3c_adoption_54729571599.webm");
    }
    if (demos.contains(QString()) || video.isEmpty()) {
        if (repo.isDownloadInProgress()) {
            log(tr("Another shared sample download is running."));
            return;
        }
        m_waitingForSample = true;
        m_sample->setEnabled(false);
        repo.startDownload(kind);
        return;
    }
    const bool videoTab =
            m_tabs->currentWidget()->findChild<RelateAnythingVideoWidget*>() ==
            m_video;
    cancel();
    m_inputs->clear();
    for (int i = 0; i < demos.size(); ++i)
        addInput(demos[i], {}, names[i] == "horse.jpg");
    m_inputs->setCurrentRow(names.indexOf("horse.jpg"));
    m_video->selectVideoFileSource();
    m_video->setVideoFilePath(video);
    m_tabs->setCurrentIndex(videoTab ? m_tabs->indexOf(m_video->parentWidget())
                                     : 0);
    log(tr("Loaded six image samples and the recommended video. The horse "
           "image is checked; check any other images to include them."));
}
QString RelateAnythingDialog::relationAssetPath(int index) const {
    std::unique_ptr<char, decltype(&aicore_relation_free_buffer)> dir(
            aicore_relation_model_cache_dir(), aicore_relation_free_buffer);
    const auto* e = aicore_relation_model_at(index);
    return e && dir ? QDir(QString::fromUtf8(dir.get()))
                              .filePath(QString::fromUtf8(e->filename))
                    : QString();
}
QString RelateAnythingDialog::modelPath() const {
    return relationAssetPath(m_model->currentData().toInt());
}
QString RelateAnythingDialog::detectorPath() const {
    const auto info = detectorInfo(m_detector->currentData().toString());
    return info.valid() && !info.dir.isEmpty()
                   ? QDir(info.dir).filePath(info.filename)
                   : QString();
}
QString RelateAnythingDialog::detectorTextModelPath() const {
    const auto info = detectorInfo(m_detector->currentData().toString());
    if (!info.valid() || !info.vocabulary) return {};
    return QDir(cacheDir(aicore_yolo_model_cache_dir(),
                         aicore_yolo_free_buffer))
            .filePath(textModelFile(info));
}
RelateAnythingSettings RelateAnythingDialog::settings() const {
    RelateAnythingSettings s;
    s.model = modelPath();
    s.detector = detectorPath();
    const auto info = detectorInfo(m_detector->currentData().toString());
    s.detectorIsRfdetr = info.rfdetr;
    if (info.vocabulary) {
        // YOLO-World / YOLOE: reproduce upstream's megasg497 detector by
        // baking the same 497 names through the matching text encoder.
        for (int i = 0; i < aicore_relation_object_vocabulary_count(); ++i)
            s.detectorClasses << QString::fromUtf8(
                    aicore_relation_object_vocabulary_at(i));
        s.detectorTextModel = detectorTextModelPath();
    }
    s.spatialThreshold = float(m_spatialThreshold->value());
    s.device = m_device->currentData().toString();
    s.threads = m_threads->value();
    s.threshold = float(m_threshold->value());
    s.maxBoxes = m_maxBoxes->value();
    s.maxRelations = m_maxRelations->value();
    s.maxSpatial = m_maxSpatial->value();
    s.detIou = float(m_detIou->value());
    s.maxBoxFraction = float(m_maxBoxFraction->value());
    s.relationThreshold = float(m_relThreshold->value());
    s.strict = m_arithmetic->currentData().toInt();
    s.decompose = m_decompose->isChecked();
    s.tracking = trackingParams();
    s.boxScoreWeight = m_boxScoreWeight->isChecked();
    // Only video / camera frames are stabilised; batches and DB images are
    // independent. submit() sets `temporal` for frames.
    s.temporal = false;
    s.vocabulary = m_vocabulary->text().trimmed();
    s.vocabularyNames = m_vocabularyNames->text().trimmed();
    s.predicateTypes = m_predicateTypes->text().trimmed();
    if (s.vocabulary.isEmpty()) s.predicates = m_predicates->text();
    if (m_objectSource->currentIndex() == 1) {
        s.regions = m_regions->text().trimmed();
        s.masks = m_masks->text().trimmed();
    }
    const auto* model =
            aicore_relation_model_at(m_model->currentData().toInt());
    for (int i = 0; model && i < aicore_relation_model_count(); ++i) {
        const auto* e = aicore_relation_model_at(i);
        if (e->role == AICORE_RELATION_MODEL_TEXT &&
            QString::fromUtf8(e->architecture) ==
                    QString::fromUtf8(model->architecture))
            s.student = relationAssetPath(i);
        if (e->role == AICORE_RELATION_MODEL_TOKENIZER)
            s.tokenizer = relationAssetPath(i);
    }
    return s;
}
bool RelateAnythingDialog::ensureModel(bool runAfter) {
    if (m_downloader->isBusy()) return false;
    if (m_objectSource->currentIndex() == 1 &&
        m_regions->text().trimmed().isEmpty()) {
        log(tr("Choose an object-box file in Options before running."));
        return false;
    }
    const auto opts = settings();
    QList<ecvModelDownloader::Request> required;
    for (int i = 0; i < aicore_relation_model_count(); ++i) {
        const auto* e = aicore_relation_model_at(i);
        const auto path = relationAssetPath(i);
        if (path != opts.model &&
            (opts.predicates.trimmed().isEmpty() ||
             (path != opts.student && path != opts.tokenizer)))
            continue;
        ecvModelDownloader::Request r;
        r.url = QString::fromUtf8(e->download_url);
        r.destPath = path;
        r.contentAnchor = {QCryptographicHash::Sha256, QByteArray(e->sha256)};
        r.ingestExactSize = e->size_bytes;
        r.requireGgufMagic = e->role != AICORE_RELATION_MODEL_TOKENIZER;
        required << r;
    }
    if (m_objectSource->currentIndex() == 0) {
        const auto info = detectorInfo(m_detector->currentData().toString());
        if (!info.valid()) return false;
        ecvModelDownloader::Request d;
        d.url = info.url;
        d.destPath = opts.detector;
        d.contentAnchor = {QCryptographicHash::Sha256,
                           ecvAssetIntegrity::PinnedDigest(info.filename)};
        required << d;
        if (!opts.detectorTextModel.isEmpty()) {
            const QString textFile =
                    QFileInfo(opts.detectorTextModel).fileName();
            for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_TEXT);
                 ++i) {
                const auto* e = aicore_yolo_model_at(i, AICORE_YOLO_ROLE_TEXT);
                if (textFile != QString::fromUtf8(e->filename)) continue;
                ecvModelDownloader::Request t;
                t.url = QString::fromUtf8(e->download_url);
                t.destPath = opts.detectorTextModel;
                t.contentAnchor = {QCryptographicHash::Sha256,
                                   ecvAssetIntegrity::PinnedDigest(textFile)};
                required << t;
            }
        }
    }
    for (const auto& r : required) {
        if (ecvAssetIntegrity::isVerified(
                    r.destPath, r.contentAnchor, r.minBytes, r.requireGgufMagic,
                    ecvAssetIntegrity::OnMiss::CheapChecksOnly,
                    r.ingestExactSize))
            continue;
        m_runAfterDownload = runAfter;
        // H2 fix: disable device and detector too, so changing them while
        // downloading doesn't trigger cancel() before finished() arrives.
        m_download->setEnabled(false);
        m_model->setEnabled(false);
        m_device->setEnabled(false);
        m_detector->setEnabled(false);
        m_run->setEnabled(false);
        m_progress->show();
        m_downloader->download(r);
        return false;
    }
    return true;
}
void RelateAnythingDialog::run() {
    if (m_busy || !m_worker) return;
    m_closed = false;
    const bool videoInput =
            m_tabs->currentWidget()->findChild<RelateAnythingVideoWidget*>() ==
            m_video;
    if (!videoInput) {
        bool checked = false;
        for (int i = 0; i < m_inputs->count(); ++i)
            checked |= m_inputs->item(i)->checkState() == Qt::Checked;
        if (!checked) {
            log(tr("Check at least one image or use sample data first."));
            return;
        }
    }
    if (!ensureModel(true)) return;
    saveSettings();
    if (m_tabs->currentWidget()->findChild<RelateAnythingVideoWidget*>() ==
        m_video) {
        m_batch = false;
        resetResults();
        if (m_video->inputSource() == VideoPlaybackWidget::InputSource::Camera)
            m_video->startCamera(m_video->selectedCameraIndex());
        else
            m_video->startVideoFile(m_video->videoFilePath());
        return;
    }
    m_queue.clear();
    for (int i = 0; i < m_inputs->count(); ++i) {
        auto* item = m_inputs->item(i);
        if (item->checkState() == Qt::Checked)
            m_queue.append({item->text(),
                            qvariant_cast<QImage>(item->data(Qt::UserRole))});
    }
    if (m_queue.isEmpty()) {
        log(tr("Check at least one image or use sample data first."));
        return;
    }
    resetResults();
    m_batch = true;
    m_batchTotal = m_queue.size();
    m_batchCompleted = 0;
    m_progress->setRange(0, m_batchTotal);
    m_progress->setValue(0);
    m_progress->show();
    m_jobSettings = settings();
    runNext();
}
void RelateAnythingDialog::runNext() {
    if (m_queue.isEmpty()) {
        m_batch = false;
        m_progress->hide();
        m_tabs->setCurrentIndex(1);
        return;
    }
    auto pair = m_queue.takeFirst();
    submit(pair.second, pair.first, false);
}
void RelateAnythingDialog::submit(QImage image,
                                  const QString& source,
                                  bool frame) {
    // L2 fix: guard worker access after shutdown.
    if (!m_worker) return;
    m_framePending = frame;
    setBusy(true);
    m_worker->resetCancel();
    auto opts = frame ? settings() : m_jobSettings;
    opts.temporal = frame && m_stabilize && m_stabilize->isChecked();
    auto generation = m_generation;
    QMetaObject::invokeMethod(
            m_worker,
            [worker = m_worker, opts, image, source, generation]() {
                worker->process(opts, image, source, generation);
            },
            Qt::QueuedConnection);
}
void RelateAnythingDialog::completed(const RelateAnythingResult& result) {
    bool current = result.generation == m_generation;
    const bool acknowledge = m_framePending;
    m_framePending = false;
    setBusy(false);
    if (current) {
        // H3 fix: check result.ok instead of error.isEmpty(), because an empty
        // error string from aicore_*_last_error(nullptr) is still a failure.
        if (!result.ok) {
            log(result.error.isEmpty() ? tr("Inference failed (no details).")
                                       : result.error);
            // M1 fix: for a batch, log the error but continue to the next image
            // unless it's a fatal session error (model/detector load failure).
            if (result.fatal) {
                m_queue.clear();
                m_batch = false;
                m_progress->hide();
                m_video->stopStream();
            } else if (m_batch) {
                // Non-fatal error in a batch: log and continue to next image.
                // Don't stop the stream or clear the queue.
            } else {
                // Non-fatal error in single-image or video mode: stop stream.
                m_video->stopStream();
            }
        } else {
            m_last = result;
            recordResult(result.json);
            showResult();
            m_preview->setPreviewImage(result.annotated, QSize(480, 280));
            const auto timings = result.json.value("timings").toObject();
            m_timing->setText(
                    tr("%1 · %2 · worker %3 ms · inference %4 ms · %5")
                            .arg(result.source, result.device)
                            .arg(result.e2eMs, 0, 'f', 1)
                            .arg(timings.value("relation_forward_ms")
                                         .toDouble(),
                                 0, 'f', 1)
                            .arg(timings.value("context_reused").toBool()
                                         ? tr("warm session")
                                         : tr("cold start")));
        }
        if (m_batch) m_progress->setValue(++m_batchCompleted);
    }
    if (acknowledge)
        m_video->acknowledge(current && result.ok ? result.annotated
                                                  : QImage());
    if (!m_heldFrameSource.isEmpty() && !m_closed) {
        QImage image = std::move(m_heldFrame);
        QString source = std::move(m_heldFrameSource);
        m_heldFrame = {};
        m_heldFrameSource.clear();
        submit(std::move(image), source, true);
    }
    if (current && m_batch)
        QTimer::singleShot(0, this, &RelateAnythingDialog::runNext);
}
void RelateAnythingDialog::showResult() {
    QList<QJsonObject> visible;
    for (const auto& v : m_last.json.value("relations").toArray()) {
        auto e = v.toObject();
        const QString text = e["subject_name"].toString() + " " +
                             e["predicate"].toString() + " " +
                             e["object_name"].toString();
        if (text.contains(m_filter->text(), Qt::CaseInsensitive)) visible << e;
    }
    m_table->setRowCount(visible.size());
    for (int i = 0; i < visible.size(); ++i) {
        const auto e = visible[i];
        QStringList values{e["subject_name"].toString(),
                           e["predicate"].toString(),
                           e["object_name"].toString(),
                           QString::number(e["score"].toDouble(), 'f', 5)};
        for (int j = 0; j < 4; ++j)
            m_table->setItem(i, j, new QTableWidgetItem(values[j]));
    }
    m_summary->setText(tr("%1 objects · %2 relations · %3")
                               .arg(m_last.json["objects"].toArray().size())
                               .arg(visible.size())
                               .arg(m_last.source));
}
void RelateAnythingDialog::setBusy(bool busy) {
    // L9 fix: only update tab state when it changes to avoid per-frame flicker.
    if (busy != m_busy) {
        m_busy = busy;
        m_tabs->setTabEnabled(2, !busy);
    }
    m_run->setEnabled(!busy && !m_downloader->isBusy());
}
void RelateAnythingDialog::resetResults() {
    m_last = RelateAnythingResult{};
    showResult();
    m_timing->clear();
    m_resultCount = 0;
    m_resultSpool.reset(new QTemporaryFile);
    if (!m_resultSpool->open()) {
        log(tr("Could not create temporary result spool: %1")
                    .arg(m_resultSpool->errorString()));
        m_resultSpool.reset();
    }
}
void RelateAnythingDialog::recordResult(const QJsonObject& result) {
    if (m_resultSpool) {
        QByteArray line = QJsonDocument(result).toJson(QJsonDocument::Compact);
        line.append('\n');
        if (m_resultSpool->write(line) != line.size())
            log(tr("Could not spool result row: %1")
                        .arg(m_resultSpool->errorString()));
        else
            ++m_resultCount;
    }
}
void RelateAnythingDialog::cancel() {
    ++m_generation;
    m_queue.clear();
    m_batch = false;
    m_framePending = false;
    m_heldFrame = {};
    m_heldFrameSource.clear();
    m_waitingForSample = false;
    if (m_sample) m_sample->setEnabled(true);
    if (m_progress) m_progress->hide();
    if (m_worker) m_worker->requestCancel();
    if (m_video) m_video->stopStream();
    if (m_downloader && m_downloader->isBusy()) {
        m_runAfterDownload = false;
        m_downloader->cancel();
    }
}
void RelateAnythingDialog::log(const QString& text) {
    CVLog::Print(QStringLiteral("[qRelateAnything] %1").arg(text));
}
void RelateAnythingDialog::exportResults() {
    if (m_resultCount == 0 || !m_resultSpool) {
        log(tr("Run inference before exporting."));
        return;
    }
    QString selectedFilter;
    auto path = QFileDialog::getSaveFileName(
            this, tr("Export results"), "relate-results.json",
            tr("JSON (*.json);;CSV (*.csv);;Text (*.txt)"), &selectedFilter);
    if (path.isEmpty()) return;
    // M6 fix: derive format from selected filter if extension is missing
    if (!path.contains('.')) {
        if (selectedFilter.contains("json", Qt::CaseInsensitive))
            path += ".json";
        else if (selectedFilter.contains("csv", Qt::CaseInsensitive))
            path += ".csv";
        else
            path += ".txt";
    }
    // M6 fix: check return value and report failure
    if (!saveResults(path)) log(tr("Failed to export results to %1").arg(path));
}
bool RelateAnythingDialog::saveResults(const QString& path) {
    if (path.isEmpty() || !m_resultCount || !m_resultSpool) return false;
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || !m_resultSpool->flush() ||
        !m_resultSpool->seek(0))
        return false;
    const bool json = path.endsWith(".json", Qt::CaseInsensitive),
               csv = path.endsWith(".csv", Qt::CaseInsensitive);
    bool ok = true, first = true;
    auto write = [&](const QByteArray& bytes) {
        return file.write(bytes) == bytes.size();
    };
    auto quote = [](QString value) -> QString {
        value.replace('"', "\"\"");
        return '"' + value + '"';
    };
    if (json) ok = write("[\n");
    if (csv) ok = write("source,subject,predicate,object,score\n");
    while (ok && !m_resultSpool->atEnd()) {
        auto line = m_resultSpool->readLine().trimmed();
        if (line.isEmpty()) continue;
        auto r = QJsonDocument::fromJson(line).object();
        if (json) {
            if (!first) ok = write(",\n");
            if (ok) ok = write(line);
            first = false;
        } else
            for (const auto& value : r["relations"].toArray()) {
                const auto e = value.toObject();
                QString row;
                if (csv)
                    row = quote(r["source"].toString()) + "," +
                          quote(e["subject_name"].toString()) + "," +
                          quote(e["predicate"].toString()) + "," +
                          quote(e["object_name"].toString()) + "," +
                          QString::number(e["score"].toDouble(), 'g', 9);
                else
                    row = r["source"].toString() + ": " +
                          e["subject_name"].toString() + " " +
                          e["predicate"].toString() + " " +
                          e["object_name"].toString() + " (" +
                          QString::number(e["score"].toDouble(), 'g', 5) + ")";
                if (!write((row + "\n").toUtf8())) {
                    ok = false;
                    break;
                }
            }
    }
    if (json && ok) ok = write("\n]\n");
    ok = ok && file.commit();
    m_resultSpool->seek(m_resultSpool->size());
    return ok;
}
