// SPDX-License-Identifier: MIT
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

RelateAnythingDialog::RelateAnythingDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("RelateAnything · Scene relations"));
    resize(1000, 620);
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
    m_model->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
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
    m_device->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
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
    auto* quickStart = new QLabel(
            tr("Click Try sample data to fill image and video inputs, then Run. "
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
                    if (!dir.isEmpty()) m_inputs->addItem(f.absoluteFilePath());
                }
            } else if (action == tr("Clear")) {
                cancel();
                m_inputs->clear();
                resetResults();
            } else
                for (const auto& pair : m_db) {
                    auto* item =
                            new QListWidgetItem("db://" + pair.first, m_inputs);
                    item->setData(Qt::UserRole, pair.second);
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
        m_worker->requestCancel();
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
    auto* ol = new QFormLayout(options);
    m_threads = new QSpinBox;
    m_threads->setObjectName("relateThreads");
    m_threads->setRange(1, 256);
    m_threads->setValue(8);
    m_detector = new QComboBox;
    m_detector->setObjectName("relateDetector");
    for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_DETECTION);
         ++i) {
        const auto* e = aicore_yolo_model_at(i, AICORE_YOLO_ROLE_DETECTION);
        m_detector->addItem(QString::fromUtf8(e->display_name), i);
    }
    m_detector->setCurrentIndex(
            aicore_yolo_model_default_index(AICORE_YOLO_ROLE_DETECTION));
    m_threshold = new QDoubleSpinBox;
    m_threshold->setRange(.01, 1);
    m_threshold->setSingleStep(.05);
    m_threshold->setValue(.25);
    m_threshold->setObjectName("relateDetectionThreshold");
    m_maxBoxes = new QSpinBox;
    m_maxBoxes->setRange(2, 60);
    m_maxBoxes->setValue(60);
    m_topk = new QSpinBox;
    m_topk->setRange(1, 1000);
    m_topk->setValue(128);
    m_decompose = new QCheckBox(tr("Separate spatial and semantic relations"));
    m_decompose->setObjectName("relateDecompose");
    m_strict = new QCheckBox(tr("Use precise model arithmetic"));
    m_strict->setChecked(aicore_relation_default_options().strict_f32 != 0);
    m_strict->setObjectName("relateStrict");
    m_predicates = new QLineEdit;
    m_predicates->setObjectName("relatePredicates");
    m_predicates->setPlaceholderText(
            tr("Optional predicates separated by |, for example riding | "
               "holding | next to"));
    ol->addRow(tr("Object detector"), m_detector);
    ol->addRow(tr("Detection confidence"), m_threshold);
    ol->addRow(tr("Maximum objects"), m_maxBoxes);
    ol->addRow(tr("Top relations"), m_topk);
    ol->addRow(tr("CPU threads"), m_threads);
    ol->addRow(m_decompose);
    ol->addRow(m_strict);
    ol->addRow(tr("Custom predicates"), m_predicates);
    m_tabs->addTab(options, tr("Options"));
    auto* videoPage = new QWidget;
    auto* videoLayout = new QVBoxLayout(videoPage);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    auto* videoGuide =
            new QLabel(tr("Use test data fills the published video path as "
                          "well as the image list. "
                          "Load a video or camera and Run. Each frame advances "
                          "when inference completes."));
    videoGuide->setWordWrap(true);
    videoLayout->addWidget(videoGuide);
    m_sampleVideo = new QPushButton(tr("Load sample video"));
    m_sampleVideo->setObjectName("relateLoadSampleVideo");
    m_sampleVideo->setEnabled(VideoPlaybackWidget::isAvailable());
    videoLayout->addWidget(m_sampleVideo, 0, Qt::AlignLeft);
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
    connect(m_sampleVideo, &QPushButton::clicked, this,
            &RelateAnythingDialog::loadSampleVideo);
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
                m_download->setEnabled(true);
                m_model->setEnabled(true);
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
    m_threshold->setValue(s.value("confidence", .25).toDouble());
    m_topk->setValue(s.value("topk", 128).toInt());
    m_maxBoxes->setValue(s.value("maxBoxes", 60).toInt());
    m_predicates->setText(s.value("predicates").toString());
    m_strict->setChecked(
            s.value("strict", aicore_relation_default_options().strict_f32 != 0)
                    .toBool());
    m_decompose->setChecked(s.value("decompose", false).toBool());
    m_detector->setCurrentIndex(
            qBound(0, s.value("detector", m_detector->currentIndex()).toInt(),
                   m_detector->count() - 1));
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
    s.setValue("topk", m_topk->value());
    s.setValue("maxBoxes", m_maxBoxes->value());
    s.setValue("predicates", m_predicates->text());
    s.setValue("strict", m_strict->isChecked());
    s.setValue("decompose", m_decompose->isChecked());
    s.setValue("detector", m_detector->currentIndex());
}
void RelateAnythingDialog::setDbImages(
        const QList<QPair<QString, QImage>>& images) {
    m_db = images;
}
void RelateAnythingDialog::addFiles() {
    m_inputs->addItems(QFileDialog::getOpenFileNames(
            this, tr("Input images"), {},
            tr("Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)")));
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
    m_inputs->addItems(demos);
    m_inputs->setCurrentRow(0);
    m_preview->setPreviewImage(QImage(demos.front()), QSize(480, 280));
    m_video->selectVideoFileSource();
    m_video->setVideoFilePath(video);
    m_tabs->setCurrentIndex(videoTab ? m_tabs->indexOf(m_video->parentWidget())
                                     : 0);
    log(tr("Loaded six published image samples and the published video "
           "sample."));
}
void RelateAnythingDialog::loadSampleVideo() {
    useTestData();
    m_tabs->setCurrentIndex(m_tabs->indexOf(m_video->parentWidget()));
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
    std::unique_ptr<char, decltype(&aicore_yolo_free_buffer)> dir(
            aicore_yolo_model_cache_dir(), aicore_yolo_free_buffer);
    const auto* e = aicore_yolo_model_at(m_detector->currentData().toInt(),
                                         AICORE_YOLO_ROLE_DETECTION);
    return e && dir ? QDir(QString::fromUtf8(dir.get()))
                              .filePath(QString::fromUtf8(e->filename))
                    : QString();
}
RelateAnythingSettings RelateAnythingDialog::settings() const {
    RelateAnythingSettings s;
    s.model = modelPath();
    s.detector = detectorPath();
    s.device = m_device->currentData().toString();
    s.threads = m_threads->value();
    s.threshold = float(m_threshold->value());
    s.maxBoxes = m_maxBoxes->value();
    s.topk = m_topk->value();
    s.strict = m_strict->isChecked();
    s.decompose = m_decompose->isChecked();
    s.predicates = m_predicates->text();
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
    const auto* detector = aicore_yolo_model_at(
            m_detector->currentData().toInt(), AICORE_YOLO_ROLE_DETECTION);
    if (!detector) return false;
    ecvModelDownloader::Request d;
    d.url = QString::fromUtf8(detector->download_url);
    d.destPath = opts.detector;
    d.contentAnchor = {QCryptographicHash::Sha256,
                       ecvAssetIntegrity::PinnedDigest(
                               QString::fromUtf8(detector->filename))};
    required << d;
    for (const auto& r : required) {
        if (ecvAssetIntegrity::isVerified(
                    r.destPath, r.contentAnchor, r.minBytes, r.requireGgufMagic,
                    ecvAssetIntegrity::OnMiss::CheapChecksOnly,
                    r.ingestExactSize))
            continue;
        m_runAfterDownload = runAfter;
        m_download->setEnabled(false);
        m_model->setEnabled(false);
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
    const auto selectedInputs = m_inputs->selectedItems();
    if (!selectedInputs.isEmpty()) {
        for (auto* item : selectedInputs)
            m_queue.append({item->text(),
                            qvariant_cast<QImage>(item->data(Qt::UserRole))});
    } else {
        for (int i = 0; i < m_inputs->count(); ++i) {
            auto* item = m_inputs->item(i);
            m_queue.append({item->text(),
                            qvariant_cast<QImage>(item->data(Qt::UserRole))});
        }
    }
    if (m_queue.isEmpty()) {
        log(tr("Add images or use sample data first."));
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
    m_framePending = frame;
    setBusy(true);
    m_worker->resetCancel();
    auto opts = frame ? settings() : m_jobSettings;
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
        if (!result.error.isEmpty()) {
            log(result.error);
            m_queue.clear();
            m_batch = false;
            m_progress->hide();
            m_video->stopStream();
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
        m_video->acknowledge(current && result.error.isEmpty()
                                     ? result.annotated
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
    m_busy = busy;
    m_tabs->setTabEnabled(2, !busy);
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
    auto path = QFileDialog::getSaveFileName(
            this, tr("Export results"), "relate-results.json",
            tr("JSON (*.json);;CSV (*.csv);;Text (*.txt)"));
    if (path.isEmpty()) return;
    saveResults(path);
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
