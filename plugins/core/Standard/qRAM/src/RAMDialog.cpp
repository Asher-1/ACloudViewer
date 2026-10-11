// SPDX-License-Identifier: MIT
#include "RAMDialog.h"

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
#include <QTabWidget>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <memory>
#include <utility>

#include "RAMVideoWidget.h"
#include "aicore/backend_capi.h"
#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "CVLog.h"
#include "ecvPersistentSettings.h"
#include "ecvTestDataRepository.h"

RAMDialog::RAMDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("RAM++ · Recognize Anything"));
    resize(1000, 620);
    ecvPS::registerSettingsGroup(QStringLiteral("qRAM"));
    qRegisterMetaType<RAMResult>();
    m_worker = new RAMWorker;
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(m_worker, &RAMWorker::completed, this, &RAMDialog::completed);
    m_thread.start();
    auto* root = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(root);
    auto* top = new QHBoxLayout;
    m_model = new QComboBox;
    m_model->setObjectName("ramModel");
    for (int i = 0; i < aicore_tagging_model_count(); ++i) {
        auto* e = aicore_tagging_model_at(i);
        m_model->addItem(QString::fromUtf8(e->display_name), i);
    }
    m_model->setCurrentIndex(aicore_tagging_model_default_index());
    m_device = new QComboBox;
    m_device->setObjectName("ramDevice");
    for (int i = 0; i < aicore_device_count(); ++i) {
        auto* d = aicore_device_at(i);
        m_device->addItem(QString::fromUtf8(d->label),
                          QString::fromUtf8(d->id));
    }
    m_download = new QPushButton(tr("Download model"));
    m_sample = ecvAICoreUi::makeSampleDataBtn(this);
    m_sample->setObjectName("ramUseTestData");
    top->addWidget(ecvAICoreUi::makeLabel(tr("Model")));
    top->addWidget(m_model, 2);
    top->addWidget(ecvAICoreUi::makeLabel(tr("Device")));
    top->addWidget(m_device, 1);
    top->addWidget(m_download);
    top->addWidget(m_sample);
    root->addLayout(top);
    auto* quickStart = new QLabel(
            tr("<b>Quick start:</b> Click <i>Use test data</i> to load four "
               "official RAM demos, then Run. <b>Tag2Text</b> generates a "
               "sentence; <b>RAM/RAM++</b> return labels. Auto device picks "
               "CUDA, Vulkan, then CPU. Per-class thresholds use the model's "
               "defaults; caption length and beam apply only to Tag2Text."));
    quickStart->setObjectName("ramQuickStart");
    quickStart->setWordWrap(true);
    root->addWidget(quickStart);
    m_tabs = new QTabWidget;
    m_tabs->setObjectName("ramTabs");
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
            } else if (action == tr("Clear"))
                m_inputs->clear();
            else
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
    m_inputs->setObjectName("ramInputs");
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
    m_video = new RAMVideoWidget;
    m_video->setEnabled(VideoPlaybackWidget::isAvailable());
    connect(m_video, &RAMVideoWidget::frameReady, this,
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
    connect(m_video, &RAMVideoWidget::invalidated, this, [this]() {
        ++m_generation;
        m_framePending = false;
        m_heldFrame = {};
        m_heldFrameSource.clear();
        m_queue.clear();
        if (m_batch && m_progress) m_progress->hide();
        m_batch = false;
        m_worker->requestCancel();
    });
    connect(m_video, &VideoPlaybackWidget::streamError, this, &RAMDialog::log);
    connect(m_video, &VideoPlaybackWidget::logMessage, this, &RAMDialog::log);
    auto* resultPage = new QWidget;
    auto* rl = new QVBoxLayout(resultPage);
    ecvAICoreUi::setupTabLayout(rl);
    m_summary = new QLabel;
    m_summary->setWordWrap(true);
    rl->addWidget(m_summary);
    auto* filterRow = new QHBoxLayout;
    m_filter = new QLineEdit;
    m_filter->setPlaceholderText(tr("Filter tags…"));
    m_filter->setObjectName("ramFilter");
    m_selected = new QCheckBox(tr("Selected tags only"));
    m_selected->setObjectName("ramSelectedOnly");
    m_selected->setChecked(true);
    filterRow->addWidget(m_filter, 1);
    filterRow->addWidget(m_selected);
    rl->addLayout(filterRow);
    m_table = new QTableWidget(0, 4);
    m_table->setObjectName("ramTags");
    m_table->setHorizontalHeaderLabels(
            {tr("Tag"), tr("Probability"), tr("Threshold"), tr("Logit")});
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_table->setMinimumHeight(220);
    m_table->setMaximumHeight(320);
    rl->addWidget(m_table);
    m_captionText = new QPlainTextEdit;
    m_captionText->setReadOnly(true);
    m_captionText->setMaximumHeight(90);
    m_captionText->setPlaceholderText(tr("Tag2Text caption"));
    rl->addWidget(m_captionText);
    auto* exports = new QHBoxLayout;
    auto* exportButton = new QPushButton(tr("Export JSON / CSV / text…"));
    auto* dbButton = new QPushButton(tr("Add result image to DB"));
    exportButton->setObjectName("ramExport");
    dbButton->setObjectName("ramAddToDb");
    exports->addWidget(exportButton);
    exports->addWidget(dbButton);
    exports->addStretch();
    rl->addLayout(exports);
    connect(exportButton, &QPushButton::clicked, this,
            &RAMDialog::exportResults);
    connect(dbButton, &QPushButton::clicked, this, [this]() {
        if (!m_last.image.isNull()) emit addToDbRequested(m_last);
    });
    connect(m_filter, &QLineEdit::textChanged, this, &RAMDialog::showResult);
    connect(m_selected, &QCheckBox::toggled, this, &RAMDialog::showResult);
    m_tabs->addTab(resultPage, tr("Results"));
    auto* options = new QWidget;
    auto* ol = new QFormLayout(options);
    m_threads = new QSpinBox;
    m_threads->setObjectName("ramThreads");
    m_threads->setRange(1, 256);
    m_threads->setValue(8);
    m_cpuPrecision = new QComboBox;
    m_cpuPrecision->setObjectName("ramCpuPrecision");
    m_cpuPrecision->addItem(tr("Auto"), AICORE_TAGGING_CPU_AUTO);
    m_cpuPrecision->addItem(tr("Upstream arithmetic"),
                            AICORE_TAGGING_CPU_UPSTREAM);
    m_defaults = new QCheckBox(tr("Use model per-class thresholds"));
    m_defaults->setObjectName("ramDefaultThresholds");
    m_defaults->setChecked(true);
    m_threshold = new QDoubleSpinBox;
    m_threshold->setObjectName("ramThreshold");
    m_threshold->setRange(0, 1);
    m_threshold->setDecimals(3);
    m_threshold->setSingleStep(.01);
    m_threshold->setValue(.68);
    m_threshold->setEnabled(false);
    m_epsilon = new QDoubleSpinBox;
    m_epsilon->setObjectName("ramThresholdEpsilon");
    m_epsilon->setRange(0, 1);
    m_epsilon->setDecimals(4);
    m_epsilon->setSingleStep(.001);
    m_epsilon->setToolTip(
            tr("Subtract this margin from Tag2Text thresholds. "
               "Zero matches the upstream default."));
    m_caption = new QCheckBox(tr("Generate caption (Tag2Text)"));
    m_length = new QSpinBox;
    m_length->setObjectName("ramCaptionLength");
    m_length->setRange(10, 512);
    m_length->setValue(30);
    m_caption->setObjectName("ramCaption");
    m_beam = new QComboBox;
    m_beam->setObjectName("ramBeam");
    m_beam->addItem(tr("Auto"), 0);
    m_beam->addItem(tr("Sequential beams"), 1);
    m_beam->addItem(tr("Batch 3 beams"), 3);
    m_specified = new QLineEdit;
    m_specified->setObjectName("ramSpecifiedTags");
    m_specified->setPlaceholderText(
            tr("Optional caption tags, separated by | or commas"));
    ol->addRow(tr("CPU threads"), m_threads);
    ol->addRow(tr("CPU precision"), m_cpuPrecision);
    ol->addRow(m_defaults);
    ol->addRow(tr("Threshold"), m_threshold);
    ol->addRow(tr("Tag2Text threshold margin"), m_epsilon);
    ol->addRow(m_caption);
    ol->addRow(tr("Caption length"), m_length);
    ol->addRow(tr("Beam execution"), m_beam);
    ol->addRow(tr("Specified tags"), m_specified);
    auto addPath = [this, ol](const QString& title, QLineEdit** edit,
                              const char* name) {
        auto* row = new QWidget;
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        *edit = new QLineEdit;
        (*edit)->setObjectName(name);
        auto* b = ecvAICoreUi::makeBrowseBtn(tr("Browse…"));
        layout->addWidget(*edit, 1);
        layout->addWidget(b);
        connect(b, &QPushButton::clicked, this, [this, edit, title]() {
            auto path = QFileDialog::getOpenFileName(this, title);
            if (!path.isEmpty()) (*edit)->setText(path);
        });
        ol->addRow(title, row);
    };
    addPath(tr("Custom label list"), &m_labels, "ramTagList");
    addPath(tr("Custom thresholds"), &m_thresholds, "ramThresholdFile");
    addPath(tr("Custom label embeddings"), &m_embeddings, "ramEmbeddingFile");
    m_tabs->addTab(options, tr("Options"));
    auto* videoPage = new QWidget;
    auto* videoLayout = new QVBoxLayout(videoPage);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    auto* videoGuide = new QLabel(
            tr("Video inference is consumer-driven. Load a video file or use "
               "the verified official RAM archive sample."));
    videoGuide->setWordWrap(true);
    videoLayout->addWidget(videoGuide);
    m_sampleVideo = new QPushButton(tr("Load sample video"));
    m_sampleVideo->setObjectName("ramLoadSampleVideo");
    m_sampleVideo->setEnabled(VideoPlaybackWidget::isAvailable());
    videoLayout->addWidget(m_sampleVideo, 0, Qt::AlignLeft);
    videoLayout->addWidget(m_video, 1);
    m_tabs->addTab(videoPage, tr("Video / camera"));
    connect(m_defaults, &QCheckBox::toggled, this,
            [this](bool v) { m_threshold->setEnabled(!v); });
    auto updateCaption = [this]() {
        auto* e = aicore_tagging_model_at(m_model->currentData().toInt());
        bool supported = e && e->supports_caption;
        m_caption->setEnabled(supported);
        if (!supported) m_caption->setChecked(false);
        m_length->setEnabled(supported);
        m_beam->setEnabled(supported);
        m_specified->setEnabled(supported);
        m_epsilon->setEnabled(supported);
    };
    connect(m_model, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this, updateCaption]() {
                cancel();
                updateCaption();
            });
    connect(m_device, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this]() { cancel(); });
    updateCaption();
    m_timing = new QLabel;
    root->addWidget(m_timing);
    m_progress = new QProgressBar;
    m_progress->hide();
    root->addWidget(m_progress);
    auto* controls = new QHBoxLayout;
    m_run = new QPushButton(tr("Run"));
    m_run->setObjectName("ramRun");
    m_cancelButton = new QPushButton(tr("Cancel / stop"));
    auto* close = new QPushButton(tr("Close"));
    m_cancelButton->setObjectName("ramCancel");
    controls->addStretch();
    controls->addWidget(m_run);
    controls->addWidget(m_cancelButton);
    controls->addWidget(close);
    root->addLayout(controls);
    connect(m_run, &QPushButton::clicked, this, &RAMDialog::run);
    connect(m_cancelButton, &QPushButton::clicked, this, &RAMDialog::cancel);
    connect(close, &QPushButton::clicked, this, &RAMDialog::close);
    connect(m_sample, &QPushButton::clicked, this, &RAMDialog::useTestData);
    connect(m_sampleVideo, &QPushButton::clicked, this,
            &RAMDialog::loadSampleVideo);
    connect(m_download, &QPushButton::clicked, this,
            [this]() { ensureModel(false); });
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &RAMDialog::log);
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
                if (ok && again) run();
            });
    auto& repository = ecvTestDataRepository::instance();
    connect(&repository, &ecvTestDataRepository::downloadFinished, this,
            [this](bool ok, ecvTestDataRepository::Dataset kind) {
                if (kind != ecvTestDataRepository::Dataset::RAM ||
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
            &RAMDialog::log);
    QSettings s;
    s.beginGroup("qRAM");
    m_model->setCurrentIndex(qBound(
            0, s.value("model", aicore_tagging_model_default_index()).toInt(),
            m_model->count() - 1));
    int index = m_device->findData(s.value("device", "auto"));
    if (index >= 0) m_device->setCurrentIndex(index);
    m_threads->setValue(s.value("threads", 8).toInt());
    m_defaults->setChecked(s.value("defaultThresholds", true).toBool());
    m_threshold->setValue(s.value("threshold", .68).toDouble());
    m_epsilon->setValue(s.value("thresholdEpsilon", 0).toDouble());
    m_caption->setChecked(m_caption->isEnabled() &&
                          s.value("caption", true).toBool());
    m_length->setValue(s.value("length", 30).toInt());
    m_beam->setCurrentIndex(qBound(0, s.value("beam", 0).toInt(), 2));
    m_cpuPrecision->setCurrentIndex(
            qBound(0, s.value("cpuPrecision", 0).toInt(), 1));
    m_labels->setText(s.value("labels").toString());
    m_thresholds->setText(s.value("thresholds").toString());
    m_embeddings->setText(s.value("embeddings").toString());
    m_specified->setText(s.value("specifiedTags").toString());
}
RAMDialog::~RAMDialog() { shutdown(); }
void RAMDialog::shutdown() {
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
void RAMDialog::closeEvent(QCloseEvent* event) {
    cancel();
    saveSettings();
    m_closed = true;
    if (m_worker)
        QMetaObject::invokeMethod(
                m_worker, [worker = m_worker]() { worker->release(); },
                Qt::QueuedConnection);
    event->accept();
}
void RAMDialog::showEvent(QShowEvent* event) {
    m_closed = false;
    QDialog::showEvent(event);
}
void RAMDialog::saveSettings() {
    QSettings s;
    s.beginGroup("qRAM");
    s.setValue("model", m_model->currentIndex());
    s.setValue("device", m_device->currentData());
    s.setValue("threads", m_threads->value());
    s.setValue("defaultThresholds", m_defaults->isChecked());
    s.setValue("threshold", m_threshold->value());
    s.setValue("thresholdEpsilon", m_epsilon->value());
    s.setValue("caption", m_caption->isChecked());
    s.setValue("length", m_length->value());
    s.setValue("beam", m_beam->currentIndex());
    s.setValue("cpuPrecision", m_cpuPrecision->currentIndex());
    s.setValue("labels", m_labels->text());
    s.setValue("thresholds", m_thresholds->text());
    s.setValue("embeddings", m_embeddings->text());
    s.setValue("specifiedTags", m_specified->text());
}
void RAMDialog::setDbImages(const QList<QPair<QString, QImage>>& images) {
    m_db = images;
}
void RAMDialog::addFiles() {
    m_inputs->addItems(QFileDialog::getOpenFileNames(
            this, tr("Input images"), {},
            tr("Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)")));
}
void RAMDialog::useTestData() {
    m_closed = false;
    auto kind = ecvTestDataRepository::Dataset::RAM;
    auto& repo = ecvTestDataRepository::instance();
    QStringList demos;
    for (int index = 1; index <= 4; ++index)
        demos << repo.findDatasetFile(kind, QStringLiteral("demo%1.jpg").arg(index));
    if (demos.front().isEmpty() && repo.isDatasetAvailable(kind)) {
        repo.extractDataset(kind);
        demos.clear();
        for (int index = 1; index <= 4; ++index)
            demos << repo.findDatasetFile(kind, QStringLiteral("demo%1.jpg").arg(index));
    }
    if (demos.front().isEmpty()) {
        if (repo.isDownloadInProgress()) {
            log(tr("Another shared sample download is running."));
            return;
        }
        m_waitingForSample = true;
        m_sample->setEnabled(false);
        repo.startDownload(kind);
        return;
    }
    if (std::any_of(demos.cbegin(), demos.cend(),
                    [](const QString& path) { return path.isEmpty(); })) {
        log(tr("The RAM sample archive is missing one or more official demos."));
        return;
    }
    cancel();
    m_inputs->clear();
    m_inputs->addItems(demos);
    m_inputs->setCurrentRow(0);
    m_preview->setPreviewImage(QImage(demos.front()), QSize(480, 280));
    m_tabs->setCurrentIndex(0);
    log(tr("Loaded official demo1–4 images; demo1 is selected."));
}
void RAMDialog::loadSampleVideo() {
    auto& repo = ecvTestDataRepository::instance();
    const auto kind = ecvTestDataRepository::Dataset::RAM;
    QString videoPath = repo.findDatasetFile(kind, "ram_demo1_to_4.avi");
    if (videoPath.isEmpty() && repo.isDatasetAvailable(kind)) {
        repo.extractDataset(kind);
        videoPath = repo.findDatasetFile(kind, "ram_demo1_to_4.avi");
    }
    const ecvAssetIntegrity::Anchor anchor{
            QCryptographicHash::Sha256,
            QByteArray("53c17ba6b53bde9b21660142a1685bfa8ce4a32d6ab9c0cc08f1c57ce944b202")};
    if (videoPath.isEmpty() ||
        !ecvAssetIntegrity::isVerified(videoPath, anchor, 0, false,
                                       ecvAssetIntegrity::OnMiss::DeepVerify,
                                       262706)) {
        log(tr("The verified RAM sample video is unavailable. Use test data to "
               "download the current official archive."));
        return;
    }
    m_video->selectVideoFileSource();
    m_video->setVideoFilePath(videoPath);
    m_tabs->setCurrentIndex(m_tabs->indexOf(m_video->parentWidget()));
    log(tr("Loaded official RAM archive sample video."));
}
QString RAMDialog::modelPath() const {
    std::unique_ptr<char, decltype(&aicore_tagging_free_buffer)> dir(
            aicore_tagging_model_cache_dir(), aicore_tagging_free_buffer);
    auto* e = aicore_tagging_model_at(m_model->currentData().toInt());
    return e && dir ? QDir(QString::fromUtf8(dir.get()))
                              .filePath(QString::fromUtf8(e->filename))
                    : QString();
}
RAMSettings RAMDialog::settings() const {
    RAMSettings s;
    s.model = modelPath();
    s.device = m_device->currentData().toString();
    s.threads = m_threads->value();
    s.caption = m_caption->isChecked();
    s.maxLength = m_length->value();
    s.beamBatch = m_beam->currentData().toInt();
    s.cpuPrecision = m_cpuPrecision->currentData().toInt();
    s.threshold = m_defaults->isChecked() ? -1 : float(m_threshold->value());
    s.thresholdEpsilon = float(m_epsilon->value());
    s.specifiedTags = m_specified->text();
    s.tagList = m_labels->text();
    s.thresholds = m_thresholds->text();
    s.embedding = m_embeddings->text();
    return s;
}
bool RAMDialog::ensureModel(bool runAfter) {
    auto* e = aicore_tagging_model_at(m_model->currentData().toInt());
    if (!e) return false;
    ecvAssetIntegrity::Anchor anchor{QCryptographicHash::Sha256,
                                     QByteArray(e->sha256)};
    if (ecvAssetIntegrity::isVerified(
                modelPath(), anchor, 65536, true,
                ecvAssetIntegrity::OnMiss::CheapChecksOnly, e->size_bytes))
        return true;
    if (m_downloader->isBusy()) return false;
    ecvModelDownloader::Request request;
    request.url = QString::fromUtf8(e->download_url);
    request.destPath = modelPath();
    request.contentAnchor = anchor;
    request.ingestExactSize = e->size_bytes;
    m_runAfterDownload = runAfter;
    m_download->setEnabled(false);
    m_model->setEnabled(false);
    m_run->setEnabled(false);
    m_progress->show();
    m_downloader->download(request);
    return false;
}
void RAMDialog::run() {
    if (m_busy || !m_worker) return;
    m_closed = false;
    if (!ensureModel(true)) return;
    saveSettings();
    if (m_tabs->currentWidget()->findChild<RAMVideoWidget*>() == m_video) {
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
void RAMDialog::runNext() {
    if (m_queue.isEmpty()) {
        m_batch = false;
        m_progress->hide();
        m_tabs->setCurrentIndex(1);
        return;
    }
    auto pair = m_queue.takeFirst();
    submit(pair.second, pair.first, false);
}
void RAMDialog::submit(QImage image, const QString& source, bool frame) {
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
void RAMDialog::completed(const RAMResult& result) {
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
            m_preview->setPreviewImage(result.image, QSize(480, 280));
            const auto timings = result.json.value("timings").toObject();
            m_timing->setText(
                    tr("%1 · %2 · worker %3 ms · inference %4 ms · %5")
                            .arg(result.source, result.device)
                            .arg(result.e2eMs, 0, 'f', 1)
                            .arg(timings.value("inference_ms").toDouble(), 0,
                                 'f', 1)
                            .arg(timings.value("context_reused").toBool()
                                         ? tr("warm session")
                                         : tr("cold start")));
        }
        if (m_batch) m_progress->setValue(++m_batchCompleted);
    }
    if (acknowledge) m_video->acknowledge();
    if (!m_heldFrameSource.isEmpty() && !m_closed) {
        QImage image = std::move(m_heldFrame);
        QString source = std::move(m_heldFrameSource);
        m_heldFrame = {};
        m_heldFrameSource.clear();
        submit(std::move(image), source, true);
    }
    if (current && m_batch) QTimer::singleShot(0, this, &RAMDialog::runNext);
}
void RAMDialog::showResult() {
    auto tags = m_last.json.value("tags").toArray();
    QList<QJsonObject> visible;
    for (auto value : tags) {
        auto t = value.toObject();
        if (m_selected->isChecked() && !t.value("selected").toBool()) continue;
        if (!t.value("name").toString().contains(m_filter->text(),
                                                 Qt::CaseInsensitive))
            continue;
        visible.append(t);
    }
    std::sort(visible.begin(), visible.end(),
              [](const QJsonObject& a, const QJsonObject& b) {
                  return a.value("probability").toDouble() >
                         b.value("probability").toDouble();
              });
    m_table->setRowCount(visible.size());
    for (int i = 0; i < visible.size(); ++i) {
        auto t = visible[i];
        QStringList values{
                t.value("name").toString(),
                QString::number(t.value("probability").toDouble(), 'f', 5),
                QString::number(t.value("threshold").toDouble(), 'f', 3),
                QString::number(t.value("logit").toDouble(), 'f', 5)};
        for (int j = 0; j < 4; ++j)
            m_table->setItem(i, j, new QTableWidgetItem(values[j]));
    }
    QStringList selected;
    for (auto value : m_last.json.value("selected_tags").toArray())
        selected << value.toString();
    m_summary->setText(selected.join(" | "));
    m_captionText->setPlainText(m_last.json.value("caption").toString());
}
void RAMDialog::setBusy(bool busy) {
    m_busy = busy;
    m_tabs->setTabEnabled(2, !busy);
    m_run->setEnabled(!busy && !m_downloader->isBusy());
}
void RAMDialog::resetResults() {
    m_last = RAMResult{};
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
void RAMDialog::recordResult(const QJsonObject& result) {
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
void RAMDialog::cancel() {
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
void RAMDialog::log(const QString& text) {
    CVLog::Print(QStringLiteral("[qRAM] %1").arg(text));
}
void RAMDialog::exportResults() {
    if (m_resultCount == 0 || !m_resultSpool) {
        log(tr("Run inference before exporting."));
        return;
    }
    auto path = QFileDialog::getSaveFileName(
            this, tr("Export results"), "ram-results.json",
            tr("JSON (*.json);;CSV (*.csv);;Text (*.txt)"));
    if (path.isEmpty()) return;
    saveResults(path);
}
bool RAMDialog::saveResults(const QString& path) {
    if (path.isEmpty() || m_resultCount == 0 || !m_resultSpool) return false;
    const bool json = path.endsWith(".json", Qt::CaseInsensitive);
    const bool csv = path.endsWith(".csv", Qt::CaseInsensitive);
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || !m_resultSpool->flush() ||
        !m_resultSpool->seek(0)) {
        log(tr("Export failed: %1").arg(file.errorString()));
        return false;
    }
    auto writeText = [&file](const QString& text) {
        const QByteArray bytes = text.toUtf8();
        return file.write(bytes) == bytes.size();
    };
    bool ok = true, first = true;
    if (json) ok = writeText("[\n");
    if (csv)
        ok = writeText(
                "source,model,device,tag,logit,probability,threshold,selected,"
                "caption\n");
    auto quote = [](QString value) -> QString {
        value.replace(QLatin1Char('"'), QStringLiteral("\"\""));
        return QLatin1Char('"') + value + QLatin1Char('"');
    };
    while (ok && !m_resultSpool->atEnd()) {
        const QByteArray line = m_resultSpool->readLine().trimmed();
        if (line.isEmpty()) continue;
        const QJsonObject result = QJsonDocument::fromJson(line).object();
        if (json) {
            if (!first) ok = file.write(",\n", 2) == 2;
            if (ok) ok = file.write(line) == line.size();
            first = false;
        } else if (csv) {
            for (const auto value : result.value("tags").toArray()) {
                const QJsonObject tag = value.toObject();
                const QString row =
                        quote(result.value("source").toString()) + "," +
                        quote(result.value("model").toString()) + "," +
                        quote(result.value("device").toString()) + "," +
                        quote(tag.value("name").toString()) + "," +
                        QString::number(tag.value("logit").toDouble(), 'g', 9) +
                        "," +
                        QString::number(tag.value("probability").toDouble(),
                                        'g', 9) +
                        "," +
                        QString::number(tag.value("threshold").toDouble(), 'g',
                                        9) +
                        "," + QString::number(tag.value("selected").toBool()) +
                        "," + quote(result.value("caption").toString()) + "\n";
                if (!writeText(row)) {
                    ok = false;
                    break;
                }
            }
        } else {
            QStringList selected;
            for (const auto value : result.value("selected_tags").toArray())
                selected << value.toString();
            ok = writeText(result.value("source").toString() + "\n" +
                           selected.join(" | ") + "\n" +
                           result.value("caption").toString() + "\n\n");
        }
    }
    if (json && ok) ok = writeText("\n]\n");
    ok = ok && file.commit();
    if (!ok)
        log(tr("Export failed: %1").arg(file.errorString()));
    else
        log(tr("Exported %1 result(s) to %2").arg(m_resultCount).arg(path));
    m_resultSpool->seek(m_resultSpool->size());
    return ok;
}
