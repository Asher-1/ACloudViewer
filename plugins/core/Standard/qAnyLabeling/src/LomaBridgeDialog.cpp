// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LomaBridgeDialog.h"

#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QImageReader>
#include <QLabel>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QTableWidget>
#include <QThread>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "aicore/loma_capi.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"

namespace {
QString cacheDirectory() {
    char* path = aicore_loma_model_cache_dir();
    const QString result = path ? QString::fromUtf8(path) : QString();
    std::free(path);
    return result;
}

QString modelPath(const QComboBox* combo) {
    return combo ? combo->currentData().toString() : QString();
}

const aicore_loma_model_entry* modelEntry(const QString& path) {
    const QByteArray name = QFileInfo(path).fileName().toUtf8();
    for (int i = 0; i < aicore_loma_model_count(); ++i) {
        const auto* entry = aicore_loma_model_at(i);
        if (entry && entry->filename && name == entry->filename) return entry;
    }
    return nullptr;
}

QImage loadImage(const QString& path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read().convertToFormat(QImage::Format_RGB888);
}

QPixmap preview(const QImage& image) {
    if (image.isNull()) return {};
    return QPixmap::fromImage(image).scaled(360, 240, Qt::KeepAspectRatio,
                                            Qt::SmoothTransformation);
}

class LomaThread final : public QThread {
    Q_OBJECT
public:
    // Joined here as well as by the dialog: a running QThread must never be
    // destroyed, whichever owner deletes it.
    ~LomaThread() override {
        if (isRunning()) {
            requestInterruption();
            wait();
        }
    }

    QString image0;
    QString image1;
    QString detector;
    QString descriptor;
    QString matcher;
    QString device;
    QVector<LomaCorrespondence> result;
    QString error;
    bool ok = false;

signals:
    void finishedResult(bool ok,
                        const QString& error,
                        const QVector<LomaCorrespondence>& matches);

protected:
    void run() override {
        const QImage source0 = loadImage(image0);
        const QImage source1 = loadImage(image1);
        if (source0.isNull() || source1.isNull()) {
            error = QStringLiteral("LoMa could not decode both images");
            emit finishedResult(false, error, result);
            return;
        }
        // The public DeDoDe descriptor contract requires 784x784 input.
        // Match COLMAP's LoMa preprocessing and convert reviewed keypoints
        // back to the source images before creating annotation geometry.
        const QImage first = source0.scaled(784, 784, Qt::IgnoreAspectRatio,
                                            Qt::SmoothTransformation)
                                     .convertToFormat(QImage::Format_RGB888);
        const QImage second = source1.scaled(784, 784, Qt::IgnoreAspectRatio,
                                             Qt::SmoothTransformation)
                                      .convertToFormat(QImage::Format_RGB888);
        aicore_loma_detector_options* detectorOptions =
                aicore_loma_detector_options_new();
        aicore_loma_descriptor_options* descriptorOptions =
                aicore_loma_descriptor_options_new();
        aicore_loma_matcher_options* matcherOptions =
                aicore_loma_matcher_options_new();
        const QByteArray selectedDevice = device.toUtf8();
        aicore_loma_detector_options_set_device(detectorOptions,
                                                selectedDevice.constData());
        aicore_loma_descriptor_options_set_device(descriptorOptions,
                                                  selectedDevice.constData());
        aicore_loma_matcher_options_set_device(matcherOptions,
                                               selectedDevice.constData());
        aicore_loma_matcher_options_set_min_score(matcherOptions, 0.1);
        const QByteArray detectorPath = detector.toUtf8();
        const QByteArray descriptorPath = descriptor.toUtf8();
        const QByteArray matcherPath = matcher.toUtf8();
        aicore_loma_detector_ctx* detectorContext = aicore_loma_detector_load(
                detectorPath.constData(), detectorOptions);
        aicore_loma_descriptor_ctx* descriptorContext =
                aicore_loma_descriptor_load(descriptorPath.constData(),
                                            descriptorOptions);
        aicore_loma_matcher_ctx* matcherContext = aicore_loma_matcher_load(
                matcherPath.constData(), matcherOptions);
        aicore_loma_detector_options_free(detectorOptions);
        aicore_loma_descriptor_options_free(descriptorOptions);
        aicore_loma_matcher_options_free(matcherOptions);
        if (!detectorContext || !descriptorContext || !matcherContext) {
            error = QStringLiteral("LoMa model loading failed");
            if (detectorContext)
                error = QString::fromUtf8(
                        aicore_loma_detector_last_error(detectorContext));
            if (descriptorContext &&
                error == QStringLiteral("LoMa model loading failed"))
                error = QString::fromUtf8(
                        aicore_loma_descriptor_last_error(descriptorContext));
            if (matcherContext &&
                error == QStringLiteral("LoMa model loading failed"))
                error = QString::fromUtf8(
                        aicore_loma_matcher_last_error(matcherContext));
            if (detectorContext) aicore_loma_detector_free(detectorContext);
            if (descriptorContext)
                aicore_loma_descriptor_free(descriptorContext);
            if (matcherContext) aicore_loma_matcher_free(matcherContext);
            emit finishedResult(false, error, result);
            return;
        }
        auto detect = [detectorContext](const QImage& image,
                                        aicore_loma_detected_features* out) {
            aicore_loma_rgb_image view{image.constBits(), image.width(),
                                       image.height(), image.bytesPerLine()};
            return aicore_loma_detector_run(detectorContext, &view, out);
        };
        aicore_loma_detected_features features0{};
        aicore_loma_detected_features features1{};
        if (detect(first, &features0) != 0 || detect(second, &features1) != 0) {
            error = QString::fromUtf8(
                    aicore_loma_detector_last_error(detectorContext));
            aicore_loma_detected_features_free(&features0);
            aicore_loma_detected_features_free(&features1);
            aicore_loma_detector_free(detectorContext);
            aicore_loma_descriptor_free(descriptorContext);
            aicore_loma_matcher_free(matcherContext);
            emit finishedResult(false, error, result);
            return;
        }
        auto describe = [descriptorContext](
                                const QImage& image,
                                const aicore_loma_detected_features& detected,
                                aicore_loma_described_features* out) {
            aicore_loma_rgb_image view{image.constBits(), image.width(),
                                       image.height(), image.bytesPerLine()};
            return aicore_loma_descriptor_run(
                    descriptorContext, &view, detected.keypoints,
                    detected.count, detected.image_width, detected.image_height,
                    out);
        };
        aicore_loma_described_features described0{};
        aicore_loma_described_features described1{};
        if (describe(first, features0, &described0) != 0 ||
            describe(second, features1, &described1) != 0) {
            error = QString::fromUtf8(
                    aicore_loma_descriptor_last_error(descriptorContext));
            aicore_loma_detected_features_free(&features0);
            aicore_loma_detected_features_free(&features1);
            aicore_loma_described_features_free(&described0);
            aicore_loma_described_features_free(&described1);
            aicore_loma_detector_free(detectorContext);
            aicore_loma_descriptor_free(descriptorContext);
            aicore_loma_matcher_free(matcherContext);
            emit finishedResult(false, error, result);
            return;
        }
        aicore_loma_features input0{
                features0.keypoints,    features0.count,
                described0.descriptors, described0.descriptor_dim,
                first.width(),          first.height()};
        aicore_loma_features input1{
                features1.keypoints,    features1.count,
                described1.descriptors, described1.descriptor_dim,
                second.width(),         second.height()};
        aicore_loma_match* matches = nullptr;
        int32_t count = 0;
        if (aicore_loma_matcher_run(matcherContext, &input0, &input1, &matches,
                                    &count) != 0) {
            error = QString::fromUtf8(
                    aicore_loma_matcher_last_error(matcherContext));
        } else {
            result.reserve(count);
            for (int32_t i = 0; i < count; ++i) {
                const auto& match = matches[i];
                if (match.idx0 < 0 || match.idx0 >= features0.count ||
                    match.idx1 < 0 || match.idx1 >= features1.count ||
                    !std::isfinite(match.score))
                    continue;
                result.append(
                        {QPointF(features0.keypoints[match.idx0].x *
                                         source0.width() / first.width(),
                                 features0.keypoints[match.idx0].y *
                                         source0.height() / first.height()),
                         QPointF(features1.keypoints[match.idx1].x *
                                         source1.width() / second.width(),
                                 features1.keypoints[match.idx1].y *
                                         source1.height() / second.height()),
                         match.score});
            }
            ok = true;
        }
        aicore_loma_free_matches(matches);
        aicore_loma_detected_features_free(&features0);
        aicore_loma_detected_features_free(&features1);
        aicore_loma_described_features_free(&described0);
        aicore_loma_described_features_free(&described1);
        aicore_loma_detector_free(detectorContext);
        aicore_loma_descriptor_free(descriptorContext);
        aicore_loma_matcher_free(matcherContext);
        emit finishedResult(ok, error, result);
    }
};
}  // namespace

LomaBridgeDialog::LomaBridgeDialog(QWidget* parent) : QDialog(parent) {
    qRegisterMetaType<LomaCorrespondence>("LomaCorrespondence");
    qRegisterMetaType<QVector<LomaCorrespondence>>(
            "QVector<LomaCorrespondence>");
    setWindowTitle(tr("LoMa two-image annotation bridge"));
    resize(900, 680);
    auto* root = new QVBoxLayout(this);
    auto* paths = new QGridLayout;
    auto* browse0 = new QPushButton(tr("Image A..."), this);
    auto* browse1 = new QPushButton(tr("Image B..."), this);
    paths->addWidget(browse0, 0, 0);
    paths->addWidget(browse1, 1, 0);
    auto* image0Label = new QLabel(this);
    auto* image1Label = new QLabel(this);
    image0Label->setObjectName(QStringLiteral("LomaImage0Path"));
    image1Label->setObjectName(QStringLiteral("LomaImage1Path"));
    paths->addWidget(image0Label, 0, 1);
    paths->addWidget(image1Label, 1, 1);
    root->addLayout(paths);
    auto* models = new QGridLayout;
    m_detector = new QComboBox(this);
    m_detector->setObjectName(QStringLiteral("LomaDetectorModel"));
    m_descriptor = new QComboBox(this);
    m_descriptor->setObjectName(QStringLiteral("LomaDescriptorModel"));
    m_matcher = new QComboBox(this);
    m_matcher->setObjectName(QStringLiteral("LomaMatcherModel"));
    m_device = new QComboBox(this);
    m_device->setObjectName(QStringLiteral("LomaBackend"));
    m_device->addItems({QStringLiteral("auto"), QStringLiteral("cpu"),
                        QStringLiteral("cuda"), QStringLiteral("vulkan"),
                        QStringLiteral("metal")});
    models->addWidget(new QLabel(tr("Detector"), this), 0, 0);
    models->addWidget(m_detector, 0, 1);
    models->addWidget(new QLabel(tr("Descriptor"), this), 1, 0);
    models->addWidget(m_descriptor, 1, 1);
    models->addWidget(new QLabel(tr("Matcher"), this), 2, 0);
    models->addWidget(m_matcher, 2, 1);
    models->addWidget(new QLabel(tr("Backend"), this), 3, 0);
    models->addWidget(m_device, 3, 1);
    root->addLayout(models);
    auto* previews = new QHBoxLayout;
    m_preview0 = new QLabel(this);
    m_preview1 = new QLabel(this);
    m_preview0->setMinimumSize(360, 240);
    m_preview1->setMinimumSize(360, 240);
    m_preview0->setAlignment(Qt::AlignCenter);
    m_preview1->setAlignment(Qt::AlignCenter);
    previews->addWidget(m_preview0);
    previews->addWidget(m_preview1);
    root->addLayout(previews);
    m_matches = new QTableWidget(this);
    m_matches->setColumnCount(4);
    m_matches->setHorizontalHeaderLabels(
            {tr("Use"), tr("Image A"), tr("Image B"), tr("Score")});
    m_matches->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    root->addWidget(m_matches, 1);
    m_status =
            new QLabel(tr("Choose two images and verified LoMa models"), this);
    root->addWidget(m_status);
    auto* buttons = new QHBoxLayout;
    m_run = new QPushButton(tr("Run LoMa"), this);
    m_accept = new QPushButton(tr("Accept selected matches"), this);
    m_accept->setEnabled(false);
    auto* cancel = new QPushButton(tr("Cancel"), this);
    buttons->addWidget(m_run);
    buttons->addStretch();
    buttons->addWidget(m_accept);
    buttons->addWidget(cancel);
    root->addLayout(buttons);
    connect(browse0, &QPushButton::clicked, this,
            &LomaBridgeDialog::browseImage0);
    connect(browse1, &QPushButton::clicked, this,
            &LomaBridgeDialog::browseImage1);
    connect(m_run, &QPushButton::clicked, this, &LomaBridgeDialog::runMatching);
    connect(m_accept, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                m_status->setText(ecvModelDownloader::formatDownloadProgress(
                        received, total));
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                if (!ok) {
                    m_pendingModels.clear();
                    setBusy(false);
                    m_status->setText(
                            tr("LoMa model download or verification failed: %1")
                                    .arg(QFileInfo(path).fileName()));
                    return;
                }
                if (!m_pendingModels.isEmpty()) m_pendingModels.removeFirst();
                if (m_pendingModels.isEmpty()) {
                    setBusy(false);
                    runMatching();
                } else {
                    downloadNextModel();
                }
            });
    refreshModelLists();
}

LomaBridgeDialog::~LomaBridgeDialog() {
    if (m_downloader && m_downloader->isBusy()) {
        m_downloader->disconnect(this);
        m_downloader->cancel();
    }
    if (m_thread) {
        m_thread->requestInterruption();
        m_thread->wait();
        delete m_thread;
    }
}

void LomaBridgeDialog::setImage0(const QString& path) {
    m_image0Path = path;
    m_image0 = loadImage(path);
    updatePreview();
}

void LomaBridgeDialog::setImage1(const QString& path) {
    m_image1Path = path;
    m_image1 = loadImage(path);
    updatePreview();
}

QVector<LomaCorrespondence> LomaBridgeDialog::acceptedMatches() const {
    QVector<LomaCorrespondence> selected;
    for (int row = 0; row < m_matches->rowCount() && row < m_result.size();
         ++row) {
        auto* item = m_matches->item(row, 0);
        if (item && item->checkState() == Qt::Checked)
            selected.append(m_result[row]);
    }
    return selected;
}

void LomaBridgeDialog::browseImage0() {
    const QString path = QFileDialog::getOpenFileName(this, tr("Image A"));
    if (!path.isEmpty()) setImage0(path);
}

void LomaBridgeDialog::browseImage1() {
    const QString path = QFileDialog::getOpenFileName(this, tr("Image B"));
    if (!path.isEmpty()) setImage1(path);
}

void LomaBridgeDialog::runMatching() {
    if (m_image0.isNull() || m_image1.isNull()) {
        m_status->setText(tr("Choose two readable images first"));
        return;
    }
    if (m_thread || m_downloader->isBusy()) return;
    for (QComboBox* combo : {m_detector, m_descriptor, m_matcher}) {
        if (!combo || !modelEntry(modelPath(combo))) {
            m_status->setText(
                    tr("Select a published LoMa model for each role"));
            return;
        }
        const QString path = modelPath(combo);
        const ecvAssetIntegrity::Anchor anchor{
                QCryptographicHash::Sha256,
                ecvAssetIntegrity::PinnedDigest(QFileInfo(path).fileName())};
        if (!ecvAssetIntegrity::isVerified(path, anchor, 1024 * 1024, true))
            m_pendingModels.append(path);
    }
    if (!m_pendingModels.isEmpty()) {
        setBusy(true);
        downloadNextModel();
        return;
    }
    auto* thread = new LomaThread;
    thread->image0 = m_image0Path;
    thread->image1 = m_image1Path;
    thread->detector = modelPath(m_detector);
    thread->descriptor = modelPath(m_descriptor);
    thread->matcher = modelPath(m_matcher);
    thread->device = m_device->currentText();
    m_thread = thread;
    connect(thread, &LomaThread::finishedResult, this,
            &LomaBridgeDialog::onFinished, Qt::QueuedConnection);
    connect(thread, &QThread::finished, this, [this, thread]() {
        thread->deleteLater();
        m_thread = nullptr;
    });
    setBusy(true);
    thread->start();
}

void LomaBridgeDialog::downloadNextModel() {
    if (m_pendingModels.isEmpty()) return;
    const QString path = m_pendingModels.first();
    const auto* entry = modelEntry(path);
    if (!entry || !entry->download_url) {
        m_pendingModels.clear();
        setBusy(false);
        m_status->setText(tr("LoMa model URL is unavailable"));
        return;
    }
    QDir().mkpath(QFileInfo(path).absolutePath());
    ecvModelDownloader::Request request;
    request.url = QString::fromUtf8(entry->download_url);
    request.destPath = path;
    request.minBytes = 1024 * 1024;
    request.contentAnchor = {
            QCryptographicHash::Sha256,
            ecvAssetIntegrity::PinnedDigest(QFileInfo(path).fileName())};
    m_status->setText(tr("Downloading %1").arg(QFileInfo(path).fileName()));
    m_downloader->download(request);
}

void LomaBridgeDialog::onFinished(bool ok,
                                  const QString& error,
                                  const QVector<LomaCorrespondence>& matches) {
    setBusy(false);
    if (!ok) {
        m_status->setText(error.isEmpty() ? tr("LoMa matching failed") : error);
        emit matchingFinished(false, 0);
        return;
    }
    m_result = matches;
    m_matches->setRowCount(0);
    for (int i = 0; i < m_result.size(); ++i) {
        const auto& match = m_result.at(i);
        m_matches->insertRow(i);
        auto* use = new QTableWidgetItem;
        use->setCheckState(Qt::Checked);
        m_matches->setItem(i, 0, use);
        m_matches->setItem(i, 1,
                           new QTableWidgetItem(
                                   QStringLiteral("(%1, %2)")
                                           .arg(match.image0.x(), 0, 'f', 1)
                                           .arg(match.image0.y(), 0, 'f', 1)));
        m_matches->setItem(i, 2,
                           new QTableWidgetItem(
                                   QStringLiteral("(%1, %2)")
                                           .arg(match.image1.x(), 0, 'f', 1)
                                           .arg(match.image1.y(), 0, 'f', 1)));
        m_matches->setItem(
                i, 3,
                new QTableWidgetItem(QString::number(match.score, 'f', 4)));
    }
    m_status->setText(tr("%1 correspondences; select the points to import")
                              .arg(m_result.size()));
    m_accept->setEnabled(!m_result.isEmpty());
    emit matchingFinished(true, m_result.size());
}

void LomaBridgeDialog::refreshModelLists() {
    const QString cache = cacheDirectory();
    for (int i = 0; i < aicore_loma_model_count(); ++i) {
        const auto* entry = aicore_loma_model_at(i);
        if (!entry || !entry->filename) continue;
        const QString path =
                QDir(cache).filePath(QString::fromUtf8(entry->filename));
        auto* combo = entry->role == AICORE_LOMA_MODEL_ROLE_DETECTOR
                              ? m_detector
                      : entry->role == AICORE_LOMA_MODEL_ROLE_DESCRIPTOR
                              ? m_descriptor
                              : m_matcher;
        combo->addItem(
                QString::fromUtf8(entry->display_name ? entry->display_name
                                                      : entry->filename),
                path);
    }
}

void LomaBridgeDialog::updatePreview() {
    if (m_preview0) m_preview0->setPixmap(preview(m_image0));
    if (m_preview1) m_preview1->setPixmap(preview(m_image1));
}

void LomaBridgeDialog::setBusy(bool busy) {
    m_run->setEnabled(!busy);
    m_accept->setEnabled(!busy && !m_result.isEmpty());
    if (busy)
        m_status->setText(
                tr("Running LoMa detector, descriptor and matcher..."));
}

#include "LomaBridgeDialog.moc"
