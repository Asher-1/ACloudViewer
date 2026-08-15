// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceRegistryWidget.h"

#include <cvFileDialog.h>

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QImageReader>
#include <QMap>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QSettings>
#include <QVBoxLayout>

#include "FaceDetectEmbedHelpers.h"
#include "FaceDetectModelContext.h"
#include "FaceDetectTestData.h"
#include "FaceDetectUiHelpers.h"

#ifdef AICore_ENABLED
#include "aicore/facedetect_capi.h"
#endif

#include <CVLog.h>

#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

namespace {

constexpr int kAuthPreviewSize = 96;

}  // namespace

FaceRegistryWidget::FaceRegistryWidget(QWidget* parent) : QWidget(parent) {
    auto* main = new QVBoxLayout(this);
    FaceDetectUi::setupCompactMainLayout(main);

    m_testDataBtn = new QPushButton(tr("\U0001f9ea  Try sample data"), this);
    m_testDataBtn->setToolTip(tr(
            "Download FriendsFaces sample pack, register gallery identities, "
            "then fill registry fields.\n\n"
            "Downloads the FriendsFaces sample pack, registers six cast "
            "members "
            "with curated gallery frontals (e.g. Joey00030.jpg), fills the "
            "registry DB path, and sets the group-photo probe for "
            "authentication."));
    // Prominent teal accent — consistent with qFreeSplatter / batch tab.
    m_testDataBtn->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold; border: none; border-radius: 4px; padding: 5px 12px; }"
            "QPushButton:hover { background: #00796b; }"
            "QPushButton:pressed { background: #00695c; }"
            "QPushButton:disabled { background: #b2dfdb; color: #e0f2f1; }");
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &FaceRegistryWidget::testDataRequested);

    auto* dbGroup = new QGroupBox(tr("Face database"), this);
    auto* dbLayout = new QGridLayout(dbGroup);
    FaceDetectUi::setupTwoColumnFormGrid(dbLayout);
    FaceDetectUi::tightenGroupBox(dbGroup);
    m_registryPathEdit = new QLineEdit(dbGroup);
    m_registryPathEdit->setPlaceholderText(
            tr("SQLite registry path (face_registry.db)"));
    auto* browseDb = FaceDetectUi::makeBrowseButton(tr("Browse…"), dbGroup);
    connect(browseDb, &QPushButton::clicked, this,
            &FaceRegistryWidget::onBrowseRegistryDb);
    connect(m_registryPathEdit, &QLineEdit::editingFinished, this, [this]() {
        if (!m_registryPathEdit) return;
        const QString path = m_registryPathEdit->text().trimmed();
        if (path.isEmpty()) return;
        setRegistryPath(path, true);
        saveSettings();
    });
    dbLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Database:")), 0, 0);
    dbLayout->addWidget(m_registryPathEdit, 0, 1, 1, 2);
    dbLayout->addWidget(browseDb, 0, 3);

    m_modelCombo = new QComboBox(dbGroup);
    m_deviceCombo = new QComboBox(dbGroup);
    m_threadsSpin = new QSpinBox(dbGroup);
    m_threadsSpin->setRange(0, 128);
    m_threadsSpin->setSpecialValueText(tr("Auto"));
    FaceDetectUi::makeCompactSpin(m_threadsSpin);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (m_syncingModelControls) return;
                updateModelPathFromCombo();
                emit modelSelectionChanged(modelFilename());
            });
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (m_syncingModelControls) return;
                m_device = deviceId();
                emit deviceSelectionChanged(m_device);
            });
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
                if (m_syncingModelControls) return;
                m_threadCount = value;
                emit threadCountChanged(value);
            });
    dbLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Detector GGUF:")), 1,
                        0);
    dbLayout->addWidget(m_modelCombo, 1, 1);
    dbLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Device:")), 2, 0);
    dbLayout->addWidget(m_deviceCombo, 2, 1);
    dbLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Threads:")), 2, 2);
    dbLayout->addWidget(m_threadsSpin, 2, 3, Qt::AlignLeft);

    m_dbStatusLabel = new QLabel(tr("—"), dbGroup);
    m_dbStatusLabel->setWordWrap(true);
    m_dbStatusLabel->setSizePolicy(QSizePolicy::Preferred,
                                   QSizePolicy::Maximum);
    m_dbStatusLabel->setStyleSheet(
            "color: palette(mid); font-size: 11px; padding: 0;");
    dbLayout->addWidget(m_dbStatusLabel, 3, 0, 1, 4);

    m_progressBar = new QProgressBar(dbGroup);
    m_progressBar->setFixedHeight(16);
    m_progressBar->setTextVisible(false);
    m_progressBar->setMaximum(100);
    m_progressBar->setValue(0);
    m_progressBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    dbLayout->addWidget(m_progressBar, 4, 0, 1, 4);
    main->addWidget(dbGroup);

    auto* regGroup = new QGroupBox(tr("Register face"), this);
    auto* regLayout = new QGridLayout(regGroup);
    FaceDetectUi::setupTwoColumnFormGrid(regLayout);
    FaceDetectUi::tightenGroupBox(regGroup);

    m_registerImagePath = new QLineEdit(regGroup);
    m_registerImagePath->setPlaceholderText(
            tr("Image path (or capture from Live tab)"));
    auto* browseReg = FaceDetectUi::makeBrowseButton(tr("Browse…"), regGroup);
    connect(browseReg, &QPushButton::clicked, this,
            &FaceRegistryWidget::onBrowseRegisterImage);
    regLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Image:")), 0, 0);
    regLayout->addWidget(m_registerImagePath, 0, 1, 1, 2);
    regLayout->addWidget(browseReg, 0, 3);

    m_nameEdit = new QLineEdit(regGroup);
    m_nameEdit->setPlaceholderText(tr("Person name"));
    m_nameEdit->setMaximumWidth(320);
    regLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Name:")), 1, 0);
    regLayout->addWidget(m_nameEdit, 1, 1, Qt::AlignLeft);

    auto* registerBtn = new QPushButton(tr("Register to database"), regGroup);
    m_registerBtn = registerBtn;
    connect(registerBtn, &QPushButton::clicked, this,
            &FaceRegistryWidget::onRegister);
    regLayout->addWidget(registerBtn, 1, 2, 1, 2, Qt::AlignRight);
    main->addWidget(regGroup);

    auto* authGroup = new QGroupBox(tr("Authenticate"), this);
    auto* authLayout = new QGridLayout(authGroup);
    FaceDetectUi::setupTwoColumnFormGrid(authLayout);
    FaceDetectUi::tightenGroupBox(authGroup);
    m_authImagePath = new QLineEdit(authGroup);
    auto* browseAuth = FaceDetectUi::makeBrowseButton(tr("Browse…"), authGroup);
    connect(browseAuth, &QPushButton::clicked, this,
            &FaceRegistryWidget::onBrowseAuthImage);
    authLayout->addWidget(FaceDetectUi::makeFormLabel(tr("Probe image:")), 0,
                          0);
    authLayout->addWidget(m_authImagePath, 0, 1, 1, 2);
    authLayout->addWidget(browseAuth, 0, 3);

    m_authThreshold = new QDoubleSpinBox(authGroup);
    m_authThreshold->setRange(0.05, 1.0);
    m_authThreshold->setSingleStep(0.01);
    m_authThreshold->setValue(0.65);
    m_authThreshold->setToolTip(
            tr("Maximum cosine distance for a match (lower = stricter)."));
    m_minDetectionScoreSpin = FaceDetectUi::makeMinDetectionScoreSpin(
            authGroup,
            tr("Minimum detector confidence when extracting embeddings for "
               "register / authenticate."));
    FaceDetectUi::makeCompactDoubleSpin(m_authThreshold);

    m_exportAuthToDbCheck =
            new QCheckBox(tr("Export auth viz to DB tree"), authGroup);
    m_exportAuthToDbCheck->setChecked(false);
    m_exportAuthToDbCheck->setToolTip(
            tr("When enabled, annotated probe image with match labels is added "
               "to the DB tree after authentication."));

    auto* threshRow = new QHBoxLayout;
    threshRow->setContentsMargins(0, 0, 0, 0);
    threshRow->setSpacing(6);
    threshRow->addWidget(FaceDetectUi::makeFormLabel(tr("Match dist:")));
    threshRow->addWidget(m_authThreshold);
    threshRow->addSpacing(8);
    threshRow->addWidget(m_exportAuthToDbCheck, 1);
    threshRow->addSpacing(8);
    threshRow->addWidget(FaceDetectUi::makeFormLabel(tr("Min det score:")));
    threshRow->addWidget(m_minDetectionScoreSpin);
    authLayout->addLayout(threshRow, 1, 0, 1, 4);

    auto* authBtn = new QPushButton(tr("Run authentication"), authGroup);
    m_authBtn = authBtn;
    connect(authBtn, &QPushButton::clicked, this,
            &FaceRegistryWidget::onAuthenticate);

    m_authPreviewLabel = new ecvClickableImageLabel(authGroup);
    m_authPreviewLabel->setFixedSize(kAuthPreviewSize, kAuthPreviewSize);
    m_authPreviewLabel->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_authPreviewLabel->setText(tr("Probe"));

    auto* authActionRow = new QHBoxLayout;
    authActionRow->setContentsMargins(0, 0, 0, 0);
    authActionRow->setSpacing(8);
    authActionRow->addWidget(ecvClickableImageLabel::wrapWithTapToPreviewHint(
            m_authPreviewLabel, authGroup));
    authActionRow->addWidget(authBtn);
    authActionRow->addStretch();
    authLayout->addLayout(authActionRow, 2, 0, 1, 4);

    m_authResultLabel = new QPlainTextEdit(authGroup);
    m_authResultLabel->setReadOnly(true);
    m_authResultLabel->setUndoRedoEnabled(false);
    m_authResultLabel->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    m_authResultLabel->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_authResultLabel->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_authResultLabel->setPlainText(tr("—"));
    m_authResultLabel->setFixedHeight(88);
    m_authResultLabel->setStyleSheet(
            "padding: 2px 4px; border-radius: 3px; background: palette(base); "
            "border: 1px solid palette(mid); font-size: 11px;");
    authLayout->addWidget(m_authResultLabel, 3, 0, 1, 4);
    main->addWidget(authGroup);

    connect(m_authImagePath, &QLineEdit::textChanged, this,
            [this](const QString&) { updateAuthPreview(); });

    auto* listGroup = new QGroupBox(tr("Registered faces"), this);
    FaceDetectUi::tightenGroupBox(listGroup);
    auto* listLayout = new QVBoxLayout(listGroup);
    listLayout->setContentsMargins(6, 4, 6, 4);
    listLayout->setSpacing(3);
    m_entryList = new QListWidget(listGroup);
    m_entryList->setAlternatingRowColors(true);
    m_entryList->setIconSize(QSize(40, 40));
    m_entryList->setUniformItemSizes(true);
    m_entryList->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    m_entryList->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_entryList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_entryList->setFixedHeight(168);
    listLayout->addWidget(m_entryList);

    auto* listBtnRow = new QHBoxLayout;
    auto* removeBtn = new QPushButton(tr("Remove selected"), listGroup);
    auto* clearBtn = new QPushButton(tr("Clear all"), listGroup);
    connect(removeBtn, &QPushButton::clicked, this,
            &FaceRegistryWidget::onRemove);
    connect(clearBtn, &QPushButton::clicked, this,
            &FaceRegistryWidget::onClear);
    listBtnRow->addWidget(m_testDataBtn);
    listBtnRow->addWidget(removeBtn);
    listBtnRow->addWidget(clearBtn);
    listBtnRow->addStretch();
    listLayout->addLayout(listBtnRow);
    main->addWidget(listGroup);
    connect(m_authThreshold,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double value) {
                emit authThresholdChanged(static_cast<float>(value));
            });
    connect(m_minDetectionScoreSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double value) {
                m_minDetectionScore = static_cast<float>(value);
                emit minDetectionScoreChanged(static_cast<float>(value));
            });
}

void FaceRegistryWidget::setMinDetectionScore(float score) {
    m_minDetectionScore = score;
    if (m_minDetectionScoreSpin) {
        m_minDetectionScoreSpin->blockSignals(true);
        m_minDetectionScoreSpin->setValue(score);
        m_minDetectionScoreSpin->blockSignals(false);
    }
}

float FaceRegistryWidget::minDetectionScore() const {
    return m_minDetectionScoreSpin
                   ? static_cast<float>(m_minDetectionScoreSpin->value())
                   : m_minDetectionScore;
}

void FaceRegistryWidget::showVerifySummary(int faceCount,
                                           int matchedCount,
                                           float threshold) {
    if (!m_authResultLabel || faceCount <= 0) return;
    m_authResultLabel->setPlainText(
            tr("Detected %1 face(s), %2 matched (threshold %3)")
                    .arg(faceCount)
                    .arg(matchedCount)
                    .arg(threshold, 0, 'f', 2));
    if (matchedCount > 0) {
        m_authResultLabel->setStyleSheet(
                "padding: 8px; border-radius: 4px; background: #ecfdf5; "
                "border: 1px solid #a7f3d0; color: #065f46;");
    } else {
        m_authResultLabel->setStyleSheet(
                "padding: 8px; border-radius: 4px; background: #fef2f2; "
                "border: 1px solid #fecaca; color: #991b1b;");
    }
}

void FaceRegistryWidget::setThreads(int threads) {
    m_threadCount = threads;
#ifdef AICore_ENABLED
    m_embedContext.release();
#endif
    if (m_threadsSpin) {
        m_syncingModelControls = true;
        m_threadsSpin->setValue(threads);
        m_syncingModelControls = false;
    }
}

void FaceRegistryWidget::fillFriendsTestBundleFields(
        const FaceDetectFriendsBundle& bundle) {
    if (m_nameEdit && !bundle.registerName.isEmpty()) {
        m_nameEdit->setText(bundle.registerName);
    }
    if (m_registerImagePath && !bundle.registerImage.isEmpty()) {
        m_registerImagePath->setText(bundle.registerImage);
    }
    if (m_authImagePath && !bundle.authProbeImage.isEmpty()) {
        m_authImagePath->setText(bundle.authProbeImage);
        updateAuthPreview();
    }
}

int FaceRegistryWidget::registerGalleryEntries(
        const QVector<FaceDetectGalleryEntry>& entries) {
    if (entries.isEmpty()) return 0;
    if (!m_store.isOpen()) {
        emit logMessage(
                tr("[Registry] Database not open — choose a valid path."));
        return 0;
    }

    QMap<QString, QStringList> grouped;
    for (const FaceDetectGalleryEntry& entry : entries) {
        if (entry.name.isEmpty() || entry.imagePath.isEmpty()) continue;
        grouped[entry.name].append(entry.imagePath);
    }

    constexpr float kGalleryMinScore = 0.0f;
    int added = 0;
    for (auto it = grouped.constBegin(); it != grouped.constEnd(); ++it) {
        const QString& name = it.key();
        bool alreadyRegistered = false;
        for (const FaceRegistryEntry& existing : m_store.entries()) {
            if (existing.name.compare(name, Qt::CaseInsensitive) == 0) {
                alreadyRegistered = true;
                break;
            }
        }
        if (alreadyRegistered) continue;

        bool registered = false;
        for (const QString& imagePath : it.value()) {
            if (registerPersonFromImage(name, imagePath, kGalleryMinScore)) {
                registered = true;
                ++added;
                break;
            }
        }
        if (registered) continue;

        const QString galleryPersonDir =
                QFileInfo(it.value().constFirst()).absolutePath();
        if (QDir(galleryPersonDir).exists()) {
            const QStringList filters = {
                    QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
                    QStringLiteral("*.png"), QStringLiteral("*.webp")};
            for (const QString& fn :
                 QDir(galleryPersonDir).entryList(filters, QDir::Files)) {
                if (registerPersonFromImage(name,
                                            QDir(galleryPersonDir).filePath(fn),
                                            kGalleryMinScore)) {
                    ++added;
                    break;
                }
            }
        }
    }

    if (added > 0) {
        refreshList();
        saveSettings();
        emit registryChanged();
        emit logMessage(
                tr("[Registry] Registered %1 gallery identities.").arg(added));
    }
    return added;
}

bool FaceRegistryWidget::registerPersonFromImage(const QString& name,
                                                 const QString& imagePath,
                                                 float minDetectionScore) {
    if (name.isEmpty() || imagePath.isEmpty()) return false;
    if (!m_store.isOpen()) return false;

    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    const QImage thumb = reader.read();

    std::vector<float> emb;
    int dim = 0;
    QString err;
    if (!embedImage(imagePath, &emb, &dim, &err, minDetectionScore)) {
        emit logMessage(
                tr("[Registry] %1 (%2): %3")
                        .arg(name, QFileInfo(imagePath).fileName(), err));
        return false;
    }

    FaceRegistryEntry e;
    e.name = name;
    e.modelFile = QFileInfo(resolveModelPath()).fileName();
    e.embedDim = dim;
    e.embedding = std::move(emb);
    e.thumbnail = thumb;
    if (!m_store.addEntry(std::move(e))) {
        emit logMessage(tr("[Registry] Failed to save '%1'.").arg(name));
        return false;
    }
    emit logMessage(tr("[Registry] Registered '%1'.").arg(name));
    return true;
}

QString FaceRegistryWidget::registryPathForModel(const QString& baseDir,
                                                 const QString& modelFilename) {
    const QString stem = QFileInfo(modelFilename).completeBaseName();
    const QString safeStem = stem.isEmpty() ? QStringLiteral("default") : stem;
    return QDir(baseDir).filePath(
            QStringLiteral("face_registry_%1.db").arg(safeStem));
}

void FaceRegistryWidget::updateAuthPreview() {
    if (!m_authPreviewLabel || !m_authImagePath) return;
    const QString path = m_authImagePath->text().trimmed();
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        m_authPreviewLabel->clearPreview();
        m_authPreviewLabel->setText(tr("Probe"));
        return;
    }
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage img = reader.read();
    if (img.isNull()) {
        m_authPreviewLabel->clearPreview();
        m_authPreviewLabel->setText(tr("?"));
        return;
    }
    m_authPreviewLabel->setPreviewImage(img, kAuthPreviewSize);
}

void FaceRegistryWidget::updateModelPathFromCombo() {
    if (!m_modelCombo) return;
    const QString fn = m_modelCombo->currentData().toString();
    if (fn.isEmpty() || fn == QStringLiteral("CUSTOM")) return;
    m_modelPath = FaceDetectEmbed::modelCacheDir() + QLatin1Char('/') + fn;
}

QString FaceRegistryWidget::modelFilename() const {
    return m_modelCombo ? m_modelCombo->currentData().toString() : QString();
}

QString FaceRegistryWidget::deviceId() const {
    return m_deviceCombo ? m_deviceCombo->currentData().toString() : m_device;
}

int FaceRegistryWidget::threadCount() const {
    return m_threadsSpin ? m_threadsSpin->value() : m_threadCount;
}

bool FaceRegistryWidget::exportAuthResultToDb() const {
    return m_exportAuthToDbCheck && m_exportAuthToDbCheck->isChecked();
}

void FaceRegistryWidget::rebuildModelCombo(const QStringList& labels,
                                           const QStringList& filenames,
                                           const QString& currentFilename) {
    if (!m_modelCombo || labels.size() != filenames.size()) return;
    m_syncingModelControls = true;
    m_modelCombo->clear();
    int selectIndex = 0;
    for (int i = 0; i < labels.size(); ++i) {
        m_modelCombo->addItem(labels.at(i), filenames.at(i));
        if (filenames.at(i) == currentFilename) selectIndex = i;
    }
    m_modelCombo->setCurrentIndex(selectIndex);
    updateModelPathFromCombo();
    m_syncingModelControls = false;
}

void FaceRegistryWidget::rebuildDeviceCombo(
        const QComboBox* sourceDeviceCombo) {
    if (!m_deviceCombo || !sourceDeviceCombo) return;
    m_syncingModelControls = true;
    m_deviceCombo->clear();
    for (int i = 0; i < sourceDeviceCombo->count(); ++i) {
        m_deviceCombo->addItem(sourceDeviceCombo->itemText(i),
                               sourceDeviceCombo->itemData(i));
    }
    m_deviceCombo->setCurrentIndex(sourceDeviceCombo->currentIndex());
    m_device = deviceId();
    m_syncingModelControls = false;
}

void FaceRegistryWidget::syncModelControlsFrom(const QComboBox* modelCombo,
                                               const QComboBox* deviceCombo,
                                               const QSpinBox* threadsSpin) {
    if (!modelCombo || !deviceCombo || !threadsSpin) return;
    QStringList labels;
    QStringList filenames;
    for (int i = 0; i < modelCombo->count(); ++i) {
        labels.append(modelCombo->itemText(i));
        filenames.append(modelCombo->itemData(i).toString());
    }
    rebuildModelCombo(labels, filenames, modelCombo->currentData().toString());
    rebuildDeviceCombo(deviceCombo);
    m_syncingModelControls = true;
    if (m_threadsSpin) m_threadsSpin->setValue(threadsSpin->value());
    m_threadCount = threadsSpin->value();
    m_syncingModelControls = false;
}

void FaceRegistryWidget::setModelPath(const QString& path) {
    if (m_modelPath != path) {
#ifdef AICore_ENABLED
        m_embedContext.release();
#endif
    }
    m_modelPath = path;
    if (!m_modelCombo) return;
    const QString fn = QFileInfo(path).fileName();
    const int idx = m_modelCombo->findData(fn);
    if (idx >= 0) {
        m_syncingModelControls = true;
        m_modelCombo->setCurrentIndex(idx);
        m_syncingModelControls = false;
    }
}

void FaceRegistryWidget::setAuthThreshold(float value) {
    if (!m_authThreshold) return;
    m_authThreshold->blockSignals(true);
    m_authThreshold->setValue(value);
    m_authThreshold->blockSignals(false);
}

float FaceRegistryWidget::authThreshold() const {
    return m_authThreshold ? static_cast<float>(m_authThreshold->value())
                           : 0.65f;
}

QString FaceRegistryWidget::resolveModelPath() const { return m_modelPath; }

void FaceRegistryWidget::setDevice(const QString& device) {
    if (m_device != device) {
#ifdef AICore_ENABLED
        m_embedContext.release();
#endif
    }
    m_device = device;
    if (!m_deviceCombo) return;
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) {
        m_syncingModelControls = true;
        m_deviceCombo->setCurrentIndex(idx);
        m_syncingModelControls = false;
    }
}

QString FaceRegistryWidget::registryPath() const {
    return m_registryPathEdit ? m_registryPathEdit->text().trimmed()
                              : m_store.path();
}

void FaceRegistryWidget::releaseStoreConnection() {
    m_store.close();
    if (m_dbStatusLabel) {
        m_dbStatusLabel->setText(
                tr("Database closed while background registration runs…"));
    }
}

void FaceRegistryWidget::setRegistryPath(const QString& path, bool userChosen) {
    m_registryPathUserChosen = userChosen;
    if (m_registryPathEdit) m_registryPathEdit->setText(path);
    m_store.rebind(path);
    refreshList();
    if (m_dbStatusLabel) {
        m_dbStatusLabel->setText(
                m_store.isOpen()
                        ? tr("%1 entries in %2")
                                  .arg(m_store.entries().size())
                                  .arg(QFileInfo(path).fileName())
                        : tr("Failed to open registry at %1").arg(path));
    }
    emit registryPathChanged(path);
}

void FaceRegistryWidget::loadSettings() {
    QSettings settings;
    QString dbPath =
            settings.value(FaceDetectTestData::manualRegistryDbSettingsKey())
                    .toString();
    m_registryPathUserChosen = !dbPath.isEmpty();
    if (dbPath.isEmpty()) {
        const QString legacyActive =
                settings.value(FaceDetectTestData::activeRegistrySettingsKey())
                        .toString();
        const QString legacyDb =
                settings.value(QStringLiteral("qFaceDetect/registryDbPath"))
                        .toString();
        const QString legacy =
                !legacyActive.isEmpty() ? legacyActive : legacyDb;
        if (!legacy.isEmpty() &&
            !FaceDetectTestData::isFriendsBundlePath(legacy)) {
            dbPath = legacy;
            m_registryPathUserChosen = true;
        }
    }
    const double thresh =
            settings.value(QStringLiteral("qFaceDetect/matchThreshold"), 0.65)
                    .toDouble();
    const double minScore =
            settings.value(QStringLiteral("qFaceDetect/minDetectionScore"),
                           settings.value(
                                   QStringLiteral("qFaceDetect/"
                                                  "registryMinDetectionScore"),
                                   0.5))
                    .toDouble();
    setAuthThreshold(static_cast<float>(thresh));
    setMinDetectionScore(static_cast<float>(minScore));
    if (m_exportAuthToDbCheck) {
        m_exportAuthToDbCheck->setChecked(
                settings.value(QStringLiteral(
                                       "qFaceDetect/exportAuthResultToDb"),
                               false)
                        .toBool());
    }
    if (m_registryPathUserChosen && !dbPath.isEmpty()) {
        setRegistryPath(dbPath, true);
    }
}

void FaceRegistryWidget::saveSettings() const {
    QSettings settings;
    if (m_registryPathUserChosen) {
        const QString path = registryPath();
        if (!path.isEmpty()) {
            settings.setValue(FaceDetectTestData::manualRegistryDbSettingsKey(),
                              path);
        } else {
            settings.remove(FaceDetectTestData::manualRegistryDbSettingsKey());
        }
    } else {
        settings.remove(FaceDetectTestData::manualRegistryDbSettingsKey());
    }
    settings.remove(FaceDetectTestData::activeRegistrySettingsKey());
    settings.remove(QStringLiteral("qFaceDetect/registryDbPath"));
    settings.setValue(QStringLiteral("qFaceDetect/matchThreshold"),
                      authThreshold());
    settings.setValue(QStringLiteral("qFaceDetect/minDetectionScore"),
                      minDetectionScore());
    settings.setValue(QStringLiteral("qFaceDetect/exportAuthResultToDb"),
                      exportAuthResultToDb());
}

void FaceRegistryWidget::refreshList() {
    m_entryList->clear();
    for (const FaceRegistryEntry& e : m_store.entries()) {
        auto* item = new QListWidgetItem(QStringLiteral("%1  (%2-d, %3)")
                                                 .arg(e.name)
                                                 .arg(e.embedDim)
                                                 .arg(e.modelFile));
        item->setData(Qt::UserRole, e.id);
        if (!e.thumbnail.isNull()) {
            item->setIcon(QIcon(QPixmap::fromImage(e.thumbnail)));
        }
        m_entryList->addItem(item);
    }
    if (m_dbStatusLabel && m_store.isOpen()) {
        m_dbStatusLabel->setText(
                tr("%1 entries in %2")
                        .arg(m_store.entries().size())
                        .arg(QFileInfo(m_store.path()).fileName()));
    }
}

void FaceRegistryWidget::onBrowseRegistryDb() {
    QSettings settings;
    const QString lastDir =
            settings.value(QStringLiteral("qFaceDetect/lastRegistryDir"),
                           FaceDetectEmbed::modelCacheDir())
                    .toString();
    const QString path = cvFileDialog::getSaveFileName(
            this, tr("Face registry database"),
            lastDir + QStringLiteral("/face_registry.db"),
            tr("SQLite database (*.db);;All files (*.*)"));
    if (path.isEmpty()) return;
    settings.setValue(QStringLiteral("qFaceDetect/lastRegistryDir"),
                      QFileInfo(path).absolutePath());
    setRegistryPath(path, true);
    saveSettings();
}

void FaceRegistryWidget::onBrowseRegisterImage() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qFaceDetect"),
            QStringLiteral("lastImageFileDir"), QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select registration image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.webp)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qFaceDetect"),
                         QStringLiteral("lastImageFileDir"), path);
    if (m_registerImagePath) m_registerImagePath->setText(path);
}

void FaceRegistryWidget::onBrowseAuthImage() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qFaceDetect"),
            QStringLiteral("lastImageFileDir"), QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select probe image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.webp)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qFaceDetect"),
                         QStringLiteral("lastImageFileDir"), path);
    if (m_authImagePath) m_authImagePath->setText(path);
    updateAuthPreview();
}

bool FaceRegistryWidget::embedImage(const QString& imagePath,
                                    std::vector<float>* out,
                                    int* outDim,
                                    QString* err,
                                    float minDetectionScore) {
#ifdef AICore_ENABLED
    const QString path = resolveModelPath();
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        if (err) *err = tr("Recognizer model not available locally.");
        return false;
    }

    FaceDetectInferenceGuard guard(m_device);
    if (!m_embedContext.ensureLoaded(path, m_device, m_threadCount)) {
        if (err) *err = tr("Failed to load face model.");
        return false;
    }

    const bool ok = FaceDetectEmbed::embedImagePathDetectAligned(
            m_embedContext.get(), imagePath, out, minDetectionScore);
    if (ok && outDim) {
        *outDim = static_cast<int>(out->size());
    }
    if (!ok && err) {
        *err = tr("Embedding failed: %1")
                       .arg(aicore_facedetect_last_error(m_embedContext.get())
                                    ? aicore_facedetect_last_error(
                                              m_embedContext.get())
                                    : "no face detected");
    }
    return ok;
#else
    Q_UNUSED(imagePath);
    Q_UNUSED(out);
    Q_UNUSED(outDim);
    if (err) *err = tr("AICore not enabled.");
    return false;
#endif
}

void FaceRegistryWidget::onRegister() {
    const QString name = m_nameEdit->text().trimmed();
    const QString path = m_registerImagePath->text().trimmed();
    if (name.isEmpty() || path.isEmpty()) {
        emit logMessage(tr("[Registry] Name and image required."));
        return;
    }
    setProcessing(true, tr("Registering…"));
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage thumb = reader.read();
    registerFromImagePath(path, thumb);
    setProcessing(false);
}

void FaceRegistryWidget::registerFromImagePath(const QString& imagePath,
                                               const QImage& thumb) {
    const QString name = m_nameEdit->text().trimmed();
    if (name.isEmpty()) {
        emit logMessage(tr("[Registry] Enter a name before registering."));
        return;
    }
    if (!registerPersonFromImage(name, imagePath, minDetectionScore())) return;

    Q_UNUSED(thumb);
    refreshList();
    saveSettings();
    emit registryChanged();
    m_nameEdit->clear();
    m_registerImagePath->clear();
}

void FaceRegistryWidget::onAuthenticate() {
    const QString path = m_authImagePath->text().trimmed();
    if (path.isEmpty()) {
        emit logMessage(tr("[Registry] Probe image required."));
        return;
    }
    setProcessing(true, tr("Authenticating…"));
    authenticateFromImagePath(path);
    setProcessing(false);
}

void FaceRegistryWidget::authenticateFromImagePath(const QString& imagePath) {
#ifdef AICore_ENABLED
    const QString modelPath = resolveModelPath();
    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        emit logMessage(tr("[Registry] Recognizer model not available."));
        return;
    }
    if (!m_store.isOpen()) {
        emit logMessage(tr("[Registry] Database not open."));
        return;
    }

    FaceDetectInferenceGuard guard(m_device);
    if (!m_embedContext.ensureLoaded(modelPath, m_device, m_threadCount)) {
        emit logMessage(tr("[Registry] Failed to load face model."));
        return;
    }
    aicore_facedetect_ctx* ctx = m_embedContext.get();

    QImage rgb = FaceDetectEmbed::loadRgbForInference(imagePath);
    if (rgb.isNull()) {
        emit logMessage(tr("[Registry] Failed to load probe image."));
        return;
    }

    const std::vector<FaceDetectBox> boxes =
            FaceDetectEmbed::detectBoxesFromRgb(ctx, rgb);

    const float minScore = minDetectionScore();
    const float thresh = static_cast<float>(m_authThreshold->value());
    QStringList lines;
    int matched = 0;
    int index = 0;
    std::vector<FaceDetectEmbed::AnnotatedFaceLabel> drawFaces;

    if (boxes.empty()) {
        std::vector<float> emb;
        if (!FaceDetectEmbed::embedImagePathWithFallback(ctx, imagePath, &emb,
                                                         minScore)) {
            emit logMessage(
                    tr("[Registry] %1")
                            .arg(aicore_facedetect_last_error(ctx)
                                         ? aicore_facedetect_last_error(ctx)
                                         : tr("Embedding failed")));
            return;
        }
        const auto match = m_store.bestMatch(emb, thresh);
        const auto nearest = m_store.nearestMatch(emb);
        if (!match) {
            const QString distText =
                    nearest ? QString::number(nearest->distance, 'f', 3)
                            : QStringLiteral("?");
            m_authResultLabel->setPlainText(
                    tr("NO MATCH (threshold %1, %2)")
                            .arg(thresh, 0, 'f', 2)
                            .arg(FaceDetectEmbed::formatNoMatchLabel(
                                    nearest ? nearest->distance : 1.f,
                                    nearest ? nearest->entry.name
                                            : QString())));
            m_authResultLabel->setStyleSheet(
                    "padding: 8px; border-radius: 4px; background: #fef2f2; "
                    "border: 1px solid #fecaca; color: #991b1b;");
            emit logMessage(tr("[Registry] Authentication failed — no match "
                               "(nearest d=%1).")
                                    .arg(distText));
            return;
        }
        FaceDetectEmbed::AnnotatedFaceLabel draw;
        draw.labelOnly = true;
        draw.label = FaceDetectEmbed::formatMatchLabel(match->entry.name,
                                                       match->distance);
        draw.matched = true;
        drawFaces.push_back(draw);
        if (exportAuthResultToDb() && !rgb.isNull()) {
            emit authResultImageReady(
                    FaceDetectEmbed::annotateLabeledFaces(rgb, drawFaces),
                    tr("Auth: %1").arg(match->entry.name));
        }
        m_authResultLabel->setPlainText(
                tr("MATCH: %1  ·  ID %2  ·  distance %3")
                        .arg(match->entry.name, match->entry.id.left(8))
                        .arg(match->distance, 0, 'f', 4));
        m_authResultLabel->setStyleSheet(
                "padding: 8px; border-radius: 4px; background: #ecfdf5; "
                "border: 1px solid #a7f3d0; color: #065f46;");
        emit logMessage(tr("[Registry] Authenticated as '%1' (distance %2).")
                                .arg(match->entry.name)
                                .arg(match->distance, 0, 'f', 4));
        return;
    }

    for (const FaceDetectBox& box : boxes) {
        ++index;
        FaceDetectEmbed::AnnotatedFaceLabel draw;
        draw.box = box;

        if (box.score < minScore) {
            draw.label = tr("skipped (det=%1)").arg(box.score, 0, 'f', 2);
            draw.matched = false;
            draw.dashed = true;
            drawFaces.push_back(draw);
            lines << tr("Face %1: skipped (det score %2 < min %3)")
                             .arg(index)
                             .arg(box.score, 0, 'f', 3)
                             .arg(minScore, 0, 'f', 2);
            continue;
        }

        if (rgb.isNull()) {
            draw.label = tr("crop failed");
            draw.matched = false;
            draw.dashed = true;
            drawFaces.push_back(draw);
            lines << tr("Face %1: crop failed (det %2)")
                             .arg(index)
                             .arg(box.score, 0, 'f', 3);
            continue;
        }

        std::vector<float> emb;
        if (!FaceDetectEmbed::embedFaceBoxFromFrame(ctx, rgb, box, minScore,
                                                    &emb)) {
            draw.label = tr("embed failed (det=%1)").arg(box.score, 0, 'f', 2);
            draw.matched = false;
            draw.dashed = true;
            drawFaces.push_back(draw);
            lines << tr("Face %1: embed failed (det %2)")
                             .arg(index)
                             .arg(box.score, 0, 'f', 3);
            emit logMessage(tr("[Registry] Face %1: embed failed (det %2)")
                                    .arg(index)
                                    .arg(box.score, 0, 'f', 3));
            continue;
        }

        {
            QStringList distParts;
            for (const FaceRegistryEntry& entry : m_store.entries()) {
                const float d =
                        FaceRegistryStore::cosineDistance(emb, entry.embedding);
                distParts << QStringLiteral("%1=%2")
                                     .arg(entry.name)
                                     .arg(d, 0, 'f', 4);
            }
            CVLog::Print(QString("[Registry] Face %1 cosine distances: [%2] "
                                 "(threshold %3)")
                                 .arg(index)
                                 .arg(distParts.join(QStringLiteral(", ")))
                                 .arg(thresh, 0, 'f', 2));
        }

        const auto match = m_store.bestMatch(emb, thresh);
        draw.label = FaceDetectEmbed::labelForEmbedding(&m_store, emb, thresh);
        draw.matched = match.has_value();

        if (match) {
            ++matched;
            lines << tr("Face %1: MATCH %2 (dist %3, det %4)")
                             .arg(index)
                             .arg(match->entry.name)
                             .arg(match->distance, 0, 'f', 4)
                             .arg(box.score, 0, 'f', 3);
            emit logMessage(tr("[Registry] Face %1 → %2 (distance %3)")
                                    .arg(index)
                                    .arg(match->entry.name)
                                    .arg(match->distance, 0, 'f', 4));
        } else {
            const auto nearest = m_store.nearestMatch(emb);
            const QString nearestName =
                    nearest ? nearest->entry.name : QStringLiteral("?");
            const float nearestDist = nearest ? nearest->distance : 1.0f;
            lines << tr("Face %1: NO MATCH nearest=%2 (d=%3, det %4)")
                             .arg(index)
                             .arg(nearestName)
                             .arg(nearestDist, 0, 'f', 4)
                             .arg(box.score, 0, 'f', 3);
            emit logMessage(tr("[Registry] Face %1: NO MATCH nearest %2 (d=%3)")
                                    .arg(index)
                                    .arg(nearestName)
                                    .arg(nearestDist, 0, 'f', 4));
        }
        drawFaces.push_back(draw);
    }

    const QString summary =
            tr("Detected %1 face(s), %2 matched (min det %3, match thresh %4)")
                    .arg(boxes.size())
                    .arg(matched)
                    .arg(minScore, 0, 'f', 2)
                    .arg(thresh, 0, 'f', 2);
    m_authResultLabel->setPlainText(summary + QStringLiteral("\n") +
                                    lines.join(QStringLiteral("\n")));
    if (exportAuthResultToDb() && !rgb.isNull() && !drawFaces.empty()) {
        emit authResultImageReady(
                FaceDetectEmbed::annotateLabeledFaces(rgb, drawFaces), summary);
    }
    if (matched > 0) {
        m_authResultLabel->setStyleSheet(
                "padding: 8px; border-radius: 4px; background: #ecfdf5; "
                "border: 1px solid #a7f3d0; color: #065f46;");
    } else {
        m_authResultLabel->setStyleSheet(
                "padding: 8px; border-radius: 4px; background: #fef2f2; "
                "border: 1px solid #fecaca; color: #991b1b;");
    }
#else
    Q_UNUSED(imagePath);
    emit logMessage(tr("[Registry] AICore not enabled."));
#endif
}

void FaceRegistryWidget::onRemove() {
    auto* item = m_entryList->currentItem();
    if (!item) return;
    const QString id = item->data(Qt::UserRole).toString();
    if (m_store.removeEntry(id)) {
        refreshList();
        saveSettings();
        emit registryChanged();
        emit logMessage(tr("[Registry] Removed entry."));
    }
}

void FaceRegistryWidget::onClear() {
    if (QMessageBox::question(this, tr("Clear registry"),
                              tr("Remove all registered faces?")) !=
        QMessageBox::Yes) {
        return;
    }
    m_store.clear();
    refreshList();
    saveSettings();
    emit registryChanged();
    emit logMessage(tr("[Registry] Cleared."));
}

void FaceRegistryWidget::setProcessing(bool busy, const QString& busyHint) {
    if (m_registerBtn) m_registerBtn->setEnabled(!busy);
    if (m_authBtn) m_authBtn->setEnabled(!busy);
    if (busy) {
        if (m_dbStatusLabel) {
            m_statusLabelSavedText = m_dbStatusLabel->text();
            if (!busyHint.isEmpty()) {
                m_dbStatusLabel->setText(busyHint);
            }
        }
        if (m_progressBar) {
            m_progressBar->setMaximum(0);  // indeterminate
            m_progressBar->setTextVisible(true);
            m_progressBar->setFormat(busyHint.isEmpty() ? tr("Processing…")
                                                        : busyHint);
        }
        setCursor(Qt::WaitCursor);
    } else {
        unsetCursor();
        if (m_dbStatusLabel && !m_statusLabelSavedText.isEmpty()) {
            m_dbStatusLabel->setText(m_statusLabelSavedText);
            m_statusLabelSavedText.clear();
        }
        if (m_progressBar) {
            m_progressBar->setMaximum(100);
            m_progressBar->setValue(0);
            m_progressBar->setTextVisible(false);
        }
    }
}
