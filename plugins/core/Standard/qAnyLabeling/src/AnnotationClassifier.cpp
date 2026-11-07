// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ecvModelDownloader.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QInputDialog>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QScrollArea>
#include <QShortcut>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <functional>

#include "AnnotationClassifierPrompt.h"
#include "AnnotationDocument.h"
#include "AnnotationWindow.h"
#include "AnnotationWorker.h"
#include "ChatPromptDialog.h"
#include "QtCompat.h"

void AnnotationWindow::openImageClassifier() {
    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("AnnotationImageClassifier"));
    dialog.setWindowTitle(tr("Image Classifier"));
    dialog.setWindowFlags(Qt::Window | Qt::WindowMinMaxButtonsHint |
                          Qt::WindowCloseButtonHint);
    dialog.resize(1200, 700);
    auto* root = new QHBoxLayout(&dialog);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(20);
    auto* splitter = new QSplitter(Qt::Horizontal, &dialog);
    splitter->setChildrenCollapsible(false);
    splitter->setHandleWidth(3);
    root->addWidget(splitter);
    auto* left = new QWidget(splitter);
    auto* leftLayout = new QVBoxLayout(left);
    auto* filename = new QLabel(left);
    filename->setAlignment(Qt::AlignCenter);
    leftLayout->addWidget(filename);
    auto* preview = new QLabel(left);
    preview->setObjectName(QStringLiteral("ClassifierPreview"));
    preview->setAlignment(Qt::AlignCenter);
    preview->setMinimumSize(400, 400);
    leftLayout->addWidget(preview, 1);
    auto* overlay = new QLabel(left);
    overlay->setAlignment(Qt::AlignCenter);
    leftLayout->addWidget(overlay);
    auto* right = new QWidget(splitter);
    auto* layout = new QVBoxLayout(right);
    auto* controls = new QGridLayout;
    layout->addLayout(controls);
    auto button = [&dialog, controls](const QString& text, const char* name,
                                      int row, int column) {
        auto* widget = new QPushButton(text, &dialog);
        widget->setObjectName(QString::fromLatin1(name));
        controls->addWidget(widget, row, column);
        return widget;
    };
    auto* exportButton = button(tr("Export"), "ClassifierExport", 0, 0);
    auto* multiclass = button(tr("MultiClass"), "ClassifierMultiClass", 0, 1);
    auto* multilabel = button(tr("MultiLabel"), "ClassifierMultiLabel", 0, 2);
    auto* autoRun = button(tr("AutoRun"), "ClassifierAutoRun", 0, 3);
    auto* add = button(tr("Add Label"), "ClassifierAddLabel", 1, 0);
    auto* remove = button(tr("Delete Label"), "ClassifierDeleteLabel", 1, 1);
    auto* rename = button(tr("Edit Label"), "ClassifierEditLabel", 1, 2);
    auto* stats = button(tr("View Statistics"), "ClassifierStatistics", 1, 3);
    auto* ai = button(tr("AI Assistant"), "ClassifierAI", 2, 3);
    auto* local = new QToolButton(&dialog);
    local->setObjectName(QStringLiteral("ClassifierLocalModel"));
    local->setText(tr("Local model"));
    local->setPopupMode(QToolButton::InstantPopup);
    auto* localMenu = new QMenu(local);
    auto* localImage = localMenu->addAction(tr("Run"));
    localImage->setObjectName(QStringLiteral("ClassifierLocalAI"));
    auto* localBatch = localMenu->addAction(tr("AutoRun"));
    localBatch->setObjectName(QStringLiteral("ClassifierLocalAutoRun"));
    local->setMenu(localMenu);
    local->setToolTip(
            tr("Uses the classification model's trained category labels."));
    controls->addWidget(local, 3, 3);
    auto* aiStatus = new QLabel(&dialog);
    aiStatus->setObjectName(QStringLiteral("ClassifierAIStatus"));
    aiStatus->setWordWrap(true);
    controls->addWidget(aiStatus, 3, 0, 1, 3);
    auto* search = new QLineEdit(&dialog);
    search->setObjectName(QStringLiteral("ClassifierSearch"));
    search->setPlaceholderText(tr("Search categories"));
    search->setClearButtonEnabled(true);
    controls->addWidget(search, 2, 0, 1, 3);
    auto* scroll = new QScrollArea(right);
    scroll->setWidgetResizable(true);
    auto* categories = new QWidget(scroll);
    auto* categoryLayout = new QVBoxLayout(categories);
    categoryLayout->setAlignment(Qt::AlignTop);
    scroll->setWidget(categories);
    layout->addWidget(scroll, 1);
    auto* nav = new QHBoxLayout;
    layout->addLayout(nav);
    auto* previous = new QPushButton(tr("Previous"), right);
    auto* next = new QPushButton(tr("Next"), right);
    previous->setObjectName(QStringLiteral("ClassifierPrevious"));
    next->setObjectName(QStringLiteral("ClassifierNext"));
    previous->setToolTip(
            tr("Previous image (A) | Previous unlabeled image (Ctrl+A)"));
    next->setToolTip(tr("Next image (D) | Next unlabeled image (Ctrl+D)"));
    auto* page = new QSpinBox(right);
    page->setObjectName(QStringLiteral("ClassifierPage"));
    page->setRange(1, std::max(1, m_files->count()));
    nav->addWidget(previous);
    nav->addWidget(page, 1);
    nav->addWidget(next);
    splitter->setSizes({600, 600});
    bool singleLabel = true;
    bool classificationStarted = false;
    bool batchStarted = false;
    bool remoteBusy = false, remoteBatch = false;
    int remoteIndex = 0, remoteFailed = 0;
    quint64 remoteGeneration = 0, remoteOperation = 0;
    QString remoteSource, remoteTarget;
    QStringList remoteFiles;
    AnnotationChatRequest remoteRequest;
    auto* remoteClient = new ChatServiceClient(&dialog);
    auto* remoteProgress = new QProgressDialog(tr("Processing images..."),
                                               tr("Cancel"), 0, 0, &dialog);
    remoteProgress->setObjectName(QStringLiteral("ClassifierVLMProgress"));
    remoteProgress->setWindowModality(Qt::WindowModal);
    remoteProgress->setMinimumDuration(0);
    remoteProgress->setAutoClose(false);
    remoteProgress->hide();
    bool updating = false;
    QStringList labels;
    for (int i = 0; i < m_flags->count(); ++i)
        labels.append(m_flags->item(i)->text());
    QVector<QCheckBox*> boxes;
    auto applyFilter = [&]() {
        const QStringList terms = search->text().toCaseFolded().split(
                QLatin1Char(' '), QtCompat::SkipEmptyParts);
        for (QCheckBox* box : boxes) {
            bool visible = true;
            for (const QString& term : terms)
                visible &= box->property("category")
                                   .toString()
                                   .toCaseFolded()
                                   .contains(term);
            box->setVisible(visible);
        }
    };
    std::function<void()> refresh;
    auto writeFlags = [&]() {
        if (updating || m_imagePath.isEmpty() || m_sidecarBlocked) return;
        QJsonObject flags;
        for (QCheckBox* box : boxes)
            flags.insert(box->property("category").toString(),
                         box->isChecked());
        m_document.insert(QStringLiteral("flags"), flags);
        m_imageFlagNames = labels;
        updateFlags();
        setDirty(true);
        QStringList selected;
        for (QCheckBox* box : boxes)
            if (box->isChecked())
                selected.append(box->property("category").toString());
        overlay->setText(selected.join(QStringLiteral(", ")));
    };
    auto rebuild = [&]() {
        for (QCheckBox* box : boxes) delete box;
        boxes.clear();
        for (const QString& label : labels) {
            auto* box = new QCheckBox(
                    QStringLiteral("%1(%2)").arg(label).arg(boxes.size()),
                    categories);
            box->setProperty("category", label);
            box->setObjectName(QStringLiteral("ClassifierCategory"));
            categoryLayout->addWidget(box);
            boxes.append(box);
            connect(box, &QCheckBox::toggled, &dialog, [&, box](bool checked) {
                if (updating) return;
                if (singleLabel && checked) {
                    for (QCheckBox* other : boxes) {
                        if (other == box) continue;
                        const QSignalBlocker block(other);
                        other->setChecked(false);
                    }
                }
                writeFlags();
            });
        }
        applyFilter();
    };
    refresh = [&]() {
        updating = true;
        const QJsonObject flags =
                m_document.value(QStringLiteral("flags")).toObject();
        for (QCheckBox* box : boxes)
            box->setChecked(
                    flags.value(box->property("category").toString()).toBool());
        filename->setText(m_imagePath.isEmpty()
                                  ? tr("No image loaded")
                                  : QFileInfo(m_imagePath).fileName());
        preview->setPixmap(QPixmap::fromImage(m_image).scaled(
                preview->size(), Qt::KeepAspectRatio,
                Qt::SmoothTransformation));
        QStringList selected;
        for (const QString& label : labels)
            if (flags.value(label).toBool()) selected.append(label);
        overlay->setText(selected.join(QStringLiteral(", ")));
        page->setValue(std::max(1, m_files->currentRow() + 1));
        page->setSuffix(tr(" / %1").arg(m_files->count()));
        previous->setEnabled(m_files->currentRow() > 0);
        next->setEnabled(m_files->currentRow() >= 0 &&
                         m_files->currentRow() + 1 < m_files->count());
        multiclass->setCheckable(true);
        multilabel->setCheckable(true);
        multiclass->setChecked(singleLabel);
        multilabel->setChecked(!singleLabel);
        const bool busy = remoteBusy || m_inferenceRunning || m_batchRunning ||
                          m_downloader->isBusy();
        previous->setEnabled(!busy && previous->isEnabled());
        next->setEnabled(!busy && next->isEnabled());
        page->setEnabled(!busy);
        multiclass->setEnabled(!busy);
        multilabel->setEnabled(!busy);
        for (QPushButton* widget :
             {add, remove, rename, exportButton, ai, autoRun})
            widget->setEnabled(!busy);
        local->setEnabled(!busy);
        for (QCheckBox* box : boxes)
            box->setEnabled(!busy && !m_sidecarBlocked &&
                            !m_imagePath.isEmpty());
        updating = false;
    };
    auto activate = [&](int index) {
        if (remoteBusy || m_inferenceRunning || m_batchRunning ||
            m_downloader->isBusy() || index < 0 || index >= m_files->count())
            return;
        openFile(m_files->item(index)->data(Qt::UserRole).toString());
        refresh();
    };
    auto readFlags = [&](const QString& image) {
        if (image == m_imagePath)
            return m_document.value(QStringLiteral("flags")).toObject();
        QFile file(sidecarPathFor(image));
        if (!file.open(QIODevice::ReadOnly)) return QJsonObject{};
        return QJsonDocument::fromJson(file.readAll())
                .object()
                .value(QStringLiteral("flags"))
                .toObject();
    };
    auto unlabeled = [&](int direction) {
        const int count = m_files->count(), current = m_files->currentRow();
        for (int step = 1; step < count; ++step) {
            const int index = (current + direction * step + count) % count;
            const QJsonObject flags = readFlags(
                    m_files->item(index)->data(Qt::UserRole).toString());
            bool labeled = false;
            for (const QString& label : labels)
                labeled |= flags.value(label).toBool();
            if (!labeled) {
                activate(index);
                return;
            }
        }
        QMessageBox::information(&dialog, tr("Info"),
                                 tr("No unlabeled images found."));
    };
    // Preserve all annotation fields while changing the flag vocabulary. Bad
    // JSON is reported and left intact. Each sidecar commits atomically.
    auto updateVocabulary = [&](const QStringList& names,
                                const QMap<QString, QString>& oldNames) {
        QProgressDialog progress(tr("Updating label files..."), tr("Cancel"), 0,
                                 m_files->count(), &dialog);
        for (int i = 0; i < m_files->count(); ++i) {
            if (progress.wasCanceled()) break;
            const QString image =
                    m_files->item(i)->data(Qt::UserRole).toString();
            const QString path = sidecarPathFor(image);
            QJsonObject object;
            if (image == m_imagePath)
                object = m_document;
            else {
                QFile file(path);
                if (!file.exists()) continue;
                if (!file.open(QIODevice::ReadOnly)) {
                    QMessageBox::warning(&dialog, tr("Image Classifier"),
                                         file.errorString());
                    return false;
                }
                QJsonParseError error;
                const QJsonDocument doc =
                        QJsonDocument::fromJson(file.readAll(), &error);
                if (error.error != QJsonParseError::NoError ||
                    !doc.isObject()) {
                    QMessageBox::warning(
                            &dialog, tr("Image Classifier"),
                            tr("Invalid annotation: %1").arg(path));
                    return false;
                }
                object = doc.object();
            }
            const QJsonObject before =
                    object.value(QStringLiteral("flags")).toObject();
            QJsonObject after;
            for (const QString& name : names)
                after.insert(name,
                             before.value(oldNames.value(name, name)).toBool());
            object.insert(QStringLiteral("flags"), after);
            if (image == m_imagePath) {
                m_document = object;
                setDirty(true);
            } else {
                QSaveFile file(path);
                const QByteArray data = QJsonDocument(object).toJson();
                if (!file.open(QIODevice::WriteOnly) ||
                    file.write(data) != data.size() || !file.commit()) {
                    QMessageBox::warning(&dialog, tr("Image Classifier"),
                                         file.errorString());
                    return false;
                }
            }
            progress.setValue(i + 1);
        }
        labels = names;
        m_imageFlagNames = labels;
        updateFlags();
        rebuild();
        refresh();
        return true;
    };
    connect(search, &QLineEdit::textChanged, &dialog,
            [&](const QString&) { applyFilter(); });
    connect(previous, &QPushButton::clicked, &dialog,
            [&]() { activate(m_files->currentRow() - 1); });
    connect(next, &QPushButton::clicked, &dialog,
            [&]() { activate(m_files->currentRow() + 1); });
    connect(page, QOverload<int>::of(&QSpinBox::valueChanged), &dialog,
            [&](int value) {
                if (!updating) activate(value - 1);
            });
    connect(multilabel, &QPushButton::clicked, &dialog, [&]() {
        singleLabel = false;
        refresh();
    });
    connect(multiclass, &QPushButton::clicked, &dialog, [&]() {
        if (!singleLabel &&
            QMessageBox::question(
                    &dialog, tr("Switch Mode"),
                    tr("Switching to MultiClass mode will only keep the first "
                       "label for each image. Other labels will be discarded. "
                       "Continue?"),
                    QMessageBox::Yes | QMessageBox::No,
                    QMessageBox::No) == QMessageBox::Yes) {
            singleLabel = true;
        }
        refresh();
    });
    connect(add, &QPushButton::clicked, &dialog, [&]() {
        bool ok = false;
        const QString text = QInputDialog::getMultiLineText(
                &dialog, tr("Add Label"), tr("One label per line"), {}, &ok);
        if (!ok) return;
        QStringList names = labels;
        for (const QString& line : text.split(QLatin1Char('\n'))) {
            const QString label = line.trimmed();
            if (!label.isEmpty() && !names.contains(label)) names.append(label);
        }
        updateVocabulary(names, {});
    });
    connect(remove, &QPushButton::clicked, &dialog, [&]() {
        bool ok = false;
        const QString name =
                QInputDialog::getItem(&dialog, tr("Delete Label"), tr("Label"),
                                      labels, 0, false, &ok);
        if (ok) {
            QStringList names = labels;
            names.removeAll(name);
            updateVocabulary(names, {});
        }
    });
    connect(rename, &QPushButton::clicked, &dialog, [&]() {
        bool ok = false;
        const QString old = QInputDialog::getItem(
                &dialog, tr("Edit Label"), tr("Label"), labels, 0, false, &ok);
        if (!ok) return;
        const QString name = QInputDialog::getText(&dialog, tr("Edit Label"),
                                                   tr("New label"),
                                                   QLineEdit::Normal, old, &ok)
                                     .trimmed();
        if (!ok || name.isEmpty() || name == old) return;
        if (labels.contains(name)) {
            QMessageBox::warning(&dialog, tr("Edit Label"),
                                 tr("Label already exists"));
            return;
        }
        QStringList names = labels;
        names[names.indexOf(old)] = name;
        updateVocabulary(names, {{name, old}});
    });
    connect(stats, &QPushButton::clicked, &dialog, [&]() {
        QMap<QString, int> counts;
        for (int i = 0; i < m_files->count(); ++i) {
            const QJsonObject flags =
                    readFlags(m_files->item(i)->data(Qt::UserRole).toString());
            for (const QString& label : labels)
                if (flags.value(label).toBool()) ++counts[label];
        }
        QStringList lines;
        for (const QString& label : labels)
            lines.append(QStringLiteral("%1: %2").arg(label).arg(
                    counts.value(label)));
        QMessageBox::information(&dialog, tr("View Statistics"),
                                 lines.join(QLatin1Char('\n')));
    });
    connect(exportButton, &QPushButton::clicked, &dialog, [&]() {
        if (!singleLabel) {
            QMessageBox::warning(&dialog, tr("Warning"),
                                 tr("Export only supports multi-class tasks!"));
            return;
        }
        if (labels.isEmpty()) {
            QMessageBox::warning(&dialog, tr("Warning"),
                                 tr("No labels configured!"));
            return;
        }
        const QString first =
                m_files->count()
                        ? m_files->item(0)->data(Qt::UserRole).toString()
                        : m_imagePath;
        const QString output = QFileDialog::getExistingDirectory(
                &dialog, tr("Export"),
                QDir(QFileInfo(first).absolutePath())
                        .filePath(QStringLiteral("classified")));
        if (output.isEmpty()) return;
        int count = 0;
        for (int i = 0; i < m_files->count(); ++i) {
            const QString image =
                    m_files->item(i)->data(Qt::UserRole).toString();
            const QJsonObject flags = readFlags(image);
            QString category;
            for (const QString& label : labels)
                if (flags.value(label).toBool()) {
                    category = label;
                    break;
                }
            if (category.isEmpty()) continue;
            const QString directory = QDir(output).filePath(category);
            const QString relative =
                    QDir(output).relativeFilePath(QDir::cleanPath(directory));
            if (category == QStringLiteral(".") ||
                category == QStringLiteral("..") ||
                category.contains(QLatin1Char('/')) ||
                category.contains(QLatin1Char('\\')) ||
                QDir::isAbsolutePath(category) ||
                relative == QStringLiteral("..") ||
                relative.startsWith(QStringLiteral("../")) ||
                QFileInfo(directory).isSymLink()) {
                QMessageBox::warning(
                        &dialog, tr("Warning"),
                        tr("Invalid category name: %1").arg(category));
                return;
            }
            const QString destination =
                    QDir(directory).filePath(QFileInfo(image).fileName());
            // QFile::copy refuses overwrite, as the source exporter does.
            if (!QDir().mkpath(directory) ||
                (!QFileInfo::exists(destination) &&
                 !QFile::copy(image, destination))) {
                QMessageBox::warning(&dialog, tr("Export"),
                                     tr("Unable to copy %1").arg(image));
                return;
            }
            ++count;
        }
        QMessageBox::information(
                &dialog, tr("Success"),
                tr("Exported %1 images to %2").arg(count).arg(output));
    });
    auto runClassification = [&](bool batch) {
        QStringList families;
        for (const Model& model : m_models)
            if (model.task == QStringLiteral("classify") &&
                !families.contains(model.family))
                families.append(model.family);
        bool ok = false;
        const QString family =
                QInputDialog::getItem(&dialog, tr("Image classification model"),
                                      tr("Model"), families, 0, false, &ok);
        if (!ok || family.isEmpty()) return;
        m_model->setCurrentIndex(m_model->findText(family));
        if (batch) {
            runBatch();
            batchStarted = m_batchRunning;
            if (batchStarted && m_batchProgress)
                connect(m_batchProgress, &QObject::destroyed, &dialog,
                        [&]() { refresh(); });
        } else {
            runDetection();
            classificationStarted = m_inferenceRunning || m_runAfterDownload;
        }
        refresh();
    };
    connect(localImage, &QAction::triggered, &dialog,
            [&]() { runClassification(false); });
    connect(localBatch, &QAction::triggered, &dialog,
            [&]() { runClassification(true); });

    // The source classifier uses the same VLM endpoint as VQA, with a
    // configurable category prompt. Fixed-category AICore classifiers remain
    // available through the separate Local model menu.
    auto finishRemote = [&](const QString& status) {
        ++remoteOperation;
        remoteClient->cancel();
        remoteBusy = false;
        remoteBatch = false;
        remoteProgress->hide();
        aiStatus->setText(status);
        refresh();
    };
    std::function<void()> requestNext;
    requestNext = [&]() {
        if (!remoteBusy) return;
        const auto operation = remoteOperation;
        if (remoteGeneration != m_sourceGeneration ||
            remoteSource != m_imagePath) {
            finishRemote(tr("Canceled"));
            return;
        }
        if (remoteIndex >= remoteFiles.size()) {
            finishRemote(tr("Completed: %1 images, %2 failed.")
                                 .arg(remoteFiles.size())
                                 .arg(remoteFailed));
            return;
        }
        remoteTarget = remoteFiles[remoteIndex];
        remoteProgress->setLabelText(
                tr("Processing: %1").arg(QFileInfo(remoteTarget).fileName()));
        if (remoteBatch) remoteProgress->setValue(remoteIndex);
        // QProgressDialog::setValue can dispatch events in modal mode.
        if (!remoteBusy || remoteGeneration != m_sourceGeneration ||
            operation != remoteOperation)
            return;
        QString error;
        const auto messages = remoteRequest.messages(remoteTarget, &error);
        if (!error.isEmpty()) {
            finishRemote(error);
            return;
        }
        remoteClient->complete(remoteRequest.settings, messages, false);
    };
    connect(remoteProgress, &QProgressDialog::canceled, &dialog,
            [&]() { finishRemote(tr("Canceled")); });
    connect(this, &AnnotationWindow::annotationSourceChanged, &dialog, [&]() {
        if (remoteBusy)
            finishRemote(tr("Canceled"));
        else
            refresh();
    });
    connect(remoteClient, &ChatServiceClient::finished, &dialog,
            [&](const QString& answer, const QString& failure) {
                if (!remoteBusy || remoteGeneration != m_sourceGeneration ||
                    remoteSource != m_imagePath)
                    return;
                QString error = failure;
                QString json = answer.trimmed();
                const auto fence =
                        QRegularExpression(
                                QStringLiteral("```json\\s*(.*?)\\s*```"),
                                QRegularExpression::DotMatchesEverythingOption)
                                .match(json);
                if (fence.hasMatch()) json = fence.captured(1);
                QJsonParseError parseError;
                const auto parsed =
                        QJsonDocument::fromJson(json.toUtf8(), &parseError);
                if (error.isEmpty() &&
                    (parseError.error != QJsonParseError::NoError ||
                     !parsed.isObject()))
                    error = tr("Failed to parse AI result");
                QJsonObject flags;
                if (error.isEmpty()) {
                    const auto object = parsed.object();
                    for (const auto& label : labels) {
                        const auto flag = object.value(label);
                        if (!flag.isUndefined() && !flag.isBool()) {
                            error = tr("Failed to parse AI result");
                            break;
                        }
                        flags.insert(label, flag.toBool());
                    }
                }
                if (error.isEmpty()) {
                    if (remoteTarget == m_imagePath) {
                        if (m_sidecarBlocked)
                            error = tr("Invalid annotation: %1")
                                            .arg(sidecarPathFor(remoteTarget));
                        else {
                            m_document.insert(QStringLiteral("flags"), flags);
                            m_imageFlagNames = labels;
                            updateFlags();
                            setDirty(true);
                        }
                    } else {
                        // Read the latest document at commit time. Preserve
                        // shapes, unknown metadata and embedded image data;
                        // never repair or overwrite malformed existing sidecars
                        // as a batch side effect.
                        auto document = loadAnnotationDocument(
                                remoteTarget, m_outputDir, m_exifScanEnabled);
                        if (!document.error.isEmpty())
                            error = document.error;
                        else {
                            auto object = document.json;
                            if (object.isEmpty()) {
                                object = QJsonObject{
                                        {"version", annotationFormatVersion()},
                                        {"shapes", QJsonArray()},
                                        {"imagePath",
                                         QFileInfo(remoteTarget).fileName()},
                                        {"imageData", QJsonValue::Null},
                                        {"imageWidth", document.image.width()},
                                        {"imageHeight",
                                         document.image.height()}};
                            }
                            object.insert(QStringLiteral("flags"), flags);
                            QSaveFile file(sidecarPathFor(remoteTarget));
                            const QByteArray data =
                                    QJsonDocument(object).toJson();
                            if (!QDir().mkpath(QFileInfo(file.fileName())
                                                       .absolutePath()) ||
                                !file.open(QIODevice::WriteOnly) ||
                                file.write(data) != data.size() ||
                                !file.commit())
                                error = file.errorString();
                        }
                    }
                }
                if (!error.isEmpty()) ++remoteFailed;
                refresh();
                if (!remoteBatch) {
                    finishRemote(error.isEmpty()
                                         ? tr("AI classification completed!")
                                         : error);
                    return;
                }
                ++remoteIndex;
                aiStatus->setText(error);
                const auto operation = remoteOperation;
                QTimer::singleShot(100, &dialog, [&, operation]() {
                    if (remoteBusy && operation == remoteOperation)
                        requestNext();
                });
            });
    auto runRemote = [&](bool batch) {
        if (remoteBusy || m_inferenceRunning || m_batchRunning ||
            m_downloader->isBusy())
            return;
        if (labels.isEmpty() || m_imagePath.isEmpty() || m_sidecarBlocked) {
            aiStatus->setText(labels.isEmpty()
                                      ? tr("Please configure labels first!")
                                      : tr("No image loaded!"));
            return;
        }
        AnnotationChatRequest request;
        if (!configureAnnotationChatRequest(
                    &dialog, tr("AI Assistant"),
                    annotationClassifierPrompt(labels, singleLabel), &request))
            return;
        QStringList files;
        if (batch) {
            for (int i = 0; i < m_files->count(); ++i)
                files.append(m_files->item(i)->data(Qt::UserRole).toString());
            if (files.isEmpty()) return;
            if (QMessageBox::question(
                        &dialog, tr("Confirmation"),
                        tr("Process %1 images with AI?").arg(files.size()),
                        QMessageBox::Yes | QMessageBox::No,
                        QMessageBox::No) != QMessageBox::Yes)
                return;
        } else
            files.append(m_imagePath);
        remoteRequest = request;
        remoteFiles = files;
        remoteSource = m_imagePath;
        remoteGeneration = m_sourceGeneration;
        remoteIndex = remoteFailed = 0;
        ++remoteOperation;
        remoteBusy = true;
        remoteBatch = batch;
        remoteProgress->reset();
        remoteProgress->setRange(0, batch ? files.size() : 0);
        remoteProgress->show();
        refresh();
        requestNext();
    };
    connect(ai, &QPushButton::clicked, &dialog, [&]() { runRemote(false); });
    connect(autoRun, &QPushButton::clicked, &dialog,
            [&]() { runRemote(true); });
    connect(m_downloader, &ecvModelDownloader::finished, &dialog,
            [&](bool, const QString&) { refresh(); });
    connect(m_worker, &AnnotationWorker::finished, &dialog,
            [&](const AnnotationInference& result) {
                if ((classificationStarted || batchStarted) &&
                    result.hasFlags && result.imagePath == m_imagePath &&
                    result.generation == m_generation &&
                    result.error.isEmpty()) {
                    m_document.insert(QStringLiteral("flags"), result.flags);
                    labels = result.flags.keys();
                    setDirty(true);
                    m_imageFlagNames = labels;
                    updateFlags();
                    rebuild();
                }
                refresh();
            });
    auto shortcut = [&](const QString& key,
                        const std::function<void()>& handler) {
        auto* action = new QShortcut(QKeySequence(key), &dialog);
        connect(action, &QShortcut::activated, &dialog, handler);
    };
    shortcut(QStringLiteral("A"),
             [&]() { activate(m_files->currentRow() - 1); });
    shortcut(QStringLiteral("D"),
             [&]() { activate(m_files->currentRow() + 1); });
    shortcut(QStringLiteral("Ctrl+A"), [&]() { unlabeled(-1); });
    shortcut(QStringLiteral("Ctrl+D"), [&]() { unlabeled(1); });
    for (int digit = 0; digit < 10; ++digit)
        shortcut(QString::number(digit), [&, digit]() {
            if (digit < boxes.size() && boxes[digit]->isEnabled())
                boxes[digit]->toggle();
        });
    rebuild();
    refresh();
    dialog.exec();
    remoteClient->cancel();
    disconnect(remoteClient, nullptr, &dialog, nullptr);
    if (batchStarted && m_batchRunning)
        cancelBatch();
    else if (classificationStarted) {
        m_runAfterDownload = false;
        if (m_downloader->isBusy()) m_downloader->cancel();
        if (m_inferenceRunning) ++m_generation;
    }
}
