// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Drives the real "Run (i)" button of the qAnyLabeling workspace with real
// AICore models: RAM image tags and RelateAnything edges. Models and sample
// images come from the shared data cache; when they are absent the test is
// skipped (exit 77) instead of downloading.

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPluginLoader>
#include <QPointer>
#include <QProgressDialog>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <cstdio>
#include <functional>

#include "AnnotationTagging.h"

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

QString dataRoot() {
    const QByteArray env = qgetenv("CLOUDVIEWER_DATA_ROOT");
    return env.isEmpty()
                   ? QDir::home().filePath(QStringLiteral("cloudViewer_data"))
                   : QString::fromLocal8Bit(env);
}

bool waitFor(const std::function<bool()>& condition, int timeoutMs) {
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < timeoutMs) {
        QApplication::processEvents(QEventLoop::AllEvents, 50);
        if (condition()) return true;
    }
    return condition();
}

QJsonObject readJson(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}

QWidget* findCanvas(QWidget* window) {
    for (QWidget* widget : window->findChildren<QWidget*>())
        if (QByteArray(widget->metaObject()->className()) == "AnnotationCanvas")
            return widget;
    return nullptr;
}

bool selectModel(QMainWindow* window,
                 const QString& family,
                 const QString& filename) {
    auto* models = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationModelFamily"));
    auto* quant = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationQuantization"));
    auto* backend =
            window->findChild<QComboBox*>(QStringLiteral("AnnotationBackend"));
    if (!models || !quant || !backend) return false;
    const int familyIndex = models->findText(family);
    if (familyIndex < 0) return false;
    models->setCurrentIndex(familyIndex);
    const int quantIndex = quant->findData(filename);
    const int cpuIndex = backend->findData(QStringLiteral("cpu"));
    if (quantIndex < 0 || cpuIndex < 0) return false;
    quant->setCurrentIndex(quantIndex);
    backend->setCurrentIndex(cpuIndex);
    return true;
}

// Click Run and wait until the status label changes to something other than
// the sentinel, i.e. the worker result has been handled by the window.
bool runAndWait(
        QMainWindow* window,
        QString* statusOut,
        qint64* msOut,
        const QString& buttonName = QStringLiteral("AnnotationRunModel")) {
    auto* status =
            window->findChild<QLabel*>(QStringLiteral("AnnotationStatus"));
    auto* run = window->findChild<QPushButton*>(buttonName);
    if (!status || !run) return false;
    const QString sentinel = QStringLiteral("<running>");
    status->setText(sentinel);
    QElapsedTimer timer;
    timer.start();
    run->click();
    // "Running ..." and "Downloading ..." are progress, not a result.
    const bool done = waitFor(
            [&]() {
                const QString text = status->text();
                return text != sentinel &&
                       !text.startsWith(QStringLiteral("Running")) &&
                       !text.startsWith(QStringLiteral("Downloading"));
            },
            600000);
    if (msOut) *msOut = timer.elapsed();
    if (statusOut) *statusOut = status->text();
    return done;
}

bool setRamOptions(QMainWindow* window,
                   const QString& mode,
                   const QStringList& deleted,
                   const QStringList& retained) {
    auto* button = window->findChild<QPushButton*>("AnnotationRamOptions");
    if (!button) return false;
    bool accepted = false;
    QTimer::singleShot(0, window, [&]() {
        auto* dialog =
                window->findChild<QDialog*>("AnnotationRamOptionsDialog");
        if (!dialog) return;
        auto* language = dialog->findChild<QComboBox*>("RamTagMode");
        auto* remove = dialog->findChild<QPlainTextEdit*>("RamDeleteTags");
        auto* keep = dialog->findChild<QPlainTextEdit*>("RamFilterTags");
        auto* buttons = dialog->findChild<QDialogButtonBox*>();
        if (!language || !remove || !keep || !buttons) {
            dialog->reject();
            return;
        }
        language->setCurrentIndex(language->findData(mode));
        remove->setPlainText(deleted.join('\n'));
        keep->setPlainText(retained.join('\n'));
        buttons->button(QDialogButtonBox::Ok)->click();
        accepted = dialog->result() == QDialog::Accepted;
    });
    button->click();
    return accepted;
}

int runOcr(QMainWindow* window,
           const QTemporaryDir& work,
           const QString& root,
           const QString& device) {
    const QString image = work.filePath("text.jpg");
    const QString second = work.filePath("other.jpg");
    const QString sidecar = work.filePath("text.json");
    const QString source = root + "ppocr_testdata/scenes/ppocrv4_en.jpg";
    if (!check(QFile::copy(source, image) && QFile::copy(source, second),
               "copy OCR fixture"))
        return 1;
    auto write = [&](const QJsonObject& document) {
        QFile file(sidecar);
        return file.open(QIODevice::WriteOnly) &&
               file.write(QJsonDocument(document).toJson()) > 0;
    };
    const QJsonObject manual{
            {"label", "manual"},
            {"shape_type", "rectangle"},
            {"points", QJsonArray{QJsonArray{1, 1}, QJsonArray{10, 10}}},
            {"description", "untouched"},
            {"fixture_extra", 17}};
    if (!write({{"imagePath", "text.jpg"},
                {"checked", true},
                {"fixture_meta", 42},
                {"shapes", QJsonArray{manual}}}))
        return 1;
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, image));
    if (!check(selectModel(window, "PP-OCRv4 (en)", "ppocr-rec-en-q8_0.gguf"),
               "select English OCR"))
        return 1;
    auto* backend = window->findChild<QComboBox*>("AnnotationBackend");
    int deviceIndex = -1;
    for (int i = 0; i < backend->count(); ++i)
        if (backend->itemData(i).toString().contains(device,
                                                     Qt::CaseInsensitive)) {
            deviceIndex = i;
            break;
        }
    if (!check(deviceIndex >= 0, "OCR requested backend is enumerated"))
        return 1;
    backend->setCurrentIndex(deviceIndex);
    auto* layout = window->findChild<QComboBox*>("AnnotationOcrLayout");
    auto* skip = window->findChild<QPushButton*>("AnnotationOcrSkipDetection");
    if (!check(layout && layout->count() == 4 &&
                       layout->currentData().toString().isEmpty() && skip &&
                       !skip->isHidden(),
               "OCR catalog layout picker defaults off and exposes skip "
               "detection"))
        return 1;
    QString status;
    qint64 elapsed = 0;
    if (!check(runAndWait(window, &status, &elapsed) &&
                       status.contains("text region"),
               "run actual OCR")) {
        std::fprintf(stderr, "%s\n", status.toUtf8().constData());
        return 1;
    }
    QJsonObject document = readJson(sidecar);
    const QJsonArray detected = document["shapes"].toArray();
    if (!check(!detected.isEmpty() && document["fixture_meta"] == 42 &&
                       !document["checked"].toBool(),
               "OCR replaces prior annotations, preserves document metadata "
               "and saves unchecked state"))
        return 1;
    for (int i = 0; i < detected.size(); ++i) {
        const auto shape = detected[i].toObject();
        if (!check(shape["label"] == "text" && shape["group_id"] == i &&
                           shape["score"].toDouble() >= 0.5 &&
                           !shape.contains("reading_order"),
                   "OCR default ignores cached layout and uses consecutive "
                   "text groups"))
            return 1;
        for (const auto& point : shape["points"].toArray())
            for (const auto& coordinate : point.toArray())
                if (!check(coordinate.toDouble() == coordinate.toInt(),
                           "OCR corners use source integer-pixel contract"))
                    return 1;
    }
    QJsonObject selected = detected.first().toObject();
    // Exercise the real window's arbitrary-polygon path, rather than only
    // the worker API. Five fractional points must survive recognition,
    // autosave and undo/redo without being replaced by the normalized quad.
    auto selectedPoints = selected["points"].toArray();
    if (!check(selectedPoints.size() == 4, "detected OCR quadrilateral"))
        return 1;
    const auto cornerA = selectedPoints[0].toArray();
    const auto cornerB = selectedPoints[1].toArray();
    selectedPoints.insert(
            1,
            QJsonArray{
                    (cornerA[0].toDouble() + cornerB[0].toDouble()) / 2 + .25,
                    (cornerA[1].toDouble() + cornerB[1].toDouble()) / 2 + .25});
    selected.insert("points", selectedPoints);
    selected.insert("shape_type", "polygon");
    selected.insert("label", "custom-text-label");
    selected.insert("description", "before recognition");
    selected.insert("score", 0.123);
    selected.insert("group_id", 91);
    selected.insert("difficult", true);
    selected.insert("flags", QJsonObject{{"verified", true}});
    selected.insert("attributes", QJsonObject{{"kind", "invoice"}});
    selected.insert("kie_linking", QJsonArray{QJsonArray{91, 92}});
    selected.insert("fixture_extra", 99);
    QJsonObject kept = detected.last().toObject();
    kept.insert("label", "manual");
    kept.insert("description", "untouched");
    kept.insert("fixture_extra", 17);
    QJsonObject locked = kept;
    locked.insert("locked", true);
    locked.insert("label", "locked-text");
    document.insert("checked", true);
    document.insert("shapes", QJsonArray{selected, kept, locked});
    if (!write(document)) return 1;
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, second));
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, image));
    auto* objects = window->findChild<QListWidget*>("AnnotationObjects");
    if (!check(objects && objects->count() == 3,
               "reload OCR selected-region fixture"))
        return 1;
    objects->item(0)->setSelected(true);
    objects->item(2)->setSelected(true);
    if (!check(runAndWait(window, &status, &elapsed,
                          "AnnotationOcrSkipDetection") &&
                       status.contains("text region"),
               "recognize selected custom-label region"))
        return 1;
    const auto updated = readJson(sidecar);
    const auto updatedShapes = updated["shapes"].toArray();
    const QJsonObject recognized = updatedShapes.first().toObject();
    const auto description = recognized.value("description");
    bool preserved = recognized.keys() == selected.keys();
    for (auto it = selected.constBegin(); it != selected.constEnd(); ++it)
        if (it.key() != "description")
            preserved &= recognized.value(it.key()) == it.value();
    if (!check(!description.toString().isEmpty() &&
                       description != "before recognition" && preserved &&
                       updatedShapes[1].toObject() == kept &&
                       updatedShapes[2].toObject() == locked &&
                       !updated["checked"].toBool(),
               "skip detection changes only the transcription, saves, and "
               "preserves score/geometry/metadata/unselected/locked shapes"))
        return 1;
    QAction* undo = nullptr;
    QAction* redo = nullptr;
    for (auto* action : window->findChildren<QAction*>()) {
        if (action->text() == "Undo") undo = action;
        if (action->text() == "Redo") redo = action;
    }
    if (!check(undo && redo && undo->isEnabled(),
               "OCR recognition has one undo checkpoint"))
        return 1;
    undo->trigger();
    if (!check(readJson(sidecar)["shapes"].toArray().first().toObject() ==
                       selected,
               "undo OCR transcription"))
        return 1;
    redo->trigger();
    if (!check(readJson(sidecar)["shapes"] == updated["shapes"],
               "redo OCR transcription")) {
        std::fprintf(stderr, "redo enabled=%d expected=%s actual=%s\n",
                     redo->isEnabled(),
                     QJsonDocument(updated["shapes"].toArray())
                             .toJson(QJsonDocument::Compact)
                             .constData(),
                     QJsonDocument(readJson(sidecar)["shapes"].toArray())
                             .toJson(QJsonDocument::Compact)
                             .constData());
        return 1;
    }
    if (!check(runAndWait(window, &status, &elapsed) &&
                       readJson(sidecar)["shapes"] == detected,
               "normal Run after skip performs detection and replaces the old "
               "objects"))
        return 1;
    const int layoutIndex = layout->findData("pplcnet-layout-q8_0.gguf");
    if (!check(layoutIndex >= 0, "layout model comes from catalog")) return 1;
    layout->setCurrentIndex(layoutIndex);
    if (!check(runAndWait(window, &status, &elapsed), "explicit layout run"))
        return 1;
    int regions = 0;
    for (const auto& value : readJson(sidecar)["shapes"].toArray()) {
        const auto shape = value.toObject();
        if (!shape.contains("reading_order")) continue;
        if (!check(shape["label"] != "layout" &&
                           !shape["label"].toString().isEmpty(),
                   "layout annotation displays its actual class name"))
            return 1;
        ++regions;
    }
    if (!check(regions > 0, "explicit layout stage produces regions")) return 1;
    layout->setCurrentIndex(0);
    if (!check(runAndWait(window, &status, &elapsed) &&
                       readJson(sidecar)["shapes"] == detected,
               "turning layout off invalidates the resident OCR session"))
        return 1;
    std::fprintf(stdout,
                 "OCR %s: %d text shapes; "
                 "skip/save/metadata/undo/redo/default/explicit-layout PASS "
                 "(%d layout regions)\n",
                 device.toUtf8().constData(), detected.size(), regions);
    return 0;
}

int runTagged(QMainWindow* window,
              const QTemporaryDir& work,
              const QString& root,
              const QString& device) {
    const QString path = work.filePath(QStringLiteral("horse.jpg"));
    if (!check(QFile::copy(
                       root + QStringLiteral(
                                      "relate_datasets/demo_images/horse.jpg"),
                       path),
               "copy tagged detector fixture"))
        return 1;
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, path));
    if (!check(selectModel(window, "yolov8s-world", "yolov8s-world-f16.gguf"),
               "select YOLO-World"))
        return 1;
    auto* backend = window->findChild<QComboBox*>("AnnotationBackend");
    int deviceIndex = -1;
    for (int i = 0; backend && i < backend->count(); ++i)
        if (backend->itemData(i).toString().contains(device,
                                                     Qt::CaseInsensitive)) {
            deviceIndex = i;
            break;
        }
    if (!check(deviceIndex >= 0, "composite requested backend is enumerated"))
        return 1;
    backend->setCurrentIndex(deviceIndex);
    auto* tagger = window->findChild<QComboBox*>("AnnotationObjectTagger");
    if (!check(tagger && !tagger->isHidden(), "object descriptions selector"))
        return 1;
    const int tagIndex = tagger->findData(
            QStringLiteral("ram_plus_swin_large_14m-q8_0.gguf"));
    if (!check(tagIndex > 0, "RAM++ catalog dependency")) return 1;
    tagger->setCurrentIndex(tagIndex);
    QLineEdit* text = nullptr;
    for (auto* field : window->findChildren<QLineEdit*>())
        if (field->placeholderText() ==
            QStringLiteral("Classes, comma separated"))
            text = field;
    if (!check(text, "world vocabulary field")) return 1;
    text->setText(QStringLiteral("person, horse"));
    auto* encoder = window->findChild<QComboBox*>("AnnotationTextEncoder");
    if (!check(encoder && encoder->findData(QStringLiteral(
                                  "clip-ViT-B-32-f16.gguf")) >= 0,
               "World text encoder"))
        return 1;
    encoder->setCurrentIndex(
            encoder->findData(QStringLiteral("clip-ViT-B-32-f16.gguf")));
    QString status;
    qint64 elapsed = 0;
    QTimer::singleShot(5, window, [tagger]() { tagger->setCurrentIndex(0); });
    if (!check(runAndWait(window, &status, &elapsed) &&
                       status.contains(QStringLiteral("changed")),
               "tagger change rejects in-flight result"))
        return 1;
    auto* objects = window->findChild<QListWidget*>("AnnotationObjects");
    if (!check(objects && objects->count() == 0,
               "stale tagged result preserves canvas"))
        return 1;
    tagger->setCurrentIndex(tagIndex);
    if (!check(runAndWait(window, &status, &elapsed),
               "tagged detection completed"))
        return 1;
    if (!check(status.contains(device, Qt::CaseInsensitive),
               "composite ran on the requested backend"))
        return 1;
    std::fprintf(stdout, "tagged status: %s (%lld ms)\n", qPrintable(status),
                 static_cast<long long>(elapsed));
    const QString sidecar = work.filePath(QStringLiteral("horse.json"));
    const auto document = readJson(sidecar);
    const auto shapes = document["shapes"].toArray();
    if (!check(!shapes.isEmpty() && document["tags"].toArray().isEmpty(),
               "object tags stay out of image-level tags"))
        return 1;
    for (const auto& value : shapes) {
        const auto shape = value.toObject();
        if (!check(!shape["description"].toString().isEmpty() &&
                           shape["shape_type"] == QStringLiteral("rectangle"),
                   "object description saved"))
            return 1;
        for (const auto& point : shape["points"].toArray()) {
            const auto p = point.toArray();
            if (!check(p[0].toDouble() == p[0].toInt() &&
                               p[1].toDouble() == p[1].toInt(),
                       "upstream integer crop endpoints"))
                return 1;
        }
    }
    QAction *undo = nullptr, *redo = nullptr;
    for (auto* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Undo")) undo = action;
        if (action->text() == QStringLiteral("Redo")) redo = action;
    }
    if (!check(undo && redo && undo->isEnabled(), "tagged result undo"))
        return 1;
    undo->trigger();
    if (!check(readJson(sidecar)["shapes"].toArray().isEmpty(),
               "undo tagged objects"))
        return 1;
    redo->trigger();
    if (!check(readJson(sidecar)["shapes"].toArray() == shapes,
               "redo tagged objects"))
        return 1;
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, path));
    if (!check(objects->count() == shapes.size(), "reload tagged objects"))
        return 1;
    objects->setCurrentRow(0);
    auto* description =
            window->findChild<QPlainTextEdit*>("AnnotationDescription");
    if (!check(description &&
                       description->toPlainText() ==
                               shapes[0].toObject()["description"].toString(),
               "reload object description"))
        return 1;
    auto* options = window->findChild<QPushButton*>("AnnotationRamOptions");
    if (!check(options && !options->isHidden() &&
                       setRamOptions(window, "zh", {}, {}) &&
                       runAndWait(window, &status, &elapsed),
               "object RAM exposes and applies the shared tag options"))
        return 1;
    const auto translated = readJson(sidecar)["shapes"].toArray();
    if (!check(translated.size() == shapes.size(),
               "tag options preserve detected objects"))
        return 1;
    for (int i = 0; i < shapes.size(); ++i) {
        const auto original = shapes[i].toObject();
        QStringList expected;
        for (const auto& name : original["description"].toString().split(" | "))
            expected.append(annotationRamTag(name, "zh"));
        auto actual = translated[i].toObject();
        if (!check(actual.take("description").toString() ==
                           expected.join(" | "),
                   "object descriptions translate after source-order tag "
                   "selection"))
            return 1;
        auto geometry = original;
        geometry.remove("description");
        if (!check(actual == geometry,
                   "RAM presentation options preserve object geometry and "
                   "metadata"))
            return 1;
    }
    if (!check(setRamOptions(window, "en", {}, {"not a model label"}) &&
                       runAndWait(window, &status, &elapsed),
               "object RAM accepts source keep-only filtering"))
        return 1;
    for (const auto& shape : readJson(sidecar)["shapes"].toArray())
        if (!check(shape.toObject()["description"].toString().isEmpty(),
                   "empty object tagging output clears the old description"))
            return 1;
    if (!check(setRamOptions(window, "en", {}, {}) &&
                       runAndWait(window, &status, &elapsed) &&
                       readJson(sidecar)["shapes"].toArray() == shapes,
               "clearing object RAM options restores the original complete "
               "result"))
        return 1;
    auto* confidence =
            window->findChild<QDoubleSpinBox*>("AnnotationConfidence");
    if (!check(confidence, "World confidence is exposed")) return 1;
    const double originalConfidence = confidence->value();
    confidence->setValue(1.0);
    if (!check(runAndWait(window, &status, &elapsed) &&
                       readJson(sidecar)["shapes"].toArray().isEmpty(),
               "successful zero-object composite result replaces old "
               "annotations"))
        return 1;
    undo->trigger();
    if (!check(readJson(sidecar)["shapes"].toArray() == shapes,
               "empty composite result has a working undo checkpoint"))
        return 1;
    confidence->setValue(originalConfidence);
    return 0;
}

int runClassifier(QMainWindow* window,
                  const QTemporaryDir& work,
                  const QString& root) {
    const QString image = work.filePath(QStringLiteral("bus.jpg"));
    const QString sidecar = work.filePath(QStringLiteral("bus.json"));
    if (!check(QFile::copy(root + "objects_detection_data/images/bus.jpg",
                           image),
               "copy classifier fixture"))
        return 1;
    const QJsonObject shape{
            {"label", "manual"},
            {"shape_type", "rectangle"},
            {"points", QJsonArray{QJsonArray{10, 10}, QJsonArray{100, 100}}},
            {"flags", QJsonObject{}},
            {"fixture_extra", 17}};
    for (const QString& name :
         {QStringLiteral("bus"), QStringLiteral("bus2")}) {
        if (name != "bus" && !QFile::copy(image, work.filePath(name + ".jpg")))
            return 1;
        QFile file(work.filePath(name + ".json"));
        if (!file.open(QIODevice::WriteOnly)) return 1;
        file.write(QJsonDocument(
                           QJsonObject{{"version", "4.1.3"},
                                       {"imagePath", QString(name + ".jpg")},
                                       {"imageData", QJsonValue::Null},
                                       {"flags",
                                        QJsonObject{{"old_manual_flag", true}}},
                                       {"shapes", QJsonArray{shape}},
                                       {"fixture_extra", "preserved"}})
                           .toJson());
    }
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, image));
    if (!check(selectModel(window, "yolo26n-cls", "yolo26n-cls-q8_0.gguf"),
               "select classifier model"))
        return 1;
    auto* action =
            window->findChild<QAction*>("AnnotationImageClassifierAction");
    if (!check(action && action->isEnabled(), "classifier action available"))
        return 1;
    bool valid = false;
    bool closedInFlight = false;
    QJsonObject savedFlags;
    QTimer::singleShot(0, window, [&]() {
        auto* dialog =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (!check(dialog &&
                           dialog->objectName() == "AnnotationImageClassifier",
                   "classifier dialog opened"))
            return;
        auto* ai = dialog->findChild<QPushButton*>("ClassifierAI");
        auto* page = dialog->findChild<QSpinBox*>("ClassifierPage");
        if (!check(ai && page, "classifier controls available")) {
            dialog->reject();
            return;
        }
        auto launch = [&](bool batch = false) {
            bool selected = false;
            QTimer choose;
            choose.setInterval(10);
            QObject::connect(&choose, &QTimer::timeout, [&]() {
                auto* input = qobject_cast<QInputDialog*>(
                        QApplication::activeModalWidget());
                if (!input) {
                    auto* range = qobject_cast<QDialog*>(
                            QApplication::activeModalWidget());
                    if (range &&
                        range->objectName() == "AnnotationBatchRangeDialog") {
                        range->findChild<QSpinBox*>("AnnotationBatchFrom")
                                ->setValue(1);
                        range->findChild<QSpinBox*>("AnnotationBatchTo")
                                ->setValue(2);
                        range->accept();
                    }
                    return;
                }
                auto* combo = input->findChild<QComboBox*>();
                const int index = combo ? combo->findText("yolo26n-cls") : -1;
                if (index < 0) {
                    input->reject();
                    return;
                }
                combo->setCurrentIndex(index);
                selected = true;
                input->accept();
            });
            choose.start();
            auto* action = dialog->findChild<QAction*>(
                    batch ? "ClassifierLocalAutoRun" : "ClassifierLocalAI");
            if (!action) return false;
            action->trigger();
            return selected;
        };
        if (!check(launch() && !ai->isEnabled() && !page->isEnabled(),
                   "classification disables conflicting controls")) {
            dialog->reject();
            return;
        }
        if (!check(waitFor(
                           [&]() {
                               return ai->isEnabled() &&
                                      dialog->findChildren<QCheckBox*>(
                                                    "ClassifierCategory")
                                                      .size() == 1000;
                           },
                           600000),
                   "classification returns all 1000 categories")) {
            dialog->reject();
            return;
        }
        const auto saved = readJson(sidecar);
        const auto flags = saved["flags"].toObject();
        savedFlags = flags;
        int selected = 0;
        for (auto it = flags.begin(); it != flags.end(); ++it)
            selected += it.value().toBool();
        valid = check(flags.size() == 1000 && selected == 1 &&
                              !flags.contains("old_manual_flag"),
                      "classification replaces flags with one trained class") &&
                check(saved["fixture_extra"] == "preserved" &&
                              saved["shapes"].toArray().size() == 1 &&
                              saved["shapes"].toArray()[0]
                                              .toObject()["fixture_extra"] ==
                                      17,
                      "classification auto-save preserves manual geometry and "
                      "metadata");
        if (valid) {
            valid = check(
                    launch(true) &&
                            waitFor(
                                    [&]() {
                                        return ai->isEnabled() &&
                                               window->findChild<QLabel*>(
                                                             "AnnotationStatus")
                                                       ->text()
                                                       .contains(
                                                               "Auto Run "
                                                               "completed");
                                    },
                                    600000),
                    "classifier batch completes and releases controls");
            for (const QString& name :
                 {QStringLiteral("bus"), QStringLiteral("bus2")}) {
                const auto batchDoc = readJson(work.filePath(name + ".json"));
                valid &= check(
                        batchDoc["flags"].toObject() == flags &&
                                batchDoc["shapes"].toArray().size() == 1 &&
                                batchDoc["shapes"]
                                                .toArray()[0]
                                                .toObject()["fixture_extra"] ==
                                        17,
                        "batch classification preserves each image's manual "
                        "geometry");
            }
        }
        if (valid) {
            closedInFlight = launch() && !ai->isEnabled();
            check(closedInFlight, "second classification starts before close");
        }
        dialog->reject();
    });
    action->trigger();
    auto* run = window->findChild<QPushButton*>("AnnotationRunModel");
    auto* status = window->findChild<QLabel*>("AnnotationStatus");
    if (!check(valid && closedInFlight && run && status &&
                       waitFor(
                               [&]() {
                                   return status->text().contains("changed");
                               },
                               600000),
               "closed classifier releases worker"))
        return 1;
    if (!check(readJson(sidecar)["flags"].toObject() == savedFlags,
               "closed classifier rejects in-flight result without rewriting "
               "flags"))
        return 1;
    std::fprintf(
            stdout,
            "Classifier real-model workflow: 1000 categories, one selected; "
            "two-image batch preserves geometry; close rejects stale result\n");
    return 0;
}
int runAttribute(QMainWindow* window,
                 const QTemporaryDir& work,
                 const QString& root,
                 const QString& device) {
    // Stage the local recipes only inside a temporary data root; the shared
    // model cache must not be modified.
    QTemporaryDir localRoot;
    if (!check(localRoot.isValid(), "Temporary attribute data root")) return 1;
    qputenv("CLOUDVIEWER_DATA_ROOT", localRoot.path().toUtf8());
    const QString image = work.filePath("bus.jpg");
    const QString sidecar = work.filePath("bus.json");
    const QString other = work.filePath("other.jpg");
    if (!check(QFile::copy(root + "objects_detection_data/images/bus.jpg",
                           image) &&
                       QFile::copy(image, other),
               "Copy the real attribute input"))
        return 1;
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, image));
    if (!check(selectModel(window, "PULC Person (local F32)",
                           "pulc-person-f32.gguf"),
               "The public local person recipe must appear in the picker"))
        return 1;
    auto* backend = window->findChild<QComboBox*>("AnnotationBackend");
    int deviceIndex = -1;
    for (int i = 0; backend && i < backend->count(); ++i)
        if (backend->itemData(i).toString().contains(device,
                                                     Qt::CaseInsensitive)) {
            deviceIndex = i;
            break;
        }
    if (!check(deviceIndex >= 0, "Real attribute backend enumeration"))
        return 1;
    backend->setCurrentIndex(deviceIndex);
    QString status;
    const QString directory = localRoot.path() + "/extract/attribute_models";
    if (!check(runAndWait(window, &status, nullptr) &&
                       status.contains(directory + "/pulc-person-f32.gguf") &&
                       !status.contains("Downloading"),
               "A missing local model must report its path and must not "
               "download"))
        return 1;
    QDir().mkpath(directory);
    if (!check(QFile::link(QString::fromLocal8Bit(qgetenv(
                                   "AICORE_TEST_ATTRIBUTE_PERSON_GGUF")),
                           directory + "/pulc-person-f32.gguf") &&
                       QFile::link(
                               QString::fromLocal8Bit(qgetenv(
                                       "AICORE_TEST_ATTRIBUTE_VEHICLE_GGUF")),
                               directory + "/pulc-vehicle-f32.gguf"),
               "Stage the real local weights"))
        return 1;
    auto* run = window->findChild<QPushButton*>("AnnotationRunModel");
    bool ignoresConfidence = false;
    for (auto* spin : window->findChildren<QDoubleSpinBox*>())
        ignoresConfidence |= !spin->isEnabled() &&
                             spin->toolTip().contains("built-in thresholds");
    if (!check(run && ignoresConfidence,
               "Attributes use the task thresholds and must not misuse "
               "detection confidence"))
        return 1;
    for (const QString& family :
         {QStringLiteral("person"), QStringLiteral("vehicle"),
          QStringLiteral("person")}) {
        if (!check(selectModel(window,
                               family == "person" ? "PULC Person (local F32)"
                                                  : "PULC Vehicle (local F32)",
                               "pulc-" + family + "-f32.gguf"),
                   "Switch the attribute model"))
            return 1;
        backend->setCurrentIndex(deviceIndex);
        if (!check(runAndWait(window, &status, nullptr) && run->isEnabled() &&
                           status.contains(device, Qt::CaseInsensitive),
                   "The attribute run completes and releases the UI"))
            return 1;
        const auto document = readJson(sidecar);
        const auto shapes = document["shapes"].toArray();
        const auto shape =
                shapes.isEmpty() ? QJsonObject{} : shapes[0].toObject();
        const auto attributes = shape["attributes"].toObject();
        if (!check(shapes.size() == 1 && shape["label"] == family &&
                           shape["shape_type"] == "rectangle" &&
                           attributes.size() == (family == "person" ? 21 : 2) &&
                           (family == "person"
                                    ? attributes["Gender"] == "Male" &&
                                              attributes["Direction"] == "Back"
                                    : attributes["Color"] == "white" &&
                                              attributes["Type"] == "truck") &&
                           document["tags"].toArray().isEmpty() &&
                           document["flags"].toObject().isEmpty(),
                   "Full-image attributes are saved and replace, and must not "
                   "become flags/tags or append boxes"))
            return 1;
    }
    const auto before = readJson(sidecar);
    QTimer::singleShot(0, window, [=]() {
        QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, other));
    });
    auto* statusLabel = window->findChild<QLabel*>("AnnotationStatus");
    // openImage updates the status first; wait for the real worker's stale
    // completion notice instead of treating load progress as completion.
    if (!check(runAndWait(window, &status, nullptr) && statusLabel &&
                       waitFor(
                               [&]() {
                                   return statusLabel->text().contains(
                                           "changed");
                               },
                               120000) &&
                       readJson(sidecar) == before &&
                       !QFileInfo::exists(work.filePath("other.json")),
               "A stale attribute result must not be written to the original "
               "or the new image"))
        return 1;
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, image));
    const auto beforeModelSwitch = readJson(sidecar);
    run->click();
    if (!check(statusLabel->text().startsWith("Running"),
               "Real attribute inference must already be submitted before "
               "switching models"))
        return 1;
    if (!check(selectModel(window, "PULC Vehicle (local F32)",
                           "pulc-vehicle-f32.gguf"),
               "Switch the attribute model during inference"))
        return 1;
    backend->setCurrentIndex(deviceIndex);
    if (!check(waitFor(
                       [&]() {
                           return statusLabel->text().contains("changed");
                       },
                       120000) &&
                       readJson(sidecar) == beforeModelSwitch,
               "The old model's result must not overwrite the current "
               "document"))
        return 1;
    if (!check(selectModel(window, "PULC Person (local F32)",
                           "pulc-person-f32.gguf"),
               "Restore the person model"))
        return 1;
    backend->setCurrentIndex(deviceIndex);

    QAction* autoRun = nullptr;
    for (auto* action : window->findChildren<QAction*>())
        if (action->property("qAnyToolbarId").toString() == "auto_run")
            autoRun = action;
    if (!check(autoRun, "Real Auto Run entry point")) return 1;
    const auto launchBatch = [&]() {
        bool selected = false;
        QTimer::singleShot(0, window, [&]() {
            auto* range =
                    window->findChild<QDialog*>("AnnotationBatchRangeDialog");
            auto* from =
                    range ? range->findChild<QSpinBox*>("AnnotationBatchFrom")
                          : nullptr;
            auto* to = range ? range->findChild<QSpinBox*>("AnnotationBatchTo")
                             : nullptr;
            if (!from || !to) return;
            from->setValue(1);
            to->setValue(2);
            selected = to->value() == 2;
            range->accept();
        });
        autoRun->trigger();
        return selected;
    };
    const auto beforeCancel = readJson(sidecar);
    if (!check(launchBatch() && statusLabel->text().startsWith("Running"),
               "The first image's real batch inference is submitted before "
               "cancelling"))
        return 1;
    auto* progress =
            window->findChild<QProgressDialog*>("AnnotationBatchProgress");
    auto* cancel = progress ? progress->findChild<QPushButton*>() : nullptr;
    if (!check(cancel, "Real batch cancel button")) return 1;
    cancel->click();
    if (!check(waitFor(
                       [&]() {
                           return statusLabel->text().contains("cancelled");
                       },
                       120000) &&
                       readJson(sidecar) == beforeCancel &&
                       !QFileInfo::exists(work.filePath("other.json")),
               "Cancelling discards the in-flight result and does not schedule "
               "the second image"))
        return 1;
    if (!check(launchBatch() &&
                       waitFor(
                               [&]() {
                                   return statusLabel->text().contains(
                                           "Auto Run completed (2 images)");
                               },
                               120000),
               "The full two-image batch can run again after cancelling"))
        return 1;
    for (const QString& name :
         {QStringLiteral("bus"), QStringLiteral("other")}) {
        const auto document = readJson(work.filePath(name + ".json"));
        const auto shapes = document["shapes"].toArray();
        if (!check(shapes.size() == 1 &&
                           shapes[0].toObject()["label"] == "person" &&
                           shapes[0].toObject()["attributes"]
                                           .toObject()
                                           .size() == 21 &&
                           document["imagePath"].toString() == name + ".jpg",
                   "Each image saves its own attributes and source identity "
                   "and must not append boxes"))
            return 1;
    }

    // The model file exists but is invalid: this must go through the real
    // worker error path, not a cache-missing precondition check.
    if (!check(selectModel(window, "PULC Vehicle (local F32)",
                           "pulc-vehicle-f32.gguf"),
               "Switch the context before the error test"))
        return 1;
    backend->setCurrentIndex(deviceIndex);
    if (!check(runAndWait(window, &status, nullptr),
               "Create the real vehicle context"))
        return 1;
    const auto beforeError = readJson(work.filePath("other.json"));
    const QString personPath = directory + "/pulc-person-f32.gguf";
    if (!check(QFile::rename(personPath, personPath + ".saved"),
               "Park the temporary recipe link"))
        return 1;
    {
        QFile corrupt(personPath);
        if (!check(corrupt.open(QIODevice::WriteOnly),
                   "Create the temporary invalid model"))
            return 1;
        corrupt.write("invalid GGUF");
    }
    if (!check(selectModel(window, "PULC Person (local F32)",
                           "pulc-person-f32.gguf"),
               "Select the invalid local model"))
        return 1;
    backend->setCurrentIndex(deviceIndex);
    if (!check(runAndWait(window, &status, nullptr) && !status.isEmpty() &&
                       !status.contains("attributes:") &&
                       !status.contains("Downloading") &&
                       readJson(work.filePath("other.json")) == beforeError,
               "A real model load failure releases the in-flight state and "
               "writes no document"))
        return 1;
    if (!check(QFile::remove(personPath) &&
                       QFile::rename(personPath + ".saved", personPath) &&
                       runAndWait(window, &status, nullptr) &&
                       status.contains("attributes:"),
               "Real attribute inference can recover after a worker error"))
        return 1;

    const auto beforeClose = readJson(work.filePath("other.json"));
    QPointer<QMainWindow> lifetime(window);
    run->click();
    if (!check(statusLabel->text().startsWith("Running"),
               "Real inference is submitted before teardown"))
        return 1;
    window->close();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    if (!check(lifetime.isNull() &&
                       readJson(work.filePath("other.json")) == beforeClose,
               "In-flight teardown waits for the worker to exit and must not "
               "save the late result"))
        return 1;
    std::fprintf(stdout,
                 "attribute %s: "
                 "picker/local-error/replace/attributes/save/switch/"
                 "source-stale/model-stale/batch-cancel/restart/worker-error/"
                 "inflight-teardown PASS\n",
                 device.toUtf8().constData());
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTemporaryDir settingsDir;
    QTemporaryDir workDir;
    if (!settingsDir.isValid() || !workDir.isValid() || (argc < 2 || argc > 4))
        return 2;
    const bool taggedOnly = argc >= 3 && QByteArray(argv[2]) == "--tagged";
    const bool classifierOnly =
            argc == 3 && QByteArray(argv[2]) == "--classifier";
    const bool ocrOnly = argc >= 3 && QByteArray(argv[2]) == "--ocr";
    const bool attributeOnly =
            argc >= 3 && QByteArray(argv[2]) == "--attribute";
    if (argc >= 3 && !taggedOnly && !classifierOnly && !ocrOnly &&
        !attributeOnly)
        return 2;
    QCoreApplication::setOrganizationName(QStringLiteral("ACloudViewerTests"));
    QCoreApplication::setApplicationName(QStringLiteral("qAnyLabelingRunPath"));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settingsDir.path());

    const QString root = dataRoot() + QStringLiteral("/extract/");
    const QString ramModel =
            QStringLiteral("ram_plus_swin_large_14m-q8_0.gguf");
    const QString relModel = QStringLiteral("relsgg-vits16plus-q8_0.gguf");
    const QString demo = root + QStringLiteral("ram_datasets/demo/demo1.jpg");
    const QString horse =
            root + QStringLiteral("relate_datasets/demo_images/horse.jpg");
    const QStringList required =
            attributeOnly
                    ? QStringList{QString::fromLocal8Bit(qgetenv(
                                          "AICORE_TEST_ATTRIBUTE_PERSON_GGUF")),
                                  QString::fromLocal8Bit(
                                          qgetenv("AICORE_TEST_ATTRIBUTE_"
                                                  "VEHICLE_GGUF")),
                                  root + "objects_detection_data/images/"
                                         "bus.jpg"}
            : ocrOnly
                    ? QStringList{root + "ocr_models/ppocr-det-ch-q8_0.gguf",
                                  root + "ocr_models/ppocr-rec-en-q8_0.gguf",
                                  root + "ocr_models/pplcnet-layout-q8_0.gguf",
                                  root + "ppocr_testdata/scenes/ppocrv4_en.jpg"}
            : classifierOnly
                    ? QStringList{root + "yolo_models/yolo26n-cls-q8_0.gguf",
                                  root + "objects_detection_data/images/"
                                         "bus.jpg"}
            : taggedOnly
                    ? QStringList{root + "ram_models/" + ramModel, horse,
                                  root + "yolo_models/yolov8s-world-f16.gguf",
                                  root + "yolo_models/clip-ViT-B-32-f16.gguf"}
                    : QStringList{
                              root + QStringLiteral("ram_models/") + ramModel,
                              root + "ram_models/tag2text_swin_14m-q8_0.gguf",
                              root + QStringLiteral("relation_models/") +
                                      relModel,
                              demo, horse};
    for (const QString& needed : required) {
        if (!QFileInfo::exists(needed)) {
            std::fprintf(
                    stderr, "%s: missing %s\n",
                    taggedOnly || classifierOnly || ocrOnly || attributeOnly
                            ? "FAIL"
                            : "SKIP",
                    needed.toLocal8Bit().constData());
            return taggedOnly || classifierOnly || ocrOnly || attributeOnly
                           ? 1
                           : 77;
        }
    }

    QPluginLoader loader(QString::fromLocal8Bit(argv[1]));
    auto* plugin = qobject_cast<ccStdPluginInterface*>(loader.instance());
    if (!plugin)
        std::fprintf(stderr, "loader: %s\n",
                     loader.errorString().toLocal8Bit().constData());
    if (!check(plugin && plugin->getActions().size() == 1,
               "plugin failed to load"))
        return 1;
    plugin->getActions().first()->trigger();
    QApplication::processEvents();
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets())
        if (widget->objectName() == QStringLiteral("qAnyLabelingWorkspace"))
            window = qobject_cast<QMainWindow*>(widget);
    if (!check(window, "workspace window not found")) return 1;
    if (taggedOnly)
        return runTagged(window, workDir, root,
                         argc == 4 ? QString::fromLocal8Bit(argv[3])
                                   : QStringLiteral("cpu"));
    if (attributeOnly)
        return runAttribute(window, workDir, root,
                            argc == 4 ? QString::fromLocal8Bit(argv[3])
                                      : QStringLiteral("cpu"));
    if (classifierOnly) return runClassifier(window, workDir, root);
    if (ocrOnly)
        return runOcr(window, workDir, root,
                      argc == 4 ? QString::fromLocal8Bit(argv[3])
                                : QStringLiteral("cpu"));

    // ---------------------------------------------------------------- RAM
    const QString ramImage = workDir.filePath(QStringLiteral("demo1.jpg"));
    if (!check(QFile::copy(demo, ramImage), "copy RAM sample")) return 1;
    {
        // A reviewed (checked) image: a model result that changes its tags
        // must send it back to unchecked, like upstream.
        QFile out(workDir.filePath(QStringLiteral("demo1.json")));
        if (!check(out.open(QIODevice::WriteOnly), "write RAM sidecar"))
            return 1;
        out.write(QJsonDocument(
                          QJsonObject{{QStringLiteral("version"),
                                       QStringLiteral("4.0.6")},
                                      {QStringLiteral("flags"), QJsonObject()},
                                      {QStringLiteral("shapes"), QJsonArray()},
                                      {QStringLiteral("checked"), true},
                                      {QStringLiteral("imagePath"),
                                       QStringLiteral("demo1.jpg")},
                                      {QStringLiteral("imageData"),
                                       QJsonValue::Null}})
                          .toJson());
    }
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, ramImage));
    QApplication::processEvents();
    if (!check(selectModel(window, QStringLiteral("ram_plus_swin_large_14m"),
                           ramModel),
               "RAM++ must be selectable in the model picker"))
        return 1;
    bool confidenceDisabled = false;
    for (auto* spin : window->findChildren<QDoubleSpinBox*>())
        if (!spin->isEnabled() &&
            spin->toolTip().contains(QStringLiteral("built-in thresholds")))
            confidenceDisabled = true;
    if (!check(confidenceDisabled,
               "RAM must disable the confidence control it does not use"))
        return 1;

    QString status;
    qint64 firstMs = 0, secondMs = 0;
    if (!check(runAndWait(window, &status, &firstMs), "RAM run did not finish"))
        return 1;
    const QString ramSidecar = workDir.filePath(QStringLiteral("demo1.json"));
    if (!check(waitFor([&]() { return QFileInfo::exists(ramSidecar); }, 20000),
               "RAM run must save the sidecar (auto-save)"))
        return 1;
    const QJsonObject ramDoc = readJson(ramSidecar);
    if (!check(!ramDoc.value(QStringLiteral("checked")).toBool(),
               "A tag-changing result must clear the checked flag"))
        return 1;
    const QJsonArray tags = ramDoc.value(QStringLiteral("tags")).toArray();
    if (!check(!tags.isEmpty() && tags.contains(QStringLiteral("couch")),
               "RAM tags must land in the sidecar 'tags' array"))
        return 1;
    if (!check(ramDoc.value(QStringLiteral("shapes")).toArray().isEmpty(),
               "RAM must not create object shapes"))
        return 1;
    auto* tagList =
            window->findChild<QListWidget*>(QStringLiteral("ImageTags"));
    if (!check(tagList && tagList->count() == tags.size(),
               "The image-tags strip must show exactly the stored tags"))
        return 1;
    std::fprintf(stdout, "RAM status: %s\n", status.toLocal8Bit().constData());

    if (!check(runAndWait(window, &status, &secondMs),
               "second RAM run did not finish"))
        return 1;
    if (!check(readJson(ramSidecar).value(QStringLiteral("tags")).toArray() ==
                       tags,
               "Re-running RAM must replace, not append, the tags"))
        return 1;
    std::fprintf(stdout,
                 "RAM run time: first=%lld ms (includes model load) "
                 "second=%lld ms\n",
                 static_cast<long long>(firstMs),
                 static_cast<long long>(secondMs));

    const QString firstTag = tags.first().toString();
    if (!check(setRamOptions(window, "zh", {}, {firstTag}) &&
                       runAndWait(window, &status, &secondMs) &&
                       readJson(ramSidecar)["tags"].toArray() ==
                               QJsonArray{annotationRamTag(firstTag, "zh")
                                                  .trimmed()},
               "RAM Chinese output and keep-only option reach the real worker "
               "and sidecar"))
        return 1;
    auto* reviewed = window->findChild<QCheckBox*>("AnnotationChecked");
    if (!check(reviewed, "review checkbox is exposed")) return 1;
    reviewed->setChecked(true);
    if (!check(runAndWait(window, &status, &secondMs) &&
                       readJson(ramSidecar)["checked"].toBool(),
               "unchanged normalized Chinese tags preserve reviewed state"))
        return 1;
    QJsonArray withoutFirst = tags;
    withoutFirst.removeAt(0);
    if (!check(setRamOptions(window, "en", {firstTag}, {firstTag}) &&
                       runAndWait(window, &status, &secondMs) &&
                       readJson(ramSidecar)["tags"].toArray() == withoutFirst,
               "RAM delete-tags takes precedence over keep-only, matching "
               "upstream"))
        return 1;
    if (!check(setRamOptions(window, "en", {}, {"not a model label"}) &&
                       runAndWait(window, &status, &secondMs) &&
                       readJson(ramSidecar)["tags"].toArray().isEmpty(),
               "RAM empty filtered output clears previously saved tags"))
        return 1;
    if (!check(setRamOptions(window, "en", {}, {}) &&
                       runAndWait(window, &status, &secondMs) &&
                       readJson(ramSidecar)["tags"].toArray() == tags,
               "RAM clearing options restores embedded per-class thresholds "
               "and original tags"))
        return 1;

    // Tag2Text adds an image caption; it must survive auto-save and reload,
    // rather than existing only in the transient status message.
    if (!check(selectModel(window, "tag2text_swin_14m",
                           "tag2text_swin_14m-q8_0.gguf") &&
                       runAndWait(window, &status, &secondMs),
               "Tag2Text caption run completes"))
        return 1;
    const auto captionDoc = readJson(ramSidecar);
    const QString caption = captionDoc["description"].toString();
    if (!check(!caption.trimmed().isEmpty() &&
                       !captionDoc["tags"].toArray().isEmpty() &&
                       captionDoc["shapes"].toArray().isEmpty(),
               "Tag2Text saves its caption and tags without object shapes"))
        return 1;
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, ramImage));
    auto* captionEditor =
            window->findChild<QPlainTextEdit*>("AnnotationDescription");
    if (!check(captionEditor && captionEditor->toPlainText() == caption,
               "Tag2Text caption is visible after reloading the sidecar"))
        return 1;

    // ----------------------------------------------------------- Relate
    const QString relImage = workDir.filePath(QStringLiteral("horse.jpg"));
    if (!check(QFile::copy(horse, relImage), "copy relate sample")) return 1;
    const auto rectangle = [](const QString& label, double x1, double y1,
                              double x2, double y2) {
        return QJsonObject{
                {QStringLiteral("label"), label},
                {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
                {QStringLiteral("points"),
                 QJsonArray{QJsonArray{x1, y1}, QJsonArray{x2, y2}}},
                {QStringLiteral("flags"), QJsonObject()}};
    };
    // Person and horse boxes YOLO26s q8_0 detects on this 1280x907 sample.
    const QJsonObject horseDoc{
            {QStringLiteral("version"), QStringLiteral("4.0.6")},
            {QStringLiteral("checked"), true},
            {QStringLiteral("flags"), QJsonObject()},
            {QStringLiteral("shapes"),
             QJsonArray{rectangle(QStringLiteral("person"), 475.3, 136.3, 643.8,
                                  626.1),
                        rectangle(QStringLiteral("horse"), 131.3, 315.7, 990.9,
                                  867.1)}},
            {QStringLiteral("imagePath"), QStringLiteral("horse.jpg")},
            {QStringLiteral("imageData"), QJsonValue::Null},
            {QStringLiteral("imageHeight"), 907},
            {QStringLiteral("imageWidth"), 1280}};
    const QString relSidecar = workDir.filePath(QStringLiteral("horse.json"));
    {
        QFile out(relSidecar);
        if (!check(out.open(QIODevice::WriteOnly), "write relate sidecar"))
            return 1;
        out.write(QJsonDocument(horseDoc).toJson());
    }
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, relImage));
    QApplication::processEvents();
    if (!check(selectModel(window, QStringLiteral("relsgg-vits16plus"),
                           relModel),
               "RelateAnything must be selectable in the model picker"))
        return 1;
    auto* objects = window->findChild<QListWidget*>(
            QStringLiteral("AnnotationObjects"));
    auto* relationTable = window->findChild<QTableWidget*>(
            QStringLiteral("AnnotationRelationTable"));
    if (!check(objects && objects->count() == 2 && relationTable,
               "relate sample must load two objects and the relation table"))
        return 1;

    // 1) one rectangle selected is not enough and the detector stage is
    //    unavailable without its cached model, OR (with it) the detector
    //    proposes the objects. Both are legitimate; assert whichever applies.
    objects->clearSelection();
    objects->item(0)->setSelected(true);
    QApplication::processEvents();
    const bool detectorCached = QFileInfo::exists(
            root + QStringLiteral("yolo_models/yoloe-26s-seg-pf-f16.gguf"));
    if (!detectorCached) {
        if (!check(runAndWait(window, &status, nullptr) &&
                           status.contains(QStringLiteral("at least two")),
                   "Run without two rectangles and without the detector must "
                   "say what to do"))
            return 1;
        if (!check(relationTable->rowCount() == 0, "no edges without a run"))
            return 1;
    }

    // 1b) the detector stage: with nothing selected and the official
    //     detector cached, the run adds the detected objects as real
    //     rectangles and the edges refer to them.
    if (detectorCached) {
        objects->clearSelection();
        QApplication::processEvents();
        const int shapesBefore = objects->count();
        const bool detectorDone = runAndWait(window, &status, nullptr);
        if (!detectorDone || !status.contains(QStringLiteral("relation")))
            std::fprintf(stderr, "detector-stage status=[%s] done=%d\n",
                         status.toLocal8Bit().constData(), int(detectorDone));
        if (!check(detectorDone && status.contains(QStringLiteral("relation")),
                   "detector stage must report its relations"))
            return 1;
        std::fprintf(stdout, "Detector-stage status: %s\n",
                     status.toLocal8Bit().constData());
        if (!check(objects->count() > shapesBefore,
                   "the detector stage must add its objects as rectangles"))
            return 1;
        if (!check(relationTable->rowCount() >= 1,
                   "the detector stage must list at least one edge"))
            return 1;
        const QString detectedSubject = relationTable->item(0, 0)->text();
        const QString detectedObject = relationTable->item(0, 2)->text();
        std::fprintf(
                stdout, "Detector-stage edge: %s -[%s]-> %s\n",
                detectedSubject.toLocal8Bit().constData(),
                relationTable->item(0, 1)->text().toLocal8Bit().constData(),
                detectedObject.toLocal8Bit().constData());
        // Clicking the first edge selects its subject on the canvas, which
        // only works if the edge points at a real, newly added rectangle.
        QMetaObject::invokeMethod(relationTable, "cellClicked", Q_ARG(int, 0),
                                  Q_ARG(int, 0));
        QApplication::processEvents();
        if (!check(objects->selectedItems().size() == 1 &&
                           objects->row(objects->selectedItems().first()) >=
                                   shapesBefore,
                   "an edge of the detector stage must select one of the "
                   "added rectangles"))
            return 1;
        // The detected rectangles are real annotations, so auto-save wrote
        // them to the sidecar (that is the point of this stage).
        if (!check(readJson(relSidecar)
                                   .value(QStringLiteral("shapes"))
                                   .toArray()
                                   .size() > 2,
                   "the detector stage must save its rectangles in the "
                   "sidecar"))
            return 1;
        // Restore the two-shape sample for the remaining checks.
        {
            QFile out(relSidecar);
            if (!check(out.open(QIODevice::WriteOnly),
                       "restore the relate sidecar"))
                return 1;
            out.write(QJsonDocument(horseDoc).toJson());
        }
        QMetaObject::invokeMethod(window, "openImage",
                                  Q_ARG(QString, ramImage));
        QApplication::processEvents();
        QMetaObject::invokeMethod(window, "openImage",
                                  Q_ARG(QString, relImage));
        QApplication::processEvents();
        if (!check(objects->count() == 2,
                   "the restored sample must have two objects again"))
            return 1;
    }

    // 1c) the detector is a choice (as in the standalone plugin): pick an
    //     RF-DETR model in the Options popup and the stage runs with it.
    if (QFileInfo::exists(
                root + QStringLiteral("rfdetr_models/rfdetr-nano-f16.gguf"))) {
        if (auto* panel =
                    window->findChild<QWidget*>(QStringLiteral("AIModel")))
            panel->show();
        QApplication::processEvents();
        auto* optionsButton = window->findChild<QPushButton*>(
                QStringLiteral("AnnotationRelateOptions"));
        const auto chooseDetector = [&](const QString& file) {
            QTimer pick;
            pick.setInterval(50);
            QObject::connect(&pick, &QTimer::timeout, [&]() {
                for (QWidget* top : QApplication::topLevelWidgets()) {
                    if (top->objectName() !=
                                QStringLiteral("RelateOptionsDialog") ||
                        !top->isVisible())
                        continue;
                    if (auto* combo = top->findChild<QComboBox*>(
                                QStringLiteral("RelateDetector")))
                        combo->setCurrentIndex(combo->findData(file));
                    if (auto* ok = top->findChild<QPushButton*>(
                                QStringLiteral("RelateOptionsOk"))) {
                        pick.stop();
                        ok->click();
                        return;
                    }
                }
            });
            pick.start();
            optionsButton->click();
            pick.stop();
        };
        if (!check(optionsButton && optionsButton->isVisibleTo(window),
                   "the Options button must be visible for RelateAnything"))
            return 1;
        chooseDetector(QStringLiteral("rfdetr-nano-f16.gguf"));
        objects->clearSelection();
        QApplication::processEvents();
        const int before = objects->count();
        const bool ok = runAndWait(window, &status, nullptr);
        std::fprintf(stdout, "RF-DETR detector-stage status: %s\n",
                     status.toLocal8Bit().constData());
        if (!check(ok && status.contains(QStringLiteral("relation")) &&
                           objects->count() > before,
                   "the detector stage must run with the RF-DETR choice and "
                   "add its objects"))
            return 1;
        // YOLO-World is open vocabulary: the run bakes the official 497
        // object names through the CLIP text encoder before detecting.
        if (QFileInfo::exists(root +
                              QStringLiteral("yolo_models/yolov8s-worldv2-f16."
                                             "gguf")) &&
            QFileInfo::exists(root +
                              QStringLiteral("yolo_models/"
                                             "clip-ViT-B-32-f16.gguf"))) {
            chooseDetector(QStringLiteral("yolov8s-worldv2-f16.gguf"));
            objects->clearSelection();
            QApplication::processEvents();
            const int worldBefore = objects->count();
            const bool worldOk = runAndWait(window, &status, nullptr);
            std::fprintf(stdout, "YOLO-World detector-stage status: %s\n",
                         status.toLocal8Bit().constData());
            if (!check(worldOk && status.contains(QStringLiteral("relation")) &&
                               objects->count() > worldBefore,
                       "the detector stage must run with the YOLO-World "
                       "choice and add its objects"))
                return 1;
        }
        chooseDetector(QStringLiteral("yoloe-26s-seg-pf-f16.gguf"));
        {
            QFile out(relSidecar);
            if (!check(out.open(QIODevice::WriteOnly),
                       "restore the relate sidecar"))
                return 1;
            out.write(QJsonDocument(horseDoc).toJson());
        }
        QMetaObject::invokeMethod(window, "openImage",
                                  Q_ARG(QString, ramImage));
        QApplication::processEvents();
        QMetaObject::invokeMethod(window, "openImage",
                                  Q_ARG(QString, relImage));
        QApplication::processEvents();
        if (!check(objects->count() == 2,
                   "the restored sample must have two objects again"))
            return 1;
    }

    // 2) two rectangles selected: real inference.
    QWidget* canvas = findCanvas(window);
    if (!check(canvas, "canvas not found")) return 1;
    objects->selectAll();
    QApplication::processEvents();
    const QImage before = canvas->grab().toImage();
    if (!check(runAndWait(window, &status, nullptr) &&
                       status.contains(QStringLiteral("relation")),
               "relate run must report its relations"))
        return 1;
    std::fprintf(stdout, "Relate status: %s\n",
                 status.toLocal8Bit().constData());
    if (!check(relationTable->rowCount() >= 1,
               "relate run must list at least one edge"))
        return 1;
    const QString subject = relationTable->item(0, 0)->text();
    const QString object = relationTable->item(0, 2)->text();
    const double score = relationTable->item(0, 4)->text().toDouble();
    std::fprintf(stdout, "Relate edge: %s -[%s %s]-> %s\n",
                 subject.toLocal8Bit().constData(),
                 relationTable->item(0, 1)->text().toLocal8Bit().constData(),
                 relationTable->item(0, 4)->text().toLocal8Bit().constData(),
                 object.toLocal8Bit().constData());
    if (!check(subject == QStringLiteral("person") &&
                       object == QStringLiteral("horse") && score >= 0.5,
               "person must relate to horse above the 0.5 floor"))
        return 1;
    if (!check(canvas->grab().toImage() != before,
               "the relation overlay must be painted on the canvas"))
        return 1;
    // Relations are display-only: the sidecar still has exactly two shapes.
    if (!check(readJson(relSidecar)
                                       .value(QStringLiteral("shapes"))
                                       .toArray()
                                       .size() == 2 &&
                       !readJson(relSidecar)
                                .contains(QStringLiteral("relations")),
               "relations must not be written to the sidecar"))
        return 1;

    // Display-only edges leave a reviewed image checked and unsaved.
    QCheckBox* checkedBox = nullptr;
    for (auto* box : window->findChildren<QCheckBox*>())
        if (box->text() == QStringLiteral("Checked")) checkedBox = box;
    if (!check(checkedBox && checkedBox->isChecked() &&
                       readJson(relSidecar)
                               .value(QStringLiteral("checked"))
                               .toBool(),
               "RelateAnything must not uncheck or rewrite a reviewed image"))
        return 1;

    // 3) clicking a row selects the subject box.
    QMetaObject::invokeMethod(relationTable, "cellClicked", Q_ARG(int, 0),
                              Q_ARG(int, 0));
    QApplication::processEvents();
    if (!check(objects->selectedItems().size() == 1 &&
                       objects->row(objects->selectedItems().first()) == 0,
               "clicking an edge must select its subject (the person box)"))
        return 1;

    // 4) Auto Run is refused up front for relate.
    QAction* autoRun = nullptr;
    for (QAction* action : window->findChildren<QAction*>())
        if (action->property("qAnyToolbarId").toString() ==
            QStringLiteral("auto_run"))
            autoRun = action;
    auto* statusLabel =
            window->findChild<QLabel*>(QStringLiteral("AnnotationStatus"));
    if (!check(autoRun && statusLabel, "Auto Run action not found")) return 1;
    statusLabel->setText(QString());
    autoRun->trigger();
    QApplication::processEvents();
    if (!check(statusLabel->text().contains(
                       QStringLiteral("not available in Auto Run")),
               "Auto Run must refuse RelateAnything with a reason"))
        return 1;

    // 4b) the Options popup: custom predicates are encoded by the relation
    //     model's own text student, so the run may only name those words.
    {
        auto* optionsButton = window->findChild<QPushButton*>(
                QStringLiteral("AnnotationRelateOptions"));
        // The AI panel starts hidden; show it the way the user does.
        if (auto* panel =
                    window->findChild<QWidget*>(QStringLiteral("AIModel")))
            panel->show();
        QApplication::processEvents();
        auto* runButton = window->findChild<QPushButton*>(
                QStringLiteral("AnnotationRunModel"));
        if (optionsButton && !optionsButton->isVisibleTo(window))
            std::fprintf(
                    stderr,
                    "options: hidden=%d  run visibleTo=%d  panel "
                    "visibleTo=%d  family=[%s]\n",
                    int(optionsButton->isHidden()),
                    int(runButton && runButton->isVisibleTo(window)),
                    int(window->findChild<QWidget*>(QStringLiteral("AIModel"))
                                ->isVisibleTo(window)),
                    window->findChild<QComboBox*>(
                                  QStringLiteral("AnnotationModelFamily"))
                            ->currentText()
                            .toLocal8Bit()
                            .constData());
        if (!check(optionsButton && optionsButton->isVisibleTo(window),
                   "the Options button must be visible for RelateAnything"))
            return 1;
        QTimer fill;
        fill.setInterval(50);
        QObject::connect(&fill, &QTimer::timeout, [&]() {
            for (QWidget* top : QApplication::topLevelWidgets()) {
                if (top->objectName() !=
                            QStringLiteral("RelateOptionsDialog") ||
                    !top->isVisible())
                    continue;
                if (auto* edit = top->findChild<QLineEdit*>(
                            QStringLiteral("RelatePredicates")))
                    edit->setText(QStringLiteral("riding | holding"));
                // The floor is lowered so the two words can surface.
                if (auto* floorBox = top->findChild<QDoubleSpinBox*>(
                            QStringLiteral("RelateRelationThreshold")))
                    floorBox->setValue(0.05);
                if (auto* ok = top->findChild<QPushButton*>(
                            QStringLiteral("RelateOptionsOk"))) {
                    fill.stop();
                    ok->click();
                    return;
                }
            }
        });
        fill.start();
        optionsButton->click();
        fill.stop();
        objects->selectAll();
        QApplication::processEvents();
        if (!check(runAndWait(window, &status, nullptr) &&
                           status.contains(QStringLiteral("relation")),
                   "a custom-predicate run must report its relations"))
            return 1;
        std::fprintf(stdout, "Custom-predicate status: %s\n",
                     status.toLocal8Bit().constData());
        bool onlyChosen = relationTable->rowCount() >= 1;
        for (int row = 0; row < relationTable->rowCount(); ++row) {
            const QString word = relationTable->item(row, 1)->text();
            std::fprintf(stdout, "  custom edge: %s\n",
                         word.toLocal8Bit().constData());
            onlyChosen = onlyChosen && (word == QStringLiteral("riding") ||
                                        word == QStringLiteral("holding"));
        }
        if (!check(onlyChosen,
                   "a custom vocabulary must restrict the edges to its "
                   "predicates"))
            return 1;
    }

    // 5) editing the shapes invalidates the index-based overlay.
    objects->selectAll();
    QApplication::processEvents();
    QKeyEvent del(QEvent::KeyPress, Qt::Key_Delete, Qt::NoModifier);
    QCoreApplication::sendEvent(canvas, &del);
    QApplication::processEvents();
    if (!check(relationTable->rowCount() == 0,
               "deleting shapes must clear the stale relation table"))
        return 1;

    // 6) Auto Run with a tagging model over the whole folder: every image
    //    gets tags, and a tags-only result must not replace its shapes.
    const QString horseShapesBefore = QString::fromUtf8(
            QJsonDocument(readJson(relSidecar)
                                  .value(QStringLiteral("shapes"))
                                  .toArray())
                    .toJson(QJsonDocument::Compact));
    // Put two shapes back on horse.json so "kept" is observable.
    {
        QJsonObject doc = horseDoc;
        QFile out(relSidecar);
        if (!check(out.open(QIODevice::WriteOnly), "rewrite horse sidecar"))
            return 1;
        out.write(QJsonDocument(doc).toJson());
    }
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, ramImage));
    QApplication::processEvents();
    QMetaObject::invokeMethod(window, "openImage", Q_ARG(QString, relImage));
    QApplication::processEvents();
    if (!check(selectModel(window, QStringLiteral("ram_plus_swin_large_14m"),
                           ramModel),
               "RAM++ must be selectable for Auto Run"))
        return 1;
    // Accept the modal range dialog (all images) as soon as it appears.
    QTimer acceptRange;
    acceptRange.setInterval(50);
    QObject::connect(&acceptRange, &QTimer::timeout, [&]() {
        for (QWidget* top : QApplication::topLevelWidgets()) {
            if (top->objectName() !=
                        QStringLiteral("AnnotationBatchRangeDialog") ||
                !top->isVisible())
                continue;
            // The range starts at the current image; cover the whole folder.
            if (auto* from = top->findChild<QSpinBox*>(
                        QStringLiteral("AnnotationBatchFrom")))
                from->setValue(1);
            for (auto* button : top->findChildren<QPushButton*>())
                if (button->text() == QStringLiteral("Run")) {
                    button->click();
                    acceptRange.stop();
                    return;
                }
        }
    });
    acceptRange.start();
    statusLabel->setText(QString());
    autoRun->trigger();
    const bool finished = waitFor(
            [&]() {
                return statusLabel->text().startsWith(
                               QStringLiteral("Auto Run completed")) ||
                       statusLabel->text().startsWith(
                               QStringLiteral("Auto Run stopped"));
            },
            600000);
    acceptRange.stop();
    if (!finished || !statusLabel->text().startsWith(
                             QStringLiteral("Auto Run completed (2")))
        std::fprintf(stderr, "Auto Run status=[%s] files=%d finished=%d\n",
                     statusLabel->text().toLocal8Bit().constData(),
                     window->findChild<QListWidget*>(
                                   QStringLiteral("AnnotationFiles"))
                             ->count(),
                     int(finished));
    if (!check(finished && statusLabel->text().startsWith(
                                   QStringLiteral("Auto Run completed (2")),
               "Auto Run over two images must complete"))
        return 1;
    const QJsonObject ramBatch = readJson(ramSidecar);
    const QJsonObject horseBatch = readJson(relSidecar);
    if (!check(!ramBatch.value(QStringLiteral("tags")).toArray().isEmpty() &&
                       !horseBatch.value(QStringLiteral("tags"))
                                .toArray()
                                .isEmpty(),
               "Auto Run must write tags for every image in the range"))
        return 1;
    if (!check(horseBatch.value(QStringLiteral("shapes")).toArray().size() == 2,
               "A tags-only Auto Run result must keep the existing shapes"))
        return 1;
    if (!check(!horseBatch.value(QStringLiteral("checked")).toBool(),
               "Tags written by Auto Run must uncheck a reviewed image"))
        return 1;
    (void)horseShapesBefore;
    return 0;
}
