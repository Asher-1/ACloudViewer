// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCommandLineParser>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QListWidget>
#include <QPluginLoader>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "RelateAnythingDialog.h"
#include "RelateAnythingVideoWidget.h"
#include "ecvClickableImageLabel.h"
#include "ecvTestDataRepository.h"
int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setOrganizationName("ACloudViewerTests");
    app.setApplicationName("RelateWorkflow");
    QTemporaryDir settings;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settings.path());
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addOption(
            {"real", "Run real-model image, export and video checks."});
    parser.addOption({"download-sample",
                      "Run shared sample download and autofill checks."});
    parser.addOption({"backend", "Device.", "device", "cuda"});
    parser.addOption({"detector", "Detector GGUF file name.", "file"});
    parser.addOption({"plugin-module", "Load plugin module.", "file"});
    parser.addOption({"output-dir", "Save evidence.", "dir"});
    parser.process(app);
    try {
        auto check = [](bool yes, const QString& message) {
            if (!yes) throw std::runtime_error(message.toStdString());
        };
        auto wait = [&](auto predicate, int limit = 240000) {
            QElapsedTimer t;
            t.start();
            while (!predicate() && t.elapsed() < limit) {
                app.processEvents(QEventLoop::AllEvents, 20);
                QThread::msleep(5);
            }
            return predicate();
        };
        if (parser.isSet("plugin-module")) {
            QPluginLoader l(parser.value("plugin-module"));
            check(l.instance(), l.errorString());
            check(l.metaData()["IID"].toString() ==
                          "cvcorp.cloudviewer.plugin.qRelateAnything",
                  "plugin IID");
            auto actions = l.instance()->findChildren<QAction*>();
            check(actions.size() == 1 && !actions[0]->icon().isNull(),
                  "plugin action/icon");
        }
        RelateAnythingDialog dialog;
        dialog.show();
        if (parser.isSet("output-dir")) {
            // Evidence of the Options layout at the default dialog size.
            QDir().mkpath(parser.value("output-dir"));
            auto* optTabs = dialog.findChild<QTabWidget*>("relateTabs");
            for (int i = 0; i < optTabs->count(); ++i)
                if (optTabs->tabText(i).contains("Options"))
                    optTabs->setCurrentIndex(i);
            app.processEvents();
            dialog.grab().save(parser.value("output-dir") + "/options.png");
            if (auto* scroll =
                        qobject_cast<QScrollArea*>(optTabs->currentWidget()))
                std::cout << "options rows content="
                          << scroll->widget()->height()
                          << "px viewport=" << scroll->viewport()->height()
                          << "px\n";
            optTabs->setCurrentIndex(0);
        }
        auto* model = dialog.findChild<QComboBox*>("relateModel");
        auto* device = dialog.findChild<QComboBox*>("relateDevice");
        auto* list = dialog.findChild<QListWidget*>("relateInputs");
        auto* sample = dialog.findChild<QPushButton*>("relateUseTestData");
        auto* run = dialog.findChild<QPushButton*>("relateRun");
        auto* table = dialog.findChild<QTableWidget*>("relateRelations");
        auto* video = dialog.findChild<RelateAnythingVideoWidget*>();
        auto* tabs = dialog.findChild<QTabWidget*>("relateTabs");
        check(model && device && list && sample && run && table && video &&
                      tabs,
              "UI controls");
        {
            // The picker lists exactly the relation-weight rows of the
            // catalog (the text students and tokenizer are not models).
            int weights = 0;
            for (int i = 0; i < aicore_relation_model_count(); ++i)
                weights += aicore_relation_model_at(i)->role ==
                           AICORE_RELATION_MODEL_RELATIONS;
            check(weights > 0 && model->count() == weights,
                  "every relation weight row is listed");
        }
        {
            // Defaults must equal upstream render_video.py / demo_webcam.py.
            auto dbl = [&](const char* name) {
                auto* w = dialog.findChild<QDoubleSpinBox*>(name);
                check(w, QString("option control ") + name);
                return w->value();
            };
            auto near = [](double a, double b) {
                return std::abs(a - b) < 1e-9;
            };
            check(near(dbl("relateDetectionThreshold"), .25) &&
                          near(dbl("relateDetectionIou"), .5) &&
                          near(dbl("relateMaxBoxFraction"), .85) &&
                          near(dbl("relateRelationThreshold"), .45) &&
                          near(dbl("relateSpatialThreshold"), .42),
                  "official video-reel threshold defaults");
            check(dialog.findChild<QSpinBox*>("relateMaxRelations")->value() ==
                                  8 &&
                          dialog.findChild<QSpinBox*>("relateMaxSpatial")
                                          ->value() == 3,
                  "official edge caps (8 total, 3 spatial)");
            check(dialog.findChild<QComboBox*>("relateProfile")
                                          ->currentData()
                                          .toInt() == 0 &&
                          dialog.findChild<QComboBox*>("relatePredicatePreset")
                                          ->currentData()
                                          .toInt() == 0,
                  "default operating point and vocabulary preset");
            // Confidence follows the detector family (make_reel.py MODES)
            // unless the profile is Custom.
            {
                auto* det = dialog.findChild<QComboBox*>("relateDetector");
                auto* thr = dialog.findChild<QDoubleSpinBox*>(
                        "relateDetectionThreshold");
                const int keep = det->currentIndex();
                const int world = det->findData("yolov8s-world-f16.gguf");
                const int plain = det->findData("yolo26s-seg-f16.gguf");
                check(world >= 0 && plain >= 0, "detector rows exist");
                det->setCurrentIndex(world);
                check(near(thr->value(), .15), "YOLO-World default 0.15");
                det->setCurrentIndex(plain);
                check(near(thr->value(), .20), "render_video default 0.20");
                det->setCurrentIndex(keep);
                check(near(thr->value(), .25), "prompt-free default 0.25");
            }
            // Upstream render_video.py tracking defaults.
            check(near(dbl("relateTrackHigh"), .25) &&
                          near(dbl("relateIouThr"), .3) &&
                          near(dbl("relateMergeIou"), .75) &&
                          near(dbl("relateKfQ"), 1.0) &&
                          near(dbl("relateKfR"), 10.0) &&
                          near(dbl("relateKfRelQ"), .05) &&
                          near(dbl("relateKfRelR"), 1.0) &&
                          near(dbl("relateScoreEma"), .35) &&
                          near(dbl("relateOffThr"), .38),
                  "official tracking / belief defaults");
            auto integer = [&](const char* name) {
                auto* w = dialog.findChild<QSpinBox*>(name);
                check(w, QString("option control ") + name);
                return w->value();
            };
            check(integer("relateMaxAge") == 8 &&
                          integer("relateMinHits") == 2 &&
                          integer("relateOnFrames") == 2 &&
                          integer("relateOffFrames") == 6 &&
                          integer("relateMaxPerPair") == 2 &&
                          integer("relateBirthTopk") == 24,
                  "official tracking counters");
            check(dialog.findChild<QCheckBox*>("relateBelief")->isChecked() &&
                          dialog.findChild<QCheckBox*>("relateStabilize")
                                  ->isChecked(),
                  "belief filter and stabilisation on by default");
            // Switching to the image demo fills its floor; editing flips to
            // Custom again.
            auto* profile = dialog.findChild<QComboBox*>("relateProfile");
            profile->setCurrentIndex(profile->findData(1));
            emit profile->activated(profile->currentIndex());
            check(near(dbl("relateRelationThreshold"), .5) &&
                          !dialog.findChild<QCheckBox*>("relateDecompose")
                                   ->isChecked(),
                  "image-demo profile fills its thresholds");
            dialog.findChild<QDoubleSpinBox*>("relateSpatialThreshold")
                    ->setValue(.3);
            check(profile->currentData().toInt() == 2,
                  "editing a threshold selects Custom");
            profile->setCurrentIndex(profile->findData(0));
            emit profile->activated(profile->currentIndex());
            check(near(dbl("relateSpatialThreshold"), .42) &&
                          dialog.findChild<QCheckBox*>("relateDecompose")
                                  ->isChecked(),
                  "video-reel profile restores defaults");
        }
        check(dialog.findChild<QComboBox*>("relateArithmetic")
                              ->currentData()
                              .toInt() == -1,
              "upstream default arithmetic");
        check(model->currentData().toInt() ==
                      aicore_relation_model_default_index(),
              "catalog default model");
        const auto info = ecvTestDataRepository::getDatasetInfo(
                ecvTestDataRepository::Dataset::RelateAnything);
        check(info.zipFileName == "relate_datasets.zip" &&
                      info.downloadUrl.endsWith("/assets/relate_datasets.zip"),
              "shared sample metadata");
        if (parser.isSet("download-sample") || parser.isSet("real")) {
            sample->click();
            check(wait([&] {
                      return list->count() == 6 && sample->isEnabled();
                  }),
                  "sample download/autofill");
            check(list->currentItem()->text().endsWith(
                          "/demo_images/horse.jpg"),
                  "recommended first image");
            for (int i = 0; i < list->count(); ++i)
                check(list->item(i)->checkState() ==
                              (list->item(i)->text().endsWith("/horse.jpg")
                                       ? Qt::Checked
                                       : Qt::Unchecked),
                      "all samples checkable, only horse checked by default");
            check(video->videoFilePath().endsWith(
                          "/demo_videos/r3c_adoption_54729571599.webm"),
                  "same click fills video path");
            check(ecvTestDataRepository::instance().isDatasetAvailable(
                          ecvTestDataRepository::Dataset::RelateAnything),
                  "cached dataset");
            sample->click();
            check(list->count() == 6,
                  "cache reuse does not duplicate image rows");
        }
        if (parser.isSet("real")) {
            const int idx = device->findData(parser.value("backend"));
            check(idx >= 0, "requested device");
            device->setCurrentIndex(idx);
            if (parser.isSet("detector")) {
                auto* detector = dialog.findChild<QComboBox*>("relateDetector");
                const int d = detector->findData(parser.value("detector"));
                check(d >= 0, "requested detector is offered in the combo");
                detector->setCurrentIndex(d);
            }
            tabs->setCurrentIndex(0);
            run->click();
            check(wait([&] {
                      return table->rowCount() > 0 && run->isEnabled();
                  }),
                  "real image pipeline");
            QTemporaryDir outputs;
            const QString dir = parser.isSet("output-dir")
                                        ? parser.value("output-dir")
                                        : outputs.path();
            QDir().mkpath(dir);
            bool savedOverlay = false;
            for (auto* preview : dialog.findChildren<ecvClickableImageLabel*>())
                if (!preview->fullImage().isNull() && !savedOverlay)
                    savedOverlay =
                            preview->fullImage().save(dir + "/annotated.png");
            check(savedOverlay, "save annotated overlay evidence");
            for (const auto& ext : {"json", "csv", "txt"})
                check(dialog.saveResults(dir + "/relations." + ext), "exports");
            QFile f(dir + "/relations.json");
            check(f.open(QIODevice::ReadOnly), "read export");
            auto results = QJsonDocument::fromJson(f.readAll()).array();
            check(results.size() == 1 && !results[0]
                                                  .toObject()["relations"]
                                                  .toArray()
                                                  .isEmpty(),
                  "typed results exported");
            check(results[0].toObject()["objects"].toArray().size() >= 2,
                  "detector consumer");
            int dbRequests = 0;
            QObject::connect(&dialog, &RelateAnythingDialog::addToDbRequested,
                             &dialog, [&](const RelateAnythingResult& result) {
                                 check(!result.annotated.isNull() &&
                                               !result.json["relations"]
                                                        .toArray()
                                                        .isEmpty(),
                                       "DB image and typed metadata");
                                 ++dbRequests;
                             });
            dialog.findChild<QPushButton*>("relateAddToDb")->click();
            check(dbRequests == 1, "DB result handoff");
            dialog.findChild<QPushButton*>("relateCheckAll")->click();
            tabs->setCurrentIndex(0);
            run->click();
            check(wait([&] {
                      return tabs->currentIndex() == 1 && run->isEnabled();
                  }),
                  "six-image batch completes");
            check(dialog.saveResults(dir + "/batch.json"), "batch export");
            QFile batch(dir + "/batch.json");
            check(batch.open(QIODevice::ReadOnly), "read batch export");
            check(QJsonDocument::fromJson(batch.readAll()).array().size() == 6,
                  "all six images exported");
            // Reuse actual detected regions to compare automatic and imported
            // input paths, then exercise masks and both vocabulary sources.
            auto* objectSource =
                    dialog.findChild<QComboBox*>("relateObjectSource");
            auto* regions = dialog.findChild<QLineEdit*>("relateRegions");
            auto* masks = dialog.findChild<QLineEdit*>("relateMasks");
            auto* predicates = dialog.findChild<QLineEdit*>("relatePredicates");
            auto* vocabulary = dialog.findChild<QLineEdit*>("relateVocabulary");
            auto* vocabularyNames =
                    dialog.findChild<QLineEdit*>("relateVocabularyNames");
            auto* types = dialog.findChild<QLineEdit*>("relatePredicateTypes");
            check(objectSource && regions && masks && predicates &&
                          vocabulary && vocabularyNames && types,
                  "official input controls");
            auto writeFile = [&](const QString& name, const QByteArray& bytes) {
                QFile file(dir + "/" + name);
                check(file.open(QIODevice::WriteOnly) &&
                              file.write(bytes) == bytes.size(),
                      "write workflow fixture: " + name);
                return file.fileName();
            };
            auto readResult = [&](const QString& name) {
                check(dialog.saveResults(dir + "/" + name),
                      "save workflow result");
                QFile file(dir + "/" + name);
                check(file.open(QIODevice::ReadOnly), "read workflow result");
                const auto values =
                        QJsonDocument::fromJson(file.readAll()).array();
                check(values.size() == 1, "one checked input");
                return values[0].toObject();
            };
            auto runImage = [&] {
                tabs->setCurrentIndex(0);
                run->click();
                check(wait([&] {
                          return tabs->currentIndex() == 1 && run->isEnabled();
                      }),
                      "imported region / vocabulary inference completes");
                check(table->rowCount() > 0,
                      "imported pipeline produces relations");
            };
            auto compareEdges = [&](const QJsonArray& expected,
                                    const QJsonArray& actual) {
                check(actual.size() == expected.size(), "relation edge count");
                for (int i = 0; i < actual.size(); ++i) {
                    const auto a = actual[i].toObject(),
                               b = expected[i].toObject();
                    for (const auto& key :
                         {"subject", "object", "predicate_index", "predicate"})
                        check(a[key] == b[key],
                              "relation identity: " + QString(key));
                    check(std::abs(a["score"].toDouble() -
                                   b["score"].toDouble()) <= 1e-5,
                          "relation score parity");
                }
            };
            sample->click();
            const QImage horse(list->currentItem()->text());
            check(!horse.isNull(), "decode horse for imported mask fixture");
            const auto baseline = results[0].toObject();
            QJsonArray boxes, labels, scores;
            for (const auto& value : baseline["objects"].toArray()) {
                const auto object = value.toObject();
                boxes.append(object["box"]);
                labels.append(object["name"]);
                scores.append(object["score"]);
            }
            QJsonObject metadata{
                    {"boxes", boxes},
                    {"labels", labels},
                    {"scores", scores},
                    {"image_size", QJsonArray{horse.width(), horse.height()}}};
            regions->setText(writeFile("objects.json",
                                       QJsonDocument(metadata).toJson()));
            objectSource->setCurrentIndex(1);
            check(!dialog.findChild<QComboBox*>("relateDetector")->isEnabled(),
                  "supplied boxes skip detector");
            runImage();
            const auto imported = readResult("supplied-boxes.json");
            check(imported["object_source"] == "supplied" &&
                          imported["timings"]
                                          .toObject()["detection_ms"]
                                          .toDouble() == 0,
                  "no detector stage for imported regions");
            compareEdges(baseline["relations"].toArray(),
                         imported["relations"].toArray());
            runImage();
            check(readResult("reused-supplied-boxes.json")["timings"]
                          .toObject()["context_reused"]
                          .toBool(),
                  "unchanged imported files retain the resident session");
            auto editedLabels = labels;
            editedLabels[0] = "Edited object label";
            metadata["labels"] = editedLabels;
            writeFile("objects.json", QJsonDocument(metadata).toJson());
            runImage();
            const auto editedRegions = readResult("edited-supplied-boxes.json");
            check(editedRegions["objects"]
                                  .toArray()[0]
                                  .toObject()["name"]
                                  .toString() == "Edited object label",
                  "same-path object JSON is reloaded");
            compareEdges(imported["relations"].toArray(),
                         editedRegions["relations"].toArray());
            QByteArray planes(boxes.size() * horse.width() * horse.height(), 0);
            for (int i = 0; i < boxes.size(); ++i) {
                const auto box = boxes[i].toArray();
                const int x1 = std::max(0, int(box[0].toDouble()));
                const int y1 = std::max(0, int(box[1].toDouble()));
                const int x2 = std::min(horse.width(), int(box[2].toDouble()));
                const int y2 = std::min(horse.height(), int(box[3].toDouble()));
                for (int y = y1; y < y2; ++y)
                    for (int x = x1; x < x2; ++x)
                        planes[(i * horse.height() + y) * horse.width() + x] =
                                1;
            }
            masks->setText(writeFile("masks.u8", planes));
            predicates->setText("riding | holding | next to | above");
            types->setText(
                    writeFile("predicate-types.json",
                              QJsonDocument(QJsonObject{{"riding", false},
                                                        {"holding", false},
                                                        {"next to", true},
                                                        {"above", true}})
                                      .toJson()));
            dialog.findChild<QCheckBox*>("relateDecompose")->setChecked(true);
            // A four-predicate custom vocabulary scores on its own scale;
            // this stage checks the decomposition, not the default floor.
            auto* relationFloor = dialog.findChild<QDoubleSpinBox*>(
                    "relateRelationThreshold");
            auto* spatialFloor =
                    dialog.findChild<QDoubleSpinBox*>("relateSpatialThreshold");
            check(relationFloor && spatialFloor &&
                          std::abs(relationFloor->value() - 0.45) < 1e-9 &&
                          std::abs(spatialFloor->value() - 0.42) < 1e-9,
                  "upstream video relation floors (0.45 / 0.42)");
            relationFloor->setValue(0.05);
            spatialFloor->setValue(0.05);
            runImage();
            const auto custom = readResult("custom-predicates-masks.json");
            check(!custom["spatial"].toArray().isEmpty() &&
                          !custom["semantic"].toArray().isEmpty(),
                  "custom predicate decomposition");
            std::unique_ptr<char, decltype(&aicore_relation_free_buffer)> cache(
                    aicore_relation_model_cache_dir(),
                    aicore_relation_free_buffer);
            const auto* relationModel =
                    aicore_relation_model_at(model->currentData().toInt());
            QString student, tokenizer;
            for (int i = 0; i < aicore_relation_model_count(); ++i) {
                const auto* entry = aicore_relation_model_at(i);
                if ((entry->role == AICORE_RELATION_MODEL_TEXT &&
                     QString::fromUtf8(entry->architecture) ==
                             QString::fromUtf8(relationModel->architecture)) ||
                    entry->role == AICORE_RELATION_MODEL_TOKENIZER) {
                    const auto path = QDir(QString::fromUtf8(cache.get()))
                                              .filePath(QString::fromUtf8(
                                                      entry->filename));
                    if (entry->role == AICORE_RELATION_MODEL_TEXT)
                        student = path;
                    else
                        tokenizer = path;
                }
            }
            const auto studentBytes = student.toUtf8(),
                       tokenizerBytes = tokenizer.toUtf8();
            const auto backendBytes = parser.value("backend").toUtf8();
            auto opts = aicore_relation_default_options();
            opts.model_path = studentBytes.constData();
            opts.tokenizer_path = tokenizerBytes.constData();
            opts.device = backendBytes.constData();
            opts.threads = 8;
            opts.text_only = 1;
            std::unique_ptr<aicore_relation_ctx,
                            decltype(&aicore_relation_free)>
                    textContext(aicore_relation_create(&opts),
                                aicore_relation_free);
            check(aicore_relation_ready(textContext.get()),
                  QString::fromUtf8(
                          aicore_relation_last_error(textContext.get())));
            const char* names[]{"riding", "holding", "next to", "above"};
            aicore_relation_embeddings* encoded = nullptr;
            check(aicore_relation_encode_text(textContext.get(), names, 4,
                                              &encoded) == 0,
                  QString::fromUtf8(
                          aicore_relation_last_error(textContext.get())));
            std::unique_ptr<aicore_relation_embeddings,
                            decltype(&aicore_relation_embeddings_free)>
                    embeddings(encoded, aicore_relation_embeddings_free);
            vocabulary->setText(writeFile(
                    "vocabulary.f32",
                    QByteArray(reinterpret_cast<const char*>(encoded->values),
                               int(encoded->count * encoded->dimension *
                                   sizeof(float)))));
            vocabularyNames->setText(
                    writeFile("predicate-names.json",
                              QJsonDocument(QJsonArray{"riding", "holding",
                                                       "next to", "above"})
                                      .toJson()));
            textContext.reset();
            check(!predicates->isEnabled(),
                  "vector file replaces predicate text input");
            runImage();
            const auto external = readResult("external-vocabulary-masks.json");
            for (const auto& key : {"relations", "spatial", "semantic"})
                compareEdges(custom[key].toArray(), external[key].toArray());
            writeFile("predicate-names.json",
                      QJsonDocument(
                              QJsonArray{"updated riding", "updated holding",
                                         "updated next to", "updated above"})
                              .toJson());
            runImage();
            const auto renamed = readResult("edited-vocabulary-names.json");
            for (const auto& value : renamed["relations"].toArray())
                check(value.toObject()["predicate"].toString().startsWith(
                              "updated "),
                      "same-path vocabulary names are reloaded");
            writeFile("masks.u8", QByteArray(1, 1));
            tabs->setCurrentIndex(0);
            run->click();
            check(wait([&] { return run->isEnabled(); }) &&
                          table->rowCount() == 0 &&
                          !dialog.saveResults(dir + "/invalid-mask.json"),
                  "bad mask fails and drains worker");
            writeFile("masks.u8", planes);
            runImage();
            const auto recoveredMask = readResult("recovered-mask.json");
            compareEdges(renamed["relations"].toArray(),
                         recoveredMask["relations"].toArray());
            masks->clear();
            regions->clear();
            vocabulary->clear();
            vocabularyNames->clear();
            types->clear();
            predicates->clear();
            dialog.findChild<QCheckBox*>("relateDecompose")->setChecked(false);
            objectSource->setCurrentIndex(0);
            int frames = 0;
            QObject::connect(video, &RelateAnythingVideoWidget::frameReady,
                             &dialog, [&](const QImage& image, int) {
                                 check(!image.isNull(), "decoded video frame");
                                 ++frames;
                             });
            tabs->setCurrentIndex(tabs->indexOf(video->parentWidget()));
            run->click();
            // Frame N+1 is only delivered after frame N's result is
            // acknowledged, so four delivered frames means three finished
            // results.
            check(wait([&] { return frames >= 4; }),
                  "consumer-driven video progresses");
            {
                // Every video frame must have gone through the ported
                // tracker + belief filter (upstream render_video.py).
                check(dialog.saveResults(dir + "/video-frames.json"),
                      "save video frame results");
                QFile vf(dir + "/video-frames.json");
                check(vf.open(QIODevice::ReadOnly), "read video results");
                const auto rows = QJsonDocument::fromJson(vf.readAll()).array();
                check(rows.size() >= 3, "one result row per video frame");
                for (const auto& row : rows)
                    check(row.toObject()["temporal"].toBool() &&
                                  row.toObject()["tracks"].toInt() >= 0,
                          "video frame used temporal stabilisation");
                check(rows.last().toObject()["tracks"].toInt() > 0,
                      "tracks persist across video frames");
            }
            auto previews = video->findChildren<ecvClickableImageLabel*>();
            check(!previews.isEmpty() &&
                          !previews.front()->pixmap(Qt::ReturnByValue).isNull(),
                  "annotated video preview");
            dialog.findChild<QPushButton*>("relateCancel")->click();
            check(wait([&] { return run->isEnabled(); }),
                  "cancel drains pending frame");
            tabs->setCurrentIndex(0);
            list->clear();
            auto* missing =
                    new QListWidgetItem(dir + "/missing-input.jpg", list);
            missing->setCheckState(Qt::Checked);
            run->click();
            check(wait([&] { return run->isEnabled(); }) &&
                          table->rowCount() == 0,
                  "invalid input clears results and releases worker");
            sample->click();
            run->click();
            check(wait([&] {
                      return table->rowCount() > 0 && run->isEnabled();
                  }),
                  "successful inference after invalid input");
            dialog.grab().save(dir + "/dialog.png");
            std::cout
                    << "Image/batch, supplied boxes/masks, custom/external "
                       "vocabulary, decomposition, DB handoff, exports, three "
                       "paced video frames and error recovery passed\n";
        }
        dialog.close();
        dialog.show();
        check(run->isEnabled(), "reopen");
        dialog.shutdown();
        std::cout << "RelateAnything workflow passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
