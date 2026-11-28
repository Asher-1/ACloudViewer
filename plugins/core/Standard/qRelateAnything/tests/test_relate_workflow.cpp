// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

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
#include <limits>
#include <memory>

#include "QtCompat.h"
#include "RelateAnythingDialog.h"
#include "RelateAnythingVideoWidget.h"
#include "ecvClickableImageLabel.h"
#include "ecvTestDataRepository.h"
namespace {
QStringList workflowArguments;

class RelateWorkflow : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(settings.isValid());
        app.setOrganizationName("ACloudViewerTests");
        app.setApplicationName("RelateWorkflow");
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                           settings.path());
        QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope,
                           settings.path());
        parser.addHelpOption();
        parser.addOption(
                {"real", "Run real-model image, export and video checks."});
        parser.addOption({"download-sample",
                          "Run shared sample download and autofill checks."});
        parser.addOption({"backend", "Device.", "device", "cuda"});
        parser.addOption({"detector", "Detector GGUF file name.", "file"});
        parser.addOption({"plugin-module", "Load plugin module.", "file"});
        parser.addOption({"output-dir", "Save evidence.", "dir"});
        parser.process(workflowArguments);
    }
    QApplication& app = *qApp;
    QTemporaryDir settings;
    QCommandLineParser parser;
};

TEST_F(RelateWorkflow, DiscoversBuiltPluginActionAndIcon) {
    if (!parser.isSet("plugin-module"))
        GTEST_SKIP() << "No plugin-module argument supplied";
    if (parser.isSet("plugin-module")) {
        QPluginLoader l(parser.value("plugin-module"));
        ASSERT_TRUE(l.instance()) << QString(l.errorString()).toStdString();
        ASSERT_TRUE(l.metaData()["IID"].toString() ==
                    "cvcorp.cloudviewer.plugin.qRelateAnything")
                << QString("plugin IID").toStdString();
        auto actions = l.instance()->findChildren<QAction*>();
        ASSERT_TRUE(actions.size() == 1 && !actions[0]->icon().isNull())
                << QString("plugin action/icon").toStdString();
    }
}

TEST_F(RelateWorkflow, DialogOptionsAndOptionalRealImageVideoLifecycle) {
    auto wait = [&](auto predicate, int limit = 240000) {
        QElapsedTimer t;
        t.start();
        while (!predicate() && t.elapsed() < limit) {
            app.processEvents(QEventLoop::AllEvents, 20);
            QThread::msleep(5);
        }
        return predicate();
    };
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
        if (auto* scroll = qobject_cast<QScrollArea*>(optTabs->currentWidget()))
            std::cout << "options rows content=" << scroll->widget()->height()
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
    ASSERT_TRUE(model && device && list && sample && run && table && video &&
                tabs)
            << QString("UI controls").toStdString();
    {
        // The picker lists exactly the relation-weight rows of the
        // catalog (the text students and tokenizer are not models).
        int weights = 0;
        for (int i = 0; i < aicore_relation_model_count(); ++i)
            weights += aicore_relation_model_at(i)->role ==
                       AICORE_RELATION_MODEL_RELATIONS;
        ASSERT_TRUE(weights > 0 && model->count() == weights)
                << QString("every relation weight row is listed").toStdString();
    }
    {
        // Defaults must equal upstream render_video.py / demo_webcam.py.
        auto dbl = [&](const char* name) {
            auto* w = dialog.findChild<QDoubleSpinBox*>(name);
            EXPECT_NE(w, nullptr) << "option control " << name;
            return w ? w->value() : std::numeric_limits<double>::quiet_NaN();
        };
        auto near = [](double a, double b) { return std::abs(a - b) < 1e-9; };
        ASSERT_TRUE(near(dbl("relateDetectionThreshold"), .25) &&
                    near(dbl("relateDetectionIou"), .5) &&
                    near(dbl("relateMaxBoxFraction"), .85) &&
                    near(dbl("relateRelationThreshold"), .45) &&
                    near(dbl("relateSpatialThreshold"), .42))
                << QString("official video-reel threshold defaults")
                           .toStdString();
        ASSERT_TRUE(
                dialog.findChild<QSpinBox*>("relateMaxRelations")->value() ==
                        8 &&
                dialog.findChild<QSpinBox*>("relateMaxSpatial")->value() == 3)
                << QString("official edge caps (8 total, 3 spatial)")
                           .toStdString();
        ASSERT_TRUE(dialog.findChild<QComboBox*>("relateProfile")
                                    ->currentData()
                                    .toInt() == 0 &&
                    dialog.findChild<QComboBox*>("relatePredicatePreset")
                                    ->currentData()
                                    .toInt() == 0)
                << QString("default operating point and vocabulary preset")
                           .toStdString();
        // Confidence follows the detector family (make_reel.py MODES)
        // unless the profile is Custom.
        {
            auto* det = dialog.findChild<QComboBox*>("relateDetector");
            auto* thr = dialog.findChild<QDoubleSpinBox*>(
                    "relateDetectionThreshold");
            const int keep = det->currentIndex();
            const int world = det->findData("yolov8s-world-f16.gguf");
            const int plain = det->findData("yolo26s-seg-f16.gguf");
            ASSERT_TRUE(world >= 0 && plain >= 0)
                    << QString("detector rows exist").toStdString();
            det->setCurrentIndex(world);
            ASSERT_TRUE(near(thr->value(), .15))
                    << QString("YOLO-World default 0.15").toStdString();
            det->setCurrentIndex(plain);
            ASSERT_TRUE(near(thr->value(), .20))
                    << QString("render_video default 0.20").toStdString();
            det->setCurrentIndex(keep);
            ASSERT_TRUE(near(thr->value(), .25))
                    << QString("prompt-free default 0.25").toStdString();
        }
        // Upstream render_video.py tracking defaults.
        ASSERT_TRUE(near(dbl("relateTrackHigh"), .25) &&
                    near(dbl("relateIouThr"), .3) &&
                    near(dbl("relateMergeIou"), .75) &&
                    near(dbl("relateKfQ"), 1.0) &&
                    near(dbl("relateKfR"), 10.0) &&
                    near(dbl("relateKfRelQ"), .05) &&
                    near(dbl("relateKfRelR"), 1.0) &&
                    near(dbl("relateScoreEma"), .35) &&
                    near(dbl("relateOffThr"), .38))
                << QString("official tracking / belief defaults").toStdString();
        auto integer = [&](const char* name) {
            auto* w = dialog.findChild<QSpinBox*>(name);
            EXPECT_NE(w, nullptr) << "option control " << name;
            return w ? w->value() : -1;
        };
        ASSERT_TRUE(integer("relateMaxAge") == 8 &&
                    integer("relateMinHits") == 2 &&
                    integer("relateOnFrames") == 2 &&
                    integer("relateOffFrames") == 6 &&
                    integer("relateMaxPerPair") == 2 &&
                    integer("relateBirthTopk") == 24)
                << QString("official tracking counters").toStdString();
        ASSERT_TRUE(
                dialog.findChild<QCheckBox*>("relateBelief")->isChecked() &&
                dialog.findChild<QCheckBox*>("relateStabilize")->isChecked())
                << QString("belief filter and stabilisation on by default")
                           .toStdString();
        // Switching to the image demo fills its floor; editing flips to
        // Custom again.
        auto* profile = dialog.findChild<QComboBox*>("relateProfile");
        profile->setCurrentIndex(profile->findData(1));
        emit profile->activated(profile->currentIndex());
        ASSERT_TRUE(
                near(dbl("relateRelationThreshold"), .5) &&
                !dialog.findChild<QCheckBox*>("relateDecompose")->isChecked())
                << QString("image-demo profile fills its thresholds")
                           .toStdString();
        dialog.findChild<QDoubleSpinBox*>("relateSpatialThreshold")
                ->setValue(.3);
        ASSERT_TRUE(profile->currentData().toInt() == 2)
                << QString("editing a threshold selects Custom").toStdString();
        profile->setCurrentIndex(profile->findData(0));
        emit profile->activated(profile->currentIndex());
        ASSERT_TRUE(
                near(dbl("relateSpatialThreshold"), .42) &&
                dialog.findChild<QCheckBox*>("relateDecompose")->isChecked())
                << QString("video-reel profile restores defaults")
                           .toStdString();
    }
    ASSERT_TRUE(dialog.findChild<QComboBox*>("relateArithmetic")
                        ->currentData()
                        .toInt() == -1)
            << QString("upstream default arithmetic").toStdString();
    ASSERT_TRUE(model->currentData().toInt() ==
                aicore_relation_model_default_index())
            << QString("catalog default model").toStdString();
    const auto info = ecvTestDataRepository::getDatasetInfo(
            ecvTestDataRepository::Dataset::RelateAnything);
    ASSERT_TRUE(info.zipFileName == "relate_datasets.zip" &&
                info.downloadUrl.endsWith("/assets/relate_datasets.zip"))
            << QString("shared sample metadata").toStdString();
    if (parser.isSet("download-sample") || parser.isSet("real")) {
        sample->click();
        ASSERT_TRUE(wait([&] {
            return list->count() == 6 && sample->isEnabled();
        })) << QString("sample download/autofill").toStdString();
        ASSERT_TRUE(
                list->currentItem()->text().endsWith("/demo_images/horse.jpg"))
                << QString("recommended first image").toStdString();
        for (int i = 0; i < list->count(); ++i)
            ASSERT_TRUE(list->item(i)->checkState() ==
                        (list->item(i)->text().endsWith("/horse.jpg")
                                 ? Qt::Checked
                                 : Qt::Unchecked))
                    << QString("all samples checkable, only horse checked by "
                               "default")
                               .toStdString();
        ASSERT_TRUE(video->videoFilePath().endsWith(
                "/demo_videos/r3c_adoption_54729571599.webm"))
                << QString("same click fills video path").toStdString();
        ASSERT_TRUE(ecvTestDataRepository::instance().isDatasetAvailable(
                ecvTestDataRepository::Dataset::RelateAnything))
                << QString("cached dataset").toStdString();
        sample->click();
        ASSERT_TRUE(list->count() == 6)
                << QString("cache reuse does not duplicate image rows")
                           .toStdString();
    }
    if (parser.isSet("real")) {
        const int idx = device->findData(parser.value("backend"));
        ASSERT_TRUE(idx >= 0) << QString("requested device").toStdString();
        device->setCurrentIndex(idx);
        if (parser.isSet("detector")) {
            auto* detector = dialog.findChild<QComboBox*>("relateDetector");
            const int d = detector->findData(parser.value("detector"));
            ASSERT_TRUE(d >= 0)
                    << QString("requested detector is offered in the combo")
                               .toStdString();
            detector->setCurrentIndex(d);
        }
        tabs->setCurrentIndex(0);
        run->click();
        ASSERT_TRUE(wait([&] {
            return table->rowCount() > 0 && run->isEnabled();
        })) << QString("real image pipeline").toStdString();
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
        ASSERT_TRUE(savedOverlay)
                << QString("save annotated overlay evidence").toStdString();
        for (const auto& ext : {"json", "csv", "txt"})
            ASSERT_TRUE(dialog.saveResults(dir + "/relations." + ext))
                    << QString("exports").toStdString();
        QFile f(dir + "/relations.json");
        ASSERT_TRUE(f.open(QIODevice::ReadOnly))
                << QString("read export").toStdString();
        auto results = QJsonDocument::fromJson(f.readAll()).array();
        ASSERT_TRUE(results.size() == 1 &&
                    !results[0].toObject()["relations"].toArray().isEmpty())
                << QString("typed results exported").toStdString();
        ASSERT_TRUE(results[0].toObject()["objects"].toArray().size() >= 2)
                << QString("detector consumer").toStdString();
        int dbRequests = 0;
        QObject::connect(
                &dialog, &RelateAnythingDialog::addToDbRequested, &dialog,
                [&](const RelateAnythingResult& result) {
                    ASSERT_TRUE(!result.annotated.isNull() &&
                                !result.json["relations"].toArray().isEmpty())
                            << QString("DB image and typed metadata")
                                       .toStdString();
                    ++dbRequests;
                });
        dialog.findChild<QPushButton*>("relateAddToDb")->click();
        ASSERT_TRUE(dbRequests == 1)
                << QString("DB result handoff").toStdString();
        dialog.findChild<QPushButton*>("relateCheckAll")->click();
        tabs->setCurrentIndex(0);
        run->click();
        ASSERT_TRUE(wait([&] {
            return tabs->currentIndex() == 1 && run->isEnabled();
        })) << QString("six-image batch completes").toStdString();
        ASSERT_TRUE(dialog.saveResults(dir + "/batch.json"))
                << QString("batch export").toStdString();
        QFile batch(dir + "/batch.json");
        ASSERT_TRUE(batch.open(QIODevice::ReadOnly))
                << QString("read batch export").toStdString();
        ASSERT_TRUE(QJsonDocument::fromJson(batch.readAll()).array().size() ==
                    6)
                << QString("all six images exported").toStdString();
        // Reuse actual detected regions to compare automatic and imported
        // input paths, then exercise masks and both vocabulary sources.
        auto* objectSource = dialog.findChild<QComboBox*>("relateObjectSource");
        auto* regions = dialog.findChild<QLineEdit*>("relateRegions");
        auto* masks = dialog.findChild<QLineEdit*>("relateMasks");
        auto* predicates = dialog.findChild<QLineEdit*>("relatePredicates");
        auto* vocabulary = dialog.findChild<QLineEdit*>("relateVocabulary");
        auto* vocabularyNames =
                dialog.findChild<QLineEdit*>("relateVocabularyNames");
        auto* types = dialog.findChild<QLineEdit*>("relatePredicateTypes");
        ASSERT_TRUE(objectSource && regions && masks && predicates &&
                    vocabulary && vocabularyNames && types)
                << QString("official input controls").toStdString();
        auto writeFile = [&](const QString& name, const QByteArray& bytes) {
            QFile file(dir + "/" + name);
            EXPECT_TRUE(file.open(QIODevice::WriteOnly) &&
                        file.write(bytes) == bytes.size())
                    << QString("write workflow fixture: " + name).toStdString();
            return file.fileName();
        };
        auto readResult = [&](const QString& name) {
            EXPECT_TRUE(dialog.saveResults(dir + "/" + name))
                    << QString("save workflow result").toStdString();
            QFile file(dir + "/" + name);
            EXPECT_TRUE(file.open(QIODevice::ReadOnly))
                    << QString("read workflow result").toStdString();
            const auto values = QJsonDocument::fromJson(file.readAll()).array();
            EXPECT_TRUE(values.size() == 1)
                    << QString("one checked input").toStdString();
            return values.isEmpty() ? QJsonObject() : values[0].toObject();
        };
        auto runImage = [&] {
            tabs->setCurrentIndex(0);
            run->click();
            ASSERT_TRUE(wait([&] {
                return tabs->currentIndex() == 1 && run->isEnabled();
            })) << QString("imported region / vocabulary inference completes")
                            .toStdString();
            ASSERT_TRUE(table->rowCount() > 0)
                    << QString("imported pipeline produces relations")
                               .toStdString();
        };
        auto compareEdges = [&](const QJsonArray& expected,
                                const QJsonArray& actual) {
            ASSERT_TRUE(actual.size() == expected.size())
                    << QString("relation edge count").toStdString();
            for (int i = 0; i < actual.size(); ++i) {
                const auto a = actual[i].toObject(), b = expected[i].toObject();
                for (const auto& key :
                     {"subject", "object", "predicate_index", "predicate"})
                    ASSERT_TRUE(a[key] == b[key])
                            << QString("relation identity: " + QString(key))
                                       .toStdString();
                ASSERT_TRUE(std::abs(a["score"].toDouble() -
                                     b["score"].toDouble()) <= 1e-5)
                        << QString("relation score parity").toStdString();
            }
        };
        sample->click();
        const QImage horse(list->currentItem()->text());
        ASSERT_TRUE(!horse.isNull())
                << QString("decode horse for imported mask fixture")
                           .toStdString();
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
        regions->setText(
                writeFile("objects.json", QJsonDocument(metadata).toJson()));
        objectSource->setCurrentIndex(1);
        ASSERT_TRUE(
                !dialog.findChild<QComboBox*>("relateDetector")->isEnabled())
                << QString("supplied boxes skip detector").toStdString();
        ASSERT_NO_FATAL_FAILURE(runImage());
        const auto imported = readResult("supplied-boxes.json");
        ASSERT_TRUE(imported["object_source"] == "supplied" &&
                    imported["timings"].toObject()["detection_ms"].toDouble() ==
                            0)
                << QString("no detector stage for imported regions")
                           .toStdString();
        ASSERT_NO_FATAL_FAILURE(compareEdges(baseline["relations"].toArray(),
                                             imported["relations"].toArray()));
        ASSERT_NO_FATAL_FAILURE(runImage());
        ASSERT_TRUE(readResult("reused-supplied-boxes.json")["timings"]
                            .toObject()["context_reused"]
                            .toBool())
                << QString("unchanged imported files retain the resident "
                           "session")
                           .toStdString();
        auto editedLabels = labels;
        editedLabels[0] = "Edited object label";
        metadata["labels"] = editedLabels;
        writeFile("objects.json", QJsonDocument(metadata).toJson());
        ASSERT_NO_FATAL_FAILURE(runImage());
        const auto editedRegions = readResult("edited-supplied-boxes.json");
        ASSERT_TRUE(editedRegions["objects"]
                            .toArray()[0]
                            .toObject()["name"]
                            .toString() == "Edited object label")
                << QString("same-path object JSON is reloaded").toStdString();
        ASSERT_NO_FATAL_FAILURE(
                compareEdges(imported["relations"].toArray(),
                             editedRegions["relations"].toArray()));
        QByteArray planes(boxes.size() * horse.width() * horse.height(), 0);
        for (int i = 0; i < boxes.size(); ++i) {
            const auto box = boxes[i].toArray();
            const int x1 = std::max(0, int(box[0].toDouble()));
            const int y1 = std::max(0, int(box[1].toDouble()));
            const int x2 = std::min(horse.width(), int(box[2].toDouble()));
            const int y2 = std::min(horse.height(), int(box[3].toDouble()));
            for (int y = y1; y < y2; ++y)
                for (int x = x1; x < x2; ++x)
                    planes[(i * horse.height() + y) * horse.width() + x] = 1;
        }
        masks->setText(writeFile("masks.u8", planes));
        predicates->setText("riding | holding | next to | above");
        types->setText(writeFile("predicate-types.json",
                                 QJsonDocument(QJsonObject{{"riding", false},
                                                           {"holding", false},
                                                           {"next to", true},
                                                           {"above", true}})
                                         .toJson()));
        dialog.findChild<QCheckBox*>("relateDecompose")->setChecked(true);
        // A four-predicate custom vocabulary scores on its own scale;
        // this stage checks the decomposition, not the default floor.
        auto* relationFloor =
                dialog.findChild<QDoubleSpinBox*>("relateRelationThreshold");
        auto* spatialFloor =
                dialog.findChild<QDoubleSpinBox*>("relateSpatialThreshold");
        ASSERT_TRUE(relationFloor && spatialFloor &&
                    std::abs(relationFloor->value() - 0.45) < 1e-9 &&
                    std::abs(spatialFloor->value() - 0.42) < 1e-9)
                << QString("upstream video relation floors (0.45 / 0.42)")
                           .toStdString();
        relationFloor->setValue(0.05);
        spatialFloor->setValue(0.05);
        ASSERT_NO_FATAL_FAILURE(runImage());
        const auto custom = readResult("custom-predicates-masks.json");
        ASSERT_TRUE(!custom["spatial"].toArray().isEmpty() &&
                    !custom["semantic"].toArray().isEmpty())
                << QString("custom predicate decomposition").toStdString();
        std::unique_ptr<char, decltype(&aicore_relation_free_buffer)> cache(
                aicore_relation_model_cache_dir(), aicore_relation_free_buffer);
        const auto* relationModel =
                aicore_relation_model_at(model->currentData().toInt());
        QString student, tokenizer;
        for (int i = 0; i < aicore_relation_model_count(); ++i) {
            const auto* entry = aicore_relation_model_at(i);
            if ((entry->role == AICORE_RELATION_MODEL_TEXT &&
                 QString::fromUtf8(entry->architecture) ==
                         QString::fromUtf8(relationModel->architecture)) ||
                entry->role == AICORE_RELATION_MODEL_TOKENIZER) {
                const auto path =
                        QDir(QString::fromUtf8(cache.get()))
                                .filePath(QString::fromUtf8(entry->filename));
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
        std::unique_ptr<aicore_relation_ctx, decltype(&aicore_relation_free)>
                textContext(aicore_relation_create(&opts),
                            aicore_relation_free);
        ASSERT_TRUE(aicore_relation_ready(textContext.get()))
                << QString(QString::fromUtf8(aicore_relation_last_error(
                                   textContext.get())))
                           .toStdString();
        const char* names[]{"riding", "holding", "next to", "above"};
        aicore_relation_embeddings* encoded = nullptr;
        ASSERT_TRUE(aicore_relation_encode_text(textContext.get(), names, 4,
                                                &encoded) == 0)
                << QString(QString::fromUtf8(aicore_relation_last_error(
                                   textContext.get())))
                           .toStdString();
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
        ASSERT_TRUE(!predicates->isEnabled())
                << QString("vector file replaces predicate text input")
                           .toStdString();
        ASSERT_NO_FATAL_FAILURE(runImage());
        const auto external = readResult("external-vocabulary-masks.json");
        for (const auto& key : {"relations", "spatial", "semantic"})
            ASSERT_NO_FATAL_FAILURE(compareEdges(custom[key].toArray(),
                                                 external[key].toArray()));
        writeFile("predicate-names.json",
                  QJsonDocument(QJsonArray{"updated riding", "updated holding",
                                           "updated next to", "updated above"})
                          .toJson());
        ASSERT_NO_FATAL_FAILURE(runImage());
        const auto renamed = readResult("edited-vocabulary-names.json");
        for (const auto& value : renamed["relations"].toArray())
            ASSERT_TRUE(value.toObject()["predicate"].toString().startsWith(
                    "updated "))
                    << QString("same-path vocabulary names are reloaded")
                               .toStdString();
        writeFile("masks.u8", QByteArray(1, 1));
        tabs->setCurrentIndex(0);
        run->click();
        ASSERT_TRUE(wait([&] { return run->isEnabled(); }) &&
                    table->rowCount() == 0 &&
                    !dialog.saveResults(dir + "/invalid-mask.json"))
                << QString("bad mask fails and drains worker").toStdString();
        writeFile("masks.u8", planes);
        ASSERT_NO_FATAL_FAILURE(runImage());
        const auto recoveredMask = readResult("recovered-mask.json");
        ASSERT_NO_FATAL_FAILURE(
                compareEdges(renamed["relations"].toArray(),
                             recoveredMask["relations"].toArray()));
        masks->clear();
        regions->clear();
        vocabulary->clear();
        vocabularyNames->clear();
        types->clear();
        predicates->clear();
        dialog.findChild<QCheckBox*>("relateDecompose")->setChecked(false);
        objectSource->setCurrentIndex(0);
        int frames = 0;
        QObject::connect(
                video, &RelateAnythingVideoWidget::frameReady, &dialog,
                [&](const QImage& image, int) {
                    ASSERT_TRUE(!image.isNull())
                            << QString("decoded video frame").toStdString();
                    ++frames;
                });
        tabs->setCurrentIndex(tabs->indexOf(video->parentWidget()));
        run->click();
        // Frame N+1 is only delivered after frame N's result is
        // acknowledged, so four delivered frames means three finished
        // results.
        ASSERT_TRUE(wait([&] { return frames >= 4; }))
                << QString("consumer-driven video progresses").toStdString();
        {
            // Every video frame must have gone through the ported
            // tracker + belief filter (upstream render_video.py).
            ASSERT_TRUE(dialog.saveResults(dir + "/video-frames.json"))
                    << QString("save video frame results").toStdString();
            QFile vf(dir + "/video-frames.json");
            ASSERT_TRUE(vf.open(QIODevice::ReadOnly))
                    << QString("read video results").toStdString();
            const auto rows = QJsonDocument::fromJson(vf.readAll()).array();
            ASSERT_TRUE(rows.size() >= 3)
                    << QString("one result row per video frame").toStdString();
            for (const auto& row : rows)
                ASSERT_TRUE(row.toObject()["temporal"].toBool() &&
                            row.toObject()["tracks"].toInt() >= 0)
                        << QString("video frame used temporal stabilisation")
                                   .toStdString();
            ASSERT_TRUE(rows.last().toObject()["tracks"].toInt() > 0)
                    << QString("tracks persist across video frames")
                               .toStdString();
        }
        auto previews = video->findChildren<ecvClickableImageLabel*>();
        ASSERT_TRUE(!previews.isEmpty() &&
                    !qtCompatLabelPixmap(previews.front()).isNull())
                << QString("annotated video preview").toStdString();
        dialog.findChild<QPushButton*>("relateCancel")->click();
        ASSERT_TRUE(wait([&] { return run->isEnabled(); }))
                << QString("cancel drains pending frame").toStdString();
        tabs->setCurrentIndex(0);
        list->clear();
        auto* missing = new QListWidgetItem(dir + "/missing-input.jpg", list);
        missing->setCheckState(Qt::Checked);
        run->click();
        ASSERT_TRUE(wait([&] { return run->isEnabled(); }) &&
                    table->rowCount() == 0)
                << QString("invalid input clears results and releases worker")
                           .toStdString();
        sample->click();
        run->click();
        ASSERT_TRUE(wait([&] {
            return table->rowCount() > 0 && run->isEnabled();
        })) << QString("successful inference after invalid input")
                        .toStdString();
        dialog.grab().save(dir + "/dialog.png");
        std::cout << "Image/batch, supplied boxes/masks, custom/external "
                     "vocabulary, decomposition, DB handoff, exports, three "
                     "paced video frames and error recovery passed\n";
    }
    dialog.close();
    dialog.show();
    ASSERT_TRUE(run->isEnabled()) << QString("reopen").toStdString();
    dialog.shutdown();
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    QApplication app(argc, argv);
    workflowArguments = app.arguments();
    return RUN_ALL_TESTS();
}
