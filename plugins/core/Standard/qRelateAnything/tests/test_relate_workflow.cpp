// SPDX-License-Identifier: MIT
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCommandLineParser>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QListWidget>
#include <QPluginLoader>
#include <QPushButton>
#include <QSettings>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <iostream>
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
        check(model->count() == 9, "all nine relation models");
        check(!dialog.findChild<QCheckBox*>("relateStrict")->isChecked(),
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
            check(list->item(0)->text().endsWith("/demo_images/skateboard.jpg"),
                  "deterministic first image");
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
                                               !result.json["relations"].toArray().isEmpty(),
                                       "DB image and typed metadata");
                                 ++dbRequests;
                             });
            dialog.findChild<QPushButton*>("relateAddToDb")->click();
            check(dbRequests == 1, "DB result handoff");
            list->clearSelection();
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
            int frames = 0;
            QObject::connect(video, &RelateAnythingVideoWidget::frameReady,
                             &dialog, [&](const QImage& image, int) {
                                 check(!image.isNull(), "decoded video frame");
                                 ++frames;
                             });
            tabs->setCurrentIndex(tabs->indexOf(video->parentWidget()));
            run->click();
            check(wait([&] { return frames >= 3; }),
                  "consumer-driven video progresses");
            auto previews = video->findChildren<ecvClickableImageLabel*>();
            check(!previews.isEmpty() &&
                          !previews.front()->pixmap(Qt::ReturnByValue).isNull(),
                  "annotated video preview");
            dialog.findChild<QPushButton*>("relateCancel")->click();
            check(wait([&] { return run->isEnabled(); }),
                  "cancel drains pending frame");
            tabs->setCurrentIndex(0);
            list->clear();
            list->addItem(dir + "/missing-input.jpg");
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
            std::cout << "Image/batch, DB handoff, JSON/CSV/text exports, three "
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
