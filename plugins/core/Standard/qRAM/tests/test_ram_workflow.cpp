// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QtCompat.h>
#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCommandLineParser>
#include <QCryptographicHash>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QPluginLoader>
#include <QProcess>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "RAMDialog.h"
#include "RAMVideoWidget.h"
#include "RAMWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvTestDataRepository.h"

namespace {
QStringList workflowArguments;

class RAMWorkflow : public ::testing::Test {
protected:
    void SetUp() override {
        qRegisterMetaType<RAMResult>("RAMResult");
        ASSERT_TRUE(settingsDir.isValid());
        app.setOrganizationName("ACloudViewerTests");
        app.setApplicationName("RAMWorkflow");
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                           settingsDir.path());
        QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope,
                           settingsDir.path());
        parser.addHelpOption();
        parser.addOption(
                {"real",
                 "Run cached real-model image, export and lifecycle checks."});
        parser.addOption(
                {"download-sample",
                 "Exercise Use test data from an empty sample cache."});
        parser.addOption(
                {"caption", "Use Tag2Text Q8 and verify generated captions."});
        parser.addOption(
                {"ram", "Use RAM Q8 and verify CPU arithmetic switching."});
        parser.addOption(
                {"backend", "Real-inference backend.", "backend", "cuda"});
        parser.addOption({"switch-backend",
                          "Switch the live model session to this backend.",
                          "backend"});
        parser.addOption({"video",
                          "Exercise consumer-driven video using this fixture.",
                          "path"});
        parser.addOption({"output-dir",
                          "Keep exports and dialog screenshots here.", "path"});
        parser.addOption({"timeout-ms",
                          "Per-operation wait limit, including downloads.",
                          "milliseconds", "120000"});
        parser.addOption({"plugin-module",
                          "Load and check the built plugin module.", "path"});
        parser.addOption({"aicore-library",
                          "Record the built AICore runtime library.", "path"});
        parser.addOption(
                {"upstream-binary",
                 "Compare UI options with this upstream ram-ggml build.",
                 "path"});
        parser.addOption(
                {"label-space-dir",
                 "Upstream cpp_ggml/benchmarks/label_spaces directory.",
                 "path"});
        parser.addOption(
                {"upstream-logit-tolerance",
                 "CPU all-class compatibility bound; GPU differences are "
                 "recorded as observational metrics.",
                 "value", "0.0001"});
        parser.process(workflowArguments);
        bool timeoutValid = false;
        workflowTimeout = parser.value("timeout-ms").toInt(&timeoutValid);
        ASSERT_TRUE(timeoutValid && workflowTimeout > 0)
                << "timeout-ms must be positive";
    }
    QApplication& app = *qApp;
    QTemporaryDir settingsDir;
    QCommandLineParser parser;
    int workflowTimeout = 0;
};

TEST_F(RAMWorkflow, DiscoversBuiltPluginActionAndIcon) {
    if (!parser.isSet("plugin-module"))
        GTEST_SKIP() << "No plugin-module argument supplied";
    if (parser.isSet("plugin-module")) {
        QPluginLoader loader(
                QFileInfo(parser.value("plugin-module")).absoluteFilePath());
        auto* plugin = loader.instance();
        ASSERT_TRUE(plugin != nullptr)
                << QString(qPrintable(loader.errorString())).toStdString();
        ASSERT_TRUE(loader.metaData().value("IID").toString() ==
                    "cvcorp.cloudviewer.plugin.qRAM")
                << QString("plugin registration IID").toStdString();
        if (plugin) {
            const auto actions = plugin->findChildren<QAction*>();
            ASSERT_TRUE(actions.size() == 1 &&
                        actions[0]->text().contains("RAM++") &&
                        !actions[0]->icon().isNull())
                    << QString("loaded plugin provides its RAM++ action and "
                               "icon")
                               .toStdString();
            std::cout << "RAM plugin module loaded through QPluginLoader\n";
        }
    }
}

TEST_F(RAMWorkflow, DialogAndOptionalRealModelLifecycle) {
    const QString configuredRoot =
            qEnvironmentVariable("CLOUDVIEWER_DATA_ROOT");
    if (!configuredRoot.isEmpty()) {
        EXPECT_TRUE(ecvTestDataRepository::dataRoot() == configuredRoot)
                << QString("sample and model caches honor the same data root")
                           .toStdString();
    }
    auto waitFor = [&](const std::function<bool()>& predicate,
                       int timeout = -1) {
        if (timeout < 0) timeout = workflowTimeout;
        QElapsedTimer timer;
        timer.start();
        while (!predicate() && timer.elapsed() < timeout) {
            app.processEvents(QEventLoop::AllEvents, 10);
            QThread::msleep(5);
        }
        return predicate();
    };
    auto readJson = [](const QString& path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return QJsonArray();
        return QJsonDocument::fromJson(file.readAll()).array();
    };
    auto sha256File = [](const QString& path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return QString();
        return QString(QCryptographicHash::hash(file.readAll(),
                                                QCryptographicHash::Sha256)
                               .toHex());
    };
    auto candidateAICoreLibrary = [&]() {
        const QString configured = parser.value("aicore-library");
        if (!configured.isEmpty()) return configured;
        const QStringList names = {
#if defined(Q_OS_WIN)
                QStringLiteral("AICore.dll"),
                QStringLiteral("AICore_d.dll"),
#elif defined(Q_OS_MACOS) || defined(Q_OS_MAC)
                QStringLiteral("libAICore.dylib"),
#else
                QStringLiteral("libAICore.so"),
#endif
        };
        QDir directory(QCoreApplication::applicationDirPath());
        for (int level = 0; level < 4; ++level) {
            for (const QString& name : names) {
                const QString candidate = directory.filePath(name);
                if (QFileInfo::exists(candidate)) return candidate;
            }
            if (!directory.cdUp()) break;
        }
        return QString();
    };
    QTemporaryDir outputs;
    const QString outputDir = parser.isSet("output-dir")
                                      ? parser.value("output-dir")
                                      : outputs.path();
    QDir().mkpath(outputDir);
    {
        RAMDialog dialog;
        auto* model = dialog.findChild<QComboBox*>("ramModel");
        auto* device = dialog.findChild<QComboBox*>("ramDevice");
        auto* precision = dialog.findChild<QComboBox*>("ramCpuPrecision");
        auto* sample = dialog.findChild<QPushButton*>("ramUseTestData");
        auto* run = dialog.findChild<QPushButton*>("ramRun");
        auto* list = dialog.findChild<QListWidget*>("ramInputs");
        auto* table = dialog.findChild<QTableWidget*>("ramTags");
        auto* tabs = dialog.findChild<QTabWidget*>("ramTabs");
        auto* caption = dialog.findChild<QCheckBox*>("ramCaption");
        EXPECT_TRUE(model && model->count() == 9)
                << QString("all catalog models in selector").toStdString();
        EXPECT_TRUE(tabs && tabs->tabText(0) == "Images / batch" &&
                    tabs->tabText(1) == "Results" &&
                    tabs->tabText(2) == "Options" &&
                    tabs->tabText(3) == "Video / camera")
                << QString("workflow tabs put video after image results and "
                           "options")
                           .toStdString();
        EXPECT_TRUE(dialog.findChild<QPushButton*>("ramLoadSampleVideo") ==
                    nullptr)
                << QString("video page reuses the shared sample button instead "
                           "of a "
                           "duplicate")
                           .toStdString();
        EXPECT_TRUE(dialog.findChild<QLabel*>("ramQuickStart") != nullptr)
                << QString("dialog exposes quick-start model guidance")
                           .toStdString();
        EXPECT_TRUE(device && device->count() >= 2)
                << QString("device selector").toStdString();
        EXPECT_TRUE(precision && precision->count() == 2 &&
                    precision->currentData().toInt() == AICORE_TAGGING_CPU_AUTO)
                << QString("CPU arithmetic defaults to automatic precision")
                           .toStdString();
        if (precision) {
            precision->setCurrentIndex(1);
            EXPECT_TRUE(precision->currentData().toInt() ==
                        AICORE_TAGGING_CPU_UPSTREAM)
                    << QString("explicit upstream CPU arithmetic remains "
                               "selectable")
                               .toStdString();
            precision->setCurrentIndex(0);
        }
        EXPECT_TRUE(sample && run && list && table && tabs && caption)
                << QString("complete workflow controls").toStdString();
        EXPECT_TRUE(dialog.findChildren<QPlainTextEdit*>().size() == 1)
                << QString("dialog keeps only the result caption text pane, "
                           "not a local log "
                           "pane")
                           .toStdString();
        if (!model || !device || !sample || !run || !list || !table || !tabs ||
            !caption)
            FAIL() << "Required workflow prerequisite failed";
        EXPECT_TRUE(model->currentData().toInt() ==
                    aicore_tagging_model_default_index())
                << QString("default model follows catalog").toStdString();
        const auto* defaultModel =
                aicore_tagging_model_at(model->currentData().toInt());
        EXPECT_TRUE(defaultModel && defaultModel->supports_caption &&
                    caption->isEnabled() && caption->isChecked())
                << QString("default Tag2Text model enables caption for "
                           "one-click sample "
                           "use")
                           .toStdString();
        model->setCurrentIndex(6);
        EXPECT_TRUE(caption->isEnabled())
                << QString("Tag2Text enables caption pipeline").toStdString();
        auto* epsilon =
                dialog.findChild<QDoubleSpinBox*>("ramThresholdEpsilon");
        EXPECT_TRUE(epsilon && epsilon->isEnabled() && epsilon->value() == 0)
                << QString("Tag2Text exposes upstream default threshold margin")
                           .toStdString();
        model->setCurrentIndex(0);
        EXPECT_TRUE(!caption->isEnabled() && !caption->isChecked())
                << QString("RAM++ disables unsupported caption").toStdString();
        EXPECT_TRUE(epsilon && !epsilon->isEnabled())
                << QString("RAM++ disables Tag2Text threshold margin")
                           .toStdString();
        auto demo = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::RAM, "demo1.jpg");
        if (parser.isSet("download-sample")) {
            EXPECT_TRUE(demo.isEmpty())
                    << QString("cold sample download starts without demo1")
                               .toStdString();
            if (!demo.isEmpty())
                FAIL() << "Required workflow prerequisite failed";
            sample->click();
            EXPECT_TRUE(waitFor([&]() {
                return sample->isEnabled() && list->count() == 4;
            })) << QString("shared sample download and extraction complete")
                            .toStdString();
            demo = ecvTestDataRepository::findDatasetFile(
                    ecvTestDataRepository::Dataset::RAM, "demo1.jpg");
        }
        if (!demo.isEmpty()) {
            sample->click();
            EXPECT_TRUE(list->count() == 4 && list->item(0)->text() == demo &&
                        list->currentRow() == 0)
                    << QString("all official demo images are filled with demo1 "
                               "selected")
                               .toStdString();
            auto* video = dialog.findChild<RAMVideoWidget*>();
            if (video && VideoPlaybackWidget::isAvailable()) {
                EXPECT_TRUE(tabs->currentIndex() == 0 &&
                            QFileInfo(video->videoFilePath()).isFile() &&
                            sha256File(video->videoFilePath()) ==
                                    "53c17ba6b53bde9b21660142a1685bfa8ce4a32d"
                                    "6ab9c0cc08f1c57ce944b202")
                        << QString("one sample action also selects the "
                                   "verified video")
                                   .toStdString();
                video->setVideoFilePath(QString());
                tabs->setCurrentIndex(3);
                sample->click();
                EXPECT_TRUE(tabs->currentIndex() == 3 && list->count() == 4 &&
                            QFileInfo(video->videoFilePath()).isFile())
                        << QString("sample action on the video page stays "
                                   "there")
                                   .toStdString();
                tabs->setCurrentIndex(0);
            }
        }
        if (parser.isSet("real")) {
            EXPECT_TRUE(!demo.isEmpty())
                    << QString("real workflow requires verified cached "
                               "official sample")
                               .toStdString();
            if (demo.isEmpty())
                FAIL() << "Required workflow prerequisite failed";
            QTemporaryDir extracted;
            const QString zip = ecvTestDataRepository::zipPath(
                    ecvTestDataRepository::Dataset::RAM);
            EXPECT_TRUE(
                    ecvTestDataRepository::extractZip(zip, extracted.path()))
                    << QString("shared ZIP extracts offline").toStdString();
            const QString member =
                    extracted.filePath("ram_datasets/demo/demo1.jpg");
            QFile file(member);
            EXPECT_TRUE(file.open(QIODevice::ReadOnly))
                    << QString("official ZIP contains demo1").toStdString();
            const QByteArray digest =
                    QCryptographicHash::hash(file.readAll(),
                                             QCryptographicHash::Sha256)
                            .toHex();
            EXPECT_TRUE(digest ==
                        "1b2906f4058a69936df49cb6156ec4cd117a286b420e1eb14764"
                        "033bf8f3c05f")
                    << QString("shared extraction preserves official bytes")
                               .toStdString();
            file.close();
            EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate))
                    << QString("isolated corruption fixture opens")
                               .toStdString();
            file.write("corrupt");
            file.close();
            EXPECT_TRUE(
                    ecvTestDataRepository::extractZip(zip, extracted.path()) &&
                    !QImage(member).isNull())
                    << QString("cached ZIP repairs corrupt member offline")
                               .toStdString();
            const bool testCaption = parser.isSet("caption");
            const int classCount = testCaption ? 3429 : 4585;
            model->setCurrentIndex(testCaption           ? 8
                                   : parser.isSet("ram") ? 5
                                                         : 2);
            caption->setChecked(testCaption);
            int selectedDevice = -1;
            for (int i = 0; i < device->count(); ++i)
                if (device->itemData(i).toString().startsWith(
                            parser.value("backend"), Qt::CaseInsensitive)) {
                    selectedDevice = i;
                    break;
                }
            EXPECT_TRUE(selectedDevice >= 0)
                    << QString("requested backend is available").toStdString();
            if (selectedDevice < 0)
                FAIL() << "Required workflow prerequisite failed";
            device->setCurrentIndex(selectedDevice);
            dialog.show();
            app.processEvents();
            dialog.grab().save(QDir(outputDir).filePath("ram-inputs.png"));
            auto* second = new QListWidgetItem("db://sample,\"quoted\"", list);
            second->setData(Qt::UserRole, QImage(demo));
            list->selectAll();
            run->click();
            EXPECT_TRUE(waitFor([&]() {
                return run->isEnabled() && tabs->currentIndex() == 1 &&
                       table->rowCount() > 0;
            })) << QString("real batch completes").toStdString();
            const QString jsonPath =
                    QDir(outputDir).filePath("ram-results.json");
            EXPECT_TRUE(dialog.saveResults(jsonPath))
                    << QString("JSON export succeeds").toStdString();
            auto rows = readJson(jsonPath);
            EXPECT_TRUE(rows.size() == 5)
                    << QString("batch export contains every input")
                               .toStdString();
            if (rows.size() == 5) {
                EXPECT_TRUE(rows[0].toObject()["tags"].toArray().size() ==
                            classCount)
                        << QString("exports every model class").toStdString();
                if (testCaption) {
                    const auto sentence =
                            rows[0].toObject()["caption"].toString();
                    EXPECT_TRUE(
                            !sentence.isEmpty() &&
                            sentence ==
                                    rows[4].toObject()["caption"].toString())
                            << QString("resident Tag2Text captions agree")
                                       .toStdString();
                }
                EXPECT_TRUE(rows[0].toObject()["selected_tags"].toArray() ==
                            rows[4].toObject()["selected_tags"].toArray())
                        << QString("resident session repeated tags agree")
                                   .toStdString();
                for (auto value : rows[0].toObject()["tags"].toArray()) {
                    auto tag = value.toObject();
                    EXPECT_TRUE(std::isfinite(tag["logit"].toDouble()) &&
                                tag["probability"].toDouble() >= 0 &&
                                tag["probability"].toDouble() <= 1)
                            << QString("finite full-class output")
                                       .toStdString();
                }
            }
            EXPECT_TRUE(dialog.saveResults(
                    QDir(outputDir).filePath("ram-results.csv")))
                    << QString("CSV export succeeds").toStdString();
            QFile csv(QDir(outputDir).filePath("ram-results.csv"));
            EXPECT_TRUE(
                    csv.open(QIODevice::ReadOnly) &&
                    csv.readAll().contains("\"db://sample,\"\"quoted\"\"\""))
                    << QString("CSV escapes commas and embedded quotes")
                               .toStdString();
            EXPECT_TRUE(dialog.saveResults(
                    QDir(outputDir).filePath("ram-results.txt")))
                    << QString("text export succeeds").toStdString();
            int published = 0;
            QObject::connect(
                    &dialog, &RAMDialog::addToDbRequested, &app,
                    [&](const RAMResult& result) {
                        EXPECT_TRUE(!result.image.isNull() &&
                                    result.annotated.size() ==
                                            result.image.size() &&
                                    result.annotated !=
                                            result.image.convertToFormat(
                                                    QImage::Format_RGB32) &&
                                    result.json["tags"].toArray().size() ==
                                            classCount)
                                << QString("DB boundary retains typed result, "
                                           "decoded "
                                           "image and tag overlay")
                                           .toStdString();
                        ++published;
                    });
            dialog.findChild<QPushButton*>("ramAddToDb")->click();
            EXPECT_TRUE(published == 1)
                    << QString("one DB publication").toStdString();
            auto* selected = dialog.findChild<QCheckBox*>("ramSelectedOnly");
            auto* filter = dialog.findChild<QLineEdit*>("ramFilter");
            selected->setChecked(false);
            EXPECT_TRUE(table->rowCount() == classCount)
                    << QString("all-class results toggle").toStdString();
            filter->setText("dog");
            EXPECT_TRUE(table->rowCount() > 0 && table->rowCount() < classCount)
                    << QString("tag search filters results").toStdString();
            filter->clear();
            selected->setChecked(true);
            dialog.grab().save(QDir(outputDir).filePath("ram-results.png"));
            tabs->setCurrentIndex(0);
            run->click();
            dialog.close();
            EXPECT_TRUE(waitFor([&]() { return run->isEnabled(); }))
                    << QString("close during inference releases pending job")
                               .toStdString();
            EXPECT_TRUE(table->rowCount() == 0)
                    << QString("cancelled generation publishes no stale result")
                               .toStdString();
            dialog.show();
            sample->click();
            run->click();
            EXPECT_TRUE(waitFor([&]() {
                return run->isEnabled() && tabs->currentIndex() == 1 &&
                       table->rowCount() > 0;
            })) << QString("reopen runs successfully").toStdString();
            if (parser.isSet("switch-backend")) {
                int target = -1;
                for (int i = 0; i < device->count(); ++i)
                    if (device->itemData(i).toString().startsWith(
                                parser.value("switch-backend"),
                                Qt::CaseInsensitive)) {
                        target = i;
                        break;
                    }
                EXPECT_TRUE(target >= 0)
                        << QString("switch target backend available")
                                   .toStdString();
                if (target >= 0) {
                    device->setCurrentIndex(target);
                    tabs->setCurrentIndex(0);
                    run->click();
                    EXPECT_TRUE(waitFor([&]() {
                        return run->isEnabled() && tabs->currentIndex() == 1 &&
                               table->rowCount() > 0;
                    })) << QString("live device change recreates session")
                                    .toStdString();
                    const QString changed =
                            QDir(outputDir).filePath("ram-device-switch.json");
                    EXPECT_TRUE(dialog.saveResults(changed))
                            << QString("device switch result export")
                                       .toStdString();
                    const auto records = readJson(changed);
                    EXPECT_TRUE(
                            records.size() == 1 &&
                            records[0]
                                    .toObject()["device"]
                                    .toString()
                                    .startsWith(parser.value("switch-backend"),
                                                Qt::CaseInsensitive))
                            << QString("result identifies switched device")
                                       .toStdString();
                }
            }
            if (parser.isSet("ram") && parser.value("backend") == "cpu" &&
                precision) {
                precision->setCurrentIndex(1);
                tabs->setCurrentIndex(0);
                run->click();
                EXPECT_TRUE(waitFor([&]() {
                    return run->isEnabled() && tabs->currentIndex() == 1 &&
                           table->rowCount() > 0;
                })) << QString("switching CPU arithmetic recreates the "
                               "resident "
                               "session")
                                .toStdString();
                const auto compatibilityPath =
                        QDir(outputDir).filePath("ram-upstream-results.json");
                EXPECT_TRUE(dialog.saveResults(compatibilityPath))
                        << QString("CPU compatibility export succeeds")
                                   .toStdString();
                const auto compatible = readJson(compatibilityPath);
                EXPECT_TRUE(
                        compatible.size() == 1 &&
                        compatible[0].toObject()["cpu_precision"].toString() ==
                                "upstream")
                        << QString("exports identify the effective CPU "
                                   "arithmetic option")
                                   .toStdString();
                precision->setCurrentIndex(0);
            }
            if (parser.isSet("video")) {
                auto* video = dialog.findChild<RAMVideoWidget*>();
                EXPECT_TRUE(video && video->isAvailable())
                        << QString("video pipeline available").toStdString();
                if (video && video->isAvailable()) {
                    int frames = 0;
                    QObject::connect(video, &RAMVideoWidget::frameReady, &app,
                                     [&](const QImage&, int) { ++frames; });
                    video->selectVideoFileSource();
                    video->setVideoFilePath(parser.value("video"));
                    // Only the consumer acknowledgement may paint the
                    // preview in ConsumerDriven mode.
                    for (auto* label :
                         video->findChildren<ecvClickableImageLabel*>())
                        label->clear();
                    tabs->setCurrentIndex(3);
                    run->click();
                    EXPECT_TRUE(waitFor([&]() {
                        return frames >= 3 && run->isEnabled();
                    })) << QString("consumer-driven video processes three "
                                   "frames")
                                    .toStdString();
                    {
                        auto labels =
                                video->findChildren<ecvClickableImageLabel*>();
                        const bool painted = std::any_of(
                                labels.cbegin(), labels.cend(),
                                [](const ecvClickableImageLabel* label) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
                                    return !label->pixmap().isNull();
#else
                                    return label->pixmap() &&
                                           !label->pixmap()->isNull();
#endif
                                });
                        EXPECT_TRUE(painted)
                                << QString("consumer-driven preview paints the "
                                           "annotated "
                                           "frame")
                                           .toStdString();
                        dialog.grab().save(
                                QDir(outputDir).filePath("ram-video.png"));
                    }
                    dialog.findChild<QPushButton*>("ramCancel")->click();
                    EXPECT_TRUE(waitFor([&]() { return run->isEnabled(); }))
                            << QString("video stop releases pending inference")
                                       .toStdString();
                    EXPECT_TRUE(dialog.saveResults(
                            QDir(outputDir).filePath("ram-video.json")))
                            << QString("video results export").toStdString();
                    EXPECT_TRUE(
                            readJson(QDir(outputDir).filePath("ram-video.json"))
                                    .size() >= 3)
                            << QString("every acknowledged frame has a result")
                                       .toStdString();
                    const int previous = frames;
                    video->restartVideoFile();
                    EXPECT_TRUE(waitFor([&]() {
                        return frames >= previous + 2 && run->isEnabled();
                    })) << QString("video restart recovers consumer gate")
                                    .toStdString();
                    dialog.findChild<QPushButton*>("ramCancel")->click();
                    EXPECT_TRUE(waitFor([&]() {
                        return run->isEnabled();
                    })) << QString("restart test begins with an idle worker")
                                    .toStdString();
                    bool restartedWhileBusy = false;
                    int restartedFrames = 0;
                    auto restartConnection = QObject::connect(
                            video, &RAMVideoWidget::frameReady, &app,
                            [&](const QImage&, int) {
                                ++restartedFrames;
                                if (restartedFrames == 1)
                                    QTimer::singleShot(5, &dialog, [&]() {
                                        restartedWhileBusy = !run->isEnabled();
                                        video->restartVideoFile();
                                    });
                            });
                    run->click();
                    EXPECT_TRUE(waitFor([&]() {
                        return restartedFrames >= 4 && run->isEnabled();
                    })) << QString("restart during inference completes "
                                   "replacement "
                                   "frames")
                                    .toStdString();
                    dialog.findChild<QPushButton*>("ramCancel")->click();
                    EXPECT_TRUE(waitFor([&]() { return run->isEnabled(); }))
                            << QString("busy restart releases the worker")
                                       .toStdString();
                    QObject::disconnect(restartConnection);
                    const QString busyRestartFile = QDir(outputDir).filePath(
                            "ram-video-restart-busy.json");
                    EXPECT_TRUE(dialog.saveResults(busyRestartFile))
                            << QString("busy restart results export")
                                       .toStdString();
                    const auto restartedRows = readJson(busyRestartFile);
                    EXPECT_TRUE(restartedRows.size() >= 2 &&
                                restartedFrames <= restartedRows.size() + 2)
                            << QString("restart holds the consumer gate "
                                       "instead of dropping "
                                       "frames")
                                       .toStdString();
                    if (!restartedRows.isEmpty())
                        EXPECT_TRUE(restartedRows.first()
                                            .toObject()["source"]
                                            .toString()
                                            .endsWith("#frame=1"))
                                << QString("replacement stream starts with its "
                                           "first frame")
                                           .toStdString();
                    if (parser.value("backend") == "cpu")
                        EXPECT_TRUE(restartedWhileBusy)
                                << QString("CPU fixture actually restarted an "
                                           "in-flight "
                                           "inference")
                                           .toStdString();
                }
            }
        }
        dialog.close();
        dialog.shutdown();
        dialog.shutdown();
    }
    if (parser.isSet("upstream-binary")) {
        // Each case drives the dialog controls, exports the typed result and
        // runs the original ram-ggml with the same model, image, backend,
        // threads and option arguments. Gates are the manifest's upstream
        // compatibility bounds: logits <= 1e-4, identical tags and captions.
        const QFileInfo upstream(parser.value("upstream-binary"));
        const QDir spaces(parser.value("label-space-dir"));
        const QString demo = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::RAM, "demo1.jpg");
        const QString space =
                spaces.filePath("openimages_common_214__ram_plus");
        const bool ready = upstream.isExecutable() && !demo.isEmpty() &&
                           QFileInfo::exists(space + ".f32");
        EXPECT_TRUE(ready) << QString("option comparison requires upstream "
                                      "binary, label spaces and "
                                      "cached demo1")
                                      .toStdString();
        if (!ready) FAIL() << "Required workflow prerequisite failed";
        RAMDialog dialog;
        auto* model = dialog.findChild<QComboBox*>("ramModel");
        auto* device = dialog.findChild<QComboBox*>("ramDevice");
        auto* sample = dialog.findChild<QPushButton*>("ramUseTestData");
        auto* run = dialog.findChild<QPushButton*>("ramRun");
        auto* tabs = dialog.findChild<QTabWidget*>("ramTabs");
        auto* caption = dialog.findChild<QCheckBox*>("ramCaption");
        auto* threads = dialog.findChild<QSpinBox*>("ramThreads");
        auto* defaults = dialog.findChild<QCheckBox*>("ramDefaultThresholds");
        auto* threshold = dialog.findChild<QDoubleSpinBox*>("ramThreshold");
        auto* epsilon =
                dialog.findChild<QDoubleSpinBox*>("ramThresholdEpsilon");
        auto* length = dialog.findChild<QSpinBox*>("ramCaptionLength");
        auto* beam = dialog.findChild<QComboBox*>("ramBeam");
        auto* specified = dialog.findChild<QLineEdit*>("ramSpecifiedTags");
        auto* tagList = dialog.findChild<QLineEdit*>("ramTagList");
        auto* thresholdFile = dialog.findChild<QLineEdit*>("ramThresholdFile");
        auto* embedding = dialog.findChild<QLineEdit*>("ramEmbeddingFile");
        const bool controls = model && device && sample && run && tabs &&
                              caption && threads && defaults && threshold &&
                              epsilon && length && beam && specified &&
                              tagList && thresholdFile && embedding;
        EXPECT_TRUE(controls)
                << QString("every option control is addressable").toStdString();
        if (!controls) FAIL() << "Required workflow prerequisite failed";
        const QString backend = parser.value("backend");
        const bool logitParityRequired =
                backend.compare("cpu", Qt::CaseInsensitive) == 0;
        bool toleranceValid = false;
        const double logitTolerance = parser.value("upstream-logit-tolerance")
                                              .toDouble(&toleranceValid);
        EXPECT_TRUE(toleranceValid && logitTolerance >= 0)
                << QString("upstream logit tolerance is a non-negative number")
                           .toStdString();
        if (!toleranceValid || logitTolerance < 0)
            FAIL() << "Required workflow prerequisite failed";
        int selectedDevice = -1;
        for (int i = 0; i < device->count(); ++i)
            if (device->itemData(i).toString().startsWith(backend,
                                                          Qt::CaseInsensitive))
                selectedDevice = i;
        EXPECT_TRUE(selectedDevice >= 0)
                << QString("option backend is available").toStdString();
        if (selectedDevice < 0)
            FAIL() << "Required workflow prerequisite failed";
        device->setCurrentIndex(selectedDevice);
        dialog.show();
        std::unique_ptr<char, decltype(&aicore_tagging_free_buffer)> cache(
                aicore_tagging_model_cache_dir(), aicore_tagging_free_buffer);
        const QDir out(outputDir);
        QFile manifest(out.filePath("ram-options-manifest.txt"));
        manifest.open(QIODevice::WriteOnly | QIODevice::Truncate);
        manifest.write(demo.toUtf8() + "\n");
        manifest.close();
        // Mirrors upstream_backend_env: the upstream process must load its own
        // GGML libraries rather than the candidate's same-soname build.
        auto env = QProcessEnvironment::systemEnvironment();
        const QString build = upstream.absolutePath(),
                      ggml = build + "/third_party/ggml/src";
        QString libraryPathVariable;
#if defined(Q_OS_WIN)
        libraryPathVariable = QStringLiteral("PATH");
#elif defined(Q_OS_MACOS) || defined(Q_OS_MAC)
        libraryPathVariable = QStringLiteral("DYLD_LIBRARY_PATH");
#else
        libraryPathVariable = QStringLiteral("LD_LIBRARY_PATH");
#endif
        QStringList libraries{ggml, ggml + "/ggml-" + backend, build + "/bin"};
        if (env.contains(libraryPathVariable))
            libraries << env.value(libraryPathVariable);
        env.insert(libraryPathVariable, libraries.join(QDir::listSeparator()));
        env.remove("GGML_BACKEND_PATH");
        struct OptionCase {
            QString name;
            int model;  // selector row: 2 RAM++ Q8, 8 Tag2Text Q8
            bool caption;
            bool customSpace;
            double threshold;  // -1 keeps per-class thresholds
            double epsilon;
            int beam;
            int length;
            QString specified;
            QStringList upstream;
        };
        const std::vector<OptionCase> cases = {
                {"custom-label-space",
                 2,
                 false,
                 true,
                 -1,
                 0,
                 0,
                 30,
                 {},
                 {"--tag-list", space + ".tags.txt", "--thresholds",
                  space + ".thresholds.txt", "--label-embedding",
                  space + ".f32"}},
                {"global-threshold",
                 2,
                 false,
                 false,
                 .6,
                 0,
                 0,
                 30,
                 {},
                 {"--threshold", "0.6"}},
                {"tag2text-batch-beams",
                 8,
                 true,
                 false,
                 -1,
                 0,
                 3,
                 30,
                 {},
                 {"--task", "caption", "--beam-batch", "3", "--max-length",
                  "30"}},
                {"tag2text-specified-sequential",
                 8,
                 true,
                 false,
                 -1,
                 .05,
                 1,
                 20,
                 "dog,grass",
                 {"--task", "caption", "--beam-batch", "1", "--max-length",
                  "20", "--specified-tags", "dog,grass", "--threshold-epsilon",
                  "0.05"}},
        };
        const QString upstreamDigest = sha256File(upstream.absoluteFilePath());
        QJsonArray evidence;
        QHash<QString, QJsonArray> exported;
        for (const auto& c : cases) {
            model->setCurrentIndex(c.model);
            caption->setChecked(c.caption);
            threads->setValue(8);
            defaults->setChecked(c.threshold < 0);
            if (c.threshold >= 0) threshold->setValue(c.threshold);
            epsilon->setValue(c.epsilon);
            length->setValue(c.length);
            beam->setCurrentIndex(beam->findData(c.beam));
            specified->setText(c.specified);
            tagList->setText(c.customSpace ? space + ".tags.txt" : QString());
            thresholdFile->setText(c.customSpace ? space + ".thresholds.txt"
                                                 : QString());
            embedding->setText(c.customSpace ? space + ".f32" : QString());
            sample->click();
            run->click();
            EXPECT_TRUE(waitFor([&]() {
                return run->isEnabled() && tabs->currentIndex() == 1;
            })) << QString("option case completes in the UI").toStdString();
            const QString exportPath =
                    out.filePath("ram-options-" + c.name + ".json");
            EXPECT_TRUE(dialog.saveResults(exportPath))
                    << QString("option case export").toStdString();
            const auto rows = readJson(exportPath);
            EXPECT_TRUE(rows.size() == 1)
                    << QString("option case exports one result").toStdString();
            if (rows.size() != 1) continue;
            const auto row = rows[0].toObject();
            const auto tags = row["tags"].toArray();
            const QString jsonl = out.filePath("ram-options-" + c.name +
                                               "-upstream.jsonl"),
                          matrix = out.filePath("ram-options-" + c.name +
                                                "-upstream.f32");
            QStringList args{
                    "--model",
                    QDir(QString::fromUtf8(cache.get()))
                            .filePath(QString::fromUtf8(
                                    aicore_tagging_model_at(
                                            model->itemData(c.model).toInt())
                                            ->filename)),
                    "--backend",
                    backend,
                    "--threads",
                    "8",
                    "--warmup",
                    "0",
                    "--repeat",
                    "1",
                    "--manifest",
                    manifest.fileName(),
                    "--jsonl",
                    jsonl,
                    "--logits-matrix",
                    matrix};
            args << c.upstream;
            QProcess process;
            process.setProcessEnvironment(env);
            process.setProcessChannelMode(QProcess::MergedChannels);
            process.start(upstream.absoluteFilePath(), args);
            const bool finished = process.waitForFinished(workflowTimeout);
            QFile log(out.filePath("ram-options-" + c.name + "-upstream.log"));
            if (log.open(QIODevice::WriteOnly | QIODevice::Truncate))
                log.write(process.readAll());
            EXPECT_TRUE(finished &&
                        process.exitStatus() == QProcess::NormalExit &&
                        process.exitCode() == 0)
                    << QString("upstream option run succeeds").toStdString();
            QFile jsonlFile(jsonl), matrixFile(matrix);
            jsonlFile.open(QIODevice::ReadOnly);
            matrixFile.open(QIODevice::ReadOnly);
            const auto reference =
                    QJsonDocument::fromJson(jsonlFile.readLine()).object();
            const QByteArray bytes = matrixFile.readAll();
            const auto* logits = reinterpret_cast<const float*>(bytes.data());
            const int classes = int(bytes.size() / sizeof(float));
            EXPECT_TRUE(classes > 0 && classes == tags.size())
                    << QString("UI and upstream expose the same class count")
                               .toStdString();
            double maxError = classes == tags.size() ? 0 : INFINITY;
            for (int i = 0; i < tags.size() && classes == tags.size(); ++i) {
                const auto tag = tags[i].toObject();
                maxError = std::max(maxError,
                                    std::abs(tag["logit"].toDouble() -
                                             logits[tag["index"].toInt()]));
            }
            QJsonArray upstreamSelected;
            for (auto value : reference["tags"].toArray())
                upstreamSelected.append(value.toObject()["name"]);
            EXPECT_TRUE(std::isfinite(maxError))
                    << QString("UI upstream logit difference is finite")
                               .toStdString();
            if (logitParityRequired)
                EXPECT_TRUE(maxError <= logitTolerance)
                        << QString("UI CPU logits match upstream compatibility "
                                   "options")
                                   .toStdString();
            EXPECT_TRUE(row["selected_tags"].toArray() == upstreamSelected)
                    << QString("UI selected tags match upstream options")
                               .toStdString();
            if (c.caption)
                EXPECT_TRUE(!row["caption"].toString().isEmpty() &&
                            row["caption"].toString() ==
                                    reference["caption"].toString())
                        << QString("UI caption matches upstream options")
                                   .toStdString();
            exported[c.name] = tags;
            evidence.append(QJsonObject{
                    {"case", c.name},
                    {"model", row["model"]},
                    {"device", row["device"]},
                    {"cpu_precision", row["cpu_precision"]},
                    {"upstream_args", QJsonArray::fromStringList(c.upstream)},
                    {"classes", tags.size()},
                    {"upstream_logit_max_error", maxError},
                    {"upstream_logit_parity_required", logitParityRequired},
                    {"selected_tags", row["selected_tags"]},
                    {"upstream_selected_tags", upstreamSelected},
                    {"caption", row["caption"]},
                    {"upstream_caption", reference["caption"]}});
            std::cout << "RAM option " << qPrintable(c.name)
                      << ": classes=" << tags.size()
                      << " selected=" << upstreamSelected.size()
                      << " logit_max_error=" << maxError << '\n';
        }
        // Exported effective thresholds prove each option reached selection.
        auto lines = [](const QString& path) {
            QFile file(path);
            file.open(QIODevice::ReadOnly);
            return QString::fromUtf8(file.readAll())
                    .split('\n', QtCompat::SkipEmptyParts);
        };
        const auto names = lines(space + ".tags.txt"),
                   values = lines(space + ".thresholds.txt");
        const auto custom = exported.value("custom-label-space");
        bool customExact =
                custom.size() == names.size() && names.size() == values.size();
        for (int i = 0; customExact && i < custom.size(); ++i) {
            const auto tag = custom[i].toObject();
            customExact = tag["name"].toString() == names[i].trimmed() &&
                          std::abs(tag["threshold"].toDouble() -
                                   values[i].toFloat()) <= 1e-6;
        }
        EXPECT_TRUE(customExact)
                << QString("UI custom names and thresholds come from files")
                           .toStdString();
        const auto global = exported.value("global-threshold");
        EXPECT_TRUE(!global.isEmpty() &&
                    std::all_of(global.begin(), global.end(),
                                [](const QJsonValue& value) {
                                    return value.toObject()["threshold"]
                                                   .toDouble() == double(.6f);
                                }))
                << QString("UI global threshold applies to every class")
                           .toStdString();
        const auto base = exported.value("tag2text-batch-beams"),
                   margin = exported.value("tag2text-specified-sequential");
        bool marginExact = !base.isEmpty() && base.size() == margin.size();
        for (int i = 0; marginExact && i < base.size(); ++i)
            marginExact =
                    std::abs(
                            margin[i].toObject()["threshold"].toDouble() -
                            (float(base[i].toObject()["threshold"].toDouble()) -
                             .05f)) <= 1e-6;
        EXPECT_TRUE(marginExact)
                << QString("UI Tag2Text margin lowers every threshold")
                           .toStdString();
        QFile report(out.filePath("ram-ui-options.json"));
        report.open(QIODevice::WriteOnly | QIODevice::Truncate);
        report.write(
                QJsonDocument(
                        QJsonObject{
                                {"image", demo},
                                {"image_sha256", sha256File(demo)},
                                {"backend", backend},
                                {"upstream_logit_tolerance", logitTolerance},
                                {"upstream_logit_parity_required",
                                 logitParityRequired},
                                {"upstream_binary",
                                 upstream.absoluteFilePath()},
                                {"upstream_binary_sha256", upstreamDigest},
                                {"candidate_aicore_sha256",
                                 sha256File(candidateAICoreLibrary())},
                                {"candidate_plugin_sha256",
                                 sha256File(parser.value("plugin-module"))},
                                {"label_space", space},
                                {"tag_list_sha256",
                                 sha256File(space + ".tags.txt")},
                                {"thresholds_sha256",
                                 sha256File(space + ".thresholds.txt")},
                                {"embedding_sha256",
                                 sha256File(space + ".f32")},
                                {"cases", evidence}})
                        .toJson());
        dialog.close();
        dialog.shutdown();
    }
}

TEST_F(RAMWorkflow, WorkerCompletesInvalidInputAndCancellationExactlyOnce) {
    {
        QThread thread;
        auto* worker = new RAMWorker;
        worker->moveToThread(&thread);
        QObject::connect(&thread, &QThread::finished, worker,
                         &QObject::deleteLater);
        thread.start();
        int completions = 0;
        QEventLoop loop;
        QObject::connect(
                worker, &RAMWorker::completed, &loop,
                [&](const RAMResult& result) {
                    ++completions;
                    EXPECT_TRUE(!result.error.isEmpty())
                            << QString("invalid input reports error")
                                       .toStdString();
                    EXPECT_TRUE(result.image.isNull())
                            << QString("invalid input publishes no image")
                                       .toStdString();
                    loop.quit();
                });
        RAMSettings settings;
        settings.model = QDir::tempPath() + "/ram-model-does-not-exist.gguf";
        settings.device = "cpu";
        QMetaObject::invokeMethod(
                worker,
                [=]() {
                    worker->process(settings, QImage(), "invalid-image", 7);
                },
                Qt::QueuedConnection);
        QTimer::singleShot(10000, &loop, &QEventLoop::quit);
        loop.exec();
        EXPECT_TRUE(completions == 1)
                << QString("one completion on load failure").toStdString();
        worker->requestCancel();
        QMetaObject::invokeMethod(
                worker,
                [=]() { worker->process(settings, QImage(), "cancelled", 8); },
                Qt::QueuedConnection);
        QTimer::singleShot(10000, &loop, &QEventLoop::quit);
        loop.exec();
        EXPECT_TRUE(completions == 2)
                << QString("one completion on cancellation").toStdString();
        QMetaObject::invokeMethod(
                worker, [=]() { worker->release(); },
                Qt::BlockingQueuedConnection);
        thread.quit();
        thread.wait();
    }
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    QApplication app(argc, argv);
    workflowArguments = app.arguments();
    return RUN_ALL_TESTS();
}
