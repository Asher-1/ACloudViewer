// SPDX-License-Identifier: MIT
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
#include "ecvTestDataRepository.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setOrganizationName("ACloudViewerTests");
    app.setApplicationName("RAMWorkflow");
    QTemporaryDir settingsDir;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settingsDir.path());
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addOption(
            {"real",
             "Run cached real-model image, export and lifecycle checks."});
    parser.addOption({"download-sample",
                      "Exercise Use test data from an empty sample cache."});
    parser.addOption(
            {"caption", "Use Tag2Text Q8 and verify generated captions."});
    parser.addOption(
            {"ram", "Use RAM Q8 and verify CPU arithmetic switching."});
    parser.addOption({"backend", "Real-inference backend.", "backend", "cuda"});
    parser.addOption({"switch-backend",
                      "Switch the live model session to this backend.",
                      "backend"});
    parser.addOption({"video",
                      "Exercise consumer-driven video using this fixture.",
                      "path"});
    parser.addOption({"output-dir", "Keep exports and dialog screenshots here.",
                      "path"});
    parser.addOption({"timeout-ms",
                      "Per-operation wait limit, including downloads.",
                      "milliseconds", "120000"});
    parser.addOption({"plugin-module",
                      "Load and check the built plugin module.", "path"});
    parser.addOption({"upstream-binary",
                      "Compare UI options with this upstream ram-ggml build.",
                      "path"});
    parser.addOption({"label-space-dir",
                      "Upstream cpp_ggml/benchmarks/label_spaces directory.",
                      "path"});
    parser.addOption({"upstream-logit-tolerance",
                      "CPU all-class compatibility bound; GPU differences are "
                      "recorded as observational metrics.",
                      "value", "0.0001"});
    parser.process(app);
    bool timeoutValid = false;
    const int workflowTimeout = parser.value("timeout-ms").toInt(&timeoutValid);
    if (!timeoutValid || workflowTimeout <= 0) return 1;
    int failures = 0;
    auto check = [&](bool ok, const char* message) {
        if (!ok) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    const QString configuredRoot =
            qEnvironmentVariable("CLOUDVIEWER_DATA_ROOT");
    if (!configuredRoot.isEmpty()) {
        check(ecvTestDataRepository::dataRoot() == configuredRoot,
              "sample and model caches honor the same data root");
    }
    if (parser.isSet("plugin-module")) {
        QPluginLoader loader(
                QFileInfo(parser.value("plugin-module")).absoluteFilePath());
        auto* plugin = loader.instance();
        check(plugin != nullptr, qPrintable(loader.errorString()));
        check(loader.metaData().value("IID").toString() ==
                      "cvcorp.cloudviewer.plugin.qRAM",
              "plugin registration IID");
        if (plugin) {
            const auto actions = plugin->findChildren<QAction*>();
            check(actions.size() == 1 && actions[0]->text().contains("RAM++") &&
                          !actions[0]->icon().isNull(),
                  "loaded plugin provides its RAM++ action and icon");
            std::cout << "RAM plugin module loaded through QPluginLoader\n";
        }
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
        check(model && model->count() == 9, "all catalog models in selector");
        check(tabs && tabs->tabText(0) == "Images / batch" &&
                      tabs->tabText(1) == "Results" &&
                      tabs->tabText(2) == "Options" &&
                      tabs->tabText(3) == "Video / camera",
              "workflow tabs put video after image results and options");
        check(dialog.findChild<QPushButton*>("ramLoadSampleVideo") != nullptr,
              "video page exposes official archive sample button");
        check(dialog.findChild<QLabel*>("ramQuickStart") != nullptr,
              "dialog exposes quick-start model guidance");
        check(device && device->count() >= 2, "device selector");
        check(precision && precision->count() == 2 &&
                      precision->currentData().toInt() ==
                              AICORE_TAGGING_CPU_AUTO,
              "CPU arithmetic defaults to automatic precision");
        if (precision) {
            precision->setCurrentIndex(1);
            check(precision->currentData().toInt() ==
                          AICORE_TAGGING_CPU_UPSTREAM,
                  "explicit upstream CPU arithmetic remains selectable");
            precision->setCurrentIndex(0);
        }
        check(sample && run && list && table && tabs && caption,
              "complete workflow controls");
        check(dialog.findChildren<QPlainTextEdit*>().size() == 1,
              "dialog keeps only the result caption text pane, not a local log pane");
        if (!model || !device || !sample || !run || !list || !table || !tabs ||
            !caption)
            return 1;
        check(model->currentData().toInt() ==
                      aicore_tagging_model_default_index(),
              "default model follows catalog");
        const auto* defaultModel =
                aicore_tagging_model_at(model->currentData().toInt());
        check(defaultModel && defaultModel->supports_caption &&
                      caption->isEnabled() && caption->isChecked(),
              "default Tag2Text model enables caption for one-click sample use");
        model->setCurrentIndex(6);
        check(caption->isEnabled(), "Tag2Text enables caption pipeline");
        auto* epsilon =
                dialog.findChild<QDoubleSpinBox*>("ramThresholdEpsilon");
        check(epsilon && epsilon->isEnabled() && epsilon->value() == 0,
              "Tag2Text exposes upstream default threshold margin");
        model->setCurrentIndex(0);
        check(!caption->isEnabled() && !caption->isChecked(),
              "RAM++ disables unsupported caption");
        check(epsilon && !epsilon->isEnabled(),
              "RAM++ disables Tag2Text threshold margin");
        auto demo = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::RAM, "demo1.jpg");
        if (parser.isSet("download-sample")) {
            check(demo.isEmpty(), "cold sample download starts without demo1");
            if (!demo.isEmpty()) return 1;
            sample->click();
            check(waitFor([&]() {
                      return sample->isEnabled() && list->count() == 4;
                  }),
                  "shared sample download and extraction complete");
            demo = ecvTestDataRepository::findDatasetFile(
                    ecvTestDataRepository::Dataset::RAM, "demo1.jpg");
        }
        if (!demo.isEmpty()) {
            sample->click();
            check(list->count() == 4 && list->item(0)->text() == demo &&
                          list->currentRow() == 0,
                  "all official demo images are filled with demo1 selected");
            auto* sampleVideo =
                    dialog.findChild<QPushButton*>("ramLoadSampleVideo");
            auto* video = dialog.findChild<RAMVideoWidget*>();
            if (sampleVideo && video && sampleVideo->isEnabled()) {
                sampleVideo->click();
                check(tabs->currentIndex() == 3 &&
                              QFileInfo(video->videoFilePath()).isFile() &&
                              sha256File(video->videoFilePath()) ==
                                      "53c17ba6b53bde9b21660142a1685bfa8ce4a32d6ab9c0cc08f1c57ce944b202",
                      "verified official archive video is selected");
                tabs->setCurrentIndex(0);
            }
        }
        if (parser.isSet("real")) {
            check(!demo.isEmpty(),
                  "real workflow requires verified cached official sample");
            if (demo.isEmpty()) return 1;
            QTemporaryDir extracted;
            const QString zip = ecvTestDataRepository::zipPath(
                    ecvTestDataRepository::Dataset::RAM);
            check(ecvTestDataRepository::extractZip(zip, extracted.path()),
                  "shared ZIP extracts offline");
            const QString member =
                    extracted.filePath("ram_datasets/demo/demo1.jpg");
            QFile file(member);
            check(file.open(QIODevice::ReadOnly),
                  "official ZIP contains demo1");
            const QByteArray digest =
                    QCryptographicHash::hash(file.readAll(),
                                             QCryptographicHash::Sha256)
                            .toHex();
            check(digest ==
                          "1b2906f4058a69936df49cb6156ec4cd117a286b420e1eb14764"
                          "033bf8f3c05f",
                  "shared extraction preserves official bytes");
            file.close();
            check(file.open(QIODevice::WriteOnly | QIODevice::Truncate),
                  "isolated corruption fixture opens");
            file.write("corrupt");
            file.close();
            check(ecvTestDataRepository::extractZip(zip, extracted.path()) &&
                          !QImage(member).isNull(),
                  "cached ZIP repairs corrupt member offline");
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
            check(selectedDevice >= 0, "requested backend is available");
            if (selectedDevice < 0) return 1;
            device->setCurrentIndex(selectedDevice);
            dialog.show();
            app.processEvents();
            dialog.grab().save(QDir(outputDir).filePath("ram-inputs.png"));
            auto* second = new QListWidgetItem("db://sample,\"quoted\"", list);
            second->setData(Qt::UserRole, QImage(demo));
            list->selectAll();
            run->click();
            check(waitFor([&]() {
                      return run->isEnabled() && tabs->currentIndex() == 1 &&
                             table->rowCount() > 0;
                  }),
                  "real batch completes");
            const QString jsonPath =
                    QDir(outputDir).filePath("ram-results.json");
            check(dialog.saveResults(jsonPath), "JSON export succeeds");
            auto rows = readJson(jsonPath);
            check(rows.size() == 5, "batch export contains every input");
            if (rows.size() == 5) {
                check(rows[0].toObject()["tags"].toArray().size() == classCount,
                      "exports every model class");
                if (testCaption) {
                    const auto sentence =
                            rows[0].toObject()["caption"].toString();
                    check(!sentence.isEmpty() &&
                                  sentence == rows[4].toObject()["caption"]
                                                      .toString(),
                          "resident Tag2Text captions agree");
                }
                check(rows[0].toObject()["selected_tags"].toArray() ==
                              rows[4].toObject()["selected_tags"].toArray(),
                      "resident session repeated tags agree");
                for (auto value : rows[0].toObject()["tags"].toArray()) {
                    auto tag = value.toObject();
                    check(std::isfinite(tag["logit"].toDouble()) &&
                                  tag["probability"].toDouble() >= 0 &&
                                  tag["probability"].toDouble() <= 1,
                          "finite full-class output");
                }
            }
            check(dialog.saveResults(
                          QDir(outputDir).filePath("ram-results.csv")),
                  "CSV export succeeds");
            QFile csv(QDir(outputDir).filePath("ram-results.csv"));
            check(csv.open(QIODevice::ReadOnly) &&
                          csv.readAll().contains(
                                  "\"db://sample,\"\"quoted\"\"\""),
                  "CSV escapes commas and embedded quotes");
            check(dialog.saveResults(
                          QDir(outputDir).filePath("ram-results.txt")),
                  "text export succeeds");
            int published = 0;
            QObject::connect(
                    &dialog, &RAMDialog::addToDbRequested, &app,
                    [&](const RAMResult& result) {
                        check(!result.image.isNull() &&
                                      result.json["tags"].toArray().size() ==
                                              classCount,
                              "DB boundary retains typed result and decoded "
                              "image");
                        ++published;
                    });
            dialog.findChild<QPushButton*>("ramAddToDb")->click();
            check(published == 1, "one DB publication");
            auto* selected = dialog.findChild<QCheckBox*>("ramSelectedOnly");
            auto* filter = dialog.findChild<QLineEdit*>("ramFilter");
            selected->setChecked(false);
            check(table->rowCount() == classCount, "all-class results toggle");
            filter->setText("dog");
            check(table->rowCount() > 0 && table->rowCount() < classCount,
                  "tag search filters results");
            filter->clear();
            selected->setChecked(true);
            dialog.grab().save(QDir(outputDir).filePath("ram-results.png"));
            tabs->setCurrentIndex(0);
            run->click();
            dialog.close();
            check(waitFor([&]() { return run->isEnabled(); }),
                  "close during inference releases pending job");
            check(table->rowCount() == 0,
                  "cancelled generation publishes no stale result");
            dialog.show();
            sample->click();
            run->click();
            check(waitFor([&]() {
                      return run->isEnabled() && tabs->currentIndex() == 1 &&
                             table->rowCount() > 0;
                  }),
                  "reopen runs successfully");
            if (parser.isSet("switch-backend")) {
                int target = -1;
                for (int i = 0; i < device->count(); ++i)
                    if (device->itemData(i).toString().startsWith(
                                parser.value("switch-backend"),
                                Qt::CaseInsensitive)) {
                        target = i;
                        break;
                    }
                check(target >= 0, "switch target backend available");
                if (target >= 0) {
                    device->setCurrentIndex(target);
                    tabs->setCurrentIndex(0);
                    run->click();
                    check(waitFor([&]() {
                              return run->isEnabled() &&
                                     tabs->currentIndex() == 1 &&
                                     table->rowCount() > 0;
                          }),
                          "live device change recreates session");
                    const QString changed =
                            QDir(outputDir).filePath("ram-device-switch.json");
                    check(dialog.saveResults(changed),
                          "device switch result export");
                    const auto records = readJson(changed);
                    check(records.size() == 1 &&
                                  records[0]
                                          .toObject()["device"]
                                          .toString()
                                          .startsWith(parser.value(
                                                              "switch-backend"),
                                                      Qt::CaseInsensitive),
                          "result identifies switched device");
                }
            }
            if (parser.isSet("ram") && parser.value("backend") == "cpu" &&
                precision) {
                precision->setCurrentIndex(1);
                tabs->setCurrentIndex(0);
                run->click();
                check(waitFor([&]() {
                          return run->isEnabled() &&
                                 tabs->currentIndex() == 1 &&
                                 table->rowCount() > 0;
                      }),
                      "switching CPU arithmetic recreates the resident "
                      "session");
                const auto compatibilityPath =
                        QDir(outputDir).filePath("ram-upstream-results.json");
                check(dialog.saveResults(compatibilityPath),
                      "CPU compatibility export succeeds");
                const auto compatible = readJson(compatibilityPath);
                check(compatible.size() == 1 &&
                              compatible[0].toObject()["cpu_precision"]
                                              .toString() == "upstream",
                      "exports identify the effective CPU arithmetic option");
                precision->setCurrentIndex(0);
            }
            if (parser.isSet("video")) {
                auto* video = dialog.findChild<RAMVideoWidget*>();
                check(video && video->isAvailable(),
                      "video pipeline available");
                if (video && video->isAvailable()) {
                    int frames = 0;
                    QObject::connect(video, &RAMVideoWidget::frameReady, &app,
                                     [&](const QImage&, int) { ++frames; });
                    video->selectVideoFileSource();
                    video->setVideoFilePath(parser.value("video"));
                    tabs->setCurrentIndex(3);
                    run->click();
                    check(waitFor([&]() {
                              return frames >= 3 && run->isEnabled();
                          }),
                          "consumer-driven video processes three frames");
                    dialog.findChild<QPushButton*>("ramCancel")->click();
                    check(waitFor([&]() { return run->isEnabled(); }),
                          "video stop releases pending inference");
                    check(dialog.saveResults(
                                  QDir(outputDir).filePath("ram-video.json")),
                          "video results export");
                    check(readJson(QDir(outputDir).filePath("ram-video.json"))
                                          .size() >= 3,
                          "every acknowledged frame has a result");
                    const int previous = frames;
                    video->restartVideoFile();
                    check(waitFor([&]() {
                              return frames >= previous + 2 && run->isEnabled();
                          }),
                          "video restart recovers consumer gate");
                    dialog.findChild<QPushButton*>("ramCancel")->click();
                    check(waitFor([&]() { return run->isEnabled(); }),
                          "restart test begins with an idle worker");
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
                    check(waitFor([&]() {
                              return restartedFrames >= 4 && run->isEnabled();
                          }),
                          "restart during inference completes replacement "
                          "frames");
                    dialog.findChild<QPushButton*>("ramCancel")->click();
                    check(waitFor([&]() { return run->isEnabled(); }),
                          "busy restart releases the worker");
                    QObject::disconnect(restartConnection);
                    const QString busyRestartFile = QDir(outputDir).filePath(
                            "ram-video-restart-busy.json");
                    check(dialog.saveResults(busyRestartFile),
                          "busy restart results export");
                    const auto restartedRows = readJson(busyRestartFile);
                    check(restartedRows.size() >= 2 &&
                                  restartedFrames <= restartedRows.size() + 2,
                          "restart holds the consumer gate instead of dropping "
                          "frames");
                    if (!restartedRows.isEmpty())
                        check(restartedRows.first()
                                      .toObject()["source"]
                                      .toString()
                                      .endsWith("#frame=1"),
                              "replacement stream starts with its first frame");
                    if (parser.value("backend") == "cpu")
                        check(restartedWhileBusy,
                              "CPU fixture actually restarted an in-flight "
                              "inference");
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
        check(ready,
              "option comparison requires upstream binary, label spaces and "
              "cached demo1");
        if (!ready) return 1;
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
        check(controls, "every option control is addressable");
        if (!controls) return 1;
        const QString backend = parser.value("backend");
        const bool logitParityRequired =
                backend.compare("cpu", Qt::CaseInsensitive) == 0;
        bool toleranceValid = false;
        const double logitTolerance =
                parser.value("upstream-logit-tolerance").toDouble(&toleranceValid);
        check(toleranceValid && logitTolerance >= 0,
              "upstream logit tolerance is a non-negative number");
        if (!toleranceValid || logitTolerance < 0) return 1;
        int selectedDevice = -1;
        for (int i = 0; i < device->count(); ++i)
            if (device->itemData(i).toString().startsWith(backend,
                                                          Qt::CaseInsensitive))
                selectedDevice = i;
        check(selectedDevice >= 0, "option backend is available");
        if (selectedDevice < 0) return 1;
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
        QStringList libraries{ggml, ggml + "/ggml-" + backend, build + "/bin"};
        if (env.contains("LD_LIBRARY_PATH"))
            libraries << env.value("LD_LIBRARY_PATH");
        env.insert("LD_LIBRARY_PATH", libraries.join(':'));
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
                {"custom-label-space", 2, false, true, -1, 0, 0, 30, {},
                 {"--tag-list", space + ".tags.txt", "--thresholds",
                  space + ".thresholds.txt", "--label-embedding",
                  space + ".f32"}},
                {"global-threshold", 2, false, false, .6, 0, 0, 30, {},
                 {"--threshold", "0.6"}},
                {"tag2text-batch-beams", 8, true, false, -1, 0, 3, 30, {},
                 {"--task", "caption", "--beam-batch", "3", "--max-length",
                  "30"}},
                {"tag2text-specified-sequential", 8, true, false, -1, .05, 1,
                 20, "dog,grass",
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
            check(waitFor([&]() {
                      return run->isEnabled() && tabs->currentIndex() == 1;
                  }),
                  "option case completes in the UI");
            const QString exportPath =
                    out.filePath("ram-options-" + c.name + ".json");
            check(dialog.saveResults(exportPath), "option case export");
            const auto rows = readJson(exportPath);
            check(rows.size() == 1, "option case exports one result");
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
                    "--backend", backend, "--threads", "8", "--warmup", "0",
                    "--repeat", "1", "--manifest", manifest.fileName(),
                    "--jsonl", jsonl, "--logits-matrix", matrix};
            args << c.upstream;
            QProcess process;
            process.setProcessEnvironment(env);
            process.setProcessChannelMode(QProcess::MergedChannels);
            process.start(upstream.absoluteFilePath(), args);
            const bool finished = process.waitForFinished(workflowTimeout);
            QFile log(out.filePath("ram-options-" + c.name + "-upstream.log"));
            if (log.open(QIODevice::WriteOnly | QIODevice::Truncate))
                log.write(process.readAll());
            check(finished && process.exitStatus() == QProcess::NormalExit &&
                          process.exitCode() == 0,
                  "upstream option run succeeds");
            QFile jsonlFile(jsonl), matrixFile(matrix);
            jsonlFile.open(QIODevice::ReadOnly);
            matrixFile.open(QIODevice::ReadOnly);
            const auto reference =
                    QJsonDocument::fromJson(jsonlFile.readLine()).object();
            const QByteArray bytes = matrixFile.readAll();
            const auto* logits = reinterpret_cast<const float*>(bytes.data());
            const int classes = int(bytes.size() / sizeof(float));
            check(classes > 0 && classes == tags.size(),
                  "UI and upstream expose the same class count");
            double maxError = classes == tags.size() ? 0 : INFINITY;
            for (int i = 0; i < tags.size() && classes == tags.size(); ++i) {
                const auto tag = tags[i].toObject();
                maxError = std::max(
                        maxError, std::abs(tag["logit"].toDouble() -
                                           logits[tag["index"].toInt()]));
            }
            QJsonArray upstreamSelected;
            for (auto value : reference["tags"].toArray())
                upstreamSelected.append(value.toObject()["name"]);
            check(std::isfinite(maxError),
                  "UI upstream logit difference is finite");
            if (logitParityRequired)
                check(maxError <= logitTolerance,
                      "UI CPU logits match upstream compatibility options");
            check(row["selected_tags"].toArray() == upstreamSelected,
                  "UI selected tags match upstream options");
            if (c.caption)
                check(!row["caption"].toString().isEmpty() &&
                              row["caption"].toString() ==
                                      reference["caption"].toString(),
                      "UI caption matches upstream options");
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
                    .split('\n', Qt::SkipEmptyParts);
        };
        const auto names = lines(space + ".tags.txt"),
                   values = lines(space + ".thresholds.txt");
        const auto custom = exported.value("custom-label-space");
        bool customExact = custom.size() == names.size() &&
                           names.size() == values.size();
        for (int i = 0; customExact && i < custom.size(); ++i) {
            const auto tag = custom[i].toObject();
            customExact = tag["name"].toString() == names[i].trimmed() &&
                          std::abs(tag["threshold"].toDouble() -
                                   values[i].toFloat()) <= 1e-6;
        }
        check(customExact, "UI custom names and thresholds come from files");
        const auto global = exported.value("global-threshold");
        check(!global.isEmpty() &&
                      std::all_of(global.begin(), global.end(),
                                  [](const QJsonValue& value) {
                                      return value.toObject()["threshold"]
                                                     .toDouble() == double(.6f);
                                  }),
              "UI global threshold applies to every class");
        const auto base = exported.value("tag2text-batch-beams"),
                   margin = exported.value("tag2text-specified-sequential");
        bool marginExact = !base.isEmpty() && base.size() == margin.size();
        for (int i = 0; marginExact && i < base.size(); ++i)
            marginExact =
                    std::abs(margin[i].toObject()["threshold"].toDouble() -
                             (float(base[i].toObject()["threshold"].toDouble()) -
                              .05f)) <= 1e-6;
        check(marginExact, "UI Tag2Text margin lowers every threshold");
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
                                {"upstream_binary", upstream.absoluteFilePath()},
                                {"upstream_binary_sha256", upstreamDigest},
                                {"candidate_aicore_sha256",
                                 sha256File(QDir(QCoreApplication::applicationDirPath())
                                                    .filePath("../libAICore.so"))},
                                {"candidate_plugin_sha256",
                                 sha256File(parser.value("plugin-module"))},
                                {"label_space", space},
                                {"tag_list_sha256",
                                 sha256File(space + ".tags.txt")},
                                {"thresholds_sha256",
                                 sha256File(space + ".thresholds.txt")},
                                {"embedding_sha256", sha256File(space + ".f32")},
                                {"cases", evidence}})
                        .toJson());
        dialog.close();
        dialog.shutdown();
    }
    {
        QThread thread;
        auto* worker = new RAMWorker;
        worker->moveToThread(&thread);
        QObject::connect(&thread, &QThread::finished, worker,
                         &QObject::deleteLater);
        thread.start();
        int completions = 0;
        QEventLoop loop;
        QObject::connect(worker, &RAMWorker::completed, &loop,
                         [&](const RAMResult& result) {
                             ++completions;
                             check(!result.error.isEmpty(),
                                   "invalid input reports error");
                             check(result.image.isNull(),
                                   "invalid input publishes no image");
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
        check(completions == 1, "one completion on load failure");
        worker->requestCancel();
        QMetaObject::invokeMethod(
                worker,
                [=]() { worker->process(settings, QImage(), "cancelled", 8); },
                Qt::QueuedConnection);
        QTimer::singleShot(10000, &loop, &QEventLoop::quit);
        loop.exec();
        check(completions == 2, "one completion on cancellation");
        QMetaObject::invokeMethod(
                worker, [=]() { worker->release(); },
                Qt::BlockingQueuedConnection);
        thread.quit();
        thread.wait();
    }
    std::cout << "RAM workflow: " << (failures ? "FAIL" : "PASS") << '\n';
    return failures ? 1 : 0;
}
