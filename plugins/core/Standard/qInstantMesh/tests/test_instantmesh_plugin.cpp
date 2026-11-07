// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QMainWindow>
#include <QPainter>
#include <QPointer>
#include <QSettings>
#include <QTemporaryDir>
#include <cstdio>
#include <future>
#include <limits>
#include <thread>

#include "InstantMeshBatchCommand.h"
#include "aicore/runtime_capi.h"
#include "ecvMainAppInterface.h"
#include "ecvMaterialSet.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "qInstantMesh.h"

class TestHost : public ecvMainAppInterface {
public:
    QMainWindow window;
    ccHObject root;
    ccHObject::Container selection;
    QMainWindow* getMainWindow() override { return &window; }
    void registerOverlayDialog(ccOverlayDialog*, Qt::Corner) override {}
    void unregisterOverlayDialog(ccOverlayDialog*) override {}
    void updateOverlayDialogsPlacement() override {}
    ccUniqueIDGenerator::Shared getUniqueIDGenerator() override {
        return ccObject::GetUniqueIDGenerator();
    }
    ccHObject* loadFile(QString, bool) override { return nullptr; }
    void addToDB(ccHObject* obj, bool, bool, bool, bool) override {
        root.addChild(obj);
    }
    void removeFromDB(ccHObject* obj, bool) override { root.removeChild(obj); }
    ccHObjectContext removeObjectTemporarilyFromDBTree(ccHObject*) override {
        return {};
    }
    void putObjectBackIntoDBTree(ccHObject*, const ccHObjectContext&) override {
    }
    void setSelectedInDB(ccHObject*, bool) override {}
    const ccHObject::Container& getSelectedEntities() const override {
        return selection;
    }
    void dispToConsole(QString, ConsoleMessageLevel) override {}
    void forceConsoleDisplay() override {}
    ccHObject* dbRootObject() override { return &root; }
    void refreshAll(bool, bool) override {}
    void enableAll() override {}
    void disableAll() override {}
    void refreshSelected(bool, bool) override {}
    void refreshObject(ccHObject*, bool, bool) override {}
    void refreshObjects(ccHObject::Container, bool, bool) override {}
    void resetSelectedBBox() override {}
    void updateUI() override {}
    void freezeUI(bool) override {}
    ccColorScalesManager* getColorScalesManager() override { return nullptr; }
    void spawnHistogramDialog(const std::vector<unsigned>&,
                              double,
                              double,
                              QString,
                              QString) override {}
    void setView(CC_VIEW_ORIENTATION) override {}
    void toggleActiveWindowCenteredPerspective() override {}
    void toggleActiveWindowViewerBasedPerspective() override {}
    void zoomOnSelectedEntities() override {}
    void zoomOnEntities(ccHObject*) override {}
    void setGlobalZoom() override {}
    void increasePointSize() override {}
    void decreasePointSize() override {}
    void addViewWidget(QWidget*) override {}
    QWidget* getActiveWindow() override { return nullptr; }
    void toggleExclusiveFullScreen(bool) override {}
    void toggle3DView(bool) override {}
};

// Replace only the CLI host and logs; use production parsing, schema, worker,
// download and export.
class TestCommandLine : public ccCommandLineInterface {
public:
    QStringList args;
    mutable QString failure;
    QStringList& arguments() override { return args; }
    const QStringList& arguments() const override { return args; }
    bool registerCommand(Command::Shared) override { return true; }
    QString getExportFilename(const CLEntityDesc&,
                              QString,
                              QString,
                              QString*,
                              bool) const override {
        return {};
    }
    QString exportEntity(CLEntityDesc&,
                         const QString&,
                         QString*,
                         ExportOptions) override {
        return {};
    }
    bool saveClouds(QString, bool, const QString*) override { return false; }
    bool saveMeshes(QString, bool, const QString*) override { return false; }
    void removeClouds(bool) override {}
    void removeMeshes(bool) override {}
    bool importFile(QString,
                    const GlobalShiftOptions&,
                    FileIOFilter::Shared) override {
        return false;
    }
    QString cloudExportFormat() const override { return {}; }
    QString cloudExportExt() const override { return {}; }
    QString meshExportFormat() const override { return {}; }
    QString meshExportExt() const override { return {}; }
    QString hierarchyExportFormat() const override { return {}; }
    QString hierarchyExportExt() const override { return {}; }
    void setCloudExportFormat(QString, QString) override {}
    void setMeshExportFormat(QString, QString) override {}
    void setHierarchyExportFormat(QString, QString) override {}
    void print(const QString&) const override {}
    void warning(const QString&) const override {}
    bool error(const QString& message) const override {
        failure = message;
        return false;
    }
};

class TestPlugin : public qInstantMesh {
public:
    using qInstantMesh::onCancelRequested;
    using qInstantMesh::onNewAction;
    using qInstantMesh::onResultReady;
    using qInstantMesh::onRunRequested;
    using qInstantMesh::onTaskFinished;
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTemporaryDir tmp;
    qputenv("CLOUDVIEWER_DATA_ROOT", tmp.path().toUtf8());
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       tmp.filePath("settings"));
    TestHost host;
    TestPlugin plugin;
    plugin.setMainAppInterface(&host);
    plugin.onNewAction();
    auto* dialog = host.window.findChild<InstantMeshDialog*>();
    if (!dialog) return 1;
    int failures = 0;
    const auto check = [&](bool ok, const char* reason) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", reason);
            ++failures;
        }
    };
    const QIcon icon(
            QStringLiteral(":/CC/plugin/qInstantMesh/images/qInstantMesh.svg"));
    QImage iconSheet(480, 320, QImage::Format_ARGB32);
    iconSheet.fill(Qt::transparent);
    QPainter iconPainter(&iconSheet);
    int iconColumn = 0;
    for (int size : {32, 64, 128}) {
        const QImage rendered = icon.pixmap(size, size).toImage();
        check(!rendered.isNull() && rendered.size() == QSize(size, size),
              "SVG resources must render at 32/64/128");
        int brandPixels = 0, clippedBrandPixels = 0;
        for (int y = size * 7 / 10; y < rendered.height(); ++y)
            for (int x = 0; x < rendered.width(); ++x) {
                const QRgb pixel = rendered.pixel(x, y);
                // At 32px, antialiasing blends white text with purple; the
                // bright green channel still distinguishes text.
                if (qRed(pixel) > 180 && qGreen(pixel) > 140 &&
                    qBlue(pixel) > 180 && qAlpha(pixel)) {
                    ++brandPixels;
                    if (x < std::max(1, size / 32) ||
                        x >= size - std::max(1, size / 32))
                        ++clippedBrandPixels;
                }
            }
        check(brandPixels > 2,
              "Qt SVG must render bottom branding text, not blank font "
              "resources");
        check(clippedBrandPixels == 0,
              "Qt SVG branding must have margins without clipping first/last "
              "letters");
        for (int row = 0; row < 2; ++row) {
            iconPainter.fillRect(iconColumn * 160, row * 160, 160, 160,
                                 row ? QColor(32, 32, 32) : Qt::white);
            iconPainter.drawImage(iconColumn * 160 + (160 - size) / 2,
                                  row * 160 + (160 - size) / 2, rendered);
        }
        ++iconColumn;
    }
    iconPainter.end();
    const QString artifactDirectory =
            qEnvironmentVariable("INSTANTMESH_TEST_ARTIFACT_DIR");
    if (!artifactDirectory.isEmpty()) {
        QDir().mkpath(artifactDirectory);
        check(iconSheet.save(
                      QDir(artifactDirectory).filePath("icon-sizes.png")),
              "The icon validation sheet must be writable");
    }
    CommandInstantMeshGenerate command;
    TestCommandLine cli;
    const QString schemaPath = tmp.filePath("schema.json");
    cli.args = QStringList{"HELP", "RESULT_JSON", schemaPath};
    check(command.process(cli), "Native HELP RESULT_JSON must succeed");
    QFile schemaFile(schemaPath);
    check(schemaFile.open(QIODevice::ReadOnly),
          "The native schema file must exist");
    const auto schema = QJsonDocument::fromJson(schemaFile.readAll()).object();
    check(schema["schema"].toString() == "acloudviewer.aicore.batch/1" &&
                  schema["task"].toString() == "instantmesh" &&
                  !schema["options"].toArray().isEmpty(),
          "Native schema must be nonempty and use the public task identity");
    for (const auto& item :
         std::vector<std::pair<QString, QString>>{{"SEED", "-1"},
                                                  {"STEPS", "1"},
                                                  {"GRID_RES", "257"},
                                                  {"TEXTURE_RES", "4097"},
                                                  {"GUIDANCE", "16"}}) {
        cli.failure.clear();
        cli.args = QStringList{"OUT_DIR", tmp.filePath("batch"), item.first,
                               item.second};
        check(!command.process(cli) &&
                      cli.failure.contains(item.first, Qt::CaseInsensitive),
              "Reject invalid batch numeric values before model "
              "checks/downloads");
    }
    // Dependency tests use geometry placeholders only; missing RMBG must fail
    // before model loading, never infer with fake weights.
    const auto geometryPaths = InstantMeshHelpers::resolvePresetFiles(
            InstantMeshHelpers::presets()[1],
            InstantMeshHelpers::modelCacheDir(), QStringLiteral("f16"));
    for (const QString& path : geometryPaths) {
        if (path.isEmpty()) continue;
        QDir().mkpath(QFileInfo(path).absolutePath());
        QFile placeholder(path);
        check(placeholder.open(QIODevice::WriteOnly),
              "Model-existence placeholder files must be creatable");
        placeholder.write("not-a-model");
    }
    cli.args = QStringList{
            "PRESET",   "geometry", "DEVICE",  "cpu",
            "USE_RMBG", "1",        "OUT_DIR", tmp.filePath("missing-rmbg")};
    cli.failure.clear();
    check(!command.process(cli) && cli.failure.contains("rmbg_f16.gguf"),
          "Batch RMBG must use the shared recommended model and "
          "missing-dependency checks");
    std::promise<int> acquired;
    std::promise<void> release;
    auto release_future = release.get_future();
    std::thread holder([&] {
        int rc = aicore_device_task_lock("cpu");
        acquired.set_value(rc);
        release_future.wait();
        if (rc == 0) aicore_device_task_unlock();
    });
    check(acquired.get_future().get() == 0,
          "The test did not acquire the device queue");
    InstantMeshDialog::Settings settings;
    settings.device = QStringLiteral("cpu");
    settings.inputPath = tmp.filePath("input.png");
    settings.modelPaths =
            QStringList{QString(),
                        QString(),
                        QString(),
                        QStringLiteral("missing-dino.gguf"),
                        QStringLiteral("missing-transformer.gguf"),
                        QStringLiteral("missing-synthesizer.gguf")};
    plugin.onRunRequested(settings);
    QPointer<InstantMeshWorker> worker = plugin.findChild<InstantMeshWorker*>();
    if (!worker) {
        release.set_value();
        holder.join();
        return 1;
    }
    InstantMeshRunResult stale;
    stale.sourceImage = QStringLiteral("stale-input.png");
    stale.verts = QVector<float>{0, 0, 0, 1, 0, 0, 0, 1, 0};
    stale.tris = QVector<int>{0, 1, 2};
    stale.vcolors = QVector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1};
    plugin.onResultReady(stale);
    check(host.root.getChildrenNumber() == 0,
          "Stale-source results must not mutate DB");
    check(dialog->lastResult().sourceImage.isEmpty(),
          "Stale-source results must not mutate UI");
    stale.sourceImage = settings.inputPath;
    stale.generation = 0;
    plugin.onResultReady(stale);
    check(host.root.getChildrenNumber() == 0,
          "Stale generations with the same source must not mutate DB");
    stale.generation = 1;
    std::thread sender([&] { emit worker->resultReady(stale); });
    sender.join();
    QCoreApplication::processEvents();
    check(host.root.getChildrenNumber() == 1,
          "Worker-thread typed results must cross the Qt event queue into DB");
    check(dialog->lastResult().sourceImage == settings.inputPath,
          "Worker-thread typed results must reach the dialog before thread "
          "completion");
    // Detect discarded per-corner UVs, GLB dependencies, borrowed freed pixels
    // and overwritten material caches.
    InstantMeshRunResult textured = stale;
    textured.verts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    textured.tris = {0, 1, 2, 0, 2, 3};
    textured.vcolors = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    textured.uvs = {0, 0, 1, 0, 1, 1, 0.25f, 0.75f, 0.75f, 1, 0, 1};
    textured.texW = textured.texH = 2;
    const unsigned char pixels[] = {255, 0, 0,   255, 0,   255, 0,   128,
                                    0,   0, 255, 64,  255, 255, 255, 255};
    for (unsigned char pixel : pixels) textured.texRgba.push_back(pixel);
    textured.glb.clear();
    QString exportError;
    const QString objPath = tmp.filePath("mesh.obj");
    check(CommandInstantMeshGenerate::saveObj(textured, objPath, &exportError),
          "Typed textured OBJ export must succeed");
    QFile objFile(objPath);
    check(objFile.open(QIODevice::ReadOnly), "OBJ must be readable");
    const QByteArray obj = objFile.readAll();
    check(obj.contains("mtllib mesh.mtl") &&
                  obj.contains("usemtl InstantMesh_atlas"),
          "OBJ must reference relative MTL and atlas material");
    check(obj.contains("vt 0.2500000 0.7500000") &&
                  obj.contains("f 1/1 2/2 3/3") &&
                  obj.contains("f 1/4 3/5 4/6"),
          "OBJ must output per-corner vt and independent corner indices");
    QFile mtlFile(tmp.filePath("mesh.mtl"));
    check(mtlFile.open(QIODevice::ReadOnly) &&
                  mtlFile.readAll().contains("map_Kd mesh_texture.png"),
          "MTL must reference textures in the same directory");
    const QImage exportedTexture(tmp.filePath("mesh_texture.png"));
    check(exportedTexture.size() == QSize(2, 2) &&
                  exportedTexture.pixel(1, 0) == qRgba(0, 255, 0, 128),
          "OBJ atlas must retain top-first RGBA");
    InstantMeshRunResult malformed = textured;
    malformed.tris[0] = -1;
    check(!CommandInstantMeshGenerate::saveObj(
                  malformed, tmp.filePath("invalid.obj"), &exportError) &&
                  !QFile::exists(tmp.filePath("invalid.obj")),
          "Invalid triangles must not produce a successful OBJ file");
    plugin.onResultReady(textured);
    check(host.root.getChildrenNumber() == 2,
          "Typed textured meshes must enter DB without GLB");
    auto* mesh = host.root.getChildrenNumber() == 2
                         ? dynamic_cast<ccMesh*>(host.root.getChild(1))
                         : nullptr;
    check(mesh && mesh->hasTextures() && mesh->materialsShown(),
          "Typed textures must create and enable real materials");
    if (mesh) {
        check(mesh->size() == 2 && mesh->getAssociatedCloud()->size() == 4,
              "DB meshes must retain shared vertices and triangle counts");
        const auto* tri = mesh->getTriangleVertIndexes(1);
        check(tri && tri->i1 == 0 && tri->i2 == 2 && tri->i3 == 3,
              "Shared vertex indices must not be rewritten");
        check(mesh->getTexCoordinatesTable() &&
                      mesh->getTexCoordinatesTable()->size() == 6,
              "Store UVs per triangle corner; shared-vertex storage must not "
              "overwrite seams");
        if (mesh->hasPerTriangleTexCoordIndexes()) {
            int a, b, c;
            mesh->getTriangleTexCoordinatesIndexes(1, a, b, c);
            check(a == 3 && b == 4 && c == 5,
                  "The second triangle must use independent corner UV indices");
            TexCoords2D *u0, *u1, *u2;
            mesh->getTriangleTexCoordinates(1, u0, u1, u2);
            check(u0 && u0->tx == 0.25f && u0->ty == 0.75f,
                  "Retain seam UVs; v=0 is the bottom");
        }
        const auto* materials = mesh->getMaterialSet();
        if (materials && materials->size() == 1) {
            check(mesh->getTriangleMtlIndex(0) == 0 &&
                          mesh->getTriangleMtlIndex(1) == 0,
                  "Every triangle must reference the atlas material");
            const auto first = materials->at(0);
            check(first->hasTextureMap(ccMaterial::TextureMapType::DIFFUSE) &&
                          first->getTextureFilenames(
                                       ccMaterial::TextureMapType::DIFFUSE)
                                          .size() == 1,
                  "Typed atlases must populate the diffuse texture-map "
                  "contract consumed by VTK");
            const QImage stored = first->getTexture();
            check(stored.size() == QSize(2, 2) &&
                          stored.pixel(0, 0) == qRgba(255, 0, 0, 255) &&
                          stored.pixel(1, 0) == qRgba(0, 255, 0, 128) &&
                          stored.pixel(0, 1) == qRgba(0, 0, 255, 64),
                  "Atlas must retain top-first RGBA and alpha");
            textured.texRgba.fill('\0');
            plugin.onResultReady(textured);
            check(first->getTexture().pixel(0, 0) == qRgba(255, 0, 0, 255),
                  "New results must not overwrite prior materials; DB pixels "
                  "must not borrow worker buffers");
        }
    }
    const auto reject = [&](const InstantMeshRunResult& result,
                            const char* reason) {
        const unsigned before = host.root.getChildrenNumber();
        const double previousRuntime = dialog->lastResult().totalRuntimeMs;
        auto tagged = result;
        tagged.totalRuntimeMs = previousRuntime + 1;
        plugin.onResultReady(tagged);
        check(host.root.getChildrenNumber() == before, reason);
        check(dialog->lastResult().totalRuntimeMs == previousRuntime,
              "Invalid typed envelopes must not update UI, preview or "
              "exportable lastResult");
    };
    InstantMeshRunResult invalid = stale;
    invalid.tris[2] = 3;
    reject(invalid, "Out-of-bounds triangle indices must not enter DB");
    invalid = stale;
    invalid.verts[0] = std::numeric_limits<float>::quiet_NaN();
    reject(invalid, "Nonfinite coordinates must not enter DB");
    invalid = textured;
    invalid.uvs[0] = std::numeric_limits<float>::infinity();
    reject(invalid, "Nonfinite UVs must not enter DB");
    invalid = textured;
    invalid.texRgba.removeLast();
    reject(invalid,
           "Truncated atlases must not degrade into successful gray meshes");
    invalid.addResultToDb = false;
    reject(invalid,
           "Reject invalid UI results even when Add to DB is unchecked");
    invalid = stale;
    invalid.vcolors[0] = std::numeric_limits<float>::quiet_NaN();
    reject(invalid, "Nonfinite vertex colors must not enter UI or DB");
    invalid = stale;
    invalid.vcolors.removeLast();
    reject(invalid,
           "Truncated colors must not degrade into successful uncolored "
           "meshes");
    invalid = textured;
    invalid.uvs[0] = 1.1f;
    reject(invalid, "Atlas UVs must stay within the public contract range");
    invalid = stale;
    invalid.texW = 1;
    reject(invalid,
           "Partial texture dimensions must not degrade into successful "
           "vertex-color results");
    InstantMeshRunResult fractional = stale;
    fractional.vcolors = {0.5f, 0.25f, 0.75f, 0, 1, 0, 0, 0, 1};
    plugin.onResultReady(fractional);
    auto* colored = dynamic_cast<ccMesh*>(
            host.root.getChild(host.root.getChildrenNumber() - 1));
    auto* points =
            colored ? dynamic_cast<ccPointCloud*>(colored->getAssociatedCloud())
                    : nullptr;
    check(points && points->getPointColor(0).r == 127 &&
                  points->getPointColor(0).g == 63 &&
                  points->getPointColor(0).b == 191,
          "Scale floating-point vertex colors before byte conversion");
    plugin.onTaskFinished(false);
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    check(worker != nullptr,
          "Logical completion must not destroy thread objects before "
          "QThread::finished");
    plugin.onCancelRequested();
    check(worker && worker->wait(1000),
          "Plugin cancellation must reach the worker's shared token");
    release.set_value();
    holder.join();
    if (worker) worker->wait();
    QCoreApplication::processEvents();
    if (argc > 1) {
        // Optional real batch gate; argv[1] is the data root (with extract/),
        // not a packaged-child gate.
        qputenv("CLOUDVIEWER_DATA_ROOT", QByteArray(argv[1]));
        const QString input = tmp.filePath("native.png");
        QImage image(32, 32, QImage::Format_RGBA8888);
        image.fill(Qt::white);
        check(image.save(input), "Real batch input must be creatable");
        const QString manifestPath = tmp.filePath("native.json");
        cli.args = QStringList{
                "PRESET",
                "geometry",
                "QUANT",
                "f16",
                "IMAGE",
                input,
                "OUT_DIR",
                tmp.filePath("native-out"),
                "RESULT_JSON",
                manifestPath,
                "DEVICE",
                argc > 2 ? QString::fromUtf8(argv[2]) : QStringLiteral("vulkan"),
                "USE_RMBG",
                "0",
                "SEED",
                "0",
                "GRID_RES",
                "8",
                "TEXTURE_RES",
                "32"};
        cli.failure.clear();
        const bool batchOk = command.process(cli);
        if (!batchOk)
            std::fprintf(stderr, "Real batch failed: %s\n",
                         qPrintable(cli.failure));
        check(batchOk, "Real geometry batch must succeed");
        QFile manifestFile(manifestPath);
        check(manifestFile.open(QIODevice::ReadOnly),
              "Real batch manifest must exist");
        const auto manifest =
                QJsonDocument::fromJson(manifestFile.readAll()).object();
        const auto records = manifest["items"].toArray();
        check(records.size() == 1 && manifest["ok"].toInt() == 1,
              "Real batch must record a successful item");
        if (records.size() == 1) {
            const auto record = records[0].toObject();
            const auto timing = record["timings"].toObject();
            check(timing["e2e_ms"].toDouble() > 0 &&
                          timing["e2e_ms"].toDouble() ==
                                  record["runtime_ms"].toDouble() &&
                          record.contains("load_ms") &&
                          record.contains("export_ms"),
                  "Batch must use common timing and separately report "
                  "load/export");
            check(record.contains("glb_bake_ms") &&
                          record["file_export_ms"].toDouble() > 0 &&
                          std::abs(record["export_ms"].toDouble() -
                                   record["glb_bake_ms"].toDouble() -
                                   record["file_export_ms"].toDouble()) < 1e-9,
                  "Export must include real GLB bake and OBJ/MTL/PNG/preview "
                  "writes, not worker bake alone");
            const auto outputs = record["outputs"].toObject();
            check(QFile::exists(outputs["mesh"].toString()) &&
                          QFile::exists(outputs["glb"].toString()),
                  "Real batch must produce OBJ and GLB");
            check(outputs.contains("material") && outputs.contains("texture") &&
                          QFile::exists(outputs["texture"].toString()),
                  "Batch manifest must register real textures and MTL");
        }
    }
    return failures ? 1 : 0;
}
