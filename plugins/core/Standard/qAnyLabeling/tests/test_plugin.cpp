// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <aicore/ocr_capi.h>
#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QApplication>
#include <QBuffer>
#include <QCheckBox>
#include <QClipboard>
#include <QColorDialog>
#include <QComboBox>
#include <QContextMenuEvent>
#include <QCryptographicHash>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QIcon>
#include <QImage>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QKeySequence>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QPluginLoader>
#include <QPushButton>
#include <QScrollArea>
#include <QSet>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QTranslator>
#include <QUrl>
#include <QtEndian>
#include <cstdio>
#include <cstdlib>

namespace {
bool reviewStatusRoundTrip(QMainWindow* window) {
    const auto action = [window](const char* id) -> QAction* {
        for (auto* item : window->findChildren<QAction*>())
            if (item->property("qAnyShortcutId").toString() ==
                QLatin1String(id))
                return item;
        return nullptr;
    };
    QTemporaryDir directory;
    if (!directory.isValid()) return false;
    const auto imagePath = directory.filePath("a.png");
    const auto sidecarPath = directory.filePath("a.json");
    QImage image(64, 48, QImage::Format_RGB32);
    image.fill(Qt::blue);
    for (const auto& name : {"a.png", "b.png", "c.png", "d.png"})
        if (!image.save(directory.filePath(name))) return false;
    const auto write = [](const QString& path, const QJsonValue& status) {
        QFile file(path);
        const auto bytes = QJsonDocument(QJsonObject{{"checked", status},
                                                     {"shapes", QJsonArray()},
                                                     {"custom", "preserve"}})
                                   .toJson();
        return file.open(QIODevice::WriteOnly) &&
               file.write(bytes) == bytes.size();
    };
    const auto read = [](const QString& path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return QJsonObject();
        return QJsonDocument::fromJson(file.readAll()).object();
    };
    if (!write(sidecarPath, "failed") ||
        !write(directory.filePath("b.json"), true) ||
        !write(directory.filePath("c.json"), "failed"))
        return false;
    if (!QMetaObject::invokeMethod(window, "openImage",
                                   Q_ARG(QString, imagePath)))
        return false;
    auto* save = action("save");
    if (!save || !save->isEnabled()) return false;
    save->trigger();
    if (read(sidecarPath).value("checked") != QJsonValue("failed")) {
        std::fprintf(stderr,
                     "Review regression: save changed failed into a boolean\n");
        return false;
    }
    auto* passed = action("toggle_annotation_checked");
    auto* failed = action("fail_annotation_checked");
    auto* cleared = action("clear_annotation_checked");
    auto* next = action("open_next_unchecked");
    auto* files = window->findChild<QListWidget*>("AnnotationFiles");
    if (!passed || !failed || !cleared || !next || !files) return false;
    const QList<QPair<QAction*, QJsonValue>> transitions = {
            {passed, true}, {failed, "failed"}, {cleared, false}};
    for (const auto& transition : transitions) {
        transition.first->trigger();
        const auto json = read(sidecarPath);
        if (json.value("checked") != transition.second ||
            json.value("custom") != QJsonValue("preserve") ||
            window->windowTitle().endsWith(" *")) {
            std::fprintf(stderr,
                         "Review regression: status action did not save "
                         "atomically\n");
            return false;
        }
        QMetaObject::invokeMethod(window, "openImage",
                                  Q_ARG(QString, imagePath));
        if (!transition.first->isChecked() ||
            passed->isChecked() + failed->isChecked() + cleared->isChecked() !=
                    1)
            return false;
        auto* item = files->currentItem();
        if (!item || item->checkState() != Qt::Checked) return false;
    }
    passed->trigger();
    next->trigger();
    if (!files->currentItem() || files->currentItem()->text() != "d.png") {
        std::fprintf(stderr,
                     "Review regression: unchecked navigation included a "
                     "reviewed file\n");
        return false;
    }
    if (files->item(0)->checkState() != Qt::Checked ||
        files->item(3)->checkState() != Qt::Unchecked)
        return false;
    std::puts(
            "Review failed/pass/clear save-reopen and unchecked navigation: "
            "PASS");
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTimer::singleShot(30000, &app, []() {
        if (auto* modal = QApplication::activeModalWidget()) {
            std::fprintf(stderr, "Timed out in dialog: %s (%s)\n",
                         qPrintable(modal->windowTitle()),
                         qPrintable(modal->objectName()));
            for (auto* label : modal->findChildren<QLabel*>())
                std::fprintf(stderr, "Dialog text: %s\n",
                             qPrintable(label->text()));
        }
        std::exit(1);
    });
    QTemporaryDir settingsDir;
    if (!settingsDir.isValid()) return 1;
    QCoreApplication::setOrganizationName(QStringLiteral("ACloudViewerTests"));
    QCoreApplication::setApplicationName(QStringLiteral("qAnyLabelingPlugin"));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settingsDir.path());
    const bool reviewOnly = argc == 3 && QByteArray(argv[2]) == "--review-only";
    if (argc != 2 && !reviewOnly) return 2;
    // QPluginLoader's metadata pass rejects a non-normalized relative path
    // with "The shared library was not found"; resolve to an absolute path
    // so the harness works from any working directory.
    // Exercise the shipped binary, from an unrelated Unicode path and cwd.
    // This target does not link qAnyLabeling.qrc: resource success therefore
    // proves that the actual plugin contains its deployment data.
    QTemporaryDir relocated;
    if (!relocated.isValid()) return 1;
    const QString relocatedDir =
            relocated.path() + QString::fromUtf8("/安装插件");
    if (!QDir().mkpath(relocatedDir)) return 1;
    const QString pluginPath =
            relocatedDir + '/' +
            QFileInfo(QString::fromLocal8Bit(argv[1])).fileName();
    if (!QFile::copy(
                QFileInfo(QString::fromLocal8Bit(argv[1])).absoluteFilePath(),
                pluginPath) ||
        !QDir::setCurrent(relocatedDir))
        return 1;
    QPluginLoader loader(pluginPath);
    QObject* pluginObject = loader.instance();
    if (!pluginObject) {
        std::fprintf(stderr, "Plugin load failed: %s\n",
                     loader.errorString().toLocal8Bit().constData());
        return 1;
    }
    for (const auto& entry :
         {qMakePair(QStringLiteral("ram_tag_list.txt"),
                    QByteArray("5d96d2f7225527480ad773aaac76b6e5c08a18adf6eae9c"
                               "a7d2ab923d5e7bd8b")),
          qMakePair(QStringLiteral("ram_tag_list_chinese.txt"),
                    QByteArray("c912657a0f30be7b2f5df6ccffa17e1d84561d0bea50852"
                               "8e38b6d0d711bcfbe"))}) {
        QFile resource(
                QStringLiteral(":/CC/plugin/qAnyLabeling/resources/ram/") +
                entry.first);
        if (!resource.open(QIODevice::ReadOnly) ||
            QCryptographicHash::hash(resource.readAll(),
                                     QCryptographicHash::Sha256)
                            .toHex() != entry.second) {
            std::fprintf(stderr, "Relocated plugin RAM resource failed\n");
            return 1;
        }
    }
    auto* plugin = qobject_cast<ccStdPluginInterface*>(pluginObject);
    const QList<QAction*> pluginActions =
            plugin ? plugin->getActions() : QList<QAction*>();
    if (!plugin || pluginActions.size() != 1 ||
        pluginActions.first()->text() !=
                QStringLiteral("X-AnyLabeling Annotation") ||
        pluginActions.first()->icon().isNull()) {
        std::fprintf(stderr,
                     "Plugin action contract failed: plugin=%p actions=%d\n",
                     static_cast<void*>(plugin), pluginActions.size());
        return 1;
    }
    pluginActions.first()->trigger();
    QCoreApplication::processEvents();
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if (widget->objectName() == QStringLiteral("qAnyLabelingWorkspace"))
            window = qobject_cast<QMainWindow*>(widget);
    }
    const int dockCount =
            window ? window->findChildren<QDockWidget*>().size() : -1;
    const int comboCount =
            window ? window->findChildren<QComboBox*>().size() : -1;
    const bool polygonIconMissing =
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/polygon.png"))
                    .isNull();
    if (!window || !window->isVisible() || dockCount < 5 || comboCount < 4 ||
        polygonIconMissing) {
        std::fprintf(stderr,
                     "Workspace init failed: window=%p visible=%d docks=%d "
                     "combos=%d polygonIconMissing=%d topLevels=%d\n",
                     static_cast<void*>(window), window && window->isVisible(),
                     dockCount, comboCount, polygonIconMissing,
                     QApplication::topLevelWidgets().size());
        return 1;
    }
    if (reviewOnly) return reviewStatusRoundTrip(window) ? 0 : 1;
    const QStringList expectedMenus = {
            QStringLiteral("File"),     QStringLiteral("Edit"),
            QStringLiteral("View"),     QStringLiteral("Theme"),
            QStringLiteral("Language"), QStringLiteral("Upload"),
            QStringLiteral("Export"),   QStringLiteral("Tool"),
            QStringLiteral("Train"),    QStringLiteral("Help")};
    QStringList actualMenus;
    QMenu* toolMenu = nullptr;
    for (QAction* action : window->menuBar()->actions()) {
        actualMenus.append(action->text());
        if (action->text() == QStringLiteral("Tool")) toolMenu = action->menu();
    }
    const QStringList expectedTools = {
            QStringLiteral("Overview"),
            QString(),
            QStringLiteral("Save Cropped Image"),
            QStringLiteral("Save Visualization Image"),
            QStringLiteral("Save Visualization Video"),
            QString(),
            QStringLiteral("Digit Shortcut Manager"),
            QStringLiteral("Label Manager"),
            QStringLiteral("Group ID Manager"),
            QStringLiteral("Shape Manager"),
            QString(),
            QStringLiteral("Shape Converter"),
            QStringLiteral("Point cloud annotation")};
    QStringList actualTools;
    if (toolMenu) {
        for (QAction* action : toolMenu->actions())
            actualTools.append(action->isSeparator() ? QString()
                                                     : action->text());
    }
    auto* instruction = window->findChild<QLabel*>(
            QStringLiteral("AnnotationInstructions"));
    if (actualMenus != expectedMenus || actualTools != expectedTools ||
        !instruction ||
        !instruction->text().contains(QStringLiteral("Mode:")) ||
        !instruction->text().contains(QStringLiteral("Shortcuts:")) ||
        !toolMenu || !toolMenu->actions().at(2)->isEnabled() ||
        !toolMenu->actions().at(3)->isEnabled() ||
        toolMenu->actions().at(4)->isEnabled() ||
        toolMenu->actions().at(9)->isEnabled() ||
        !toolMenu->actions().at(11)->isEnabled() ||
        !toolMenu->actions().at(12)->isEnabled() ||
        toolMenu->actions().at(12)->shortcut() !=
                QKeySequence(Qt::CTRL + Qt::Key_6) ||
        toolMenu->actions().at(12)->property("qAnyShortcutId").toString() !=
                QStringLiteral("open_pointcloud")) {
        std::fprintf(stderr,
                     "Menu contract failed: menus=%s tools=%s instruction=%s\n",
                     actualMenus.join('|').toLocal8Bit().constData(),
                     actualTools.join('|').toLocal8Bit().constData(),
                     instruction ? instruction->text().toLocal8Bit().constData()
                                 : "missing");
        return 1;
    }
    QMenu* fileMenu = nullptr;
    for (QAction* action : window->menuBar()->actions()) {
        if (action->text() == QStringLiteral("File")) {
            fileMenu = action->menu();
            break;
        }
    }
    const QStringList expectedFilePrefix = {
            QStringLiteral("Open File"),
            QStringLiteral("Next Image"),
            QStringLiteral("Prev Image"),
            QStringLiteral("Next Unchecked Image"),
            QStringLiteral("Prev Unchecked Image"),
            QStringLiteral("Open Dir"),
            QStringLiteral("Open Video"),
            QStringLiteral("Compare View"),
            QStringLiteral("Open Recent"),
            QStringLiteral("Save"),
            QStringLiteral("Save As"),
            QStringLiteral("Save Automatically"),
            QStringLiteral("Change Output Dir"),
            QStringLiteral("Save With Image Data"),
            QStringLiteral("Close")};
    bool fileOrderMatches = fileMenu != nullptr;
    if (fileOrderMatches) {
        const QList<QAction*> actions = fileMenu->actions();
        if (actions.size() < expectedFilePrefix.size())
            fileOrderMatches = false;
        for (int i = 0; fileOrderMatches && i < expectedFilePrefix.size(); ++i)
            fileOrderMatches =
                    !actions.at(i)->isSeparator() &&
                    actions.at(i)->text() == expectedFilePrefix.at(i);
    }
    QAction* changeOutput = window->findChild<QAction*>(
            QStringLiteral("AnnotationChangeOutputDirectory"));
    QAction* saveAutomatically = nullptr;
    QAction* saveWithImageData = nullptr;
    if (fileMenu) {
        for (QAction* action : fileMenu->actions()) {
            if (action->text() == QStringLiteral("Save Automatically"))
                saveAutomatically = action;
            if (action->text() == QStringLiteral("Save With Image Data"))
                saveWithImageData = action;
        }
    }
    if (!fileOrderMatches || !changeOutput || !changeOutput->isEnabled() ||
        !saveAutomatically || !saveAutomatically->isCheckable() ||
        !saveWithImageData || !saveWithImageData->isCheckable()) {
        std::fprintf(stderr, "File menu contract failed\n");
        return 1;
    }
    QMenu* viewMenu = nullptr;
    for (QAction* action : window->menuBar()->actions()) {
        if (action->text() == QStringLiteral("View")) {
            viewMenu = action->menu();
            break;
        }
    }
    const QStringList expectedViewPrefix = {
            QStringLiteral("Navigator"),
            QStringLiteral("Image Tags"),
            QStringLiteral("Fill Drawing Polygon"),
            QStringLiteral("Loop Through Labels"),
            QStringLiteral("Loop Select Labels"),
            QString(),
            QStringLiteral("Zoom In"),
            QStringLiteral("Zoom Out"),
            QStringLiteral("Original Size"),
            QString(),
            QStringLiteral("Keep Previous Scale"),
            QStringLiteral("Keep Previous Brightness"),
            QStringLiteral("Keep Previous Contrast"),
            QString(),
            QStringLiteral("Fit Window"),
            QStringLiteral("Fit Width"),
            QString(),
            QStringLiteral("Show Masks"),
            QStringLiteral("Show Texts"),
            QStringLiteral("Show Labels"),
            QStringLiteral("Show Scores"),
            QStringLiteral("Show Degress"),
            QStringLiteral("Show Attributes"),
            QStringLiteral("Show KIE Linking"),
            QStringLiteral("Show Groups"),
            QStringLiteral("Hide selected polygons"),
            QStringLiteral("Show hidden polygons"),
            QStringLiteral("Group selected shapes"),
            QStringLiteral("Ungroup selected shapes")};
    bool viewOrderMatches = viewMenu != nullptr;
    if (viewOrderMatches) {
        const QList<QAction*> actions = viewMenu->actions();
        if (actions.size() < expectedViewPrefix.size())
            viewOrderMatches = false;
        for (int i = 0; viewOrderMatches && i < expectedViewPrefix.size();
             ++i) {
            const QString expected = expectedViewPrefix.at(i);
            viewOrderMatches =
                    expected.isEmpty()
                            ? actions.at(i)->isSeparator()
                            : !actions.at(i)->isSeparator() &&
                                      actions.at(i)->text() == expected;
        }
    }
    QAction* fillDrawingAction = nullptr;
    QAction* keepPreviousScale = nullptr;
    if (viewMenu) {
        for (QAction* action : viewMenu->actions()) {
            if (action->text() == QStringLiteral("Fill Drawing Polygon"))
                fillDrawingAction = action;
            if (action->text() == QStringLiteral("Keep Previous Scale"))
                keepPreviousScale = action;
        }
    }
    if (!viewOrderMatches || !fillDrawingAction ||
        !fillDrawingAction->isCheckable() || !keepPreviousScale ||
        !keepPreviousScale->isCheckable()) {
        std::fprintf(stderr, "View menu source order contract failed\n");
        return 1;
    }
    QMenu* editMenuContract = nullptr;
    for (QAction* action : window->menuBar()->actions()) {
        if (action->text() == QStringLiteral("Edit")) {
            editMenuContract = action->menu();
            break;
        }
    }
    // The Edit menu follows the source order: create modes, edit modes,
    // object commands, undo, coordinates/merge, then the mode toggles.
    const QStringList expectedEditPrefix = {
            QStringLiteral("Polygon"),
            QStringLiteral("Create brush polygon"),
            QStringLiteral("Magic wand"),
            QStringLiteral("Rectangle"),
            QStringLiteral("Cuboid"),
            QStringLiteral("Rotation"),
            QStringLiteral("Quadrilateral"),
            QStringLiteral("Circle"),
            QStringLiteral("Line"),
            QStringLiteral("Point"),
            QStringLiteral("Line strip"),
            QString(),
            QStringLiteral("Edit"),
            QStringLiteral("Brush edit"),
            QStringLiteral("Edit label"),
            QStringLiteral("Duplicate objects"),
            QStringLiteral("Delete object"),
            QStringLiteral("Copy objects"),
            QStringLiteral("Paste objects"),
            QString(),
            QStringLiteral("Undo"),
            QStringLiteral("Undo last point"),
            QString(),
            QStringLiteral("Copy Coordinates"),
            QStringLiteral("Remove selected point"),
            QStringLiteral("Union selected objects"),
            QString(),
            QStringLiteral("Toggle keep previous image adjustments"),
            QStringLiteral("Auto use last label"),
            QStringLiteral("Toggle auto use last group ID"),
            QStringLiteral("Use System Clipboard"),
            QStringLiteral("Show Shapes")};
    bool editOrderMatches = editMenuContract != nullptr;
    if (editOrderMatches) {
        const QList<QAction*> actions = editMenuContract->actions();
        if (actions.size() < expectedEditPrefix.size())
            editOrderMatches = false;
        for (int i = 0; editOrderMatches && i < expectedEditPrefix.size();
             ++i) {
            const QString expected = expectedEditPrefix.at(i);
            editOrderMatches =
                    expected.isEmpty()
                            ? actions.at(i)->isSeparator()
                            : !actions.at(i)->isSeparator() &&
                                      actions.at(i)->text() == expected;
        }
    }
    if (!editOrderMatches) {
        std::fprintf(stderr, "Edit menu source order contract failed\n");
        return 1;
    }
    fillDrawingAction->setChecked(false);
    fillDrawingAction->setChecked(true);
    QAction* workspaceOpenVideo = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Open Video")) {
            workspaceOpenVideo = action;
            break;
        }
    }
    if (!workspaceOpenVideo) {
        std::fprintf(stderr, "Workspace video action not found\n");
        return 1;
    }
    workspaceOpenVideo->trigger();
    QCoreApplication::processEvents();
    QDialog* videoDialog = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if (widget->windowTitle() == QStringLiteral("SAM3 video tracking"))
            videoDialog = qobject_cast<QDialog*>(widget);
    }
    if (!videoDialog || !videoDialog->isVisible()) {
        std::fprintf(stderr,
                     "Video dialog failed: dialog=%p visible=%d topLevels=%d\n",
                     static_cast<void*>(videoDialog),
                     videoDialog && videoDialog->isVisible(),
                     QApplication::topLevelWidgets().size());
        return 1;
    }
    videoDialog->close();
    bool hasRFDetr = false, hasSAM = false;
    for (QComboBox* combo : window->findChildren<QComboBox*>()) {
        hasRFDetr |= combo->findText(QStringLiteral("rfdetr-seg-small")) >= 0;
        hasSAM |= combo->findText(QStringLiteral("sam2.1_hiera_tiny")) >= 0;
    }
    if (!hasRFDetr || !hasSAM) {
        std::fprintf(stderr,
                     "RF-DETR or SAM catalog is unavailable in the model "
                     "selector\n");
        return 1;
    }
    QAction* autoLabelPanelOpen = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Auto Labeling")) {
            autoLabelPanelOpen = action;
            break;
        }
    }
    if (!autoLabelPanelOpen) {
        std::fprintf(stderr, "Auto labeling panel action not found\n");
        return 1;
    }
    QAction* autoRunProbe = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Auto Run")) {
            autoRunProbe = action;
            break;
        }
    }
    if (!autoRunProbe || autoRunProbe->isEnabled()) {
        std::fprintf(
                stderr,
                "Auto Run must start disabled until the model panel opens\n");
        return 1;
    }
    autoLabelPanelOpen->trigger();
    QCoreApplication::processEvents();
    if (!autoRunProbe->isEnabled()) {
        std::fprintf(stderr,
                     "Auto Run did not follow the model panel visibility\n");
        return 1;
    }
    auto* modelPanel = window->findChild<QWidget*>(QStringLiteral("AIModel"));
    auto* canvasScrollForModel = window->findChild<QScrollArea*>(
            QStringLiteral("AnnotationCanvasScrollArea"));
    const QPoint modelBottom =
            modelPanel
                    ? modelPanel->mapTo(window, QPoint(0, modelPanel->height()))
                    : QPoint();
    const QPoint canvasTop =
            canvasScrollForModel
                    ? canvasScrollForModel->mapTo(window, QPoint(0, 0))
                    : QPoint();
    const QPoint instructionBottom =
            instruction ? instruction->mapTo(window,
                                             QPoint(0, instruction->height()))
                        : QPoint();
    if (!modelPanel || qobject_cast<QDockWidget*>(modelPanel) ||
        !modelPanel->isVisible() || !canvasScrollForModel ||
        instructionBottom.y() > modelPanel->mapTo(window, QPoint(0, 0)).y() ||
        modelBottom.y() > canvasTop.y()) {
        std::fprintf(stderr,
                     "Auto labeling panel is not embedded between the "
                     "instruction row and the canvas\n");
        return 1;
    }
    auto* models = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationModelFamily"));
    auto* samText =
            window->findChild<QLineEdit*>(QStringLiteral("SamTextPrompt"));
    auto* textEncoder = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationTextEncoder"));
    auto* quantization = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationQuantization"));
    auto* backend =
            window->findChild<QComboBox*>(QStringLiteral("AnnotationBackend"));
    auto* brushRadius = window->findChild<QSpinBox*>(
            QStringLiteral("AnnotationBrushRadius"));
    auto* brushMode = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationBrushMode"));
    auto* brushTarget = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationBrushTarget"));
    if (!models || !samText || samText->isVisible() || !textEncoder ||
        !quantization || !backend || backend->count() == 0 || !brushRadius ||
        !brushMode || !brushTarget || brushRadius->minimum() != 1 ||
        brushRadius->maximum() != 200 || brushMode->findData(0) < 0 ||
        brushMode->findData(1) < 0 || brushMode->findData(2) < 0 ||
        !brushTarget->isEditable() ||
        models->findText(QStringLiteral("sam3")) < 0 ||
        models->findText(QStringLiteral("gkd_fullset")) < 0 ||
        models->findText(QStringLiteral("buffalo_sc")) < 0 ||
        models->findText(QStringLiteral("reid-yolo26n")) >= 0 ||
        models->findText(QStringLiteral("depth-anything-nested-metric")) >= 0 ||
        models->findText(QStringLiteral("depth-anything-nested-anyview")) < 0 ||
        models->findText(QStringLiteral("deeplsd_wireframe")) < 0 ||
        models->findText(QStringLiteral("deeplsd_md")) < 0 ||
        models->findText(QStringLiteral("aliked-n16rot")) < 0 ||
        models->findText(QStringLiteral("PP-OCRv4 (ch)")) < 0 ||
        models->findText(QStringLiteral("PP-OCRv4 (en)")) < 0 ||
        models->findText(QStringLiteral("yolov8s-world")) < 0 ||
        models->findText(QStringLiteral("yoloe-26n-seg-pf")) < 0 ||
        models->findText(QStringLiteral("ram_plus_swin_large_14m")) < 0 ||
        models->findText(QStringLiteral("tag2text_swin_14m")) < 0 ||
        models->findText(QStringLiteral("relsgg-vits16plus")) < 0 ||
        models->findText(QStringLiteral("relsgg-convnext-tiny")) < 0 ||
        models->findText(QStringLiteral("relsgg-vits16plus-text-student")) >=
                0) {
        std::fprintf(stderr,
                     "SAM text prompt visibility or model catalog failed\n");
        std::fprintf(
                stderr,
                "Model controls failed: models=%p samText=%p textEncoder=%p "
                "quant=%p backend=%p radius=%p mode=%p target=%p\n",
                static_cast<void*>(models), static_cast<void*>(samText),
                static_cast<void*>(textEncoder),
                static_cast<void*>(quantization), static_cast<void*>(backend),
                static_cast<void*>(brushRadius), static_cast<void*>(brushMode),
                static_cast<void*>(brushTarget));
        return 1;
    }
    // Every family published by AICore must expose at least one exact GGUF
    // filename in the quantization selector. This catches catalog/UI drift
    // without duplicating the catalog in the plugin test.
    for (int familyIndex = 0; familyIndex < models->count(); ++familyIndex) {
        models->setCurrentIndex(familyIndex);
        QCoreApplication::processEvents();
        if (quantization->count() == 0) {
            std::fprintf(stderr, "Model family has no quantization: %s\n",
                         models->currentText().toLocal8Bit().constData());
            return 1;
        }
        for (int quantIndex = 0; quantIndex < quantization->count();
             ++quantIndex) {
            if (quantization->itemData(quantIndex).toString().isEmpty()) {
                std::fprintf(stderr,
                             "Quantization row has no GGUF filename: %s\n",
                             models->currentText().toLocal8Bit().constData());
                return 1;
            }
        }
    }
    auto* ocrDirection = window->findChild<QCheckBox*>(
            QStringLiteral("AnnotationOcrAngleClassifier"));
    if (!ocrDirection || !ocrDirection->isChecked()) {
        std::fprintf(stderr,
                     "OCR orientation classifier missing or not enabled by "
                     "default\n");
        return 1;
    }
    // The public catalog defines the file set; the real picker must not omit
    // published quantizations, mix recipes or duplicate files. Unpublished
    // recipes retain only local F32 fallback. O(P*C): preset/catalog tables are
    // small and bounded; enumerate independently to verify the UI projection.
    for (int pi = 0; pi < aicore_ocr_preset_count(); ++pi) {
        const auto* preset = aicore_ocr_preset_at(pi);
        if (!preset || !preset->display_name || !preset->rec_stem) return 1;
        const QString family = QString::fromUtf8(preset->display_name);
        QSet<QString> expectedFiles;
        for (int ci = 0; ci < aicore_ocr_model_count(); ++ci) {
            const auto* entry = aicore_ocr_model_at(ci);
            if (!entry || !entry->filename || !entry->role ||
                qstrcmp(entry->role, "rec") != 0)
                continue;
            if (aicore_ocr_preset_by_rec_filename(entry->filename) == preset)
                expectedFiles.insert(QString::fromUtf8(entry->filename));
        }
        if (expectedFiles.isEmpty())
            expectedFiles.insert(QString::fromUtf8(preset->rec_stem) +
                                 QStringLiteral("-f32.gguf"));
        if (models->findText(family) < 0) {
            std::fprintf(stderr, "Missing OCR recipe family: %s\n",
                         qPrintable(family));
            return 1;
        }
        models->setCurrentText(family);
        QSet<QString> actualFiles;
        for (int qi = 0; qi < quantization->count(); ++qi)
            actualFiles.insert(quantization->itemData(qi).toString());
        if (!ocrDirection->isVisible() || actualFiles != expectedFiles ||
            actualFiles.size() != quantization->count()) {
            std::fprintf(
                    stderr,
                    "OCR catalog/picker mismatch: %s actual=%s expected=%s\n",
                    qPrintable(family),
                    qPrintable(QStringList(actualFiles.values()).join(',')),
                    qPrintable(QStringList(expectedFiles.values()).join(',')));
            return 1;
        }
    }
    models->setCurrentText(QStringLiteral("sam3"));
    if (ocrDirection->isVisible()) return 1;
    if (!samText->isVisible()) {
        std::fprintf(stderr, "SAM 3 text prompt is hidden\n");
        return 1;
    }
    models->setCurrentText(QStringLiteral("sam2.1_hiera_tiny"));
    if (samText->isVisible()) {
        std::fprintf(stderr, "Visual-only SAM model exposes a text prompt\n");
        return 1;
    }
    models->setCurrentText(QStringLiteral("gkd_fullset"));
    if (!samText->isVisible() || textEncoder->isVisible()) return 1;
    models->setCurrentText(QStringLiteral("depth-anything-nested-anyview"));
    if (samText->isVisible() || textEncoder->isVisible() ||
        quantization->findData(QStringLiteral(
                "depth-anything-nested-anyview-q8_0.gguf")) < 0 ||
        quantization->findData(
                QStringLiteral("depth-anything-nested-anyview-q4_k.gguf")) < 0)
        return 1;
    models->setCurrentText(QStringLiteral("yolov8s-world"));
    if (!samText->isVisible() || !textEncoder->isVisible() ||
        textEncoder->findText(QStringLiteral("F16")) < 0)
        return 1;
    models->setCurrentText(QStringLiteral("yoloe-26n-seg-pf"));
    if (samText->isVisible() || textEncoder->isVisible()) return 1;
    QTemporaryDir directory;
    if (!directory.isValid()) return 1;
    const QString imagePath = directory.path() + QStringLiteral("/sample.png");
    const QString sidecarPath =
            directory.path() + QStringLiteral("/sample.json");
    const QString checkedImagePath =
            directory.path() + QStringLiteral("/sample2.png");
    const QString uncheckedImagePath =
            directory.path() + QStringLiteral("/sample3.png");
    QImage image(64, 48, QImage::Format_RGB32);
    image.fill(QColor(80, 120, 140));
    if (!image.save(imagePath) || !image.save(checkedImagePath) ||
        !image.save(uncheckedImagePath))
        return 1;
    QFile checkedSidecar(directory.path() + QStringLiteral("/sample2.json"));
    if (!checkedSidecar.open(QIODevice::WriteOnly) ||
        checkedSidecar.write("{\"checked\":true,\"shapes\":[]}") < 0)
        return 1;
    checkedSidecar.close();
    QJsonObject shape{
            {QStringLiteral("label"), QStringLiteral("vehicle")},
            {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{1, 2}, QJsonArray{20, 2}, QJsonArray{20, 30},
                        QJsonArray{1, 30}}},
            {QStringLiteral("score"), 0.92},
            {QStringLiteral("group_id"), 7}};
    QJsonObject document{
            {QStringLiteral("version"), QStringLiteral("4.0.6")},
            {QStringLiteral("imagePath"), QStringLiteral("sample.png")},
            {QStringLiteral("imageData"), QJsonValue::Null},
            {QStringLiteral("shapes"), QJsonArray{shape}},
            {QStringLiteral("tags"),
             QJsonArray{QStringLiteral(" machine "), QStringLiteral("camera"),
                        QStringLiteral("machine"), 12}},
            {QStringLiteral("custom_field"), QStringLiteral("keep")}};
    QFile sidecar(sidecarPath);
    if (!sidecar.open(QIODevice::WriteOnly) ||
        sidecar.write(QJsonDocument(document).toJson()) < 0)
        return 1;
    sidecar.close();

    QAction* save = nullptr;
    QAction* copy = nullptr;
    QAction* paste = nullptr;
    QAction* editLabel = nullptr;
    QAction* labelManager = nullptr;
    QAction* duplicate = nullptr;
    QAction* hideSelected = nullptr;
    QAction* showHidden = nullptr;
    QAction* nextUnchecked = nullptr;
    QAction* poseExport = nullptr;
    QAction* cocoPoseExport = nullptr;
    QAction* maskExport = nullptr;
    QAction* importMenu = nullptr;
    QAction* sam3ArchiveImport = nullptr;
    QAction* sam3Workspace = nullptr;
    QAction* cocoImport = nullptr;
    QAction* group = nullptr;
    QAction* ungroup = nullptr;
    QAction* objectUnion = nullptr;
    QAction* undo = nullptr;
    QAction* motsExport = nullptr;
    QAction* vlmAction = nullptr;
    QAction* openImage = nullptr;
    QAction* openVideo = nullptr;
    QAction* openFolder = nullptr;
    QAction* autoLabel = nullptr;
    QAction* polygonTool = nullptr;
    QAction* brushTool = nullptr;
    QAction* fitImage = nullptr;
    QAction* compareView = nullptr;
    QAction* saveAs = nullptr;
    QAction* closeImage = nullptr;
    QAction* deleteLabelFile = nullptr;
    QAction* deleteImageFile = nullptr;
    QAction* autoLabelPanel = nullptr;
    QAction* zoomOriginal = nullptr;
    QAction* fitWidth = nullptr;
    QAction* showShapes = nullptr;
    QAction* showLabels = nullptr;
    QAction* keepPreviousBrightness = nullptr;
    QAction* keepPreviousContrast = nullptr;
    QAction* autoUseLastLabel = nullptr;
    QAction* createBrush = nullptr;
    QAction* imageTags = nullptr;
    QAction* imageFlagsAction = nullptr;
    QAction* settingsAction = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Save")) save = action;
        if (action->text() == QStringLiteral("Copy objects")) copy = action;
        if (action->text() == QStringLiteral("Paste objects")) paste = action;
        if (action->text() == QStringLiteral("Edit label")) editLabel = action;
        if (action->text() == QStringLiteral("Label Manager"))
            labelManager = action;
        if (action->text() == QStringLiteral("Duplicate objects"))
            duplicate = action;
        if (action->text() == QStringLiteral("Hide selected polygons"))
            hideSelected = action;
        if (action->text() == QStringLiteral("Show hidden polygons"))
            showHidden = action;
        if (action->text() == QStringLiteral("Next Unchecked Image"))
            nextUnchecked = action;
        if (action->text() == QStringLiteral("YOLO Pose")) poseExport = action;
        if (action->text() == QStringLiteral("COCO Keypoints"))
            cocoPoseExport = action;
        if (action->text() == QStringLiteral("MASK")) maskExport = action;
        if (action->text() == QStringLiteral("Upload")) importMenu = action;
        if (action->text() == QStringLiteral("SAM3 tracking archive"))
            sam3ArchiveImport = action;
        if (action->text() == QStringLiteral("Open SAM3 archive workspace"))
            sam3Workspace = action;
        if (action->text() == QStringLiteral("COCO Keypoints"))
            cocoImport = action;
        if (action->text() == QStringLiteral("Group selected shapes"))
            group = action;
        if (action->text() == QStringLiteral("Ungroup selected shapes"))
            ungroup = action;
        if (action->text() == QStringLiteral("Union selected objects"))
            objectUnion = action;
        if (action->text() == QStringLiteral("Undo")) undo = action;
        if (action->text() == QStringLiteral("MOTS")) motsExport = action;
        if (action->text() == QStringLiteral("VLM-R1 OVD")) vlmAction = action;
        if (action->text() == QStringLiteral("Open File")) openImage = action;
        if (action->text() == QStringLiteral("Open Video")) openVideo = action;
        if (action->text() == QStringLiteral("Open Dir")) openFolder = action;
        if (action->text() == QStringLiteral("Auto Run")) autoLabel = action;
        if (action->text() == QStringLiteral("Polygon")) polygonTool = action;
        if (action->text() == QStringLiteral("Brush edit")) brushTool = action;
        if (action->text() == QStringLiteral("Fit Window")) fitImage = action;
        if (action->text() == QStringLiteral("Compare View"))
            compareView = action;
        if (action->text() == QStringLiteral("Save As")) saveAs = action;
        if (action->text() == QStringLiteral("Close")) closeImage = action;
        if (action->text() == QStringLiteral("Delete File"))
            deleteLabelFile = action;
        if (action->text() == QStringLiteral("Delete Image File"))
            deleteImageFile = action;
        if (action->text() == QStringLiteral("Auto Labeling"))
            autoLabelPanel = action;
        if (action->text() == QStringLiteral("Original Size"))
            zoomOriginal = action;
        if (action->text() == QStringLiteral("Fit Width")) fitWidth = action;
        if (action->text() == QStringLiteral("Show Shapes"))
            showShapes = action;
        if (action->text() == QStringLiteral("Show Labels"))
            showLabels = action;
        if (action->text() == QStringLiteral("Keep Previous Brightness"))
            keepPreviousBrightness = action;
        if (action->text() == QStringLiteral("Keep Previous Contrast"))
            keepPreviousContrast = action;
        if (action->text() == QStringLiteral("Auto use last label"))
            autoUseLastLabel = action;
        if (action->text() == QStringLiteral("Create brush polygon"))
            createBrush = action;
        if (action->text() == QStringLiteral("Image Tags")) imageTags = action;
        if (action->text() == QStringLiteral("Image Flags"))
            imageFlagsAction = action;
        if (action->objectName() == QStringLiteral("AnnotationSettings"))
            settingsAction = action;
    }
    // Source lifecycle state binding: commands that depend on an image or
    // the file list start disabled before any image is opened.
    QAction* nextImage = nullptr;
    QAction* prevImage = nullptr;
    QAction* markChecked = nullptr;
    QAction* zoomIn = nullptr;
    QAction* zoomOut = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Next Image")) nextImage = action;
        if (action->text() == QStringLiteral("Prev Image")) prevImage = action;
        if (action->text() == QStringLiteral("Mark as Checked"))
            markChecked = action;
        if (action->text() == QStringLiteral("Zoom In")) zoomIn = action;
        if (action->text() == QStringLiteral("Zoom Out")) zoomOut = action;
    }
    if (!nextImage || !prevImage || !markChecked || !zoomIn || !zoomOut ||
        save->isEnabled() || nextImage->isEnabled() || prevImage->isEnabled() ||
        nextUnchecked->isEnabled() || markChecked->isEnabled() ||
        deleteLabelFile->isEnabled() || saveAs->isEnabled() ||
        fitImage->isEnabled() || fitWidth->isEnabled() || zoomIn->isEnabled() ||
        zoomOut->isEnabled() || zoomOriginal->isEnabled() ||
        !closeImage->isEnabled()) {
        std::fprintf(stderr, "Image commands must start disabled\n");
        return 1;
    }
    // The focus-free review popup shows the source loop messages without
    // stealing canvas focus, and Ctrl+Q quits through the close confirmation.
    auto* reviewPopup = window->findChild<QWidget*>(
            QStringLiteral("AnnotationReviewPopup"));
    QAction* loopThrough = nullptr;
    QAction* quitAction = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Loop Through Labels"))
            loopThrough = action;
        if (action->text() == QStringLiteral("Quit")) quitAction = action;
    }
    Q_UNUSED(loopThrough);
    if (!reviewPopup || !loopThrough || !quitAction ||
        quitAction->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_Q)) {
        std::fprintf(stderr,
                     "Review popup or quit action missing: popup=%p loop=%p "
                     "quit=%p\n",
                     (void*)reviewPopup, (void*)loopThrough, (void*)quitAction);
        return 1;
    }
    if (!reviewPopup->windowFlags().testFlag(Qt::WindowDoesNotAcceptFocus) ||
        !reviewPopup->windowFlags().testFlag(Qt::WindowTransparentForInput)) {
        std::fprintf(stderr,
                     "Review popup must be focus-free and click-through\n");
        return 1;
    }
    const bool imageOpened =
            QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, imagePath));
    QMenu* uploadMenu = importMenu ? importMenu->menu() : nullptr;
    QMenu* exportMenu = nullptr;
    for (QAction* action : window->menuBar()->actions()) {
        if (action->text() == QStringLiteral("Export")) {
            exportMenu = action->menu();
            break;
        }
    }
    const auto menuHasIcons = [](QMenu* menu) {
        if (!menu) return false;
        for (QAction* action : menu->actions()) {
            if (!action->isSeparator() && action->icon().isNull()) return false;
        }
        return true;
    };
    const auto disabledTodo = [uploadMenu](const QString& objectName) {
        if (!uploadMenu) return false;
        for (QAction* action : uploadMenu->actions()) {
            if (action->objectName() == objectName)
                return !action->isEnabled() && !action->icon().isNull();
        }
        return false;
    };
    // The implemented template uploads keep their official names, icons
    // and tooltips but are now enabled.
    const auto enabledUpload = [uploadMenu](const QString& objectName) {
        if (!uploadMenu) return false;
        for (QAction* action : uploadMenu->actions()) {
            if (action->objectName() == objectName)
                return action->isEnabled() && !action->icon().isNull() &&
                       !action->toolTip().isEmpty();
        }
        return false;
    };
    if (!save || !copy || !paste || !editLabel || !labelManager || !duplicate ||
        !hideSelected || !showHidden || !nextUnchecked || !poseExport ||
        !cocoPoseExport || !maskExport || !importMenu || !sam3ArchiveImport ||
        !sam3Workspace || !cocoImport || !group || !ungroup || !objectUnion ||
        !undo || !motsExport || !vlmAction || !openImage || !openVideo ||
        !openFolder || !autoLabel || !polygonTool || !brushTool || !fitImage ||
        !compareView || !saveAs || !closeImage || !deleteLabelFile ||
        !deleteImageFile || !autoLabelPanel || !zoomOriginal || !fitWidth ||
        !showShapes || !showLabels || !imageTags || !keepPreviousBrightness ||
        !keepPreviousContrast || !autoUseLastLabel || !createBrush ||
        !autoUseLastLabel->isCheckable() || !imageFlagsAction ||
        imageFlagsAction->icon().isNull() ||
        imageFlagsAction->toolTip() !=
                QStringLiteral("Upload Custom Image Flags File") ||
        !settingsAction ||
        settingsAction->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_0) ||
        !keepPreviousBrightness->isCheckable() ||
        !keepPreviousContrast->isCheckable() || !menuHasIcons(uploadMenu) ||
        !menuHasIcons(exportMenu) ||
        !enabledUpload(QStringLiteral("AnnotationUploadLabelFlags")) ||
        !enabledUpload(QStringLiteral("AnnotationUploadAttributes")) ||
        !enabledUpload(QStringLiteral("AnnotationUploadLabelClasses")) ||
        !disabledTodo(
                QStringLiteral("AnnotationTODOImport_MM-Grounding-DINO")) ||
        !imageOpened) {
        std::fprintf(
                stderr,
                "Action/data contract failed: save=%p copy=%p paste=%p edit=%p "
                "label=%p duplicate=%p hide=%p show=%p next=%p pose=%p coco=%p "
                "mask=%p import=%p archive=%p workspace=%p group=%p ungroup=%p "
                "union=%p undo=%p mots=%p vlm=%p openImage=%p openVideo=%p "
                "openFolder=%p auto=%p polygon=%p brush=%p fit=%p compare=%p "
                "saveAs=%p close=%p deleteLabel=%p deleteImage=%p panel=%p "
                "zoom=%p fitWidth=%p showShapes=%p showLabels=%p tags=%p "
                "brightness=%p contrast=%p last=%p createBrush=%p settings=%p "
                "imageOpened=%d\n",
                static_cast<void*>(save), static_cast<void*>(copy),
                static_cast<void*>(paste), static_cast<void*>(editLabel),
                static_cast<void*>(labelManager), static_cast<void*>(duplicate),
                static_cast<void*>(hideSelected),
                static_cast<void*>(showHidden),
                static_cast<void*>(nextUnchecked),
                static_cast<void*>(poseExport),
                static_cast<void*>(cocoPoseExport),
                static_cast<void*>(maskExport), static_cast<void*>(importMenu),
                static_cast<void*>(sam3ArchiveImport),
                static_cast<void*>(sam3Workspace), static_cast<void*>(group),
                static_cast<void*>(ungroup), static_cast<void*>(objectUnion),
                static_cast<void*>(undo), static_cast<void*>(motsExport),
                static_cast<void*>(vlmAction), static_cast<void*>(openImage),
                static_cast<void*>(openVideo), static_cast<void*>(openFolder),
                static_cast<void*>(autoLabel), static_cast<void*>(polygonTool),
                static_cast<void*>(brushTool), static_cast<void*>(fitImage),
                static_cast<void*>(compareView), static_cast<void*>(saveAs),
                static_cast<void*>(closeImage),
                static_cast<void*>(deleteLabelFile),
                static_cast<void*>(deleteImageFile),
                static_cast<void*>(autoLabelPanel),
                static_cast<void*>(zoomOriginal), static_cast<void*>(fitWidth),
                static_cast<void*>(showShapes), static_cast<void*>(showLabels),
                static_cast<void*>(imageTags),
                static_cast<void*>(keepPreviousBrightness),
                static_cast<void*>(keepPreviousContrast),
                static_cast<void*>(autoUseLastLabel),
                static_cast<void*>(createBrush),
                static_cast<void*>(settingsAction), imageOpened);
        return 1;
    }
    // After opening an image the lifecycle-bound commands become available;
    // the fixture image has a sidecar and shapes, so Delete File and Save As
    // follow it too.
    if (!nextImage->isEnabled() || !prevImage->isEnabled() ||
        !nextUnchecked->isEnabled() || !markChecked->isEnabled() ||
        !fitImage->isEnabled() || !fitWidth->isEnabled() ||
        !zoomIn->isEnabled() || !zoomOut->isEnabled() ||
        !zoomOriginal->isEnabled() || !save->isEnabled() ||
        !saveAs->isEnabled() || !closeImage->isEnabled() ||
        !deleteImageFile->isEnabled() || !deleteLabelFile->isEnabled()) {
        std::fprintf(stderr,
                     "Image lifecycle state binding failed after load: "
                     "next=%d prev=%d nextU=%d mark=%d fit=%d fitW=%d "
                     "zoomIn=%d zoomOut=%d zoomOrg=%d save=%d saveAs=%d "
                     "close=%d delImg=%d delLabel=%d\n",
                     nextImage->isEnabled(), prevImage->isEnabled(),
                     nextUnchecked->isEnabled(), markChecked->isEnabled(),
                     fitImage->isEnabled(), fitWidth->isEnabled(),
                     zoomIn->isEnabled(), zoomOut->isEnabled(),
                     zoomOriginal->isEnabled(), save->isEnabled(),
                     saveAs->isEnabled(), closeImage->isEnabled(),
                     deleteImageFile->isEnabled(),
                     deleteLabelFile->isEnabled());
        return 1;
    }
    auto* visualizationVideo = window->findChild<QAction*>(
            QStringLiteral("AnnotationVisualizationVideoAction"));
    if (!visualizationVideo) return 1;
    if (visualizationVideo->isEnabled()) {
        if (!sidecar.open(QIODevice::ReadOnly)) return 1;
        const QByteArray annotationBeforeExport = sidecar.readAll();
        sidecar.close();
        for (int pass = 0; pass < 2; ++pass) {
            const QString videoPath = directory.filePath(
                    QStringLiteral("visualization-%1.mp4").arg(pass));
            bool controlsMatch = false;
            QTimer::singleShot(0, [&]() {
                auto* dialog = qobject_cast<QDialog*>(
                        QApplication::activeModalWidget());
                if (!dialog ||
                    dialog->objectName() !=
                            QStringLiteral("AnnotationVisualizationVideo"))
                    return;
                auto* fps =
                        dialog->findChild<QSpinBox*>("VisualizationVideoFps");
                auto* rectangle = dialog->findChild<QCheckBox*>(
                        "VisualizationType_rectangle");
                auto* skip =
                        dialog->findChild<QCheckBox*>("VisualizationSkipEmpty");
                auto* destination =
                        dialog->findChild<QLineEdit*>("VisualizationVideoPath");
                const QString sourceDefaultPath =
                        QDir::cleanPath(directory.path() +
                                        QStringLiteral("/../visualizations/") +
                                        QFileInfo(directory.path()).fileName() +
                                        QStringLiteral("_visualization.mp4"));
                controlsMatch = fps && rectangle && skip && destination &&
                                fps->value() == 25 && fps->minimum() == 1 &&
                                fps->maximum() == 120 &&
                                !rectangle->isChecked() && !skip->isChecked() &&
                                destination->text() == sourceDefaultPath;
                if (!controlsMatch) {
                    dialog->reject();
                    return;
                }
                rectangle->setChecked(true);
                skip->setChecked(pass == 0);
                fps->setValue(pass == 0 ? 25 : 17);
                destination->setText(videoPath);
                dialog->findChild<QDialogButtonBox*>()
                        ->button(QDialogButtonBox::Ok)
                        ->click();
            });
            visualizationVideo->trigger();
            QFile video(videoPath);
            if (!controlsMatch || !video.open(QIODevice::ReadOnly)) {
                std::fprintf(stderr,
                             "visualization video dialog/export failed\n");
                return 1;
            }
            const QByteArray encoded = video.readAll();
            const int sampleTable = encoded.indexOf("stsz");
            if (!encoded.contains("ftyp") || sampleTable < 0 ||
                sampleTable + 16 > encoded.size() ||
                qFromBigEndian<quint32>(reinterpret_cast<const uchar*>(
                        encoded.constData() + sampleTable + 12)) !=
                        quint32(pass == 0 ? 1 : 3)) {
                std::fprintf(stderr,
                             "visualization skip-empty/frame count mismatch\n");
                return 1;
            }
        }
        if (!sidecar.open(QIODevice::ReadOnly) ||
            sidecar.readAll() != annotationBeforeExport) {
            std::fprintf(stderr,
                         "visualization export modified annotation document\n");
            return 1;
        }
        sidecar.close();
    } else if (!visualizationVideo->toolTip().contains(
                       QStringLiteral("unavailable"))) {
        return 1;
    }
    // Selection-driven enable states: one rectangle sidecar shape exists.
    QAction* copyCoordinates = nullptr;
    QAction* loopSelect = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->objectName() == QStringLiteral("AnnotationCopyCoordinates"))
            copyCoordinates = action;
        if (action->text() == QStringLiteral("Loop Select Labels"))
            loopSelect = action;
    }
    if (!copyCoordinates || copyCoordinates->isEnabled() || !loopSelect) {
        std::fprintf(
                stderr,
                "Copy Coordinates must start disabled without a selection\n");
        return 1;
    }
    auto* objectList = window->findChild<QListWidget*>(
            QStringLiteral("AnnotationObjects"));
    if (!objectList || objectList->count() != 1) {
        std::fprintf(stderr, "Object list missing for selection contract\n");
        return 1;
    }
    objectList->setCurrentRow(0);
    QCoreApplication::processEvents();
    if (!copyCoordinates->isEnabled()) {
        std::fprintf(stderr, "Copy Coordinates did not follow the selection\n");
        return 1;
    }
    copyCoordinates->trigger();
    if (QApplication::clipboard()->text() != QStringLiteral("[1, 2, 20, 30]")) {
        std::fprintf(
                stderr, "Copy Coordinates payload mismatch: %s\n",
                QApplication::clipboard()->text().toLocal8Bit().constData());
        return 1;
    }
    // Label dialog fixture: exec() opens a nested loop, so a queued timer
    // exercises sort/completion/accept from inside it.
    QTimer::singleShot(0, window, [window]() {
        QDialog* modal =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (!modal || modal->windowTitle() != QStringLiteral("Label object")) {
            std::fprintf(stderr, "Label dialog fixture: dialog missing\n");
            if (modal) modal->reject();
            return;
        }
        auto* list = modal->findChild<QListWidget*>(
                QStringLiteral("AnnotationLabelDialogList"));
        auto* edit = modal->findChild<QLineEdit*>();
        if (!list || list->count() != 1 ||
            list->item(0)->text() != QStringLiteral("vehicle")) {
            std::fprintf(stderr, "Label dialog fixture: list content wrong\n");
            modal->reject();
            return;
        }
        if (edit) {
            edit->setText(QStringLiteral("veh"));
            if (list->count() != 1) {
                std::fprintf(stderr,
                             "Label dialog fixture: completion filter wrong\n");
                modal->reject();
                return;
            }
            edit->setText(QStringLiteral("vehicle"));
        }
        modal->accept();
    });
    editLabel->trigger();
    QCoreApplication::processEvents();
    {
        QDialog* leftover =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (leftover &&
            leftover->windowTitle() == QStringLiteral("Label object")) {
            std::fprintf(stderr, "Label dialog fixture: dialog stayed open\n");
            leftover->reject();
            return 1;
        }
    }
    // Loop Select Labels walks the object list and clears the selection at
    // the end, matching the source loop_select_labels behavior.
    loopSelect->trigger();
    QCoreApplication::processEvents();
    if (objectList->selectedItems().size() != 1) {
        std::fprintf(stderr, "Loop Select Labels did not select the object\n");
        return 1;
    }
    loopSelect->trigger();
    QCoreApplication::processEvents();
    if (!objectList->selectedItems().isEmpty()) {
        std::fprintf(
                stderr,
                "Loop Select Labels did not clear the selection at the end\n");
        return 1;
    }
    for (const auto& name : {QStringLiteral("AnnotationOpenChatbot"),
                             QStringLiteral("AnnotationOpenVQA")}) {
        auto* action = window->findChild<QAction*>(name);
        if (!action || !action->isEnabled() || action->icon().isNull())
            return 1;
        action->trigger();
        QCoreApplication::processEvents();
        auto* dialog = window->findChild<QDialog*>(
                name.endsWith("VQA") ? QStringLiteral("AnnotationVQA")
                                     : QStringLiteral("AnnotationChatbot"));
        if (!dialog || !dialog->isVisible()) return 1;
        auto* address = dialog->findChild<QLineEdit*>("ChatAddress");
        auto* temperature =
                dialog->findChild<QDoubleSpinBox*>("ChatTemperature");
        auto* tokens = dialog->findChild<QSpinBox*>("ChatMaxTokens");
        if (!address || address->text() != "http://localhost:11434/v1" ||
            !temperature || temperature->value() != 1 || !tokens ||
            tokens->value() != 0)
            return 1;
        dialog->close();
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    }
    // Upstream Ctrl+4 opens its dedicated PaddleOCR document workspace.
    {
        QAction* paddleOcr = window->findChild<QAction*>(
                QStringLiteral("AnnotationOpenPaddleOCR"));
        if (!paddleOcr || !paddleOcr->isEnabled() ||
            paddleOcr->icon().isNull()) {
            std::fprintf(stderr,
                         "PaddleOCR toolbar entry must be live with an icon\n");
            return 1;
        }
        paddleOcr->trigger();
        auto* dialog = window->findChild<QDialog*>(
                QStringLiteral("AnnotationPaddleOCR"));
        if (!dialog || !dialog->isVisible() ||
            dialog->minimumSize() != QSize(1200, 760) ||
            !dialog->findChild<QListWidget*>(QStringLiteral("PpocrRecords"))) {
            std::fprintf(stderr,
                         "PaddleOCR entry must open its dedicated document "
                         "workspace\n");
            return 1;
        }
        paddleOcr->trigger();
        if (window->findChildren<QDialog*>(
                          QStringLiteral("AnnotationPaddleOCR"))
                    .size() != 1)
            return 1;
        dialog->close();
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    }
    // The zh_CN catalog is embedded in the plugin resources and translates
    // the core menu set; untranslated entries fall back to the source text.
    {
        QTranslator catalog;
        const bool loaded = catalog.load(QStringLiteral(
                ":/qAnyLabeling/translations/qanylabeling_zh_CN.qm"));
        const QString translated =
                loaded ? catalog.translate("AnnotationWindow", "File")
                       : QString();
        if (!loaded || translated != QStringLiteral("文件") ||
            catalog.translate("PointCloudWindow", "Perspective") !=
                    QStringLiteral("透视投影") ||
            catalog.translate("PointCloudWindow", "Dataset") !=
                    QStringLiteral("数据集") ||
            catalog.translate("AnnotationChat", "Model settings") !=
                    QStringLiteral("模型设置") ||
            catalog.translate("ChatServiceClient", "Chat request timed out.") !=
                    QStringLiteral("聊天请求超时。")) {
            std::fprintf(
                    stderr,
                    "zh_CN translation catalog failed to load: "
                    "exists=%d size=%lld translated=[%s]\n",
                    QFile::exists(QStringLiteral(":/qAnyLabeling/translations/"
                                                 "qanylabeling_zh_CN.qm")),
                    QFileInfo(QStringLiteral(":/qAnyLabeling/translations/"
                                             "qanylabeling_zh_CN.qm"))
                            .size(),
                    translated.toUtf8().constData());
            return 1;
        }
    }
    if (qgetenv("QANYLABELING_HOLD_WINDOW") == QByteArrayLiteral("1")) {
        QTimer::singleShot(300, window, [window]() {
            const QByteArray screenshot = qgetenv("QANYLABELING_SCREENSHOT");
            if (!screenshot.isEmpty())
                window->grab().save(QString::fromLocal8Bit(screenshot));
            QTimer::singleShot(300, qApp, &QCoreApplication::quit);
        });
        return app.exec();
    }
    auto* recentMenu =
            window->findChild<QMenu*>(QStringLiteral("AnnotationRecentMenu"));
    if (!recentMenu || recentMenu->actions().isEmpty()) {
        std::fprintf(stderr, "Recent image menu was not initialized\n");
        return 1;
    }
    QMainWindow hostWindow;
    auto* hostGlobal = new QAction(QStringLiteral("Host global"), &hostWindow);
    hostGlobal->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_9));
    hostGlobal->setShortcutContext(Qt::ApplicationShortcut);
    hostWindow.addAction(hostGlobal);
    auto* hostSave = new QAction(QStringLiteral("Host save"), &hostWindow);
    hostSave->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_S));
    hostSave->setShortcutContext(Qt::ApplicationShortcut);
    hostWindow.addAction(hostSave);
    hostWindow.show();
    window->activateWindow();
    QCoreApplication::processEvents();
    QCoreApplication::processEvents();
    if (hostSave->isEnabled()) {
        std::fprintf(stderr, "Host default shortcut was not suspended\n");
        return 1;
    }
    bool settingsDialogReady = false;
    QTimer::singleShot(0, window, [&settingsDialogReady] {
        for (QWidget* widget : QApplication::topLevelWidgets()) {
            auto* dialog = qobject_cast<QDialog*>(widget);
            if (!dialog || dialog->objectName() !=
                                   QStringLiteral("AnnotationSettingsDialog"))
                continue;
            auto* radius = dialog->findChild<QSpinBox*>(
                    QStringLiteral("AnnotationSettingsBrushRadius"));
            auto* opacity = dialog->findChild<QSlider*>(
                    QStringLiteral("AnnotationSettingsMaskOpacity"));
            auto* showScores = dialog->findChild<QCheckBox*>(
                    QStringLiteral("AnnotationSettingsShowScores"));
            auto* fileEditable = dialog->findChild<QCheckBox*>(
                    QStringLiteral("AnnotationSettingsFileListEditable"));
            auto* shortcuts = dialog->findChild<QTableWidget*>(
                    QStringLiteral("AnnotationShortcutTable"));
            auto* navigation = dialog->findChild<QListWidget*>(
                    QStringLiteral("AnnotationSettingsNavigation"));
            auto* saveShortcut = dialog->findChild<QKeySequenceEdit*>(
                    QStringLiteral("AnnotationShortcutEditor_save"));
            auto* buttons = dialog->findChild<QDialogButtonBox*>();
            if (!radius || !opacity || !showScores || !fileEditable ||
                fileEditable->isChecked() || !showScores->isChecked() ||
                !shortcuts || shortcuts->rowCount() == 0 || !navigation ||
                !saveShortcut || !buttons || navigation->count() != 4 ||
                navigation->currentItem()->text() !=
                        QStringLiteral("Shortcuts")) {
                dialog->reject();
                return;
            }
            saveShortcut->setKeySequence(
                    QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_9));
            if (buttons->button(QDialogButtonBox::Ok)->isEnabled()) {
                std::fprintf(
                        stderr,
                        "Host global shortcut conflict was not detected\n");
                dialog->reject();
                return;
            }
            saveShortcut->setKeySequence(
                    QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_8));
            if (!buttons->button(QDialogButtonBox::Ok)->isEnabled()) {
                std::fprintf(stderr, "Unique shortcut remained blocked\n");
                dialog->reject();
                return;
            }
            const QByteArray screenshot =
                    qgetenv("QANYLABELING_SETTINGS_SCREENSHOT");
            if (!screenshot.isEmpty())
                dialog->grab().save(QString::fromLocal8Bit(screenshot));
            settingsDialogReady = true;
            radius->setValue(33);
            opacity->setValue(61);
            showScores->setChecked(false);
            fileEditable->setChecked(true);
            buttons->button(QDialogButtonBox::Ok)->click();
        }
    });
    settingsAction->trigger();
    auto* configuredRadius = window->findChild<QSpinBox*>(
            QStringLiteral("AnnotationBrushRadius"));
    auto* configuredOpacity =
            window->findChild<QSlider*>(QStringLiteral("AnnotationOpacity"));
    if (!settingsDialogReady || !configuredRadius ||
        configuredRadius->value() != 33 || !configuredOpacity ||
        configuredOpacity->value() != 100 ||
        QSettings().value(QStringLiteral("qAnyLabeling/brushRadius")) != 33 ||
        QSettings().value(QStringLiteral("qAnyLabeling/maskOpacity")) != 61 ||
        QSettings().value(QStringLiteral("qAnyLabeling/showScores")).toBool() ||
        !QSettings()
                 .value(QStringLiteral("qAnyLabeling/fileListCheckboxEditable"))
                 .toBool() ||
        save->shortcut() != QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_8) ||
        QSettings().value(QStringLiteral("qAnyLabeling/shortcuts/save")) !=
                QStringLiteral("Ctrl+Alt+8") ||
        !hostSave->isEnabled()) {
        std::fprintf(stderr,
                     "Settings dialog did not apply persisted values\n");
        return 1;
    }
    QAction* showScoresAction = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->text() == QStringLiteral("Show Scores")) {
            showScoresAction = action;
            break;
        }
    }
    if (!showScoresAction || showScoresAction->isChecked()) {
        std::fprintf(stderr, "Canvas display action ignored settings\n");
        return 1;
    }
    showScoresAction->setChecked(true);
    if (!QSettings()
                 .value(QStringLiteral("qAnyLabeling/showScores"))
                 .toBool()) {
        std::fprintf(stderr, "Canvas display action was not persisted\n");
        return 1;
    }
    bool restoredDisplayState = false;
    QTimer::singleShot(0, window, [&restoredDisplayState] {
        for (QWidget* widget : QApplication::topLevelWidgets()) {
            auto* dialog = qobject_cast<QDialog*>(widget);
            if (!dialog || dialog->objectName() !=
                                   QStringLiteral("AnnotationSettingsDialog"))
                continue;
            auto* showScores = dialog->findChild<QCheckBox*>(
                    QStringLiteral("AnnotationSettingsShowScores"));
            restoredDisplayState = showScores && showScores->isChecked();
            dialog->reject();
        }
    });
    settingsAction->trigger();
    if (!restoredDisplayState) {
        std::fprintf(stderr, "Canvas display setting was not restored\n");
        return 1;
    }
    auto* flagList = window->findChild<QListWidget*>(
            QStringLiteral("AnnotationImageFlags"));
    auto* flagsDock =
            window->findChild<QDockWidget*>(QStringLiteral("ImageFlags"));
    if (!flagList || !flagsDock || !flagsDock->isHidden() ||
        window->findChild<QToolButton*>(
                QStringLiteral("AnnotationAddImageFlag")) ||
        window->findChild<QToolButton*>(
                QStringLiteral("AnnotationRemoveImageFlag")))
        return 1;
    const QString imageFlagsPath =
            directory.path() + QStringLiteral("/image_flags.txt");
    QFile imageFlagsFile(imageFlagsPath);
    if (!imageFlagsFile.open(QIODevice::WriteOnly) ||
        imageFlagsFile.write("occluded\nverified\n") !=
                QByteArrayLiteral("occluded\nverified\n").size())
        return 1;
    imageFlagsFile.close();
    QTimer::singleShot(0, window, [imageFlagsPath] {
        for (QWidget* widget : QApplication::topLevelWidgets()) {
            auto* dialog = qobject_cast<QFileDialog*>(widget);
            if (!dialog) continue;
            dialog->selectFile(imageFlagsPath);
            QMetaObject::invokeMethod(dialog, "accept", Qt::DirectConnection);
        }
    });
    imageFlagsAction->trigger();
    if (!flagsDock->isVisible() || flagList->count() != 2 ||
        flagList->item(0)->text() != QStringLiteral("occluded") ||
        flagList->item(1)->text() != QStringLiteral("verified") ||
        flagList->item(0)->checkState() != Qt::Unchecked ||
        flagList->item(1)->checkState() != Qt::Unchecked) {
        std::fprintf(stderr, "Image flag template import failed\n");
        return 1;
    }
    flagList->item(0)->setCheckState(Qt::Checked);
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const bool storedFlag = QJsonDocument::fromJson(sidecar.readAll())
                                    .object()
                                    .value(QStringLiteral("flags"))
                                    .toObject()
                                    .value(QStringLiteral("occluded"))
                                    .toBool();
    sidecar.close();
    if (!storedFlag) {
        std::fprintf(stderr, "Checked image flag was not saved\n");
        return 1;
    }
    // Classification is a view of the main document flags: category changes
    // must preserve geometry and unrelated sidecar fields under auto-save.
    auto* classifierAction = window->findChild<QAction*>(
            QStringLiteral("AnnotationImageClassifierAction"));
    bool classifierReady = false;
    int classifierStage = 0;
    QTimer::singleShot(
            0, window,
            [&classifierReady, &classifierStage, save, window, &directory]() {
                classifierStage = 1;
                auto* files = window->findChild<QListWidget*>(
                        QStringLiteral("AnnotationFiles"));
                auto* dialog = qobject_cast<QDialog*>(
                        QApplication::activeModalWidget());
                if (!dialog ||
                    dialog->objectName() !=
                            QStringLiteral("AnnotationImageClassifier"))
                    return;
                auto* multi = dialog->findChild<QPushButton*>(
                        QStringLiteral("ClassifierMultiClass"));
                auto* many = dialog->findChild<QPushButton*>(
                        QStringLiteral("ClassifierMultiLabel"));
                auto* search = dialog->findChild<QLineEdit*>(
                        QStringLiteral("ClassifierSearch"));
                const auto categories = dialog->findChildren<QCheckBox*>(
                        QStringLiteral("ClassifierCategory"));
                if (!multi || !multi->isChecked() || !many || !search ||
                    categories.size() != 2) {
                    dialog->reject();
                    return;
                }
                classifierStage = 2;
                categories[1]->setChecked(true);
                if (categories[0]->isChecked() || !categories[1]->isChecked()) {
                    dialog->reject();
                    return;
                }
                classifierStage = 3;
                many->click();
                categories[0]->setChecked(true);
                if (!categories[0]->isChecked() ||
                    !categories[1]->isChecked()) {
                    dialog->reject();
                    return;
                }
                classifierStage = 4;
                search->setText(QStringLiteral("verified missing"));
                if (!categories[0]->isHidden() || !categories[1]->isHidden()) {
                    dialog->reject();
                    return;
                }
                search->setText(QStringLiteral("verified"));
                if (!categories[0]->isHidden() || categories[1]->isHidden()) {
                    dialog->reject();
                    return;
                }
                search->clear();
                classifierStage = 5;
                save->trigger();
                auto* page = dialog->findChild<QSpinBox*>(
                        QStringLiteral("ClassifierPage"));
                if (!page || files->count() < 2) {
                    dialog->reject();
                    return;
                }
                const int originalPage = page->value();
                classifierStage = 6;
                page->setValue(originalPage == 1 ? 2 : 1);
                if (files->currentRow() + 1 == originalPage) {
                    dialog->reject();
                    return;
                }
                classifierStage = 7;
                page->setValue(originalPage);
                if (files->currentRow() + 1 != originalPage ||
                    !categories[0]->isChecked() ||
                    !categories[1]->isChecked()) {
                    dialog->reject();
                    return;
                }
                QTimer::singleShot(0, dialog, [] {
                    if (auto* question = qobject_cast<QMessageBox*>(
                                QApplication::activeModalWidget()))
                        question->button(QMessageBox::Yes)->click();
                });
                classifierStage = 8;
                multi->click();
                if (!multi->isChecked()) {
                    dialog->reject();
                    return;
                }
                classifierStage = 9;
                const QString exportPath =
                        directory.path() + QStringLiteral("/classified");
                if (!QDir().mkpath(exportPath)) {
                    dialog->reject();
                    return;
                }
                QTimer::singleShot(0, dialog, [exportPath, dialog] {
                    if (auto* chooser = qobject_cast<QFileDialog*>(
                                QApplication::activeModalWidget())) {
                        chooser->setDirectory(exportPath);
                        chooser->selectFile(exportPath);
                        QTimer::singleShot(0, dialog, [] {
                            if (auto* message = qobject_cast<QMessageBox*>(
                                        QApplication::activeModalWidget()))
                                message->accept();
                        });
                        QMetaObject::invokeMethod(chooser, "accept",
                                                  Qt::DirectConnection);
                    }
                });
                auto* exportButton = dialog->findChild<QPushButton*>(
                        QStringLiteral("ClassifierExport"));
                if (!exportButton) {
                    dialog->reject();
                    return;
                }
                classifierStage = 10;
                exportButton->click();
                classifierStage = 11;
                const QString sourceName = files->currentItem()->text();
                if (!QFileInfo::exists(exportPath +
                                       QStringLiteral("/occluded/") +
                                       sourceName)) {
                    dialog->reject();
                    return;
                }
                classifierReady = true;
                dialog->accept();
            });
    if (!classifierAction || !classifierAction->isEnabled()) return 1;
    classifierAction->trigger();
    if (!classifierReady) {
        std::fprintf(
                stderr,
                "Image Classifier interaction contract failed at stage %d\n",
                classifierStage);
        return 1;
    }
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonObject classifiedDocument =
            QJsonDocument::fromJson(sidecar.readAll()).object();
    sidecar.close();
    const QJsonObject classifiedFlags =
            classifiedDocument.value(QStringLiteral("flags")).toObject();
    if (!classifiedFlags.value(QStringLiteral("occluded")).toBool() ||
        !classifiedFlags.value(QStringLiteral("verified")).toBool() ||
        classifiedDocument.value(QStringLiteral("shapes"))
                .toArray()
                .isEmpty()) {
        std::fprintf(stderr,
                     "Classifier must save flags without losing shapes\n");
        return 1;
    }
    if (!deleteLabelFile->isEnabled()) {
        std::fprintf(stderr, "Delete File did not follow the saved sidecar\n");
        return 1;
    }
    saveWithImageData->setChecked(true);
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonValue embeddedImage =
            QJsonDocument::fromJson(sidecar.readAll())
                    .object()
                    .value(QStringLiteral("imageData"));
    sidecar.close();
    saveWithImageData->setChecked(false);
    if (!embeddedImage.isString() || embeddedImage.toString().isEmpty()) {
        std::fprintf(stderr, "Embedded image data was not saved\n");
        return 1;
    }
    auto* tags = window->findChild<QListWidget*>(QStringLiteral("ImageTags"));
    auto* tagInput =
            window->findChild<QLineEdit*>(QStringLiteral("ImageTagInput"));
    auto* addTag =
            window->findChild<QPushButton*>(QStringLiteral("AddImageTag"));
    auto* editTag =
            window->findChild<QPushButton*>(QStringLiteral("EditImageTag"));
    auto* deleteTags =
            window->findChild<QPushButton*>(QStringLiteral("DeleteImageTags"));
    auto* copyTags =
            window->findChild<QPushButton*>(QStringLiteral("CopyImageTags"));
    auto* batchTags = window->findChild<QPushButton*>(
            QStringLiteral("BatchDeleteImageTags"));
    if (!tags || !tagInput || !addTag || !editTag || !deleteTags || !copyTags ||
        !batchTags || !imageTags->isChecked() || !tags->isVisible() ||
        imageTags->shortcut() !=
                QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_T) ||
        tags->count() != 2 ||
        tags->item(0)->text() != QStringLiteral("machine") ||
        tags->item(1)->text() != QStringLiteral("camera")) {
        std::fprintf(stderr,
                     "Image tag panel failed: tags=%p input=%p add=%p edit=%p "
                     "delete=%p copy=%p checked=%d visible=%d shortcut=%s "
                     "count=%d\n",
                     static_cast<void*>(tags), static_cast<void*>(tagInput),
                     static_cast<void*>(addTag), static_cast<void*>(editTag),
                     static_cast<void*>(deleteTags),
                     static_cast<void*>(copyTags), imageTags->isChecked(),
                     tags && tags->isVisible(),
                     imageTags->shortcut().toString().toLocal8Bit().constData(),
                     tags ? tags->count() : -1);
        return 1;
    }
    addTag->click();
    tagInput->setText(QStringLiteral("  inspected  "));
    addTag->click();
    if (tags->count() != 3 ||
        tags->item(2)->text() != QStringLiteral("inspected")) {
        std::fprintf(stderr, "Image tag add failed: count=%d\n", tags->count());
        return 1;
    }
    tags->setCurrentRow(2);
    editTag->click();
    auto* tagEditor = tags->findChild<QLineEdit*>();
    if (!tagEditor) {
        std::fprintf(stderr, "Image tag inline editor did not open\n");
        return 1;
    }
    tagEditor->setText(QStringLiteral("reviewed"));
    QKeyEvent acceptTag(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
    QApplication::sendEvent(tagEditor, &acceptTag);
    QApplication::processEvents();
    if (tags->count() != 3 ||
        tags->item(2)->text() != QStringLiteral("reviewed")) {
        std::fprintf(stderr, "Image tag rename failed\n");
        return 1;
    }
    copyTags->click();
    if (QApplication::clipboard()->text() !=
        QStringLiteral("machine,camera,reviewed")) {
        std::fprintf(stderr, "Image tag copy failed\n");
        return 1;
    }
    batchTags->click();
    if (addTag->isEnabled() || batchTags->text() != QStringLiteral("x")) {
        std::fprintf(stderr, "Image tag batch mode did not start\n");
        return 1;
    }
    tags->item(0)->setSelected(true);
    if (!tags->item(0)->font().strikeOut()) {
        std::fprintf(stderr, "Image tag batch selection was not rendered\n");
        return 1;
    }
    batchTags->click();
    if (!addTag->isEnabled() || batchTags->text() != QStringLiteral("-")) {
        std::fprintf(stderr, "Image tag batch mode did not cancel\n");
        return 1;
    }
    tags->clearSelection();
    tags->setCurrentItem(tags->item(0));
    tags->item(0)->setSelected(true);
    QTimer::singleShot(0, window, [] {
        for (QWidget* widget : QApplication::topLevelWidgets()) {
            if (auto* dialog = qobject_cast<QMessageBox*>(widget))
                dialog->button(QMessageBox::Yes)->click();
        }
    });
    deleteTags->click();
    if (tags->count() != 2 ||
        tags->item(0)->text() != QStringLiteral("camera")) {
        std::fprintf(stderr, "Image tag delete failed: count=%d\n",
                     tags->count());
        return 1;
    }
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonObject tagged =
            QJsonDocument::fromJson(sidecar.readAll()).object();
    sidecar.close();
    if (tagged.value(QStringLiteral("tags")).toArray() !=
                QJsonArray{QStringLiteral("camera"),
                           QStringLiteral("reviewed")} ||
        tagged.value(QStringLiteral("custom_field")).toString() !=
                QStringLiteral("keep")) {
        std::fprintf(stderr, "Image tag save failed\n");
        return 1;
    }
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, imagePath)) ||
        tags->count() != 2 ||
        tags->item(1)->text() != QStringLiteral("reviewed")) {
        std::fprintf(stderr, "Image tag reopen failed\n");
        return 1;
    }
    addTag->click();
    if (tagInput->text() != QStringLiteral("tag") || tagInput->isHidden()) {
        std::fprintf(stderr, "Image tag default value failed\n");
        return 1;
    }
    QKeyEvent escape(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
    QApplication::sendEvent(tagInput, &escape);
    if (!tagInput->isHidden() || tags->count() != 2) {
        std::fprintf(stderr, "Image tag Escape cancel failed\n");
        return 1;
    }
    addTag->click();
    tagInput->setText(QStringLiteral("pending-switch"));
    QTimer::singleShot(0, window, [] {
        for (QWidget* widget : QApplication::topLevelWidgets()) {
            if (auto* dialog = qobject_cast<QMessageBox*>(widget))
                dialog->button(QMessageBox::Save)->click();
        }
    });
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, checkedImagePath)) ||
        !tagInput->isHidden() || tags->count() != 0 || imageTags->isChecked()) {
        std::fprintf(stderr, "Image tag switch failed\n");
        return 1;
    }
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonArray switchedTags = QJsonDocument::fromJson(sidecar.readAll())
                                            .object()
                                            .value(QStringLiteral("tags"))
                                            .toArray();
    sidecar.close();
    if (switchedTags != QJsonArray{QStringLiteral("camera"),
                                   QStringLiteral("reviewed"),
                                   QStringLiteral("pending-switch")}) {
        std::fprintf(stderr, "Pending image tag was lost on image switch\n");
        return 1;
    }
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, imagePath)) ||
        tags->count() != 3 || !imageTags->isChecked()) {
        std::fprintf(stderr, "Image tag switch-back failed\n");
        return 1;
    }
    QListWidgetItem* reorderedTag = tags->takeItem(0);
    tags->insertItem(3, reorderedTag);
    QApplication::processEvents();
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonArray reorderedTags = QJsonDocument::fromJson(sidecar.readAll())
                                             .object()
                                             .value(QStringLiteral("tags"))
                                             .toArray();
    sidecar.close();
    if (reorderedTags != QJsonArray{QStringLiteral("reviewed"),
                                    QStringLiteral("pending-switch"),
                                    QStringLiteral("camera")}) {
        std::fprintf(stderr, "Image tag reorder was not saved\n");
        return 1;
    }
    if (editLabel->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_E) ||
        editLabel->shortcutContext() != Qt::WindowShortcut ||
        labelManager->shortcut() != QKeySequence(Qt::ALT + Qt::Key_L) ||
        labelManager->shortcutContext() != Qt::WindowShortcut ||
        openImage->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_I) ||
        openVideo->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_O) ||
        openFolder->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_U) ||
        autoLabel->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_B) ||
        duplicate->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_D) ||
        hideSelected->shortcut() != QKeySequence(Qt::Key_S) ||
        showHidden->shortcut() != QKeySequence(Qt::Key_W) ||
        polygonTool->shortcut() != QKeySequence(Qt::Key_P) ||
        brushTool->shortcut() != QKeySequence(Qt::SHIFT + Qt::Key_B) ||
        fitImage->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_F) ||
        compareView->shortcut() !=
                QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_C) ||
        saveAs->shortcut() != QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_S) ||
        closeImage->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_W) ||
        deleteLabelFile->shortcut() !=
                QKeySequence(Qt::CTRL + Qt::Key_Delete) ||
        deleteImageFile->shortcut() !=
                QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_Delete) ||
        autoLabelPanel->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_A) ||
        zoomOriginal->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_Equal) ||
        fitWidth->shortcut() !=
                QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_F) ||
        showShapes->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_H) ||
        showLabels->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_L) ||
        autoUseLastLabel->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_Y) ||
        createBrush->shortcut() != QKeySequence(Qt::CTRL + Qt::Key_N) ||
        polygonTool->shortcutContext() != Qt::WindowShortcut ||
        brushTool->shortcutContext() != Qt::WindowShortcut ||
        openVideo->shortcutContext() != Qt::WindowShortcut ||
        autoUseLastLabel->shortcutContext() != Qt::WindowShortcut ||
        createBrush->shortcutContext() != Qt::WindowShortcut ||
        duplicate->shortcutContext() != Qt::WindowShortcut ||
        hideSelected->shortcutContext() != Qt::WindowShortcut ||
        showHidden->shortcutContext() != Qt::WindowShortcut) {
        std::fprintf(stderr, "Source-compatible annotation shortcuts failed\n");
        return 1;
    }
    auto* objects = window->findChild<QListWidget*>(
            QStringLiteral("AnnotationObjects"));
    auto* toolbarScroll = window->findChild<QScrollArea*>(
            QStringLiteral("AnnotationToolbarScrollArea"));
    auto* canvasScroll = window->findChild<QScrollArea*>(
            QStringLiteral("AnnotationCanvasScrollArea"));
    auto* workspaceSplitter = window->findChild<QSplitter*>(
            QStringLiteral("AnnotationWorkspaceSplitter"));
    auto* rightSidebar = window->findChild<QWidget*>(
            QStringLiteral("AnnotationRightSidebar"));
    auto* sidebarToggle = window->findChild<QToolButton*>(
            QStringLiteral("AnnotationSidebarToggle"));
    auto* annotationToolbar =
            window->findChild<QToolBar*>(QStringLiteral("AnnotationToolbar"));
    if (!toolbarScroll || toolbarScroll->width() != 46 || !canvasScroll ||
        !workspaceSplitter || !rightSidebar || !sidebarToggle ||
        !rightSidebar->isVisible() || sidebarToggle->width() != 26 ||
        !annotationToolbar) {
        std::fprintf(stderr,
                     "Upstream workspace layout failed to initialize\n");
        return 1;
    }
    QStringList toolbarIds;
    for (QAction* action : annotationToolbar->actions()) {
        const QVariant id = action->property("qAnyToolbarId");
        if (id.isValid()) toolbarIds.append(id.toString());
    }
    const QStringList upstreamToolbarIds = {
            QStringLiteral("open_dir"),
            QStringLiteral("open_next"),
            QStringLiteral("open_prev"),
            QStringLiteral("save"),
            QStringLiteral("delete_file"),
            QStringLiteral("tool_polygon"),
            QStringLiteral("create_brush_polygon"),
            QStringLiteral("tool_magic_wand"),
            QStringLiteral("tool_rectangle"),
            QStringLiteral("tool_cuboid"),
            QStringLiteral("tool_rotation"),
            QStringLiteral("tool_quadrilateral"),
            QStringLiteral("tool_circle"),
            QStringLiteral("tool_line"),
            QStringLiteral("tool_point"),
            QStringLiteral("tool_linestrip"),
            QStringLiteral("edit_brush_mode"),
            QStringLiteral("tool_edit"),
            QStringLiteral("delete_polygon"),
            QStringLiteral("undo"),
            QStringLiteral("loop_thru_labels"),
            QStringLiteral("loop_select_labels"),
            QStringLiteral("select_toggle_shapes"),
            QStringLiteral("auto_run"),
            QStringLiteral("auto_label"),
            QStringLiteral("open_chatbot"),
            QStringLiteral("open_vqa"),
            QStringLiteral("open_pointcloud"),
            QStringLiteral("open_image_classifier"),
            QStringLiteral("open_video_classifier"),
            QStringLiteral("open_paddleocr"),
            QStringLiteral("fit_width")};
    if (toolbarIds.size() < upstreamToolbarIds.size() ||
        toolbarIds.mid(0, upstreamToolbarIds.size()) != upstreamToolbarIds) {
        std::fprintf(stderr, "Upstream toolbar action order drifted: %s\n",
                     toolbarIds.join(QStringLiteral(", "))
                             .toLocal8Bit()
                             .constData());
        return 1;
    }
    for (const QString& id : upstreamToolbarIds) {
        bool found = false;
        for (QAction* action : annotationToolbar->actions()) {
            if (action->property("qAnyToolbarId").toString() != id) continue;
            found = true;
            if (action->icon().isNull()) {
                std::fprintf(stderr, "Upstream toolbar icon missing: %s\n",
                             id.toLocal8Bit().constData());
                return 1;
            }
            break;
        }
        if (!found) {
            std::fprintf(stderr, "Upstream toolbar action missing: %s\n",
                         id.toLocal8Bit().constData());
            return 1;
        }
    }
    auto* zoomWidget = window->findChild<QSpinBox*>(
            QStringLiteral("AnnotationZoomWidget"));
    if (!zoomWidget || zoomWidget->minimum() != 1 ||
        zoomWidget->maximum() != 1000 ||
        zoomWidget->suffix() != QStringLiteral("%")) {
        std::fprintf(stderr, "Source zoom widget contract failed\n");
        return 1;
    }
    sidebarToggle->click();
    QCoreApplication::processEvents();
    if (rightSidebar->isVisible()) {
        std::fprintf(stderr, "Right sidebar toggle did not hide the sidebar\n");
        return 1;
    }
    sidebarToggle->click();
    QCoreApplication::processEvents();
    if (!rightSidebar->isVisible()) {
        std::fprintf(stderr,
                     "Right sidebar toggle did not restore the sidebar\n");
        return 1;
    }
    if (!objects ||
        objects->selectionMode() != QAbstractItemView::ExtendedSelection ||
        objects->count() != 1 ||
        !objects->item(0)->text().contains(QStringLiteral("[7]"))) {
        std::fprintf(stderr, "Grouped object list failed to initialize\n");
        return 1;
    }
    objects->setCurrentRow(0);
    auto* description = window->findChild<QPlainTextEdit*>(
            QStringLiteral("AnnotationDescription"));
    auto* settingsButton = window->findChild<QToolButton*>(
            QStringLiteral("AnnotationSettingsButton"));
    if (!description || !settingsButton || !settingsButton->isVisible() ||
        !description->isEnabled()) {
        std::fprintf(stderr, "Description or settings panel missing\n");
        return 1;
    }
    description->setPlainText(QStringLiteral("First object description"));
    description->moveCursor(QTextCursor::End);
    description->insertPlainText(QStringLiteral(" continued"));
    if (description->toPlainText() !=
                QStringLiteral("First object description continued") ||
        objects->selectedItems().size() != 1) {
        std::fprintf(stderr, "Object description edit lost its selection\n");
        return 1;
    }
    objects->clearSelection();
    description->setPlainText(QStringLiteral("Image description"));
    objects->setCurrentRow(0);
    objects->item(0)->setSelected(true);
    if (description->toPlainText() !=
        QStringLiteral("First object description continued")) {
        std::fprintf(stderr, "Object and image descriptions were mixed\n");
        return 1;
    }
    brushTarget->setEditText(QStringLiteral("new-mask"));
    createBrush->trigger();
    QCoreApplication::processEvents();
    if (!brushTool->isChecked() || !objects->selectedItems().isEmpty()) {
        std::fprintf(stderr, "New brush polygon edited the selected object\n");
        return 1;
    }
    brushTool->setChecked(false);
    openVideo->trigger();
    QCoreApplication::processEvents();
    QDialog* embeddedVideo = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if (widget->windowTitle() == QStringLiteral("SAM3 video tracking") &&
            widget->isVisible())
            embeddedVideo = qobject_cast<QDialog*>(widget);
    }
    if (!embeddedVideo || embeddedVideo == videoDialog) {
        std::fprintf(stderr, "Image workspace video tracking did not open\n");
        return 1;
    }
    auto* videoModel = embeddedVideo->findChild<QComboBox*>(
            QStringLiteral("SAM3VideoModel"));
    auto* videoDevice = embeddedVideo->findChild<QComboBox*>(
            QStringLiteral("SAM3VideoDevice"));
    auto* videoBackend = embeddedVideo->findChild<QLabel*>(
            QStringLiteral("SAM3VideoBackend"));
    if (!videoModel || videoModel->count() == 0 || !videoDevice ||
        !videoDevice->isVisible() || videoDevice->count() == 0 ||
        !videoBackend || !videoBackend->isVisible()) {
        std::fprintf(
                stderr,
                "Standalone video model/device/backend controls missing\n");
        return 1;
    }
    for (int i = 0; i < videoDevice->count(); ++i) {
        if (videoDevice->itemData(i).toString().isEmpty()) {
            std::fprintf(stderr, "Video device row has no AICore ID\n");
            return 1;
        }
    }
    embeddedVideo->close();
    objects->setCurrentRow(0);
    copy->trigger();
    paste->trigger();
    if (objects->count() != 2) {
        std::fprintf(stderr, "Object copy/paste failed\n");
        return 1;
    }
    objects->setCurrentRow(1);
    duplicate->trigger();
    // Canvas context-menu end-to-end fixture: exec() runs a nested loop that
    // a queued timer uses to inspect and close the popup.
    QWidget* canvasWidget = canvasScroll->widget();
    QTimer::singleShot(0, window, [window, canvasWidget]() {
        QWidget* popup = QApplication::activePopupWidget();
        QMenu* menu = popup ? qobject_cast<QMenu*>(popup) : nullptr;
        if (!menu || menu->actions().size() < 12) {
            std::fprintf(stderr, "Canvas menu fixture: popup missing\n");
            if (popup) popup->close();
            return;
        }
        const QList<QAction*> actions = menu->actions();
        if (!actions.at(0)->menu() ||
            actions.at(0)->menu()->title() !=
                    QStringLiteral("Filter by Label") ||
            !actions.at(1)->menu() ||
            actions.at(1)->menu()->title() !=
                    QStringLiteral("Filter by Group ID")) {
            std::fprintf(stderr,
                         "Canvas menu fixture: filter submenus missing\n");
            popup->close();
            return;
        }
        QMenu* labelMenu = actions.at(0)->menu();
        labelMenu->popup(QPoint(0, 0));
        const QList<QAction*> options = labelMenu->actions();
        bool hasAll = false, hasVehicle = false;
        for (QAction* option : options) {
            hasAll |= option->text() == QStringLiteral("All Labels");
            hasVehicle |= option->text() == QStringLiteral("vehicle");
        }
        if (!hasAll || !hasVehicle) {
            std::fprintf(stderr,
                         "Canvas menu fixture: label options missing\n");
            popup->close();
            return;
        }
        popup->close();
    });
    {
        QContextMenuEvent contextEvent(
                QContextMenuEvent::Mouse, QPoint(40, 40),
                canvasWidget->mapToGlobal(QPoint(40, 40)));
        QCoreApplication::sendEvent(canvasWidget, &contextEvent);
        QCoreApplication::processEvents();
    }
    if (QApplication::activePopupWidget()) {
        std::fprintf(stderr, "Canvas menu fixture: menu stayed open\n");
        if (QApplication::activePopupWidget())
            QApplication::activePopupWidget()->close();
        return 1;
    }
    // SAM tools keep the right click for negative prompts: no popup opens.
    QAction* samTool = nullptr;
    QAction* editTool = nullptr;
    for (QAction* action : window->findChildren<QAction*>()) {
        if (action->data().toString() == QStringLiteral("sam_point"))
            samTool = action;
        if (action->data().toString() == QStringLiteral("edit"))
            editTool = action;
    }
    if (!samTool || !editTool) return 1;
    samTool->trigger();
    QCoreApplication::processEvents();
    {
        QContextMenuEvent contextEvent(
                QContextMenuEvent::Mouse, QPoint(40, 40),
                canvasWidget->mapToGlobal(QPoint(40, 40)));
        QCoreApplication::sendEvent(canvasWidget, &contextEvent);
        QCoreApplication::processEvents();
    }
    const bool samSuppressed = QApplication::activePopupWidget() == nullptr;
    if (QApplication::activePopupWidget())
        QApplication::activePopupWidget()->close();
    editTool->trigger();
    QCoreApplication::processEvents();
    if (!samSuppressed) {
        std::fprintf(stderr, "Canvas menu must stay suppressed in SAM mode\n");
        return 1;
    }
    if (objects->count() != 3) {
        std::fprintf(stderr, "Object duplication failed\n");
        return 1;
    }
    hideSelected->trigger();
    if (window->findChild<QListWidget*>(QStringLiteral("AnnotationObjects"))
                ->count() != 3) {
        std::fprintf(stderr, "Object hiding changed object count\n");
        return 1;
    }
    showHidden->trigger();
    undo->trigger();
    undo->trigger();
    undo->trigger();
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonObject saved =
            QJsonDocument::fromJson(sidecar.readAll()).object();
    const QJsonArray savedShapes =
            saved.value(QStringLiteral("shapes")).toArray();
    if (saved.value(QStringLiteral("custom_field")).toString() !=
                QStringLiteral("keep") ||
        savedShapes.size() != 2 ||
        savedShapes[0].toObject().value(QStringLiteral("score")).toDouble() !=
                0.92 ||
        savedShapes[0].toObject().value(QStringLiteral("group_id")).toInt() !=
                7 ||
        savedShapes[1].toObject().value(QStringLiteral("group_id")).toInt() ==
                7 ||
        savedShapes[1].toObject()
                        .value(QStringLiteral("points"))
                        .toArray()[0]
                        .toArray()[0]
                        .toInt() != 11 ||
        saved.value(QStringLiteral("imageWidth")).toInt() != 64) {
        std::fprintf(stderr, "Annotation sidecar round-trip failed\n");
        return 1;
    }
    sidecar.close();
    objects->item(0)->setSelected(true);
    objects->item(1)->setSelected(true);
    group->trigger();
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonArray grouped = QJsonDocument::fromJson(sidecar.readAll())
                                       .object()
                                       .value(QStringLiteral("shapes"))
                                       .toArray();
    sidecar.close();
    if (grouped.size() != 2 ||
        grouped[0].toObject().value(QStringLiteral("group_id")).toInt() != 7 ||
        grouped[1].toObject().value(QStringLiteral("group_id")).toInt() != 7) {
        std::fprintf(stderr, "Grouping did not merge selected group IDs\n");
        return 1;
    }
    objects->item(0)->setSelected(true);
    ungroup->trigger();
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonArray ungrouped = QJsonDocument::fromJson(sidecar.readAll())
                                         .object()
                                         .value(QStringLiteral("shapes"))
                                         .toArray();
    sidecar.close();
    if (ungrouped.size() != 2 ||
        !ungrouped[0].toObject().value(QStringLiteral("group_id")).isNull() ||
        !ungrouped[1].toObject().value(QStringLiteral("group_id")).isNull()) {
        std::fprintf(stderr,
                     "Ungrouping did not clear the entire selected group\n");
        return 1;
    }
    objects->item(0)->setSelected(true);
    objects->item(1)->setSelected(true);
    objectUnion->trigger();
    save->trigger();
    if (!sidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonArray merged = QJsonDocument::fromJson(sidecar.readAll())
                                      .object()
                                      .value(QStringLiteral("shapes"))
                                      .toArray();
    sidecar.close();
    if (merged.size() != 1 ||
        merged[0].toObject().value(QStringLiteral("shape_type")).toString() !=
                QStringLiteral("rectangle") ||
        merged[0].toObject()
                        .value(QStringLiteral("points"))
                        .toArray()[0]
                        .toArray() != QJsonArray{1, 2} ||
        merged[0].toObject()
                        .value(QStringLiteral("points"))
                        .toArray()[2]
                        .toArray() != QJsonArray{30, 40}) {
        std::fprintf(stderr, "Merged JSON: %s\n",
                     QJsonDocument(merged)
                             .toJson(QJsonDocument::Compact)
                             .constData());
        std::fprintf(stderr,
                     "Rectangle union did not preserve the combined bounds\n");
        return 1;
    }
    undo->trigger();
    if (objects->count() != 2) {
        std::fprintf(stderr, "Undo did not restore merged objects\n");
        return 1;
    }
    save->trigger();

    const QString polygonPath =
            directory.path() + QStringLiteral("/polygons.png");
    if (!image.save(polygonPath)) return 1;
    QJsonObject polygonA{
            {QStringLiteral("label"), QStringLiteral("area")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{5, 5}, QJsonArray{20, 5}, QJsonArray{20, 20},
                        QJsonArray{5, 20}}}};
    QJsonObject polygonB{
            {QStringLiteral("label"), QStringLiteral("area")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{15, 15}, QJsonArray{30, 15},
                        QJsonArray{30, 30}, QJsonArray{15, 30}}}};
    QFile polygonSidecar(directory.path() + QStringLiteral("/polygons.json"));
    if (!polygonSidecar.open(QIODevice::WriteOnly) ||
        polygonSidecar.write(
                QJsonDocument(QJsonObject{{QStringLiteral("shapes"),
                                           QJsonArray{polygonA, polygonB}}})
                        .toJson()) < 0)
        return 1;
    polygonSidecar.close();
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, polygonPath)) ||
        objects->count() != 2)
        return 1;
    recentMenu->aboutToShow();
    bool recentSample = false;
    for (QAction* action : recentMenu->actions())
        recentSample |= action->text().contains(QStringLiteral("sample.png"));
    if (!recentSample) {
        std::fprintf(stderr, "Recent image was not listed after navigation\n");
        return 1;
    }
    objects->item(0)->setSelected(true);
    objects->item(1)->setSelected(true);
    objectUnion->trigger();
    save->trigger();
    if (!polygonSidecar.open(QIODevice::ReadOnly)) return 1;
    const QJsonArray polygonUnion =
            QJsonDocument::fromJson(polygonSidecar.readAll())
                    .object()
                    .value(QStringLiteral("shapes"))
                    .toArray();
    polygonSidecar.close();
    if (polygonUnion.size() != 1 ||
        polygonUnion[0].toObject()
                        .value(QStringLiteral("shape_type"))
                        .toString() != QStringLiteral("polygon") ||
        polygonUnion[0].toObject()
                        .value(QStringLiteral("points"))
                        .toArray()
                        .size() < 6) {
        std::fprintf(stderr,
                     "Overlapping polygon union did not produce one external "
                     "polygon\n");
        return 1;
    }
    auto* files =
            window->findChild<QListWidget*>(QStringLiteral("AnnotationFiles"));
    if (!files ||
        !QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, checkedImagePath)))
        return 1;
    auto* brightnessSlider =
            window->findChild<QSlider*>(QStringLiteral("AnnotationBrightness"));
    auto* contrastSlider =
            window->findChild<QSlider*>(QStringLiteral("AnnotationContrast"));
    if (!brightnessSlider || !contrastSlider) {
        std::fprintf(stderr, "Canvas adjustment sliders are not addressable\n");
        return 1;
    }
    keepPreviousBrightness->setChecked(true);
    keepPreviousContrast->setChecked(true);
    brightnessSlider->setValue(83);
    contrastSlider->setValue(67);
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, uncheckedImagePath)) ||
        brightnessSlider->value() != 83 || contrastSlider->value() != 67) {
        std::fprintf(stderr,
                     "Keep-previous display adjustments did not follow the "
                     "next image\n");
        return 1;
    }
    keepPreviousBrightness->setChecked(false);
    keepPreviousContrast->setChecked(false);
    nextUnchecked->trigger();
    if (!files ||
        files->currentItem()->text() != QStringLiteral("sample3.png")) {
        std::fprintf(stderr, "Unchecked image navigation failed\n");
        return 1;
    }
    models->setCurrentText(QStringLiteral("yoloe-26n-seg-pf"));
    bool rangeChecked = false;
    QTimer::singleShot(0, window, [&] {
        auto* dialog = window->findChild<QDialog*>(
                QStringLiteral("AnnotationBatchRangeDialog"));
        auto* from = dialog ? dialog->findChild<QSpinBox*>(
                                      QStringLiteral("AnnotationBatchFrom"))
                            : nullptr;
        auto* to = dialog ? dialog->findChild<QSpinBox*>(
                                    QStringLiteral("AnnotationBatchTo"))
                          : nullptr;
        if (from && to && from->value() == files->currentRow() + 1 &&
            to->value() == files->count()) {
            from->setValue(1);
            to->setValue(2);
            rangeChecked = to->value() == 2;
        }
        if (dialog) dialog->reject();
    });
    autoLabel->trigger();
    if (!rangeChecked || window->findChild<QDialog*>(
                                 QStringLiteral("AnnotationBatchProgress"))) {
        std::fprintf(stderr, "Auto Run range or cancellation failed\n");
        return 1;
    }
    const QString droppedImage =
            directory.path() + QStringLiteral("/dropped.png");
    if (!image.save(droppedImage)) return 1;
    QMimeData droppedMime;
    droppedMime.setUrls({QUrl::fromLocalFile(droppedImage)});
    QDragEnterEvent enter(QPoint(20, 20), Qt::CopyAction, &droppedMime,
                          Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(window->centralWidget(), &enter);
    if (!enter.isAccepted()) {
        std::fprintf(stderr, "Image drag was not accepted\n");
        return 1;
    }
    QDragMoveEvent dragMove(QPoint(20, 20), Qt::CopyAction, &droppedMime,
                            Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(window->centralWidget(), &dragMove);
    if (!dragMove.isAccepted()) return 1;
    QDropEvent drop(QPoint(20, 20), Qt::CopyAction, &droppedMime,
                    Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(window->centralWidget(), &drop);
    if (!drop.isAccepted() || !files->currentItem() ||
        files->currentItem()->text() != QStringLiteral("dropped.png")) {
        std::fprintf(
                stderr, "Dropped image did not open (accepted=%d current=%s)\n",
                int(drop.isAccepted()),
                files->currentItem()
                        ? files->currentItem()->text().toLocal8Bit().constData()
                        : "<none>");
        return 1;
    }
    QMimeData invalidMime;
    QFile unsupportedFile(directory.path() +
                          QStringLiteral("/unsupported.txt"));
    if (!unsupportedFile.open(QIODevice::WriteOnly)) return 1;
    unsupportedFile.close();
    invalidMime.setUrls({QUrl::fromLocalFile(unsupportedFile.fileName())});
    QDragEnterEvent invalidEnter(QPoint(20, 20), Qt::CopyAction, &invalidMime,
                                 Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(window->centralWidget(), &invalidEnter);
    if (invalidEnter.isAccepted()) {
        std::fprintf(stderr, "Unsupported drag was accepted\n");
        return 1;
    }
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, imagePath)))
        return 1;
    bool managerChecked = false;
    QTimer::singleShot(0, window, [&] {
        auto* dialog = window->findChild<QDialog*>(
                QStringLiteral("AnnotationLabelManager"));
        auto* table = dialog ? dialog->findChild<QTableWidget*>(
                                       QStringLiteral("AnnotationLabelTable"))
                             : nullptr;
        auto* from = dialog ? dialog->findChild<QSpinBox*>(
                                      QStringLiteral("AnnotationLabelFrom"))
                            : nullptr;
        auto* to = dialog ? dialog->findChild<QSpinBox*>(
                                    QStringLiteral("AnnotationLabelTo"))
                          : nullptr;
        auto* buttons =
                dialog ? dialog->findChild<QDialogButtonBox*>(
                                 QStringLiteral("AnnotationLabelButtons"))
                       : nullptr;
        if (table && from && to && buttons && table->columnCount() == 5 &&
            table->rowCount() >= 1) {
            int vehicleRow = -1;
            for (int row = 0; row < table->rowCount(); ++row)
                if (table->item(row, 0)->text() == QStringLiteral("vehicle"))
                    vehicleRow = row;
            if (vehicleRow >= 0) {
                auto* replacement = qobject_cast<QLineEdit*>(
                        table->cellWidget(vehicleRow, 2));
                auto* visible = table->cellWidget(vehicleRow, 3)
                                        ->findChild<QCheckBox*>();
                if (replacement && visible) {
                    replacement->setText(QStringLiteral("car"));
                    visible->setChecked(false);
                    for (int row = 0; row < files->count(); ++row)
                        if (files->item(row)->data(Qt::UserRole).toString() ==
                            imagePath) {
                            from->setValue(row + 1);
                            to->setValue(row + 1);
                        }
                    managerChecked = true;
                    buttons->button(QDialogButtonBox::Ok)->click();
                }
            }
        }
        if (dialog && !managerChecked) dialog->reject();
    });
    labelManager->trigger();
    if (!managerChecked || !sidecar.open(QIODevice::ReadOnly)) {
        std::fprintf(stderr, "Label manager did not apply changes\n");
        return 1;
    }
    const QJsonObject managed =
            QJsonDocument::fromJson(sidecar.readAll()).object();
    sidecar.close();
    const QJsonArray managedShapes =
            managed.value(QStringLiteral("shapes")).toArray();
    if (managed.value(QStringLiteral("custom_field")).toString() !=
                QStringLiteral("keep") ||
        managedShapes.size() != 2 ||
        managedShapes.at(0)
                        .toObject()
                        .value(QStringLiteral("label"))
                        .toString() != QStringLiteral("car") ||
        managedShapes.at(0)
                .toObject()
                .value(QStringLiteral("visible"))
                .toBool(true)) {
        std::fprintf(stderr,
                     "Label manager lost sidecar fields or visibility\n");
        return 1;
    }
    QImage ring(8, 8, QImage::Format_ARGB32);
    ring.fill(Qt::transparent);
    for (int y = 1; y < 7; ++y)
        for (int x = 1; x < 7; ++x)
            if (x < 3 || x > 4 || y < 3 || y > 4)
                ring.setPixelColor(x, y, QColor(255, 0, 0, 120));
    QByteArray ringBytes;
    QBuffer ringBuffer(&ringBytes);
    if (!ringBuffer.open(QIODevice::WriteOnly) ||
        !ring.save(&ringBuffer, "PNG"))
        return 1;
    QJsonObject ringShape{
            {QStringLiteral("label"), QStringLiteral("ring")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{1, 1}, QJsonArray{6, 1}, QJsonArray{6, 6},
                        QJsonArray{1, 6}}},
            {QStringLiteral("acloudviewer_mask_png"),
             QString::fromLatin1(ringBytes.toBase64())}};
    if (!checkedSidecar.open(QIODevice::WriteOnly | QIODevice::Truncate) ||
        checkedSidecar.write(
                QJsonDocument(QJsonObject{{QStringLiteral("checked"), true},
                                          {QStringLiteral("shapes"),
                                           QJsonArray{ringShape}}})
                        .toJson()) < 0)
        return 1;
    checkedSidecar.close();
    bool colorChanged = false;
    QTimer::singleShot(0, window, [&] {
        auto* dialog = window->findChild<QDialog*>(
                QStringLiteral("AnnotationLabelManager"));
        auto* table = dialog ? dialog->findChild<QTableWidget*>(
                                       QStringLiteral("AnnotationLabelTable"))
                             : nullptr;
        auto* from = dialog ? dialog->findChild<QSpinBox*>(
                                      QStringLiteral("AnnotationLabelFrom"))
                            : nullptr;
        auto* to = dialog ? dialog->findChild<QSpinBox*>(
                                    QStringLiteral("AnnotationLabelTo"))
                          : nullptr;
        auto* buttons =
                dialog ? dialog->findChild<QDialogButtonBox*>(
                                 QStringLiteral("AnnotationLabelButtons"))
                       : nullptr;
        if (!table || !from || !to || !buttons) {
            if (dialog) dialog->reject();
            return;
        }
        int ringRow = -1;
        for (int row = 0; row < table->rowCount(); ++row)
            if (table->item(row, 0)->text() == QStringLiteral("ring"))
                ringRow = row;
        auto* swatch = ringRow >= 0 ? qobject_cast<QPushButton*>(
                                              table->cellWidget(ringRow, 4))
                                    : nullptr;
        if (!swatch) {
            dialog->reject();
            return;
        }
        for (int row = 0; row < files->count(); ++row)
            if (files->item(row)->data(Qt::UserRole).toString() ==
                checkedImagePath) {
                from->setValue(row + 1);
                to->setValue(row + 1);
            }
        QTimer::singleShot(0, window, [&] {
            for (QWidget* widget : QApplication::topLevelWidgets()) {
                auto* picker = qobject_cast<QColorDialog*>(widget);
                if (!picker || !picker->isVisible()) continue;
                picker->setCurrentColor(QColor(12, 200, 45, 180));
                picker->accept();
                colorChanged = true;
                break;
            }
        });
        swatch->click();
        if (colorChanged)
            buttons->button(QDialogButtonBox::Ok)->click();
        else
            dialog->reject();
    });
    labelManager->trigger();
    if (!colorChanged || !checkedSidecar.open(QIODevice::ReadOnly)) {
        std::fprintf(stderr, "Label color picker did not apply\n");
        return 1;
    }
    const QJsonObject colorDocument =
            QJsonDocument::fromJson(checkedSidecar.readAll()).object();
    checkedSidecar.close();
    const QJsonObject coloredRing =
            colorDocument.value(QStringLiteral("shapes"))
                    .toArray()
                    .at(0)
                    .toObject();
    const QJsonArray rgba =
            coloredRing.value(QStringLiteral("acloudviewer_color")).toArray();
    const QImage coloredMask = QImage::fromData(
            QByteArray::fromBase64(
                    coloredRing.value(QStringLiteral("acloudviewer_mask_png"))
                            .toString()
                            .toLatin1()),
            "PNG");
    if (rgba != QJsonArray{12, 200, 45, 180} || coloredMask.isNull() ||
        coloredMask.pixelColor(1, 1) != QColor(12, 200, 45, 120) ||
        coloredMask.pixelColor(3, 3).alpha() != 0 ||
        colorDocument.value(QStringLiteral("checked")).toBool() != true) {
        std::fprintf(stderr,
                     "Label color change lost mask alpha or metadata\n");
        return 1;
    }
    const QByteArray screenshot = qgetenv("QANYLABELING_SCREENSHOT");
    if (!screenshot.isEmpty()) {
        models->setCurrentText(QStringLiteral("sam3"));
        samText->setText(QStringLiteral("bus"));
        window->grab().save(QString::fromLocal8Bit(screenshot));
    }
    // Exercise the real window snapshot/undo/redo and sidecar path, beyond
    // the geometry-only cuboid tests. Unknown metadata survives all edits.
    save->trigger();
    QImage cuboidImage(400, 400, QImage::Format_RGB32);
    cuboidImage.fill(Qt::black);
    const QString cuboidImagePath =
            directory.path() + QStringLiteral("/cuboid.png");
    if (!cuboidImage.save(cuboidImagePath)) return 1;
    const QJsonArray cuboidPoints{QJsonArray{80, 100},  QJsonArray{200, 100},
                                  QJsonArray{200, 240}, QJsonArray{80, 240},
                                  QJsonArray{120, 60},  QJsonArray{240, 60},
                                  QJsonArray{240, 200}, QJsonArray{120, 200}};
    const QJsonObject cuboidShape{
            {QStringLiteral("label"), QStringLiteral("car")},
            {QStringLiteral("shape_type"), QStringLiteral("cuboid")},
            {QStringLiteral("points"), cuboidPoints},
            {QStringLiteral("cuboid3d"),
             QJsonObject{{QStringLiteral("source"), QStringLiteral("fixture")},
                         {QStringLiteral("custom"), 42}}}};
    QFile cuboidSidecar(directory.path() + QStringLiteral("/cuboid.json"));
    if (!cuboidSidecar.open(QIODevice::WriteOnly) ||
        cuboidSidecar.write(
                QJsonDocument(QJsonObject{{QStringLiteral("shapes"),
                                           QJsonArray{cuboidShape}}})
                        .toJson()) < 0)
        return 1;
    cuboidSidecar.close();
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, cuboidImagePath)))
        return 1;
    editTool->trigger();
    QCoreApplication::processEvents();
    zoomOriginal->trigger();
    QCoreApplication::processEvents();
    const QPointF cuboidOrigin((canvasWidget->width() - 400) / 2.0,
                               (canvasWidget->height() - 400) / 2.0);
    const auto cuboidMouse = [&](QEvent::Type type, const QPointF& point) {
        QMouseEvent event(type, cuboidOrigin + point, Qt::LeftButton,
                          type == QEvent::MouseButtonRelease ? Qt::NoButton
                                                             : Qt::LeftButton,
                          Qt::NoModifier);
        QCoreApplication::sendEvent(canvasWidget, &event);
    };
    cuboidMouse(QEvent::MouseButtonPress, QPointF(80, 100));
    cuboidMouse(QEvent::MouseMove, QPointF(60, 80));
    cuboidMouse(QEvent::MouseButtonRelease, QPointF(60, 80));
    const auto savedCuboid = [&]() {
        save->trigger();
        if (!cuboidSidecar.open(QIODevice::ReadOnly)) return QJsonObject();
        const QJsonArray shapes =
                QJsonDocument::fromJson(cuboidSidecar.readAll())
                        .object()
                        .value(QStringLiteral("shapes"))
                        .toArray();
        cuboidSidecar.close();
        return shapes.size() == 1 ? shapes.first().toObject() : QJsonObject();
    };
    const QJsonObject editedCuboid = savedCuboid();
    const QJsonArray editedPoints =
            editedCuboid.value(QStringLiteral("points")).toArray();
    if (editedPoints.size() != 8 ||
        editedPoints[0].toArray() != QJsonArray{60, 80} ||
        editedPoints[1].toArray() != QJsonArray{200, 80}) {
        std::fprintf(stderr,
                     "Window cuboid drag/save failed: canvas=%dx%d shapes=%s\n",
                     canvasWidget->width(), canvasWidget->height(),
                     QJsonDocument(editedCuboid)
                             .toJson(QJsonDocument::Compact)
                             .constData());
        return 1;
    }
    undo->trigger();
    if (savedCuboid().value(QStringLiteral("points")).toArray() !=
        cuboidPoints) {
        std::fprintf(stderr,
                     "Window cuboid undo did not restore all eight points\n");
        return 1;
    }
    QAction* cuboidRedo = nullptr;
    for (QAction* action : window->findChildren<QAction*>())
        if (action->text() == QStringLiteral("Redo")) cuboidRedo = action;
    if (!cuboidRedo || !cuboidRedo->isEnabled()) return 1;
    cuboidRedo->trigger();
    const QJsonObject redoneCuboid = savedCuboid();
    if (redoneCuboid.value(QStringLiteral("points")).toArray() !=
                editedPoints ||
        redoneCuboid.value(QStringLiteral("cuboid3d"))
                        .toObject()
                        .value(QStringLiteral("custom"))
                        .toInt() != 42) {
        std::fprintf(stderr,
                     "Window cuboid redo lost geometry or source metadata\n");
        return 1;
    }
    // Open a JSON outside the image folder through the real drag/drop path.
    // Saving must update that exact file with a path relative to its folder,
    // and must not overwrite the adjacent image's independent annotations.
    QDir root(directory.path());
    if (!root.mkdir(QStringLiteral("labels"))) return 1;
    QFile external(root.filePath(QStringLiteral("labels/review.json")));
    QJsonObject externalJson{
            {QStringLiteral("shapes"), QJsonArray{cuboidShape}},
            {QStringLiteral("imagePath"), QStringLiteral("../cuboid.png")},
            {QStringLiteral("imageData"), QJsonValue::Null},
            {QStringLiteral("custom_document"), 73}};
    if (!external.open(QIODevice::WriteOnly) ||
        external.write(QJsonDocument(externalJson).toJson()) < 0)
        return 1;
    external.close();
    QMimeData jsonMime;
    jsonMime.setUrls({QUrl::fromLocalFile(external.fileName())});
    QDragEnterEvent jsonEnter(QPoint(20, 20), Qt::CopyAction, &jsonMime,
                              Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(window->centralWidget(), &jsonEnter);
    if (!jsonEnter.isAccepted()) return 1;
    QDropEvent jsonDrop(QPoint(20, 20), Qt::CopyAction, &jsonMime,
                        Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(window->centralWidget(), &jsonDrop);
    save->trigger();
    if (!jsonDrop.isAccepted() || !external.open(QIODevice::ReadOnly)) return 1;
    const auto savedExternal =
            QJsonDocument::fromJson(external.readAll()).object();
    external.close();
    if (savedExternal.value(QStringLiteral("imagePath")).toString() !=
                QStringLiteral("../cuboid.png") ||
        savedExternal.value(QStringLiteral("custom_document")).toInt() != 73 ||
        savedExternal.value(QStringLiteral("shapes"))
                        .toArray()
                        .first()
                        .toObject()
                        .value(QStringLiteral("points")) !=
                cuboidShape.value(QStringLiteral("points"))) {
        std::fprintf(stderr,
                     "JSON-first open/save lost path or document identity\n");
        return 1;
    }
    if (!cuboidSidecar.open(QIODevice::ReadOnly)) return 1;
    const auto adjacent =
            QJsonDocument::fromJson(cuboidSidecar.readAll()).object();
    cuboidSidecar.close();
    if (adjacent.value(QStringLiteral("shapes")).toArray().first().toObject() !=
        redoneCuboid)
        return 1;
    // A malformed document must leave the current document usable and saveable.
    QFile broken(root.filePath(QStringLiteral("broken.json")));
    if (!broken.open(QIODevice::WriteOnly) || broken.write("{") != 1) return 1;
    broken.close();
    bool errorShown = false;
    QTimer::singleShot(0, window, [&]() {
        if (auto* box = qobject_cast<QMessageBox*>(
                    QApplication::activeModalWidget())) {
            errorShown = true;
            box->accept();
        }
    });
    if (!QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                   Q_ARG(QString, broken.fileName())))
        return 1;
    save->trigger();
    if (!errorShown || !external.open(QIODevice::ReadOnly)) return 1;
    const auto afterError =
            QJsonDocument::fromJson(external.readAll()).object();
    external.close();
    if (afterError != savedExternal) {
        std::fprintf(stderr, "Failed JSON load changed the active document\n");
        return 1;
    }

    // Drive the real batch dialog through its nested confirmation loop.
    if (!root.mkdir(QStringLiteral("batch"))) return 1;
    QDir batch(root.filePath(QStringLiteral("batch")));
    for (int i = 0; i < 3; ++i)
        if (!QFile::copy(cuboidImagePath,
                         batch.filePath(QStringLiteral("%1.png").arg(i))))
            return 1;
    QFile firstBatch(batch.filePath(QStringLiteral("0.json")));
    if (!firstBatch.open(QIODevice::WriteOnly) ||
        firstBatch.write(QJsonDocument(QJsonObject{{QStringLiteral("shapes"),
                                                    QJsonArray{cuboidShape}}})
                                 .toJson()) < 0)
        return 1;
    firstBatch.close();
    if (!QMetaObject::invokeMethod(
                window, "openImage", Qt::DirectConnection,
                Q_ARG(QString, batch.filePath(QStringLiteral("0.png")))))
        return 1;
    auto* batchObjects = window->findChild<QListWidget*>(
            QStringLiteral("AnnotationObjects"));
    auto* managerAction = window->findChild<QAction*>(
            QStringLiteral("qAnyShortcut_edit_shapes"));
    if (!batchObjects || batchObjects->count() != 1 || !managerAction ||
        !managerAction->isEnabled())
        return 1;
    batchObjects->item(0)->setSelected(true);
    bool batchControls = false, batchConfirmed = false;
    QTimer::singleShot(0, window, [&]() {
        auto* dialog =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (!dialog ||
            dialog->objectName() != QStringLiteral("AnnotationShapeManager"))
            return;
        auto* from = dialog->findChild<QSpinBox*>(
                QStringLiteral("AnnotationShapeFrom"));
        auto* to = dialog->findChild<QSpinBox*>(
                QStringLiteral("AnnotationShapeTo"));
        auto* add = dialog->findChild<QCheckBox*>(
                QStringLiteral("AnnotationShapeOperation3"));
        auto* remove = dialog->findChild<QCheckBox*>(
                QStringLiteral("AnnotationShapeOperation2"));
        auto* go = dialog->findChild<QPushButton*>(
                QStringLiteral("AnnotationShapeGo"));
        batchControls = from && to && add && remove && go &&
                        from->value() == 1 && to->value() == 0 &&
                        add->isEnabled() && remove->isEnabled();
        if (!batchControls) {
            dialog->reject();
            return;
        }
        remove->setChecked(true);
        add->setChecked(true);
        batchControls &= !remove->isChecked();
        to->setValue(3);
        QTimer::singleShot(0, window, [&]() {
            if (auto* box = qobject_cast<QMessageBox*>(
                        QApplication::activeModalWidget())) {
                batchConfirmed = true;
                box->button(QMessageBox::Yes)->click();
            }
        });
        go->click();
    });
    managerAction->trigger();
    if (!batchControls || !batchConfirmed) return 1;
    for (int i = 0; i < 3; ++i) {
        QFile file(batch.filePath(QStringLiteral("%1.json").arg(i)));
        if (!file.open(QIODevice::ReadOnly)) return 1;
        const auto shapes = QJsonDocument::fromJson(file.readAll())
                                    .object()
                                    .value(QStringLiteral("shapes"))
                                    .toArray();
        if (shapes.size() != 1 ||
            shapes.first().toObject().value(QStringLiteral("points")) !=
                    cuboidShape.value(QStringLiteral("points"))) {
            std::fprintf(stderr,
                         "Shape Manager add/deduplicate did not persist "
                         "selected cuboid\n");
            return 1;
        }
    }
    window->close();
    QCoreApplication::processEvents();
    return 0;
}
