// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QMainWindow>
#include <QPointer>
#include <QStringList>
#include <QThread>
#include <QVector>
#include <functional>

#include "AnnotationCanvas.h"
#include "AnnotationExporter.h"
#include "AnnotationImporter.h"
#include "AnnotationMask.h"
#include "AnnotationMot.h"
#include "AnnotationOdvg.h"
#include "AnnotationPpocr.h"
#include "AnnotationVlm.h"
#include "AnnotationWorker.h"

class QActionGroup;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QDragEnterEvent;
class QDragMoveEvent;
class QDropEvent;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QListWidget;
class QFileInfo;
class QDockWidget;
class QFrame;
class QHBoxLayout;
class QSlider;
class QScrollArea;
class QSplitter;
class QToolButton;
class QTimer;
class QTableWidget;
class QToolBar;
class QProgressDialog;
class QPushButton;
class QSpinBox;
class QDialog;
class QWidget;
class QVBoxLayout;
class ecvMainAppInterface;
class ecvModelDownloader;
class AnnotationNavigator;
class AnnotationPopup;
class QFontComboBox;
class PointCloudWindow;
class QMenu;

class AnnotationWindow final : public QMainWindow {
    Q_OBJECT
public:
    explicit AnnotationWindow(QWidget* parent = nullptr);
    ~AnnotationWindow() override;
    Q_INVOKABLE void openImage(const QString& file);
    /** Pass the host interface to embedded AICore views (for DB export). */
    void setAppInterface(ecvMainAppInterface* app);

protected:
    void closeEvent(QCloseEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    struct Model {
        QString filename;
        QString url;
        QString family;
        QString quant;
        QString task;
        bool supportsText = false;
        // Catalog interaction defaults (facedetect rows carry the upstream
        // scrfd/yunet yaml conf); <=0 means "no catalog default".
        float confDefault = 0.0f;
    };

    void createUi();
    void createActions();
    void populateModels();
    void refreshQuantizations();
    void openFile(const QString& file);
    void openVideoTracking();
    void openPointCloudAnnotation();
    void saveCroppedImage();
    void saveVisualizationImage();
    void openShapeConverter();
    void openDirectory();
    void openDirectoryPath(const QString& path);
    void editLabels();
    void updateRecentMenu();
    void rememberRecentFile(const QString& file);
    void refreshFileList();
    bool saveDocument();
    bool saveDocumentTo(const QString& sidecar);
    void saveDocumentAs();
    /** Source-compatible per-image sidecar resolution honoring the
     *  Change Output Dir contract: when an alternate output directory is
     *  active only that directory is consulted for load/save. */
    QString sidecarPathFor(const QString& imagePath) const;
    /** Load/save/delete candidates: the active output directory first, then
     *  the image directory (the delete path falls back like the source). */
    QStringList sidecarCandidates(const QString& imagePath) const;
    void changeOutputDir();
    void closeImage();
    void deleteLabelFile();
    void deleteImageFile();
    void importYoloLabels(YoloFormat format);
    void importVocDota(bool voc, bool polygons = false);
    void importCocoLabels(CocoFormat format);
    void importMasks();
    void importMotLabels();
    void importSam3Archive();
    void openSam3ArchiveWorkspace();
    void importOdvgLabels();
    void importVlmLabels();
    void importPpocrLabels(PpocrMode mode);
    void importImageFlags();
    void importLabelFolder(
            const QString& title,
            const QStringList& extensions,
            const std::function<AnnotationImport(const QByteArray&,
                                                 const QFileInfo&)>& convert,
            const QString& protectedInput = QString());
    int writeImportedDocuments(const QString& title,
                               const QHash<QString, QJsonObject>& documents,
                               const QString& protectedInput = QString());
    int importNamedDocuments(const QString& title,
                             const QHash<QString, QJsonObject>& documents,
                             const QString& protectedInput);
    void exportYolo(YoloFormat format);
    void exportVocDota(bool voc, bool polygons = false);
    void exportCoco(bool polygons, bool pose = false);
    void exportMasks();
    void exportMot(bool segmentation = false);
    void exportOdvg();
    void exportVlm();
    void exportPpocr(PpocrMode mode);
    bool confirmSave();
    void setDirty(bool dirty);
    void updateObjects();
    void updateDescriptionEditor();
    void updateFileStatus();
    void snapshot();
    void undo();
    void redo();
    void groupSelectedShapes();
    void ungroupSelectedShapes();
    void unionSelectedShapes();
    void copyShapes();
    void pasteShapes();
    void editSelectedLabel();
    void runDetection();
    void runBatch();
    void startBatchNext();
    void finishBatch(const QString& message);
    void startInference(const QString& modelPath);
    void onInference(const AnnotationInference& result);
    void updateFlags();
    void updateTags();
    void setTagBatchMode(bool enabled);
    void finishTagEditing();
    void syncTagsToDocument();
    void commitTagInput();
    void cancelTagInput();
    void updateNavigator();
    void toggleCompareView();
    void closeCompareView();
    void closeCompareView(bool confirm);
    void loadCompareImage();
    void openSettings();
    void updateInstruction();
    void applyTheme(const QString& mode);
    void syncHostShortcuts();
    void configureDigitShortcuts();
    void activateDigitShortcut(int digit);
    void updateActionState();
    void copyShapeCoordinates();
    void populateLabelFilterMenu(QMenu* menu);
    void populateGidFilterMenu(QMenu* menu);
    void applyObjectFilter();
    void applyFileListCheckboxPolicy();
    /** Source-compatible label editing dialog honoring sort_labels,
     *  show_label_text_field, label_completion and fit_to_content. When
     *  flagsOut is given the dialog renders the Label Flags template fields
     *  for the chosen label and returns the checked states. */
    QString openLabelDialog(const QString& initial,
                            bool* accepted,
                            QJsonObject* flagsOut = nullptr,
                            const QJsonObject& currentFlags = {});
    /** Upload template editors (source Upload menu). */
    void uploadLabelFlags();
    void uploadShapeAttributes();
    void uploadLabelClasses();
    /** Edit the uploaded shape-attribute template on the selected shape. */
    void editShapeAttributes();

    AnnotationCanvas* m_canvas = nullptr;
    QWidget* m_workspace = nullptr;
    QWidget* m_toolbarHost = nullptr;
    QVBoxLayout* m_toolbarHostLayout = nullptr;
    QScrollArea* m_toolbarScrollArea = nullptr;
    QSplitter* m_workspaceSplitter = nullptr;
    QScrollArea* m_canvasScrollArea = nullptr;
    QWidget* m_rightSidebar = nullptr;
    QVBoxLayout* m_rightSidebarLayout = nullptr;
    QToolButton* m_sidebarToggle = nullptr;
    int m_rightSidebarWidth = 320;
    AnnotationNavigator* m_navigator = nullptr;
    QDockWidget* m_navigatorDock = nullptr;
    // The auto-labeling controls are a plain central-widget section between
    // the instruction row and the canvas, matching the upstream workbench.
    // The object name stays "AIModel" as the visible/test handle.
    QWidget* m_modelPanel = nullptr;
    QListWidget* m_files = nullptr;
    QListWidget* m_objects = nullptr;
    QListWidget* m_labels = nullptr;
    QListWidget* m_flags = nullptr;
    QDockWidget* m_flagsDock = nullptr;
    QStringList m_imageFlagNames;
    QPlainTextEdit* m_description = nullptr;
    QLabel* m_descriptionTitle = nullptr;
    QListWidget* m_tags = nullptr;
    QLineEdit* m_tagInput = nullptr;
    QWidget* m_tagsPanel = nullptr;
    QAction* m_tagsAction = nullptr;
    bool m_tagBatchMode = false;
    QLineEdit* m_fileFilter = nullptr;
    QComboBox* m_model = nullptr;
    QComboBox* m_quant = nullptr;
    QComboBox* m_backend = nullptr;
    QComboBox* m_textModel = nullptr;
    QComboBox* m_faceLandmark = nullptr;
    QComboBox* m_brushMode = nullptr;
    QComboBox* m_brushTarget = nullptr;
    QSpinBox* m_brushRadius = nullptr;
    QSpinBox* m_zoomWidget = nullptr;
    QLineEdit* m_samText = nullptr;
    QPushButton* m_gkdSupportButton = nullptr;
    QDoubleSpinBox* m_confidence = nullptr;
    QCheckBox* m_slicedDetection = nullptr;
    QSpinBox* m_sliceWidth = nullptr;
    QSpinBox* m_sliceHeight = nullptr;
    QDoubleSpinBox* m_sliceOverlap = nullptr;
    QDoubleSpinBox* m_sliceMergeThreshold = nullptr;
    QCheckBox* m_checked = nullptr;
    QLabel* m_instruction = nullptr;
    QLabel* m_status = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    QThread* m_thread = nullptr;
    AnnotationWorker* m_worker = nullptr;
    QActionGroup* m_tools = nullptr;
    QAction* m_undoAction = nullptr;
    QAction* m_undoLastPointAction = nullptr;
    QAction* m_redoAction = nullptr;
    QAction* m_saveAction = nullptr;
    QAction* m_saveAsAction = nullptr;
    QAction* m_closeAction = nullptr;
    QAction* m_deleteLabelFileAction = nullptr;
    QAction* m_autoRunAction = nullptr;
    QAction* m_toggleCheckedAction = nullptr;
    QAction* m_nextAction = nullptr;
    QAction* m_previousAction = nullptr;
    QAction* m_nextUncheckedAction = nullptr;
    QAction* m_previousUncheckedAction = nullptr;
    QAction* m_zoomInAction = nullptr;
    QAction* m_zoomOutAction = nullptr;
    QAction* m_zoomOriginalAction = nullptr;
    QAction* m_fitWindowAction = nullptr;
    QAction* m_fitWidthAction = nullptr;
    QAction* m_loopThroughLabelsAction = nullptr;
    QAction* m_loopSelectLabelsAction = nullptr;
    QAction* m_pasteAction = nullptr;
    QAction* m_deletePolygonAction = nullptr;
    QAction* m_copyCoordinatesAction = nullptr;
    QAction* m_useSystemClipboardAction = nullptr;
    QAction* m_compareAction = nullptr;
    AnnotationPopup* m_labelLoopPopup = nullptr;
    QAction* m_keepBrightnessAction = nullptr;
    QAction* m_keepContrastAction = nullptr;
    QAction* m_autoUseLastLabelAction = nullptr;
    QAction* m_showShapesAction = nullptr;
    QAction* m_selectToggleShapesAction = nullptr;
    QAction* m_showMasksAction = nullptr;
    QAction* m_showLabelsAction = nullptr;
    QAction* m_showScoresAction = nullptr;
    QAction* m_showGroupsAction = nullptr;
    QAction* m_showTextsAction = nullptr;
    QAction* m_showAttributesAction = nullptr;
    QAction* m_showDegreesAction = nullptr;
    QAction* m_showLinkingAction = nullptr;
    QAction* m_settingsAction = nullptr;
    QMenu* m_recentMenu = nullptr;
    QToolBar* m_compareBar = nullptr;
    QSlider* m_compareSlider = nullptr;
    QLabel* m_compareValue = nullptr;
    QFrame* m_adjustmentPanel = nullptr;
    QToolButton* m_adjustmentToggle = nullptr;
    QSlider* m_opacitySlider = nullptr;
    QSlider* m_brightnessSlider = nullptr;
    QSlider* m_contrastSlider = nullptr;
    QLabel* m_opacityValue = nullptr;
    QLabel* m_brightnessValue = nullptr;
    QLabel* m_contrastValue = nullptr;
    QTimer* m_adjustmentTimer = nullptr;
    QVector<Model> m_models;
    QVector<Model> m_textModels;
    QVector<QVector<AnnotationShape>> m_undo;
    QVector<QVector<AnnotationShape>> m_redo;
    QJsonObject m_document;
    QImage m_image;
    QImage m_gkdSupportImage;
    QVector<QPointF> m_gkdSupportPoints;
    QString m_imagePath;
    QString m_directory;
    // Session-scoped alternate annotation directory (Change Output Dir);
    // the source keeps it as runtime workbench state, not a saved setting.
    QString m_outputDir;
    QStringList m_recentFiles;
    // Source Label Flags / Attributes templates uploaded through the
    // Upload menu: per-label flag names and attribute definitions with
    // their widget types (combobox/radiobutton/group_id/lineedit).
    QHash<QString, QStringList> m_labelFlags;
    QHash<QString, QJsonObject> m_attributes;
    QHash<QString, QJsonObject> m_attributeWidgetTypes;
    bool m_dirty = false;
    // Match xanylabeling_config.yaml: edits are saved automatically unless
    // the user explicitly disables the option in File > Save Automatically.
    bool m_autoSave = true;
    bool m_saveWithImageData = false;
    bool m_autoSaveInProgress = false;
    bool m_updating = false;
    bool m_ocrSkipRequested = false;
    bool m_inferenceRunning = false;
    bool m_batchRunning = false;
    bool m_batchCancelRequested = false;
    bool m_runAfterDownload = false;
    bool m_downloadDependencyModel = false;
    bool m_ignoreNextChange = false;
    bool m_sidecarBlocked = false;
    int m_pasteCount = 0;
    QJsonArray m_internalShapeClipboard;
    quint64 m_generation = 0;
    quint64 m_promptRevision = 0;
    quint64 m_downloadGeneration = 0;
    QString m_downloadImagePath;
    QString m_downloadFilename;
    QStringList m_batchFiles;
    int m_batchIndex = -1;
    QProgressDialog* m_batchProgress = nullptr;
    QString m_compareDirectory;
    QHash<QString, QString> m_compareFiles;
    QHash<int, QPair<QString, QString>> m_digitShortcuts;
    QHash<QString, QPair<int, int>> m_displayAdjustments;
    QHash<QAction*, bool> m_suspendedHostActions;
    bool m_keepPreviousBrightness = false;
    bool m_keepPreviousContrast = false;
    bool m_keepPreviousScale = false;
    bool m_autoUseLastLabel = false;
    bool m_autoHighlightShape = true;
    bool m_autoSwitchToEditMode = true;
    bool m_exifScanEnabled = true;
    bool m_systemClipboard = false;
    bool m_switchToChecked = false;
    int m_defaultBrushRadius = 12;
    int m_defaultShapeOpacity = 100;
    int m_defaultMaskOpacity = 80;
    int m_magicDefaultThreshold = 15;
    double m_magicDragSensitivity = 3.0;
    double m_magicLuminanceWeight = 0.5;
    int m_shapePointSize = 10;
    double m_shapeLineWidth = 4.0;
    double m_canvasEpsilon = 10.0;
    bool m_doubleClickClose = true;
    bool m_doubleClickEditLabel = true;
    int m_numBackups = 10;
    bool m_wheelRectangleEditing = true;
    double m_wheelAdjustStep = 2.0;
    double m_wheelScaleStep = 0.05;
    bool m_crosshairVisible = true;
    double m_crosshairWidth = 2.0;
    QColor m_crosshairColor = QColor(QStringLiteral("#00FF00"));
    double m_crosshairOpacity = 0.5;
    double m_rotationLargeIncrement = 1.0;
    double m_rotationSmallIncrement = 0.1;
    double m_brushPointDistance = 25.0;
    double m_brushSimplifyEpsilon = 2.0;
    int m_brushMaxUndoSteps = 30;
    int m_brushMaxUndoMemoryMb = 128;
    double m_magicSimplifyEpsilon = 0.5;
    double m_magicOpacity = 0.6;
    double m_cuboidMinDepth = 5.0;
    bool m_tagsVisibilityExplicit = false;
    // Source label-review loop state (reset on image change/close).
    bool m_labelLoopActive = false;
    int m_labelLoopCount = -1;
    int m_labelLoopTotal = 0;
    int m_selectLoopCount = -1;
    // Canvas context-menu filter state ("Filter by Label/Group ID").
    QString m_labelFilter;
    int m_gidFilter = -1;
    bool m_displayLabelPopup = true;
    bool m_fileListCheckboxEditable = true;
    // Source label-dialog and canvas move-mode settings.
    bool m_sortLabels = true;
    bool m_showLabelTextField = true;
    QString m_labelCompletion = QStringLiteral("startswith");
    bool m_fitToContentColumn = true;
    bool m_fitToContentRow = false;
    QString m_moveMode = QStringLiteral("auto");
    QString m_fontFamily;
    QPointer<QDialog> m_videoDialog;
    QPointer<PointCloudWindow> m_pointCloudWindow;
    ecvMainAppInterface* m_app = nullptr;

signals:
    void inferenceRequested(const QString& modelPath,
                            const QString& metricModelPath,
                            const QString& device,
                            const QString& task,
                            const QString& imagePath,
                            const QImage& image,
                            quint64 generation,
                            double confidence,
                            quint64 promptRevision,
                            const AnnotationPrompt& prompt);
};
