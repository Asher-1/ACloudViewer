// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// SAM3 interactive segmentation dialog.
//
// Qt-based replacement for the upstream ImGui demo (examples/main_image.cpp).
// Layout (mirrors the upstream single-panel strategy):
//   Shared device row at the top, then one tab per model family:
//     SAM 3 Full (ViT + text detector) → Points / Box (PVS) / Exemplar (PCS)
//     SAM 3 Visual (no text encoder)   → Points / Box (PVS)
//     SAM 2 / 2.1 (visual-only Hiera)  → Points / Box (PVS)
//     Video (self-contained tracking tab)
//   Each image tab is self-contained and compact: two control rows, the
//   canvas expanding to fill all remaining space, a compact bottom bar
//   (score / show masks / export to DB / multimask / Clear / Export masks /
//   status) and the detection list. No tab's layout affects another one.
//   Canvas mouse interaction: left-click → +point, right-click → -point,
//                             drag → bounding box

#pragma once

#include <aicore/sam3_capi.h>

#include <QDialog>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPixmap>
#include <QThread>
#include <QVector>
#include <QWidget>

#include "SAM3Worker.h"
#include "ecvTestDataRepository.h"

class QDragEnterEvent;
class QDropEvent;
class QResizeEvent;

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QSlider;
class QTabWidget;
class QTextBrowser;
class QTimer;
class ecvMainAppInterface;
class ecvModelDownloader;
class VideoTab;  // video tracking tab (VideoTab.h; built with OpenCV only)

/** Detection box + label drawn on the image canvas, mirroring upstream
 *  examples/main_image.cpp detection boxes. */
struct SAM3DetBox {
    QRectF box;
    int instanceId = -1;
    float score = 0.0f;
    QColor color;
};

// QLabel subclass that captures mouse events for canvas interaction
class SAM3Canvas : public QLabel {
    Q_OBJECT
public:
    explicit SAM3Canvas(QWidget* parent = nullptr);

    void setInteractive(bool on);
    void setImage(const QImage& img);
    void updateOverlay();
    void clearOverlay();
    /** Shows the step-by-step usage hint when no image is loaded. */
    void updatePlaceholderText();

    /** Sets the mask overlay layer (blended at original image resolution);
     *  drawn between the image and the point/box annotations. */
    void setMaskOverlay(const QImage& mask);
    void clearMaskOverlay();
    /** Sets the detection boxes + labels drawn atop the mask overlay
     *  (upstream main_image.cpp draws a rect + "#id score" per detection). */
    void setDetections(const QVector<SAM3DetBox>& detections);
    void clearDetections();
    /** Sets the positive exemplar boxes (Exemplar / PCS mode) drawn as green
     *  rects, mirroring upstream examples/main_image.cpp pos_exemplars. */
    void setExemplars(const QVector<QRectF>& exemplars);
    void clearExemplars();
    /** In Exemplar (PCS) mode the freshly dragged box is one of the green
     *  exemplar rects (upstream draws pos_exemplars only); the cyan
     *  "confirmed PVS box" is hidden. */
    void setExemplarMode(bool on);
    /** True when \p p (in original image pixels) lies inside the image. */
    bool isInsideImage(const QPointF& p) const;

    const QVector<QPointF>& posPoints() const { return m_posPoints; }
    const QVector<QPointF>& negPoints() const { return m_negPoints; }

    void addPosPoint(const QPointF& p);
    void addNegPoint(const QPointF& p);
    void clearPoints();
    void setBox(const QRectF& box);
    bool hasBox() const { return m_hasBox; }
    QRectF box() const { return m_box; }

signals:
    void pointAdded(int type);  // 0 = positive, 1 = negative
    void boxDrawn();
    void imageDropped(const QString& path);

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseDoubleClickEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void paintEvent(QPaintEvent* e) override;
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

private:
    void drawAnnotations(QPainter& p, const QImage& display);
    QPointF screenToImage(const QPointF& screen) const;

    bool m_interactive = false;
    QImage m_original;
    QImage m_overlay;
    QImage m_maskOverlay;
    QVector<SAM3DetBox> m_detections;
    QVector<QRectF> m_exemplars;  // Exemplar (PCS) mode, green rects
    bool m_exemplarMode = false;  // hide the cyan confirmed box in PCS mode

    // Points / box state
    QVector<QPointF> m_posPoints;
    QVector<QPointF> m_negPoints;
    bool m_hasBox = false;
    QRectF m_box;
    bool m_dragging = false;
    QPointF m_dragStart;
    QRectF m_dragRect;
};

/** Widgets + per-tab state of one image segmentation tab. Each tab owns
 *  its own canvas and bottom bar so the three image tabs are fully
 *  self-contained and never affected by the Video tab's layout. */
struct ImageTabUi {
    // Tab page + control rows
    QWidget* tab = nullptr;
    QRadioButton* modePoints = nullptr;
    QRadioButton* modeBox = nullptr;
    QRadioButton* modeExemplar = nullptr;  // SAM 3 Full only
    QLineEdit* textPrompt = nullptr;       // SAM 3 Full only
    QPushButton* segmentBtn = nullptr;     // SAM 3 Full only
    QComboBox* modelCombo = nullptr;
    QPushButton* downloadBtn = nullptr;  // downloads the selected catalog GGUF
    QPushButton* loadBtn = nullptr;
    QComboBox* testDataCombo = nullptr;    // test-image picker (SAM3 dataset)
    QPushButton* testDataBtn = nullptr;
    // Canvas (fills the tab) + bottom bar
    SAM3Canvas* canvas = nullptr;
    QLabel* detLabel = nullptr;
    QDoubleSpinBox* scoreSpin = nullptr;
    QCheckBox* showMasks = nullptr;
    QCheckBox* multimask = nullptr;
    QCheckBox* exportToDbCheckBox = nullptr;
    QPushButton* clearBtn = nullptr;
    QPushButton* exportBtn = nullptr;
    QLabel* statusLabel = nullptr;
    QTextBrowser* detectionLabel = nullptr;
    // Per-tab state
    QImage currentImage;
    QString currentImagePath;
    SAM3WorkerResult lastResult;
    /** Positive exemplar boxes collected in Exemplar (PCS) mode, mirroring
     *  upstream examples/main_image.cpp pos_exemplars. */
    QVector<QRectF> posExemplars;
};

class SAM3Dialog : public QDialog {
    Q_OBJECT
public:
    struct Settings {
        QString device = "auto";
        int threads = 4;
        int encodeImgSize = 0;
        float scoreThreshold = 0.5f;
        float nmsThreshold = 0.1f;
        float assocIouThreshold = 0.1f;
        int hotstartDelay = 15;
        int maxKeepAlive = 30;
        int reconditionEvery = 16;
        int fillHoleArea = 16;
        // Last selected model filename per tab ("" = none)
        QString modelFull;
        QString modelVisual;
        QString modelSam2;
    };

    explicit SAM3Dialog(QWidget* parent = nullptr);
    ~SAM3Dialog() override;

    void appendLog(const QString& msg);
    Settings settings() const { return m_settings; }
    bool isRunning() const { return m_worker && m_worker->isRunning(); }

public slots:
    void applyDbTreeSelection(const QStringList& names);

    /** Pass the app interface for DB-tree export. Must be called before
     *  the dialog is shown (typically from qSAM3::showDialog); forwarded to
     *  the Video tab as well so its Export to DB works too. */
    void setAppInterface(ecvMainAppInterface* app);

private slots:
    void onLoadModel();
    void onRunSegment();
    void onClear();
    void onExportMasks();
    void onWorkerFinished(bool ok);
    void onWorkerProgress(int current, int total);
    void onWorkerLog(const QString& msg);
    void onWorkerResult(const SAM3WorkerResult& result);
    void onCanvasPoint(int type);
    void onCanvasBox();
    void onModeChanged();
    void onDeviceChanged(int idx);
    void onTabChanged(int index);
    void requestTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);

protected:
    void showEvent(QShowEvent* e) override;
    /** Keeps the busy overlay covering the whole dialog. */
    void resizeEvent(QResizeEvent* e) override;

private:
    /** Model families grouped per tab. */
    enum class Sam3Tab { Full = 0, Visual, Sam2 };

    void setupUi();
    void populateModelCombo(QComboBox* combo, Sam3Tab tab);
    void selectModelByFilename(QComboBox* combo, const QString& filename);
    static bool familyMatches(const char* family, Sam3Tab tab);
    void loadSettings();
    void saveSettings();
    QString modelPath() const;
    /** Download the GGUF currently selected in the active tab's model combo
     *  into the shared AICore model cache (sam3_models). Naming mirrors the
     *  qDA3/qYOLO/qDeepLSD startDownload convention. When \p thenRun is
     *  true the pending operation is re-executed once the download finishes. */
    void startDownload(bool thenRun);
    /** Refresh every tab's Download button state (cached / missing). */
    void updateDownloadButtons();
    /** Re-fill all three model combos, keeping the current selection. */
    void refreshModelCombos();
    void startWorker(SAM3WorkerAction action);
    void stopWorker();
    /** Run a segmentation request. Canvas-originated requests always use
     *  PVS, even when the text field is non-empty, matching main_image.cpp. */
    void runSegmentation(bool canvasPrompt);
    /** If a model file for the current combo entry exists locally and it is
     *  not loaded yet (or the combo switched to a different model), start
     *  loading it right away. Loads lazily on first use (Segment / click /
     *  box), never on dialog show. Returns true when a load was started. */
    bool autoLoadModelIfAvailable();
    /** Ensures a model matching the current combo selection is loaded and
     *  ready for inference. Starts a lazy load when needed (the pending
     *  operation is re-run once the load finishes); shows a message box
     *  when the GGUF file is missing. Returns true when ready to run. */
    bool ensureModelReady();
    /** True when the loaded model differs from the current combo entry
     *  (model switched / tab changed / nothing loaded yet). */
    bool modelSelectionChanged() const;
    void setBusy(bool busy);
    /** Re-evaluate whether the Segment button can run now (image + not
     *  busy; the model loads lazily on first use) and keep the canvas
     *  interactivity in sync. */
    void updateSegmentButtonState();
    /** Refresh the target tab's canvas from its lastResult (defaults to the
     *  active tab; worker callbacks pass the task's tab explicitly). */
    void updateCanvasFromResult(ImageTabUi* target = nullptr);
    void updateDetectionList(ImageTabUi* target = nullptr);
    /** Clear prompts and rendered inference output while preserving the image,
     *  selected model and (optionally) the text prompt. */
    void clearInteractionState(ImageTabUi& ui, bool clearTextPrompt);
    void updateStatus(const QString& msg);
    bool loadRequestedTestData();
    /** Load the sample image selected in the given tab's test-data picker
     *  into that tab's canvas (combo switch auto-loads per tab). */
    bool loadTestImageInto(ImageTabUi& u);
    /** (Re)fill the per-tab test-image pickers from the extracted SAM3
     *  dataset (images/ subdirectory). Keeps the three tabs in sync. */
    void populateTestDataCombos();
    /** Test-image picker of the active tab. */
    QComboBox* currentTestDataCombo() const;
    void setTestDataControlsEnabled(bool enabled);

    Sam3Tab currentTab() const;
    /** Widgets + state of the active image tab (Video tab never reaches
     *  here; guarded by the callers). */
    ImageTabUi& currentUi() { return m_tabsUi[static_cast<int>(currentTab())]; }
    const ImageTabUi& currentUi() const {
        return m_tabsUi[static_cast<int>(currentTab())];
    }
    QComboBox* currentModelCombo() const;
    bool currentPcsMode() const;
    QRadioButton* currentPointsRadio() const;
    QRadioButton* currentBoxRadio() const;

    /** Export the target tab's lastResult to the DB tree as a ccImage
     *  (defaults to the active tab). */
    void exportToDb(ImageTabUi* target = nullptr);

    // UI widgets (top bar)
    QTabWidget* m_tabs = nullptr;
    /** Widgets + state of the three image tabs (indexed by Sam3Tab). */
    ImageTabUi m_tabsUi[3];

    // Device (shared across the image tabs)
    QComboBox* m_deviceCombo = nullptr;
    QLabel* m_backendLabel = nullptr;

    // Video tracking tab (self-contained; VideoTab.h, OpenCV only)
    VideoTab* m_videoTab = nullptr;

    // Busy overlay (spinner + dim, mirroring upstream main_image.cpp
    // draw_busy_overlay)
    QLabel* m_busyOverlay = nullptr;
    QTimer* m_busyTimer = nullptr;
    int m_busyFrame = 0;

    // Worker
    SAM3Worker* m_worker = nullptr;

    // State
    Settings m_settings;
    QString m_modelPath;
    bool m_visualOnly = false;
    bool m_busy = false;
    /** Set when the user triggered Segment / point / box while the model
     *  was still loading; the operation is re-run once the load finishes. */
    bool m_retryAfterModelLoad = false;
    bool m_retryCanvasPrompt = false;
    bool m_reloadAfterCurrentTask = false;
    /** Tab index (0..2) that owns the running worker task; all worker
     *  callbacks (result / model-ready / finished / log) target this tab's
     *  UI so a task started on one tab never writes another tab's state. */
    int m_taskTab = 0;
    /** Tab index that requested the test-data download; the auto-load after
     *  extraction lands on that tab even if the user switched away. */
    int m_testDataTab = 0;

    // Test data (shared ecvTestDataRepository, ObjectsDetection dataset)
    bool m_testDataDownloadInProgress = false;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    bool m_firstShow = true;

    // Model downloader (shared ecvModelDownloader, qDA3-style)
    ecvModelDownloader* m_modelDownloader = nullptr;
    bool m_downloadInProgress = false;
    /** Re-run the pending operation once the download completes. */
    bool m_downloadThenRun = false;
    QString m_downloadTargetFilename;
    /** Tab index that owns the in-flight download (0..2). */
    int m_downloadTab = 0;
    /** Prevent re-prompting after the user declined the download dialog. */
    bool m_downloadPrompted = false;

    // App interface (set via setAppInterface; used for DB-tree export)
    ecvMainAppInterface* m_app = nullptr;
};
