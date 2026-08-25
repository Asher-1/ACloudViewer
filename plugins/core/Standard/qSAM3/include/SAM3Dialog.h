// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// SAM3 interactive segmentation dialog.
//
// Qt-based replacement for the upstream ImGui demo (examples/main_image.cpp).
// Layout:
//   Tabs per model family:
//     SAM 3 Full (ViT + text detector) → Points / Box (PVS) / Exemplar (PCS)
//     SAM 3 Visual (no text encoder)   → Points / Box (PVS)
//     SAM 2 / 2.1 (visual-only Hiera)  → Points / Box (PVS)
//   Each tab: model combo (filtered by family) + Load + "Try sample data"
//   Shared: device combo, image canvas, score threshold, show masks,
//           multimask, status, detection list, Clear / Export buttons
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
class QTimer;
class ecvMainAppInterface;
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

    // Points / box state
    QVector<QPointF> m_posPoints;
    QVector<QPointF> m_negPoints;
    bool m_hasBox = false;
    QRectF m_box;
    bool m_dragging = false;
    QPointF m_dragStart;
    QRectF m_dragRect;
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
        bool exportToDb = true;
    };

    explicit SAM3Dialog(QWidget* parent = nullptr);
    ~SAM3Dialog() override;

    void appendLog(const QString& msg);
    Settings settings() const { return m_settings; }
    bool isRunning() const { return m_worker && m_worker->isRunning(); }

public slots:
    void applyDbTreeSelection(const QStringList& names);

    /** Pass the app interface for DB-tree export. Must be called before
     *  the dialog is shown (typically from qSAM3::showDialog). */
    void setAppInterface(ecvMainAppInterface* app) { m_app = app; }

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
    void startWorker(SAM3WorkerAction action);
    void stopWorker();
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
    void updateCanvasFromResult();
    void updateDetectionList();
    void updateStatus(const QString& msg);
    bool loadRequestedTestData();
    /** (Re)fill the per-tab test-image pickers from the extracted SAM3
     *  dataset (images/ subdirectory). Keeps the three tabs in sync. */
    void populateTestDataCombos();
    /** Test-image picker of the active tab. */
    QComboBox* currentTestDataCombo() const;
    void setTestDataControlsEnabled(bool enabled);

    Sam3Tab currentTab() const;
    QComboBox* currentModelCombo() const;
    bool currentPcsMode() const;
    QRadioButton* currentPointsRadio() const;
    QRadioButton* currentBoxRadio() const;

    /** Export the current m_lastResult to the DB tree as a ccImage. */
    void exportToDb();

    // UI widgets (top bar)
    QTabWidget* m_tabs = nullptr;
    QLineEdit* m_textPrompt = nullptr;
    QPushButton* m_segmentBtn = nullptr;
    QPushButton* m_clearBtn = nullptr;
    QPushButton* m_exportBtn = nullptr;

    // ── SAM3 Full tab (text + detector) ──
    QRadioButton* m_modePoints = nullptr;
    QRadioButton* m_modeBox = nullptr;
    QRadioButton* m_modeExemplar = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QPushButton* m_loadBtn = nullptr;
    QComboBox* m_testDataCombo = nullptr;  // test-image picker (SAM3 dataset)
    QPushButton* m_testDataBtn = nullptr;

    // ── SAM3 Visual tab (visual-only) ──
    QRadioButton* m_modePointsV = nullptr;
    QRadioButton* m_modeBoxV = nullptr;
    QComboBox* m_modelComboV = nullptr;
    QPushButton* m_loadBtnV = nullptr;
    QComboBox* m_testDataComboV = nullptr;
    QPushButton* m_testDataBtnV = nullptr;

    // ── SAM 2 / 2.1 tab (visual-only Hiera) ──
    QRadioButton* m_modePointsS = nullptr;
    QRadioButton* m_modeBoxS = nullptr;
    QComboBox* m_modelComboS = nullptr;
    QPushButton* m_loadBtnS = nullptr;
    QComboBox* m_testDataComboS = nullptr;
    QPushButton* m_testDataBtnS = nullptr;

    // Device
    QComboBox* m_deviceCombo = nullptr;
    QLabel* m_backendLabel = nullptr;

    // Canvas
    SAM3Canvas* m_canvas = nullptr;
    VideoTab* m_videoTab = nullptr;

    // Busy overlay (spinner + dim, mirroring upstream main_image.cpp
    // draw_busy_overlay)
    QLabel* m_busyOverlay = nullptr;
    QTimer* m_busyTimer = nullptr;
    int m_busyFrame = 0;

    // Bottom panel
    QDoubleSpinBox* m_scoreSpin = nullptr;
    QCheckBox* m_showMasks = nullptr;
    QCheckBox* m_multimask = nullptr;
    QCheckBox* m_exportToDbCheckBox = nullptr;
    QLabel* m_statusLabel = nullptr;
    QLabel* m_detectionLabel = nullptr;

    // Worker
    QThread* m_workerThread = nullptr;
    SAM3Worker* m_worker = nullptr;
    SAM3WorkerResult m_lastResult;

    // State
    Settings m_settings;
    QString m_modelPath;
    int m_mode = 0;  // 0=Points, 1=Box, 2=Exemplar
    bool m_visualOnly = false;
    bool m_busy = false;
    /** Positive exemplar boxes collected in Exemplar (PCS) mode, mirroring
     *  upstream examples/main_image.cpp pos_exemplars. */
    QVector<QRectF> m_posExemplars;
    QImage m_currentImage;
    QString m_currentImagePath;
    bool m_encoded = false;
    /** Set when the user triggered Segment / point / box while the model
     *  was still loading; the operation is re-run once the load finishes. */
    bool m_retryAfterModelLoad = false;

    // Test data (shared ecvTestDataRepository, ObjectsDetection dataset)
    bool m_testDataDownloadInProgress = false;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    bool m_firstShow = true;

    // App interface (set via setAppInterface; used for DB-tree export)
    ecvMainAppInterface* m_app = nullptr;
};