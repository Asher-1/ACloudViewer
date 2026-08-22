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

#include "SAM3Worker.h"

#include <aicore/sam3_capi.h>

#include "ecvTestDataRepository.h"

#include <QDialog>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPixmap>
#include <QThread>
#include <QVector>
#include <QWidget>

class QDragEnterEvent;
class QDropEvent;
class QResizeEvent;

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QRadioButton;
class QSlider;
class QTabWidget;
class VideoTab;  // video tracking tab (VideoTab.h; built with OpenCV only)

// QLabel subclass that captures mouse events for canvas interaction
class SAM3Canvas : public QLabel {
    Q_OBJECT
public:
    explicit SAM3Canvas(QWidget* parent = nullptr);

    void setInteractive(bool on);
    void setImage(const QImage& img);
    void updateOverlay();
    void clearOverlay();

    /** Sets the mask overlay layer (blended at original image resolution);
     *  drawn between the image and the point/box annotations. */
    void setMaskOverlay(const QImage& mask);
    void clearMaskOverlay();
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
    };

    explicit SAM3Dialog(QWidget* parent = nullptr);
    ~SAM3Dialog() override;

    void appendLog(const QString& msg);
    Settings settings() const { return m_settings; }
    bool isRunning() const { return m_worker && m_worker->isRunning(); }

public slots:
    void applyDbTreeSelection(const QStringList& names);

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
    void setBusy(bool busy);
    void updateCanvasFromResult();
    void updateDetectionList();
    void updateStatus(const QString& msg);
    bool loadRequestedTestData();
    void setTestDataControlsEnabled(bool enabled);

    Sam3Tab currentTab() const;
    QComboBox* currentModelCombo() const;
    bool currentPcsMode() const;
    QRadioButton* currentPointsRadio() const;
    QRadioButton* currentBoxRadio() const;

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
    QPushButton* m_testDataBtn = nullptr;

    // ── SAM3 Visual tab (visual-only) ──
    QRadioButton* m_modePointsV = nullptr;
    QRadioButton* m_modeBoxV = nullptr;
    QComboBox* m_modelComboV = nullptr;
    QPushButton* m_loadBtnV = nullptr;
    QPushButton* m_testDataBtnV = nullptr;

    // ── SAM 2 / 2.1 tab (visual-only Hiera) ──
    QRadioButton* m_modePointsS = nullptr;
    QRadioButton* m_modeBoxS = nullptr;
    QComboBox* m_modelComboS = nullptr;
    QPushButton* m_loadBtnS = nullptr;
    QPushButton* m_testDataBtnS = nullptr;

    // Device
    QComboBox* m_deviceCombo = nullptr;
    QLabel* m_backendLabel = nullptr;

    // Canvas
    SAM3Canvas* m_canvas = nullptr;
    VideoTab* m_videoTab = nullptr;

    // Bottom panel
    QDoubleSpinBox* m_scoreSpin = nullptr;
    QCheckBox* m_showMasks = nullptr;
    QCheckBox* m_multimask = nullptr;
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
    QImage m_currentImage;
    QString m_currentImagePath;
    bool m_encoded = false;

    // Test data (shared ecvTestDataRepository, ObjectsDetection dataset)
    bool m_testDataDownloadInProgress = false;
};