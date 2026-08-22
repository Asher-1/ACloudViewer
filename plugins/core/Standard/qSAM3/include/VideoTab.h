// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Video segmentation & tracking tab for qSAM3.
//
// Qt re-implementation of the upstream sam3-ggml ImGui demo
// (examples/main_video.cpp). Layout:
//   Row 1: Mode (Text | Box | Points) + text prompt + Open video +
//          Play / Pause / Step / Reset
//   Row 2: Model combo (all catalog families) + Load + Device + Backend
//   Canvas: video frame with per-instance mask tints; click a mask to
//           refine, drag a box / click points to add an instance
//   Timeline: scrubber bar + per-instance presence bands (click to seek)
//   Bottom: Show masks, playback speed, Export frame masks, instance list,
//           status line

#pragma once

#include "VideoCanvas.h"
#include "VideoTimeline.h"
#include "VideoWorker.h"

#include <QTimer>
#include <QWidget>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QRadioButton;
class QSlider;

class VideoFrameReader;
class ecvMainAppInterface;

namespace cv {
class Mat;
}

class VideoTab : public QWidget {
    Q_OBJECT
public:
    explicit VideoTab(QWidget* parent = nullptr);
    ~VideoTab() override;

    void setDevice(const QString& device);  // propagate shared device combo
    /** Pass the app interface for DB-tree export. */
    void setAppInterface(ecvMainAppInterface* app) { m_app = app; }

private slots:
    void onOpenVideo();
    void onLoadModel();
    void onPlayPause();
    void onStep();
    void onReset();
    void onModelReady(const QString& backend, bool visualOnly);
    void onFrameReady(const cv::Mat& rgbFrame, int frameIndex);
    void onFrameResult(const SAM3WorkerResult& result, int frameIndex);
    void onInstanceAdded(int instanceId);
    void onInstanceRefined(int instanceId, bool ok);
    void onBusyChanged(bool busy);
    void onLog(const QString& msg);
    void onSeek(int frame);
    void onModeChanged();
    void onCanvasBox();
    void onCanvasInstanceClicked(int id);
    void onCanvasPosPoint(const QPointF& p);
    void onCanvasNegPoint(const QPointF& p);
    void onExportMasks();

private:
    void setupUi();
    void populateModelCombo();
    QString modelPath() const;
    void openVideoFile(const QString& path);
    void trackNextFrame();
    void schedulePlayback();
    void addInstanceFromPrompts();
    void refineInstance(int id, const QVector<QPointF>& pos,
                        const QVector<QPointF>& neg);
    void updateCanvasInstances();
    void updateTimeline(int frameIndex, const SAM3WorkerResult& result);
    void appendLog(const QString& msg);
    void setStatus(const QString& msg);
    QColor instanceColor(int id) const;
    void resetPrompts();
    /** Export the current frame result to the DB tree as a ccImage. */
    void exportCurrentFrameToDb();

    VideoWorker* m_worker = nullptr;
    VideoFrameReader* m_reader = nullptr;
    QTimer m_playTimer;

    // Controls
    QRadioButton* m_modeText = nullptr;
    QRadioButton* m_modeBox = nullptr;
    QRadioButton* m_modePoints = nullptr;
    QLineEdit* m_textPrompt = nullptr;
    QPushButton* m_openBtn = nullptr;
    QPushButton* m_playBtn = nullptr;
    QPushButton* m_stepBtn = nullptr;
    QPushButton* m_resetBtn = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QPushButton* m_loadBtn = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QLabel* m_backendLabel = nullptr;
    VideoCanvas* m_canvas = nullptr;
    VideoTimeline* m_timeline = nullptr;
    QCheckBox* m_showMasks = nullptr;
    QCheckBox* m_exportToDbCheckBox = nullptr;
    QSlider* m_speedSlider = nullptr;
    QLabel* m_speedLabel = nullptr;
    QPushButton* m_exportBtn = nullptr;
    QLabel* m_instanceLabel = nullptr;
    QLabel* m_statusLabel = nullptr;

    // State
    QString m_videoPath;
    int m_totalFrames = 0;
    int m_currentFrame = 0;
    int m_processedMax = -1;
    double m_fps = 0.0;
    bool m_playing = false;
    bool m_busy = false;
    bool m_trackerActive = false;
    bool m_visualOnly = false;

    SAM3WorkerResult m_lastResult;
    QVector<VideoTimelineEntry> m_timelineEntries;
    QVector<int> m_timelineInstanceIds;

    // DB-tree export state
    ecvMainAppInterface* m_app = nullptr;
    QImage m_currentFrameImage;
};
