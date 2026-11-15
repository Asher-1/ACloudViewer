// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

#include "VideoSegmentExporter.h"
#include "VideoSidecar.h"

class QComboBox;
class QDoubleSpinBox;
class QDragEnterEvent;
class QDragMoveEvent;
class QDropEvent;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QPushButton;
class VideoPlaybackWidget;
class SegmentTimeline;

class VideoAnnotationDialog final : public QDialog {
    Q_OBJECT
public:
    explicit VideoAnnotationDialog(QWidget* parent = nullptr);
    bool openVideo(const QString& path);
    /** Export the selected segment with exact decoded frame boundaries. */
    bool exportSelectedSegmentVideo(const QString& outputPath,
                                    QString* error = nullptr);
    /** Export the selected segment as img_00001.jpg raw frames. */
    bool exportSelectedSegmentFrames(const QString& outputDirectory,
                                     QString* error = nullptr);

protected:
    void closeEvent(QCloseEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    struct EditSnapshot {
        VideoSidecar data;
        int selectedRow = -1;
    };

    bool confirmSave();
    bool save();
    void refreshLabels();
    void refreshSegments();
    void refreshTimeline();
    void selectSegment(int row);
    void seek(qint64 ms);
    void addSegment(qint64 startMs, qint64 endMs);
    void splitSelectedSegment();
    void pushUndoSnapshot();
    void undoLastEdit();
    void redoLastEdit();
    void restoreSnapshot(const EditSnapshot& snapshot);
    void refreshHistoryActions();
    bool exportSelectedSegment(VideoSegmentExportFormat format,
                               const QString& outputPath,
                               QString* error);
    void setDirty();

    VideoPlaybackWidget* m_player = nullptr;
    SegmentTimeline* m_timeline = nullptr;
    QListWidget* m_labels = nullptr;
    QListWidget* m_segments = nullptr;
    QComboBox* m_activeLabel = nullptr;
    QDoubleSpinBox* m_start = nullptr;
    QDoubleSpinBox* m_end = nullptr;
    QPlainTextEdit* m_description = nullptr;
    QLabel* m_info = nullptr;
    QPushButton* m_undoButton = nullptr;
    QPushButton* m_redoButton = nullptr;
    QPushButton* m_exportDataset = nullptr;
    VideoSidecar m_data;
    QString m_videoPath;
    qint64 m_currentMs = 0;
    qint64 m_markStart = -1;
    qint64 m_markEnd = -1;
    quint64 m_thumbnailRequest = 0;
    bool m_updating = false;
    bool m_dirty = false;
    QString m_descriptionUndoId;
    QString m_rangeUndoId;
    bool m_timelineUndoActive = false;
    QVector<EditSnapshot> m_undoStack;
    QVector<EditSnapshot> m_redoStack;
};
