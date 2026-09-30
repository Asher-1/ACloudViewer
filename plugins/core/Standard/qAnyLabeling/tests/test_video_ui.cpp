// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QLineEdit>
#include <QListWidget>
#include <QMimeData>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QShortcut>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QUrl>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "VideoAnnotationDialog.h"
#include "VideoPlaybackWidget.h"
#include "VideoSegmentExporter.h"

int main(int argc, char** argv) {
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QTemporaryDir folder;
    if (!folder.isValid()) return 1;
    const QString path = folder.filePath(QStringLiteral("clip.avi"));
    cv::VideoWriter writer(path.toStdString(),
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5.0,
                           cv::Size(64, 64));
    if (!writer.isOpened()) {
        std::fprintf(stderr, "MJPG writer unavailable\n");
        return 77;
    }
    for (int i = 0; i < 10; ++i)
        writer.write(cv::Mat(64, 64, CV_8UC3, cv::Scalar(i * 20, 80, 200)));
    writer.release();

    VideoSidecar original;
    original.video = QStringLiteral("clip.avi");
    original.fps = 5;
    original.durationMs = 2000;
    original.labels.append(QStringLiteral("walking"));
    original.labelColors.insert(QStringLiteral("walking"),
                                QStringLiteral("#EF476F"));
    original.segments.append({QStringLiteral("s1234567890"),
                              QStringLiteral("walking"), 100, 500, 1, 3,
                              QStringLiteral("person")});
    QString error;
    if (!saveVideoSidecar(path, original, &error)) return 1;

    VideoAnnotationDialog dialog;
    dialog.show();
    app.processEvents();
    if (!dialog.openVideo(path)) {
        std::fprintf(stderr, "video open failed\n");
        return 1;
    }
    auto* player = dialog.findChild<VideoPlaybackWidget*>();
    auto* labels =
            dialog.findChild<QListWidget*>(QStringLiteral("VideoLabels"));
    auto* segments =
            dialog.findChild<QListWidget*>(QStringLiteral("VideoSegments"));
    auto* timeline =
            dialog.findChild<QWidget*>(QStringLiteral("VideoSegmentTimeline"));
    auto* dataset = dialog.findChild<QPushButton*>(
            QStringLiteral("ExportVideoDataset"));
    if (!player || !labels || !segments || !timeline || !dataset ||
        !dataset->isEnabled() || segments->currentRow() != -1 ||
        player->totalVideoFrames() != 10 || labels->count() != 1 ||
        segments->count() != 1) {
        std::fprintf(stderr, "video workspace metadata missing\n");
        return 1;
    }
    QElapsedTimer thumbnailsTimer;
    thumbnailsTimer.start();
    while (timeline->property("thumbnailCount").toInt() != 10 &&
           thumbnailsTimer.elapsed() < 3000)
        app.processEvents(QEventLoop::AllEvents, 20);
    if (timeline->property("thumbnailCount").toInt() != 10) {
        std::fprintf(stderr, "timeline thumbnails were not decoded\n");
        return 1;
    }
    const QImage timelineImage = timeline->grab().toImage();
    const int thumbnailX = timeline->width() * 3 / 4;
    const int thumbnailY = timeline->height() / 2;
    if (timelineImage.isNull() || thumbnailX < 0 || thumbnailY < 0 ||
        thumbnailX >= timelineImage.width() ||
        thumbnailY >= timelineImage.height() ||
        timelineImage.pixelColor(thumbnailX, thumbnailY).red() < 70) {
        std::fprintf(stderr, "timeline thumbnails were not painted\n");
        return 1;
    }
    const QString labelsPath = folder.filePath(QStringLiteral("labels.txt"));
    QFile labelsFile(labelsPath);
    if (!labelsFile.open(QIODevice::WriteOnly) ||
        labelsFile.write("\xef\xbb\xbf"
                         "jumping\nwalking\njumping\n") < 0)
        return 1;
    labelsFile.close();
    auto* settings = dialog.findChild<QPushButton*>(
            QStringLiteral("VideoLabelSettings"));
    if (!settings) return 1;
    QTimer::singleShot(0, &dialog, [&]() {
        auto* editor =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        auto* table = editor ? editor->findChild<QTableWidget*>(QStringLiteral(
                                       "VideoLabelSettingsTable"))
                             : nullptr;
        if (!table || !table->cellWidget(0, 2)) return;
        qobject_cast<QLineEdit*>(table->cellWidget(0, 2))
                ->setText(QStringLiteral("running"));
        QTimer::singleShot(0, editor, [&]() {
            auto* fileDialog = qobject_cast<QFileDialog*>(
                    QApplication::activeModalWidget());
            if (!fileDialog) return;
            fileDialog->selectFile(labelsPath);
            QMetaObject::invokeMethod(fileDialog, "accept",
                                      Qt::QueuedConnection);
        });
        editor->findChild<QPushButton*>(QStringLiteral("ImportVideoLabels"))
                ->click();
        editor->findChild<QDialogButtonBox*>()
                ->button(QDialogButtonBox::Ok)
                ->click();
    });
    settings->click();
    if (labels->count() != 2 ||
        labels->item(0)->text() != QStringLiteral("jumping") ||
        labels->item(1)->text() != QStringLiteral("running")) {
        std::fprintf(stderr, "label import or rename failed\n");
        return 1;
    }
    auto* shortcuts =
            dialog.findChild<QPushButton*>(QStringLiteral("VideoShortcuts"));
    if (!shortcuts) {
        std::fprintf(stderr, "video shortcuts action is unavailable\n");
        return 1;
    }
    QShortcut* labelShortcut = nullptr;
    QShortcut* resetShortcut = nullptr;
    for (QShortcut* shortcut : dialog.findChildren<QShortcut*>()) {
        if (shortcut->key() == QKeySequence(Qt::Key_1)) {
            labelShortcut = shortcut;
        } else if (shortcut->key() == QKeySequence(Qt::Key_0)) {
            resetShortcut = shortcut;
        }
    }
    if (!labelShortcut || !resetShortcut ||
        !QMetaObject::invokeMethod(labelShortcut, "activated") ||
        labels->currentRow() != 1) {
        std::fprintf(stderr, "digit shortcut did not select the label\n");
        return 1;
    }
    if (!QMetaObject::invokeMethod(resetShortcut, "activated") ||
        labels->currentRow() != 0) {
        std::fprintf(stderr, "zero shortcut did not restore the first label\n");
        return 1;
    }
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < 250 && !timeline->isVisible()) app.processEvents();
    const int trackLeft = 12;
    const int trackWidth = timeline->width() - 24;
    const int trackY = timeline->height() / 2;
    const QPoint first(trackLeft + trackWidth * 55 / 100, trackY);
    const QPoint last(trackLeft + trackWidth * 70 / 100, trackY);
    QMouseEvent press(QEvent::MouseButtonPress, first, Qt::LeftButton,
                      Qt::LeftButton, Qt::NoModifier);
    QMouseEvent move(QEvent::MouseMove, last, Qt::NoButton, Qt::LeftButton,
                     Qt::NoModifier);
    QMouseEvent release(QEvent::MouseButtonRelease, last, Qt::LeftButton,
                        Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(timeline, &press);
    QApplication::sendEvent(timeline, &move);
    QApplication::sendEvent(timeline, &release);
    if (segments->count() != 2) {
        std::fprintf(stderr, "timeline drag did not create a segment\n");
        return 1;
    }
    const auto sendDrag = [timeline](const QPoint& from, const QPoint& to) {
        QMouseEvent press(QEvent::MouseButtonPress, from, Qt::LeftButton,
                          Qt::LeftButton, Qt::NoModifier);
        QMouseEvent move(QEvent::MouseMove, to, Qt::NoButton, Qt::LeftButton,
                         Qt::NoModifier);
        QMouseEvent release(QEvent::MouseButtonRelease, to, Qt::LeftButton,
                            Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(timeline, &press);
        QApplication::sendEvent(timeline, &move);
        QApplication::sendEvent(timeline, &release);
    };
    const QPoint rightHandle(trackLeft + trackWidth * 70 / 100, trackY);
    const QPoint extendedRight(trackLeft + trackWidth * 75 / 100, trackY);
    sendDrag(rightHandle, extendedRight);
    const QPoint bodyStart(trackLeft + trackWidth * 64 / 100, trackY);
    const QPoint movedBody(trackLeft + trackWidth * 80 / 100, trackY);
    sendDrag(bodyStart, movedBody);
    if (segments->currentRow() != 1) {
        std::fprintf(stderr, "new segment is not selected\n");
        return 1;
    }
    const QString clipExport = folder.filePath(QStringLiteral("segment.avi"));
    const QString frameExport = folder.filePath(QStringLiteral("rawframes"));
    if (!dialog.exportSelectedSegmentVideo(clipExport, &error) ||
        !dialog.exportSelectedSegmentFrames(frameExport, &error)) {
        std::fprintf(stderr, "segment export failed: %s\n", qPrintable(error));
        return 1;
    }
    cv::VideoCapture exportedClip(clipExport.toStdString());
    cv::Mat exportedFirst;
    if (!exportedClip.isOpened() ||
        int(exportedClip.get(cv::CAP_PROP_FRAME_COUNT)) != 2 ||
        !exportedClip.read(exportedFirst) || exportedFirst.empty() ||
        std::abs(int(exportedFirst.at<cv::Vec3b>(0, 0)[0]) - 140) > 4 ||
        QDir(frameExport)
                        .entryList({QStringLiteral("img_*.jpg")}, QDir::Files)
                        .size() != 2) {
        std::fprintf(stderr, "exported frame range is incorrect\n");
        return 1;
    }
    const cv::Mat firstJpeg =
            cv::imread(QDir(frameExport)
                               .filePath(QStringLiteral("img_00001.jpg"))
                               .toStdString());
    if (firstJpeg.empty() ||
        std::abs(int(firstJpeg.at<cv::Vec3b>(0, 0)[0]) - 140) > 4 ||
        !dialog.exportSelectedSegmentVideo(clipExport, &error)) {
        std::fprintf(stderr, "exported clip replace or JPEG failed: %s\n",
                     qPrintable(error));
        return 1;
    }
    const QString cancelled = folder.filePath(QStringLiteral("cancelled"));
    if (exportVideoSegment(
                {path, cancelled, 1400, 1800, 5,
                 VideoSegmentExportFormat::RawFrames},
                [](int, int) { return false; }, &error) ||
        QDir(cancelled).exists()) {
        std::fprintf(stderr, "cancelled export left output behind\n");
        return 1;
    }
    bool saved = false;
    for (QPushButton* button : dialog.findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("Save")) {
            button->click();
            saved = true;
            break;
        }
    }
    VideoSidecar reread;
    if (!saved || !loadVideoSidecar(path, &reread, &error) ||
        reread.segments.size() != 2 || reread.labels.size() != 2 ||
        reread.segments.first().label != QStringLiteral("running") ||
        reread.segments.last().startMs != 1400 ||
        reread.segments.last().endMs != 1800 ||
        reread.segments.last().startFrame != 7 ||
        reread.segments.last().endFrame != 9) {
        std::fprintf(stderr, "timeline segment did not survive save: %s\n",
                     qPrintable(error));
        return 1;
    }
    auto* split =
            dialog.findChild<QPushButton*>(QStringLiteral("VideoSplitSegment"));
    auto* description = dialog.findChild<QPlainTextEdit*>();
    if (!split || !description) return 1;
    player->stopStream();
    segments->setCurrentRow(1);
    description->setPlainText(QStringLiteral("moving person"));
    segments->setCurrentRow(0);
    const QPoint playhead(trackLeft + trackWidth * 80 / 100, 15);
    QMouseEvent seekPress(QEvent::MouseButtonPress, playhead, Qt::LeftButton,
                          Qt::LeftButton, Qt::NoModifier);
    QMouseEvent seekRelease(QEvent::MouseButtonRelease, playhead,
                            Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(timeline, &seekPress);
    QApplication::sendEvent(timeline, &seekRelease);
    split->click();
    if (segments->count() != 3 || segments->currentRow() != 2) {
        std::fprintf(stderr, "split did not select the second segment\n");
        return 1;
    }
    for (QPushButton* button : dialog.findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("Save")) {
            button->click();
            break;
        }
    }
    if (!loadVideoSidecar(path, &reread, &error) ||
        reread.segments.size() != 3 || reread.segments[1].startMs != 1400 ||
        reread.segments[1].endMs != 1600 ||
        reread.segments[2].startMs != 1600 ||
        reread.segments[2].endMs != 1800 || reread.segments[1].endFrame != 8 ||
        reread.segments[2].startFrame != 8 ||
        reread.segments[1].label != QStringLiteral("jumping") ||
        reread.segments[2].label != QStringLiteral("jumping") ||
        reread.segments[1].description != QStringLiteral("moving person") ||
        reread.segments[2].description != QStringLiteral("moving person") ||
        reread.segments[1].id == reread.segments[2].id ||
        reread.segments[1].id == reread.segments[0].id ||
        reread.segments[2].id == reread.segments[0].id) {
        std::fprintf(stderr, "split actual count=%d\n", reread.segments.size());
        for (const auto& segment : reread.segments)
            std::fprintf(stderr,
                         "segment %s label=%s start=%lld end=%lld sf=%d ef=%d "
                         "desc=%s\n",
                         qPrintable(segment.id), qPrintable(segment.label),
                         static_cast<long long>(segment.startMs),
                         static_cast<long long>(segment.endMs),
                         segment.startFrame, segment.endFrame,
                         qPrintable(segment.description));
        std::fprintf(stderr,
                     "split boundaries or metadata did not survive save\n");
        return 1;
    }
    auto* undo = dialog.findChild<QPushButton*>(QStringLiteral("VideoUndo"));
    auto* redo = dialog.findChild<QPushButton*>(QStringLiteral("VideoRedo"));
    if (!undo || !redo || !undo->isEnabled() || redo->isEnabled()) {
        std::fprintf(stderr, "video edit history actions are unavailable\n");
        return 1;
    }
    undo->click();
    if (segments->count() != 2 || !redo->isEnabled()) {
        std::fprintf(stderr,
                     "video undo did not restore the original segment\n");
        return 1;
    }
    redo->click();
    if (segments->count() != 3 || !undo->isEnabled()) {
        std::fprintf(stderr, "video redo did not restore the split\n");
        return 1;
    }
    for (QPushButton* button : dialog.findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("Save")) {
            button->click();
            break;
        }
    }
    QMimeData droppedMime;
    droppedMime.setUrls({QUrl::fromLocalFile(path)});
    QDragEnterEvent enter(QPoint(20, 20), Qt::CopyAction, &droppedMime,
                          Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(player, &enter);
    QDragMoveEvent dragMove(QPoint(20, 20), Qt::CopyAction, &droppedMime,
                            Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(player, &dragMove);
    QDropEvent drop(QPoint(20, 20), Qt::CopyAction, &droppedMime,
                    Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(player, &drop);
    if (!enter.isAccepted() || !dragMove.isAccepted() || !drop.isAccepted() ||
        player->totalVideoFrames() != 10) {
        std::fprintf(stderr, "Video drag and drop failed\n");
        return 1;
    }
    if (undo->isEnabled() || redo->isEnabled()) {
        std::fprintf(stderr, "video history was not cleared after reopening\n");
        return 1;
    }
    dialog.close();
    return 0;
}
