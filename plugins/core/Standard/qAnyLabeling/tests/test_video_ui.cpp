// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QAction>
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
#include <QSettings>
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

class VideoUiTest : public ::testing::Test {
protected:
    QTemporaryDir folder;
    QString path, error;
    void SetUp() override {
        ASSERT_FALSE((!folder.isValid()));
        path = folder.filePath(QStringLiteral("clip.avi"));
        cv::VideoWriter writer(path.toStdString(),
                               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5.0,
                               cv::Size(64, 64));
        if (!writer.isOpened()) {
            std::fprintf(stderr, "MJPG writer unavailable\n");
            GTEST_SKIP() << "MJPG writer unavailable";
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
        ASSERT_TRUE(saveVideoSidecar(path, original, &error))
                << error.toStdString();
    }
};

TEST_F(VideoUiTest, VisualizationExportPreservesRgbFpsAndAtomicOutput) {
    const QString visualization =
            folder.filePath(QStringLiteral("visualization.mp4"));
    const auto renderFrame = [](int index, QString*) {
        QImage frame(64, 64, QImage::Format_RGB888);
        frame.fill(QColor(200, 80, index * 40));
        return frame;
    };
    ASSERT_FALSE((!exportVisualizationVideo(visualization, 25, 3, renderFrame,
                                            {}, &error)))
            << "visualization export failed: %s\n";
    cv::VideoCapture visualizationReader(visualization.toStdString());
    cv::Mat visualizationFirst;
    ASSERT_FALSE(
            (!visualizationReader.isOpened() ||
             std::abs(visualizationReader.get(cv::CAP_PROP_FPS) - 25) > .01 ||
             int(visualizationReader.get(cv::CAP_PROP_FRAME_COUNT)) != 3 ||
             !visualizationReader.read(visualizationFirst) ||
             visualizationFirst.size() != cv::Size(64, 64) ||
             std::abs(int(visualizationFirst.at<cv::Vec3b>(32, 32)[2]) - 200) >
                     8 ||
             visualizationFirst.at<cv::Vec3b>(32, 32)[0] > 8))
            << "visualization FPS/count/size/RGB order mismatch\n";
    visualizationReader.release();
    QFile committed(visualization);
    ASSERT_FALSE((!committed.open(QIODevice::ReadOnly)));
    const QByteArray originalVideo = committed.readAll();
    committed.close();
    for (int mode = 0; mode < 4; ++mode) {
        const auto failingFrame = [&, mode](int index, QString* message) {
            if (mode == 1 && index == 1) {
                *message = QStringLiteral("fixture decode error");
                return QImage();
            }
            if (mode == 2 && index == 1) {
                QImage different(66, 64, QImage::Format_RGB888);
                different.fill(Qt::red);
                return different;
            }
            return renderFrame(index, message);
        };
        ASSERT_FALSE((
                exportVisualizationVideo(
                        visualization, mode == 3 ? 0 : 25, 3, failingFrame,
                        [mode](int done, int) { return mode != 0 || done < 1; },
                        &error) ||
                error.isEmpty() || !committed.open(QIODevice::ReadOnly)));
        ASSERT_FALSE((committed.readAll() != originalVideo))
                << "failed visualization replaced committed output\n";
        committed.close();
    }
    ASSERT_FALSE((!QDir(folder.path())
                           .entryList({QStringLiteral(".qany-video-*")},
                                      QDir::Dirs | QDir::Hidden |
                                              QDir::NoDotAndDotDot)
                           .isEmpty()));
}

class VideoWorkspaceTest : public VideoUiTest {
protected:
    VideoAnnotationDialog dialog;
    VideoPlaybackWidget* player = nullptr;
    QListWidget* labels = nullptr;
    QListWidget* segments = nullptr;
    QWidget* timeline = nullptr;
    QPushButton* dataset = nullptr;
    void SetUp() override {
        VideoUiTest::SetUp();
        if (HasFatalFailure() || IsSkipped()) return;
        dialog.show();
        QApplication::processEvents();
        ASSERT_FALSE((!dialog.openVideo(path))) << "video open failed\n";
        player = dialog.findChild<VideoPlaybackWidget*>();
        labels = dialog.findChild<QListWidget*>(QStringLiteral("VideoLabels"));
        segments =
                dialog.findChild<QListWidget*>(QStringLiteral("VideoSegments"));
        timeline = dialog.findChild<QWidget*>(
                QStringLiteral("VideoSegmentTimeline"));
        dataset = dialog.findChild<QPushButton*>(
                QStringLiteral("ExportVideoDataset"));
        ASSERT_FALSE((!player || !labels || !segments || !timeline ||
                      !dataset || !dataset->isEnabled() ||
                      segments->currentRow() != -1 ||
                      player->totalVideoFrames() != 10 ||
                      labels->count() != 1 || segments->count() != 1))
                << "video workspace metadata missing\n";
    }
    void TearDown() override { dialog.close(); }
};

TEST_F(VideoWorkspaceTest, TimelineDecodesAndPaintsEveryThumbnail) {
    QElapsedTimer thumbnailsTimer;
    thumbnailsTimer.start();
    while (timeline->property("thumbnailCount").toInt() != 10 &&
           thumbnailsTimer.elapsed() < 3000)
        QApplication::processEvents(QEventLoop::AllEvents, 20);
    ASSERT_FALSE((timeline->property("thumbnailCount").toInt() != 10))
            << "timeline thumbnails were not decoded\n";
    const QImage timelineImage = timeline->grab().toImage();
    const int thumbnailX = timeline->width() * 3 / 4;
    const int thumbnailY = timeline->height() / 2;
    ASSERT_FALSE((timelineImage.isNull() || thumbnailX < 0 || thumbnailY < 0 ||
                  thumbnailX >= timelineImage.width() ||
                  thumbnailY >= timelineImage.height() ||
                  timelineImage.pixelColor(thumbnailX, thumbnailY).red() < 70))
            << "timeline thumbnails were not painted\n";
}

TEST_F(VideoWorkspaceTest, LabelTimelineExportsSplitHistoryAndDropRoundTrip) {
    const QString labelsPath = folder.filePath(QStringLiteral("labels.txt"));
    QFile labelsFile(labelsPath);
    ASSERT_FALSE((!labelsFile.open(QIODevice::WriteOnly) ||
                  labelsFile.write("\xef\xbb\xbf"
                                   "jumping\nwalking\njumping\n") < 0));
    labelsFile.close();
    auto* settings = dialog.findChild<QPushButton*>(
            QStringLiteral("VideoLabelSettings"));
    ASSERT_FALSE((!settings));
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
    ASSERT_FALSE((labels->count() != 2 ||
                  labels->item(0)->text() != QStringLiteral("jumping") ||
                  labels->item(1)->text() != QStringLiteral("running")))
            << "label import or rename failed\n";
    auto* shortcuts =
            dialog.findChild<QPushButton*>(QStringLiteral("VideoShortcuts"));
    ASSERT_FALSE((!shortcuts)) << "video shortcuts action is unavailable\n";
    auto* labelAction = dialog.findChild<QAction*>(
            QStringLiteral("VideoAnnotation_label-slot-1"));
    auto* resetAction = dialog.findChild<QAction*>(
            QStringLiteral("VideoAnnotation_label-slot-0"));
    ASSERT_TRUE(labelAction && resetAction);
    EXPECT_TRUE(labelAction->shortcuts().contains(QKeySequence(Qt::Key_1)));
    EXPECT_TRUE(resetAction->shortcuts().contains(QKeySequence(Qt::Key_0)));
    labelAction->trigger();
    ASSERT_EQ(labels->currentRow(), 1)
            << "digit command did not select the label";
    resetAction->trigger();
    ASSERT_EQ(labels->currentRow(), 0)
            << "zero command did not restore the first label";
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < 250 && !timeline->isVisible())
        QApplication::processEvents();
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
    ASSERT_FALSE((segments->count() != 2))
            << "timeline drag did not create a segment\n";
    const auto sendDrag = [this](const QPoint& from, const QPoint& to) {
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
    ASSERT_FALSE((segments->currentRow() != 1))
            << "new segment is not selected\n";
    const QString clipExport = folder.filePath(QStringLiteral("segment.avi"));
    const QString frameExport = folder.filePath(QStringLiteral("rawframes"));
    ASSERT_FALSE((!dialog.exportSelectedSegmentVideo(clipExport, &error) ||
                  !dialog.exportSelectedSegmentFrames(frameExport, &error)))
            << "segment export failed: %s\n";
    cv::VideoCapture exportedClip(clipExport.toStdString());
    cv::Mat exportedFirst;
    ASSERT_FALSE(
            (!exportedClip.isOpened() ||
             int(exportedClip.get(cv::CAP_PROP_FRAME_COUNT)) != 2 ||
             !exportedClip.read(exportedFirst) || exportedFirst.empty() ||
             std::abs(int(exportedFirst.at<cv::Vec3b>(0, 0)[0]) - 140) > 4 ||
             QDir(frameExport)
                             .entryList({QStringLiteral("img_*.jpg")},
                                        QDir::Files)
                             .size() != 2))
            << "exported frame range is incorrect\n";
    const cv::Mat firstJpeg =
            cv::imread(QDir(frameExport)
                               .filePath(QStringLiteral("img_00001.jpg"))
                               .toStdString());
    ASSERT_FALSE((firstJpeg.empty() ||
                  std::abs(int(firstJpeg.at<cv::Vec3b>(0, 0)[0]) - 140) > 4 ||
                  !dialog.exportSelectedSegmentVideo(clipExport, &error)))
            << "exported clip replace or JPEG failed: %s\n";
    const QString cancelled = folder.filePath(QStringLiteral("cancelled"));
    ASSERT_FALSE((exportVideoSegment(
                          {path, cancelled, 1400, 1800, 5,
                           VideoSegmentExportFormat::RawFrames},
                          [](int, int) { return false; }, &error) ||
                  QDir(cancelled).exists()))
            << "cancelled export left output behind\n";
    bool saved = false;
    for (QPushButton* button : dialog.findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("Save")) {
            button->click();
            saved = true;
            break;
        }
    }
    VideoSidecar reread;
    ASSERT_FALSE((!saved || !loadVideoSidecar(path, &reread, &error) ||
                  reread.segments.size() != 2 || reread.labels.size() != 2 ||
                  reread.segments.first().label != QStringLiteral("running") ||
                  reread.segments.last().startMs != 1400 ||
                  reread.segments.last().endMs != 1800 ||
                  reread.segments.last().startFrame != 7 ||
                  reread.segments.last().endFrame != 9))
            << "timeline segment did not survive save: %s\n";
    auto* split =
            dialog.findChild<QPushButton*>(QStringLiteral("VideoSplitSegment"));
    auto* description = dialog.findChild<QPlainTextEdit*>();
    ASSERT_FALSE((!split || !description));
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
    ASSERT_FALSE((segments->count() != 3 || segments->currentRow() != 2))
            << "split did not select the second segment\n";
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
        FAIL() << "Contract precondition or invariant failed";
    }
    auto* undo = dialog.findChild<QPushButton*>(QStringLiteral("VideoUndo"));
    auto* redo = dialog.findChild<QPushButton*>(QStringLiteral("VideoRedo"));
    ASSERT_FALSE((!undo || !redo || !undo->isEnabled() || redo->isEnabled()))
            << "video edit history actions are unavailable\n";
    undo->click();
    ASSERT_FALSE((segments->count() != 2 || !redo->isEnabled()))
            << "video undo did not restore the original segment\n";
    redo->click();
    ASSERT_FALSE((segments->count() != 3 || !undo->isEnabled()))
            << "video redo did not restore the split\n";
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
    ASSERT_FALSE((!enter.isAccepted() || !dragMove.isAccepted() ||
                  !drop.isAccepted() || player->totalVideoFrames() != 10))
            << "Video drag and drop failed\n";
    ASSERT_FALSE((undo->isEnabled() || redo->isEnabled()))
            << "video history was not cleared after reopening\n";
    dialog.close();
}

int main(int argc, char** argv) {
    QTemporaryDir settingsDir;
    if (!settingsDir.isValid()) return 1;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settingsDir.path());
    QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope,
                       settingsDir.path());
    QCoreApplication::setOrganizationName(QStringLiteral("ACloudViewerTests"));
    QCoreApplication::setApplicationName(QStringLiteral("test_video_ui"));
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
