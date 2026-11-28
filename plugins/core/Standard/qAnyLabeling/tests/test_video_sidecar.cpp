// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>

#include "VideoSidecar.h"

class VideoSidecarTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(folder.isValid());
        video = folder.filePath(QStringLiteral("clip.mp4"));
        sidecar = videoSidecarPath(video);
        data.fps = 29.97;
        data.durationMs = 4000;
        data.width = 1920;
        data.height = 1080;
        data.video = QStringLiteral("clip.mp4");
        data.labels.append(QStringLiteral("running"));
        data.labelColors.insert(QStringLiteral("running"),
                                QStringLiteral("#EF476F"));
        data.segments.append(
                {QStringLiteral("s1234567890"), QStringLiteral("running"), 1000,
                 2500, videoFrameAt(1000, data.fps),
                 videoFrameAt(2500, data.fps), QStringLiteral("fast")});
    }
    QTemporaryDir folder;
    QString video;
    QString sidecar;
    VideoSidecar data;
};

TEST_F(VideoSidecarTest, RoundTripsCurrentSchemaAndFrameRate) {
    QString error;
    ASSERT_TRUE(saveVideoSidecar(video, data, &error)) << error.toStdString();
    VideoSidecar loaded;
    ASSERT_TRUE(loadVideoSidecar(video, &loaded, &error))
            << error.toStdString();
    EXPECT_EQ(loaded.toJson(), data.toJson());
    EXPECT_EQ(videoFrameAt(1000, 29.97), 30);
}

TEST_F(VideoSidecarTest, LoadsAndRewritesLegacyMinimalSchema) {
    QFile legacy(sidecar);
    ASSERT_TRUE(legacy.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_GE(legacy.write("{\"video\":\"clip.mp4\",\"segments\":[]}"), 0);
    legacy.close();
    VideoSidecar loaded;
    QString error;
    ASSERT_TRUE(loadVideoSidecar(video, &loaded, &error))
            << error.toStdString();
    EXPECT_EQ(loaded.video, QStringLiteral("clip.mp4"));
    EXPECT_TRUE(loaded.segments.isEmpty());
    EXPECT_TRUE(loaded.labels.isEmpty());
    EXPECT_TRUE(saveVideoSidecar(video, loaded, &error)) << error.toStdString();
}

TEST_F(VideoSidecarTest, RefusesToOverwriteForeignAnnotationSidecar) {
    QFile foreign(sidecar);
    ASSERT_TRUE(foreign.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_GE(foreign.write("{\"version\": \"5.0.1\", \"shapes\": []}"), 0);
    foreign.close();
    QString error;
    EXPECT_FALSE(saveVideoSidecar(video, data, &error));
    ASSERT_TRUE(foreign.open(QIODevice::ReadOnly));
    EXPECT_TRUE(foreign.readAll().contains("shapes"));
}
