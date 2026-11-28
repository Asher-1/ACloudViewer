// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QPointer>

#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

TEST(DatasetLifecycle,
     RecreatingApplicationRecreatesAndReleasesNetworkServices) {
    for (int cycle = 0; cycle < 2; ++cycle) {
        SCOPED_TRACE(cycle);
        QPointer<ecvTestDataRepository> repository;
        QPointer<ecvModelDownloader> downloader;
        {
            int argc = 1;
            char program[] = "dataset-lifecycle-test";
            char* argv[] = {program, nullptr};
            QCoreApplication app(argc, argv);
            repository = &ecvTestDataRepository::instance();
            downloader = repository->findChild<ecvModelDownloader*>();
            ASSERT_TRUE(downloader);
            ASSERT_EQ(repository.data(), &ecvTestDataRepository::instance());
        }
        EXPECT_TRUE(repository.isNull());
        EXPECT_TRUE(downloader.isNull());
    }
}
