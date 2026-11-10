// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QPointer>
#include <cstdio>

#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

int main(int argc, char** argv) {
    // Each application must release its network resources before DLL teardown.
    // Recreating the application must also recreate the shared repository.
    for (int cycle = 0; cycle < 2; ++cycle) {
        QPointer<ecvTestDataRepository> repository;
        QPointer<ecvModelDownloader> downloader;
        {
            QCoreApplication app(argc, argv);
            repository = &ecvTestDataRepository::instance();
            downloader = repository->findChild<ecvModelDownloader*>();
            if (!downloader ||
                repository != &ecvTestDataRepository::instance()) {
                std::fprintf(stderr, "cycle %d: shared service missing\n", cycle);
                return 1;
            }
        }
        if (repository || downloader) {
            std::fprintf(stderr,
                         "cycle %d: network service outlived the application\n",
                         cycle);
            return 1;
        }
    }
    std::puts("dataset repository application lifetime passed");
    return 0;
}
