// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Test_Updater.h"
#include "Test_Downloader.h"
#include "Test_QSimpleUpdater.h"

int main(int argc, char *argv[])
{
   QApplication app(argc, argv);

   app.setApplicationName("QSimpleUpdater Tests");
   app.setOrganizationName("The QSimpleUpdater Library");

   QTest::qExec(new Test_Updater, argc, argv);
   QTest::qExec(new Test_Downloader, argc, argv);
   QTest::qExec(new Test_QSimpleUpdater, argc, argv);

   QTimer::singleShot(1000, Qt::PreciseTimer, qApp, SLOT(quit()));

   return app.exec();
}
