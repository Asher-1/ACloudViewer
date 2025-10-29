// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qMeshIO.h"

#include "Assimp3DMax.h"
#include "AssimpBlender.h"
#include "AssimpCollada.h"
#include "AssimpCommonFilter.h"
#include "AssimpGltf.h"
#include "AssimpIfc.h"
#include "assimp/version.h"

qMeshIO::qMeshIO(QObject *parent)
    : QObject(parent), ccIOPluginInterface(":/asmaloney/qMeshIO/info.json") {
    const QString cAssimpVer =
            QStringLiteral("\t[qMeshIO] Using Assimp %1.%2 (%3-%4)")
                    .arg(QString::number(aiGetVersionMajor()),
                         QString::number(aiGetVersionMinor()))
                    .arg(aiGetVersionRevision(), 0, 16)
                    .arg(aiGetBranchName());

    CVLog::Print(cAssimpVer);
}

void qMeshIO::registerCommands(ccCommandLineInterface *inCmdLine) {
    Q_UNUSED(inCmdLine);
}

ccIOPluginInterface::FilterList qMeshIO::getFilters() {
    return {
            FileIOFilter::Shared(new AssimpIfc),
            FileIOFilter::Shared(new Assimp3DMax),
            FileIOFilter::Shared(new AssimpGltf),
            FileIOFilter::Shared(new AssimpBlender),
            FileIOFilter::Shared(new AssimpCollada),
            FileIOFilter::Shared(new AssimpCommonFilter),
    };
}
