// Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "assimp/version.h"

#include "qMeshIO.h"

#include "AssimpIfc.h"
#include "AssimpGltf.h"
#include "Assimp3DMax.h"
#include "AssimpCollada.h"
#include "AssimpBlender.h"
#include "AssimpCommonFilter.h"


qMeshIO::qMeshIO(QObject *parent) :
   QObject( parent ),
   ccIOPluginInterface( ":/asmaloney/qMeshIO/info.json" )
{
   const QString    cAssimpVer = QStringLiteral( "\t[qMeshIO] Using Assimp %1.%2 (%3-%4)" )
                                 .arg( QString::number( aiGetVersionMajor() ),
                                       QString::number( aiGetVersionMinor() ) )
                                 .arg( aiGetVersionRevision(), 0, 16 )
                                 .arg( aiGetBranchName() );
   
   CVLog::Print( cAssimpVer );
}

void qMeshIO::registerCommands( ccCommandLineInterface *inCmdLine )
{
   Q_UNUSED( inCmdLine );
}

ccIOPluginInterface::FilterList qMeshIO::getFilters()
{
   return {
        FileIOFilter::Shared( new AssimpIfc ),
        FileIOFilter::Shared( new Assimp3DMax ),
        FileIOFilter::Shared( new AssimpGltf ),
        FileIOFilter::Shared( new AssimpBlender ),
        FileIOFilter::Shared( new AssimpCollada ),
        FileIOFilter::Shared( new AssimpCommonFilter ),
   };
}
