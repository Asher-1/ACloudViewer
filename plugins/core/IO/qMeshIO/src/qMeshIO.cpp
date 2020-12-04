// Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "assimp/version.h"

#include "qMeshIO.h"

#include "COLLADA.h"
#include "glTF.h"
#include "IFC.h"

#include "AssimpCommonFilter.h"
#include "AssimpAC3D.h"
#include "AssimpBlender.h"
#include "AssimpDirectX.h"
#include "AssimpFBX.h"
#include "Assimp3DMax.h"


qMeshIO::qMeshIO(QObject *parent) :
   QObject( parent ),
   ccIOPluginInterface( ":/asmaloney/qMeshIO/info.json" )
{
   const QString    cAssimpVer = QStringLiteral( "[qMeshIO] Using Assimp %1.%2 (%3-%4)" )
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
        FileIOFilter::Shared( new COLLADAFilter ),
        FileIOFilter::Shared( new glTFFilter ),
        FileIOFilter::Shared( new IFCFilter ),
        FileIOFilter::Shared( new AssimpCommonFilter),
        FileIOFilter::Shared( new AssimpFBX ),
        FileIOFilter::Shared( new AssimpAC3D ),
        FileIOFilter::Shared( new Assimp3DMax ),
        FileIOFilter::Shared( new AssimpBlender ),
        FileIOFilter::Shared( new AssimpDirectX ),
   };
}
