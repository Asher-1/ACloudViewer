// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QProgressDialog>

#ifdef QT_DEBUG
#include <iostream>
#endif

#include "ecvMesh.h"
#include "ecvMaterialSet.h"

#include "assimp/DefaultLogger.hpp"
#include "assimp/Importer.hpp"
#include "assimp/ProgressHandler.hpp"

#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include "IoAbstractLoader.h"
#include "IoUtils.h"

namespace
{
   class _LogStream : public Assimp::LogStream
   {
      void	write( const char *inMessage ) override
      {
         QString   message = QString( inMessage ).trimmed();
         int	messageLevel = CVLog::LOG_STANDARD;
      
         if ( message.startsWith( "Warn" ) )
         {
            messageLevel = CVLog::LOG_WARNING;
         }
         else if ( message.startsWith( "Error" ) )
         {
            messageLevel = CVLog::LOG_ERROR;
         }
         else if ( message.startsWith( "Info" ) )
         {
            messageLevel = CVLog::LOG_VERBOSE;
         }
      
         message.prepend( "[qMeshIO] ai - " );
      
         CVLog::LogMessage( message, messageLevel );
      }
   };

   class _ProgressHandler : public QProgressDialog, public Assimp::ProgressHandler
   {
    public:
      _ProgressHandler( const QString &inText ) :
         cText( inText )
      {
         setWindowModality( Qt::WindowModal );
         setWindowTitle( tr( "Import Mesh" ) );
      
         setMinimumDuration( 0 );
      
         setMinimumSize( 400, 100 );
      }
   
      bool	Update( float inPercent ) override
      {
         int   value = qRound( inPercent * 100.0f );
      
         setLabelText( QStringLiteral( "Loading %1: %2%" ).arg( cText, QString::number( value ) ) );
      
         setValue( value );
      
         return wasCanceled();
      }
   
    private:
      const QString    cText;
   };

   class _Loader
   {
    public:
      _Loader( const aiScene *inScene, const QString &inFileName, const QString &inPath ) :
         cScene( inScene ),
         cFileName( inFileName ),
         cPath( inPath )
      {
         _initCameraNames();
      }
   
      void	load( ccHObject &ioContainer )
      {
         if ( cScene->HasMeshes() )
         {
            CVLog::Print( QStringLiteral( "[qMeshIO] The file '%1' has %2 meshes" ).arg(
               cFileName,
               QLocale::system().toString( cScene->mNumMeshes ) ) );
         }
      
         if ( cScene->HasCameras() )
         {
            CVLog::Print( QStringLiteral( "[qMeshIO] The file '%1' has %2 cameras" ).arg(
               cFileName,
               QLocale::system().toString( cScene->mNumCameras ) ) );
         }
      
         _recursiveAddNode( cScene->mRootNode, cScene, &ioContainer );
      
         ioContainer.applyGLTransformation_recursive();
         ioContainer.resetGLTransformationHistory();
      
         _pruneTree( &ioContainer );
      }
   
    private:
      void	_initCameraNames()
      {
         for ( unsigned int i = 0; i < cScene->mNumCameras; ++i )
         {
            const auto     cCamera = cScene->mCameras[i];
            const QString	cCameraName( cCamera->mName.C_Str() );
         
            mCameraMap[cCameraName] = cCamera;
         }
      }
   
      void	_recursiveAddNode( const aiNode *inNode, const aiScene *inScene, ccHObject *ioParentObject )
      {
   #ifdef QT_DEBUG
         std::cout << "Process node: " << inNode->mName.C_Str() << std::endl;
         std::cout << "  num children: " << inNode->mNumChildren << std::endl;
         std::cout << "  num meshes: " << inNode->mNumMeshes << std::endl;
   #endif
      
         auto currentObject = new ccHObject( inNode->mName.C_Str() );
      
         ioParentObject->addChild( currentObject );
      
         const bool    cNodeHasTransform = !inNode->mTransformation.IsIdentity();
      
         if ( cNodeHasTransform )
         {
            ccGLMatrix	transform = IoUtils::convertMatrix( inNode->mTransformation );
         
            currentObject->setGLTransformation( transform );
         }
      
         // meshes
         for ( unsigned int j = 0; j < inNode->mNumMeshes; ++j )
         {
            const auto	cMeshIndex = inNode->mMeshes[j];
            const auto	mesh = inScene->mMeshes[cMeshIndex];
         
            ccMesh  *newMesh = IoUtils::newCCMeshFromAIMesh( mesh );
         
            if ( newMesh == nullptr )
            {
               continue;
            }
         
            auto    materialSet = IoUtils::createMaterialSetForMesh( mesh, cPath, inScene );
         
            if ( materialSet != nullptr )
            {
               newMesh->setMaterialSet( materialSet );
               newMesh->showMaterials( true );
            }
         
            currentObject->addChild( newMesh );
         }
      
         // metadata
         if ( inNode->mMetaData != nullptr )
         {
            const auto	data = inNode->mMetaData;
         
            for ( unsigned int i = 0; i < data->mNumProperties; ++i )
            {
               const auto   cMetaKey = data->mKeys[i].C_Str();
               QVariant     metaValue = IoUtils::convertMetaValueToVariant( data, i );
            
   #ifdef QT_DEBUG
               std::cout << "Setting meta: " << cMetaKey << " = " << metaValue.toString().toLatin1().constData() << std::endl;
   #endif
      
               currentObject->setMetaData( cMetaKey, metaValue );
            }
         }
            
         for ( unsigned int i = 0; i < inNode->mNumChildren; ++i )
         {
            const auto     cChild = inNode->mChildren[i];
         
            _recursiveAddNode( cChild, inScene, currentObject );
         }
      }
   
      void	_pruneTree( ccHObject *ioCurrentObject )
      {
         auto    childCount = ioCurrentObject->getChildrenNumber();
      
         std::vector<ccHObject *>	children;
      
         // Because the indices change when we delete children, save a list and process that instead
         for ( unsigned int i = 0; i < childCount; ++i )
         {
            children.push_back( ioCurrentObject->getChild( i ) );
         }
      
         for ( auto child : children )
         {
            _pruneTree( child );
         }
      
         // If we are not a "naked" hierarchy object, then we contain useful info, so return
         if ( ioCurrentObject->getClassID() != CV_TYPES::HIERARCHY_OBJECT )
         {
            return;
         }
      
         // If we don't have a parent, then we are the top level, so return
         auto  parent = ioCurrentObject->getParent();
      
         if ( parent == nullptr )
         {
            return;
         }
      
         // Our child count will be different now if we deleted some objects
         childCount = ioCurrentObject->getChildrenNumber();;
      
         // If we don't have children, then we can be pruned
         if ( childCount == 0 )
         {
   #ifdef QT_DEBUG
            std::cout << "pruning: " << ioCurrentObject->getName().toLatin1().constData()
                      << "  from parent: " << parent->getName().toLatin1().constData() << std::endl;
   #endif
         
            parent->detachChild( ioCurrentObject );
         
            delete ioCurrentObject;
         }
         else if ( (childCount == 1) &&
                   ioCurrentObject->metaData().empty() )
         {
            // If we have one child, and it doesn't have useful data,
            // Then we can reparent it
         
            auto	child = ioCurrentObject->getChild( 0 );
         
            if ( child != nullptr )
            {
   #ifdef QT_DEBUG
               std::cout << "reparenting: " << child->getName().toLatin1().constData()
                         << " from " << ioCurrentObject->getName().toLatin1().constData()
                         << " to " << parent->getName().toLatin1().constData()
                         << std::endl;
   #endif
            
               ioCurrentObject->detachChild( child );
               child->setName( ioCurrentObject->getName() );
            
               parent->addChild( child );
            
               delete ioCurrentObject;
            }
         }
      }
   
      const aiScene    *cScene;
      const QString    cFileName;
      const QString    cPath;
   
      QMap<QString, const aiCamera *>  mCameraMap;
   };
}


IoAbstractLoader::IoAbstractLoader( const FileIOFilter::FilterInfo &info ) :
   FileIOFilter( info )
{
}

bool IoAbstractLoader::canSave( CV_CLASS_ENUM type, bool &multiple, bool &exclusive ) const
{
   Q_UNUSED( type );
   Q_UNUSED( multiple );
   Q_UNUSED( exclusive );
   
   return false;
}

CC_FILE_ERROR IoAbstractLoader::loadFile( const QString &inFileName, ccHObject &ioContainer, FileIOFilter::LoadParameters &inParameters )
{
   Q_UNUSED( inParameters )
	
   const auto	cFileName = QFileInfo( inFileName ).fileName();
   const auto	cPath = QFileInfo( inFileName ).absoluteDir().path();
   
   CVLog::Print( QStringLiteral( "[qMeshIO] Loading file '%1'" ).arg( inFileName ) );
   
   Assimp::DefaultLogger::create( "", Assimp::Logger::NORMAL, aiDefaultLogStream_STDOUT );
   
   unsigned int	loggingSeverity = Assimp::Logger::Err | Assimp::Logger::Warn;
   
#ifdef QT_DEBUG
   loggingSeverity |= Assimp::Logger::Info;
   loggingSeverity |= Assimp::Logger::Debugging;
#endif
   
   Assimp::DefaultLogger::get()->attachStream( new _LogStream, loggingSeverity );
   
   Assimp::Importer importer;
   
   importer.SetProgressHandler( new _ProgressHandler( cFileName ) );
   
   // removes things we don't care about from the import
   importer.SetPropertyInteger( AI_CONFIG_PP_RVC_FLAGS,
                                aiComponent_ANIMATIONS |
                                aiComponent_BONEWEIGHTS |
                                aiComponent_CAMERAS |
                                aiComponent_LIGHTS );
   
   importer.SetPropertyBool( AI_CONFIG_IMPORT_NO_SKELETON_MESHES, true );
   importer.SetPropertyBool( AI_CONFIG_IMPORT_COLLADA_USE_COLLADA_NAMES, true );
   importer.SetPropertyBool( AI_CONFIG_PP_FD_REMOVE, true );
   importer.SetPropertyBool( AI_CONFIG_PP_FID_IGNORE_TEXTURECOORDS, true );
   
   const aiScene *cScene = importer.ReadFile( inFileName.toStdString(),
											  aiProcess_FindInvalidData |
											  aiProcess_JoinIdenticalVertices |
											  aiProcess_RemoveComponent |
											  aiProcess_Triangulate |
											  aiProcess_ValidateDataStructure );
   
   if ( cScene == nullptr )
   {
      CVLog::Warning( QStringLiteral( "[qMeshIO] The file '%1' has errors: %2" ).arg( cFileName, importer.GetErrorString() ) );
      
      Assimp::DefaultLogger::kill();
      
      return CC_FERR_READING;
   }
   
   _Loader  loader( cScene, cFileName, cPath );
   
   loader.load( ioContainer );
   
   Assimp::DefaultLogger::kill();
   
   // allow individual loaders to do some processing on the results
   _postProcess( ioContainer );
   
   return CC_FERR_NO_ERROR;
}

void IoAbstractLoader::_postProcess( ccHObject &ioContainer )
{
   Q_UNUSED( ioContainer );
}