// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMeshGroup.h"

// system
#include <assert.h>

void ccMeshGroup::drawMeOnly(CC_DRAW_CONTEXT& context) {
    // does nothing
    return;
}

bool ccMeshGroup::toFile_MeOnly(QFile& out) const {
    CVLog::Error("[Mesh groups are not handled any more!");
    return false;
}

bool ccMeshGroup::fromFile_MeOnly(QFile& in,
                                  short dataVersion,
                                  int flags,
                                  LoadedIDMap& oldToNewIDMap) {
    // Mesh groups are deprecated since version 2.9
    assert(dataVersion < 29);
    if (dataVersion >= 29) return false;

    if (!ccGenericMesh::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    /*** we simply read the data as it was before, so as to be able to read the
     * other entities from the file! ***/

    // as the associated cloud (=vertices) can't be saved directly (as it may be
    // shared by multiple meshes) we only store its unique ID (dataVersion>=20)
    // --> we hope we will find it at loading time (i.e. this is the
    // responsibility of the caller to make sure that all dependencies are saved
    // together)
    uint32_t vertUniqueID = 0;
    if (in.read((char*)&vertUniqueID, 4) < 0) return ReadError();
    //[DIRTY] WARNING: temporarily, we set the vertices unique ID in the
    //'m_associatedCloud' pointer!!!
    //*(uint32_t*)(&m_associatedCloud) = vertUniqueID;

    // per-triangle normals array (dataVersion>=20)
    {
        // as the associated normals array can't be saved directly (as it may be
        // shared by multiple meshes) we only store its unique ID
        // (dataVersion>=20) --> we hope we will find it at loading time (i.e.
        // this is the responsibility of the caller to make sure that all
        // dependencies are saved together)
        uint32_t normArrayID = 0;
        if (in.read((char*)&normArrayID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the array unique ID in the
        //'m_triNormals' pointer!!!
        //*(uint32_t*)(&m_triNormals) = normArrayID;
    }

    // texture coordinates array (dataVersion>=20)
    {
        // as the associated texture coordinates array can't be saved directly
        // (as it may be shared by multiple meshes) we only store its unique ID
        // (dataVersion>=20) --> we hope we will find it at loading time (i.e.
        // this is the responsibility of the caller to make sure that all
        // dependencies are saved together)
        uint32_t texCoordArrayID = 0;
        if (in.read((char*)&texCoordArrayID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the array unique ID in the
        //'m_texCoords' pointer!!!
        //*(uint32_t*)(&m_texCoords) = texCoordArrayID;
    }

    // materials
    {
        // as the associated materials can't be saved directly (as it may be
        // shared by multiple meshes) we only store its unique ID
        // (dataVersion>=20) --> we hope we will find it at loading time (i.e.
        // this is the responsibility of the caller to make sure that all
        // dependencies are saved together)
        uint32_t matSetID = 0;
        if (in.read((char*)&matSetID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the array unique ID in the
        //'m_materials' pointer!!!
        //*(uint32_t*)(&m_materials) = matSetID;
    }

    return true;
}
