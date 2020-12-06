// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "AssimpGltf.h"


AssimpGltf::AssimpGltf() :
   IoAbstractLoader( {
      "_glTF Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{ "gltf", "glb" },
      "gltf",
      QStringList{ "qMeshIO - glTF file (*.gltf *.glb)" },
      QStringList(),
      Import
   } )   
{   
}
