// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "AssimpFBX.h"


AssimpFBX::AssimpFBX() :
   mioAbstractLoader( {
      "_FBX Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{ "fbx"},
      "fbx",
      QStringList{ "qMeshIO - FBX file (*.fbx)" },
      QStringList(),
      Import
   } )   
{   
}
