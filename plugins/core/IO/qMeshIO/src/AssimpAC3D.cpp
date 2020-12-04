// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "AssimpAC3D.h"


AssimpAC3D::AssimpAC3D() :
   mioAbstractLoader( {
      "_AC3D Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{ "ac" },
      "ac",
      QStringList{ "qMeshIO - AC3D file (*.ac)" },
      QStringList(),
      Import
   } )   
{   
}
