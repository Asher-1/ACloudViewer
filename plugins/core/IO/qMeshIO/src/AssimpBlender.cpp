// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "AssimpBlender.h"


AssimpBlender::AssimpBlender() :
   mioAbstractLoader( {
      "_Blender Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{ "blend" },
      "blend",
      QStringList{ "qMeshIO - Blender file (*.blend)" },
      QStringList(),
      Import
   } )   
{   
}
