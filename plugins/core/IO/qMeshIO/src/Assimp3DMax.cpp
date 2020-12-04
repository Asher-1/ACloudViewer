// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "Assimp3DMax.h"


Assimp3DMax::Assimp3DMax() :
   mioAbstractLoader( {
      "_3DMax Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{ "3ds", "ase" },
      "3ds",
      QStringList{ "qMeshIO - 3DMax file (*.3ds *.ase)" },
      QStringList(),
      Import
   } )   
{   
}
