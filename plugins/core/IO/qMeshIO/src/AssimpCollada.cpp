// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "AssimpCollada.h"


AssimpCollada::AssimpCollada() :
   IoAbstractLoader( {
      "_COLLADA Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{ "dae" },
      "dae",
      QStringList{ "qMeshIO - COLLADA file (*.dae)" },
      QStringList(),
      Import
   } ) 
{    
}
