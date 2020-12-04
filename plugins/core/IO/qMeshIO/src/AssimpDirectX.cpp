// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "AssimpDirectX.h"


AssimpDirectX::AssimpDirectX() :
   mioAbstractLoader( {
      "_DirectX Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{ "x" },
      "x",
      QStringList{ "qMeshIO - DirectX file (*.x)" },
      QStringList(),
      Import
   } )   
{   
}
