// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "AssimpCommonFilter.h"


AssimpCommonFilter::AssimpCommonFilter() :
   mioAbstractLoader( {
      "_AssimpCommonFormat Filter",
      FileIOFilter::DEFAULT_PRIORITY,	// priority
      QStringList{"lws", "lwo", "lxo", "ms3d", "cob", "scn", "xgl", "zgl", "ifc", "ply", "dxf",
                "irrmesh", "irr", "md1", "md2", "md3", "pk3", "mdc", "md5", "smd", "obj", "stl", "off",
                "mdl", "vta", "ogex", "3d", "b3d", "q3d", "q3s", "nff", "raw", "ter", "hmp", "ndo"},
      "ms3d",
      QStringList{ "qMeshIO - AssimpCommonFormat file "
      "( *.lws *.lwo *.lxo *.ms3d *.cob *.scn *.xgl *.zgl *.ifc *.obj *.ply" 
       " *.irrmesh *.irr *.md1 *.md2 *.md3 *.pk3 *.mdc *.md5 *.smd *.mdl *.stl"
       " *.vta *.ogex *.3d *.b3d *.q3d *.q3s *.nff *.raw *.ter *.hmp *.ndo *.dxf *.off)" },
      QStringList(),
      Import
   } )   
{   
}
