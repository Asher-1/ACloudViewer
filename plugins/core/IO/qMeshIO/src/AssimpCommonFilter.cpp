// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AssimpCommonFilter.h"

AssimpCommonFilter::AssimpCommonFilter()
    : IoAbstractLoader(
              {"_AssimpCommonFormat Filter",
               FileIOFilter::DEFAULT_PRIORITY,  // priority
               QStringList{"lws", "lwo", "lxo", "ms3d", "cob", "scn",     "xgl",
                           "zgl", "ifc", "ply", "dxf",  "fbx", "irrmesh", "irr",
                           "md1", "md2", "md3", "pk3",  "mdc", "md5",     "smd",
                           "obj", "stl", "off", "mdl",  "vta", "ogex",    "3d",
                           "b3d", "q3d", "q3s", "nff",  "raw", "ter",     "hmp",
                           "ndo", "ac",  "x"},
               "ms3d",
               QStringList{"qMeshIO - AssimpCommonFormat file "
                           "( *.lws *.lwo *.lxo *.ms3d *.cob *.scn *.xgl *.zgl "
                           "*.ifc *.obj *.ply *.fbx"
                           " *.irrmesh *.irr *.md1 *.md2 *.md3 *.pk3 *.mdc "
                           "*.md5 *.smd *.mdl *.stl *.ac"
                           " *.vta *.ogex *.3d *.b3d *.q3d *.q3s *.nff *.raw "
                           "*.ter *.hmp *.ndo *.dxf *.off *.x)"},
               QStringList(), Import}) {}
