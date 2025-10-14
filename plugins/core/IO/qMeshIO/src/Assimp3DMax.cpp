// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Assimp3DMax.h"

Assimp3DMax::Assimp3DMax()
    : IoAbstractLoader({"_3DMax Filter",
                        FileIOFilter::DEFAULT_PRIORITY,  // priority
                        QStringList{"3ds", "ase"}, "3ds",
                        QStringList{"qMeshIO - 3DMax file (*.3ds *.ase)"},
                        QStringList(), Import}) {}
