// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AssimpCollada.h"

AssimpCollada::AssimpCollada()
    : IoAbstractLoader({"_COLLADA Filter",
                        FileIOFilter::DEFAULT_PRIORITY,  // priority
                        QStringList{"dae"}, "dae",
                        QStringList{"qMeshIO - COLLADA file (*.dae)"},
                        QStringList(), Import}) {}
