// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ccCorkDlg.h"

#include <mesh/corkMesh.h>

#include <ecvMesh.h>

#include <QString>

class ecvMainAppInterface;

bool ToCorkMesh(const ccMesh* in,
                CorkMesh& out,
                ecvMainAppInterface* app = nullptr);
ccMesh* FromCorkMesh(const CorkMesh& in,
                      ecvMainAppInterface* app = nullptr);
bool qCorkPerformBooleanOp(ccCorkDlg::CSG_OPERATION operation,
                           CorkMesh& corkA,
                           CorkMesh& corkB,
                           const QString& nameA,
                           const QString& nameB,
                           ecvMainAppInterface* app);
