// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

#include "OpenPCDetModelCatalog.h"

class ccHObject;
class ccPointCloud;

/** Pure constructors for the DB-tree detection exports. Split from
 *  qOpenPCDet::addBoxesToDb / exportAnnotatedCloud so the metadata and
 *  coloring contract is unit-testable without an ecvMainAppInterface
 *  (the plugin methods only add the returned object to the DB tree). */

/** One per-class-colored wireframe polyline per detection inside a group
 *  carrying the OpenPCDet metadata (Model/Count/Runtime/Device/Scene...).
 *  Returns null only on allocation failure (partial groups are cleaned up). */
ccHObject* BuildDetectionGroup(const OpenPCDetRunResult& result,
                               const QString& modelTag);

/** Clone of \p source with in-box points tinted by detection class color.
 *  \p tintedOut receives the number of tinted points. Returns null with
 *  \p error set on clone/RGB-allocation failure. */
ccPointCloud* BuildAnnotatedCloud(ccPointCloud* source,
                                  const OpenPCDetRunResult& result,
                                  const QString& modelTag,
                                  unsigned* tintedOut,
                                  QString* error);
