// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

class ccHObject;

namespace spirula {

//! Resolves a user pick into an importable splat file:
//!   run directory      -> <dir>/splat.ply
//!   step-*.ckpt dir    -> its splat.ply (same rule as upstream --resume)
//!   .ply file          -> itself
QString resolveSplatPath(const QString& picked);

//! Light parser for a 3DGS PLY (binary_little_endian; x/y/z + f_dc_0..2 +
//! opacity, scale_0..2 optional). Produces a colored ccPointCloud
//! (SH0 -> RGB, low-opacity gaussians pruned) with Opacity/Scale scalar
//! fields; the caller owns the returned object (nullptr on error).
//!
//! Memory: parses in one pass into a temporary per-vertex buffer
//! (~27 bytes/vertex; 10M gaussians ~= 270 MB) before building the cloud,
//! which keeps the disk read single-pass without a second scan for the
//! opacity prune.
ccHObject* loadSplatPointCloud(const QString& plyPath, QString* errorOut);

//! Loads a mesh.ply through the regular I/O stack (FileIOFilter auto-detect).
ccHObject* loadMesh(const QString& plyPath, QString* errorOut);

}  // namespace spirula
