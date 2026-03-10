// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file vtkRenderWindowInteractorFix.h
 *  @brief Factory for vtkRenderWindowInteractor with platform-specific fixes
 */

#include <vtkRenderWindowInteractor.h>

/// @return New vtkRenderWindowInteractor instance (caller owns, use Delete())
vtkRenderWindowInteractor* vtkRenderWindowInteractorFixNew();
