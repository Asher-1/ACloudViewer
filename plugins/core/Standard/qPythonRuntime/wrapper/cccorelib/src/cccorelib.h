// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PYTHON_PLUGIN_CCCORELIB_H
#define PYTHON_PLUGIN_CCCORELIB_H

#include <pybind11/pybind11.h>

#include <AutoSegmentationTools.h>
#include <DgmOctree.h>
#include <GenericDistribution.h>
#include <TrueKdTree.h>

PYBIND11_MAKE_OPAQUE(std::vector<unsigned>)
PYBIND11_MAKE_OPAQUE(cloudViewer::ReferenceCloudContainer)

PYBIND11_MAKE_OPAQUE(cloudViewer::DgmOctree::NeighbourCellsSet)
PYBIND11_MAKE_OPAQUE(cloudViewer::DgmOctree::NeighboursSet)
PYBIND11_MAKE_OPAQUE(cloudViewer::GenericDistribution::ScalarContainer)

PYBIND11_MAKE_OPAQUE(cloudViewer::TrueKdTree::LeafVector)

#endif // PYTHON_PLUGIN_CCCORELIB_H
