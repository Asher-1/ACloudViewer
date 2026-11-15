// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace reconstruction {
namespace feature {

void pybind_feature(py::module &m);

// Binds the pipelines/feature.h entry points on the ``feature`` submodule.
// Their default-argument values (FeatureExtractionOptions,
// FeatureMatchingOptions, TwoViewGeometryOptions, *PairingOptions) must
// already be registered, so this runs after the feature, estimators and
// pipeline binders.
void pybind_feature_methods(py::module &m);

}  // namespace feature
}  // namespace reconstruction
}  // namespace cloudViewer