// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/reconstruction.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/reconstruction/database/database.h"
#include "pybind/reconstruction/feature/feature.h"
#include "pybind/reconstruction/gui/gui.h"
#include "pybind/reconstruction/image/image.h"
#include "pybind/reconstruction/model/model.h"
#include "pybind/reconstruction/mvs/multi_views_stereo.h"
#include "pybind/reconstruction/reconstruction_options.h"
#include "pybind/reconstruction/sfm/structure_from_motion.h"
#include "pybind/reconstruction/vocab_tree/vocab_tree.h"

namespace cloudViewer {
namespace reconstruction {

void pybind_reconstruction(py::module& m) {
    py::module m_reconstruction = m.def_submodule("reconstruction");
    options::pybind_reconstruction_options(m_reconstruction);
    database::pybind_database(m_reconstruction);
    feature::pybind_feature(m_reconstruction);
    image::pybind_image(m_reconstruction);
    model::pybind_model(m_reconstruction);
    mvs::pybind_multi_views_stereo(m_reconstruction);
    sfm::pybind_structure_from_motion(m_reconstruction);
    vocab_tree::pybind_vocab_tree(m_reconstruction);
    gui::pybind_gui(m_reconstruction);
}

}  // namespace reconstruction
}  // namespace cloudViewer
