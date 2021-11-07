// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/reconstruction.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/reconstruction/reconstruction_options.h"
#include "pybind/reconstruction/feature/feature.h"
#include "pybind/reconstruction/database/database.h"
#include "pybind/reconstruction/image/image.h"
#include "pybind/reconstruction/model/model.h"
#include "pybind/reconstruction/mvs/multi_views_stereo.h"
#include "pybind/reconstruction/gui/gui.h"
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
