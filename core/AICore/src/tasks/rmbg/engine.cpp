// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/rmbg/rmbg.hpp"
#include "tasks/rmbg/rmbg_graph.hpp"
#include "tasks/rmbg/rmbg_preprocess.hpp"

namespace rmbg {

bool remove_background(Model& m,
                       const void* image_bytes,
                       int image_len,
                       std::vector<uint8_t>& out_png,
                       std::string& err) {
    if (!m.backend || !m.graph || !m.graph_ready) {
        err = "model not loaded";
        return false;
    }
    std::vector<uint8_t> rgba;
    std::vector<float> input, alpha;
    int width = 0, height = 0;
    if (!decode_preprocess(image_bytes, image_len, m.cfg.input_size, m.cfg.mean,
                           m.cfg.std, rgba, width, height, input, err))
        return false;
    if (!m.graph->forward(input, alpha, err)) return false;
    return encode_result_png(rgba, width, height, alpha, m.cfg.input_size,
                             m.cfg.input_size, out_png, err);
}

}  // namespace rmbg
