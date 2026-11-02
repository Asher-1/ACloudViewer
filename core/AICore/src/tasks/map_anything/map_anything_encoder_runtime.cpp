// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "map_anything_encoder_runtime.hpp"

namespace map_anything {

EncoderRuntime::~EncoderRuntime() {
    // The graph runtime references staged weights that live in the model's
    // staged backend buffer, and both sit on the lease's physical device:
    // destroy in dependency order (runtime -> model -> backend).
    teardown_runtime();
    if (model_) {
        model_.reset();
    }
    if (be_.valid()) {
        free_backend(be_);
    }
}

void EncoderRuntime::teardown_runtime() {
    if (rt_) {
        rt_.reset();
    }
    cur_s_ = cur_h_ = cur_w_ = 0;
}

void EncoderRuntime::teardown_all() {
    teardown_runtime();
    model_.reset();
    if (be_.valid()) {
        free_backend(be_);
        be_ = Backend{};
    }
}

bool EncoderRuntime::load(const Config& cfg, std::string* error) {
    teardown_all();
    cfg_ = cfg;
    error_.clear();

    model_ = load_gguf(cfg_.encoder_gguf);
    if (!model_) {
        error_ = "failed to load the encoder GGUF: " + cfg_.encoder_gguf;
        if (error) *error = error_;
        return false;
    }

    std::string be_err;
    be_ = init_backend(cfg_.device, cfg_.n_threads, &be_err);
    if (!be_.valid()) {
        model_.reset();
        error_ = be_err.empty() ? "encoder backend initialization failed"
                                : be_err;
        if (error) *error = error_;
        return false;
    }

    ropts_ = RuntimeOptions{};
    ropts_.expose_tokens = true;
    ropts_.f32_accuracy = cfg_.f32_accuracy;
    ropts_.dump_dir = cfg_.dump_dir;
    if (error) error->clear();
    return true;
}

bool EncoderRuntime::ready_for(int n_views, int height, int width) const {
    return rt_ != nullptr && rt_->valid() && cur_s_ == n_views &&
           cur_h_ == height && cur_w_ == width;
}

bool EncoderRuntime::ensure_geometry(int n_views, int height, int width,
                                     std::string* error) {
    error_.clear();
    if (!valid()) {
        error_ = "encoder runtime is not loaded";
        if (error) *error = error_;
        return false;
    }
    if (ready_for(n_views, height, width)) {
        if (error) error->clear();
        return true;
    }
    // rebuild for the new geometry; weights re-stage (the map_anything capi
    // contract — sweeping resolutions pays the staging cost each time)
    teardown_runtime();
    rt_.reset(new VGGTRuntime(*model_, be_, n_views, height, width, ropts_));
    if (!rt_->valid()) {
        error_ = rt_->error().empty()
                         ? "encoder runtime failed to build for the requested "
                           "geometry"
                         : rt_->error();
        rt_.reset();
        if (error) *error = error_;
        return false;
    }
    cur_s_ = n_views;
    cur_h_ = height;
    cur_w_ = width;
    if (error) error->clear();
    return true;
}

bool EncoderRuntime::run(const float* images_f32, VGGTOutputs& out,
                         std::string* error) {
    error_.clear();
    if (!rt_ || !rt_->valid()) {
        error_ = "encoder runtime is not built for the current geometry";
        if (error) *error = error_;
        return false;
    }
    if (!rt_->run(images_f32, out)) {
        error_ = rt_->error().empty() ? "encoder run failed"
                                      : rt_->error();
        if (error) *error = error_;
        return false;
    }
    if (error) error->clear();
    return true;
}

}  // namespace map_anything
