// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// EulerAncestralDiscreteScheduler implementation. See scheduler.hpp.
// Numerics mirror diffusers 0.39 exactly (float32, same order of operations):
//   betas  = linspace(beta_start, beta_end, T)              (linear)
//   ac     = cumprod(1 - betas)
//   sigmas = ((1-ac)/ac)^0.5
//   set_timesteps(trailing): step_ratio = T/steps;
//     timesteps = round(arange(T, 0, -step_ratio)) - 1  (float32)
//     sigmas = interp(timesteps, arange(T), sigmas) ++ [0]
//   step (v_prediction):
//                   pred_x0   = mo*(-sigma/sqrt(sigma^2+1)) + x/(sigma^2+1)
//                   sigma_up  = sqrt(s2^2*(s1^2-s2^2)/s1^2)
//                   sigma_down= sqrt(s2^2 - sigma_up^2)
//                   d         = (x - pred_x0)/sigma
//                   prev      = x + d*(sigma_down - sigma) + noise*sigma_up
#include "scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
namespace {

// float32 linspace, matching torch.linspace for these endpoints.
std::vector<float> linspace(float start, float stop, int n) {
    std::vector<float> out(n);
    const float step = (n == 1) ? 0.f : (stop - start) / (n - 1);
    for (int i = 0; i < n; ++i) out[i] = start + step * i;
    return out;
}

}  // namespace

EulerAncestralScheduler::EulerAncestralScheduler()
    : EulerAncestralScheduler(Config()) {}

EulerAncestralScheduler::EulerAncestralScheduler(const Config& cfg)
    : cfg_(cfg) {
    compute_alphas_cumprod();
}

void EulerAncestralScheduler::set_alphas_cumprod(const float* ac, int n) {
    if (n != cfg_.num_train_timesteps) return;  // mismatched curve: ignore
    alphas_cumprod_.assign(ac, ac + n);
    alphas_injected_ = true;
}

void EulerAncestralScheduler::compute_alphas_cumprod() {
    if (alphas_injected_) return;
    const int T = cfg_.num_train_timesteps;
    std::vector<float> betas(T);
    if (cfg_.beta_schedule == "scaled_linear") {
        std::vector<float> sq = linspace(std::sqrt(cfg_.beta_start),
                                         std::sqrt(cfg_.beta_end), T);
        for (int i = 0; i < T; ++i) betas[i] = sq[i] * sq[i];
    } else {  // "linear"
        betas = linspace(cfg_.beta_start, cfg_.beta_end, T);
    }
    alphas_cumprod_.resize(T);
    float acc = 1.f;
    for (int i = 0; i < T; ++i) {
        acc *= 1.f - betas[i];
        alphas_cumprod_[i] = acc;
    }
}

void EulerAncestralScheduler::set_timesteps(int num_inference_steps) {
    const int T = cfg_.num_train_timesteps;
    // numpy's np.interp evaluates in float64; mirror that. But the y values
    // themselves come from torch float32 `((1-ac)/ac)**0.5` — sqrt in float32
    // first, then lift to double, exactly like the reference.
    std::vector<double> train_sigma(T);
    for (int i = 0; i < T; ++i) {
        train_sigma[i] = (double)std::sqrt((1.f - alphas_cumprod_[i]) /
                                           alphas_cumprod_[i]);
    }

    timesteps_.assign(num_inference_steps, 0.f);
    if (cfg_.timestep_spacing == "trailing") {
        const float step_ratio = (float)T / (float)num_inference_steps;
        for (int i = 0; i < num_inference_steps; ++i) {
            // np.arange(T, 0, -step_ratio)[i] = T - i*step_ratio
            const float t = (float)T - (float)i * step_ratio;
            timesteps_[i] =
                    std::nearbyint(t) - 1.f;  // np.round + (-1), float32
        }
    } else if (cfg_.timestep_spacing == "leading") {
        const int step_ratio = T / num_inference_steps;
        for (int i = 0; i < num_inference_steps; ++i) {
            timesteps_[i] = std::round((float)i * step_ratio) +
                            (float)cfg_.steps_offset;
        }
        // reverse
        std::reverse(timesteps_.begin(), timesteps_.end());
    } else {  // linspace
        std::vector<float> ls =
                linspace(0.f, (float)(T - 1), num_inference_steps);
        for (int i = 0; i < num_inference_steps; ++i)
            timesteps_[i] = ls[num_inference_steps - 1 - i];
    }

    // np.interp(timesteps, arange(T), train_sigma): numpy clips outside the
    // range to the edge values — trailing timesteps are all within [0, T-1].
    sigmas_.assign(num_inference_steps + 1, 0.f);
    for (int i = 0; i < num_inference_steps; ++i) {
        const double t = (double)timesteps_[i];
        const int i0 = (int)t;  // np.interp uses floor; t >= 0 here
        const double frac = t - (double)i0;
        const int i1 = (i0 + 1 < T) ? i0 + 1 : T - 1;
        sigmas_[i] = (float)(train_sigma[i0] * (1.0 - frac) +
                             train_sigma[i1] * frac);
    }
    step_index_ = -1;
}

float EulerAncestralScheduler::sigma_at_t(float t) const {
    // Resolve a timestep value to a sigma via exact match against the
    // schedule (diffusers index_for_timestep semantics).
    const int n = (int)timesteps_.size();
    for (int i = 0; i < n; ++i) {
        if (timesteps_[i] == t) return sigmas_[i];
    }
    // Fall back to the training-curve interpolation for unseen t.
    const int T = cfg_.num_train_timesteps;
    const int i0 = (int)t;
    const float frac = t - (float)i0;
    auto ac = [&](int i) {
        return std::sqrt((1.f - alphas_cumprod_[i]) / alphas_cumprod_[i]);
    };
    const int i1 = (i0 + 1 < T) ? i0 + 1 : T - 1;
    return ac(i0) * (1.f - frac) + ac(i1) * frac;
}

void EulerAncestralScheduler::scale_model_input(const float* x,
                                                int n,
                                                float* out) const {
    const int i = (step_index_ < 0) ? 0 : step_index_;
    const float s = sigmas_[i];
    const float inv = 1.f / std::sqrt(s * s + 1.f);
    for (int k = 0; k < n; ++k) out[k] = x[k] * inv;
}

void EulerAncestralScheduler::add_noise(const float* x0,
                                        const float* noise,
                                        int n,
                                        float timestep,
                                        float* out) const {
    const float s = sigma_at_t(timestep);
    for (int k = 0; k < n; ++k) out[k] = x0[k] + noise[k] * s;
}

void EulerAncestralScheduler::step(const float* model_output,
                                   const float* sample,
                                   const float* noise,
                                   int n,
                                   float* prev_out) {
    int i = step_index_;
    if (i < 0) i = 0;
    const float sigma = sigmas_[i];
    const float sigma_from = sigma;
    const float sigma_to = sigmas_[i + 1];

    const float sigma_up =
            std::sqrt(sigma_to * sigma_to *
                      (sigma_from * sigma_from - sigma_to * sigma_to) /
                      (sigma_from * sigma_from));
    const float sigma_down =
            std::sqrt(sigma_to * sigma_to - sigma_up * sigma_up);
    const float dt = sigma_down - sigma;

    for (int k = 0; k < n; ++k) {
        const float x = sample[k];
        float pred_x0;
        if (cfg_.prediction_type == "v_prediction") {
            // pred_x0 = mo * (-sigma/sqrt(sigma^2+1)) + x/(sigma^2+1)
            const float denom = sigma * sigma + 1.f;
            pred_x0 = model_output[k] * (-sigma / std::sqrt(denom)) + x / denom;
        } else {  // epsilon
            pred_x0 = x - sigma * model_output[k];
        }
        const float d = (x - pred_x0) / sigma;
        float prev = x + d * dt;
        if (noise) prev += noise[k] * sigma_up;
        prev_out[k] = prev;
    }
    step_index_ = i + 1;
}
