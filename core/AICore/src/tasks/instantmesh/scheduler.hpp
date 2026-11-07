// EulerAncestralDiscreteScheduler — faithful port of diffusers'
// EulerAncestralDiscreteScheduler as configured by the official InstantMesh
// run.py:
//   EulerAncestralDiscreteScheduler.from_config(scheduler.config,
//                                               timestep_spacing='trailing')
// (beta_schedule="linear", beta_start=0.00085, beta_end=0.012,
//  num_train_timesteps=1000, prediction_type="v_prediction").
//
// All math is float32, matching numpy/torch upcast semantics of the
// reference. Noise is supplied by the caller (deterministic parity) or drawn
// from an internal mt19937_64 generator for standalone runs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class EulerAncestralScheduler {
public:
    struct Config {
        int    num_train_timesteps = 1000;
        float  beta_start          = 0.00085f;
        float  beta_end            = 0.012f;
        std::string beta_schedule  = "linear";       // zero123plus-v1.2 scheduler_config.json
        std::string prediction_type = "v_prediction"; // ditto (run.py only overrides spacing)
        std::string timestep_spacing = "trailing";   // run.py override
        int    steps_offset        = 1;             // unused by "trailing"
    };

    EulerAncestralScheduler();
    explicit EulerAncestralScheduler(const Config & cfg);

    // Inject the reference alphas_cumprod curve (float32, num_train_timesteps
    // entries). torch.cumprod's CPU rounding is not reproducible bit-exactly
    // in portable C++, so the curve is treated as model data: produced at
    // conversion time (written into the GGUF alongside the weights) and
    // loaded here. Without injection the constructor computes a close
    // approximation (float32 cumulative product).
    void set_alphas_cumprod(const float * ac, int n);

    // num_inference_steps: e.g. 75 (run.py default).
    void set_timesteps(int num_inference_steps);

    // x / sqrt(sigma^2 + 1) at current step index.
    // `x` and `out` are contiguous float32 arrays of n elements.
    void scale_model_input(const float * x, int n, float * out) const;

    // x + noise * sigma(t)  (t resolved to the matching index in timesteps()).
    void add_noise(const float * x0, const float * noise, int n, float timestep, float * out) const;

    // One ancestral Euler update. `noise` (n floats) implements the ancestral
    // randn; pass a fixed array for parity runs. Advances the step index.
    void step(const float * model_output, const float * sample, const float * noise,
              int n, float * prev_out);

    int    num_inference_steps() const { return (int) timesteps_.size(); }
    int    step_index()          const { return step_index_; }
    float  sigma(int i)          const { return sigmas_[i]; }
    const std::vector<float> & timesteps() const { return timesteps_; }
    const std::vector<float> & sigmas()    const { return sigmas_; }

private:
    // sigma(t): linear interpolation of the training sigma curve at t,
    // matching np.interp(timesteps, arange(len(sigmas)), sigmas).
    float sigma_at_t(float t) const;

    void compute_alphas_cumprod();

    Config cfg_;
    std::vector<float> alphas_cumprod_;  // computed or injected
    bool alphas_injected_ = false;
    std::vector<float> sigmas_;          // [num_inference_steps + 1] (last = 0)
    std::vector<float> timesteps_;       // [num_inference_steps]
    int step_index_ = -1;

    std::vector<uint64_t> mt_state_;     // reserved: deterministic RNG option
};
