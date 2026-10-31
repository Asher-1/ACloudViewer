// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "reference_inputs.hpp"
#include "src/tasks/instantmesh/scheduler.hpp"

namespace {
std::vector<float> readReference(const std::filesystem::path& path,
                                 size_t count) {
    std::vector<float> data(count);
    FILE* file = std::fopen(path.string().c_str(), "rb");
    if (!file) return {};
    const size_t got = std::fread(data.data(), sizeof(float), count, file);
    const bool exact = got == count && std::fgetc(file) == EOF;
    std::fclose(file);
    if (!exact || !std::all_of(data.begin(), data.end(), [](float value) {
            return std::isfinite(value);
        }))
        return {};
    return data;
}

double maxError(const std::vector<float>& actual,
                const std::vector<float>& expected) {
    if (actual.size() != expected.size()) return INFINITY;
    double error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) return INFINITY;
        error = std::max(error, std::abs(double(actual[i]) - expected[i]));
    }
    return error;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || !argv[1][0]) {
        std::fprintf(stderr,
                     "INCOMPLETE: missing pinned diffusers scheduler reference "
                     "path\n");
        return 77;
    }
    const std::filesystem::path root(argv[1]);
    constexpr int steps = 75, n = 4 * 80 * 120;
    const std::vector<std::pair<const char*, size_t>> files = {
            {"timesteps.bin", steps},
            {"sigmas.bin", steps + 1},
            {"alphas_cumprod.bin", 1000},
            {"x0.bin", n},
            {"noise_init.bin", n},
            {"x_init.bin", n},
            {"x_scaled0.bin", n},
            {"eps.bin", size_t(steps) * n},
            {"anc_noise.bin", size_t(steps) * n},
            {"x_final.bin", n}};
    auto identity = instantmesh_reference::identity(
            "cpp_ggml/convert/parity_scheduler.py",
            "59eb41b92f8477c8380772d872870ee6a326bfae0a8fbc7bc75cd4eb9e199d3e");
    identity.insert("steps", steps);
    identity.insert("latent_elements", n);
    const int status =
            instantmesh_reference::authenticate(root, identity, files);
    if (status) return status;
    std::vector<std::vector<float>> refs;
    for (const auto& file : files) {
        const auto path = root / file.first;
        if (!std::filesystem::is_regular_file(path)) {
            std::fprintf(stderr, "INCOMPLETE: missing reference %s\n",
                         path.string().c_str());
            return 77;
        }
        auto data = readReference(path, file.second);
        if (data.empty()) {
            std::fprintf(stderr,
                         "FAIL: invalid reference size or finiteness: %s\n",
                         path.string().c_str());
            return 1;
        }
        refs.push_back(std::move(data));
    }
    EulerAncestralScheduler scheduler;
    scheduler.set_alphas_cumprod(refs[2].data(), 1000);
    scheduler.set_timesteps(steps);
    const double timestepError = maxError(scheduler.timesteps(), refs[0]);
    const double sigmaError = maxError(scheduler.sigmas(), refs[1]);
    std::vector<float> output(n);
    scheduler.add_noise(refs[3].data(), refs[4].data(), n, refs[0][0],
                        output.data());
    const double noiseError = maxError(output, refs[5]);
    scheduler.scale_model_input(refs[5].data(), n, output.data());
    const double scaleError = maxError(output, refs[6]);
    std::vector<float> current = refs[5];
    for (int step = 0; step < steps; ++step) {
        scheduler.step(refs[7].data() + size_t(step) * n, current.data(),
                       refs[8].data() + size_t(step) * n, n, output.data());
        current.swap(output);
    }
    const double walkError = maxError(current, refs[9]);
    // Original executable gates; historical measurements or candidate results
    // must not relax them.
    const bool passed = timestepError < 1e-6 && sigmaError < 1e-6 &&
                        noiseError < 1e-5 && scaleError < 1e-5 &&
                        walkError < 1e-4;
    if (!std::isfinite(timestepError + sigmaError + noiseError + scaleError +
                       walkError)) {
        std::fprintf(stderr,
                     "FAIL: scheduler output contains nonfinite errors\n");
        return 1;
    }
    std::printf(
            "{\"timesteps_max_abs\":%.9g,\"sigmas_max_abs\":%.9g,"
            "\"add_noise_max_abs\":%.9g,\"scale_input_max_abs\":%.9g,"
            "\"walk75_max_abs\":%.9g,\"steps\":75}\n",
            timestepError, sigmaError, noiseError, scaleError, walkError);
    return passed ? 0 : 1;
}
