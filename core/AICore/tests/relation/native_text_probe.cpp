// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Links only the pinned upstream static library, never AICore.
#include <json/json.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "relateanything/full_model.hpp"
int main(int argc, char** argv) {
    try {
        if (argc != 10 && argc != 11)
            throw std::runtime_error(
                    "model tokenizer texts backend threads warmup repeats "
                    "output strict [token-output]");
        relateanything::ModelOptions o;
        o.text_student = argv[1];
        o.tokenizer = argv[2];
        o.backend = argv[4];
        o.threads = std::stoi(argv[5]);
        const int arithmetic = std::stoi(argv[9]);
        o.strict_f32 = arithmetic > 0;
        o.strict_f32_auto = arithmetic < 0;
        if (argc == 11) o.token_output = argv[10];
        Json::Value json;
        std::ifstream input(argv[3]);
        input >> json;
        std::vector<std::string> texts;
        for (const auto& v : json) texts.push_back(v.asString());
        std::vector<double> times;
        std::vector<float> values;
        for (int i = -std::stoi(argv[6]); i < std::stoi(argv[7]); ++i) {
            auto start = std::chrono::steady_clock::now();
            values = relateanything::encode_predicates(o, texts);
            if (i >= 0)
                times.push_back(
                        std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count());
        }
        std::ofstream output(argv[8], std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()),
                     values.size() * sizeof(float));
        if (!output || times.empty())
            throw std::runtime_error("invalid output/repetitions");
        std::cout << "iteration_ms=";
        for (size_t i = 0; i < times.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << times[i];
        }
        std::cout << '\n';
        std::sort(times.begin(), times.end());
        std::cout << "latency_ms=" << times[times.size() / 2] << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
