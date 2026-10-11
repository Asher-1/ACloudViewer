// SPDX-License-Identifier: MIT
// Public-ABI accuracy, repeatability, and timing probe for RAM-family models.
#include <QCoreApplication>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "aicore/runtime_capi.h"
#include "aicore/tagging_capi.h"

namespace {
std::string reference_file(const std::string& directory,
                           const std::string& model) {
    const auto name =
            model.substr(model.find_last_of("/\\") == std::string::npos
                                 ? 0
                                 : model.find_last_of("/\\") + 1);
    if (name.find("ram_plus_") != 0 && name.find("ram_") != 0 &&
        name.find("tag2text_") != 0)
        throw std::runtime_error("model is not in the RAM catalog: " + name);
    return directory + "/" + name.substr(0, name.size() - 5) +
           "-demo1-pytorch-gguf-logits.f32";
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<size_t>(std::ceil(p * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}
void verify_contracts(aicore_tagging_ctx* ctx,
                      const aicore_tagging_options& options,
                      const std::string& path) {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    const QImage rgb = QImage(QString::fromStdString(path))
                               .scaled(385, 383, Qt::IgnoreAspectRatio)
                               .convertToFormat(QImage::Format_RGB888);
    require(!rgb.isNull() && rgb.bytesPerLine() > rgb.width() * 3,
            "padded image fixture");
    auto request = aicore_tagging_default_request();
    aicore_image_view view{rgb.constBits(), rgb.width(), rgb.height(),
                           size_t(rgb.bytesPerLine()), AICORE_IMAGE_RGB8};
    auto infer = [&](aicore_tagging_ctx* context,
                     const aicore_image_view& input) {
        aicore_tagging_result* result = nullptr;
        require(aicore_tagging_infer_image(context, &input, &request,
                                           &result) == 0 &&
                        result,
                "borrowed image inference failed");
        return std::unique_ptr<aicore_tagging_result,
                               decltype(&aicore_tagging_result_free)>(
                result, aicore_tagging_result_free);
    };
    auto first = infer(ctx, view);
    const std::string firstName = first->tags[0].name;
    request.threshold_epsilon = .01f;
    auto margin = infer(ctx, view);
    const bool tag2text = std::string(options.model_path).find("tag2text_") !=
                          std::string::npos;
    for (size_t i = 0; i < margin->tag_count; ++i) {
        require(margin->tags[i].logit == first->tags[i].logit,
                "threshold margin changed graph outputs");
        require(margin->tags[i].threshold ==
                        first->tags[i].threshold - (tag2text ? .01f : 0.f),
                "threshold margin differs from upstream family semantics");
    }
    request.threshold_epsilon = 0;
    for (auto format : {AICORE_IMAGE_RGB8, AICORE_IMAGE_BGR8,
                        AICORE_IMAGE_RGBA8, AICORE_IMAGE_BGRA8}) {
        const int channels =
                format == AICORE_IMAGE_RGB8 || format == AICORE_IMAGE_BGR8 ? 3
                                                                           : 4;
        const size_t stride = size_t(rgb.width() * channels + 17);
        std::vector<uint8_t> bytes(stride * rgb.height(), 0xCD);
        for (int y = 0; y < rgb.height(); ++y)
            for (int x = 0; x < rgb.width(); ++x) {
                const auto* source = rgb.constScanLine(y) + x * 3;
                auto* target = bytes.data() + y * stride + x * channels;
                const bool bgr = format == AICORE_IMAGE_BGR8 ||
                                 format == AICORE_IMAGE_BGRA8;
                target[0] = source[bgr ? 2 : 0];
                target[1] = source[1];
                target[2] = source[bgr ? 0 : 2];
                if (channels == 4) target[3] = 255;
            }
        auto result = infer(
                ctx, {bytes.data(), rgb.width(), rgb.height(), stride, format});
        require(result->tag_count == first->tag_count,
                "format changes class count");
        for (size_t i = 0; i < result->tag_count; ++i)
            require(result->tags[i].logit == first->tags[i].logit,
                    "stride/channel parity failed");
    }
    QImage gray = rgb.convertToFormat(QImage::Format_Grayscale8);
    QImage grayRgb = gray.convertToFormat(QImage::Format_RGB888);
    auto mono = infer(ctx, {gray.constBits(), gray.width(), gray.height(),
                            size_t(gray.bytesPerLine()), AICORE_IMAGE_GRAY8});
    auto expanded =
            infer(ctx, {grayRgb.constBits(), grayRgb.width(), grayRgb.height(),
                        size_t(grayRgb.bytesPerLine()), AICORE_IMAGE_RGB8});
    for (size_t i = 0; i < mono->tag_count; ++i)
        require(mono->tags[i].logit == expanded->tags[i].logit,
                "gray expansion parity failed");
    auto invalid = view;
    invalid.row_stride_bytes = 1;
    aicore_tagging_result* output = reinterpret_cast<aicore_tagging_result*>(1);
    require(aicore_tagging_infer_image(ctx, &invalid, &request, &output) ==
                            -1 &&
                    !output,
            "invalid stride must clear output");
    aicore_pipeline_timings timing{};
    require(aicore_tagging_last_pipeline_timings(ctx, &timing) == -1,
            "failed call must clear timings");
    request.threshold_epsilon = -1;
    require(aicore_tagging_infer_image(ctx, &view, &request, &output) == -1 &&
                    !output,
            "invalid threshold margin must clear output");
    request.threshold_epsilon = 0;
    auto* token = aicore_cancel_token_new();
    require(token, "cancel token allocation failed");
    aicore_cancel_token_request(token);
    aicore_cancel_scope_begin(token);
    const int cancelled =
            aicore_tagging_infer_image(ctx, &view, &request, &output);
    aicore_cancel_scope_end(token);
    aicore_cancel_token_free(token);
    require(cancelled == 1 && !output,
            "cooperative cancellation must clear output");
    aicore_tagging_shutdown();
    aicore_tagging_shutdown();
    auto afterShutdown = infer(ctx, view);
    require(afterShutdown->tags[0].logit == first->tags[0].logit,
            "shutdown destroyed live context");
    {
        std::unique_ptr<aicore_tagging_ctx, decltype(&aicore_tagging_free)>
                peer(aicore_tagging_create(&options), aicore_tagging_free);
        require(aicore_tagging_ready(peer.get()),
                "second context creation failed");
        auto peerResult = infer(peer.get(), view);
        peer.reset();
        require(peerResult->tag_count == first->tag_count &&
                        firstName == peerResult->tags[0].name,
                "result lifetime depends on context");
    }
    auto afterPeer = infer(ctx, view);
    require(firstName == first->tags[0].name &&
                    afterPeer->tags[0].logit == first->tags[0].logit,
            "peer destruction invalidated live owner");
    std::cerr << "RAM public lifecycle/stride/cancel contracts: PASS\n";
}
}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("options need values");
            args[argv[i]] = argv[i + 1];
        }
        auto value = [&](const char* key, const std::string& fallback = "") {
            const auto it = args.find(key);
            return it == args.end() ? fallback : it->second;
        };

        const auto model = value("--model");
        const auto device = value("--backend", "cpu");
        const auto reference_dir = value("--reference-dir");
        const auto labels = value("--tag-list");
        const auto thresholds = value("--thresholds");
        const auto embedding = value("--label-embedding");
        const auto specified = value("--specified-tags");
        if (model.empty()) return 77;

        auto options = aicore_tagging_default_options();
        options.model_path = model.c_str();
        options.device = device.c_str();
        options.threads = std::stoi(value("--threads", "8"));
        if (options.threads <= 0) options.threads = 8;
        options.generate_caption = value("--task", "tags") == "caption";
        options.max_length = std::stoi(value("--max-length", "30"));
        options.beam_batch = std::stoi(value("--beam-batch", "0"));
        options.cpu_precision = std::stoi(value("--cpu-precision", "0"));
        options.tag_list_path = labels.empty() ? nullptr : labels.c_str();
        options.threshold_path =
                thresholds.empty() ? nullptr : thresholds.c_str();
        options.label_embedding_path =
                embedding.empty() ? nullptr : embedding.c_str();

        std::unique_ptr<aicore_tagging_ctx, decltype(&aicore_tagging_free)> ctx(
                aicore_tagging_create(&options), aicore_tagging_free);
        if (!aicore_tagging_ready(ctx.get()))
            throw std::runtime_error(aicore_tagging_last_error(ctx.get()));

        auto request = aicore_tagging_default_request();
        request.threshold = std::stof(value("--threshold", "-1"));
        request.threshold_epsilon =
                std::stof(value("--threshold-epsilon", "0"));
        request.specified_tags = specified.c_str();

        std::vector<std::string> images;
        const auto manifest = value("--manifest");
        if (manifest.empty()) {
            images.push_back(value("--image"));
        } else {
            std::ifstream input(manifest);
            for (std::string line; std::getline(input, line);)
                if (!line.empty()) images.push_back(line);
        }
        if (images.empty() || images.front().empty())
            throw std::runtime_error("no image inputs");
        if (value("--verify-contracts", "0") == "1")
            verify_contracts(ctx.get(), options, images.front());
        if (!reference_dir.empty() && images.size() != 1)
            throw std::runtime_error(
                    "demo1 reference requires exactly one image");

        const int warmup = std::stoi(value("--warmup", "2"));
        const int repeat = std::stoi(value("--repeat", "10"));
        if (warmup < 0 || repeat < 1)
            throw std::runtime_error("invalid iteration count");

        std::ofstream jsonl(value("--jsonl"));
        std::ofstream matrix(value("--logits-matrix"), std::ios::binary);
        double pipeline_total = 0;
        double inference_total = 0;
        double preprocess_total = 0;
        double decode_total = 0;
        double postprocess_total = 0;
        double e2e_total = 0;
        size_t value_count = 0;
        double maximum_difference = 0;
        double absolute_error_sum = 0;
        double probability_maximum_difference = 0;
        double probability_absolute_error_sum = 0;
        double upstream_cpu_maximum_difference = 0;
        double cpu_arithmetic_probability_maximum_difference = 0;
        double cpu_arithmetic_probability_error_sum = 0;
        const bool cpu_quantized_reference =
                QString::fromUtf8(aicore_tagging_context_device(ctx.get()))
                        .startsWith("cpu", Qt::CaseInsensitive) &&
                QString::fromStdString(model).endsWith("-q8_0.gguf");
        const bool automatic_cpu_reference =
                cpu_quantized_reference &&
                options.cpu_precision == AICORE_TAGGING_CPU_AUTO &&
                QFileInfo(QString::fromStdString(model))
                        .fileName()
                        .startsWith("ram_swin_");
        size_t comparison_count = 0;
        std::vector<double> latencies;
        QJsonArray rows;

        for (size_t image_index = 0; image_index < images.size();
             ++image_index) {
            const auto& image_path = images[image_index];
            QJsonObject row;
            QJsonArray selected_tags;
            QJsonArray all_tags;
            double pipeline = 0;
            double inference = 0;
            double preprocessing = 0;
            double decoding = 0;
            double postprocessing = 0;
            double e2e = 0;
            std::vector<float> first_logits;
            QString first_caption;

            for (int iteration = image_index == 0 ? -warmup : 0;
                 iteration < repeat; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                QImage image(QString::fromStdString(image_path));
                if (image.isNull())
                    throw std::runtime_error("cannot decode " + image_path);
                image = image.convertToFormat(QImage::Format_RGB888);
                const auto decoded = std::chrono::steady_clock::now();

                aicore_image_view view{};
                view.data = image.constBits();
                view.width = image.width();
                view.height = image.height();
                view.row_stride_bytes = image.bytesPerLine();
                view.format = AICORE_IMAGE_RGB8;
                aicore_tagging_result* raw = nullptr;
                const int status = aicore_tagging_infer_image(ctx.get(), &view,
                                                              &request, &raw);
                std::unique_ptr<aicore_tagging_result,
                                decltype(&aicore_tagging_result_free)>
                        result(raw, aicore_tagging_result_free);
                if (status != 0)
                    throw std::runtime_error(
                            aicore_tagging_last_error(ctx.get()));

                if (iteration >= 0) {
                    const double elapsed =
                            std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    wall_start)
                                    .count();
                    latencies.push_back(elapsed);
                    pipeline += elapsed;
                    inference += raw->timings.inference_ms;
                    preprocessing += raw->timings.preprocess_ms;
                    decoding += std::chrono::duration<double, std::milli>(
                                        decoded - wall_start)
                                        .count();
                    postprocessing += raw->timings.postprocess_ms;
                    e2e += raw->timings.e2e_ms;
                    std::vector<float> logits;
                    logits.reserve(raw->tag_count);
                    for (size_t i = 0; i < raw->tag_count; ++i)
                        logits.push_back(raw->tags[i].logit);
                    const QString caption = QString::fromUtf8(raw->caption);
                    if (iteration == 0) {
                        first_logits = logits;
                        first_caption = caption;
                    } else if (logits != first_logits ||
                               caption != first_caption) {
                        throw std::runtime_error("unstable repeated output");
                    }
                    if (iteration == repeat - 1) {
                        for (size_t i = 0; i < raw->tag_count; ++i) {
                            const auto& tag = raw->tags[i];
                            QJsonObject item{
                                    {"name", QString::fromUtf8(tag.name)},
                                    {"index", tag.index},
                                    {"probability", tag.probability},
                                    {"logit", tag.logit},
                                    {"threshold", tag.threshold}};
                            all_tags.append(item);
                            if (tag.selected) selected_tags.append(item);
                        }
                    }
                }
            }

            if (!reference_dir.empty()) {
                const auto filename = reference_file(reference_dir, model);
                std::ifstream reference(filename,
                                        std::ios::binary | std::ios::ate);
                if (!reference)
                    throw std::runtime_error("missing PyTorch reference: " +
                                             filename);
                const auto bytes = static_cast<size_t>(reference.tellg());
                if (bytes != first_logits.size() * sizeof(float))
                    throw std::runtime_error(
                            "PyTorch truth class count differs for " + model);
                std::vector<float> truth(first_logits.size());
                reference.seekg(0);
                reference.read(reinterpret_cast<char*>(truth.data()), bytes);
                if (!reference)
                    throw std::runtime_error("cannot read PyTorch reference: " +
                                             filename);
                for (size_t i = 0; i < truth.size(); ++i) {
                    const double diff =
                            std::abs(double(first_logits[i]) - truth[i]);
                    const double p =
                            1.0 / (1.0 + std::exp(-double(first_logits[i])));
                    const double reference_p =
                            1.0 / (1.0 + std::exp(-double(truth[i])));
                    maximum_difference = std::max(maximum_difference, diff);
                    absolute_error_sum += diff;
                    probability_maximum_difference =
                            std::max(probability_maximum_difference,
                                     std::abs(p - reference_p));
                    probability_absolute_error_sum += std::abs(p - reference_p);
                }
                comparison_count += truth.size();
                if (cpu_quantized_reference) {
                    // GGML CPU quantizes activations for Q8 dot products.
                    // Compare that raw arithmetic with the pinned upstream
                    // CPU graph, while retaining independent PyTorch score
                    // errors above. Floating/GPU graphs keep the PyTorch
                    // full-logit gate.
                    const auto cpu_file =
                            QString::fromStdString(filename)
                                    .replace("-demo1-pytorch-gguf-logits.f32",
                                             automatic_cpu_reference
                                                     ? ".cpu-auto-pytorch.f32"
                                                     : ".cpu.f32")
                                    .toStdString();
                    std::ifstream cpu_truth(cpu_file,
                                            std::ios::binary | std::ios::ate);
                    if (!cpu_truth || size_t(cpu_truth.tellg()) != bytes)
                        throw std::runtime_error(
                                "missing/invalid upstream CPU reference: " +
                                cpu_file);
                    cpu_truth.seekg(0);
                    cpu_truth.read(reinterpret_cast<char*>(truth.data()),
                                   bytes);
                    if (!cpu_truth)
                        throw std::runtime_error("cannot read " + cpu_file);
                    for (size_t i = 0; i < truth.size(); ++i) {
                        upstream_cpu_maximum_difference = std::max(
                                upstream_cpu_maximum_difference,
                                std::abs(double(first_logits[i]) - truth[i]));
                        const double difference = std::abs(
                                1. / (1. + std::exp(-double(first_logits[i]))) -
                                1. / (1. + std::exp(-double(truth[i]))));
                        cpu_arithmetic_probability_maximum_difference = std::max(
                                cpu_arithmetic_probability_maximum_difference,
                                difference);
                        cpu_arithmetic_probability_error_sum += difference;
                    }
                }
            }

            row = {{"image", QString::fromStdString(image_path)},
                   {"latency_ms", pipeline / repeat},
                   {"inference_ms", inference / repeat},
                   {"preprocess_ms", preprocessing / repeat},
                   {"decode_ms", decoding / repeat},
                   {"postprocess_ms", postprocessing / repeat},
                   {"e2e_ms", e2e / repeat},
                   {"tags", selected_tags},
                   {"caption", first_caption}};
            if (images.size() == 1) row.insert("all_tags", all_tags);
            if (matrix)
                matrix.write(reinterpret_cast<const char*>(first_logits.data()),
                             static_cast<std::streamsize>(first_logits.size() *
                                                          sizeof(float)));
            if (jsonl)
                jsonl << QJsonDocument(row)
                                 .toJson(QJsonDocument::Compact)
                                 .constData()
                      << '\n';
            rows.append(row);
            pipeline_total += pipeline / repeat;
            inference_total += inference / repeat;
            preprocess_total += preprocessing / repeat;
            decode_total += decoding / repeat;
            postprocess_total += postprocessing / repeat;
            e2e_total += e2e / repeat;
            value_count += first_logits.size();
        }

        QJsonObject report{
                {"device",
                 QString::fromUtf8(aicore_tagging_context_device(ctx.get()))},
                {"images", int(images.size())},
                {"threads", options.threads},
                {"cpu_precision", options.cpu_precision},
                {"class_values", int(value_count)},
                {"pipeline_ms", pipeline_total / images.size()},
                {"inference_ms", inference_total / images.size()},
                {"preprocess_ms", preprocess_total / images.size()},
                {"decode_ms", decode_total / images.size()},
                {"postprocess_ms", postprocess_total / images.size()},
                {"e2e_ms", e2e_total / images.size()},
                {"p50_ms", percentile(latencies, 0.50)},
                {"p95_ms", percentile(latencies, 0.95)},
                {"stable", true},
                {"rows", rows}};
        if (comparison_count != 0) {
            report.insert("max_diff", maximum_difference);
            report.insert("mean_diff", absolute_error_sum / comparison_count);
            report.insert("probability_max_diff",
                          probability_maximum_difference);
            report.insert("probability_mean_diff",
                          probability_absolute_error_sum / comparison_count);
            report.insert("accuracy_max_diff",
                          cpu_quantized_reference
                                  ? upstream_cpu_maximum_difference
                                  : maximum_difference);
            report.insert("raw_logit_reference",
                          automatic_cpu_reference ? "pytorch_cpu_q8_auto"
                          : cpu_quantized_reference
                                  ? "upstream_cpu_q8"
                                  : "pytorch_gguf_dequantized");
            report.insert(
                    "cpu_arithmetic_probability_max_diff",
                    cpu_quantized_reference
                            ? cpu_arithmetic_probability_maximum_difference
                            : probability_maximum_difference);
            report.insert("cpu_arithmetic_probability_mean_diff",
                          cpu_quantized_reference
                                  ? cpu_arithmetic_probability_error_sum /
                                            comparison_count
                                  : probability_absolute_error_sum /
                                            comparison_count);
            if (cpu_quantized_reference)
                report.insert("upstream_cpu_max_diff",
                              upstream_cpu_maximum_difference);
        }

        const auto output = value("--output");
        if (!output.empty()) {
            std::ofstream file(output);
            file << QJsonDocument(report).toJson().constData();
        }
        std::cout << QJsonDocument(report)
                             .toJson(QJsonDocument::Compact)
                             .constData()
                  << '\n';
        ctx.reset();
        aicore_tagging_shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
