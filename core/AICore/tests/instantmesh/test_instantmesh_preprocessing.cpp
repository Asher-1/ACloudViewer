// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QImage>

#include "../../src/tasks/instantmesh/instantmesh_pipeline.cpp"
#include "reference_inputs.hpp"

static QJsonObject e2e_identity() {
    auto identity = instantmesh_reference::identity(
            "cpp_ggml/convert/dump_e2e.py",
            "10a714c70b0c9e87defe2295e47e0bc1b0fbc59b5b9158e21bb6fb7b38d26cb9");
    identity.remove("seed");
    identity.insert("kind",
                    "independent-75-step-diffusion-reference-not-complete-"
                    "geometry-texture-or-performance");
    identity.insert("checkpoint_revision",
                    "2da07e89919e1a130c9b5add1584c70c7aa065fd");
    identity.insert("override_revision",
                    "b785b4ecfb6636ef34a08c748f96f6a5686244d0");
    identity.insert(
            "input_sha256",
            "db468bad8a04f1474a8d68140c07501b013b3ec6124b911fb7852675d64c05ee");
    identity.insert("parameters", QJsonObject{{"steps", 75},
                                              {"seed", 42},
                                              {"dtype", "fp32"},
                                              {"device", "cpu"},
                                              {"threads", 2}});
    return identity;
}

static int diagnose_reference(const char* directory, const char* input) {
    const std::filesystem::path root(directory);
    const auto identity = e2e_identity();
    const instantmesh_reference::Files files{{"img_vae.bin", 3 * 512 * 512},
                                             {"img_clip.bin", 3 * 224 * 224}};
    if (const int status =
                instantmesh_reference::authenticate(root, identity, files))
        return status;
    QFile imageFile(QString::fromUtf8(input));
    QCryptographicHash inputHash(QCryptographicHash::Sha256);
    if (!imageFile.open(QIODevice::ReadOnly) ||
        !inputHash.addData(&imageFile) ||
        inputHash.result().toHex() !=
                identity.value("input_sha256").toString().toLatin1()) {
        std::fprintf(stderr, "FAIL: input image digest/read mismatch: %s\n",
                     input);
        return 1;
    }
    QImage image(QString::fromUtf8(input));
    if (image.isNull()) {
        std::fprintf(
                stderr,
                "FAIL: Qt could not decode the authenticated input image\n");
        return 1;
    }
    image = image.convertToFormat(QImage::Format_RGBA8888);
    std::vector<uint8_t> rgb;
    // Call the real production white-alpha composite, including native row
    // stride; do not duplicate its arithmetic.
    instantmesh::composite_rgba_white(image.constBits(), image.height(),
                                      image.width(), image.bytesPerLine(), rgb);
    std::vector<float> images[2];
    if (!instantmesh::condition_images(rgb.data(), image.height(),
                                       image.width(), 512, 224, images[0],
                                       images[1]))
        return 1;
    QJsonObject metrics{
            {"kind",
             "preprocessing-diagnostic-not-complete-accuracy-or-performance"},
            {"reference_authenticated", 1},
            {"independent_reference_complete", 0}};
    for (int branch = 0; branch < 2; ++branch) {
        const auto& actual = images[branch];
        QFile file(QString::fromUtf8(
                (root / files[branch].first).u8string().c_str()));
        if (!file.open(QIODevice::ReadOnly)) {
            std::fprintf(
                    stderr,
                    "FAIL: cannot reopen authenticated preprocessing tensor\n");
            return 1;
        }
        std::vector<float> expected(actual.size());
        if (file.read(reinterpret_cast<char*>(expected.data()),
                      qint64(expected.size() * sizeof(float))) !=
            qint64(expected.size() * sizeof(float))) {
            std::fprintf(
                    stderr,
                    "FAIL: cannot read authenticated preprocessing tensor\n");
            return 1;
        }
        double maximum = 0, square = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index]) ||
                !std::isfinite(expected[index])) {
                std::fprintf(stderr,
                             "FAIL: preprocessing tensor contains non-finite "
                             "values\n");
                return 1;
            }
            const double difference = double(actual[index]) - expected[index];
            maximum = std::max(maximum, std::abs(difference));
            square += difference * difference;
        }
        const QString prefix = branch == 0 ? "vae_input" : "clip_input";
        metrics.insert(prefix + "_max_abs", maximum);
        metrics.insert(prefix + "_rmse", std::sqrt(square / actual.size()));
    }
    std::puts(
            QJsonDocument(metrics).toJson(QJsonDocument::Compact).constData());
    return 0;  // Diagnostic execution only; no uncalibrated accuracy threshold
               // or release PASS.
}

static int image_probe(char** argv, bool normalize) {
    int dimensions[4] = {};
    const int count = normalize ? 3 : 4;
    for (int i = 0; i < count; ++i) {
        bool valid = false;
        dimensions[i] = QString::fromUtf8(argv[3 + i]).toInt(&valid);
        if (!valid || dimensions[i] <= 0) return 2;
    }
    const std::string filter(argv[3 + count]);
    if (filter != "bilinear" && filter != "bicubic") return 2;
    const int width = dimensions[0], height = dimensions[1];
    if (size_t(height) > std::numeric_limits<size_t>::max() / 3 / size_t(width))
        return 2;
    const size_t bytes = size_t(height) * width * 3;
    QFile input(QString::fromUtf8(argv[2]));
    if (!input.open(QIODevice::ReadOnly) || uint64_t(input.size()) != bytes)
        return 2;
    const auto pixels = input.readAll();
    if (size_t(pixels.size()) != bytes) return 2;
    std::vector<uint8_t> resized;
    std::vector<float> normalized;
    const auto rgb = reinterpret_cast<const uint8_t*>(pixels.constData());
    const char* data = nullptr;
    size_t length = 0;
    if (normalize) {
        std::vector<float> vae, clip;
        if (!instantmesh::condition_images(rgb, height, width, dimensions[2],
                                           dimensions[2], vae, clip))
            return 2;
        normalized = filter == "bilinear" ? std::move(vae) : std::move(clip);
        data = reinterpret_cast<const char*>(normalized.data());
        length = normalized.size() * sizeof(float);
    } else {
        if (!instantmesh::resize_input_aa(
                    rgb, height, width, dimensions[3], dimensions[2],
                    filter == "bilinear" ? instantmesh::InputFilter::Bilinear
                                         : instantmesh::InputFilter::Bicubic,
                    resized))
            return 2;
        data = reinterpret_cast<const char*>(resized.data());
        length = resized.size();
    }
    QFile output(QString::fromUtf8(argv[4 + count]));
    if (!output.open(QIODevice::WriteOnly | QIODevice::NewOnly) ||
        output.write(data, qint64(length)) != qint64(length))
        return 2;
    return 0;
}

static int composite_probe(char** argv) {
    int dimensions[3];
    for (int i = 0; i < 3; ++i) {
        bool valid = false;
        dimensions[i] = QString::fromUtf8(argv[3 + i]).toInt(&valid);
        if (!valid || dimensions[i] <= 0) return 2;
    }
    const int width = dimensions[0], height = dimensions[1],
              stride = dimensions[2];
    if (int64_t(width) * 4 > stride) return 2;
    QFile input(QString::fromUtf8(argv[2]));
    if (!input.open(QIODevice::ReadOnly) ||
        input.size() != int64_t(height) * stride)
        return 2;
    const auto pixels = input.readAll();
    if (pixels.size() != input.size()) return 2;
    std::vector<uint8_t> result;
    instantmesh::composite_rgba_white(
            reinterpret_cast<const uint8_t*>(pixels.constData()), height, width,
            stride, result);
    QFile output(QString::fromUtf8(argv[6]));
    if (!output.open(QIODevice::WriteOnly | QIODevice::NewOnly) ||
        output.write(reinterpret_cast<const char*>(result.data()),
                     qint64(result.size())) != qint64(result.size()))
        return 2;
    return 0;
}

static int diagnose_noise_reference(const char* directory) {
    const std::filesystem::path root(directory);
    std::vector<std::string> names;
    names.reserve(151);
    instantmesh_reference::Files files;
    names.emplace_back("latents0.bin");
    files.emplace_back(names.back().c_str(), 38400);
    for (int step = 0; step < 75; ++step) {
        char name[64];
        std::snprintf(name, sizeof(name), "cond_noise_%03d.bin", step);
        names.emplace_back(name);
        files.emplace_back(names.back().c_str(), 32768);
        std::snprintf(name, sizeof(name), "step_noise_%03d.bin", step);
        names.emplace_back(name);
        files.emplace_back(names.back().c_str(), 38400);
    }
    if (const int status = instantmesh_reference::authenticate(
                root, e2e_identity(), files))
        return status;
    instantmesh::Rng rng{42};
    double maximum[3] = {}, square[3] = {};
    size_t counts[3] = {};
    for (size_t index = 0; index < files.size(); ++index) {
        const int category = index == 0 ? 0 : (index % 2 ? 1 : 2);
        const auto& entry = files[index];
        QFile file(QString::fromUtf8((root / entry.first).u8string().c_str()));
        if (!file.open(QIODevice::ReadOnly)) return 1;
        std::vector<float> expected(entry.second), actual(entry.second);
        if (file.read(reinterpret_cast<char*>(expected.data()),
                      qint64(entry.second * sizeof(float))) !=
            qint64(entry.second * sizeof(float)))
            return 1;
        rng.randn(actual.data(), actual.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]))
                return 1;
            const double difference = double(actual[i]) - expected[i];
            maximum[category] =
                    std::max(maximum[category], std::abs(difference));
            square[category] += difference * difference;
        }
        counts[category] += actual.size();
    }
    QJsonObject metrics{
            {"kind", "authenticated-noise-input-not-full-chain-accuracy"},
            {"reference_authenticated", 1},
            {"independent_reference_complete", 0}};
    const QString prefixes[3] = {"initial", "condition", "ancestral"};
    for (int category = 0; category < 3; ++category) {
        metrics.insert(prefixes[category] + "_max_abs", maximum[category]);
        metrics.insert(prefixes[category] + "_rmse",
                       std::sqrt(square[category] / counts[category]));
    }
    std::puts(
            QJsonDocument(metrics).toJson(QJsonDocument::Compact).constData());
    return *std::max_element(maximum, maximum + 3) < 1e-6 ? 0 : 1;
}

static int noise_probe(char** argv) {
    bool valid = false;
    const auto seed = QString::fromUtf8(argv[2]).toULongLong(&valid);
    if (!valid || argv[2][0] == '-') return 2;
    instantmesh::Rng rng{uint32_t(seed)};
    QFile output(QString::fromUtf8(argv[4]));
    if (!output.open(QIODevice::WriteOnly | QIODevice::NewOnly)) return 2;
    const auto sizes = QString::fromUtf8(argv[3]).split(QLatin1Char(','));
    for (const auto& size : sizes) {
        const int count = size.toInt(&valid);
        if (!valid || count < 16 || count % 16 != 0 || count > 10000000)
            return 2;
        std::vector<float> values(count);
        rng.randn(values.data(), values.size());
        const auto bytes = qint64(values.size() * sizeof(float));
        if (output.write(reinterpret_cast<const char*>(values.data()), bytes) !=
            bytes)
            return 2;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 5 && std::strcmp(argv[1], "noise") == 0)
        return noise_probe(argv);
    if (argc == 7 && std::strcmp(argv[1], "composite") == 0)
        return composite_probe(argv);
    if (argc == 9 && std::strcmp(argv[1], "resize") == 0)
        return image_probe(argv, false);
    if (argc == 8 && std::strcmp(argv[1], "normalize") == 0)
        return image_probe(argv, true);
    if (argc == 3 && std::strcmp(argv[1], "noise-reference") == 0)
        return diagnose_noise_reference(argv[2]);
    if (argc == 3) return diagnose_reference(argv[1], argv[2]);
    if (argc != 1) return 2;
    int failures = 0;
    instantmesh::Rng noise{42};
    float initial[16];
    noise.randn(initial, 16);
    const float expectedNoise[3] = {1.9269153f, 1.4872841f, 0.9007172f};
    for (int i = 0; i < 3; ++i)
        if (std::abs(initial[i] - expectedNoise[i]) > 1e-6f) ++failures;
    const uint8_t rgba[] = {64, 32, 128, 0,   64, 32,  128, 255, 9, 9, 9, 9,
                            64, 32, 128, 128, 0,  255, 64,  255, 9, 9, 9, 9};
    const std::vector<uint8_t> expectedComposite{255, 255, 255, 64, 32,  128,
                                                 159, 143, 191, 0,  255, 64};
    std::vector<uint8_t> composite;
    instantmesh::composite_rgba_white(rgba, 2, 2, 12, composite);
    if (composite != expectedComposite) {
        std::fprintf(stderr, "FAIL: padded RGBA white-alpha composition\n");
        ++failures;
    }
    const uint8_t transparentDark[] = {0, 0, 0, 128};
    instantmesh::composite_rgba_white(transparentDark, 1, 1, 4, composite);
    if (composite != std::vector<uint8_t>{127, 127, 127}) ++failures;
    // Symmetric weights around the center preserve these hand-derived channel
    // values.
    const uint8_t line[] = {0, 30, 60, 70, 100, 130, 140, 170, 200};
    for (auto filter : {instantmesh::InputFilter::Bilinear,
                        instantmesh::InputFilter::Bicubic}) {
        std::vector<uint8_t> resized;
        if (!instantmesh::resize_input_aa(line, 1, 3, 1, 1, filter, resized) ||
            resized != std::vector<uint8_t>{70, 100, 130})
            ++failures;
        if (instantmesh::resize_input_aa(line, 0, 3, 1, 1, filter, resized))
            ++failures;
    }
    std::vector<float> vae, clip;
    if (instantmesh::condition_images(nullptr, 1, 1, 1, 1, vae, clip) ||
        instantmesh::condition_images(line, 0, 3, 1, 1, vae, clip))
        ++failures;
    if (!instantmesh::condition_images(line, 1, 3, 1, 1, vae, clip) ||
        vae.size() != 3 || clip.size() != 3) {
        ++failures;
    } else if (std::abs(vae[0] - (70.f - 127.5f) / 204.f) > 1e-6f ||
               std::abs(clip[0] - (70.f - 0.48145466f * 255.f) /
                                          (0.26862954f * 255.f)) > 1e-6f) {
        ++failures;
    }
    const float means[3][3] = {{0.5f, 0.5f, 0.5f},
                               {0.48145466f, 0.4578275f, 0.40821073f},
                               {0.485f, 0.456f, 0.406f}};
    const float deviations[3][3] = {{0.8f, 0.8f, 0.8f},
                                    {0.26862954f, 0.26130258f, 0.27577711f},
                                    {0.229f, 0.224f, 0.225f}};
    const uint8_t varied[] = {255, 0,   64, 128, 32,  192,
                              0,   255, 16, 96,  160, 224};
    for (int branch = 0; branch < 3; ++branch) {
        std::vector<float> output;
        instantmesh::preprocess_image(varied, 2, 2, 2, means[branch],
                                      deviations[branch], output);
        for (int channel = 0; channel < 3; ++channel)
            for (int pixel = 0; pixel < 4; ++pixel) {
                const float expected = (varied[pixel * 3 + channel] / 255.f -
                                        means[branch][channel]) /
                                       deviations[branch][channel];
                if (std::abs(output[channel * 4 + pixel] - expected) > 1e-6f) {
                    std::fprintf(stderr,
                                 "NCHW mismatch: branch=%d channel=%d pixel=%d "
                                 "actual=%.9g expected=%.9g\n",
                                 branch, channel, pixel,
                                 output[channel * 4 + pixel], expected);
                    ++failures;
                }
            }
        // Identity resize isolates layout/normalization from interpolation
        // error; include rectangular center-crop and native encoder sizes.
        for (const auto shape :
             {std::pair<int, int>{2, 2}, {3, 7}, {224, 224}, {512, 512}}) {
            const int height = shape.first, width = shape.second, size = height;
            std::vector<uint8_t> constant((size_t)width * height * 3);
            for (size_t i = 0; i < constant.size(); ++i)
                constant[i] = varied[i % 3];
            instantmesh::preprocess_image(constant.data(), height, width, size,
                                          means[branch], deviations[branch],
                                          output);
            for (int channel = 0; channel < 3; ++channel) {
                const float expected =
                        (varied[channel] / 255.f - means[branch][channel]) /
                        deviations[branch][channel];
                for (int pixel = 0; pixel < size * size; ++pixel)
                    if (std::abs(output[channel * size * size + pixel] -
                                 expected) > 1e-6f)
                        ++failures;
            }
        }
    }
    std::printf("preprocess_nchw_failures=%d\n", failures);
    return failures ? 1 : 0;
}
