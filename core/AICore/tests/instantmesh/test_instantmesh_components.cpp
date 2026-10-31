// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "aicore/asset_digests.h"
#include "geometry_reference.hpp"
#include "ggml-alloc.h"
#include "reference_inputs.hpp"
#include "src/common/ggml_backend_registry.hpp"
#include "src/tasks/instantmesh/clip_vision.hpp"
#include "src/tasks/instantmesh/synthesizer.hpp"
#include "src/tasks/instantmesh/unet.hpp"
#include "src/tasks/instantmesh/vae.hpp"

namespace {
std::vector<float> read(const std::filesystem::path& path, size_t count) {
    QFile file(QString::fromUtf8(path.u8string().c_str()));
    if (!file.open(QIODevice::ReadOnly)) return {};
    std::vector<float> result(count);
    if (file.read(reinterpret_cast<char*>(result.data()),
                  qint64(count * sizeof(float))) !=
        qint64(count * sizeof(float)))
        return {};
    if (!std::all_of(result.begin(), result.end(),
                     [](float x) { return std::isfinite(x); }))
        return {};
    return result;
}

bool measure(const float* actual,
             const std::vector<float>& expected,
             const char* prefix,
             double limit,
             QJsonObject& metrics) {
    if (!actual) return false;
    double maxAbs = 0, squared = 0, absolute = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i])) return false;
        const double difference = double(actual[i]) - expected[i];
        maxAbs = std::max(maxAbs, std::abs(difference));
        squared += difference * difference;
        absolute += std::abs(difference);
    }
    metrics.insert(QString::fromUtf8(prefix) + "_max_abs", maxAbs);
    metrics.insert(QString::fromUtf8(prefix) + "_rmse",
                   std::sqrt(squared / expected.size()));
    metrics.insert(QString::fromUtf8(prefix) + "_mean_abs",
                   absolute / expected.size());
    return maxAbs < limit;
}
// Removing load validation must fail this gate: valid GGUF containers must not
// defer aborts or out-of-bounds access to forward.
int synthesizer_load_contract(const char* device) {
    std::string error;
    auto lease = aicore::runtime::acquire_backend_lease(device, 4, &error);
    if (!lease) return 1;
    auto lock = lease.lock();
    QTemporaryDir directory;
    if (!directory.isValid()) return 1;
    bool passed = true;
    for (int mutation = 0; mutation < 7; ++mutation) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                ggml_init({2u << 20, nullptr, false}), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
                gguf_init_empty(), gguf_free);
        if (!ctx || !gguf) return 1;
        for (const char* branch :
             {"net_sdf", "net_deformation", "net_weight", "net_rgb"}) {
            const int output = std::string(branch) == "net_sdf"      ? 1
                               : std::string(branch) == "net_weight" ? 21
                                                                     : 3;
            for (int layer = 0; layer < 4; ++layer) {
                const int input =
                        layer == 0 ? (std::string(branch) == "net_weight" ? 1920
                                                                          : 240)
                                   : 64;
                const int rows = layer == 3 ? output : 64;
                const std::string name = std::string("decoder.") + branch +
                                         "." + std::to_string(2 * layer);
                for (bool bias : {false, true}) {
                    if (mutation == 1 && name == "decoder.net_rgb.6" && bias)
                        continue;
                    auto* tensor =
                            bias ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32,
                                                      rows)
                                 : ggml_new_tensor_2d(
                                           ctx.get(), GGML_TYPE_F32,
                                           mutation == 6 && name == "decoder."
                                                                    "net_"
                                                                    "weight.0"
                                                   ? input - 1
                                                   : input,
                                           rows);
                    ggml_set_name(
                            tensor,
                            (name + (bias ? ".bias" : ".weight")).c_str());
                    ggml_set_zero(tensor);
                    gguf_add_tensor(gguf.get(), tensor);
                }
            }
        }
        if (mutation == 2)
            gguf_set_val_i32(gguf.get(), "synthesizer.plane_dim", 129);
        if (mutation == 3)
            gguf_set_val_i32(gguf.get(), "synthesizer.hidden", 0);
        if (mutation == 4)
            gguf_set_val_i32(gguf.get(), "synthesizer.num_layers", 0);
        if (mutation == 5)
            gguf_set_val_str(gguf.get(), "synthesizer.plane_dim", "invalid");
        const std::string path =
                directory.filePath(QString::number(mutation) + ".gguf")
                        .toStdString();
        if (!gguf_write_to_file(gguf.get(), path.c_str(), false)) return 1;
        instantmesh::SynthesizerModel model;
        error.clear();
        const bool loaded = instantmesh::synthesizer_load(path, lease.handle(),
                                                          model, &error);
        bool row = mutation == 0
                           ? loaded
                           : !loaded && !error.empty() && !model.backend &&
                                     !model.gguf.ctx && !model.gguf.gguf &&
                                     !model.gguf.buffer &&
                                     model.gguf.tensors.empty();
        if (mutation != 0)
            row = row && !instantmesh::synthesizer_load(path, lease.handle(),
                                                        model, nullptr);
        std::printf("synthesizer mutation=%d loaded=%d contract=%d error=%s\n",
                    mutation, int(loaded), int(row), error.c_str());
        passed = row && passed;
    }
    std::printf("actual_backend=%s\n",
                ggml_backend_dev_name(ggml_backend_get_device(lease.handle())));
    return passed ? 0 : 1;
}
// net_weight requires each corner's three plane features to be adjacent, not
// feature-major. Select the second corner's third plane by hand: expect
// 1000*0.1=100; the wrong permutation yields 20*0.1=2.
int synthesizer_layout_contract(const char* device) {
    std::string error;
    auto lease = aicore::runtime::acquire_backend_lease(device, 4, &error);
    if (!lease) return 1;
    auto lock = lease.lock();
    instantmesh::SynthesizerModel model;
    model.backend = lease.handle();
    model.hp.plane_dim = 1;
    model.hp.hidden = 32;
    model.gguf.ctx = ggml_init({2u << 20, nullptr, true});
    if (!model.gguf.ctx) return 1;
    for (const char* branch :
         {"net_sdf", "net_deformation", "net_weight", "net_rgb"}) {
        const bool weight = std::string(branch) == "net_weight";
        const int output = weight                             ? 21
                           : std::string(branch) == "net_sdf" ? 1
                                                              : 3;
        for (int layer = 0; layer < 4; ++layer) {
            const int input = layer == 0 ? (weight ? 24 : 3) : 32;
            const int rows = layer == 3 ? output : 32;
            const std::string prefix = std::string("decoder.") + branch + "." +
                                       std::to_string(2 * layer);
            model.gguf.tensors.emplace(
                    prefix + ".weight",
                    ggml_new_tensor_2d(model.gguf.ctx, GGML_TYPE_F32, input,
                                       rows));
            model.gguf.tensors.emplace(
                    prefix + ".bias",
                    ggml_new_tensor_1d(model.gguf.ctx, GGML_TYPE_F32, rows));
        }
    }
    model.gguf.buffer =
            ggml_backend_alloc_ctx_tensors(model.gguf.ctx, model.backend);
    if (!model.gguf.buffer) return 1;
    for (const auto& entry : model.gguf.tensors) {
        std::vector<float> data(ggml_nelements(entry.second), 0);
        if (entry.first.find("decoder.net_weight.") == 0 &&
            entry.first.find(".weight") != std::string::npos) {
            if (entry.first == "decoder.net_weight.0.weight")
                data[5] = 1;
            else if (entry.first == "decoder.net_weight.6.weight")
                for (int row = 0; row < 21; ++row) data[row * 32] = 1;
            else
                data[0] = 1;
        }
        ggml_backend_tensor_set(entry.second, data.data(), 0,
                                data.size() * sizeof(float));
    }
    const float batch0[] = {10,  20,  30,   40,   100,  200,
                            300, 400, 1000, 2000, 3000, 4000};
    std::vector<float> planes(batch0, batch0 + 12);
    for (float value : batch0) planes.push_back(value + 1);
    const float vertices[] = {-0.5f, -0.5f, -0.5f, 0.5f, -0.5f, -0.5f,
                              -0.5f, 0.5f,  -0.5f, 0.5f, 0.5f,  -0.5f,
                              -0.5f, -0.5f, 0.5f,  0.5f, -0.5f, 0.5f,
                              -0.5f, 0.5f,  0.5f,  0.5f, 0.5f,  0.5f};
    const int32_t cubes[] = {7, 0, 5, 2, 6, 1, 4, 3};
    float *sdf = nullptr, *deformation = nullptr, *weight = nullptr;
    instantmesh::synthesizer_forward(model, planes.data(), 2, 2, 2, vertices, 8,
                                     cubes, 1, &sdf, &deformation, &weight);
    using Buffer = std::unique_ptr<float, decltype(&std::free)>;
    Buffer s(sdf, std::free), d(deformation, std::free), w(weight, std::free);
    if (!s || !d || !w) return 1;
    double maximum = 0;
    for (int batch = 0; batch < 2; ++batch)
        for (int channel = 0; channel < 21; ++channel)
            maximum = std::max(maximum,
                               std::abs(double(w.get()[batch * 21 + channel]) -
                                        (batch ? 100.1 : 100.0)));
    std::printf(
            "synth corner-major batch0=%.9g batch1=%.9g max_abs=%.9g "
            "actual_backend=%s\n",
            w.get()[0], w.get()[21], maximum,
            ggml_backend_dev_name(ggml_backend_get_device(lease.handle())));
    return maximum < 2e-5 ? 0 : 1;
}
// Extend the same component probe; the layout exception bypasses only weight's
// old-output compatibility failure and retains all raw differences.
int synthesizer_reference(int argc, char** argv) {
    using namespace instantmesh_geometry_reference;
    if (argc < 10) return 77;
    const Path model_path(argv[2]), root(argv[3]), baseline(argv[7]),
            geometry(argv[8]), quant_root(argv[9]);
    QJsonObject manifest, metadata, capture, quant_metadata;
    if (const int status = json(Path(argv[6]), manifest)) return status;
    const auto gate =
            manifest.value("instantmesh_synthesizer_component_accuracy")
                    .toObject();
    if (gate.isEmpty()) return 77;
    if (const int status = json(root / "provenance.json", metadata))
        return status;
    if (const int status = json(baseline / "capture.json", capture))
        return status;
    if (const int status = json(quant_root / "provenance.json", quant_metadata))
        return status;
    const auto matches = [&](const char* key, const Path& file) {
        return gate.value(key).toString().toLatin1() == digest(file);
    };
    const auto exception = gate.value("compatibility_exception").toObject();
    const auto backends = gate.value("backends").toArray();
    const QJsonArray plane_shape{1, 3, 80, 64, 64};
    if (gate.value("kind").toString() !=
                "frozen-synthesizer-fields-not-full-pipeline-or-performance" ||
        gate.value("upstream_revision") !=
                metadata.value("upstream_revision") ||
        gate.value("plane_shape").toArray() != plane_shape ||
        metadata.value("plane_shape").toArray() != plane_shape ||
        gate.value("scale").toDouble() != 2.1 || backends.size() != 2 ||
        !backends.contains("cpu") || !backends.contains("vulkan") ||
        !matches("reference_receipt_sha256", root / "provenance.json") ||
        !matches("geometry_reference_receipt_sha256",
                 geometry / "provenance.json") ||
        !matches("baseline_capture_sha256", baseline / "capture.json") ||
        !matches("weight_quantization_reference_sha256",
                 quant_root / "provenance.json") ||
        !gate.value("locked_before_candidate_model_sampling").toBool() ||
        gate.value("absolute_multiplier").toDouble() != 1.10 ||
        gate.value("added_multiplier").toDouble() != 0.10 ||
        gate.value("epsilon").toDouble() != 1e-5 ||
        gate.value("weight_quantization_multiplier").toDouble() != 1.10 ||
        gate.value("weight_epsilon").toDouble() != 2e-5 ||
        gate.value("weight_migration_limit").toDouble() != 2e-5 ||
        exception.value("id").toString() !=
                "instantmesh-net-weight-corner-major-layout" ||
        exception.value("authorization").toString() !=
                "user-approved-correct-layout-with-explicit-compatibility-"
                "exception" ||
        exception.value("output").toString() != "weight" ||
        !exception.value("other_outputs_and_dino_lrm_gates_unchanged")
                 .toBool() ||
        !exception.value("preserve_original_baseline").toBool() ||
        !exception.value("preserve_raw_added_metrics").toBool() ||
        metadata.value("upstream_revision").toString() !=
                "32b14574edc9caba95f03672bfa1eeddd13c3af7" ||
        metadata.value("checkpoint_sha256").toString() !=
                "329f7aae9b583fd7c1b27d0221463db7b808a03c48a1f7aa26649af6a03b91"
                "a1" ||
        gate.value("grid_res").toInt() != 8 ||
        metadata.value("grid_res").toInt() != 8 ||
        metadata.value("scale").toDouble() != 2.1) {
        std::fprintf(
                stderr,
                "FAIL: Synthesizer reference/frozen-gate identity mismatch\n");
        return 1;
    }
    const std::string filename = model_path.filename().u8string();
    const QString model_name = QString::fromStdString(filename),
                  backend_name = QString::fromUtf8(argv[4]).toLower();
    QString quant;
    for (const char* value : {"f32", "f16", "q8"})
        if (model_name == QString("synthesizer_") + value + ".gguf")
            quant = value;
    if (quant.isEmpty()) return 77;
    struct SynthCase {
        QString name;
        std::vector<float> planes, vertices, weight_reference;
        std::vector<int32_t> cubes;
        std::array<std::vector<float>, 4> expected, original;
        std::array<QJsonObject, 4> bounds;
    };
    constexpr size_t vertices = 729, cubes = 512;
    const char* branches[] = {"sdf", "deformation", "weight", "rgb"};
    const size_t counts[] = {vertices, 3 * vertices, 21 * cubes, 3 * vertices};
    std::vector<SynthCase> cases;
    for (const char* scenario : scenarios) {
        SynthCase item;
        item.name = scenario;
        const auto record = metadata.value("scenarios")
                                    .toObject()
                                    .value(item.name)
                                    .toObject();
        const auto files = record.value("files").toObject();
        const auto row =
                capture.value("runs")
                        .toObject()
                        .value(item.name + "/" + backend_name + "/" + quant)
                        .toObject();
        if (row.isEmpty()) return 77;
        const QString actual = row.value("reported_backend").toString();
        if (actual.isEmpty() || (backend_name == "cpu") != (actual == "CPU"))
            return 1;
        const Path scenario_path = root / scenario;
        if (const int status = numeric(scenario_path / "verts.bin",
                                       files.value("verts.bin").toObject(),
                                       vertices * 3, item.vertices))
            return status;
        const auto plane_hash = record.value("input_triplanes_sha256");
        if (const int status = numeric(
                    geometry / scenario / "triplanes.bin",
                    {{"sha256", plane_hash}, {"bytes", 3 * 80 * 64 * 64 * 4}},
                    3 * 80 * 64 * 64, item.planes))
            return status;
        const Path cube_path = scenario_path / "cubes.bin";
        QFile cube_file(QString::fromStdString(cube_path.u8string()));
        if (!cube_file.exists()) return 77;
        const auto cube_record = files.value("cubes.bin").toObject();
        if (cube_file.size() != qint64(cubes * 8 * 4) ||
            cube_record.value("bytes").toDouble() != double(cubes * 8 * 4) ||
            digest(cube_path) !=
                    cube_record.value("sha256").toString().toLatin1() ||
            !cube_file.open(QIODevice::ReadOnly))
            return 1;
        item.cubes.resize(cubes * 8);
        if (cube_file.read(reinterpret_cast<char*>(item.cubes.data()),
                           cubes * 8 * 4) != qint64(cubes * 8 * 4) ||
            !std::all_of(
                    item.cubes.begin(), item.cubes.end(),
                    [](int32_t value) { return value >= 0 && value < 729; }))
            return 1;
        for (int index = 0; index < 4; ++index) {
            const QString branch = branches[index], file = branch + ".bin";
            if (const int status = numeric(scenario_path / file.toStdString(),
                                           files.value(file).toObject(),
                                           counts[index], item.expected[index]))
                return status;
            if (const int status = numeric(
                        baseline / scenario / backend_name.toStdString() /
                                quant.toStdString() / file.toStdString(),
                        row.value("outputs")
                                .toObject()
                                .value(branch)
                                .toObject(),
                        counts[index], item.original[index]))
                return status;
            item.bounds[index] = row.value("frozen_limits")
                                         .toObject()
                                         .value(branch)
                                         .toObject();
            const auto measured =
                    errors(item.original[index].data(),
                           item.expected[index].data(), counts[index]);
            for (const char* key : {"max_abs", "rmse"}) {
                const double error = measured.value(key).toDouble();
                if (std::abs(item.bounds[index]
                                     .value(QString("absolute_") + key)
                                     .toDouble(-1) -
                             (1.10 * error + 1e-5)) > 1e-10 ||
                    std::abs(item.bounds[index]
                                     .value(QString("added_") + key)
                                     .toDouble(-1) -
                             (0.10 * error + 1e-5)) > 1e-10)
                    return 1;
            }
        }
        const auto weight = quant_metadata.value("records")
                                    .toObject()
                                    .value(item.name + "/" + model_name)
                                    .toObject();
        const QString file = weight.value("file").toString();
        if (file.isEmpty() || file.contains('/') || file.contains('\\') ||
            file.contains(".."))
            return 1;
        if (const int status = numeric(quant_root / file.toStdString(), weight,
                                       counts[2], item.weight_reference))
            return status;
        const auto measured = errors(item.weight_reference.data(),
                                     item.expected[2].data(), counts[2]);
        for (const char* key : {"max_abs", "rmse"}) {
            if (std::abs(weight.value(QString("absolute_") + key).toDouble(-1) -
                         (1.10 * measured.value(key).toDouble() + 2e-5)) >
                        1e-10 ||
                weight.value(QString("migration_") + key).toDouble() != 2e-5)
                return 1;
        }
        item.bounds[2] = weight;
        cases.push_back(std::move(item));
    }
    if (!QFile::exists(QString::fromStdString(model_path.u8string()))) {
        std::fprintf(stderr, "INCOMPLETE: missing Synthesizer GGUF\n");
        return 77;
    }
    const char* catalog = aicore::AssetDigestForFile(
            ("instantmesh_models/" + filename).c_str());
    if (!catalog || digest(model_path) != QByteArray(catalog)) return 1;
    std::string error;
    auto lease = aicore::runtime::acquire_backend_lease(
            argv[4], std::atoi(argv[5]), &error);
    if (!lease) return 1;
    auto lock = lease.lock();
    instantmesh::SynthesizerModel model;
    if (!instantmesh::synthesizer_load(model_path.u8string(), lease.handle(),
                                       model, &error)) {
        std::fprintf(stderr, "FAIL: Synthesizer load: %s\n", error.c_str());
        return 1;
    }
    QJsonObject report{
            {"reference_authenticated", 1},
            {"frozen_limits_authenticated", 1},
            {"known_upstream_layout_exception", 1},
            {"component_reference_complete", 1},
            {"actual_backend",
             ggml_backend_dev_name(ggml_backend_get_device(lease.handle()))}};
    bool passed = true;
    double max_absolute_ratio = 0, max_migration_ratio = 0;
    for (const auto& item : cases) {
        float *sdf = nullptr, *deform = nullptr, *weight = nullptr,
              *rgb = nullptr;
        instantmesh::synthesizer_forward(
                model, item.planes.data(), 1, 64, 64, item.vertices.data(),
                vertices, item.cubes.data(), cubes, &sdf, &deform, &weight);
        using Buffer = std::unique_ptr<float, decltype(&std::free)>;
        Buffer s(sdf, std::free), d(deform, std::free), w(weight, std::free);
        if (!instantmesh::synthesizer_texture_forward(
                    model, item.planes.data(), 1, 64, 64, item.vertices.data(),
                    vertices, &rgb))
            return 1;
        Buffer r(rgb, std::free);
        const float* outputs[] = {s.get(), d.get(), w.get(), r.get()};
        for (int index = 0; index < 4; ++index) {
            if (!outputs[index]) return 1;
            const auto absolute = errors(
                    outputs[index], item.expected[index].data(), counts[index]);
            const auto added = errors(
                    outputs[index], item.original[index].data(), counts[index]);
            const auto migration =
                    index == 2 ? errors(outputs[index],
                                        item.weight_reference.data(),
                                        counts[index])
                               : added;
            if (absolute.isEmpty() || added.isEmpty() || migration.isEmpty())
                return 1;
            const QString prefix = item.name + "_" + branches[index];
            for (auto it = absolute.begin(); it != absolute.end(); ++it) {
                report.insert(prefix + "_" + it.key(), it.value());
                report.insert(prefix + "_legacy_added_" + it.key(),
                              added.value(it.key()));
                report.insert(prefix + "_migration_" + it.key(),
                              migration.value(it.key()));
                if (it.key() == "max_abs" || it.key() == "rmse") {
                    const double limit = item.bounds[index]
                                                 .value("absolute_" + it.key())
                                                 .toDouble();
                    const double budget =
                            item.bounds[index]
                                    .value((index == 2 ? "migration_"
                                                       : "added_") +
                                           it.key())
                                    .toDouble();
                    const double ratio = it.value().toDouble() / limit,
                                 added_ratio =
                                         migration.value(it.key()).toDouble() /
                                         budget;
                    report.insert(prefix + "_" + it.key() + "_ratio", ratio);
                    report.insert(prefix + "_migration_" + it.key() + "_ratio",
                                  added_ratio);
                    max_absolute_ratio = std::max(max_absolute_ratio, ratio);
                    max_migration_ratio =
                            std::max(max_migration_ratio, added_ratio);
                    passed = ratio <= 1 && added_ratio <= 1 && passed;
                }
            }
        }
    }
    report.insert("max_absolute_error_ratio", max_absolute_ratio);
    report.insert("max_migration_error_ratio", max_migration_ratio);
    std::puts(QJsonDocument(report).toJson(QJsonDocument::Compact).constData());
    return passed ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "synth")
        return synthesizer_reference(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "synth-layout-contract")
        return synthesizer_layout_contract(argc > 2 ? argv[2] : "cpu");
    if (argc > 1 && std::string(argv[1]) == "synth-load-contract")
        return synthesizer_load_contract(argc > 2 ? argv[2] : "cpu");
    if (argc < 5) {
        std::fprintf(stderr,
                     "INCOMPLETE: requires clip|vae|unet|dino|lrm, GGUF, "
                     "reference, device "
                     "[threads]\n");
        return 77;
    }
    if (std::string(argv[1]) == "dino" || std::string(argv[1]) == "lrm")
        return instantmesh_geometry_reference::run(argc, argv);
    const bool clip = std::string(argv[1]) == "clip";
    const bool unet = std::string(argv[1]) == "unet";
    if (!clip && !unet && std::string(argv[1]) != "vae") return 2;
    const std::filesystem::path root(argv[3]), modelPath(argv[2]);
    const instantmesh_reference::Files files =
            unet   ? instantmesh_reference::Files{{"unet_in.bin",
                                                   2 * 4 * 120 * 80},
                                                  {"unet_ref_in.bin",
                                                   2 * 4 * 120 * 80},
                                                  {"unet_ctx.bin", 2 * 77 * 1024},
                                                  {"unet_t.bin", 1},
                                                  {"unet_eps_out.bin",
                                                   2 * 4 * 120 * 80}}
            : clip ? instantmesh_reference::Files{{"input.bin", 3 * 224 * 224},
                                                  {"embeds.bin", 1024}}
                   : instantmesh_reference::Files{
                             {"enc_in.bin", 3 * 512 * 512},
                             {"enc_lat.bin", 4 * 64 * 64},
                             {"dec_lat.bin", 4 * 80 * 120},
                             {"dec_img.bin", 3 * 640 * 960}};
    auto identity = instantmesh_reference::identity(
            clip ? "cpp_ggml/convert/parity_clip_vision.py"
                 : "cpp_ggml/convert/parity_vae.py",
            clip ? "d11fc8ed7fd9ac4702867c67df936b8877d06ed5c6083cefdf46457fcbc"
                   "c760d"
                 : "a36c852bfa2c9df72be266584b97a9950bc3775d7544f04ca01201a212c"
                   "2451f");
    identity.insert("checkpoint_revision",
                    "2da07e89919e1a130c9b5add1584c70c7aa065fd");
    identity.insert("checkpoint_sha256",
                    clip ? "ae616c24393dd1854372b0639e5541666f7521cbe219669255e"
                           "865cb7f89466a"
                         : "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667f"
                           "e1f45b90daf9a");
    if (unet) {
        identity = instantmesh_reference::identity(
                "cpp_ggml/convert/dump_unet_stages.py",
                "36b67f2611ddc53fe8aa2163be42e5f3e43c3608a2c2c39143e87047a0bbc5"
                "c6");
        identity.insert("seed", 7);
        identity.insert("checkpoint_revision",
                        "2da07e89919e1a130c9b5add1584c70c7aa065fd");
        identity.insert("checkpoint_sha256",
                        "481a8c8d268810095c6bc85ea71b44a0bb9a5ea6457d6c8792275f"
                        "43989442a3");
        identity.insert("override_revision",
                        "b785b4ecfb6636ef34a08c748f96f6a5686244d0");
        identity.insert("override_sha256",
                        "9d78d78421d1feb6871d13e13e86ed8099628648d7d9c51ffca901"
                        "5b7d5fa3c4");
        identity.insert("pipeline_sha256",
                        "186272f54c39172c5fb4b9b8817821d523248edb4699b274fa7ba4"
                        "fbb4cc34db");
        identity.insert("checkpoint_config_sha256",
                        "3da423cd254e7b4f5911b9c72f3d4ee11807084bdb8914475ef727"
                        "80a2ff95bc");
        identity.insert("batch", 2);
        identity.insert("height", 120);
        identity.insert("width", 80);
        identity.insert("reference_height", 120);
        identity.insert("reference_width", 80);
        identity.insert("context_length", 77);
    }
    const int status =
            instantmesh_reference::authenticate(root, identity, files);
    if (status) return status;
    std::vector<std::vector<float>> inputs;
    for (const auto& file : files) {
        auto values = read(root / file.first, file.second);
        if (values.empty()) {
            std::fprintf(
                    stderr,
                    "FAIL: reference contains invalid floating-point values\n");
            return 1;
        }
        inputs.push_back(std::move(values));
    }
    QFile weights(QString::fromUtf8(modelPath.u8string().c_str()));
    if (!weights.exists()) {
        std::fprintf(stderr, "INCOMPLETE: missing component GGUF\n");
        return 77;
    }
    const std::string filename = modelPath.filename().u8string();
    if (filename != (unet   ? "zero123pp_unet_f16.gguf"
                     : clip ? "zero123pp_cond_f32.gguf"
                            : "zero123pp_vae_f32.gguf"))
        return 2;
    const char* digest = aicore::AssetDigestForFile(
            ("instantmesh_models/" + filename).c_str());
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!digest || !weights.open(QIODevice::ReadOnly) ||
        !hash.addData(&weights) ||
        hash.result().toHex() != QByteArray(digest)) {
        std::fprintf(stderr,
                     "FAIL: GGUF digest does not match the public catalog\n");
        return 1;
    }
    std::string error;
    auto lease = aicore::runtime::acquire_backend_lease(
            argv[4], argc > 5 ? std::atoi(argv[5]) : 4, &error);
    if (!lease) {
        std::fprintf(stderr, "FAIL: explicit backend unavailable: %s\n",
                     error.c_str());
        return 1;
    }
    auto lock = lease.lock();
    const auto backend = lease.handle();
    QJsonObject metrics{
            {"actual_backend",
             ggml_backend_dev_name(ggml_backend_get_device(backend))},
            {"reference_authenticated", 1}};
    bool passed = false;
    using Buffer = std::unique_ptr<float, decltype(&std::free)>;
    if (unet) {
        instantmesh::UnetModel model;
        if (!instantmesh::unet_load(modelPath.u8string(), backend, model,
                                    &error)) {
            std::fprintf(stderr, "FAIL: UNet load：%s\n", error.c_str());
            return 1;
        }
        int h = 0, w = 0;
        if (inputs[3][0] != 999.0f) return 1;
        Buffer output(instantmesh::unet_forward_refonly(
                              model, inputs[0].data(), inputs[1].data(),
                              inputs[2].data(), 77, 2, 120, 80, 120, 80,
                              inputs[3][0], &h, &w),
                      std::free);
        passed = h == 120 && w == 80 &&
                 measure(output.get(), inputs[4], "unet", 0.35, metrics);
    } else if (clip) {
        instantmesh::ClipVisionModel model;
        if (!instantmesh::clip_vision_load(modelPath.u8string(), backend, model,
                                           &error)) {
            std::fprintf(stderr, "FAIL: CLIP load：%s\n", error.c_str());
            return 1;
        }
        int dimension = 0;
        Buffer output(instantmesh::clip_vision_encode(model, inputs[0].data(),
                                                      1, &dimension),
                      std::free);
        passed = dimension == 1024 &&
                 measure(output.get(), inputs[1], "clip", 1e-3, metrics);
    } else {
        instantmesh::VaeModel model;
        if (!instantmesh::vae_load(modelPath.u8string(), backend, model,
                                   &error)) {
            std::fprintf(stderr, "FAIL: VAE load：%s\n", error.c_str());
            return 1;
        }
        int h = 0, w = 0;
        Buffer encoded(instantmesh::vae_encode_mode(model, inputs[0].data(), 1,
                                                    512, 512, &h, &w),
                       std::free);
        const bool encode =
                h == 64 && w == 64 &&
                measure(encoded.get(), inputs[1], "vae_encode", 5e-4, metrics);
        Buffer decoded(instantmesh::vae_decode(model, inputs[2].data(), 1, 80,
                                               120, &h, &w),
                       std::free);
        const bool decode =
                h == 640 && w == 960 &&
                measure(decoded.get(), inputs[3], "vae_decode", 5e-3, metrics);
        passed = encode && decode;
    }
    std::puts(
            QJsonDocument(metrics).toJson(QJsonDocument::Compact).constData());
    return passed ? 0 : 1;
}
