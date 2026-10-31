// SPDX-License-Identifier: MIT
// Test-only: frozen four-image component references, per-output absolute error and added error against same-weight upstream.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>

#include "aicore/asset_digests.h"
#include "reference_inputs.hpp"
#include "src/common/ggml_backend_registry.hpp"
#include "src/tasks/instantmesh/dino.hpp"
#include "src/tasks/instantmesh/lrm_transformer.hpp"

namespace instantmesh_geometry_reference {
using Path = std::filesystem::path;
constexpr std::array<const char*, 4> scenarios{{"blue_cat", "cute_horse", "fox", "robot"}};

inline int json(const Path& path, QJsonObject& object) {
    QFile file(QString::fromUtf8(path.u8string().c_str()));
    if (!file.exists()) {
        std::fprintf(stderr, "INCOMPLETE: missing reference/frozen gate: %s\n", path.u8string().c_str());
        return 77;
    }
    QJsonParseError error;
    if (!file.open(QIODevice::ReadOnly)) return 1;
    const auto document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        std::fprintf(stderr, "FAIL: invalid reference/frozen-gate JSON\n");
        return 1;
    }
    object = document.object();
    return 0;
}

inline QByteArray digest(const Path& path) {
    QFile file(QString::fromUtf8(path.u8string().c_str()));
    QCryptographicHash hash(QCryptographicHash::Sha256);
    return file.open(QIODevice::ReadOnly) && hash.addData(&file)
                   ? hash.result().toHex() : QByteArray();
}

inline int numeric(const Path& path, const QJsonObject& record,
                   size_t count, std::vector<float>& values) {
    QFile file(QString::fromUtf8(path.u8string().c_str()));
    if (!file.exists()) {
        std::fprintf(stderr, "INCOMPLETE: missing component reference %s\n", path.u8string().c_str());
        return 77;
    }
    const auto hash = record.value("sha256").toString().toLatin1();
    if (hash.size() != 64 || file.size() != qint64(count * sizeof(float)) ||
        record.value("bytes").toDouble(-1) != double(count * sizeof(float)) ||
        digest(path) != hash || !file.open(QIODevice::ReadOnly)) {
        std::fprintf(stderr, "FAIL: component reference size/digest mismatch\n");
        return 1;
    }
    values.resize(count);
    if (file.read(reinterpret_cast<char*>(values.data()), qint64(count * sizeof(float))) !=
        qint64(count * sizeof(float)) ||
        !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
        std::fprintf(stderr, "FAIL: component reference contains invalid floating-point values\n");
        return 1;
    }
    return 0;
}

struct Case {
    QString scenario;
    std::vector<float> image, camera, expected, upstream;
    QJsonObject frozen;
};

inline int prepare(const Path& root, bool dino, std::vector<Case>& cases) {
    QJsonObject metadata;
    if (const int status = json(root / "provenance.json", metadata)) return status;
    const QJsonObject identity{
        {"kind", "independent-geometry-components-not-full-pipeline-reference"},
        {"upstream_revision", "32b14574edc9caba95f03672bfa1eeddd13c3af7"},
        {"checkpoint_revision", "b785b4ecfb6636ef34a08c748f96f6a5686244d0"},
        {"checkpoint_sha256", "329f7aae9b583fd7c1b27d0221463db7b808a03c48a1f7aa26649af6a03b91a1"},
        {"checkpoint_bytes", 1514818077}, {"seed", 42}, {"views", 6},
        {"image_size", 224}, {"dino_sequence", 197},
        {"input_archive_sha256", "490f85a45c37f278bff8011109e91f66488d155c438015e3fdd7b95bbc297cac"}};
    for (auto it = identity.begin(); it != identity.end(); ++it) {
        if (metadata.value(it.key()) != it.value()) {
            std::fprintf(stderr, "FAIL: component reference identity mismatch: %s\n", it.key().toUtf8().constData());
            return 1;
        }
    }
    const QJsonObject sources{
        {"cpp_ggml/convert/prep_input.py", "4e2c4d43f967990742b0e2cb0d87c29d72509c528b0aec834ac0881ff6244b0f"},
        {"cpp_ggml/convert/parity_dino.py", "b869c20e959acf3e19a15ac9a9d1162f0dd2aa2edc3e3e43a47f77314aa06550"},
        {"cpp_ggml/convert/parity_lrm.py", "c30929464e65ffc63729982a53e71972aa206b49e4c6b0e55c7210057a2a3a13"}};
    if (metadata.value("generator_sources_sha256").toObject() != sources) {
        std::fprintf(stderr, "FAIL: independent-framework generator_sources_sha256 mismatch\n");
        return 1;
    }
    for (const auto* name : scenarios) {
        Case item;
        item.scenario = QString::fromUtf8(name);
        const auto data = metadata.value("scenarios").toObject().value(item.scenario).toObject();
        const auto hashes = data.value("files_sha256").toObject();
        const auto bytes = data.value("files_bytes").toObject();
        const auto load = [&](const char* file, size_t count, std::vector<float>& values) {
            const QString key = QString::fromUtf8(file);
            return numeric(root / name / file,
                           {{"sha256", hashes.value(key)}, {"bytes", bytes.value(key)}}, count, values);
        };
        if (dino) {
            if (const int status = load("image.bin", 6 * 3 * 224 * 224, item.image)) return status;
            if (const int status = load("camera.bin", 6 * 16, item.camera)) return status;
            if (const int status = load("dino_features.bin", 6 * 197 * 768, item.expected)) return status;
        } else {
            if (const int status = load("dino_features.bin", 6 * 197 * 768, item.image)) return status;
            if (const int status = load("triplanes.bin", 3 * 80 * 64 * 64, item.expected)) return status;
        }
        cases.push_back(std::move(item));
    }
    return 0;
}

inline QJsonObject errors(const float* actual, const float* expected, size_t count);

inline int frozen_limits(const Path& manifest, const Path& baseline, const Path& root,
                         const QString& backend, const QString& quant, bool dino,
                         const QString& filename, std::vector<Case>& cases) {
    QJsonObject document;
    if (const int status = json(manifest, document)) return status;
    const auto limits = document.value("instantmesh_geometry_component_accuracy").toObject();
    if (limits.isEmpty()) {
        std::fprintf(stderr, "INCOMPLETE: missing frozen four-image component numeric gates\n");
        return 77;
    }
    if (limits.value("reference_receipt_sha256").toString().toLatin1() != digest(root / "provenance.json") ||
        limits.value("locked_before_candidate_component_sampling") != QJsonValue(true)) {
        std::fprintf(stderr, "FAIL: component reference/frozen-gate identity mismatch\n");
        return 1;
    }
    const auto model = limits.value("models").toObject().value(filename).toObject();
    const char* catalog = aicore::AssetDigestForFile(("instantmesh_models/" + filename).toUtf8().constData());
    if (!catalog || model.value("sha256").toString() != QString::fromLatin1(catalog)) {
        std::fprintf(stderr, "FAIL: frozen GGUF does not match the public catalog\n");
        return 1;
    }
    const QString component = dino ? "dino" : "lrm";
    for (auto& item : cases) {
        item.frozen = limits.value("cases").toObject().value(item.scenario + "/" + backend + "/" + quant).toObject();
        if (item.frozen.isEmpty()) {
            std::fprintf(stderr, "INCOMPLETE: upstream component gates are not frozen for this backend/quant\n");
            return 77;
        }
        const auto record = item.frozen.value("baseline_files").toObject().value(component).toObject();
        const auto file = record.value("file").toString();
        const auto actual = record.value("actual_backend").toString();
        if (file.isEmpty() || file.contains('/') || file.contains('\\') || file.contains("..") ||
            actual.isEmpty() || (backend == "cpu") != (actual == "CPU")) {
            std::fprintf(stderr, "FAIL: invalid upstream path or actual-backend identity\n");
            return 1;
        }
        if (const int status = numeric(baseline / file.toStdString(), record, item.expected.size(), item.upstream)) return status;
        const int parts = dino ? 6 : 3;
        const size_t count = item.expected.size() / parts;
        for (int i = 0; i < parts; ++i) {
            const QString prefix = component + (dino ? "_view" : "_plane") + QString::number(i);
            const auto measured = errors(item.upstream.data() + i * count, item.expected.data() + i * count, count);
            for (const auto* metric : {"max_abs", "rmse"}) {
                const QString key = prefix + "_" + metric;
                const auto upstream = item.frozen.value("upstream_errors").toObject().value(key);
                if (!upstream.isDouble() || !std::isfinite(upstream.toDouble()) ||
                    std::abs(upstream.toDouble() - measured.value(metric).toDouble()) >
                        1e-12 + 1e-10 * upstream.toDouble()) {
                    std::fprintf(stderr, "FAIL: frozen upstream per-output error mismatch\n");
                    return 1;
                }
                for (const auto* bounds : {"absolute_bounds", "added_error_bounds"}) {
                    const auto value = item.frozen.value(bounds).toObject().value(key);
                    const double multiplier = std::string(bounds) == "absolute_bounds" ? 1.10 : 0.10;
                    const double frozen = multiplier * upstream.toDouble() + 1e-5;
                    if (!value.isDouble() || !std::isfinite(value.toDouble()) || value.toDouble() <= 0 ||
                        std::abs(value.toDouble() - frozen) > 1e-12 + 1e-10 * frozen) {
                        std::fprintf(stderr, "FAIL: per-output frozen numeric gate missing/invalid/relaxed\n");
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

inline QJsonObject errors(const float* actual, const float* expected, size_t count) {
    double maximum = 0, squared = 0, absolute = 0, signs = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) return {};
        const double difference = double(actual[i]) - expected[i];
        maximum = std::max(maximum, std::abs(difference));
        squared += difference * difference;
        absolute += std::abs(difference);
        signs += std::signbit(actual[i]) != std::signbit(expected[i]);
    }
    return {{"max_abs", maximum}, {"rmse", std::sqrt(squared / count)},
            {"mean_abs", absolute / count}, {"sign_changes", signs}};
}

inline int run(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "INCOMPLETE: component requires GGUF, four-image reference, device, threads, manifest, baseline\n");
        return 77;
    }
    const bool dino = std::string(argv[1]) == "dino";
    const Path modelPath(argv[2]), root(argv[3]);
    std::vector<Case> cases;
    if (const int status = prepare(root, dino, cases)) return status;
    const QString filename = QString::fromUtf8(modelPath.filename().u8string().c_str());
    const QString stem = dino ? "dino" : "lrm_transformer";
    QString quant;
    for (const auto* candidate : {"f32", "f16", "q8"})
        if (filename == stem + "_" + candidate + ".gguf") quant = candidate;
    if (quant.isEmpty()) {
        std::fprintf(stderr, "INCOMPLETE: missing published component GGUF\n");
        return 77;
    }
    const QString backendName = QString::fromUtf8(argv[4]).toLower();
    if (const int status = frozen_limits(Path(argv[6]), Path(argv[7]), root, backendName, quant, dino, filename, cases)) return status;
    if (!QFile::exists(QString::fromUtf8(modelPath.u8string().c_str()))) {
        std::fprintf(stderr, "INCOMPLETE: missing component GGUF\n");
        return 77;
    }
    const char* hash = aicore::AssetDigestForFile(("instantmesh_models/" + filename).toUtf8().constData());
    if (!hash || digest(modelPath) != QByteArray(hash)) {
        std::fprintf(stderr, "FAIL: GGUF digest does not match the public catalog\n");
        return 1;
    }
    std::string error;
    auto lease = aicore::runtime::acquire_backend_lease(argv[4], std::atoi(argv[5]), &error);
    if (!lease) {
        std::fprintf(stderr, "FAIL: explicit backend unavailable: %s\n", error.c_str());
        return 1;
    }
    auto lock = lease.lock();
    instantmesh::DinoModel dinoModel;
    instantmesh::LrmTransformerModel lrmModel;
    const bool loaded = dino ? instantmesh::dino_load(modelPath.u8string(), lease.handle(), dinoModel, &error)
                             : instantmesh::lrm_transformer_load(modelPath.u8string(), lease.handle(), lrmModel, &error);
    if (!loaded) {
        std::fprintf(stderr, "FAIL: component load: %s\n", error.c_str());
        return 1;
    }
    QJsonObject metrics{{"reference_authenticated", 1}, {"frozen_limits_authenticated", 1},
        {"actual_backend", ggml_backend_dev_name(ggml_backend_get_device(lease.handle()))},
        {"reference_receipt_sha256", QString::fromLatin1(digest(root / "provenance.json"))}};
    bool passed = true;
    for (const auto& item : cases) {
        int seq = 0, hidden = 0, planes = 0, dimension = 0, h = 0, w = 0;
        using Buffer = std::unique_ptr<float, decltype(&std::free)>;
        Buffer output(dino ? instantmesh::dino_encode(dinoModel, item.image.data(), item.camera.data(), 6, 224, 224, &seq, &hidden)
                           : instantmesh::lrm_transformer_forward(lrmModel, item.image.data(), 1, 1182, &planes, &dimension, &h, &w), std::free);
        if (!output || (dino ? seq != 197 || hidden != 768 : planes != 3 || dimension != 80 || h != 64 || w != 64)) return 1;
        const size_t parts = dino ? 6 : 3, count = item.expected.size() / parts;
        const QString component = dino ? "dino" : "lrm";
        for (size_t i = 0; i < parts; ++i) {
            const auto absolute = errors(output.get() + i * count, item.expected.data() + i * count, count);
            const auto upstream = errors(item.upstream.data() + i * count, item.expected.data() + i * count, count);
            const auto added = errors(output.get() + i * count, item.upstream.data() + i * count, count);
            if (absolute.isEmpty() || upstream.isEmpty() || added.isEmpty()) return 1;
            const QString part = component + (dino ? "_view" : "_plane") + QString::number(i);
            const QString prefix = item.scenario + "_" + part;
            for (auto it = absolute.begin(); it != absolute.end(); ++it) {
                metrics.insert(prefix + "_" + it.key(), it.value());
                metrics.insert(prefix + "_upstream_" + it.key(), upstream.value(it.key()));
                metrics.insert(prefix + "_added_" + it.key(), added.value(it.key()));
                if (it.key() == "max_abs" || it.key() == "rmse") {
                    const double bound = item.frozen.value("absolute_bounds").toObject().value(part + "_" + it.key()).toDouble();
                    const double budget = item.frozen.value("added_error_bounds").toObject().value(part + "_" + it.key()).toDouble();
                    const double ratio = it.value().toDouble() / bound;
                    const double addedRatio = added.value(it.key()).toDouble() / budget;
                    metrics.insert(prefix + "_" + it.key() + "_ratio", ratio);
                    metrics.insert(prefix + "_added_" + it.key() + "_ratio", addedRatio);
                    metrics.insert(prefix + "_" + it.key() + "_limit", bound);
                    metrics.insert(prefix + "_added_" + it.key() + "_limit", budget);
                    passed = passed && ratio <= 1 && addedRatio <= 1;
                }
            }
        }
    }
    std::puts(QJsonDocument(metrics).toJson(QJsonDocument::Compact).constData());
    return passed ? 0 : 1;
}
}  // namespace instantmesh_geometry_reference
