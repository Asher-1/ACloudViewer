// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThreadPool>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "aicore/relation_capi.h"
#include "aicore/runtime_capi.h"
namespace {
void check(bool yes, const char* msg) {
    if (!yes) throw std::runtime_error(msg);
}
QJsonDocument json(const QString& path) {
    QFile f(path);
    check(f.open(QIODevice::ReadOnly), "cannot open input JSON");
    QJsonParseError e;
    auto d = QJsonDocument::fromJson(f.readAll(), &e);
    check(e.error == QJsonParseError::NoError, "invalid input JSON");
    return d;
}
template <class T>
void write(std::ofstream& f, const T* p, size_t n) {
    f.write(reinterpret_cast<const char*>(p), sizeof(T) * n);
}
void raw(const std::string& path, const aicore_relation_result* r) {
    std::ofstream f(path, std::ios::binary);
    const char magic[8] = {'R', 'A', 'F', 'O', 'v', '1', 0, 0};
    f.write(magic, 8);
    uint32_t k = r->pair_count, v = r->predicate_count;
    write(f, &k, 1);
    write(f, &v, 1);
    write(f, r->valid_mask, k);
    write(f, r->subject_indices, k);
    write(f, r->object_indices, k);
    write(f, r->pair_logits, k);
    write(f, r->predicate_logits, size_t(k) * v);
    check(bool(f), "raw output write failed");
}
void write_catalog_image_ppm(const std::string& catalog_path,
                             int sample,
                             const std::string& output_path) {
    const auto samples = json(QString::fromStdString(catalog_path))
                                 .object()["samples"]
                                 .toArray();
    check(sample >= 0 && sample < samples.size(),
          "invalid sample index for image decode");
    const auto row = samples[sample].toObject();
    QImage image(
            QDir(QFileInfo(QString::fromStdString(catalog_path)).absolutePath())
                    .filePath(row["image"].toString()));
    image = image.convertToFormat(QImage::Format_RGB888);
    check(!image.isNull(), "cannot decode image");
    std::ofstream f(output_path, std::ios::binary);
    check(bool(f), "cannot create decoded PPM");
    f << "P6\n" << image.width() << ' ' << image.height() << "\n255\n";
    for (int y = 0; y < image.height(); ++y)
        f.write(reinterpret_cast<const char*>(image.constScanLine(y)),
                image.width() * 3);
    check(bool(f), "decoded PPM write failed");
}
}  // namespace
int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--contracts" || a == "--no-strict" ||
                a == "--auto-arithmetic" || a == "--decompose" ||
                a == "--decode-only")
                args[a] = "1";
            else {
                check(i + 1 < argc, "missing option value");
                args[a] = argv[++i];
            }
        }
        auto option = [&](const char* key, const char* def = "") {
            auto i = args.find(key);
            return i == args.end() ? std::string(def) : i->second;
        };
        if (args.count("--decode-only")) {
            const auto catalog = option("--catalog");
            const auto pixels = option("--pixels");
            check(!catalog.empty() && !pixels.empty(),
                  "--decode-only requires --catalog and --pixels");
            write_catalog_image_ppm(catalog, std::stoi(option("--sample", "0")),
                                    pixels);
            return 0;
        }
        const auto model = option("--model"),
                   backend = option("--backend", "cpu"),
                   tokenizer = option("--tokenizer"), dump = option("--dump");
        check(!model.empty(), "--model is required");
        const int warmup = std::stoi(option("--warmup", "2")),
                  repeat = std::stoi(option("--repeat", "5"));
        check(warmup >= 0 && repeat > 0, "invalid repetitions");
        auto o = aicore_relation_default_options();
        o.model_path = model.c_str();
        o.device = backend.c_str();
        o.threads = std::stoi(option("--threads", "4"));
        o.raw_logits = 1;
        o.strict_f32 = args.count("--auto-arithmetic")
                               ? -1
                               : !args.count("--no-strict");
        o.decompose = args.count("--decompose");
        const bool text = args.count("--texts");
        o.text_only = text;
        o.tokenizer_path = tokenizer.c_str();
        const auto student = option("--student");
        std::vector<QByteArray> predicateNames;
        std::vector<const char*> predicatePointers;
        if (args.count("--predicate-texts")) {
            for (const auto& v :
                 json(QString::fromStdString(option("--predicate-texts")))
                         .array())
                predicateNames.push_back(v.toString().toUtf8());
            for (const auto& v : predicateNames)
                predicatePointers.push_back(v.constData());
            o.predicate_texts = predicatePointers.data();
            o.predicate_text_count = predicatePointers.size();
            o.text_student_path = student.c_str();
        }
        if (!dump.empty()) o.diagnostic_directory = dump.c_str();
        std::unique_ptr<aicore_relation_ctx, decltype(&aicore_relation_free)>
                ctx(aicore_relation_create(&o), aicore_relation_free);
        check(aicore_relation_ready(ctx.get()),
              aicore_relation_last_error(ctx.get()));
        std::vector<double> times, forwards;
        std::vector<float> logits;
        uint64_t hash = 1469598103934665603ull;
        auto fingerprint = [&](const void* p, size_t n) {
            auto* b = static_cast<const uint8_t*>(p);
            for (size_t i = 0; i < n; ++i)
                hash = (hash ^ b[i]) * 1099511628211ull;
        };
        double stability = 0;
        size_t outputCount = 0;
        if (text) {
            auto values =
                    json(QString::fromStdString(option("--texts"))).array();
            std::vector<QByteArray> names;
            for (const auto& v : values) names.push_back(v.toString().toUtf8());
            std::vector<const char*> pointers;
            for (const auto& v : names) pointers.push_back(v.constData());
            for (int i = -warmup; i < repeat; ++i) {
                aicore_relation_embeddings* p = nullptr;
                check(aicore_relation_encode_text(ctx.get(), pointers.data(),
                                                  pointers.size(), &p) == 0,
                      aicore_relation_last_error(ctx.get()));
                std::unique_ptr<aicore_relation_embeddings,
                                decltype(&aicore_relation_embeddings_free)>
                        r(p, aicore_relation_embeddings_free);
                check(r->dimension == 512 && r->count == pointers.size(),
                      "bad text dimensions");
                size_t n = r->count * r->dimension;
                if (i >= 0) {
                    times.push_back(r->timings.e2e_ms);
                    if (logits.empty())
                        logits.assign(r->values, r->values + n);
                    else
                        for (size_t j = 0; j < n; ++j)
                            stability = std::max(
                                    stability,
                                    double(std::abs(logits[j] - r->values[j])));
                }
                if (i == repeat - 1) {
                    if (args.count("--contracts")) {
                        const char* changed[] = {"under", "Σς İ", "new"};
                        auto encode = [&](aicore_relation_ctx* target) {
                            aicore_relation_embeddings* raw = nullptr;
                            check(aicore_relation_encode_text(target, changed,
                                                              3, &raw) == 0,
                                  aicore_relation_last_error(target));
                            return std::unique_ptr<
                                    aicore_relation_embeddings,
                                    decltype(&aicore_relation_embeddings_free)>(
                                    raw, aicore_relation_embeddings_free);
                        };
                        auto hot = encode(ctx.get());
                        std::unique_ptr<aicore_relation_ctx,
                                        decltype(&aicore_relation_free)>
                                peer(aicore_relation_create(&o),
                                     aicore_relation_free);
                        check(aicore_relation_ready(peer.get()),
                              aicore_relation_last_error(peer.get()));
                        auto cold = encode(peer.get());
                        peer.reset();
                        aicore_relation_shutdown();
                        auto again = encode(ctx.get());
                        check(hot->count == 3 && cold->count == 3 &&
                                      again->count == 3,
                              "changed text request dimensions");
                        for (size_t j = 0; j < 3 * 512; ++j) {
                            check(std::abs(hot->values[j] - cold->values[j]) <
                                          .0002f,
                                  "resident text matches independent context");
                            check(hot->values[j] == again->values[j],
                                  "shutdown preserves resident text session");
                        }
                        for (size_t j = 0; j < n; ++j)
                            check(r->values[j] == logits[j],
                                  "later text requests preserve owned result");
                        aicore_relation_embeddings* invalid =
                                reinterpret_cast<aicore_relation_embeddings*>(
                                        1);
                        check(aicore_relation_encode_text(ctx.get(), nullptr, 1,
                                                          &invalid) < 0 &&
                                      !invalid,
                              "invalid text clears output");
                        aicore_pipeline_timings cleared{};
                        check(aicore_relation_last_pipeline_timings(
                                      ctx.get(), &cleared) < 0,
                              "invalid text clears timings");
                        auto recovered = encode(ctx.get());
                        check(recovered->values[0] == hot->values[0],
                              "resident text recovers after invalid request");
                    }
                    ctx.reset();
                    check(std::isfinite(r->values[0]), "text result lifetime");
                    std::ofstream f(option("--output"), std::ios::binary);
                    write(f, r->values, n);
                    check(bool(f), "write text output");
                    fingerprint(r->values, sizeof(float) * n);
                    outputCount = n;
                }
            }
        } else {
            const QString catalog = QString::fromStdString(option("--catalog"));
            const auto samples = json(catalog).object()["samples"].toArray();
            int sample = std::stoi(option("--sample", "0"));
            check(sample >= 0 && sample < samples.size(),
                  "invalid sample index");
            const auto s = samples[sample].toObject();
            QImage image(QDir(QFileInfo(catalog).absolutePath())
                                 .filePath(s["image"].toString()));
            image = image.convertToFormat(QImage::Format_RGB888);
            check(!image.isNull(), "cannot decode image");
            if (args.count("--pixels")) {
                std::ofstream f(option("--pixels"), std::ios::binary);
                for (int y = 0; y < image.height(); ++y)
                    write(f, image.constScanLine(y), image.width() * 3);
            }
            aicore_image_view view{image.constBits(), image.width(),
                                   image.height(), size_t(image.bytesPerLine()),
                                   AICORE_IMAGE_RGB8};
            std::vector<float> boxes;
            for (const auto& obj : s["objects"].toArray())
                for (const auto& v : obj.toObject()["box_xyxy"].toArray())
                    boxes.push_back(float(v.toDouble()));
            auto q = aicore_relation_default_request();
            q.boxes_xyxy = boxes.data();
            q.object_count = boxes.size() / 4;
            std::vector<float> scores;
            QByteArray masks;
            if (args.count("--scores")) {
                for (const auto& v :
                     json(QString::fromStdString(option("--scores"))).array())
                    scores.push_back(float(v.toDouble()));
                check(scores.size() == q.object_count, "invalid score count");
                q.box_scores = scores.data();
            }
            if (args.count("--masks")) {
                QFile f(QString::fromStdString(option("--masks")));
                check(f.open(QIODevice::ReadOnly), "cannot open masks");
                masks = f.readAll();
                q.masks = reinterpret_cast<const uint8_t*>(masks.constData());
                q.mask_bytes = size_t(masks.size());
            }
            check(q.object_count >= 2, "need at least two boxes");
            auto infer = [&](const aicore_image_view& v) {
                aicore_relation_result* p = nullptr;
                check(aicore_relation_infer_image(ctx.get(), &v, &q, &p) == 0,
                      aicore_relation_last_error(ctx.get()));
                return std::unique_ptr<aicore_relation_result,
                                       decltype(&aicore_relation_result_free)>(
                        p, aicore_relation_result_free);
            };
            if (args.count("--contracts")) {
                auto bad = view;
                bad.row_stride_bytes = 1;
                aicore_relation_result* p =
                        reinterpret_cast<aicore_relation_result*>(1);
                check(aicore_relation_infer_image(ctx.get(), &bad, &q, &p) <
                                      0 &&
                              !p,
                      "invalid stride");
                aicore_pipeline_timings t{};
                check(aicore_relation_last_pipeline_timings(ctx.get(), &t) < 0,
                      "failure clears timings");
                auto* token = aicore_cancel_token_new();
                aicore_cancel_token_request(token);
                aicore_cancel_scope_begin(token);
                check(aicore_relation_infer_image(ctx.get(), &view, &q, &p) ==
                                      1 &&
                              !p,
                      "cancel before infer");
                aicore_cancel_scope_end(token);
                aicore_cancel_token_free(token);
                auto first = infer(view);
                std::unique_ptr<aicore_relation_ctx,
                                decltype(&aicore_relation_free)>
                        peer(aicore_relation_create(&o), aicore_relation_free);
                check(aicore_relation_ready(peer.get()),
                      aicore_relation_last_error(peer.get()));
                aicore_relation_result* peerRaw = nullptr;
                check(aicore_relation_infer_image(peer.get(), &view, &q,
                                                  &peerRaw) == 0,
                      aicore_relation_last_error(peer.get()));
                std::unique_ptr<aicore_relation_result,
                                decltype(&aicore_relation_result_free)>
                        peerResult(peerRaw, aicore_relation_result_free);
                peer.reset();
                check(peerResult->predicate_count == first->predicate_count,
                      "independent context result");
                aicore_relation_shutdown();
                auto second = infer(view);
                check(first->pair_count == second->pair_count,
                      "shutdown live context");
                QImage rgba = image.convertToFormat(QImage::Format_RGBA8888);
                aicore_image_view rv{rgba.constBits(), rgba.width(),
                                     rgba.height(), size_t(rgba.bytesPerLine()),
                                     AICORE_IMAGE_RGBA8};
                auto third = infer(rv);
                check(first->pair_count == third->pair_count,
                      "format parity count");
                for (size_t j = 0;
                     j < first->pair_count * first->predicate_count; ++j)
                    check(std::abs(first->predicate_logits[j] -
                                   third->predicate_logits[j]) < .0001,
                          "format parity");
                const size_t stride = size_t(image.width()) * 3 + 13;
                std::vector<uint8_t> padded(stride * image.height(), 0xcd);
                for (int y = 0; y < image.height(); ++y)
                    std::memcpy(padded.data() + stride * y,
                                image.constScanLine(y),
                                size_t(image.width()) * 3);
                auto paddedView = view;
                paddedView.data = padded.data();
                paddedView.row_stride_bytes = stride;
                auto fourth = infer(paddedView);
                check(fourth->pair_count == first->pair_count,
                      "padded stride shape");
                for (size_t j = 0;
                     j < first->pair_count * first->predicate_count; ++j) {
                    check(std::abs(first->predicate_logits[j] -
                                   second->predicate_logits[j]) < .0001,
                          "shutdown preserves values");
                    check(std::abs(first->predicate_logits[j] -
                                   fourth->predicate_logits[j]) < .0001,
                          "padded stride values");
                }
                auto invalidRequest = q;
                invalidRequest.mask_bytes = 1;
                p = reinterpret_cast<aicore_relation_result*>(1);
                check(aicore_relation_infer_image(ctx.get(), &view,
                                                  &invalidRequest, &p) < 0 &&
                              !p,
                      "invalid mask capacity clears output");
            }
            // Qt parallelizes image conversion. Retire those temporary workers
            // before timing so their idle-expiry teardown cannot interrupt the
            // pinned inference cores (the native CLI has no Qt worker pool).
            check(QThreadPool::globalInstance()->waitForDone(5000),
                  "image conversion workers did not finish");
            using Result = decltype(infer(view));
            Result r(nullptr, aicore_relation_result_free);
            for (int i = -warmup; i < repeat; ++i) {
                r = infer(view);
                check(r->predicate_logits && r->pair_logits &&
                              r->predicate_names,
                      "missing raw tensors");
                check(r->timings.valid_fields ==
                              (AICORE_TIMING_PREPROCESS | AICORE_TIMING_E2E),
                      "truthful timing stages");
                if (i >= 0) {
                    times.push_back(r->timings.e2e_ms);
                    forwards.push_back(r->forward_ms);
                }
            }
            ctx.reset();
            const size_t n = r->pair_count * r->predicate_count;
            check(n > 0, "empty timed output");
            for (size_t j = 0; j < n; ++j)
                check(std::isfinite(r->predicate_logits[j]),
                      "nonfinite timed logits");
            {
                std::cout << "forward_ms=" << r->forward_ms
                          << "\npairs=" << r->pair_count
                          << "\npredicates=" << r->predicate_count
                          << "\nbackend_buffer_bytes="
                          << r->backend_buffer_bytes << "\n";
                if (args.count("--edges")) {
                    auto encode = [&](const aicore_relation_edge* p,
                                      size_t count) {
                        QJsonArray edges;
                        for (size_t j = 0; j < count; ++j)
                            edges.append(QJsonObject{
                                    {"subject", int(p[j].subject)},
                                    {"object", int(p[j].object)},
                                    {"predicate_id", int(p[j].predicate)},
                                    {"predicate",
                                     QString::fromUtf8(
                                             r->predicate_names
                                                     [p[j].predicate])},
                                    {"score", p[j].score}});
                        return edges;
                    };
                    QJsonObject output{
                            {"relations",
                             encode(r->relations, r->relation_count)}};
                    if (o.decompose) {
                        output["spatial"] =
                                encode(r->spatial_relations, r->spatial_count);
                        output["semantic"] = encode(r->semantic_relations,
                                                    r->semantic_count);
                    }
                    QFile f(QString::fromStdString(option("--edges")));
                    check(f.open(QIODevice::WriteOnly),
                          "cannot write typed edges");
                    auto bytes = QJsonDocument(output).toJson();
                    check(f.write(bytes) == bytes.size(), "typed edges write");
                }
                check(std::isfinite(r->predicate_logits[0]) &&
                              r->predicate_names[0],
                      "result lifetime");
                raw(option("--output"), r.get());
                fingerprint(r->predicate_logits, n * sizeof(float));
                outputCount = n;
            }
        }
        std::cout << "e2e_iteration_ms=";
        for (size_t i = 0; i < times.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << times[i];
        }
        std::cout << '\n';
        std::sort(times.begin(), times.end());
        if (!forwards.empty()) {
            std::cout << "forward_iteration_ms=";
            for (size_t i = 0; i < forwards.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << forwards[i];
            }
            std::cout << '\n';
            std::sort(forwards.begin(), forwards.end());
            std::cout << "forward_median_ms=" << forwards[forwards.size() / 2]
                      << "\nforward_mean_ms="
                      << std::accumulate(forwards.begin(), forwards.end(),
                                         0.0) /
                                 forwards.size()
                      << "\n";
        }
        std::cout << "latency_ms=" << times[times.size() / 2] << "\np95_ms="
                  << times[size_t(std::ceil(times.size() * .95)) - 1]
                  << "\nstability_max_diff=" << stability
                  << "\nfingerprint=" << hash
                  << "\noutput_count=" << outputCount << "\n";
        check(stability < .0001, "unstable outputs");
        aicore_relation_shutdown();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
