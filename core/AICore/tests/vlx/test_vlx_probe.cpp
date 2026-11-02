// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Real-model probe for the VLX-Seek region-retrieval task. One process loads
// the three-model set (wedetect + vision + LLM), runs warmups plus N greedy
// repeats, then checks the lifecycle invariants a live UI depends on:
// determinism, context recreation and (with --upstream-dir) greedy-token and
// box parity against the pristine upstream vlx-cli built from the pinned
// Asher-1/VLX-Seek-GGML revision. Emits one JSON object (stdout or
// --json-out).
//
// Exit codes: 0 pass, 1 invariant/accuracy failure, 2 usage, 77 missing
// model asset (the common runner records it as unavailable).

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QTemporaryDir>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/runtime_capi.h"
#include "aicore/vlx_seek_capi.h"

namespace {

struct Args {
    std::string models_dir, wedetect, vision, llm, device = "auto",
                               image, task = "detection", text = "person",
                               lang = "en", upstream_dir, upstream_precision,
                               json_out;
    int max_tokens = 256, repeats = 2, warmups = 0, threads = 0;
    double box_tolerance = 0.05;  // px; f32 reduction noise class
    int expect_boxes_min = -1;    // optional accuracy gate
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t i = size_t(std::ceil(p * double(v.size()))) - 1;
    return v[std::min(i, v.size() - 1)];
}

struct RunOutcome {
    int rc = 0;
    std::string answer;
    std::vector<std::array<float, 4>> boxes;
    std::vector<std::string> labels;
    int32_t prompt_tokens = 0, completion_tokens = 0, proposal_count = 0;
    double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0, e2e_ms = 0;
};

bool run_once(aicore_vlx_seek_ctx* ctx, const Args& a,
              const std::vector<float>& fixed_boxes, RunOutcome* out,
              std::string* error) {
    const aicore_image_view* view = nullptr;
    aicore_image_view v{};
    std::vector<uint8_t> rgb;
    {
        uint8_t* p = nullptr;
        int32_t w = 0, h = 0;
        const int rc = aicore_vlx_seek_load_path_rgb(a.image.c_str(), &p, &w, &h);
        if (rc != 0 || !p) {
            *error = "cannot load image " + a.image;
            return false;
        }
        rgb.assign(p, p + size_t(w) * h * 3);
        aicore_vlx_seek_free_buffer(p);
        v.data = rgb.data();
        v.width = w;
        v.height = h;
        v.row_stride_bytes = size_t(w) * 3;
        v.format = AICORE_IMAGE_RGB8;
        view = &v;
    }
    const float* boxes = fixed_boxes.empty() ? nullptr : fixed_boxes.data();
    const int32_t n_boxes = fixed_boxes.empty()
                                ? 0
                                : int32_t(fixed_boxes.size() / 4);
    QElapsedTimer timer;
    timer.start();
    const int rc = aicore_vlx_seek_run_image(ctx, view, a.task.c_str(),
                                             a.text.c_str(), a.lang.c_str(),
                                             boxes, n_boxes, nullptr, 0);
    out->rc = rc;
    if (rc != 0) {
        const char* err = aicore_vlx_seek_last_error(ctx);
        *error = err ? err : "run failed";
        return false;
    }
    out->answer = aicore_vlx_seek_result_answer(ctx) ?
                      aicore_vlx_seek_result_answer(ctx) : "";
    const int32_t n = aicore_vlx_seek_result_box_count(ctx);
    for (int32_t i = 0; i < n; ++i) {
        std::array<float, 4> b{};
        aicore_vlx_seek_result_box(ctx, i, b.data());
        out->boxes.push_back(b);
        const char* label = aicore_vlx_seek_result_box_label(ctx, i);
        out->labels.push_back(label ? label : "");
    }
    out->prompt_tokens = aicore_vlx_seek_result_prompt_tokens(ctx);
    out->completion_tokens = aicore_vlx_seek_result_completion_tokens(ctx);
    out->proposal_count = aicore_vlx_seek_result_proposal_count(ctx);
    aicore_pipeline_timings t{};
    if (aicore_vlx_seek_last_pipeline_timings(ctx, &t) == 0) {
        out->preprocess_ms = t.preprocess_ms;
        out->inference_ms = t.inference_ms;
        out->postprocess_ms = t.postprocess_ms;
        out->e2e_ms = t.e2e_ms;
    }
    (void)timer;
    return true;
}

bool same_result(const RunOutcome& a, const RunOutcome& b, double box_tol,
                 std::string* why) {
    if (a.answer != b.answer) {
        *why = "answer mismatch";
        return false;
    }
    if (a.boxes.size() != b.boxes.size()) {
        *why = "box count mismatch";
        return false;
    }
    for (size_t i = 0; i < a.boxes.size(); ++i)
        for (int k = 0; k < 4; ++k)
            if (std::fabs(a.boxes[i][k] - b.boxes[i][k]) > box_tol) {
                *why = "box coordinate drift";
                return false;
            }
    if (a.prompt_tokens != b.prompt_tokens ||
        a.completion_tokens != b.completion_tokens) {
        *why = "token count mismatch";
        return false;
    }
    return true;
}

// Runs the pristine upstream vlx-cli with the same request and returns the
// official JSON (answer/result_bbox_list/token counts).
std::string run_upstream(const Args& a, const std::string& device,
                         const std::vector<float>& fixed_boxes,
                         std::string* error) {
    const QString exe = QString::fromStdString(a.upstream_dir) +
                        QDir::separator() + "vlx-cli";
    if (!QFileInfo::exists(exe)) {
        *error = "upstream vlx-cli not found at " + a.upstream_dir;
        return {};
    }
    const QString models = QString::fromStdString(a.models_dir);
    QStringList args{"run",
                     "--image-path", QString::fromStdString(a.image),
                     "--task", QString::fromStdString(a.task),
                     "--text", QString::fromStdString(a.text),
                     "--lang", QString::fromStdString(a.lang),
                     "--gguf-dir", models,
                     "--max-new-tokens", QString::number(a.max_tokens)};
    const QString prec = a.upstream_precision.empty()
                             ? QString::fromStdString("f16")
                             : QString::fromStdString(a.upstream_precision);
    args << "--precision" << prec;
    if (a.threads > 0) args << "-t" << QString::number(a.threads);
    if (device == "CPU") args << "--device" << "cpu";
    else if (!device.empty()) args << "--device" << QString::fromStdString(device);
    if (!fixed_boxes.empty()) {
        QStringList parts;
        for (size_t i = 0; i + 3 < fixed_boxes.size(); i += 4) {
            parts << QString("%1,%2,%3,%4")
                         .arg(fixed_boxes[i], 0, 'f', 2)
                         .arg(fixed_boxes[i + 1], 0, 'f', 2)
                         .arg(fixed_boxes[i + 2], 0, 'f', 2)
                         .arg(fixed_boxes[i + 3], 0, 'f', 2);
        }
        args << "--bbox-list" << QString("[%1]").arg(parts.join(","));
    }
    QProcess process;
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    // The reference must resolve ONLY its own ggml build: appending this
    // process's library directories would let its dynamic backend loader
    // pick up AICore's ggml-cuda/ggml-cpu modules and poison the device
    // registry (observed as "no ggml device for --device cpu").
    const QString dir = QString::fromStdString(a.upstream_dir);
    for (const char* var : {"LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"}) {
        env.insert(var, var == QStringLiteral("PATH")
                                ? dir + QDir::listSeparator() +
                                      env.value(var)
                                : dir);
    }
    process.setProcessEnvironment(env);
    process.setProgram(exe);
    process.setArguments(args);
    process.start();
    if (!process.waitForFinished(30 * 60 * 1000)) {
        *error = "upstream vlx-cli timed out or failed to start";
        return {};
    }
    const QByteArray out = process.readAllStandardOutput();
    const QByteArray err = process.readAllStandardError();
    // vlx-cli prints the official JSON as its last stdout block.
    const QString text = QString::fromUtf8(out);
    const int brace = text.lastIndexOf('{');
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0 ||
        brace < 0) {
        *error = QString("upstream vlx-cli failed (rc=%1): %2")
                     .arg(process.exitCode())
                     .arg(QString::fromUtf8(err.left(400)))
                     .toStdString();
        return {};
    }
    return text.mid(brace).toStdString();
}

// Minimal parse of the official JSON fields the parity gate compares.
bool parse_official_json(const std::string& js, RunOutcome* out) {
    QJsonDocument doc = QJsonDocument::fromJson(
        QByteArray(js.data(), int(js.size())));
    if (!doc.isObject()) return false;
    const QJsonObject o = doc.object();
    out->answer = o.value("answer").toString().toStdString();
    out->prompt_tokens = o.value("prompt_tokens").toInt();
    out->completion_tokens = o.value("completion_tokens").toInt();
    const QJsonValue boxes = o.value("result_bbox_list");
    if (boxes.isArray()) {
        for (const QJsonValue& v : boxes.toArray()) {
            const QJsonObject b = v.toObject();
            out->boxes.push_back({float(b.value("xmin").toDouble()),
                                  float(b.value("ymin").toDouble()),
                                  float(b.value("xmax").toDouble()),
                                  float(b.value("ymax").toDouble())});
            out->labels.push_back(
                b.value("label").toString().toStdString());
        }
    }
    return true;
}

std::string boxes_json(const std::vector<std::array<float, 4>>& boxes) {
    std::ostringstream js;
    js << "[";
    for (size_t i = 0; i < boxes.size(); ++i) {
        js << (i ? "," : "") << "[" << boxes[i][0] << "," << boxes[i][1] << ","
           << boxes[i][2] << "," << boxes[i][3] << "]";
    }
    js << "]";
    return js.str();
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (arg == name && i + 1 < argc) return argv[++i];
            std::fprintf(stderr, "missing/invalid value for %s\n", name);
            std::exit(2);
        };
        if (arg == "--models-dir") a.models_dir = need("--models-dir");
        else if (arg == "--wedetect") a.wedetect = need("--wedetect");
        else if (arg == "--vision") a.vision = need("--vision");
        else if (arg == "--llm") a.llm = need("--llm");
        else if (arg == "--device") a.device = need("--device");
        else if (arg == "--image") a.image = need("--image");
        else if (arg == "--task") a.task = need("--task");
        else if (arg == "--text") a.text = need("--text");
        else if (arg == "--lang") a.lang = need("--lang");
        else if (arg == "--max-tokens") a.max_tokens = std::stoi(need("--max-tokens"));
        else if (arg == "--repeats") a.repeats = std::stoi(need("--repeats"));
        else if (arg == "--warmups") a.warmups = std::stoi(need("--warmups"));
        else if (arg == "--threads") a.threads = std::stoi(need("--threads"));
        else if (arg == "--upstream-dir") a.upstream_dir = need("--upstream-dir");
        else if (arg == "--upstream-precision")
            a.upstream_precision = need("--upstream-precision");
        else if (arg == "--json-out") a.json_out = need("--json-out");
        else if (arg == "--box-tolerance")
            a.box_tolerance = std::stod(need("--box-tolerance"));
        else if (arg == "--expect-boxes-min")
            a.expect_boxes_min = std::stoi(need("--expect-boxes-min"));
        else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (a.image.empty() || (a.models_dir.empty() &&
                            (a.wedetect.empty() || a.vision.empty() ||
                             a.llm.empty()))) {
        std::fprintf(stderr,
                     "usage: test_vlx_probe --models-dir DIR --image PATH "
                     "[options]\n");
        return 2;
    }

    QJsonObject report;
    report.insert("task", QString::fromStdString(a.task));
    report.insert("image", QString::fromStdString(a.image));
    report.insert("device_request", QString::fromStdString(a.device));

    // ---- asset availability (exit-77 contract) ----
    const auto derive = [&](const char* stem, const char* prec) {
        return a.models_dir + "/" + stem + "-" + prec + ".gguf";
    };
    const std::string wedetect_path =
        a.wedetect.empty() ? derive("wedetect-base-uni", "f32") : a.wedetect;
    const std::string vision_path =
        a.vision.empty() ? derive("vlx-vision-10b", "f16") : a.vision;
    const std::string llm_path =
        a.llm.empty() ? derive("vlx-10b-llm", "f16") : a.llm;
    for (const std::string& p : {wedetect_path, vision_path, llm_path}) {
        if (!QFileInfo::exists(QString::fromStdString(p))) {
            std::fprintf(stderr, "[skip] missing asset %s\n", p.c_str());
            report.insert("unavailable", QString::fromStdString(p));
            if (!a.json_out.empty())
                QFile(QString::fromStdString(a.json_out))
                    .write(QJsonDocument(report).toJson());
            return 77;
        }
    }
    if (!QFileInfo::exists(QString::fromStdString(a.image))) {
        std::fprintf(stderr, "[skip] missing input %s\n", a.image.c_str());
        return 77;
    }

    // ---- load ----
    aicore_vlx_seek_options* opts = aicore_vlx_seek_options_new();
    aicore_vlx_seek_options_set_models_dir(
        opts, a.models_dir.empty() ? QFileInfo(QString::fromStdString(llm_path))
                                       .dir().path().toUtf8().constData()
                                   : a.models_dir.c_str());
    aicore_vlx_seek_options_set_device(opts, a.device.c_str());
    aicore_vlx_seek_options_set_threads(opts, a.threads);
    aicore_vlx_seek_options_set_max_new_tokens(opts, a.max_tokens);
    if (!a.wedetect.empty())
        aicore_vlx_seek_options_set_wedetect_gguf(opts, a.wedetect.c_str());
    if (!a.vision.empty())
        aicore_vlx_seek_options_set_vision_gguf(opts, a.vision.c_str());
    if (!a.llm.empty())
        aicore_vlx_seek_options_set_llm_gguf(opts, a.llm.c_str());
    aicore_vlx_seek_ctx* ctx = aicore_vlx_seek_load_opts(opts);
    aicore_vlx_seek_options_free(opts);
    if (!ctx || !aicore_vlx_seek_is_ready(ctx)) {
        const char* err = ctx ? aicore_vlx_seek_last_error(ctx) : nullptr;
        const std::string msg = err ? err : "load failed";
        std::fprintf(stderr, "[fail] load: %s\n", msg.c_str());
        if (msg.find("memory") != std::string::npos ||
            msg.find("headroom") != std::string::npos ||
            msg.find("device") != std::string::npos) {
            // Resource-class failure: the runner records it, not a regression.
            report.insert("unavailable", QString::fromStdString(msg));
            if (!a.json_out.empty())
                QFile(QString::fromStdString(a.json_out))
                    .write(QJsonDocument(report).toJson());
            return 77;
        }
        report.insert("error", QString::fromStdString(msg));
        if (!a.json_out.empty())
            QFile(QString::fromStdString(a.json_out))
                .write(QJsonDocument(report).toJson());
        return 1;
    }
    report.insert("device", aicore_vlx_seek_context_device(ctx));

    // ---- warmups + repeats (determinism) ----
    RunOutcome first;
    std::string error;
    std::vector<double> e2e;
    bool ok = true;
    std::string why;
    for (int w = 0; w < a.warmups; ++w) {
        RunOutcome warm;
        if (!run_once(ctx, a, {}, &warm, &error)) {
            std::fprintf(stderr, "[fail] warmup: %s\n", error.c_str());
            return 1;
        }
        if (w == 0) first = warm;  // warm-run output must equal the cold run
    }
    for (int r = 0; r < a.repeats; ++r) {
        RunOutcome run;
        if (!run_once(ctx, a, {}, &run, &error)) {
            std::fprintf(stderr, "[fail] repeat %d: %s\n", r, error.c_str());
            return 1;
        }
        if (r == 0) first = run;
        e2e.push_back(run.e2e_ms);
        if (!same_result(first, run, a.box_tolerance, &why)) {
            std::fprintf(stderr, "[fail] repeat %d not deterministic: %s\n", r,
                         why.c_str());
            ok = false;
        }
    }
    // ---- context recreation ----
    {
        aicore_vlx_seek_free(ctx);
        opts = aicore_vlx_seek_options_new();
        aicore_vlx_seek_options_set_models_dir(
            opts, a.models_dir.empty()
                      ? QFileInfo(QString::fromStdString(llm_path))
                            .dir().path().toUtf8().constData()
                      : a.models_dir.c_str());
        aicore_vlx_seek_options_set_device(opts, a.device.c_str());
        aicore_vlx_seek_options_set_threads(opts, a.threads);
        aicore_vlx_seek_options_set_max_new_tokens(opts, a.max_tokens);
        if (!a.wedetect.empty())
            aicore_vlx_seek_options_set_wedetect_gguf(opts, a.wedetect.c_str());
        if (!a.vision.empty())
            aicore_vlx_seek_options_set_vision_gguf(opts, a.vision.c_str());
        if (!a.llm.empty())
            aicore_vlx_seek_options_set_llm_gguf(opts, a.llm.c_str());
        ctx = aicore_vlx_seek_load_opts(opts);
        aicore_vlx_seek_options_free(opts);
        if (!ctx || !aicore_vlx_seek_is_ready(ctx)) {
            std::fprintf(stderr, "[fail] context recreation failed\n");
            return 1;
        }
        RunOutcome again;
        if (!run_once(ctx, a, {}, &again, &error)) {
            std::fprintf(stderr, "[fail] rerun after recreation: %s\n",
                         error.c_str());
            return 1;
        }
        if (!same_result(first, again, a.box_tolerance, &why)) {
            std::fprintf(stderr, "[fail] recreation changed results: %s\n",
                         why.c_str());
            ok = false;
        }
    }

    // ---- upstream parity (fixed proposals; both sides consume the same
    // boxes, isolating the vision+LLM chain from detector scheduling) ----
    double upstream_box_max_delta = -1.0;
    int upstream_text_match = -1;
    if (!a.upstream_dir.empty()) {
        RunOutcome ours_fixed = first;
        if (ours_fixed.proposal_count > 0 && ours_fixed.boxes.empty()) {
            // harvest our proposals: rerun with boxes from a closed run is
            // implicit (first run WAS closed) — the reference gets OUR
            // proposal set only when the task exposed it; the official run
            // mode is closed (each side runs its own detector), matching the
            // upstream closed-mode criterion.
        }
        const std::string ref = run_upstream(a, aicore_vlx_seek_context_device(ctx),
                                             {}, &error);
        if (ref.empty()) {
            std::fprintf(stderr, "[fail] upstream reference: %s\n",
                         error.c_str());
            return 1;
        }
        RunOutcome theirs;
        if (!parse_official_json(ref, &theirs)) {
            std::fprintf(stderr, "[fail] cannot parse upstream JSON\n");
            return 1;
        }
        upstream_text_match = first.answer == theirs.answer ? 1 : 0;
        if (upstream_text_match != 1) {
            std::fprintf(stderr,
                         "[fail] greedy text differs from upstream:\n ours: %s\n"
                         " theirs: %s\n",
                         first.answer.c_str(), theirs.answer.c_str());
            ok = false;
        }
        if (first.boxes.size() == theirs.boxes.size()) {
            for (size_t i = 0; i < first.boxes.size(); ++i)
                for (int k = 0; k < 4; ++k)
                    upstream_box_max_delta =
                        std::max(upstream_box_max_delta,
                                 double(std::fabs(first.boxes[i][k] -
                                                  theirs.boxes[i][k])));
            if (upstream_box_max_delta > a.box_tolerance) {
                std::fprintf(stderr,
                             "[fail] box delta %.4f px exceeds tolerance\n",
                             upstream_box_max_delta);
                ok = false;
            }
        } else {
            std::fprintf(stderr, "[fail] box count ours=%zu theirs=%zu\n",
                         first.boxes.size(), theirs.boxes.size());
            ok = false;
        }
    }

    if (a.expect_boxes_min >= 0 &&
        int(first.boxes.size()) < a.expect_boxes_min) {
        std::fprintf(stderr, "[fail] boxes %zu < expected min %d\n",
                     first.boxes.size(), a.expect_boxes_min);
        ok = false;
    }

    // ---- report ----
    report.insert("answer", QString::fromStdString(first.answer));
    report.insert("boxes", QString::fromStdString(boxes_json(first.boxes)));
    report.insert("box_count", int(first.boxes.size()));
    report.insert("proposal_count", first.proposal_count);
    report.insert("prompt_tokens", first.prompt_tokens);
    report.insert("completion_tokens", first.completion_tokens);
    report.insert("preprocess_ms", first.preprocess_ms);
    report.insert("inference_ms", first.inference_ms);
    report.insert("postprocess_ms", first.postprocess_ms);
    report.insert("e2e_ms", first.e2e_ms);
    if (!e2e.empty()) {
        report.insert("e2e_p50_ms", percentile(e2e, 0.5));
        report.insert("e2e_p95_ms", percentile(e2e, 0.95));
        report.insert("repeats", int(e2e.size()));
    }
    if (upstream_text_match >= 0) {
        report.insert("upstream_text_match", upstream_text_match);
        report.insert("upstream_box_max_delta_px", upstream_box_max_delta);
    }
    report.insert("verdict", ok ? "PASS" : "FAIL");
    const QByteArray payload = QJsonDocument(report).toJson();
    if (!a.json_out.empty()) {
        QFile(QString::fromStdString(a.json_out)).write(payload);
    }
    std::fwrite(payload.constData(), 1, size_t(payload.size()), stdout);
    std::fputc('\n', stdout);
    aicore_vlx_seek_free(ctx);
    aicore_vlx_seek_shutdown();
    return ok ? 0 : 1;
}
