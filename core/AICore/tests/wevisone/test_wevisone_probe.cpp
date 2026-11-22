// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Real-model probe for the WeVisOne-VL unified perception task. One process
// loads the three-model set (VLM + mmproj + pixel decoder), runs warmups plus
// N greedy repeats, then checks the lifecycle invariants a live UI depends
// on: determinism (answer, boxes AND the first matte), context recreation,
// the run_batch == per-image run_image contract (with --image2) and — with
// --upstream-dir — greedy-answer / box / matte parity against the pristine
// upstream wevisone-cli built from the pinned Asher-1/WeVisOne-VL-GGML
// revision (tests/wevisone/prepare_wevisone_upstream_reference.py).
// Upstream parity is coordinate-aware for generation rows: labels, OCR,
// structure and coordinate COUNT must match exactly, while integer
// coordinate bins may drift within --coord-bin-tolerance (cross-ggml fp32
// reduction noise flips near-tie argmax). GT-conditioned (--response) rows
// keep exact answer parity and the strict matte band; same-implementation
// determinism always uses exact equality.
// --response + --prompt [+ --gt-boxes] drives the GT-conditioned
// run_decode_masks path and mirrors it through upstream --response.
// Emits one JSON object (stdout or --json-out).
//
// Exit codes: 0 pass, 1 invariant/accuracy failure, 2 usage, 77 missing
// model asset (the common runner records it as unavailable).

#include <QCryptographicHash>
#include <QDir>
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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/runtime_capi.h"
#include "aicore/wevisone_capi.h"

namespace {

struct Args {
    std::string models_dir, vlm, mmproj, decoder,
            size = "2b", precision = "f16", decoder_precision = "f32",
            device = "auto", image, image2, task = "detection",
            query = "person", ref_boxes, prompt, response, gt_boxes,
            upstream_dir, json_out, dump_dir;
    int max_tokens = 2048, repeats = 2, warmups = 0, threads = 0,
        decode_masks = 1;
    // Test-only diagnostic: -1 keeps the task default (INFO). 0 (DEBUG)
    // exposes llama reserve/ubatch traces for buffer-size root-causing.
    int log_level = -1;
    double box_tolerance = 0.05;    // px; f32 reduction noise class
    double matte_tolerance = 0.05;  // alpha in [0,1]; decoder noise class
    // Cross-implementation coordinate bin tolerance for the GENERATION
    // upstream-parity gate only. Distributed fp32 reduction noise across
    // ggml versions can flip a near-tie greedy argmax by one coordinate
    // bin (0-1000 grid); labels/OCR/structure/count must stay identical.
    // GT-conditioned (--response) rows force this to 0 (exact parity).
    int coord_bin_tolerance = 2;
    int expect_boxes_min = -1;   // optional accuracy gate
    int expect_mattes_min = -1;  // optional accuracy gate (mask tasks)
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t i = size_t(std::ceil(p * double(v.size()))) - 1;
    return v[std::min(i, v.size() - 1)];
}

// Exactly one sink: the runner merges stdout JSON lines and the report
// file, and a metric present in both would resolve as ambiguous. QFile
// must be opened before write(); a default-constructed QFile::write() is
// a silent no-op.
void emit_report(const std::string& path, const QByteArray& payload) {
    if (!path.empty()) {
        QFile file(QString::fromStdString(path));
        if (file.open(QIODevice::WriteOnly)) {
            file.write(payload);
            return;
        }
        std::fprintf(stderr, "[warn] cannot write --json-out %s: %s\n",
                     path.c_str(), file.errorString().toUtf8().constData());
    }
    std::fwrite(payload.constData(), 1, size_t(payload.size()), stdout);
    std::fputc('\n', stdout);
}

// Upstream parse_ref_boxes semantics: ';' separates boxes, ',' separates
// the four coordinates, exactly four floats per box, empty spec = no boxes.
bool parse_boxes(const std::string& spec, std::vector<float>* out) {
    if (spec.empty()) return true;
    size_t start = 0;
    while (start <= spec.size()) {
        const size_t semi = spec.find(';', start);
        const std::string box =
                spec.substr(start, semi == std::string::npos ? std::string::npos
                                                             : semi - start);
        float v[4] = {0, 0, 0, 0};
        int got = 0;
        size_t bs = 0;
        while (got < 4 && bs <= box.size()) {
            const size_t comma = box.find(',', bs);
            const std::string tok = box.substr(bs, comma == std::string::npos
                                                           ? std::string::npos
                                                           : comma - bs);
            if (!tok.empty()) v[got++] = std::strtof(tok.c_str(), nullptr);
            if (comma == std::string::npos) break;
            bs = comma + 1;
        }
        if (got != 4) return false;
        out->insert(out->end(), v, v + 4);
        if (semi == std::string::npos) break;
        start = semi + 1;
    }
    return !out->empty();
}

struct RunOutcome {
    int rc = 0;
    std::string answer;
    std::vector<std::array<float, 4>> boxes;  // original-image pixels
    int has_masks = 0;
    int32_t matte_count = 0, matte_w = 0, matte_h = 0;
    std::vector<float> matte;  // first matte copy (upstream RunResult keeps 0)
    int32_t prompt_tokens = 0, completion_tokens = 0;
    double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0, e2e_ms = 0;
};

struct ImageBuf {
    std::vector<uint8_t> rgb;
    aicore_image_view view{};
    bool load(const std::string& path, std::string* error) {
        uint8_t* p = nullptr;
        int32_t w = 0, h = 0;
        if (aicore_wevisone_load_path_rgb(path.c_str(), &p, &w, &h) != 0 ||
            !p) {
            *error = "cannot load image " + path;
            return false;
        }
        rgb.assign(p, p + size_t(w) * h * 3);
        aicore_wevisone_free_buffer(p);
        view.data = rgb.data();
        view.width = w;
        view.height = h;
        view.row_stride_bytes = size_t(w) * 3;
        view.format = AICORE_IMAGE_RGB8;
        return true;
    }
};

// Collects result `index` of the most recent run/run_batch through the
// indexed getters (run_image holds exactly one result at index 0).
void collect_result(aicore_wevisone_ctx* ctx, int32_t index, RunOutcome* out) {
    const char* answer = aicore_wevisone_result_answer_at(ctx, index);
    out->answer = answer ? answer : "";
    const int32_t n = aicore_wevisone_result_box_count_at(ctx, index);
    for (int32_t i = 0; i < n; ++i) {
        std::array<float, 4> b{};
        aicore_wevisone_result_box_at(ctx, index, i, b.data());
        out->boxes.push_back(b);
    }
    out->has_masks = aicore_wevisone_result_has_masks_at(ctx, index);
    out->matte_count = aicore_wevisone_result_matte_count_at(ctx, index);
    if (out->matte_count > 0 &&
        aicore_wevisone_result_matte_size_at(ctx, index, 0, &out->matte_w,
                                             &out->matte_h) == 0) {
        const float* m = aicore_wevisone_result_matte_data_at(ctx, index, 0);
        if (m && out->matte_w > 0 && out->matte_h > 0)
            out->matte.assign(m, m + size_t(out->matte_w) * out->matte_h);
    }
    // Per-index token counts: the non-indexed getters surface the LAST
    // batch entry, so index 0 of a batch must read its own JSON payload.
    char* js = aicore_wevisone_result_json_at(ctx, index);
    if (js) {
        const QJsonObject o = QJsonDocument::fromJson(QByteArray(js)).object();
        out->prompt_tokens = o.value("prompt_tokens").toInt();
        out->completion_tokens = o.value("completion_tokens").toInt();
        aicore_wevisone_free_buffer(js);
    }
    aicore_pipeline_timings t{};
    if (aicore_wevisone_last_pipeline_timings(ctx, &t) == 0) {
        out->preprocess_ms = t.preprocess_ms;
        out->inference_ms = t.inference_ms;
        out->postprocess_ms = t.postprocess_ms;
        out->e2e_ms = t.e2e_ms;
    }
}

bool run_once(aicore_wevisone_ctx* ctx,
              const Args& a,
              const aicore_image_view* view,
              const std::vector<float>& ref_boxes,
              const std::vector<float>& gt_boxes,
              RunOutcome* out,
              std::string* error) {
    int rc;
    if (!a.response.empty()) {
        // GT-conditioned decoder pass: teacher-forced prompt + response,
        // no generation (the evaluation box_source=gt protocol).
        rc = aicore_wevisone_run_decode_masks(
                ctx, view, a.prompt.c_str(), a.response.c_str(),
                gt_boxes.empty() ? nullptr : gt_boxes.data(),
                int32_t(gt_boxes.size() / 4));
    } else {
        rc = aicore_wevisone_run_image(
                ctx, view, a.task.c_str(), a.query.c_str(),
                ref_boxes.empty() ? nullptr : ref_boxes.data(),
                int32_t(ref_boxes.size() / 4));
    }
    out->rc = rc;
    if (rc != 0) {
        const char* err = aicore_wevisone_last_error(ctx);
        *error = err ? err : "run failed";
        return false;
    }
    collect_result(ctx, 0, out);
    return true;
}

double matte_max_delta(const RunOutcome& a, const RunOutcome& b) {
    if (a.matte.size() != b.matte.size()) return 1.0;
    double d = 0;
    for (size_t i = 0; i < a.matte.size(); ++i)
        d = std::max(d, double(std::fabs(a.matte[i] - b.matte[i])));
    return d;
}

bool same_result(const RunOutcome& a,
                 const RunOutcome& b,
                 double box_tol,
                 double matte_tol,
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
    if (a.has_masks != b.has_masks || a.matte_count != b.matte_count ||
        a.matte_w != b.matte_w || a.matte_h != b.matte_h) {
        *why = "matte shape/count mismatch";
        return false;
    }
    if (matte_max_delta(a, b) > matte_tol) {
        *why = "matte drift";
        return false;
    }
    return true;
}

// Coordinate-aware answer equivalence for CROSS-IMPLEMENTATION upstream
// parity. The greedy answer is a stream of literal text (labels, OCR,
// structure tags such as <ref>/<box>/<point>/<OBJ>, punctuation) and
// integer coordinate tokens written as <NNN> on a 0-1000 grid. Two ggml
// builds can flip a near-tie coordinate argmax by a bin while emitting
// identical literals; that is reduction noise, not a semantic divergence.
// This requires the literal skeleton AND the coordinate count/positions to
// match exactly, and every coordinate to agree within bin_tol. It returns
// the max |bin delta| (0 when byte-identical). Same-implementation
// determinism still uses exact equality (same_result), never this.
bool answers_equivalent(const std::string& a,
                        const std::string& b,
                        int bin_tol,
                        int* max_bin_delta,
                        int* skeleton_match = nullptr) {
    *max_bin_delta = 0;
    // Split into a literal skeleton (coordinate tokens replaced by a
    // single marker) plus the ordered coordinate values. A coordinate
    // token is '<' digits-only '>'; everything else is literal. The digit
    // run is capped so a pathological long number cannot overflow stoi.
    auto tokenize = [](const std::string& s, std::string* skeleton,
                       std::vector<int>* coords) {
        size_t i = 0;
        while (i < s.size()) {
            if (s[i] == '<') {
                const size_t close = s.find('>', i + 1);
                if (close != std::string::npos) {
                    const std::string inner = s.substr(i + 1, close - i - 1);
                    bool coord = !inner.empty() && inner.size() <= 9;
                    for (char c : inner)
                        if (!std::isdigit(static_cast<unsigned char>(c))) {
                            coord = false;
                            break;
                        }
                    if (coord) {
                        coords->push_back(std::stoi(inner));
                        skeleton->push_back('\x01');
                        i = close + 1;
                        continue;
                    }
                }
            }
            skeleton->push_back(s[i]);
            ++i;
        }
    };
    std::string sa, sb;
    std::vector<int> ca, cb;
    tokenize(a, &sa, &ca);
    tokenize(b, &sb, &cb);
    // Skeleton match isolates the literal structure (labels, OCR text, tag
    // sequence, coordinate count) from the coordinate values. It separates a
    // pure coordinate jitter (skeleton identical, coords drift) from a
    // semantic divergence (different label/text/count/object).
    const bool skel = (sa == sb && ca.size() == cb.size());
    if (skeleton_match != nullptr) *skeleton_match = skel ? 1 : 0;
    if (!skel) return false;
    // Walk EVERY coordinate so max_bin_delta reports the true worst drift
    // even when the gate fails; an early return would truncate the metric at
    // the first out-of-tolerance coordinate and understate the divergence.
    bool equiv = true;
    for (size_t k = 0; k < ca.size(); ++k) {
        const int d = std::abs(ca[k] - cb[k]);
        *max_bin_delta = std::max(*max_bin_delta, d);
        if (d > bin_tol) equiv = false;
    }
    return equiv;
}

// The upstream CLI validates --device against exactly cpu|cuda|vulkan;
// the lease-resolved device is the ggml device name (CUDA0 / Vulkan0 /
// CPU) and may instead carry a public-index suffix (cuda:1).
std::string upstream_device(const std::string& resolved) {
    std::string dev = resolved.substr(0, resolved.find(':'));
    std::transform(dev.begin(), dev.end(), dev.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    while (!dev.empty() &&
           std::isdigit(static_cast<unsigned char>(dev.back()))) {
        dev.pop_back();
    }
    return dev;
}

void append_runtime_opts(QStringList* args,
                         const Args& a,
                         const std::string& vlm,
                         const std::string& mmproj,
                         const std::string& decoder,
                         const std::string& device) {
    *args << "--vlm-gguf" << QString::fromStdString(vlm) << "--mmproj-gguf"
          << QString::fromStdString(mmproj) << "--decoder-gguf"
          << QString::fromStdString(decoder);
    if (!a.models_dir.empty())
        *args << "--models-dir" << QString::fromStdString(a.models_dir);
    *args << "--device" << QString::fromStdString(upstream_device(device))
          << "--max-new-tokens" << QString::number(a.max_tokens);
    if (a.threads > 0) *args << "--threads" << QString::number(a.threads);
}

// Runs the pristine upstream wevisone-cli with an isolated library path
// (appending this process's directories would let its dynamic backend
// loader pick up AICore's ggml modules and poison the device registry —
// the vlx probe's observed failure mode).
bool run_upstream_exe(const Args& a,
                      const QStringList& args,
                      QByteArray* err_out,
                      std::string* error) {
    const QString exe = QString::fromStdString(a.upstream_dir) +
                        QDir::separator() + "wevisone-cli";
    if (!QFileInfo::exists(exe)) {
        *error = "upstream wevisone-cli not found at " + a.upstream_dir;
        return false;
    }
    QProcess process;
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    const QString dir = QString::fromStdString(a.upstream_dir);
    for (const char* var : {"LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"}) {
        env.insert(var, var == QStringLiteral("PATH")
                                ? dir + QDir::listSeparator() + env.value(var)
                                : dir);
    }
    process.setProcessEnvironment(env);
    process.setProgram(exe);
    process.setArguments(args);
    process.start();
    if (!process.waitForFinished(60 * 60 * 1000)) {
        *error = "upstream wevisone-cli timed out or failed to start";
        return false;
    }
    *err_out = process.readAllStandardError();
    if (process.exitStatus() != QProcess::NormalExit ||
        process.exitCode() != 0) {
        // The crash reason lives at the TAIL of stderr; the head is only the
        // ggml_cuda_init banner. Capture the tail so the real failure (OOM,
        // assert, segfault backtrace) is reported, not the device banner.
        const QByteArray tail = err_out->right(800);
        *error = QString("upstream wevisone-cli failed (rc=%1, %2): %3")
                         .arg(process.exitCode())
                         .arg(process.exitStatus() == QProcess::NormalExit
                                      ? QStringLiteral("exit")
                                      : QStringLiteral("crash"))
                         .arg(QString::fromUtf8(tail))
                         .toStdString();
        return false;
    }
    return true;
}

// Minimal parse of the official run --json payload (answer, box_count,
// NORMALIZED box_coords, mask_count). Boxes are scaled back to pixels so
// both sides compare in the same space; std::to_string's 6-decimal
// serialization quantizes <= 0.5e-6 * max(w,h) px, inside box_tolerance.
bool parse_official_json(const QByteArray& js,
                         int32_t w,
                         int32_t h,
                         RunOutcome* out) {
    QJsonDocument doc = QJsonDocument::fromJson(js);
    if (!doc.isObject()) return false;
    const QJsonObject o = doc.object();
    out->answer = o.value("answer").toString().toStdString();
    out->has_masks = o.value("mask_count").toInt() > 0 ? 1 : 0;
    const QJsonValue boxes = o.value("box_coords");
    if (boxes.isArray()) {
        for (const QJsonValue& v : boxes.toArray()) {
            const QJsonArray b = v.toArray();
            if (b.size() != 4) return false;
            out->boxes.push_back(
                    {float(b[0].toDouble() * w), float(b[1].toDouble() * h),
                     float(b[2].toDouble() * w), float(b[3].toDouble() * h)});
        }
    }
    return true;
}

bool read_raw_matte(const std::string& path,
                    int32_t* w,
                    int32_t* h,
                    std::vector<float>* out) {
    QFile shape(QString::fromStdString(path + ".shape"));
    if (!shape.open(QIODevice::ReadOnly)) return false;
    const QStringList dims =
            QString::fromUtf8(shape.readAll().trimmed()).split(',');
    if (dims.size() != 2) return false;
    *h = dims[0].toInt();
    *w = dims[1].toInt();
    QFile raw(QString::fromStdString(path));
    if (!raw.open(QIODevice::ReadOnly)) return false;
    const QByteArray bytes = raw.readAll();
    if (*w <= 0 || *h <= 0 ||
        bytes.size() != qint64(*w) * *h * qint64(sizeof(float)))
        return false;
    out->resize(size_t(*w) * *h);
    std::memcpy(out->data(), bytes.constData(), size_t(bytes.size()));
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
        if (arg == "--models-dir")
            a.models_dir = need("--models-dir");
        else if (arg == "--vlm")
            a.vlm = need("--vlm");
        else if (arg == "--mmproj")
            a.mmproj = need("--mmproj");
        else if (arg == "--decoder")
            a.decoder = need("--decoder");
        else if (arg == "--size")
            a.size = need("--size");
        else if (arg == "--precision")
            a.precision = need("--precision");
        else if (arg == "--decoder-precision")
            a.decoder_precision = need("--decoder-precision");
        else if (arg == "--device")
            a.device = need("--device");
        else if (arg == "--image")
            a.image = need("--image");
        else if (arg == "--image2")
            a.image2 = need("--image2");
        else if (arg == "--task")
            a.task = need("--task");
        else if (arg == "--query")
            a.query = need("--query");
        else if (arg == "--ref-boxes")
            a.ref_boxes = need("--ref-boxes");
        else if (arg == "--prompt")
            a.prompt = need("--prompt");
        else if (arg == "--response")
            a.response = need("--response");
        else if (arg == "--gt-boxes")
            a.gt_boxes = need("--gt-boxes");
        else if (arg == "--max-tokens")
            a.max_tokens = std::stoi(need("--max-tokens"));
        else if (arg == "--repeats")
            a.repeats = std::stoi(need("--repeats"));
        else if (arg == "--warmups")
            a.warmups = std::stoi(need("--warmups"));
        else if (arg == "--threads")
            a.threads = std::stoi(need("--threads"));
        else if (arg == "--decode-masks")
            a.decode_masks = std::stoi(need("--decode-masks"));
        else if (arg == "--log-level")
            a.log_level = std::stoi(need("--log-level"));
        else if (arg == "--upstream-dir")
            a.upstream_dir = need("--upstream-dir");
        else if (arg == "--json-out")
            a.json_out = need("--json-out");
        else if (arg == "--dump-dir")
            a.dump_dir = need("--dump-dir");
        else if (arg == "--box-tolerance")
            a.box_tolerance = std::stod(need("--box-tolerance"));
        else if (arg == "--matte-tolerance")
            a.matte_tolerance = std::stod(need("--matte-tolerance"));
        else if (arg == "--coord-bin-tolerance")
            a.coord_bin_tolerance = std::stoi(need("--coord-bin-tolerance"));
        else if (arg == "--expect-boxes-min")
            a.expect_boxes_min = std::stoi(need("--expect-boxes-min"));
        else if (arg == "--expect-mattes-min")
            a.expect_mattes_min = std::stoi(need("--expect-mattes-min"));
        else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (a.image.empty() ||
        (a.models_dir.empty() &&
         (a.vlm.empty() || a.mmproj.empty() || a.decoder.empty()))) {
        std::fprintf(stderr,
                     "usage: test_wevisone_probe --models-dir DIR --image PATH "
                     "[options]\n");
        return 2;
    }
    if (!a.response.empty() && a.prompt.empty()) {
        // The upstream --response path without --prompt rebuilds the task
        // template internally; the probe mirrors it with an explicit raw
        // prompt handed to BOTH sides, keeping the comparison exact.
        std::fprintf(stderr, "--response requires --prompt (raw template)\n");
        return 2;
    }
    if (!a.response.empty() && !a.image2.empty()) {
        std::fprintf(stderr, "--image2 (batch) is a generation-path check\n");
        return 2;
    }
    std::vector<float> ref_boxes, gt_boxes;
    if (!parse_boxes(a.ref_boxes, &ref_boxes) ||
        !parse_boxes(a.gt_boxes, &gt_boxes)) {
        std::fprintf(stderr, "malformed --ref-boxes/--gt-boxes spec\n");
        return 2;
    }

    QJsonObject report;
    report.insert("task", QString::fromStdString(a.task));
    report.insert("image", QString::fromStdString(a.image));
    report.insert("size", QString::fromStdString(a.size));
    report.insert("precision", QString::fromStdString(a.precision));
    report.insert("decoder_precision",
                  QString::fromStdString(a.decoder_precision));
    report.insert("device_request", QString::fromStdString(a.device));

    // ---- asset availability (exit-77 contract) ----
    const auto derive = [&](const std::string& name) {
        return a.models_dir + "/" + name;
    };
    const std::string vlm_path = a.vlm.empty()
                                         ? derive("wevisone-vlm-" + a.size +
                                                  "-" + a.precision + ".gguf")
                                         : a.vlm;
    const std::string mmproj_path =
            a.mmproj.empty()
                    ? derive("mmproj-wevisone-vlm-" + a.size + "-f16.gguf")
                    : a.mmproj;
    const std::string decoder_path =
            a.decoder.empty() ? derive("wevisone-mask-decoder-" + a.size + "-" +
                                       a.decoder_precision + ".gguf")
                              : a.decoder;
    for (const std::string& p : {vlm_path, mmproj_path, decoder_path}) {
        if (!QFileInfo::exists(QString::fromStdString(p))) {
            std::fprintf(stderr, "[skip] missing asset %s\n", p.c_str());
            report.insert("unavailable", QString::fromStdString(p));
            emit_report(a.json_out, QJsonDocument(report).toJson());
            return 77;
        }
    }
    for (const std::string& p : {a.image, a.image2}) {
        if (!p.empty() && !QFileInfo::exists(QString::fromStdString(p))) {
            std::fprintf(stderr, "[skip] missing input %s\n", p.c_str());
            return 77;
        }
    }

    // ---- load ----
    const auto make_opts = [&]() {
        aicore_wevisone_options* opts = aicore_wevisone_options_new();
        aicore_wevisone_options_set_models_dir(
                opts, a.models_dir.empty()
                              ? QFileInfo(QString::fromStdString(vlm_path))
                                        .dir()
                                        .path()
                                        .toUtf8()
                                        .constData()
                              : a.models_dir.c_str());
        aicore_wevisone_options_set_size(opts, a.size.c_str());
        aicore_wevisone_options_set_precision(opts, a.precision.c_str());
        aicore_wevisone_options_set_decoder_precision(
                opts, a.decoder_precision.c_str());
        aicore_wevisone_options_set_device(opts, a.device.c_str());
        aicore_wevisone_options_set_threads(opts, a.threads);
        aicore_wevisone_options_set_max_new_tokens(opts, a.max_tokens);
        aicore_wevisone_options_set_decode_masks(opts, a.decode_masks);
        aicore_wevisone_options_set_vlm_gguf(opts, vlm_path.c_str());
        aicore_wevisone_options_set_mmproj_gguf(opts, mmproj_path.c_str());
        aicore_wevisone_options_set_decoder_gguf(opts, decoder_path.c_str());
        if (!a.dump_dir.empty())
            aicore_wevisone_options_set_dump_dir(opts, a.dump_dir.c_str());
        if (a.log_level >= 0)
            aicore_wevisone_options_set_log_level(opts, a.log_level);
        return opts;
    };
    aicore_wevisone_options* opts = make_opts();
    aicore_wevisone_ctx* ctx = aicore_wevisone_load_opts(opts);
    aicore_wevisone_options_free(opts);
    if (!ctx || !aicore_wevisone_is_ready(ctx)) {
        const char* err = ctx ? aicore_wevisone_last_error(ctx) : nullptr;
        const std::string msg = err ? err : "load failed";
        std::fprintf(stderr, "[fail] load: %s\n", msg.c_str());
        if (msg.find("memory") != std::string::npos ||
            msg.find("headroom") != std::string::npos ||
            msg.find("device") != std::string::npos) {
            // Resource-class failure: the runner records it, not a regression.
            report.insert("unavailable", QString::fromStdString(msg));
            emit_report(a.json_out, QJsonDocument(report).toJson());
            return 77;
        }
        report.insert("error", QString::fromStdString(msg));
        emit_report(a.json_out, QJsonDocument(report).toJson());
        return 1;
    }
    const std::string resolved_device =
            aicore_wevisone_context_device(ctx)
                    ? aicore_wevisone_context_device(ctx)
                    : "";
    report.insert("device", QString::fromStdString(resolved_device));

    ImageBuf img1;
    std::string error;
    if (!img1.load(a.image, &error)) {
        std::fprintf(stderr, "[fail] %s\n", error.c_str());
        return 1;
    }

    // ---- warmups + repeats (determinism incl. the matte plane) ----
    RunOutcome first;
    std::vector<double> e2e;
    bool ok = true;
    std::string why;
    for (int w = 0; w < a.warmups; ++w) {
        RunOutcome warm;
        if (!run_once(ctx, a, &img1.view, ref_boxes, gt_boxes, &warm, &error)) {
            std::fprintf(stderr, "[fail] warmup: %s\n", error.c_str());
            return 1;
        }
        if (w == 0) first = warm;  // warm-run output must equal the cold run
    }
    for (int r = 0; r < a.repeats; ++r) {
        RunOutcome run;
        if (!run_once(ctx, a, &img1.view, ref_boxes, gt_boxes, &run, &error)) {
            std::fprintf(stderr, "[fail] repeat %d: %s\n", r, error.c_str());
            return 1;
        }
        if (r == 0) first = run;
        e2e.push_back(run.e2e_ms);
        if (!same_result(first, run, a.box_tolerance, a.matte_tolerance,
                         &why)) {
            std::fprintf(stderr, "[fail] repeat %d not deterministic: %s\n", r,
                         why.c_str());
            ok = false;
        }
    }
    // ---- context recreation ----
    {
        aicore_wevisone_free(ctx);
        opts = make_opts();
        ctx = aicore_wevisone_load_opts(opts);
        aicore_wevisone_options_free(opts);
        if (!ctx || !aicore_wevisone_is_ready(ctx)) {
            std::fprintf(stderr, "[fail] context recreation failed\n");
            return 1;
        }
        RunOutcome again;
        if (!run_once(ctx, a, &img1.view, ref_boxes, gt_boxes, &again,
                      &error)) {
            std::fprintf(stderr, "[fail] rerun after recreation: %s\n",
                         error.c_str());
            return 1;
        }
        if (!same_result(first, again, a.box_tolerance, a.matte_tolerance,
                         &why)) {
            std::fprintf(stderr, "[fail] recreation changed results: %s\n",
                         why.c_str());
            ok = false;
        }
    }

    // ---- run_batch contract (one load, N images, per-image results) ----
    int batch_index0_match = -1, batch_index1_match = -1;
    // CPU-side copies of the batch results, hoisted so the upstream batch
    // parity check below can reuse them after the GPU context is freed.
    RunOutcome batch_b0, batch_b1;
    if (!a.image2.empty()) {
        ImageBuf img2;
        if (!img2.load(a.image2, &error)) {
            std::fprintf(stderr, "[fail] %s\n", error.c_str());
            return 1;
        }
        RunOutcome single2;
        if (!run_once(ctx, a, &img2.view, ref_boxes, gt_boxes, &single2,
                      &error)) {
            std::fprintf(stderr, "[fail] image2 single run: %s\n",
                         error.c_str());
            return 1;
        }
        const aicore_image_view views[2] = {img1.view, img2.view};
        const int rc = aicore_wevisone_run_batch(
                ctx, views, 2, a.task.c_str(), a.query.c_str(),
                ref_boxes.empty() ? nullptr : ref_boxes.data(),
                int32_t(ref_boxes.size() / 4));
        if (rc != 0 || aicore_wevisone_result_count(ctx) != 2) {
            std::fprintf(stderr, "[fail] run_batch rc=%d count=%d\n", rc,
                         aicore_wevisone_result_count(ctx));
            return 1;
        }
        RunOutcome b0, b1;
        collect_result(ctx, 0, &b0);
        collect_result(ctx, 1, &b1);
        batch_b0 = b0;
        batch_b1 = b1;
        batch_index0_match =
                same_result(first, b0, a.box_tolerance, a.matte_tolerance, &why)
                        ? 1
                        : 0;
        batch_index1_match = same_result(single2, b1, a.box_tolerance,
                                         a.matte_tolerance, &why)
                                     ? 1
                                     : 0;
        if (batch_index0_match != 1 || batch_index1_match != 1) {
            std::fprintf(stderr,
                         "[fail] batch != per-image runs (idx0=%d idx1=%d)\n",
                         batch_index0_match, batch_index1_match);
            ok = false;
        }
    }

    // ---- upstream parity (same gguf bytes on both sides) ----
    int upstream_text_match = -1, batch_text_match = -1;
    int upstream_coord_bin_delta = 0, batch_coord_bin_delta = 0;
    int upstream_skeleton_match = -1, batch_skeleton_match = -1;
    double upstream_box_max_delta = -1.0, upstream_matte_max_delta = -1.0,
           upstream_matte_mean_delta = -1.0, batch_box_max_delta = -1.0;
    // Generation rows tolerate a coordinate-bin flip from cross-ggml
    // reduction noise; GT-conditioned (--response) rows demand exact parity.
    const int eff_bin_tol = a.response.empty() ? a.coord_bin_tolerance : 0;
    // A coordinate bin is 1/1000 of the image side; the pixel box gate must
    // admit the same drift the coordinate gate admits (plus f32 noise). The
    // 1e-3 px slack absorbs float rounding of the bin->pixel conversion so a
    // coordinate exactly at the bin boundary is not failed by fp epsilon.
    const double bin_px_band =
            eff_bin_tol / 1000.0 *
            double(std::max(img1.view.width, img1.view.height));
    const double parity_box_tol = std::max(
            a.box_tolerance, bin_px_band + (eff_bin_tol > 0 ? 1e-3 : 0.0));
    // Release our GPU context before spawning the upstream reference. Both
    // sides load the full VLM + mmproj + decoder; on a 24 GB card two 4b-f16
    // contexts (~10 GB each) plus KV/compute buffers exhaust VRAM and the
    // upstream decoder graph alloc fails with cudaMalloc OOM (observed peak
    // 23146/24564 MiB). All comparison data (first, batch_b0/b1) is already
    // collected CPU-side, so the context is not needed while upstream runs.
    if (!a.upstream_dir.empty() && ctx != nullptr) {
        aicore_wevisone_free(ctx);
        ctx = nullptr;
    }
    if (!a.upstream_dir.empty()) {
        QTemporaryDir tmp;
        if (!tmp.isValid()) {
            std::fprintf(stderr, "[fail] cannot create temp dir\n");
            return 1;
        }
        const std::string json_path =
                tmp.filePath("upstream.json").toStdString();
        const std::string matte_path =
                tmp.filePath("upstream.matte").toStdString();
        QStringList args{"run",
                         "--image",
                         QString::fromStdString(a.image),
                         "--task",
                         QString::fromStdString(a.task),
                         "--query",
                         QString::fromStdString(a.query)};
        append_runtime_opts(&args, a, vlm_path, mmproj_path, decoder_path,
                            resolved_device);
        if (!ref_boxes.empty())
            args << "--ref-boxes" << QString::fromStdString(a.ref_boxes);
        if (!a.response.empty()) {
            args << "--prompt" << QString::fromStdString(a.prompt)
                 << "--response" << QString::fromStdString(a.response);
            if (!gt_boxes.empty())
                args << "--gt-boxes" << QString::fromStdString(a.gt_boxes);
        }
        args << "--json" << QString::fromStdString(json_path) << "--matte-out"
             << QString::fromStdString(matte_path);
        QByteArray err_out;
        if (!run_upstream_exe(a, args, &err_out, &error)) {
            std::fprintf(stderr, "[fail] upstream reference: %s\n",
                         error.c_str());
            return 1;
        }
        QFile jf(QString::fromStdString(json_path));
        if (!jf.open(QIODevice::ReadOnly)) {
            std::fprintf(stderr, "[fail] upstream wrote no --json payload\n");
            return 1;
        }
        RunOutcome theirs;
        if (!parse_official_json(jf.readAll(), img1.view.width,
                                 img1.view.height, &theirs)) {
            std::fprintf(stderr, "[fail] cannot parse upstream JSON\n");
            return 1;
        }
        upstream_text_match =
                answers_equivalent(first.answer, theirs.answer, eff_bin_tol,
                                   &upstream_coord_bin_delta,
                                   &upstream_skeleton_match)
                        ? 1
                        : 0;
        if (upstream_text_match != 1) {
            std::fprintf(stderr,
                         "[fail] greedy answer diverges from upstream beyond "
                         "coord-bin tolerance %d:\n ours: %s\n theirs: %s\n",
                         eff_bin_tol, first.answer.c_str(),
                         theirs.answer.c_str());
            ok = false;
        }
        if (first.boxes.size() == theirs.boxes.size()) {
            for (size_t i = 0; i < first.boxes.size(); ++i)
                for (int k = 0; k < 4; ++k)
                    upstream_box_max_delta =
                            std::max(upstream_box_max_delta,
                                     double(std::fabs(first.boxes[i][k] -
                                                      theirs.boxes[i][k])));
            if (upstream_box_max_delta > parity_box_tol) {
                std::fprintf(stderr,
                             "[fail] box delta %.4f px exceeds tolerance "
                             "%.4f\n",
                             upstream_box_max_delta, parity_box_tol);
                ok = false;
            }
        } else {
            std::fprintf(stderr, "[fail] box count ours=%zu theirs=%zu\n",
                         first.boxes.size(), theirs.boxes.size());
            ok = false;
        }
        // The upstream CLI has no decode-masks toggle (it always decodes);
        // mask/matte parity is only comparable when this side decodes too.
        if (a.decode_masks) {
            if (first.has_masks != theirs.has_masks) {
                std::fprintf(stderr, "[fail] mask presence ours=%d theirs=%d\n",
                             first.has_masks, theirs.has_masks);
                ok = false;
            }
            if (!first.matte.empty()) {
                int32_t tw = 0, th = 0;
                std::vector<float> tm;
                if (!read_raw_matte(matte_path, &tw, &th, &tm)) {
                    std::fprintf(stderr,
                                 "[fail] upstream wrote no matte dump\n");
                    ok = false;
                } else if (tw != first.matte_w || th != first.matte_h ||
                           tm.size() != first.matte.size()) {
                    std::fprintf(stderr,
                                 "[fail] matte shape ours=%dx%d theirs=%dx%d\n",
                                 first.matte_w, first.matte_h, tw, th);
                    ok = false;
                } else {
                    double sum = 0;
                    for (size_t i = 0; i < tm.size(); ++i) {
                        const double d = std::fabs(first.matte[i] - tm[i]);
                        upstream_matte_max_delta =
                                std::max(upstream_matte_max_delta, d);
                        sum += d;
                    }
                    upstream_matte_mean_delta = sum / double(tm.size());
                    // The numeric matte band is a decoder-precision gate and
                    // is only meaningful on the GT-isolated rows: they
                    // teacher-force the identical prompt/response AND use the
                    // f16 VLM, so the matte isolates the pixel decoder. On
                    // generation rows the matte also depends on the quantized
                    // VLM features feeding the decoder, which legitimately
                    // differ across quantizations even when the box matches;
                    // there the matte is gated structurally (presence/shape/
                    // count above), and a coordinate flip additionally shifts
                    // the conditioning box. The strict numeric decoder gate
                    // lives on the GT-conditioned matting rows.
                    if (!a.response.empty() &&
                        upstream_matte_max_delta > a.matte_tolerance) {
                        std::fprintf(stderr,
                                     "[fail] matte max delta %.5f exceeds "
                                     "tolerance\n",
                                     upstream_matte_max_delta);
                        ok = false;
                    }
                }
            } else if (theirs.has_masks) {
                std::fprintf(stderr,
                             "[fail] upstream decoded a matte, we did not\n");
                ok = false;
            }
        }
        // ---- upstream batch parity (JSONL per-image payloads) ----
        if (!a.image2.empty()) {
            const std::string list_path =
                    tmp.filePath("batch.txt").toStdString();
            const std::string jsonl_path =
                    tmp.filePath("batch.jsonl").toStdString();
            {
                std::ofstream lf(list_path);
                lf << a.image << "\n" << a.image2 << "\n";
            }
            QStringList bargs{"batch",
                              "--list",
                              QString::fromStdString(list_path),
                              "--task",
                              QString::fromStdString(a.task),
                              "--query",
                              QString::fromStdString(a.query),
                              "--jsonl",
                              QString::fromStdString(jsonl_path)};
            append_runtime_opts(&bargs, a, vlm_path, mmproj_path, decoder_path,
                                resolved_device);
            if (!run_upstream_exe(a, bargs, &err_out, &error)) {
                std::fprintf(stderr, "[fail] upstream batch: %s\n",
                             error.c_str());
                return 1;
            }
            QFile jlf(QString::fromStdString(jsonl_path));
            if (!jlf.open(QIODevice::ReadOnly)) {
                std::fprintf(stderr, "[fail] upstream wrote no JSONL\n");
                return 1;
            }
            const QList<QByteArray> lines = jlf.readAll().split('\n');
            QList<QByteArray> rows;
            for (const QByteArray& line : lines)
                if (!line.trimmed().isEmpty()) rows.append(line);
            if (rows.size() != 2) {
                std::fprintf(stderr, "[fail] upstream JSONL rows=%lld\n",
                             (long long)rows.size());
                return 1;
            }
            // Our batch results were collected CPU-side before the context
            // was freed for the upstream run; reuse those copies (the last
            // batch is unchanged - no new run happened in between).
            RunOutcome b0 = batch_b0, b1 = batch_b1, t0, t1;
            ImageBuf img2;
            if (!img2.load(a.image2, &error) ||
                !parse_official_json(rows[0], img1.view.width, img1.view.height,
                                     &t0) ||
                !parse_official_json(rows[1], img2.view.width, img2.view.height,
                                     &t1)) {
                std::fprintf(stderr, "[fail] cannot parse upstream JSONL\n");
                return 1;
            }
            int d0 = 0, d1 = 0, sk0 = 0, sk1 = 0;
            const bool eq0 = answers_equivalent(b0.answer, t0.answer,
                                                eff_bin_tol, &d0, &sk0);
            const bool eq1 = answers_equivalent(b1.answer, t1.answer,
                                                eff_bin_tol, &d1, &sk1);
            batch_coord_bin_delta = std::max(d0, d1);
            batch_skeleton_match = (sk0 && sk1) ? 1 : 0;
            batch_text_match = (eq0 && eq1) ? 1 : 0;
            if (batch_text_match != 1) {
                std::fprintf(stderr,
                             "[fail] batch greedy answer diverges from "
                             "upstream beyond coord-bin tolerance %d\n",
                             eff_bin_tol);
                ok = false;
            }
            const RunOutcome* ours[2] = {&b0, &b1};
            const RunOutcome* ref[2] = {&t0, &t1};
            const int dims[2][2] = {{img1.view.width, img1.view.height},
                                    {img2.view.width, img2.view.height}};
            for (int k = 0; k < 2; ++k) {
                if (ours[k]->boxes.size() != ref[k]->boxes.size()) {
                    std::fprintf(stderr,
                                 "[fail] batch box count image%d ours=%zu "
                                 "theirs=%zu\n",
                                 k, ours[k]->boxes.size(),
                                 ref[k]->boxes.size());
                    ok = false;
                    continue;
                }
                double img_delta = 0;
                for (size_t i = 0; i < ours[k]->boxes.size(); ++i)
                    for (int c = 0; c < 4; ++c)
                        img_delta = std::max(
                                img_delta,
                                double(std::fabs(ours[k]->boxes[i][c] -
                                                 ref[k]->boxes[i][c])));
                batch_box_max_delta = std::max(batch_box_max_delta, img_delta);
                // Same bin->pixel band as the single-image gate, including
                // the 1e-3 px fp slack so a coordinate exactly at the bin
                // boundary is not failed by float rounding of the conversion.
                const double img_bin_band =
                        eff_bin_tol / 1000.0 *
                        double(std::max(dims[k][0], dims[k][1]));
                const double img_box_tol =
                        std::max(a.box_tolerance,
                                 img_bin_band + (eff_bin_tol > 0 ? 1e-3 : 0.0));
                if (img_delta > img_box_tol) {
                    std::fprintf(stderr,
                                 "[fail] batch box delta image%d %.4f px "
                                 "exceeds tolerance %.4f\n",
                                 k, img_delta, img_box_tol);
                    ok = false;
                }
            }
        }
    }

    if (a.expect_boxes_min >= 0 &&
        int(first.boxes.size()) < a.expect_boxes_min) {
        std::fprintf(stderr, "[fail] boxes %zu < expected min %d\n",
                     first.boxes.size(), a.expect_boxes_min);
        ok = false;
    }
    if (a.expect_mattes_min >= 0 && first.matte_count < a.expect_mattes_min) {
        std::fprintf(stderr, "[fail] mattes %d < expected min %d\n",
                     first.matte_count, a.expect_mattes_min);
        ok = false;
    }

    // ---- report ----
    report.insert("answer", QString::fromStdString(first.answer));
    report.insert("boxes", QString::fromStdString(boxes_json(first.boxes)));
    report.insert("box_count", int(first.boxes.size()));
    report.insert("has_masks", first.has_masks);
    report.insert("matte_count", first.matte_count);
    report.insert("matte_width", first.matte_w);
    report.insert("matte_height", first.matte_h);
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
    if (batch_index0_match >= 0) {
        report.insert("batch_index0_match", batch_index0_match);
        report.insert("batch_index1_match", batch_index1_match);
    }
    if (upstream_text_match >= 0) {
        report.insert("upstream_text_match", upstream_text_match);
        report.insert("upstream_box_max_delta_px", upstream_box_max_delta);
        report.insert("upstream_coord_max_bin_delta", upstream_coord_bin_delta);
        report.insert("upstream_skeleton_match", upstream_skeleton_match);
        if (upstream_matte_max_delta >= 0) {
            report.insert("upstream_matte_max_delta", upstream_matte_max_delta);
            report.insert("upstream_matte_mean_delta",
                          upstream_matte_mean_delta);
        }
        if (batch_text_match >= 0) {
            report.insert("batch_text_match", batch_text_match);
            report.insert("batch_box_max_delta_px", batch_box_max_delta);
            report.insert("batch_coord_max_bin_delta", batch_coord_bin_delta);
            report.insert("batch_skeleton_match", batch_skeleton_match);
        }
    }
    // Stable-output fingerprint (the runner's stability gate hashes this
    // field across repeats): answer + boxes + token counts + matte plane.
    {
        QCryptographicHash digest(QCryptographicHash::Sha256);
        digest.addData(QByteArray::fromStdString(first.answer));
        for (const auto& b : first.boxes) {
            digest.addData(reinterpret_cast<const char*>(b.data()),
                           sizeof(float) * 4);
        }
        digest.addData(reinterpret_cast<const char*>(&first.prompt_tokens),
                       sizeof(first.prompt_tokens));
        digest.addData(reinterpret_cast<const char*>(&first.completion_tokens),
                       sizeof(first.completion_tokens));
        if (!first.matte.empty()) {
            digest.addData(reinterpret_cast<const char*>(first.matte.data()),
                           qint64(first.matte.size() * sizeof(float)));
        }
        report.insert("token_hash",
                      QString::fromLatin1(digest.result().toHex()));
    }
    report.insert("verdict", ok ? "PASS" : "FAIL");
    const QByteArray payload =
            QJsonDocument(report).toJson(QJsonDocument::Compact);
    emit_report(a.json_out, payload);
    if (ctx != nullptr) aicore_wevisone_free(ctx);
    aicore_wevisone_shutdown();
    return ok ? 0 : 1;
}