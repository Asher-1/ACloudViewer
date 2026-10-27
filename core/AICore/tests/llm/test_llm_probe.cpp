// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Real-model probe for the local LLM / VLM task. One process loads a model
// (and optional projector), runs warmups plus N greedy repeats of the same
// request, then checks the lifecycle invariants that a live UI depends on:
// determinism, context recreation, streaming == final text, cooperative
// cancel and stop sequences. Emits one JSON object (stdout or --json-out).
//
// Exit codes: 0 pass, 1 invariant/accuracy failure, 2 usage, 77 missing
// model asset (the common runner records it as unavailable).

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QLibrary>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QStringList>
#include <QTemporaryDir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "aicore/llm_capi.h"
#include "aicore/runtime_capi.h"

namespace {

struct Args {
    std::string model, mmproj, device = "auto", prompt, system, image,
            upstream_dir, json_out, profile_nvtx;
    bool upstream_requested = false;
    int max_tokens = 64, repeats = 3, warmups = 1, threads = 0;
    std::string contains;  // optional case-insensitive answer gate
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t i = size_t(std::ceil(p * double(v.size()))) - 1;
    return v[std::min(i, v.size() - 1)];
}

std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

uint64_t fnv1a(const int32_t* ids, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        const auto* b = reinterpret_cast<const unsigned char*>(&ids[i]);
        for (size_t k = 0; k < sizeof(int32_t); ++k) {
            h ^= b[k];
            h *= 1099511628211ull;
        }
    }
    return h;
}

struct Stream {
    std::string text;
    int calls = 0;
    int stop_after = -1;  // cancel through the callback after N chunks
};
int stream_cb(const char* chunk, size_t bytes, void* user) {
    auto* s = static_cast<Stream*>(user);
    s->text.append(chunk, bytes);
    ++s->calls;
    return s->stop_after >= 0 && s->calls >= s->stop_after ? 1 : 0;
}

// Runs <upstream-dir>/llama-mtmd-cli (vision) or llama-completion (text),
// built from the same pinned llama.cpp archive, with the probe's
// request and returns the generated text. Device names follow ggml's
// registry ("CUDA0", "Vulkan0"); CPU disables every offload.
std::string run_upstream(const Args& a,
                         const std::string& device,
                         const std::string& image_png,
                         std::string* error) {
    const bool vision = !a.image.empty();
    QStringList args{"-m", QString::fromStdString(a.model), "-p",
                     QString::fromStdString(a.prompt), "--temp", "0", "-n",
                     QString::number(a.max_tokens), "--no-warmup"};
    if (device == "CPU")
        args << "-dev" << "none" << "-ngl" << "0";
    else
        args << "-dev" << QString::fromStdString(device) << "-ngl" << "99";
    if (a.threads > 0) args << "-t" << QString::number(a.threads);
    if (!a.system.empty())
        args << "-sys" << QString::fromStdString(a.system);
    QProcess process;
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    // The runner points the loader at the patched production ggml, whose op
    // enum differs from the pristine one: the reference tool must resolve its
    // own libraries first or it mixes two ggml builds and aborts.
    const QString dir = QString::fromStdString(a.upstream_dir);
    for (const char* var : {"LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"}) {
        const QString old = env.value(var);
        env.insert(var, old.isEmpty() ? dir
                                      : dir + QDir::listSeparator() + old);
    }
    const QString marker = "<<<AICORE-UPSTREAM-RESPONSE>>>";
    if (vision) {
        args << "--mmproj" << QString::fromStdString(a.mmproj) << "--image"
             << QString::fromStdString(image_png);
        if (device == "CPU") args << "--no-mmproj-offload";
        env.insert("MTMD_TEST_RESPONSE_MARKER", marker);
    } else {
        args << "-cnv" << "-st" << "--no-display-prompt";
    }
    process.setProcessEnvironment(env);
    process.setStandardInputFile(QProcess::nullDevice());
#ifdef _WIN32
    const QString suffix = ".exe";
#else
    const QString suffix;
#endif
    const QString tool = dir +
                         (vision ? "/llama-mtmd-cli" : "/llama-completion") +
                         suffix;
    // dlopen 失败在 Release 工具中可能只让设备消失；必须先核实同名
    // backend 可用，不能把空输出误报成模型数值分叉。
    if (device != "CPU") {
        process.start(tool, {"--list-devices"});
        if (!process.waitForFinished(60000) ||
            process.exitStatus() != QProcess::NormalExit ||
            process.exitCode() != 0) {
            *error = "cannot enumerate upstream devices: " +
                     process.readAllStandardError().right(400).toStdString();
            return {};
        }
        const QByteArray devices = process.readAllStandardOutput() +
                                   process.readAllStandardError();
        if (!devices.contains((QString::fromStdString(device) + ":").toUtf8())) {
            *error = "upstream device unavailable: " + device + "; " +
                     devices.right(400).toStdString();
            return {};
        }
    }
    process.start(tool, args);
    if (!process.waitForFinished(600000) ||
        process.exitStatus() != QProcess::NormalExit ||
        process.exitCode() != 0) {
        *error = "exit " + std::to_string(process.exitCode()) + ": " +
                 process.readAllStandardError().right(400).toStdString();
        return {};
    }
    std::string out = process.readAllStandardOutput().toStdString();
    if (vision) {
        const std::string m = marker.toStdString() + "\n";
        const size_t at = out.find(m);
        if (at == std::string::npos) {
            *error = "response marker missing";
            return {};
        }
        out = out.substr(at + m.size());
    } else {
        const std::string eot = " [end of text]";
        const size_t at = out.rfind(eot);
        if (at != std::string::npos) out.resize(at);
    }
    if (out.find_first_not_of(" \r\n\t") == std::string::npos) {
        *error = "upstream returned no response: " +
                 process.readAllStandardError().right(400).toStdString();
        return {};
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--model") a.model = next();
        else if (k == "--mmproj") a.mmproj = next();
        else if (k == "--device") a.device = next();
        else if (k == "--prompt") a.prompt = next();
        else if (k == "--system") a.system = next();
        else if (k == "--image") a.image = next();
        else if (k == "--max-tokens") a.max_tokens = std::stoi(next());
        else if (k == "--repeats") a.repeats = std::stoi(next());
        else if (k == "--warmups") a.warmups = std::stoi(next());
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--upstream-dir") {
            a.upstream_dir = next();
            a.upstream_requested = true;
        }
        else if (k == "--contains") a.contains = next();
        else if (k == "--json-out") a.json_out = next();
        else if (k == "--profile-nvtx") a.profile_nvtx = next();
        else {
            std::fprintf(stderr, "unknown argument %s\n", k.c_str());
            return 2;
        }
    }
    if (a.model.empty() || a.prompt.empty() || a.repeats < 1) {
        std::fprintf(stderr,
                     "usage: test_llm_probe --model M.gguf [--mmproj P.gguf] "
                     "--prompt TEXT [--image I] [--device D] [--max-tokens N] "
                     "[--repeats N] [--warmups N] [--upstream-dir DIR] "
                     "[--contains WORD] [--json-out F]\n");
        return 2;
    }
    if (a.mmproj == "auto") {
        // Pair through the task catalog, never a private naming rule.
        const std::string name = QFileInfo(QString::fromStdString(a.model))
                                         .fileName()
                                         .toStdString();
        const auto* entry = aicore_llm_model_by_filename(name.c_str());
        a.mmproj = entry && entry->mmproj_filename
                           ? QFileInfo(QString::fromStdString(a.model))
                                             .absolutePath()
                                             .toStdString() +
                                     "/" + entry->mmproj_filename
                           : std::string();
    }
    for (const auto& path : {a.model, a.mmproj, a.image}) {
        if (!path.empty() && !QFileInfo::exists(QString::fromStdString(path))) {
            std::fprintf(stderr, "asset unavailable: %s\n", path.c_str());
            return 77;
        }
    }
    // An unconfigured {reference:llama_upstream} expands to "": skip, never
    // pass silently (the runner counts the skip against the gate).
    if (a.upstream_requested) {
        const QString tool = QString::fromStdString(a.upstream_dir) +
                             (a.image.empty() ? "/llama-completion"
                                              : "/llama-mtmd-cli");
        if (a.upstream_dir.empty() ||
            !(QFileInfo::exists(tool) || QFileInfo::exists(tool + ".exe"))) {
            std::fprintf(stderr, "upstream reference unavailable: %s\n",
                         tool.toStdString().c_str());
            return 77;
        }
    }

    // 可选诊断使用调用方显式提供的 NVTX 库；普通 probe 无 SDK 编译/链接依赖。
#ifdef Q_OS_WIN
    using NvtxPush = int (__stdcall *)(const char*);
    using NvtxPop = int (__stdcall *)();
#else
    using NvtxPush = int (*)(const char*);
    using NvtxPop = int (*)();
#endif
    QLibrary nvtx;
    NvtxPush nvtxPush = nullptr;
    NvtxPop nvtxPop = nullptr;
    QString nvtxDigest;
    QJsonArray measuredRanges;
    if (!a.profile_nvtx.empty()) {
        const QString path = QString::fromStdString(a.profile_nvtx);
        nvtx.setFileName(path);
        if (nvtx.load()) {
            nvtxPush = reinterpret_cast<NvtxPush>(nvtx.resolve("nvtxRangePushA"));
            nvtxPop = reinterpret_cast<NvtxPop>(nvtx.resolve("nvtxRangePop"));
        }
        QFile file(path);
        QCryptographicHash hash(QCryptographicHash::Sha256);
        if (!nvtxPush || !nvtxPop || !file.open(QIODevice::ReadOnly) || !hash.addData(&file)) {
            std::fprintf(stderr, "NVTX 诊断库无法加载或缺少 range API：%s\n", a.profile_nvtx.c_str());
            return 1;
        }
        nvtxDigest = QString::fromLatin1(hash.result().toHex());
    }

    QImage image;
    aicore_image_view view{};
    if (!a.image.empty()) {
        image = QImage(QString::fromStdString(a.image))
                        .convertToFormat(QImage::Format_RGB888);
        if (image.isNull()) {
            std::fprintf(stderr, "cannot decode image %s\n", a.image.c_str());
            return 1;
        }
        view = {image.constBits(), image.width(), image.height(),
                size_t(image.bytesPerLine()), AICORE_IMAGE_RGB8};
    }

    auto options = aicore_llm_default_options();
    options.model_path = a.model.c_str();
    options.mmproj_path = a.mmproj.empty() ? nullptr : a.mmproj.c_str();
    options.device = a.device.c_str();
    options.threads = a.threads;

    std::vector<aicore_llm_message> messages;
    if (!a.system.empty()) messages.push_back({"system", a.system.c_str()});
    messages.push_back({"user", a.prompt.c_str()});
    auto request = aicore_llm_default_request();
    request.messages = messages.data();
    request.message_count = messages.size();
    request.images = a.image.empty() ? nullptr : &view;
    request.image_count = a.image.empty() ? 0 : 1;
    request.max_tokens = uint32_t(a.max_tokens);

    QJsonObject report;
    QJsonArray failures;
    auto fail = [&](const std::string& why) {
        failures.append(QString::fromStdString(why));
    };
    std::vector<double> e2e, prefill, decode, decode_tps, process_cpu;
    std::string first_text;
    uint64_t first_hash = 0;
    uint32_t first_tokens = 0, prompt_tokens = 0;
    std::string device;

    // Two independent contexts: the second proves recreation reproduces the
    // first context's greedy output exactly.
    for (int round = 0; round < 2; ++round) {
        aicore_llm_ctx* ctx = aicore_llm_create(&options);
        if (!aicore_llm_ready(ctx)) {
            fail(std::string("create: ") + aicore_llm_last_error(ctx));
            aicore_llm_free(ctx);
            break;
        }
        device = aicore_llm_context_device(ctx);
        if (!a.image.empty() && !aicore_llm_supports_vision(ctx))
            fail("projector loaded but vision unsupported");
        const int runs = round == 0 ? a.warmups + a.repeats : 1;
        for (int i = 0; i < runs; ++i) {
            Stream stream;
            request.stream_cb = stream_cb;
            request.stream_user_data = &stream;
            aicore_llm_result* r = nullptr;
            const bool measured = round == 0 && i >= a.warmups;
            bool rangeStarted = false;
            if (measured && nvtxPush) {
                const QString name = QStringLiteral("aicore/llm/measured/%1").arg(i - a.warmups);
                // 动态 NVTX 的返回值不证明 profiler 已采集；本机 Nsight
                // 在真实记录范围时也返回 -1。每次 push 都必须配对 pop，
                // 真实采集数量/持续时间由 Nsight 事件表验证，不能由返回值推断。
                nvtxPush(name.toUtf8().constData());
                rangeStarted = true;
                measuredRanges.append(name);
            }
            const std::clock_t cpu_start = std::clock();
            const int rc = aicore_llm_generate(ctx, &request, &r);
            const double cpu_ms = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
            // generate 是同步公共调用；即使失败也先关闭范围，不标记 warmup/recreate/cancel。
            if (rangeStarted) nvtxPop();
            if (rc != 0 || !r) {
                fail(std::string("generate rc=") + std::to_string(rc) + ": " +
                     aicore_llm_last_error(ctx));
                aicore_llm_result_free(r);
                break;
            }
            const std::string text(r->text, r->text_bytes);
            const uint64_t hash = fnv1a(r->token_ids, r->generated_tokens);
            if (stream.text != text) fail("streamed text != final text");
            if (!(r->timings.valid_fields & AICORE_TIMING_E2E))
                fail("e2e timing not reported");
            if (first_text.empty() && round == 0 && i == 0) {
                first_text = text;
                first_hash = hash;
                first_tokens = r->generated_tokens;
                prompt_tokens = r->prompt_tokens;
            } else if (text != first_text || hash != first_hash) {
                fail(round == 0 ? "greedy output differs across repeats"
                                : "recreated context output differs");
            }
            if (measured) {
                e2e.push_back(r->timings.e2e_ms);
                prefill.push_back(r->prefill_ms);
                decode.push_back(r->decode_ms);
                process_cpu.push_back(cpu_ms);
                if (r->generated_tokens && r->decode_ms > 0)
                    decode_tps.push_back(1000.0 * r->generated_tokens /
                                         r->decode_ms);
            }
            aicore_llm_result_free(r);
        }
        if (round == 1) {
            // Cooperative cancel through the stream callback: partial result.
            Stream stream;
            stream.stop_after = 2;
            request.stream_user_data = &stream;
            aicore_llm_result* r = nullptr;
            const int rc = aicore_llm_generate(ctx, &request, &r);
            if (first_tokens > 3) {
                if (rc != 1 || !r ||
                    r->finish_reason != AICORE_LLM_FINISH_CANCELLED ||
                    r->generated_tokens >= first_tokens)
                    fail("stream-callback cancel did not stop generation");
                else if (first_text.compare(0, r->text_bytes, r->text,
                                            r->text_bytes) != 0)
                    fail("cancelled partial text is not a prefix");
            }
            aicore_llm_result_free(r);
            // Stop sequence: cut at the first character of the reference.
            if (first_text.size() > 4) {
                const std::string stop = first_text.substr(2, 2);
                const char* stops[] = {stop.c_str()};
                auto stopped = request;
                stopped.stream_cb = nullptr;
                stopped.stop_sequences = stops;
                stopped.stop_count = 1;
                r = nullptr;
                if (aicore_llm_generate(ctx, &stopped, &r) != 0 || !r ||
                    r->finish_reason != AICORE_LLM_FINISH_STOP ||
                    std::string(r->text, r->text_bytes) !=
                            first_text.substr(0, first_text.find(stop)))
                    fail("stop sequence did not trim the output");
                aicore_llm_result_free(r);
            }
        }
        aicore_llm_free(ctx);
    }

    bool accuracy = true;
    if (!a.upstream_dir.empty() && device.empty()) {
        report["upstream_text_match"] = 0;
        accuracy = false;  // no context: nothing to compare against
    } else if (!a.upstream_dir.empty()) {
        // Transient truth from the upstream tool of the same pinned llama.cpp
        // revision, same model/projector/prompt/device, greedy. The tool
        // reads a lossless PNG of the exact pixels this probe decoded: the
        // decoded image is caller-owned input, and JPEG decoders differ
        // (Qt/libjpeg vs stb_image) enough to flip late greedy tokens.
        std::string error, png;
        QTemporaryDir scratch;
        if (!a.image.empty()) {
            png = scratch.filePath("input.png").toStdString();
            if (!scratch.isValid() ||
                !image.save(QString::fromStdString(png), "PNG"))
                error = "cannot write the reference PNG";
        }
        const std::string expected =
                error.empty() ? run_upstream(a, device, png, &error)
                              : std::string();
        auto trim = [](std::string s) {
            size_t b = 0;
            while (b < s.size() && (s[b] == ' ' || s[b] == '\n')) ++b;
            size_t e = s.size();
            while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\n' ||
                             s[e - 1] == '\r'))
                --e;
            return s.substr(b, e - b);
        };
        if (!error.empty()) {
            fail("upstream reference failed: " + error);
            accuracy = false;
        } else {
            accuracy = trim(first_text) == trim(expected);
        }
        report["upstream_text"] = QString::fromStdString(expected);
        report["upstream_text_match"] = accuracy ? 1 : 0;
        if (error.empty() && !accuracy)
            fail("greedy text differs from the upstream tool");
    }
    if (!a.contains.empty()) {
        const bool hit =
                lower(first_text).find(lower(a.contains)) != std::string::npos;
        report["contains"] = QString::fromStdString(a.contains);
        report["answer_contains_hit"] = hit ? 1 : 0;
        if (!hit) fail("answer does not contain '" + a.contains + "'");
        accuracy = accuracy && hit;
    }

    report["model"] = QString::fromStdString(a.model);
    report["mmproj"] = QString::fromStdString(a.mmproj);
    report["device"] = QString::fromStdString(device);
    report["prompt_tokens"] = int(prompt_tokens);
    report["generated_tokens"] = int(first_tokens);
    report["text"] = QString::fromStdString(first_text);
    report["token_hash"] = QString::number(first_hash, 16);
    report["accuracy_pass"] = accuracy;
    // *_ms leaves are recorded by the common runner as timing metrics.
    report["e2e_p50_ms"] = percentile(e2e, 0.5);
    report["e2e_p95_ms"] = percentile(e2e, 0.95);
    report["prefill_p50_ms"] = percentile(prefill, 0.5);
    report["decode_p50_ms"] = percentile(decode, 0.5);
    report["decode_tokens_per_s_p50"] = percentile(decode_tps, 0.5);
    // 保留 measured 顺序以定位尾部；warmup、重建与生命周期调用不混入样本。
    // process_cpu 是本进程各线程的 CPU 时间总和，不是 GPU event 时间。
    const auto samples = [](const std::vector<double>& values) {
        QJsonArray array;
        for (double value : values) array.append(value);
        return array;
    };
    report["e2e_samples_ms"] = samples(e2e);
    report["prefill_samples_ms"] = samples(prefill);
    report["decode_samples_ms"] = samples(decode);
    report["process_cpu_samples_ms"] = samples(process_cpu);
    report["repeats"] = a.repeats;
    report["warmups"] = a.warmups;
    if (nvtxPush) {
        report["profiling_only"] = true;
        // 这是已发出的成对范围数，不是独立的 profiler 采集证明。
        report["measured_nvtx_ranges"] = measuredRanges.size();
        report["measured_nvtx_names"] = measuredRanges;
        report["nvtx_library_sha256"] = nvtxDigest;
    }
    report["failures"] = failures;
    report["pass"] = failures.isEmpty();
    // Exactly one sink: the runner merges stdout JSON lines and the report
    // file, and a metric present in both would resolve as ambiguous.
    const QByteArray json = QJsonDocument(report).toJson(QJsonDocument::Compact);
    if (!a.json_out.empty()) {
        std::ofstream(a.json_out, std::ios::binary)
                .write(json.constData(), json.size());
    } else {
        std::fwrite(json.constData(), 1, size_t(json.size()), stdout);
        std::fputc('\n', stdout);
    }
    aicore_llm_shutdown();
    return failures.isEmpty() ? 0 : 1;
}
