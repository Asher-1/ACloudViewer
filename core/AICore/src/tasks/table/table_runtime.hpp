// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// AICore `table` task runtime: SLANet_plus table structure recognition.
//
// The exported Paddle program is a straight-line encoder (PP-LCNet +
// CSP-PAN, ending at hardswish_72) followed by one `while` op whose
// sub-block is the autoregressive structure decoder. The encoder runs on
// the audited OCR graph interpreter (the converter truncates the program at
// the encoder output); the decoder step is written here in ggml from the
// sub-block's 72 ops:
//
//   Fp      = F W0                                  (loop invariant)
//   e_t     = tanh(Fp + h W1 + b1);  a_t = softmax_T(e_t W2)
//   ctx_t   = a_t^T F
//   x_t     = [ctx_t, onehot(y_{t-1})]           -> Paddle GRUCell (r,z,c)
//   h_t     = (h_{t-1} - c) * z + c
//   logit_t = (h_t W3 + b3) W4 + b4;  loc_t = sigmoid((h_t W5 + b5) W6 + b6)
//   y_t     = argmax(logit_t)                      until y_t == eos
//
// onehot(y) W_ih^T is evaluated as a row gather of the token block of
// W_ih, which is exact (the one-hot product has a single non-zero term).
// K steps are unrolled per graph with the argmax fed back in-graph, so a
// table costs ceil(steps / K) graph launches instead of one per token.

#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "common/paddle_graph.hpp"

namespace aicore {
namespace table {

struct TableStructureOutput {
    /** Argmax token per decoded step; ends with eos when it was emitted. */
    std::vector<int32_t> tokens;
    /** Max softmax probability per step. */
    std::vector<float> scores;
    /** steps x locReg cell polygon values, sigmoid domain (padded input). */
    std::vector<float> loc;
    /** steps x vocab softmax rows (only filled when requested). */
    std::vector<float> probs;
    double encoderMs = 0.0;
    double decoderMs = 0.0;
};

class TableStructureModel {
public:
    TableStructureModel() = default;
    ~TableStructureModel();
    TableStructureModel(const TableStructureModel&) = delete;
    TableStructureModel& operator=(const TableStructureModel&) = delete;

    bool load(const std::string& path, std::string* error);

    /** Run one padded padSize x padSize NCHW (BGR-normalized) input. `gpu`
     *  is null for a CPU session; both handles must outlive the model (the
     *  weights realize against the device on first use). */
    bool infer(ggml_backend_t gpu,
               ggml_backend_t cpu,
               const float* nchw,
               TableStructureOutput& out,
               std::string& error,
               bool keepProbs = false);

    /** Encoder routing gate (see OcrWeights::sched_gpu_tagging). */
    void setSchedGpuTagging(bool on) { encoder_.sched_gpu_tagging = on; }

    const std::vector<std::string>& vocabulary() const { return vocab_; }
    int padSize() const { return padSize_; }
    int maxLen() const { return maxLen_; }
    int maxSteps() const { return maxSteps_; }
    int locReg() const { return locReg_; }
    int eosIndex() const { return static_cast<int>(vocab_.size()) - 1; }

    /** Drop every graph and device buffer (context teardown). */
    void release();

private:
    struct Graph {
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t alloc = nullptr;
        void clear();
    };
    bool loadDecoderWeights(std::string* error);
    bool ensureDecoder(ggml_backend_t backend, std::string& error);
    bool runEncoder(ggml_backend_t gpu,
                    ggml_backend_t cpu,
                    const float* nchw,
                    std::string& error);
    bool buildPrepGraph(std::string& error);
    bool buildStepGraph(std::string& error);

    paddle::OcrWeights encoder_;
    std::vector<std::string> vocab_;
    int padSize_ = 488;
    int maxLen_ = 488;
    int maxSteps_ = 501;
    int locReg_ = 8;
    int featC_ = 0;     // encoder channels (96)
    int hidden_ = 0;    // decoder hidden size (256)
    int64_t tokens_ = 0;  // encoder positions T = H' * W'

    // Host copies of the decoder weights in ggml mul_mat orientation.
    struct HostTensor {
        std::vector<float> data;
        int64_t ne0 = 0, ne1 = 1;
    };
    std::vector<std::pair<std::string, HostTensor>> hostDecoder_;

    // Encoder graph (fixed padded geometry -> built once per context).
    paddle::OcrExecPlan encPlan_;
    ggml_context* encCtx_ = nullptr;
    ggml_cgraph* encGraph_ = nullptr;
    ggml_tensor* encInput_ = nullptr;
    ggml_tensor* encOutput_ = nullptr;
    paddle::OcrGraphExec encExec_;
    std::shared_ptr<paddle::OcrFoldPool> encFold_;
    std::vector<float> feature_;  // Ft: T x C, channel-major (ne={T, C})

    // Decoder state on one backend.
    ggml_backend_t decBackend_ = nullptr;
    ggml_context* decWeightCtx_ = nullptr;
    ggml_backend_buffer_t decWeightBuf_ = nullptr;
    struct DecoderTensors {
        ggml_tensor *w0t = nullptr, *w1t = nullptr, *b1 = nullptr,
                    *w2t = nullptr, *wihC = nullptr, *wihY = nullptr,
                    *bih = nullptr, *whh = nullptr, *bhh = nullptr,
                    *w3t = nullptr, *b3 = nullptr, *w4t = nullptr,
                    *b4 = nullptr, *w5t = nullptr, *b5 = nullptr,
                    *w6t = nullptr, *b6 = nullptr;
    } dw_;
    Graph prep_;
    ggml_tensor* prepFt_ = nullptr;
    ggml_tensor* prepFp_ = nullptr;
    Graph step_;
    ggml_tensor* stepFt_ = nullptr;
    ggml_tensor* stepFp_ = nullptr;
    ggml_tensor* stepH0_ = nullptr;
    ggml_tensor* stepY0_ = nullptr;
    ggml_tensor* stepProbs_ = nullptr;  // {vocab, K}
    ggml_tensor* stepLoc_ = nullptr;    // {locReg, K}
    ggml_tensor* stepH_ = nullptr;      // {hidden}
    std::vector<ggml_tensor*> stepY_;   // K x I32[1]
};

}  // namespace table
}  // namespace aicore
