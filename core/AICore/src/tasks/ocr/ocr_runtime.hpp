// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "common/paddle_graph.hpp"

namespace aicore {
namespace ocr {
using paddle::GraphOp;
using paddle::OcrProgram;
using paddle::OcrProbeOptions;
using paddle::OcrBuildOptions;
using paddle::OcrFoldPool;
using paddle::OcrWeights;
using paddle::buildGraph;
using paddle::OcrExecPlan;
using paddle::makeExecPlan;
using paddle::execBuildOptions;
using paddle::OcrGraphExec;
using paddle::prepareGraph;
using paddle::computeGraph;

/** CTC greedy decode over a (T, vocab) probability row set. Index 0 is the
 *  CTC blank, the last index is the word separator; repeated labels
 *  collapse. Returns UTF-8 text and the mean non-blank score. */
std::string ctcGreedyDecode(const float* probs,
                            int timeSteps,
                            int vocab,
                            const std::vector<std::string>& dictionary,
                            float* meanScore);

/** Same decode over a pre-reduced row set: `bestIndex[t]` is the argmax
 *  class and `bestProb[t]` its probability (the two scalars the decoder
 *  actually consumes), as produced by the in-graph argmax/get_rows head on
 *  GPU sessions. Shares the collapse/dictionary logic with ctcGreedyDecode. */
std::string ctcGreedyDecodeIndexed(const int32_t* bestIndex,
                                   const float* bestProb,
                                   int timeSteps,
                                   int vocab,
                                   const std::vector<std::string>& dictionary,
                                   float* meanScore);

/** Recognition preprocessing: resize the input to height 48 preserving
 *  aspect (clamped to `targetWidth`), normalize with the PP-OCR ImageNet
 *  statistics, and emit an NCHW float buffer. */
std::vector<float> preprocessRec(const unsigned char* rgba,
                                 int width,
                                 int height,
                                 int channels,
                                 int targetWidth,
                                 int* outW);
}  // namespace ocr
}  // namespace aicore
