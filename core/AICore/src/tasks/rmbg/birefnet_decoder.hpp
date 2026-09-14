#pragma once
#include <string>
#include <vector>

#include "tasks/rmbg/swin_backbone.hpp"


namespace rmbg {

/// BiRefNet decoder scaffold (lateral fusion + ASPPDeformable + head).
/// Phase 3: parity against PyTorch taps before full ggml graph.
/// Multi-scale encoder outputs matching BiRefNet `forward_enc` (mul_scl cat + cxt on x4).
struct Encoder4ScaleOutput {
    std::vector<float> x1, x2, x3, x4;
};

bool forward_encoder_4scale(const std::vector<float> & nchw_in, int H, int W,
                          SwinBackboneForward & bb,
                          Encoder4ScaleOutput & out,
                          std::string & err);

/// Full eval decoder: squeezed x4 + encoder scales + RGB -> 1ch logits @ HxW.
bool forward_decoder_eval(const std::vector<float> & nchw_in, int H, int W,
                          const Encoder4ScaleOutput & enc,
                          const std::vector<float> & x4_sq,
                          const WeightMap & w,
                          std::vector<float> & alpha_logits,
                          std::string & err);

/// Encoder + squeeze + decoder (BiRefNet eval, no grad).
bool forward_alpha(const std::vector<float> & nchw_in, int H, int W,
                   SwinBackboneForward & bb, const WeightMap & w,
                   std::vector<float> & alpha_logits,
                   std::string & err);

struct BiRefNetDecoderForward {
    bool init(const char * gguf_path, std::string & err);

    /// Multi-scale encoder features [s3,s2,s1,s0] -> 1-channel logits (pre-sigmoid).
    bool forward(const std::vector<std::vector<float>> & enc_feats,
                 int H, int W,
                 std::vector<float> & alpha_logits,
                 std::string & err) const;

    /// squeeze_module on deepest encoder map (x4).
    bool forward_squeeze(const std::vector<float> & x4, int H, int W,
                         std::vector<float> & out, std::string & err) const;

    bool loaded() const { return loaded_; }

private:
    bool loaded_ = false;
    WeightMap weights_;
};

} // namespace rmbg
