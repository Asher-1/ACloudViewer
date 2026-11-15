// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/facedetect/model.hpp"

#include <algorithm>
#include <stdexcept>

#include "tasks/facedetect/align.hpp"
#include "tasks/facedetect/antispoof_graph.hpp"
#include "tasks/facedetect/arcface_graph.hpp"
#include "tasks/facedetect/backend.hpp"
#include "tasks/facedetect/common.hpp"
#include "tasks/facedetect/detect.hpp"
#include "tasks/facedetect/genderage_graph.hpp"
#include "tasks/facedetect/landmark_graph.hpp"
#include "tasks/facedetect/sface_graph.hpp"

namespace fd {

std::unique_ptr<Model> Model::load(const std::string& gguf_path) {
    auto m = std::unique_ptr<Model>(new Model());
    if (!m->loader_.load(gguf_path)) {
        FD_LOG("Model::load: failed to load %s", gguf_path.c_str());
        return nullptr;
    }
    return m;
}

std::vector<Detection> Model::detect(const Image& img) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    // Detector graph stage is deferred; scrfd_detect is a stub for now.
    return scrfd_detect(loader_, img);
}

std::vector<float> Model::embed(const Image& img) const {
    return embed(img, 0.0f);
}

namespace {

bool looksLikePortraitCrop(const Image& img) {
    if (img.width < 48 || img.height < 48) return false;
    const float aspect =
            static_cast<float>(img.width) / static_cast<float>(img.height);
    return aspect >= 0.45f && aspect <= 2.2f;
}

Landmarks5 portraitTemplateLandmarks(int w, int h) {
    Landmarks5 lm{};
    // InsightFace ArcFace reference layout scaled to the crop (frontal
    // portrait).
    lm[0] = {0.341916f * w, 0.461574f * h};
    lm[1] = {0.656534f * w, 0.459999f * h};
    lm[2] = {0.500000f * w, 0.640000f * h};
    lm[3] = {0.370000f * w, 0.824000f * h};
    lm[4] = {0.631000f * w, 0.823000f * h};
    return lm;
}

std::vector<float> embedAlignedPortrait(const Model& model,
                                        const Image& img,
                                        const Landmarks5& landmarks) {
    Image aligned;
    if (!norm_crop(img, landmarks, aligned,
                   static_cast<int>(model.config().rec_input_size))) {
        throw std::runtime_error("facedetect: alignment failed");
    }
    if (model.config().recognizer == "sface") {
        return sface_embed(model.loader(), aligned, global_backend());
    }
    return arcface_embed(model.loader(), aligned, global_backend());
}

}  // namespace

std::vector<float> Model::embed(const Image& img,
                                float min_detection_score) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    std::vector<Detection> dets = scrfd_detect(loader_, img);
    if (min_detection_score > 0.0f) {
        dets.erase(std::remove_if(dets.begin(), dets.end(),
                                  [min_detection_score](const Detection& d) {
                                      return d.score < min_detection_score;
                                  }),
                   dets.end());
    }
    if (dets.empty()) {
        if (!looksLikePortraitCrop(img)) {
            throw std::runtime_error("facedetect: no face detected");
        }
        return embedAlignedPortrait(
                *this, img, portraitTemplateLandmarks(img.width, img.height));
    }
    const Detection& primary =
            *std::max_element(dets.begin(), dets.end(),
                              [](const Detection& a, const Detection& b) {
                                  return (a.x2 - a.x1) * (a.y2 - a.y1) <
                                         (b.x2 - b.x1) * (b.y2 - b.y1);
                              });
    return embedAlignedPortrait(*this, img, primary.landmarks);
}

std::vector<float> Model::embed(const Image& img, const Detection& det) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    return embedAlignedPortrait(*this, img, det.landmarks);
}

std::vector<Face> Model::analyze(const Image& img) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    // Per detected face, run the genderage head when the pack carries it (the
    // `ga.` weights). Anti-spoof remains a later stage. Each face is
    // independently warped to the 96x96 genderage crop (its own detection box),
    // so multi-face images get per-face age/gender rather than only the
    // primary's.
    std::vector<Detection> dets = scrfd_detect(loader_, img);
    const bool have_ga = loader_.config().genderage_present;
    std::vector<Face> faces;
    faces.reserve(dets.size());
    for (const Detection& d : dets) {
        Face f;
        f.det = d;
        if (have_ga) {
            std::pair<int, int> ga =
                    genderage(loader_, img, d, global_backend());
            f.gender = (ga.first == 1) ? 'M' : 'F';
            f.age = ga.second;
        }
        faces.push_back(std::move(f));
    }
    return faces;
}

bool Model::is_real(const Image& img, const Detection& d) const {
    if (!loader_.config().antispoof_present)
        return true;  // no models -> no veto
    return antispoof_score(*this, img, d) >= 0.5f;
}

std::vector<DenseLandmarkFace> Model::dense_landmarks(
        const Image& img, const std::vector<Detection>& dets) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    if (dets.empty()) return {};
    Backend& be = global_backend();
    std::vector<DenseLandmarkFace> out;
    out.reserve(dets.size());
    for (const Detection& d : dets) {
        DenseLandmarkFace face;
        face.det = d;
        face.points_2d = landmarks(loader_, false, img, d, be);
        try {
            face.points_3d = landmarks(loader_, true, img, d, be);
        } catch (const std::exception&) {
            face.points_3d.clear();
        }
        out.push_back(std::move(face));
    }
    return out;
}

}  // namespace fd
