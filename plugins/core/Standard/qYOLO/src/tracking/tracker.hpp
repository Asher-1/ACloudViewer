// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
// Ported from ultralytics-ggml cpp_ggml/src/tracker.{hpp,cpp} (commit
// a77081c2): only the namespace (qyolo::track), the frame type (self-
// contained RgbFrame instead of yolo::Image) and the OpenCV build gate
// (QYOLO_WITH_OPENCV instead of YOLO_WITH_OPENCV) differ. Semantics are
// aligned line-by-line with the official ultralytics runtime (8.4.x):
// byte_tracker / bot_sort / oc_sort / deep_oc_sort / fast_tracker /
// track_tracker + utils/{matching,kalman_filter,gmc,stracks,reid}.py —
// including the OCR unmatched-detection remap, the TTSTrack promotion rule
// (tracklet_len + 1 >= min_track_len), state-based track routing, the
// DeepOCSort det_thresh = track_high_thresh coupling, the FASTTracker init
// suppression stack, and the shared smooth_feature ReID EMA contract.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qyolo {
namespace track {

/* Borrowed tightly-packed RGB8 frame (row-major, 3 bytes/pixel) handed to
 * the tracker for GMC. The caller owns the storage; the tracker reads it
 * synchronously inside update() and never retains the pointer. */
struct RgbFrame {
    int w = 0;
    int h = 0;
    const std::uint8_t* rgb = nullptr;
};

/* Multi-object tracking aligned 1:1 with ultralytics/trackers/ (Python).
 *
 * Every tracker mirrors its Python counterpart (byte_tracker.BYTETracker,
 * bot_sort.BOTSORT, oc_sort.OCSORT, deep_oc_sort.DeepOCSORT,
 * fast_tracker.FASTTracker, track_tracker.TRACKTRACK) algorithm-for-algorithm on
 * the with_reid=False configuration path, which is the default of all six
 * official tracker YAMLs. Trackers consume per-frame detections in original
 * image coordinates — the same space Python feeds tracker.update(results)
 * after non_max_suppression's scale_boxes — plus the raw frame so GMC can
 * estimate camera motion.
 */

// One YAML key per member. Defaults per tracker type come from
// ultralytics/cfg/trackers/<type>.yaml (see default_tracker_config); a
// `--tracker <file>.yaml` overrides any subset of these.
struct TrackConfig {
    std::string tracker_type = "tracktrack";
    float track_high_thresh = 0.25f;
    float track_low_thresh = 0.1f;
    float new_track_thresh = 0.25f;
    int track_buffer = 30;
    float match_thresh = 0.8f;
    bool fuse_score = true;
    // OC-SORT specifics (ocsort.yaml / deepocsort.yaml)
    int delta_t = 3;
    float inertia = 0.2f;
    bool use_byte = false;
    // BOTSORT / DeepOCSORT / TRACKTRACK specifics
    std::string gmc_method = "sparseOptFlow";  // sparseOptFlow|orb|sift|ecc|none (extras need an OpenCV build)
    float proximity_thresh = 0.5f;
    float appearance_thresh = 0.8f;
    bool with_reid = false;  // model="auto": ReID engages through FrameInput.feats (no encoder ships in-tree)
    std::string model = "auto";
    float alpha_fixed_emb = 0.95f;
    // FastTracker specifics (fasttrack.yaml)
    int reset_velocity_offset_occ = 5;
    int reset_pos_offset_occ = 3;
    float enlarge_bbox_occ = 1.1f;
    float dampen_motion_occ = 0.5f;
    int active_occ_to_lost_thresh = 10;
    float occ_cover_thresh = 0.7f;
    int occ_reappear_window = 40;
    float init_iou_suppress = 0.7f;
    // TrackTrack specifics (tracktrack.yaml)
    float lost_match_thr = 0.0f;
    float penalty_p = 0.2f;
    float penalty_q = 0.4f;
    float reduce_step = 0.05f;
    float iou_weight = 0.5f;
    float reid_weight = 0.5f;
    float conf_weight = 0.1f;
    float angle_weight = 0.05f;
    float tai_thr = 0.55f;
    int min_track_len = 3;
};

// Built-in defaults for bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack,
// each mirroring ultralytics/cfg/trackers/<type>.yaml. Unknown names return false.
bool default_tracker_config(const std::string& type, TrackConfig& out);

// Load a tracker config from an official ultralytics/cfg/trackers/*.yaml file.
// The files are flat "key: value" maps; unknown keys are ignored (matching the
// Python getattr-with-default reads). Returns false on unreadable files.
bool load_tracker_config_yaml(const std::string& path, TrackConfig& out);

// Resolve --tracker: a bare type name uses the built-in defaults, a *.yaml /
// *.yml path is parsed. Unknown forms return false with a message on stderr.
bool resolve_tracker_config(const std::string& spec, TrackConfig& out);

// Detection handed to a tracker. Center format so OBB (cx, cy, w, h, angle)
// and axis-aligned boxes share one layout; angle < -pi means axis-aligned.
struct TrackDet {
    float cx = 0, cy = 0, w = 0, h = 0;
    float angle = -10.0f;  // -10 (outside [-pi, pi]) = axis-aligned; NaN never valid
    float score = 0;
    int class_id = 0;
    int idx = 0;  // index in the full frame detection set (joins masks/keypoints when drawing)
    bool angled() const { return angle >= -3.1515927f && angle <= 3.1515927f; }
};

// One tracked output: the detection carrying this frame's track id, mirroring
// the Python result row [coords..., id, score, cls, idx].
struct TrackedBox {
    TrackDet det;
    int track_id = 0;
};

struct FrameInput {
    std::vector<TrackDet> dets;          // full post-NMS frame detections, original-image coords
    const RgbFrame* frame = nullptr;  // raw RGB frame (GMC input; may be null to disable GMC)
    std::vector<TrackDet> dets_del;      // TRACKTRACK loose-NMS recoveries (detect/obb only; may be empty)
    // Official model="auto" ReID features (predict.py get_obj_feats),
    // index-aligned with dets: one (unnormalized) vector per detection;
    // empty when ReID is off, the head is end2end, or the runtime provides
    // no features. dets_del rows carry no features (upstream parity).
    std::vector<std::vector<float>> feats;
};

struct Tracker {
    virtual ~Tracker() = default;
    // Advance one frame; returns the active tracks. Mirrors tracker.update().
    virtual std::vector<TrackedBox> update(const FrameInput& in) = 0;
    virtual void reset() = 0;
};

// Factory: bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack.
// with_reid=true is accepted on the BoT-SORT family: the ReID cosine term
// engages through FrameInput.feats (the official model="auto" detector-
// feature path); a stream without features degrades to motion-only
// association, mirroring the upstream "feats missing" behavior. The
// factory rejects with_reid only for the trackers that never consume
// features and, on builds without OpenCV, the orb/sift/ecc GMC methods.
std::unique_ptr<Tracker> create_tracker(const TrackConfig& cfg);

}  // namespace track
}  // namespace qyolo
