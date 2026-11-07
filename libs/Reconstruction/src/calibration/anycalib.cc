// SPDX-License-Identifier: BSD-3-Clause

#include "calibration/anycalib.h"

#include "calibration/single_view_calibrator.h"
#include "util/download.h"
#include "util/file.h"
#include "util/logging.h"
#ifdef AICore_ENABLED
#include "aicore/anycalib_capi.h"
#include "aicore/asset_digests.h"
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>

namespace colmap {
namespace {

// Fixed square network input size: round(sqrt(102400) / 14) * 14, the square
// training resolution of AnyCalib. See `scripts/anycalib/export_onnx.py` for
// why dynamic input sizes are not supported.
constexpr int kAnyCalibInputSize = 322;

// Upsample small images in place, preserving the aspect ratio, and update
// `scale_xy` with the applied per-axis rescaling.
void UpsampleToMinSize(Bitmap& image, int min_size, Eigen::Vector2d& scale_xy) {
  const int width = image.Width();
  const int height = image.Height();
  if (width >= min_size && height >= min_size) {
    return;
  }
  const double s = std::max(static_cast<double>(min_size) / height,
                            static_cast<double>(min_size) / width);
  const int new_width = std::lround(width * s);
  const int new_height = std::lround(height * s);
  image.Rescale(new_width, new_height);
  scale_xy.x() = static_cast<double>(image.Width()) / width;
  scale_xy.y() = static_cast<double>(image.Height()) / height;
}

// Origin (left, top) of the centered square crop of a `width` x `height`
// image. The integer divisions center the crop up to one pixel.
Eigen::Vector2i CenterCropOrigin(int width, int height) {
  const int side = std::min(width, height);
  return Eigen::Vector2i((width - side) / 2, (height - side) / 2);
}

#ifdef AICore_ENABLED

// Default published GGUF: the catalog's recommended half-precision variant.
constexpr const char* kDefaultAnyCalibModelFile =
        "anycalib_gen_v1.0.0_322x322.f16.gguf";

// Resolve the anycalib GGUF path. An explicit local path wins; otherwise the
// AICore task catalog supplies the published URL and the pinned digest from
// aicore/asset_digests.h, and the shared download utility fills the
// cross-platform cache directory (DA3 resolver pattern). Returns an empty
// string on any resolution or integrity failure (the caller throws).
std::string ResolveAnyCalibModelPath(const std::string& model_path) {
  if (!model_path.empty()) {
    if (ExistsFile(model_path)) {
      return model_path;
    }
    LOG(ERROR) << "AnyCalib: explicit model path does not exist: "
               << model_path;
    return "";
  }

  const aicore_anycalib_model_entry* entry =
      aicore_anycalib_model_by_filename(kDefaultAnyCalibModelFile);
  if (entry == nullptr) {
    LOG(ERROR) << "AnyCalib: model not in the AICore catalog: "
               << kDefaultAnyCalibModelFile;
    return "";
  }
  const char* pinned_digest =
      aicore::AssetDigestForFile(kDefaultAnyCalibModelFile);
  if (pinned_digest == nullptr) {
    LOG(ERROR) << "AnyCalib: no pinned digest for model asset: "
               << kDefaultAnyCalibModelFile;
    return "";
  }

  char* cache_dir_c = aicore_anycalib_model_cache_dir();
  if (cache_dir_c == nullptr) {
    LOG(ERROR) << "AnyCalib: failed to query the model cache directory";
    return "";
  }
  const std::filesystem::path cache_dir(cache_dir_c);
  aicore_anycalib_free_buffer(cache_dir_c);
  std::error_code ec;
  std::filesystem::create_directories(cache_dir, ec);

  const auto cached_path = cache_dir / kDefaultAnyCalibModelFile;
  if (std::filesystem::exists(cached_path)) {
    // Verify the cached model before trusting it: a truncated or corrupted
    // cache must never silently feed the calibration.
    const std::string cached_digest = ComputeFileSHA256(cached_path);
    if (cached_digest == pinned_digest) {
      return cached_path.string();
    }
    LOG(WARNING) << "AnyCalib: cached model failed digest check (got "
                 << cached_digest << ", expected " << pinned_digest
                 << "), re-downloading: " << cached_path.string();
    std::filesystem::remove(cached_path);
  }

  const auto blob = DownloadFile(entry->download_url);
  if (!blob.has_value()) {
    LOG(ERROR) << "AnyCalib: failed to download model from: "
               << entry->download_url;
    return "";
  }
  // Verify before caching: a corrupted download must never poison the cache.
  const std::string blob_digest =
      ComputeSHA256(std::string_view(blob->data(), blob->size()));
  if (blob_digest != pinned_digest) {
    LOG(ERROR) << "AnyCalib: downloaded model failed digest check (got "
               << blob_digest << ", expected " << pinned_digest << "): "
               << entry->download_url;
    return "";
  }
  std::ofstream ofs(cached_path, std::ios::binary);
  if (!ofs) {
    LOG(ERROR) << "AnyCalib: failed to write cached model to: "
               << cached_path.string();
    return "";
  }
  ofs.write(blob->data(), static_cast<std::streamsize>(blob->size()));
  ofs.close();
  return cached_path.string();
}

// Single-view calibrator backed by the AICore anycalib task (libAICore):
// the ggml graph returns typed per-pixel rays for the 322x322 network crop;
// intrinsics fitting is shared with the ray-based backend interface.
class AnyCalibCalibrator : public SingleViewCalibrator {
 public:
  explicit AnyCalibCalibrator(const SingleViewCalibrationOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
    THROW_CHECK_NOTNULL(options_.anycalib);
    const std::string model_path =
        ResolveAnyCalibModelPath(options_.anycalib->model_path);
    THROW_CHECK(!model_path.empty())
        << "AnyCalib: failed to resolve the anycalib GGUF model";
    aicore_anycalib_options* opts = aicore_anycalib_options_new();
    THROW_CHECK_NOTNULL(opts);
    // CPU routing until the task gains backend parity (the task itself
    // rejects non-CPU devices).
    aicore_anycalib_options_set_device(opts, "cpu");
    aicore_anycalib_options_set_threads(
        opts, options_.num_threads > 0 ? options_.num_threads : 0);
    ctx_ = aicore_anycalib_load_opts(model_path.c_str(), opts);
    aicore_anycalib_options_free(opts);
    THROW_CHECK(ctx_ != nullptr && aicore_anycalib_is_ready(ctx_))
        << "AnyCalib: failed to load the anycalib GGUF: "
        << (ctx_ != nullptr && aicore_anycalib_last_error(ctx_) != nullptr
                ? aicore_anycalib_last_error(ctx_)
                : "unknown error");
  }

  ~AnyCalibCalibrator() override {
    if (ctx_ != nullptr) {
      aicore_anycalib_free(ctx_);
      ctx_ = nullptr;
    }
  }

  bool Calibrate(const Bitmap& bitmap,
                 Camera* camera,
                 const PosePrior& pose_prior) const override {
    THROW_CHECK_NOTNULL(camera);
    // The network needs RGB input; convert grayscale images.
    Bitmap rgb_bitmap;
    const Bitmap* input_bitmap = &bitmap;
    if (!bitmap.IsRGB()) {
      rgb_bitmap = bitmap.CloneAsRGB();
      input_bitmap = &rgb_bitmap;
    }

    // An empty target model preserves the camera's existing model. Fitting a
    // non-perspective model fails gracefully below (`FitCameraFromRays`
    // rejects it), leaving the camera unmodified.
    const CameraModelId model_id =
        options_.camera_model.empty()
            ? camera->ModelId()
            : CameraModelNameToId(options_.camera_model);
    // The camera-model switches throw on invalid models instead of returning
    // false, so reject them explicitly before running inference.
    if (model_id == CameraModelId::kInvalid) {
      LOG(WARNING) << "Cannot calibrate camera with invalid model";
      return false;
    }

    AnyCalibInput input = PrepareAnyCalibInput(*input_bitmap, pose_prior);
    float* rays_buffer = nullptr;
    float* tangent_buffer = nullptr;
    int32_t count = 0;
    const int rc = aicore_anycalib_run(ctx_,
                                       input.data.data(),
                                       &rays_buffer,
                                       &tangent_buffer,
                                       &count);
    if (rc != 0 || rays_buffer == nullptr ||
        count != kAnyCalibInputSize * kAnyCalibInputSize) {
      LOG(ERROR) << "AnyCalib: inference failed (rc=" << rc
                 << ", count=" << count << ")";
      aicore_anycalib_free_buffer(rays_buffer);
      aicore_anycalib_free_buffer(tangent_buffer);
      return false;
    }
    const float* rays_data = rays_buffer;

    const int size = kAnyCalibInputSize;
    std::vector<Eigen::Vector2d> img_points;
    std::vector<Eigen::Vector3d> cam_rays;
    img_points.reserve(size * size);
    cam_rays.reserve(size * size);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const size_t i = static_cast<size_t>(y) * size + x;
        // The network predicts one ray per pixel; reference pixel centers.
        img_points.push_back(
            input.ImgToOrig(Eigen::Vector2d(x + 0.5, y + 0.5)));
        cam_rays.push_back(input.CamToOrig(Eigen::Vector3d(
            rays_data[3 * i], rays_data[3 * i + 1], rays_data[3 * i + 2])));
      }
    }
    aicore_anycalib_free_buffer(rays_buffer);
    aicore_anycalib_free_buffer(tangent_buffer);

    std::vector<double> prior_focal_lengths;
    if (options_.anycalib->fitting.prior_focal_length_weight > 0.0 &&
        camera->HasPriorFocalLength() && camera->IsPerspective()) {
      if (CameraModelFocalLengthIdxs(model_id).size() == 1) {
        prior_focal_lengths.push_back(camera->MeanFocalLength());
      } else {
        prior_focal_lengths = {camera->FocalLengthX(), camera->FocalLengthY()};
      }
    }

    const FittedCamera fitted = FitCameraFromRays(model_id,
                                                  img_points,
                                                  cam_rays,
                                                  options_.anycalib->fitting,
                                                  prior_focal_lengths);
    if (!fitted.success) {
      return false;
    }
    Camera calibrated;
    calibrated.SetCameraId(camera->CameraId());
    calibrated.SetModelId(model_id);
    calibrated.SetWidth(bitmap.Width());
    calibrated.SetHeight(bitmap.Height());
    calibrated.Params() = fitted.params;
    // Upstream parity (f1ea91d0): single-view calibration output.
    calibrated.SetSource(CameraSource::SINGLE_VIEW);
    if (!IsValidCalibration(calibrated)) {
      return false;
    }
    // Reject implausible predictions, which are numerically well behaved but
    // far from the true intrinsics. NOTE: this is only checked per image,
    // because the coefficient-wise median of calibrations that all satisfy
    // these per-coefficient bounds satisfies them as well.
    if (calibrated.HasBogusParams(options_.min_focal_length_ratio,
                                  options_.max_focal_length_ratio,
                                  options_.max_extra_param)) {
      // Log the rejected parameters: a systematic rejection here (e.g. by
      // `max_extra_param` for high-order models) would otherwise silently fall
      // back to the existing intrinsics for every image.
      LOG(WARNING) << "Rejecting implausible learned calibration for "
                   << calibrated.ModelName()
                   << " camera: " << calibrated.ParamsToString();
      return false;
    }
    *camera = calibrated;
    return true;
  }

 private:
  SingleViewCalibrationOptions options_;
  aicore_anycalib_ctx* ctx_ = nullptr;
};

#endif

}  // namespace

bool AnyCalibOptions::Check() const {
  // NOTE: `model_path` is intentionally not validated here: like the
  // feature extractor model paths, it may be empty until set, and a missing
  // file surfaces as an exception when the calibrator is created.
  CHECK_OPTION(fitting.Check());
  return true;
}

Eigen::Vector2d AnyCalibInput::ImgToOrig(const Eigen::Vector2d& point) const {
  Eigen::Vector2d upright = (point - shift_xy).cwiseQuotient(scale_xy);
  switch (image_rot90) {
    case 0:
      return upright;
    case 1:
      return Eigen::Vector2d(upright_height - upright.y(), upright.x());
    case 2:
      return Eigen::Vector2d(upright_width - upright.x(),
                             upright_height - upright.y());
    case 3:
      return Eigen::Vector2d(upright.y(), upright_width - upright.x());
    default:
      break;
  }
  LOG_FATAL_THROW(std::invalid_argument) << "Invalid image rotation: " << image_rot90;
  return Eigen::Vector2d::Zero();
}

Eigen::Vector3d AnyCalibInput::CamToOrig(const Eigen::Vector3d& ray) const {
  switch (image_rot90) {
    case 0:
      return ray;
    case 1:
      return Eigen::Vector3d(-ray.y(), ray.x(), ray.z());
    case 2:
      return Eigen::Vector3d(-ray.x(), -ray.y(), ray.z());
    case 3:
      return Eigen::Vector3d(ray.y(), -ray.x(), ray.z());
    default:
      break;
  }
  LOG_FATAL_THROW(std::invalid_argument) << "Invalid image rotation: " << image_rot90;
  return Eigen::Vector3d::Zero();
}

AnyCalibInput PrepareAnyCalibInput(const Bitmap& bitmap,
                                   const PosePrior& pose_prior) {
  THROW_CHECK(bitmap.IsRGB());
  THROW_CHECK_GT(bitmap.Width(), 0);
  THROW_CHECK_GT(bitmap.Height(), 0);
  constexpr int kSize = kAnyCalibInputSize;
  AnyCalibInput input;
  Eigen::Vector2d& scale_xy = input.scale_xy;
  Eigen::Vector2d& shift_xy = input.shift_xy;

  Bitmap image = bitmap.Clone();
  input.image_rot90 =
      pose_prior.HasGravity() ? ComputeRot90FromGravity(pose_prior.gravity) : 0;
  if (input.image_rot90 != 0) {
    image.Rot90(input.image_rot90);
  }
  input.upright_width = image.Width();
  input.upright_height = image.Height();

  // Upsample small images, preserving the aspect ratio (set_im_size step 1).
  UpsampleToMinSize(image, kSize, scale_xy);

  // Center-crop to square (set_im_size step 2 with target aspect ratio 1).
  // Separate fixed landscape and portrait exports preserve more image
  // content, but the landscape export consistently reduced reconstruction AUC
  // relative to this square model on ETH3D DSLR. That dataset did not
  // exercise the portrait export, so keep the empirically stronger single
  // square model.
  // NOTE: upstream bicubic resampling is replaced with COLMAP's bilinear
  // `Bitmap::Rescale`; the network is robust to the resampling kernel.
  const Eigen::Vector2i crop_origin =
      CenterCropOrigin(image.Width(), image.Height());
  const int side = std::min(image.Width(), image.Height());
  // Cropping translates image coordinates by the negative crop origin.
  shift_xy = Eigen::Vector2d(-crop_origin.x(), -crop_origin.y());
  image.Crop(crop_origin.x(), crop_origin.y(), side, side);

  // Downsample to the network resolution (set_im_size step 3).
  const Eigen::Vector2d scale_2(static_cast<double>(kSize) / image.Width(),
                                static_cast<double>(kSize) / image.Height());
  image.Rescale(kSize, kSize);
  scale_xy = scale_xy.cwiseProduct(scale_2);
  shift_xy = shift_xy.cwiseProduct(scale_2);

  // Convert to row-major [C, H, W] float tensor, normalized to [0, 1].
  const int num_pixels = kSize * kSize;
  input.data.resize(3 * num_pixels);
  const std::vector<uint8_t>& data = image.RowMajorData();
  const int pitch = image.Pitch();
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      for (int c = 0; c < 3; ++c) {
        constexpr float kImageNormalization = 1.0f / 255.0f;
        input.data[c * num_pixels + y * kSize + x] =
            kImageNormalization * data[y * pitch + 3 * x + c];
      }
    }
  }
  return input;
}

std::unique_ptr<SingleViewCalibrator> CreateAnyCalibCalibrator(
    const SingleViewCalibrationOptions& options) {
#ifdef AICore_ENABLED
  // Validate before constructing: the constructor dereferences
  // `options.anycalib` and resolves/loads the GGUF model, so invalid options
  // must be rejected before any of that happens.
  THROW_CHECK(options.Check());
  return std::make_unique<AnyCalibCalibrator>(options);
#else
  throw std::runtime_error("AnyCalib calibration requires AICore support.");
#endif
}

}  // namespace colmap
