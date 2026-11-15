#include "sensor/bitmap.h"

#include <OpenImageIO/color.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <regex>
#include <string>

#include "VLFeat/imopv.h"
#include "sensor/database.h"
#include "util/logging.h"
#include "math/math.h"
#include "util/misc.h"

namespace colmap {

struct Bitmap::Storage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;
  // Keep the native image specification so numeric EXIF and GPS attributes
  // retain their type instead of being lossy-converted to strings.
  OIIO::ImageSpec image_spec;
};

namespace {
std::string MetadataKey(const BitmapMetadataModel model, const std::string& name) {
  if (model == BitmapMetadataModel::kExif) return "Exif:" + name;
  if (model == BitmapMetadataModel::kGps) return "GPS:" + name;
  return name;
}
const OIIO::ParamValue* FindMetadata(const OIIO::ImageSpec& image_spec,
                                     const std::string& name) {
  return image_spec.find_attribute(name);
}
bool GetFloatMetadata(const OIIO::ImageSpec& image_spec,
                      const std::string& name,
                      float* value) {
  const OIIO::ParamValue* attribute = FindMetadata(image_spec, name);
  if (!attribute) return false;
  *value = attribute->get_float();
  return true;
}
bool GetIntMetadata(const OIIO::ImageSpec& image_spec,
                    const std::string& name,
                    int* value) {
  const OIIO::ParamValue* attribute = FindMetadata(image_spec, name);
  if (!attribute) return false;
  *value = attribute->get_int();
  return true;
}
bool GetPointMetadata(const OIIO::ImageSpec& image_spec,
                      const std::string& name,
                      float value[3]) {
  // Upstream parity (d3ccaf35): read through the typed getattribute() so
  // OIIO converts the stored data (e.g. a GPS float[3] array, whose
  // TypeDesc::aggregate is 1 - only the arraylen encodes the component
  // count - so an aggregate-based check wrongly rejects it and the GPS
  // position was lost, fixed 2026-09-17).
  return image_spec.getattribute(
      name, OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, 3), value);
}
void SetImageSpecColorSpace(OIIO::ImageSpec& image_spec,
                            const std::string& colorspace) {
  if (colorspace.empty()) {
    image_spec.erase_attribute("oiio:ColorSpace");
  } else {
    image_spec.attribute("oiio:ColorSpace", colorspace);
  }
}

bool IsLinearColorSpace(const std::string& colorspace) {
  return colorspace == "linear" || colorspace == "lin_srgb" ||
         colorspace == "lin_rec709P";
}

std::vector<uint8_t> ConvertColorSpace(const uint8_t* src_data,
                                       const int width,
                                       const int height,
                                       const int channels,
                                       const std::string& from,
                                       const std::string& to) {
  const OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::UINT8);
  OIIO::ImageBuf source(spec, const_cast<uint8_t*>(src_data));
  std::vector<uint8_t> target_data(
      static_cast<size_t>(width) * height * channels);
  OIIO::ImageBuf target(spec, target_data.data());
  THROW_CHECK(OIIO::ImageBufAlgo::colorconvert(target, source, from, to));
  return target_data;
}
}  // namespace

Bitmap::Bitmap() : data_(nullptr), width_(0), height_(0), channels_(0) {}
Bitmap::~Bitmap() = default;
Bitmap::Bitmap(const Bitmap& other) : Bitmap() {
  if (other.data_) data_ = std::make_unique<Storage>(*other.data_);
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  linear_colorspace_ = other.linear_colorspace_;
}
Bitmap::Bitmap(Bitmap&& other) noexcept
    : data_(std::move(other.data_)),
      width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      linear_colorspace_(other.linear_colorspace_) {
  other.width_ = other.height_ = other.channels_ = 0;
  other.linear_colorspace_ = false;
}
Bitmap& Bitmap::operator=(const Bitmap& other) {
  if (this != &other) {
    Bitmap copy(other);
    *this = std::move(copy);
  }
  return *this;
}
Bitmap& Bitmap::operator=(Bitmap&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    linear_colorspace_ = other.linear_colorspace_;
    other.width_ = other.height_ = other.channels_ = 0;
    other.linear_colorspace_ = false;
  }
  return *this;
}

Bitmap::Bitmap(const int width,
               const int height,
               const bool as_rgb,
               const bool linear_colorspace) {
  THROW_CHECK(Allocate(width, height, as_rgb))
      << "Failed to allocate bitmap of size " << width << "x" << height;
  linear_colorspace_ = linear_colorspace;
  SetImageSpecColorSpace(data_->image_spec,
                         linear_colorspace_ ? "linear" : "sRGB");
}

bool Bitmap::Allocate(const int width, const int height, const bool as_rgb) {
  if (width <= 0 || height <= 0) return false;
  data_ = std::make_unique<Storage>(); data_->width = width; data_->height = height; data_->channels = as_rgb ? 3 : 1;
  data_->pixels.resize(static_cast<size_t>(width) * height * data_->channels);
  data_->image_spec = OIIO::ImageSpec(width, height, data_->channels,
                                      OIIO::TypeDesc::UINT8);
  width_ = width;
  height_ = height;
  channels_ = data_->channels;
  linear_colorspace_ = false;
  SetImageSpecColorSpace(data_->image_spec, "sRGB");
  return true;
}
void Bitmap::Deallocate() {
  data_.reset();
  width_ = height_ = channels_ = 0;
  linear_colorspace_ = false;
}
size_t Bitmap::NumBytes() const { return data_ ? data_->pixels.size() : 0; }
std::vector<uint8_t> Bitmap::ConvertToRawBits() const { return ConvertToRowMajorArray(); }
std::vector<uint8_t> Bitmap::ConvertToRowMajorArray() const { return data_ ? data_->pixels : std::vector<uint8_t>(); }
std::vector<uint8_t> Bitmap::ConvertToColMajorArray() const {
  std::vector<uint8_t> out(static_cast<size_t>(width_) * height_ * channels_); size_t i = 0;
  for (int d = 0; d < channels_; ++d) for (int x = 0; x < width_; ++x) for (int y = 0; y < height_; ++y)
    out[i++] = data_->pixels[(static_cast<size_t>(y) * width_ + x) * channels_ + d];
  return out;
}
bool Bitmap::GetPixel(const int x, const int y, BitmapColor<uint8_t>* color) const {
  if (!data_ || !color || x < 0 || x >= width_ || y < 0 || y >= height_) return false;
  const uint8_t* p =
      &data_->pixels[(static_cast<size_t>(y) * width_ + x) * channels_];
  color->r = p[0];
  if (channels_ == 3) {
    color->g = p[1];
    color->b = p[2];
  } else {
    color->g = p[0];
    color->b = p[0];
  }
  return true;
}
std::optional<BitmapColor<uint8_t>> Bitmap::GetPixel(const int x,
                                                     const int y) const {
  BitmapColor<uint8_t> color;
  if (GetPixel(x, y, &color)) {
    return color;
  }
  return std::nullopt;
}
bool Bitmap::SetPixel(const int x, const int y, const BitmapColor<uint8_t>& color) {
  if (!data_ || x < 0 || x >= width_ || y < 0 || y >= height_) return false;
  uint8_t* p = &data_->pixels[(static_cast<size_t>(y) * width_ + x) * channels_]; p[0] = color.r;
  if (channels_ == 3) { p[1] = color.g; p[2] = color.b; } return true;
}
const uint8_t* Bitmap::GetScanline(const int y) const { CHECK_GE(y, 0); CHECK_LT(y, height_); return &data_->pixels[static_cast<size_t>(y) * width_ * channels_]; }
void Bitmap::Fill(const BitmapColor<uint8_t>& color) { if (data_) for (int y = 0; y < height_; ++y) for (int x = 0; x < width_; ++x) SetPixel(x, y, color); }
bool Bitmap::InterpolateNearestNeighbor(const double x, const double y, BitmapColor<uint8_t>* color) const { return GetPixel(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)), color); }
bool Bitmap::InterpolateBilinear(const double x, const double y, BitmapColor<float>* color) const {
  const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y)), x1 = x0 + 1, y1 = y0 + 1;
  if (!data_ || !color || x0 < 0 || y0 < 0 || x1 >= width_ || y1 >= height_) return false;
  const double dx = x - x0, dy = y - y0;
  for (int d = 0; d < 3; ++d) { auto s = [&](int xx, int yy) { return static_cast<double>(data_->pixels[(static_cast<size_t>(yy) * width_ + xx) * channels_ + (channels_ == 1 ? 0 : d)]); }; const double v = (1 - dy) * ((1 - dx) * s(x0, y0) + dx * s(x1, y0)) + dy * ((1 - dx) * s(x0, y1) + dx * s(x1, y1)); if (d == 0) color->r = v; else if (d == 1) color->g = v; else color->b = v; }
  return true;
}

bool Bitmap::Read(const std::filesystem::path& path,
                  const bool as_rgb,
                  const bool linearize_colorspace) {
  if (!ExistsFile(path)) return false;
  OIIO::ImageSpec config;
  config["oiio:reorient"] = 0;
  config["oiio:UnassociatedAlpha"] = 1;
  auto input = OIIO::ImageInput::open(path.string(), &config);
  if (!input) {
    // OIIO stores failures in thread-local pending state; consume it before
    // returning so a later operation does not observe this stale error.
    static_cast<void>(OIIO::geterror());
    return false;
  }
  const OIIO::ImageSpec spec = input->spec();
  if (spec.width <= 0 || spec.height <= 0 ||
      (spec.nchannels != 1 && spec.nchannels != 2 &&
       spec.nchannels != 3 && spec.nchannels != 4)) {
    input->close();
    return false;
  }
  const int file_channels = spec.nchannels == 4 ? 3 :
                            spec.nchannels == 2 ? 1 : spec.nchannels;
  auto storage = std::make_unique<Storage>();
  storage->width = spec.width;
  storage->height = spec.height;
  storage->channels = file_channels;
  storage->pixels.resize(static_cast<size_t>(spec.width) * spec.height * file_channels);
  if (!input->read_image(0, 0, 0, file_channels, OIIO::TypeDesc::UINT8,
                         storage->pixels.data())) {
    input->close();
    return false;
  }
  input->close();
  storage->image_spec = spec;
  storage->image_spec.nchannels = file_channels;
  data_ = std::move(storage);
  width_ = spec.width;
  height_ = spec.height;
  channels_ = data_->channels;
  const std::string colorspace =
      data_->image_spec.get_string_attribute("oiio:ColorSpace");
  linear_colorspace_ = linearize_colorspace;
  if (linearize_colorspace && !colorspace.empty() &&
      !IsLinearColorSpace(colorspace)) {
    data_->pixels = ConvertColorSpace(data_->pixels.data(),
                                      width_,
                                      height_,
                                      channels_,
                                      colorspace,
                                      "linear");
    SetImageSpecColorSpace(data_->image_spec, "linear");
  }
  // Upstream parity: honor the as_rgb request (the patch-match workspace
  // depends on as_rgb=false producing a grey bitmap).
  if (as_rgb && channels_ != 3) {
    *this = CloneAsRGB();
  } else if (!as_rgb && channels_ != 1) {
    *this = CloneAsGrey();
  }
  return true;
}
bool Bitmap::Write(const std::filesystem::path& path,
                   const bool delinearize_colorspace) const {
  if (!data_) return false;
  const std::string filename = path.string();
  OIIO::ImageSpec spec = data_->image_spec;
  spec.width = width_;
  spec.height = height_;
  spec.nchannels = channels_;
  spec.format = OIIO::TypeDesc::UINT8;
  const std::string extension = path.extension().string();
  const bool is_jpeg = extension == ".jpg" || extension == ".jpeg" ||
                       extension == ".JPG" || extension == ".JPEG";
  // Upstream parity (sensor/bitmap.cc "Save JPEG in superb quality by
  // default"): an unset quality (-1) must not fall back to the OIIO encoder
  // default (75) but to superb quality 100, otherwise undistorted images
  // are written visibly smaller and noisier than upstream.
  if (is_jpeg) {
    const int quality = jpeg_quality_ > 0 ? std::min(100, jpeg_quality_) : 100;
    spec.attribute("Compression", "jpeg:" + std::to_string(quality));
  }
  const uint8_t* output_data = data_->pixels.data();
  std::vector<uint8_t> converted_output;
  if (delinearize_colorspace && linear_colorspace_) {
    std::string target_colorspace =
        spec.get_string_attribute("oiio:ColorSpace");
    if (target_colorspace.empty()) {
      target_colorspace = "sRGB";
      SetImageSpecColorSpace(spec, target_colorspace);
    }
    if (!IsLinearColorSpace(target_colorspace)) {
      converted_output = ConvertColorSpace(data_->pixels.data(),
                                           width_,
                                           height_,
                                           channels_,
                                           "linear",
                                           target_colorspace);
      output_data = converted_output.data();
    }
  }
  auto output = OIIO::ImageOutput::create(filename);
  if (!output || !output->open(filename, spec)) return false;
  const bool ok = output->write_image(OIIO::TypeDesc::UINT8, output_data) &&
                  output->close();
  return ok;
}
void Bitmap::Smooth(const float sigma_x, const float sigma_y) { if (!data_) return; std::vector<float> in(static_cast<size_t>(width_) * height_), out(in.size()); for (int d = 0; d < channels_; ++d) { for (int y=0;y<height_;++y) for(int x=0;x<width_;++x) in[static_cast<size_t>(y)*width_+x] = data_->pixels[(static_cast<size_t>(y)*width_+x)*channels_+d]; vl_imsmooth_f(out.data(), width_, in.data(), width_, height_, width_, sigma_x, sigma_y); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x) data_->pixels[(static_cast<size_t>(y)*width_+x)*channels_+d] = TruncateCast<float,uint8_t>(out[static_cast<size_t>(y)*width_+x]); } }
void Bitmap::Rescale(const int new_width,
                     const int new_height,
                     const RescaleFilter filter) {
  if (!data_ || new_width <= 0 || new_height <= 0) return;
  const OIIO::ImageBuf source(
      OIIO::ImageSpec(width_, height_, channels_, OIIO::TypeDesc::UINT8),
      data_->pixels.data());
  std::vector<uint8_t> resized(
      static_cast<size_t>(new_width) * new_height * channels_);
  OIIO::ImageBuf destination(
      OIIO::ImageSpec(new_width, new_height, channels_, OIIO::TypeDesc::UINT8),
      resized.data());
  const char* filter_name =
      filter == RescaleFilter::kBox ? "box" : "triangle";
#if defined(OIIO_VERSION_MAJOR) && OIIO_VERSION_MAJOR >= 3
  const bool ok = OIIO::ImageBufAlgo::resize(
      destination, source, {{"filtername", filter_name}});
#else
  const bool ok = OIIO::ImageBufAlgo::resize(
      destination, source, filter_name, 0.0f);
#endif
  if (!ok) return;
  width_ = new_width;
  height_ = new_height;
  data_->width = new_width;
  data_->height = new_height;
  data_->pixels = std::move(resized);
  data_->image_spec.width = new_width;
  data_->image_spec.height = new_height;
}
Bitmap Bitmap::Clone() const { return Bitmap(*this); }
Bitmap Bitmap::CloneAsGrey() const { if (IsGrey()) return Clone(); Bitmap out; out.Allocate(width_,height_,false); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x){ BitmapColor<uint8_t> c; GetPixel(x,y,&c); // Upstream parity (d3ccaf35 sensor/bitmap.cc): BT.709 weights with an
 // added 0.5 before truncation (equivalent to std::round). The old fork
 // used BT.601 weights (.299/.587/.114) with plain truncation, shifting
 // every grey value by up to tens of counts and perturbing the SIFT
 // DoG responses downstream.
 out.SetPixel(x,y,BitmapColor<uint8_t>(static_cast<uint8_t>(.2126f*c.r+.7152f*c.g+.0722f*c.b+.5f))); } if (data_ && out.data_) out.data_->image_spec = data_->image_spec; return out; }
Bitmap Bitmap::CloneAsRGB() const { if (IsRGB()) return Clone(); Bitmap out; out.Allocate(width_,height_,true); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x){ BitmapColor<uint8_t> c; GetPixel(x,y,&c); out.SetPixel(x,y,BitmapColor<uint8_t>(c.r,c.r,c.r)); } if (data_ && out.data_) out.data_->image_spec = data_->image_spec; return out; }
void Bitmap::SetMetaData(const std::string_view name,
                         const std::string_view type,
                         const void* value) {
  THROW_CHECK(data_ != nullptr);
  THROW_CHECK_NE(type, "string");
  OIIO::TypeDesc type_desc;
  type_desc.fromstring(std::string(type));
  THROW_CHECK_NE(type_desc, OIIO::TypeDesc::UNKNOWN);
  data_->image_spec.attribute(std::string(name), type_desc, value);
}

void Bitmap::SetMetaData(const std::string_view name,
                         const std::string_view value) {
  THROW_CHECK(data_ != nullptr);
  data_->image_spec.attribute(std::string(name), std::string(value));
}

bool Bitmap::GetMetaData(const std::string_view name,
                         const std::string_view type,
                         void* value) const {
  if (!data_) return false;
  THROW_CHECK_NE(type, "string");
  OIIO::TypeDesc type_desc;
  type_desc.fromstring(std::string(type));
  THROW_CHECK_NE(type_desc, OIIO::TypeDesc::UNKNOWN);
  return data_->image_spec.getattribute(std::string(name), type_desc, value);
}

std::optional<std::string> Bitmap::GetMetaData(
    const std::string_view name) const {
  if (!data_) return std::nullopt;
  const OIIO::ParamValue* attribute =
      data_->image_spec.find_attribute(std::string(name));
  if (!attribute) return std::nullopt;
  return attribute->get_string();
}

void Bitmap::CloneMetadata(Bitmap* target) const {
  CHECK_NOTNULL(target);
  if (!target->data_ || !data_) return;
  target->data_->image_spec = data_->image_spec;
  target->data_->image_spec.width = target->width_;
  target->data_->image_spec.height = target->height_;
  target->data_->image_spec.nchannels = target->channels_;
}
bool Bitmap::ReadExifTag(const BitmapMetadataModel model, const std::string& tag_name,
                         std::string* result) const {
  if (!data_ || !result) return false;
  const std::string key = MetadataKey(model, tag_name);
  const OIIO::ParamValue* attribute = data_->image_spec.find_attribute(key);
  if (!attribute) attribute = data_->image_spec.find_attribute(tag_name);
  if (!attribute) { result->clear(); return false; }
  *result = attribute->get_string();
  return true;
}
bool Bitmap::ExifCameraModel(std::string* out) const {
  if (!out || !data_) return false;
  std::string make, model;
  float focal = 0.0f;
  if (!ReadExifTag(BitmapMetadataModel::kMain, "Make", &make) ||
      !ReadExifTag(BitmapMetadataModel::kMain, "Model", &model) ||
      (!GetFloatMetadata(data_->image_spec, "Exif:FocalLengthIn35mmFilm", &focal) &&
       !GetFloatMetadata(data_->image_spec, "Exif:FocalLength", &focal))) return false;
  *out = make + "-" + model + "-" + std::to_string(focal) + "-" +
         std::to_string(width_) + "x" + std::to_string(height_);
  return true;
}
bool Bitmap::ExifFocalLength(double* focal_length) const {
  if (!data_ || !focal_length) return false;
  float focal_35 = 0.0f;
  if (GetFloatMetadata(data_->image_spec, "Exif:FocalLengthIn35mmFilm", &focal_35) && focal_35 > 0) {
    *focal_length = focal_35 / 43.27 * std::hypot(width_, height_);
    return true;
  }
  float focal_mm = 0.0f;
  if (!GetFloatMetadata(data_->image_spec, "Exif:FocalLength", &focal_mm) || focal_mm <= 0) return false;
  float resolution = 0.0f;
  int unit = 0;
  if (GetFloatMetadata(data_->image_spec, "Exif:FocalPlaneXResolution", &resolution) &&
      GetIntMetadata(data_->image_spec, "Exif:FocalPlaneResolutionUnit", &unit)) {
    const double scales[] = {0, 0, 1.0 / 25.4, 1.0 / 10.0, 1.0, 1000.0};
    if (unit >= 2 && unit <= 5 && resolution > 0) { *focal_length = focal_mm * resolution * scales[unit]; return true; }
  }
  std::string make, model;
  double sensor_width = 0.0;
  if (ReadExifTag(BitmapMetadataModel::kMain, "Make", &make) &&
      ReadExifTag(BitmapMetadataModel::kMain, "Model", &model) &&
      CameraDatabase().QuerySensorWidth(make, model, &sensor_width)) {
    *focal_length = focal_mm / sensor_width * std::max(width_, height_);
    return true;
  }
  return false;
}
std::optional<int> Bitmap::ExifOrientation() const {
  if (!data_) return std::nullopt;
  int orientation = 0;
  if (!GetIntMetadata(data_->image_spec, "Orientation", &orientation)) {
    return std::nullopt;
  }
  return orientation;
}

bool Bitmap::ExifLatitude(double* value) const {
  if (!data_ || !value) return false;
  float dms[3] = {0, 0, 0};
  if (!GetPointMetadata(data_->image_spec, "GPS:Latitude", dms)) return false;
  *value = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0;
  std::string ref;
  if (ReadExifTag(BitmapMetadataModel::kGps, "LatitudeRef", &ref) &&
      !ref.empty() && (ref[0] == 'S' || ref[0] == 's')) *value = -*value;
  return true;
}
bool Bitmap::ExifLongitude(double* value) const {
  if (!data_ || !value) return false;
  float dms[3] = {0, 0, 0};
  if (!GetPointMetadata(data_->image_spec, "GPS:Longitude", dms)) return false;
  *value = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0;
  std::string ref;
  if (ReadExifTag(BitmapMetadataModel::kGps, "LongitudeRef", &ref) &&
      !ref.empty() && (ref[0] == 'W' || ref[0] == 'w')) *value = -*value;
  return true;
}
bool Bitmap::ExifAltitude(double* value) const {
  if (!data_ || !value) return false;
  float altitude = 0.0f;
  if (!GetFloatMetadata(data_->image_spec, "GPS:Altitude", &altitude)) return false;
  *value = altitude;
  std::string ref;
  if (ReadExifTag(BitmapMetadataModel::kGps, "AltitudeRef", &ref) && ref == "1") *value = -*value;
  return true;
}

void* Bitmap::Data() { return data_.get(); }
const void* Bitmap::Data() const { return data_.get(); }
int Bitmap::Width() const { return width_; } int Bitmap::Height() const { return height_; } int Bitmap::Channels() const { return channels_; }
unsigned int Bitmap::BitsPerPixel() const { return channels_ * 8; } unsigned int Bitmap::ScanWidth() const { return width_ * channels_; }
bool Bitmap::IsRGB() const { return channels_ == 3; } bool Bitmap::IsGrey() const { return channels_ == 1; }

float JetColormap::Red(const float g){return Base(g-.25f);} float JetColormap::Green(const float g){return Base(g);} float JetColormap::Blue(const float g){return Base(g+.25f);} float JetColormap::Base(const float v){if(v<=.125f)return 0;if(v<=.375f)return Interpolate(2*v-1,0,-.75f,1,-.25f);if(v<=.625f)return 1;if(v<=.87f)return Interpolate(2*v-1,1,.25f,0,.75f);return 0;} float JetColormap::Interpolate(const float v,const float y0,const float x0,const float y1,const float x1){return(v-x0)*(y1-y0)/(x1-x0)+y0;}

void Bitmap::Rot90(bool counterclockwise) {
  if (!data_) return;
  const int new_width = height_;
  const int new_height = width_;
  std::vector<uint8_t> rotated(static_cast<size_t>(new_width) * new_height *
                               channels_);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const int nx = counterclockwise ? y : (height_ - 1 - y);
      const int ny = counterclockwise ? (width_ - 1 - x) : x;
      for (int c = 0; c < channels_; ++c) {
        rotated[(static_cast<size_t>(ny) * new_width + nx) * channels_ + c] =
            data_->pixels[(static_cast<size_t>(y) * width_ + x) * channels_ +
                          c];
      }
    }
  }
  data_->pixels = std::move(rotated);
  std::swap(width_, height_);
}

void Bitmap::SetJPEGQuality(const int quality) {
  THROW_CHECK_GT(quality, 0);
  THROW_CHECK_LE(quality, 100);
  jpeg_quality_ = quality;
}

int Bitmap::JPEGQuality() const { return jpeg_quality_; }

int Bitmap::Pitch() const { return width_ * channels_; }

bool Bitmap::IsEmpty() const { return width_ <= 0 || height_ <= 0; }

double Bitmap::Thumbnail(const int max_image_size,
                         const RescaleFilter filter) {
    CHECK_GT(max_image_size, 0);
    if (width_ <= max_image_size && height_ <= max_image_size) {
        return 1.0;
    }
    // Fit the down-sampled version exactly into the max dimensions.
    const double scale =
            static_cast<double>(max_image_size) / std::max(width_, height_);
    Rescale(static_cast<int>(std::round(width_ * scale)),
            static_cast<int>(std::round(height_ * scale)), filter);
    return scale;
}

std::vector<uint8_t>& Bitmap::RowMajorData() { return data_->pixels; }

const std::vector<uint8_t>& Bitmap::RowMajorData() const {
  return data_->pixels;
}

// Upstream parity (d3ccaf35 sensor/bitmap.cc Rot90(int)): rotate by
// `num_quarter_turns` counter-clockwise quarter turns. Delegates to the
// boolean single-turn implementation (already validated by the bitmap
// round-trip tests), always in the counter-clockwise direction so the
// per-turn effects accumulate instead of cancelling.
void Bitmap::Rot90(int num_quarter_turns) {
  if (!data_) return;
  int k = num_quarter_turns % 4;
  if (k < 0) {
    k += 4;
  }
  for (int turn = 0; turn < k; ++turn) {
    Rot90(/*counterclockwise=*/true);
  }
}

// Upstream parity (d3ccaf35 sensor/bitmap.cc Crop): crop the image in place
// to the given rectangle. Cropping to the full image is a no-op.
void Bitmap::Crop(const int left,
                  const int top,
                  const int width,
                  const int height) {
  THROW_CHECK_GE(left, 0);
  THROW_CHECK_GE(top, 0);
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  THROW_CHECK_LE(left + width, width_);
  THROW_CHECK_LE(top + height, height_);
  if (!data_) return;
  if (left == 0 && top == 0 && width == width_ && height == height_) {
    return;
  }

  std::vector<uint8_t> cropped(static_cast<size_t>(width) * height *
                               channels_);
  for (int y = 0; y < height; ++y) {
    const uint8_t* src_row =
        data_->pixels.data() + (static_cast<size_t>(top + y) * width_ + left) *
                                   channels_;
    std::copy(src_row,
              src_row + static_cast<size_t>(width) * channels_,
              cropped.begin() + static_cast<size_t>(y) * width * channels_);
  }
  data_->pixels = std::move(cropped);
  width_ = width;
  height_ = height;
}

}  // namespace colmap
