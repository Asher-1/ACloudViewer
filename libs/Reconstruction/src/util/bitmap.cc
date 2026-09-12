#include "util/bitmap.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebufalgo.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <regex>

#include "VLFeat/imopv.h"
#include "sensor/database.h"
#include "util/logging.h"
#include "util/math.h"
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
  const OIIO::ParamValue* attribute = FindMetadata(image_spec, name);
  if (!attribute || attribute->nvalues() < 3) return false;
  for (int i = 0; i < 3; ++i) value[i] = attribute->get_float_indexed(i);
  return true;
}
}  // namespace

Bitmap::Bitmap() : data_(nullptr), width_(0), height_(0), channels_(0) {}
Bitmap::~Bitmap() = default;
Bitmap::Bitmap(const Bitmap& other) : Bitmap() {
  if (other.data_) data_ = std::make_unique<Storage>(*other.data_);
  width_ = other.width_; height_ = other.height_; channels_ = other.channels_;
}
Bitmap::Bitmap(Bitmap&& other) noexcept
    : data_(std::move(other.data_)), width_(other.width_), height_(other.height_), channels_(other.channels_) {
  other.width_ = other.height_ = other.channels_ = 0;
}
Bitmap& Bitmap::operator=(const Bitmap& other) { if (this != &other) { Bitmap copy(other); *this = std::move(copy); } return *this; }
Bitmap& Bitmap::operator=(Bitmap&& other) noexcept {
  if (this != &other) { data_ = std::move(other.data_); width_ = other.width_; height_ = other.height_; channels_ = other.channels_; other.width_ = other.height_ = other.channels_ = 0; }
  return *this;
}

bool Bitmap::Allocate(const int width, const int height, const bool as_rgb) {
  if (width <= 0 || height <= 0) return false;
  data_ = std::make_unique<Storage>(); data_->width = width; data_->height = height; data_->channels = as_rgb ? 3 : 1;
  data_->pixels.resize(static_cast<size_t>(width) * height * data_->channels);
  data_->image_spec = OIIO::ImageSpec(width, height, data_->channels,
                                      OIIO::TypeDesc::UINT8);
  width_ = width; height_ = height; channels_ = data_->channels; return true;
}
void Bitmap::Deallocate() { data_.reset(); width_ = height_ = channels_ = 0; }
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
  const uint8_t* p = &data_->pixels[(static_cast<size_t>(y) * width_ + x) * channels_]; color->r = p[0];
  if (channels_ == 3) { color->g = p[1]; color->b = p[2]; } return true;
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

bool Bitmap::Read(const std::filesystem::path& path, const bool as_rgb) {
  if (!ExistsFile(path)) return false;
  OIIO::ImageSpec config;
  config["oiio:reorient"] = 0;
  config["oiio:UnassociatedAlpha"] = 1;
  auto input = OIIO::ImageInput::open(path.string(), &config);
  if (!input) return false;
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
  data_ = std::move(storage); width_ = spec.width; height_ = spec.height; channels_ = data_->channels; return true;
}
bool Bitmap::Write(const std::filesystem::path& path, const BitmapFormat format, const int flags) const {
  if (!data_) return false;
  std::string filename = path.string();
  if (format != BitmapFormat::kUnknown &&
      path.extension().empty()) {
    filename += format == BitmapFormat::kJpeg
                    ? ".jpg"
                    : format == BitmapFormat::kTiff ? ".tif" : ".png";
  }
  OIIO::ImageSpec spec = data_->image_spec;
  spec.width = width_;
  spec.height = height_;
  spec.nchannels = channels_;
  spec.format = OIIO::TypeDesc::UINT8;
  if (flags > 0 && format == BitmapFormat::kJpeg)
    spec.attribute("Compression", "jpeg:" + std::to_string(std::min(100, flags)));
  auto output = OIIO::ImageOutput::create(filename);
  if (!output || !output->open(filename, spec)) return false;
  const bool ok = output->write_image(OIIO::TypeDesc::UINT8, data_->pixels.data()) && output->close();
  return ok;
}
void Bitmap::Smooth(const float sigma_x, const float sigma_y) { if (!data_) return; std::vector<float> in(static_cast<size_t>(width_) * height_), out(in.size()); for (int d = 0; d < channels_; ++d) { for (int y=0;y<height_;++y) for(int x=0;x<width_;++x) in[static_cast<size_t>(y)*width_+x] = data_->pixels[(static_cast<size_t>(y)*width_+x)*channels_+d]; vl_imsmooth_f(out.data(), width_, in.data(), width_, height_, width_, sigma_x, sigma_y); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x) data_->pixels[(static_cast<size_t>(y)*width_+x)*channels_+d] = TruncateCast<float,uint8_t>(out[static_cast<size_t>(y)*width_+x]); } }
void Bitmap::Rescale(const int new_width, const int new_height,
                     const BitmapRescaleFilter filter) {
  if (!data_ || new_width <= 0 || new_height <= 0) return;
  const OIIO::ImageBuf source(
      OIIO::ImageSpec(width_, height_, channels_, OIIO::TypeDesc::UINT8),
      data_->pixels.data());
  std::vector<uint8_t> resized(
      static_cast<size_t>(new_width) * new_height * channels_);
  OIIO::ImageBuf destination(
      OIIO::ImageSpec(new_width, new_height, channels_, OIIO::TypeDesc::UINT8),
      resized.data());
  const char* filter_name = filter == BitmapRescaleFilter::kBox ? "box" : "triangle";
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
Bitmap Bitmap::CloneAsGrey() const { if (IsGrey()) return Clone(); Bitmap out; out.Allocate(width_,height_,false); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x){ BitmapColor<uint8_t> c; GetPixel(x,y,&c); out.SetPixel(x,y,BitmapColor<uint8_t>(static_cast<uint8_t>(.299*c.r+.587*c.g+.114*c.b))); } return out; }
Bitmap Bitmap::CloneAsRGB() const { if (IsRGB()) return Clone(); Bitmap out; out.Allocate(width_,height_,true); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x){ BitmapColor<uint8_t> c; GetPixel(x,y,&c); out.SetPixel(x,y,BitmapColor<uint8_t>(c.r,c.r,c.r)); } return out; }
void Bitmap::CloneMetadata(Bitmap* target) const {
  CHECK_NOTNULL(target);
  if (target->data_ && data_) target->data_->image_spec = data_->image_spec;
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
}  // namespace colmap
