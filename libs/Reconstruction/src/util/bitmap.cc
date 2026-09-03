#include "util/bitmap.h"

#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <regex>
#include <unordered_map>

#include "VLFeat/imopv.h"
#include "base/camera_database.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"

namespace colmap {

struct Bitmap::Storage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;
  std::unordered_map<std::string, std::string> metadata;
};

namespace {
std::string MetadataKey(const BitmapMetadataModel model, const std::string& name) {
  if (model == BitmapMetadataModel::kExif) return "Exif:" + name;
  if (model == BitmapMetadataModel::kGps) return "Exif:" + name;
  return name;
}
void ReadMetadata(const OIIO::ImageSpec& spec,
                  std::unordered_map<std::string, std::string>* metadata) {
  for (const OIIO::ParamValue& p : spec.extra_attribs) {
    if (p.type() == OIIO::TypeDesc::STRING)
      (*metadata)[p.name().string()] = p.get_string();
  }
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

bool Bitmap::Read(const std::string& path, const bool as_rgb) {
  if (!ExistsFile(path)) return false;
  auto input = OIIO::ImageInput::open(path);
  if (!input) return false;
  const OIIO::ImageSpec spec = input->spec(); if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0) { input->close(); return false; }
  auto storage = std::make_unique<Storage>(); storage->width = spec.width; storage->height = spec.height; storage->channels = as_rgb ? 3 : 1;
  std::vector<uint8_t> src(static_cast<size_t>(spec.width) * spec.height * spec.nchannels); storage->pixels.resize(static_cast<size_t>(spec.width) * spec.height * storage->channels);
  if (!input->read_image(OIIO::TypeDesc::UINT8, src.data())) { input->close(); return false; } input->close(); ReadMetadata(spec, &storage->metadata);
  for (int y = 0; y < spec.height; ++y) for (int x = 0; x < spec.width; ++x) { const uint8_t* s = &src[(static_cast<size_t>(y) * spec.width + x) * spec.nchannels]; uint8_t* d = &storage->pixels[(static_cast<size_t>(y) * spec.width + x) * storage->channels]; d[0] = s[0]; if (as_rgb) { d[1] = spec.nchannels > 1 ? s[1] : s[0]; d[2] = spec.nchannels > 2 ? s[2] : s[0]; } }
  data_ = std::move(storage); width_ = spec.width; height_ = spec.height; channels_ = data_->channels; return true;
}
bool Bitmap::Write(const std::string& path, const BitmapFormat format, const int flags) const {
  if (!data_) return false;
  std::string filename = path;
  if (format != BitmapFormat::kUnknown &&
      std::filesystem::path(path).extension().empty()) {
    filename += format == BitmapFormat::kJpeg
                    ? ".jpg"
                    : format == BitmapFormat::kTiff ? ".tif" : ".png";
  }
  OIIO::ImageSpec spec(width_, height_, channels_, OIIO::TypeDesc::UINT8); if (flags > 0 && format == BitmapFormat::kJpeg) spec.attribute("CompressionQuality", std::min(100, flags)); auto output = OIIO::ImageOutput::create(filename); if (!output || !output->open(filename, spec)) return false; const bool ok = output->write_image(OIIO::TypeDesc::UINT8, data_->pixels.data()); output->close(); return ok;
}
void Bitmap::Smooth(const float sigma_x, const float sigma_y) { if (!data_) return; std::vector<float> in(static_cast<size_t>(width_) * height_), out(in.size()); for (int d = 0; d < channels_; ++d) { for (int y=0;y<height_;++y) for(int x=0;x<width_;++x) in[static_cast<size_t>(y)*width_+x] = data_->pixels[(static_cast<size_t>(y)*width_+x)*channels_+d]; vl_imsmooth_f(out.data(), width_, in.data(), width_, height_, width_, sigma_x, sigma_y); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x) data_->pixels[(static_cast<size_t>(y)*width_+x)*channels_+d] = TruncateCast<float,uint8_t>(out[static_cast<size_t>(y)*width_+x]); } }
void Bitmap::Rescale(const int new_width, const int new_height, const BitmapRescaleFilter filter) { if (!data_ || new_width <= 0 || new_height <= 0) return; const int ow=width_, oh=height_; std::vector<uint8_t> old=std::move(data_->pixels); width_=new_width; height_=new_height; data_->width=new_width; data_->height=new_height; data_->pixels.resize(static_cast<size_t>(new_width)*new_height*channels_); for(int y=0;y<new_height;++y) for(int x=0;x<new_width;++x){ const double sx=(x+.5)*ow/new_width-.5, sy=(y+.5)*oh/new_height-.5; const int ix=std::clamp(static_cast<int>(std::round(sx)),0,ow-1), iy=std::clamp(static_cast<int>(std::round(sy)),0,oh-1); for(int d=0;d<channels_;++d) data_->pixels[(static_cast<size_t>(y)*new_width+x)*channels_+d]=old[(static_cast<size_t>(iy)*ow+ix)*channels_+d]; } (void)filter; }
Bitmap Bitmap::Clone() const { return Bitmap(*this); }
Bitmap Bitmap::CloneAsGrey() const { if (IsGrey()) return Clone(); Bitmap out; out.Allocate(width_,height_,false); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x){ BitmapColor<uint8_t> c; GetPixel(x,y,&c); out.SetPixel(x,y,BitmapColor<uint8_t>(static_cast<uint8_t>(.299*c.r+.587*c.g+.114*c.b))); } return out; }
Bitmap Bitmap::CloneAsRGB() const { if (IsRGB()) return Clone(); Bitmap out; out.Allocate(width_,height_,true); for(int y=0;y<height_;++y) for(int x=0;x<width_;++x){ BitmapColor<uint8_t> c; GetPixel(x,y,&c); out.SetPixel(x,y,BitmapColor<uint8_t>(c.r,c.r,c.r)); } return out; }
void Bitmap::CloneMetadata(Bitmap* target) const { CHECK_NOTNULL(target); if (target->data_) target->data_->metadata = data_ ? data_->metadata : std::unordered_map<std::string,std::string>(); }
bool Bitmap::ReadExifTag(const BitmapMetadataModel model,const std::string& tag_name,std::string* result) const { if(!data_||!result)return false; auto it=data_->metadata.find(MetadataKey(model,tag_name)); if(it==data_->metadata.end())it=data_->metadata.find(tag_name); if(it==data_->metadata.end()){*result="";return false;}*result=it->second;return true; }
bool Bitmap::ExifCameraModel(std::string* out) const { std::string a,b,c; *out=""; if(!ReadExifTag(BitmapMetadataModel::kMain,"Make",&a)||!ReadExifTag(BitmapMetadataModel::kMain,"Model",&b)||(!ReadExifTag(BitmapMetadataModel::kExif,"FocalLengthIn35mmFilm",&c)&&!ReadExifTag(BitmapMetadataModel::kExif,"FocalLength",&c)))return false;*out=a+"-"+b+"-"+c+"-"+std::to_string(width_)+"x"+std::to_string(height_);return true; }
bool Bitmap::ExifFocalLength(double* focal_length) const { std::string s; std::smatch m; if (ReadExifTag(BitmapMetadataModel::kExif,"FocalLengthIn35mmFilm",&s) && std::regex_search(s,m,std::regex("([0-9.]+)"))) { *focal_length=std::stod(m[1])/35.0*std::max(width_,height_); return *focal_length>0; } if (!ReadExifTag(BitmapMetadataModel::kExif,"FocalLength",&s) || !std::regex_search(s,m,std::regex("([0-9.]+)"))) return false; const double focal_mm=std::stod(m[1]); std::string make,model; double sensor_width; if(ReadExifTag(BitmapMetadataModel::kMain,"Make",&make)&&ReadExifTag(BitmapMetadataModel::kMain,"Model",&model)&&CameraDatabase().QuerySensorWidth(make,model,&sensor_width)){*focal_length=focal_mm/sensor_width*std::max(width_,height_);return true;}return false; }
bool Bitmap::ExifLatitude(double* v) const { std::string s, ref; std::smatch m; if(!ReadExifTag(BitmapMetadataModel::kGps,"GPSLatitude",&s)||!std::regex_search(s,m,std::regex("([0-9.]+):([0-9.]+):([0-9.]+)")))return false;*v=std::stod(m[1])+std::stod(m[2])/60+std::stod(m[3])/3600;if(ReadExifTag(BitmapMetadataModel::kGps,"GPSLatitudeRef",&ref)&&!ref.empty()&&(ref[0]=='S'||ref[0]=='s'))*v=-*v;return true; }
bool Bitmap::ExifLongitude(double* v) const { std::string s, ref; std::smatch m; if(!ReadExifTag(BitmapMetadataModel::kGps,"GPSLongitude",&s)||!std::regex_search(s,m,std::regex("([0-9.]+):([0-9.]+):([0-9.]+)")))return false;*v=std::stod(m[1])+std::stod(m[2])/60+std::stod(m[3])/3600;if(ReadExifTag(BitmapMetadataModel::kGps,"GPSLongitudeRef",&ref)&&!ref.empty()&&(ref[0]=='W'||ref[0]=='w'))*v=-*v;return true; }
bool Bitmap::ExifAltitude(double* v) const { std::string s;std::smatch m;if(!ReadExifTag(BitmapMetadataModel::kGps,"GPSAltitude",&s)||!std::regex_search(s,m,std::regex("([0-9.]+).*?/.*?([0-9.]+)")))return false;*v=std::stod(m[1])/std::stod(m[2]);return true; }

void* Bitmap::Data() { return data_.get(); }
const void* Bitmap::Data() const { return data_.get(); }
int Bitmap::Width() const { return width_; } int Bitmap::Height() const { return height_; } int Bitmap::Channels() const { return channels_; }
unsigned int Bitmap::BitsPerPixel() const { return channels_ * 8; } unsigned int Bitmap::ScanWidth() const { return width_ * channels_; }
bool Bitmap::IsRGB() const { return channels_ == 3; } bool Bitmap::IsGrey() const { return channels_ == 1; }

float JetColormap::Red(const float g){return Base(g-.25f);} float JetColormap::Green(const float g){return Base(g);} float JetColormap::Blue(const float g){return Base(g+.25f);} float JetColormap::Base(const float v){if(v<=.125f)return 0;if(v<=.375f)return Interpolate(2*v-1,0,-.75f,1,-.25f);if(v<=.625f)return 1;if(v<=.87f)return Interpolate(2*v-1,1,.25f,0,.75f);return 0;} float JetColormap::Interpolate(const float v,const float y0,const float x0,const float y1,const float x1){return(v-x0)*(y1-y0)/(x1-x0)+y0;}
}  // namespace colmap
