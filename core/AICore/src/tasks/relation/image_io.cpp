// Pillow-compatible fixed-point resize. The coefficient construction follows
// Pillow's HPND-licensed Resample.c; see third_party/patches/README.md.
// Local precedent: General-Keypoint-Detection-GGML/cpp_ggml/src/image_io.cpp.
#include "relateanything/full_model.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include "common/image_utils.hpp"

namespace relateanything {
namespace {
struct Pixels {
    int width = 0, height = 0;
    std::vector<unsigned char> rgb;
    const unsigned char * borrowed = nullptr;
    std::size_t stride = 0;
    const unsigned char * row(int y) const {
        return borrowed ? borrowed + y * stride : rgb.data() + static_cast<std::size_t>(y) * width * 3;
    }
};
Pixels load_rgb(const std::filesystem::path & path) {
    aicore::img::Image image;
    if (!aicore::img::load_image_rgb(path.string(), image))
        throw std::runtime_error("cannot decode image: " + path.string());
    Pixels out; out.width = image.width; out.height = image.height;
    out.rgb = std::move(image.rgb); return out;
}
struct Axis {
    std::vector<int> start;
    std::vector<std::vector<int>> weights;
};
Axis axis(int src, int dst) {
    Axis out;
    const double scale = double(src) / dst, support = std::max(scale, 1.0);
    for (int i = 0; i < dst; ++i) {
        const double center = (i + .5) * scale;
        const int lo = std::max(0, int(center - support + .5));
        const int hi = std::min(src, int(center + support + .5));
        std::vector<double> raw; double sum = 0;
        for (int j = lo; j < hi; ++j) {
            const double value = std::max(0.0, 1.0 - std::abs((j - center + .5) / support));
            raw.push_back(value); sum += value;
        }
        std::vector<int> weights;
        for (double value : raw) weights.push_back(int(.5 + value / sum * (1 << 22)));
        out.start.push_back(lo); out.weights.push_back(std::move(weights));
    }
    return out;
}
Pixels resize(const Pixels & src) {
    constexpr int size = 448;
    const Axis xs = axis(src.width, size), ys = axis(src.height, size);
    std::vector<unsigned char> tmp(static_cast<std::size_t>(size) * src.height * 3);
    for (int y = 0; y < src.height; ++y) for (int x = 0; x < size; ++x) {
        int a0 = 1 << 21, a1 = a0, a2 = a0;
        for (std::size_t j = 0; j < xs.weights[x].size(); ++j) {
            const auto * pixel = src.row(y) + (xs.start[x] + j) * 3;
            const int weight = xs.weights[x][j];
            a0 += pixel[0] * weight; a1 += pixel[1] * weight; a2 += pixel[2] * weight;
        }
        auto * pixel = tmp.data() + (static_cast<std::size_t>(y) * size + x) * 3;
        pixel[0] = std::clamp(a0 >> 22, 0, 255);
        pixel[1] = std::clamp(a1 >> 22, 0, 255);
        pixel[2] = std::clamp(a2 >> 22, 0, 255);
    }
    Pixels dst{size, size, std::vector<unsigned char>(size * size * 3)};
    std::vector<int> row(size * 3);
    for (int y = 0; y < size; ++y) {
        std::fill(row.begin(), row.end(), 1 << 21);
        for (std::size_t j = 0; j < ys.weights[y].size(); ++j) {
            const auto * source = tmp.data() + (ys.start[y] + j) * size * 3;
            const int weight = ys.weights[y][j];
            // Contiguous row arithmetic allows compiler SIMD while preserving
            // the fixed-point summation and rounding of the PIL reference.
            for (int x = 0; x < size * 3; ++x) row[x] += source[x] * weight;
        }
        auto * target = dst.rgb.data() + y * size * 3;
        for (int x = 0; x < size * 3; ++x)
            target[x] = std::clamp(row[x] >> 22, 0, 255);
    }
    return dst;
}
} // namespace

static ImageInput prepare_pixels(const Pixels & original, const ImageOptions & options) {
    const auto pixels = resize(original);
    ImageInput input;
    input.rgb_chw.resize(3 * 448 * 448);
    constexpr float mean[] = {.485f, .456f, .406f}, deviation[] = {.229f, .224f, .225f};
    for (int c = 0; c < 3; ++c) for (int i = 0; i < 448 * 448; ++i)
        input.rgb_chw[c * 448 * 448 + i] = (pixels.rgb[3 * i + c] / 255.0f - mean[c]) / deviation[c];
    if (options.boxes_xyxy.size() % 4) throw std::runtime_error("boxes must contain four coordinates each");
    input.object_count = options.boxes_xyxy.size() / 4;
    if (!options.box_scores.empty() && options.box_scores.size() != input.object_count)
        throw std::runtime_error("box scores must have one value per box");
    input.box_scores = options.box_scores;
    const std::size_t mask_pixels = static_cast<std::size_t>(original.width) * original.height;
    if (!options.masks.empty()) {
        if (options.masks.size() != input.object_count * mask_pixels)
            throw std::runtime_error("masks must have original-image [N,H,W] dimensions");
        input.coverage.resize(input.object_count * 32 * 32);
        input.fill.resize(input.object_count);
        // Exact disjoint box filter, matching mask_raster's reduceat boundaries.
        for (std::size_t i = 0; i < input.object_count; ++i) {
            const auto * mask = options.masks.data() + i * mask_pixels;
            double area = 0;
            for (std::size_t j = 0; j < mask_pixels; ++j) {
                if (mask[j] > 1) throw std::runtime_error("binary masks require values 0 or 1");
                area += mask[j];
            }
            const auto * b = options.boxes_xyxy.data() + i * 4;
            const float box_area = std::max((b[2] - b[0]) * (b[3] - b[1]), 1.0f);
            input.fill[i] = static_cast<float>(std::clamp(area / box_area, 0.0, 1.0));
            for (int y = 0; y < 32; ++y) for (int x = 0; x < 32; ++x) {
                const int y0 = y * original.height / 32, y1 = (y + 1) * original.height / 32;
                const int x0 = x * original.width / 32, x1 = (x + 1) * original.width / 32;
                double sum = 0;
                for (int py = y0; py < std::max(y0 + 1, y1); ++py)
                    for (int px = x0; px < std::max(x0 + 1, x1); ++px)
                        sum += mask[static_cast<std::size_t>(py) * original.width + px];
                input.coverage[i * 1024 + y * 32 + x] =
                    static_cast<float>(sum / std::max((y1 - y0) * (x1 - x0), 1));
            }
        }
    }
    for (std::size_t i = 0; i < input.object_count; ++i) {
        const float * b = options.boxes_xyxy.data() + i * 4;
        // Match upstream operation order: normalize corners before converting
        // to cxcywh. A one-ULP geometry difference is amplified by Fourier PE.
        const float x1 = b[0] / original.width, y1 = b[1] / original.height;
        const float x2 = b[2] / original.width, y2 = b[3] / original.height;
        input.boxes_cxcywh.insert(input.boxes_cxcywh.end(), {
            (x1 + x2) * .5f, (y1 + y2) * .5f, x2 - x1, y2 - y1});
    }
    return input;
}

ImageInput prepare_image(const ImageOptions & options) {
    return prepare_pixels(load_rgb(options.image), options);
}

ImageInput prepare_image(const ImageView & image, const ImageOptions & regions) {
    if (!image.data || !image.width || !image.height || image.width > 20000 || image.height > 20000)
        throw std::invalid_argument("invalid borrowed image dimensions or data");
    const int channels = image.format == ImageFormat::gray8 ? 1 :
        image.format == ImageFormat::rgb8 || image.format == ImageFormat::bgr8 ? 3 :
        image.format == ImageFormat::rgba8 || image.format == ImageFormat::bgra8 ? 4 : 0;
    if (!channels) throw std::invalid_argument("unsupported borrowed image format");
    const std::size_t row_bytes = image.width * channels;
    if (image.row_stride_bytes < row_bytes ||
        image.row_stride_bytes > (SIZE_MAX - row_bytes) / image.height ||
        image.data_bytes < (image.height - 1) * image.row_stride_bytes + row_bytes)
        throw std::invalid_argument("borrowed image stride or capacity is too small");
    Pixels pixels;
    pixels.width = static_cast<int>(image.width); pixels.height = static_cast<int>(image.height);
    if (image.format == ImageFormat::rgb8) {
        pixels.borrowed = image.data; pixels.stride = image.row_stride_bytes;
    } else {
        pixels.rgb.resize(image.width * image.height * 3);
        const bool bgr = image.format == ImageFormat::bgr8 || image.format == ImageFormat::bgra8;
        for (std::size_t y = 0; y < image.height; ++y) for (std::size_t x = 0; x < image.width; ++x) {
            const auto * source = image.data + y * image.row_stride_bytes + x * channels;
            auto * target = pixels.rgb.data() + (y * image.width + x) * 3;
            for (int c = 0; c < 3; ++c) target[c] = source[channels == 1 ? 0 : bgr ? 2-c : c];
        }
    }
    return prepare_pixels(pixels, regions);
}

ImageInput read_image_request(const std::filesystem::path & path) {
    std::ifstream stream(path);
    std::string magic, image;
    std::size_t count = 0;
    stream >> magic >> std::quoted(image) >> count;
    if (!stream || (magic != "RAIP1" && magic != "RAIP2") || count > 10000)
        throw std::runtime_error("invalid RAIP request");
    ImageOptions options;
    options.image = std::filesystem::path(image);
    if (options.image.is_relative()) options.image = path.parent_path() / options.image;
    options.boxes_xyxy.resize(count * 4);
    // Python JSON numbers are parsed as double before np.float32 conversion.
    // Match that rounding for decimal coordinates exactly on a float midpoint.
    for (float & value : options.boxes_xyxy) {
        double parsed = 0;
        stream >> parsed;
        value = static_cast<float>(parsed);
    }
    if (!stream) throw std::runtime_error("truncated RAIP boxes");
    if (magic == "RAIP2") {
        int scored = 0;
        std::string mask_path;
        stream >> scored;
        if (scored != 0 && scored != 1) throw std::runtime_error("invalid RAIP score flag");
        if (scored) {
            options.box_scores.resize(count);
            for (float & value : options.box_scores) {
                double parsed = 0;
                stream >> parsed;
                value = static_cast<float>(parsed);
            }
        }
        stream >> std::quoted(mask_path);
        if (!stream) throw std::runtime_error("truncated RAIP optional regions");
        if (!mask_path.empty()) {
            auto file = std::filesystem::path(mask_path);
            if (file.is_relative()) file = path.parent_path() / file;
            std::ifstream masks(file, std::ios::binary);
            if (!masks) throw std::runtime_error("cannot open masks: " + file.string());
            const auto bytes = std::filesystem::file_size(file);
            const auto pixels = load_rgb(options.image);
            if (bytes != count * static_cast<std::size_t>(pixels.width) * pixels.height)
                throw std::runtime_error("invalid mask byte count");
            options.masks.resize(bytes);
            masks.read(reinterpret_cast<char *>(options.masks.data()), bytes);
            if (!masks) throw std::runtime_error("truncated masks");
        }
    }
    return prepare_image(options);
}
} // namespace relateanything
