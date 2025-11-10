// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ImageIO.h"

// clang-format off
#include <cstddef>
#include <cstdio>
#include <jpeglib.h>  // Include after cstddef to define size_t
#include <png.h>
// clang-format on
#include <FileSystem.h>
#include <Logging.h>

#include <array>
#include <fstream>
#include <unordered_map>

namespace cloudViewer {

namespace {
using namespace io;

/// Convert libjpeg error messages to std::runtime_error. This prevents
/// libjpeg from exit() in case of errors.
void jpeg_error_throw(j_common_ptr p_cinfo) {
    if (p_cinfo->is_decompressor)
        jpeg_destroy_decompress(
                reinterpret_cast<jpeg_decompress_struct *>(p_cinfo));
    else
        jpeg_destroy_compress(
                reinterpret_cast<jpeg_compress_struct *>(p_cinfo));
    char buffer[JMSG_LENGTH_MAX];
    (*p_cinfo->err->format_message)(p_cinfo, buffer);
    throw std::runtime_error(buffer);
}

void SetPNGImageFromImage(const geometry::Image &image,
                          int quality,
                          png_image &pngimage) {
    pngimage.width = image.width_;
    pngimage.height = image.height_;
    pngimage.format = pngimage.flags = 0;

    if (image.bytes_per_channel_ == 2) {
        pngimage.format |= PNG_FORMAT_FLAG_LINEAR;
    }
    if (image.num_of_channels_ >= 3) {
        pngimage.format |= PNG_FORMAT_FLAG_COLOR;
    }
    if (image.num_of_channels_ == 4) {
        pngimage.format |= PNG_FORMAT_FLAG_ALPHA;
    }
    if (quality <= 2) {
        pngimage.flags |= PNG_IMAGE_FLAG_FAST;
    }
}

using signature_decoder_t =
        std::pair<std::string,
                  std::function<bool(const std::string &, geometry::Image &)>>;
static const std::array<signature_decoder_t, 2> signature_decoder_list{
        {{"\x89\x50\x4e\x47\xd\xa\x1a\xa", ReadImageFromPNG},
         {"\xFF\xD8\xFF", ReadImageFromJPG}}};
static constexpr uint8_t MAX_SIGNATURE_LEN = 8;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const geometry::Image &, int)>>
        file_extension_to_image_write_function{
                {"png", WriteImageToPNG},
                {"jpg", WriteImageToJPG},
                {"jpeg", WriteImageToJPG},
        };

}  // unnamed namespace

namespace io {
std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename) {
    auto image = std::make_shared<geometry::Image>();
    ReadImage(filename, *image);
    return image;
}

std::shared_ptr<geometry::Image> CreateImageFromMemory(
        const std::string &image_format,
        const unsigned char *image_data_ptr,
        size_t image_data_size) {
    auto image = std::make_shared<geometry::Image>();
    ReadImageFromMemory(image_format, image_data_ptr, image_data_size, *image);
    return image;
}

bool ReadImage(const std::string &filename, geometry::Image &image) {
    std::string signature_buffer(MAX_SIGNATURE_LEN, 0);
    std::ifstream file(filename, std::ios::binary);
    file.read(&signature_buffer[0], MAX_SIGNATURE_LEN);
    std::string err_msg;
    if (!file) {
        err_msg = "Read geometry::Image failed for file {}. I/O error.";
    } else {
        file.close();
        for (const auto &signature_decoder : signature_decoder_list) {
            if (signature_buffer.compare(0, signature_decoder.first.size(),
                                         signature_decoder.first) == 0) {
                return signature_decoder.second(filename, image);
            }
        }
        err_msg =
                "Read geometry::Image failed for file {}. Unknown file "
                "signature, only PNG and JPG are supported.";
    }
    image.Clear();
    utility::LogWarning(err_msg.c_str(), filename);
    return false;
}

bool ReadImageFromMemory(const std::string &image_format,
                         const unsigned char *image_data_ptr,
                         size_t image_data_size,
                         geometry::Image &image) {
    std::string format = image_format;
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);
    if (format == "png") {
        return ReadPNGFromMemory(image_data_ptr, image_data_size, image);
    } else if (format == "jpg") {
        return ReadJPGFromMemory(image_data_ptr, image_data_size, image);
    } else {
        utility::LogWarning("The format of {} is not supported", format);
        return false;
    }
}

bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality /* = kCloudViewerImageIODefaultQuality*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_write_function.find(filename_ext);
    if (map_itr == file_extension_to_image_write_function.end()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    return map_itr->second(filename, image, quality);
}

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_file(&pngimage, filename.c_str()) == 0) {
        utility::LogWarning("Read PNG failed: unable to parse header.");
        return false;
    }

    // Clear colormap flag if necessary to ensure libpng expands the colo
    // indexed pixels to full color
    if (pngimage.format & PNG_FORMAT_FLAG_COLORMAP) {
        pngimage.format &= ~PNG_FORMAT_FLAG_COLORMAP;
    }

    image.Prepare(pngimage.width, pngimage.height,
                  PNG_IMAGE_SAMPLE_CHANNELS(pngimage.format),
                  PNG_IMAGE_SAMPLE_COMPONENT_SIZE(pngimage.format));

    if (png_image_finish_read(&pngimage, NULL, image.data_.data(), 0, NULL) ==
        0) {
        utility::LogWarning("Read PNG failed: unable to read file: {}",
                            filename);
        utility::LogWarning("PNG error: {}", pngimage.message);
        return false;
    }
    return true;
}

bool ReadPNGFromMemory(const unsigned char *image_data_ptr,
                       size_t image_data_size,
                       geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_memory(&pngimage, image_data_ptr,
                                         image_data_size) == 0) {
        utility::LogWarning("Read PNG failed: unable to parse header.");
        image.Clear();
        return false;
    }

    // Clear colormap flag if necessary to ensure libpng expands the colo
    // indexed pixels to full color
    if (pngimage.format & PNG_FORMAT_FLAG_COLORMAP) {
        pngimage.format &= ~PNG_FORMAT_FLAG_COLORMAP;
    }

    image.Prepare(pngimage.width, pngimage.height,
                  PNG_IMAGE_SAMPLE_CHANNELS(pngimage.format),
                  PNG_IMAGE_SAMPLE_COMPONENT_SIZE(pngimage.format));

    if (png_image_finish_read(&pngimage, NULL, image.data_.data(), 0, NULL) ==
        0) {
        utility::LogWarning("PNG error: {}", pngimage.message);
        image.Clear();
        return false;
    }
    return true;
}

bool WriteImageToPNG(const std::string &filename,
                     const geometry::Image &image,
                     int quality /* = kCloudViewerImageIODefaultQuality*/) {
    if (!image.HasData()) {
        utility::LogWarning("Write PNG failed: image has no data.");
        return false;
    }
    if (quality == kCloudViewerImageIODefaultQuality)  // Set default quality
        quality = 6;
    if (quality < 0 || quality > 9) {
        utility::LogWarning(
                "Write PNG failed: quality ({}) must be in the range [0,9]",
                quality);
        return false;
    }

    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    SetPNGImageFromImage(image, quality, pngimage);
    if (png_image_write_to_file(&pngimage, filename.c_str(), 0,
                                image.data_.data(), 0, NULL) == 0) {
        utility::LogWarning("Write PNG failed: unable to write file: {}",
                            filename);
        return false;
    }
    return true;
}

bool ReadImageFromJPG(const std::string &filename, geometry::Image &image) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file_in;
    JSAMPARRAY buffer;

    if ((file_in = utility::filesystem::FOpen(filename, "rb")) == NULL) {
        utility::LogWarning("Read JPG failed: unable to open file: {}",
                            filename);
        image.Clear();
        return false;
    }

    try {
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = jpeg_error_throw;
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, file_in);
        jpeg_read_header(&cinfo, TRUE);

        // We only support two channel types: gray, and RGB.
        int num_of_channels = 3;
        int bytes_per_channel = 1;
        switch (cinfo.jpeg_color_space) {
            case JCS_RGB:
            case JCS_YCbCr:
                cinfo.out_color_space = JCS_RGB;
                cinfo.out_color_components = 3;
                num_of_channels = 3;
                break;
            case JCS_GRAYSCALE:
                cinfo.jpeg_color_space = JCS_GRAYSCALE;
                cinfo.out_color_components = 1;
                num_of_channels = 1;
                break;
            case JCS_CMYK:
            case JCS_YCCK:
            default:
                utility::LogWarning(
                        "Read JPG failed: color space not supported.");
                jpeg_destroy_decompress(&cinfo);
                fclose(file_in);
                image.Clear();
                return false;
        }
        jpeg_start_decompress(&cinfo);
        image.Prepare(cinfo.output_width, cinfo.output_height, num_of_channels,
                      bytes_per_channel);
        int row_stride = cinfo.output_width * cinfo.output_components;
        buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                            row_stride, 1);
        uint8_t *pdata = image.data_.data();
        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            memcpy(pdata, buffer[0], row_stride);
            pdata += row_stride;
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(file_in);
        return true;
    } catch (const std::runtime_error &err) {
        fclose(file_in);
        image.Clear();
        utility::LogWarning("libjpeg error: {}", err.what());
        return false;
    }
}

bool ReadJPGFromMemory(const unsigned char *image_data_ptr,
                       size_t image_data_size,
                       geometry::Image &image) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPARRAY buffer;

    try {
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = jpeg_error_throw;
        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, image_data_ptr, image_data_size);
        jpeg_read_header(&cinfo, TRUE);

        // We only support two channel types: gray, and RGB.
        int num_of_channels = 3;
        int bytes_per_channel = 1;
        switch (cinfo.jpeg_color_space) {
            case JCS_RGB:
            case JCS_YCbCr:
                cinfo.out_color_space = JCS_RGB;
                cinfo.out_color_components = 3;
                num_of_channels = 3;
                break;
            case JCS_GRAYSCALE:
                cinfo.jpeg_color_space = JCS_GRAYSCALE;
                cinfo.out_color_components = 1;
                num_of_channels = 1;
                break;
            case JCS_CMYK:
            case JCS_YCCK:
            default:
                utility::LogWarning(
                        "Read JPG failed: color space not supported.");
                jpeg_destroy_decompress(&cinfo);
                image.Clear();
                return false;
        }
        jpeg_start_decompress(&cinfo);
        image.Prepare(cinfo.output_width, cinfo.output_height, num_of_channels,
                      bytes_per_channel);
        int row_stride = cinfo.output_width * cinfo.output_components;
        buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                            row_stride, 1);
        uint8_t *pdata = image.data_.data();
        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            memcpy(pdata, buffer[0], row_stride);
            pdata += row_stride;
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        return true;
    } catch (const std::runtime_error &err) {
        image.Clear();
        utility::LogWarning(err.what());
        return false;
    }
}

bool WriteImageToJPG(const std::string &filename,
                     const geometry::Image &image,
                     int quality /* = kCloudViewerImageIODefaultQuality*/) {
    if (!image.HasData()) {
        utility::LogWarning("Write JPG failed: image has no data.");
        return false;
    }
    if (image.bytes_per_channel_ != 1 ||
        (image.num_of_channels_ != 1 && image.num_of_channels_ != 3)) {
        utility::LogWarning("Write JPG failed: unsupported image data.");
        return false;
    }
    if (quality == kCloudViewerImageIODefaultQuality)  // Set default quality
        quality = 90;
    if (quality < 0 || quality > 100) {
        utility::LogWarning(
                "Write JPG failed: image quality should be in the range "
                "[0,100].");
        return false;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file_out;
    JSAMPROW row_pointer[1];

    if ((file_out = utility::filesystem::FOpen(filename, "wb")) == NULL) {
        utility::LogWarning("Write JPG failed: unable to open file: {}",
                            filename);
        return false;
    }

    try {
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = jpeg_error_throw;
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, file_out);
        cinfo.image_width = image.width_;
        cinfo.image_height = image.height_;
        cinfo.input_components = image.num_of_channels_;
        cinfo.in_color_space =
                (cinfo.input_components == 1 ? JCS_GRAYSCALE : JCS_RGB);
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, quality, TRUE);
        jpeg_start_compress(&cinfo, TRUE);
        int row_stride = image.width_ * image.num_of_channels_;
        const uint8_t *pdata = image.data_.data();
        std::vector<uint8_t> buffer(row_stride);
        while (cinfo.next_scanline < cinfo.image_height) {
            memcpy(buffer.data(), pdata, row_stride);
            row_pointer[0] = buffer.data();
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
            pdata += row_stride;
        }
        jpeg_finish_compress(&cinfo);
        fclose(file_out);
        jpeg_destroy_compress(&cinfo);
        return true;
    } catch (const std::runtime_error &err) {
        fclose(file_out);
        utility::LogWarning(err.what());
        return false;
    }
}

}  // namespace io
}  // namespace cloudViewer
