// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "ImageIO.h"

#include <unordered_map>

#include <cstddef>
#include <cstdio>

#include <png.h>
#include <jpeglib.h>  // Include after cstddef to define size_t

#include <Console.h>
#include <FileSystem.h>

namespace cloudViewer {

namespace {
using namespace io;

void SetPNGImageFromImage(const geometry::Image &image, png_image &pngimage) {
	pngimage.width = image.width_;
	pngimage.height = image.height_;
	pngimage.format = 0;
	if (image.bytes_per_channel_ == 2) {
		pngimage.format |= PNG_FORMAT_FLAG_LINEAR;
	}
	if (image.num_of_channels_ == 3) {
		pngimage.format |= PNG_FORMAT_FLAG_COLOR;
	}
}

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::Image &)>>
        file_extension_to_image_read_function{
                {"png", ReadImageFromPNG},
                {"jpg", ReadImageFromJPG},
                {"jpeg", ReadImageFromJPG},
        };

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
	using namespace CVLib;
std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename) {
    auto image = std::make_shared<geometry::Image>();
    ReadImage(filename, *image);
    return image;
}

bool ReadImage(const std::string &filename, geometry::Image &image) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::Image failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_read_function.find(filename_ext);
    if (map_itr == file_extension_to_image_read_function.end()) {
        utility::LogWarning(
                "Read geometry::Image failed: unknown file extension.");
        return false;
    }
    return map_itr->second(filename, image);
}

bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality /* = 90*/) {
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

	// We only support two channel types: gray, and RGB.
	// There is no alpha channel.
	// bytes_per_channel is determined by PNG_FORMAT_FLAG_LINEAR flag.
	image.Prepare(pngimage.width, pngimage.height,
		(pngimage.format & PNG_FORMAT_FLAG_COLOR) ? 3 : 1,
		(pngimage.format & PNG_FORMAT_FLAG_LINEAR) ? 2 : 1);
	SetPNGImageFromImage(image, pngimage);
	if (png_image_finish_read(&pngimage, NULL, image.data_.data(), 0, NULL) ==
		0) {
		utility::LogWarning("Read PNG failed: unable to read file: {}",
			filename);
		return false;
	}
	return true;
}

bool WriteImageToPNG(const std::string &filename,
	const geometry::Image &image,
	int quality) {
	if (image.HasData() == false) {
		utility::LogWarning("Write PNG failed: image has no data.");
		return false;
	}
	png_image pngimage;
	memset(&pngimage, 0, sizeof(pngimage));
	pngimage.version = PNG_IMAGE_VERSION;
	SetPNGImageFromImage(image, pngimage);
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
		return false;
	}

	cinfo.err = jpeg_std_error(&jerr);
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
		utility::LogWarning("Read JPG failed: color space not supported.");
		jpeg_destroy_decompress(&cinfo);
		fclose(file_in);
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
}

bool WriteImageToJPG(const std::string &filename,
	const geometry::Image &image,
	int quality /* = 90*/) {
	if (image.HasData() == false) {
		utility::LogWarning("Write JPG failed: image has no data.");
		return false;
	}
	if (image.bytes_per_channel_ != 1 ||
		(image.num_of_channels_ != 1 && image.num_of_channels_ != 3)) {
		utility::LogWarning("Write JPG failed: unsupported image data.");
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

	cinfo.err = jpeg_std_error(&jerr);
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
}


}  // namespace io
}  // namespace cloudViewer
