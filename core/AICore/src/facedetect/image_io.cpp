#include "image_io.hpp"
#include "common.hpp"

// Vendored public-domain single-header decoder. Used for PNG/BMP and as a
// fallback for any non-JPEG input. JPEG goes through libjpeg-turbo instead (see
// below) so the decode matches cv2.imread (insightface's loader) bit-for-bit.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_STDIO_GIF  // (no-op guard; GIF disabled by not enabling it)
#include "stb_image.h"

#include <cstdio>
#include <cstring>

#if FACEDETECT_HAVE_LIBJPEG
#include <csetjmp>
extern "C" {
#include <jpeglib.h>
}
#endif

namespace fd {

#if FACEDETECT_HAVE_LIBJPEG
namespace {

// libjpeg's default error handler calls exit() on a fatal error. Route errors
// through setjmp/longjmp so a corrupt/truncated JPEG fails gracefully (return
// false) instead of tearing down the process.
struct jpeg_error_jmp {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

void jpeg_error_exit_longjmp(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<jpeg_error_jmp*>(cinfo->err);
    char buf[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buf);
    FD_LOG("load_image_rgb: libjpeg error: %s", buf);
    longjmp(err->setjmp_buffer, 1);
}

// Decode a JPEG file to tightly-packed RGB using libjpeg-turbo with the same
// settings OpenCV's imread uses (JDCT_ISLOW IDCT + fancy chroma upsampling, both
// the libjpeg defaults), so the output matches cv2.imread pixel-for-pixel.
bool load_jpeg_rgb(const std::string& path, Image& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        FD_LOG("load_image_rgb: cannot open %s", path.c_str());
        return false;
    }

    jpeg_decompress_struct cinfo;
    jpeg_error_jmp jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit_longjmp;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(f);
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    jpeg_read_header(&cinfo, TRUE);

    // Match cv2.imread / libjpeg defaults explicitly so the decode is reproducible
    // regardless of any future libjpeg default changes.
    cinfo.out_color_space     = JCS_RGB;     // 3-channel R,G,B output
    cinfo.dct_method          = JDCT_ISLOW;  // accurate integer IDCT (default)
    cinfo.do_fancy_upsampling = TRUE;        // smooth chroma upsampling (default)

    jpeg_start_decompress(&cinfo);

    const int w = (int)cinfo.output_width;
    const int h = (int)cinfo.output_height;
    const int comps = cinfo.output_components;  // == 3 for JCS_RGB
    if (comps != 3) {
        FD_LOG("load_image_rgb: unexpected output_components=%d for %s", comps, path.c_str());
        jpeg_destroy_decompress(&cinfo);
        std::fclose(f);
        return false;
    }

    out.width = w;
    out.height = h;
    out.rgb.resize((size_t)w * h * 3);

    const int row_stride = w * comps;
    while ((int)cinfo.output_scanline < h) {
        unsigned char* rowptr = out.rgb.data() + (size_t)cinfo.output_scanline * row_stride;
        JSAMPROW rows[1] = { rowptr };
        jpeg_read_scanlines(&cinfo, rows, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(f);
    return true;
}

// Cheap JPEG sniff by the SOI marker (FF D8 FF). Avoids relying on the file
// extension while keeping non-JPEG inputs (PNG/BMP) on the stb path.
bool is_jpeg(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char magic[3] = {0, 0, 0};
    size_t n = std::fread(magic, 1, 3, f);
    std::fclose(f);
    return n == 3 && magic[0] == 0xFF && magic[1] == 0xD8 && magic[2] == 0xFF;
}

}  // namespace
#endif  // FACEDETECT_HAVE_LIBJPEG

bool load_image_rgb(const std::string& path, Image& out) {
#if FACEDETECT_HAVE_LIBJPEG
    // JPEG -> libjpeg-turbo (cv2.imread parity). Fall through to stb only if the
    // libjpeg decode fails, so a non-fatal libjpeg quirk never blocks loading.
    if (is_jpeg(path) && load_jpeg_rgb(path, out)) {
        return true;
    }
#endif
    int w = 0, h = 0, c = 0;
    // Request 3 components: stb converts grayscale/RGBA to RGB for us.
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        FD_LOG("load_image_rgb: failed to decode %s (%s)", path.c_str(), stbi_failure_reason());
        return false;
    }
    out.width = w;
    out.height = h;
    out.rgb.assign(data, data + (size_t)w * h * 3);
    stbi_image_free(data);
    return true;
}

bool image_from_rgb(const uint8_t* rgb, int width, int height, Image& out) {
    if (!rgb || width <= 0 || height <= 0) return false;
    out.width = width;
    out.height = height;
    out.rgb.assign(rgb, rgb + (size_t)width * height * 3);
    return true;
}

} // namespace fd
