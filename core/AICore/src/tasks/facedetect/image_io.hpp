#pragma once

// The image toolkit is owned by src/common/ (shared with the ocr task and
// any future image-input task); this header keeps the fd:: names working
// for the facedetect task's graph helpers.
#include "common/image_utils.hpp"

namespace fd {

using aicore::img::Image;
using aicore::img::cv_resize_linear_u8;
using aicore::img::image_from_rgb;
using aicore::img::image_from_view;
using aicore::img::load_image_rgb;

}  // namespace fd
