// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "util/bitmap"
#include "util/testing.h"

#include <boost/filesystem.hpp>

#include <OpenImageIO/imageio.h>

#include <cstdlib>
#include <vector>

#include "util/bitmap.h"

using namespace colmap;

TEST(util_bitmap, TestBitmapColorEmpty) {
  BitmapColor<uint8_t> color;
  EXPECT_EQ(color.r, 0);
  EXPECT_EQ(color.g, 0);
  EXPECT_EQ(color.b, 0);
  EXPECT_EQ(color, BitmapColor<uint8_t>(0));
  EXPECT_EQ(color, BitmapColor<uint8_t>(0, 0, 0));
}

TEST(util_bitmap, TestBitmapGrayColor) {
  BitmapColor<uint8_t> color(5);
  EXPECT_EQ(color.r, 5);
  EXPECT_EQ(color.g, 5);
  EXPECT_EQ(color.b, 5);
}

TEST(util_bitmap, TestBitmapColorCast) {
  BitmapColor<float> color1(1.1, 2.9, -3.0);
  BitmapColor<uint8_t> color2 = color1.Cast<uint8_t>();
  EXPECT_EQ(color2.r, 1);
  EXPECT_EQ(color2.g, 3);
  EXPECT_EQ(color2.b, 0);
}

TEST(util_bitmap, TestEmpty) {
  Bitmap bitmap;
  EXPECT_EQ(bitmap.Width(), 0);
  EXPECT_EQ(bitmap.Height(), 0);
  EXPECT_EQ(bitmap.Channels(), 0);
  EXPECT_EQ(bitmap.IsRGB(), false);
  EXPECT_EQ(bitmap.IsGrey(), false);
}

TEST(util_bitmap, TestAllocateRGB) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  EXPECT_EQ(bitmap.Width(), 100);
  EXPECT_EQ(bitmap.Height(), 100);
  EXPECT_EQ(bitmap.Channels(), 3);
  EXPECT_EQ(bitmap.IsRGB(), true);
  EXPECT_EQ(bitmap.IsGrey(), false);
}

TEST(util_bitmap, TestAllocateGrey) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  EXPECT_EQ(bitmap.Width(), 100);
  EXPECT_EQ(bitmap.Height(), 100);
  EXPECT_EQ(bitmap.Channels(), 1);
  EXPECT_EQ(bitmap.IsRGB(), false);
  EXPECT_EQ(bitmap.IsGrey(), true);
}

TEST(util_bitmap, TestDeallocate) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  bitmap.Deallocate();
  EXPECT_EQ(bitmap.Width(), 0);
  EXPECT_EQ(bitmap.Height(), 0);
  EXPECT_EQ(bitmap.Channels(), 0);
  EXPECT_EQ(bitmap.NumBytes(), 0);
  EXPECT_EQ(bitmap.IsRGB(), false);
  EXPECT_EQ(bitmap.IsGrey(), false);
}

TEST(util_bitmap, TestBitsPerPixel) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  EXPECT_EQ(bitmap.BitsPerPixel(), 24);
  bitmap.Allocate(100, 100, false);
  EXPECT_EQ(bitmap.BitsPerPixel(), 8);
}

TEST(util_bitmap, TestNumBytes) {
  Bitmap bitmap;
  EXPECT_EQ(bitmap.NumBytes(), 0);
  bitmap.Allocate(100, 100, true);
  EXPECT_EQ(bitmap.NumBytes(), 3 * 100 * 100);
  bitmap.Allocate(100, 100, false);
  EXPECT_EQ(bitmap.NumBytes(), 100 * 100);
}

TEST(util_bitmap, TestConvertToRowMajorArrayRGB) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToRowMajorArray();
  EXPECT_EQ(array.size(), 12);
  EXPECT_EQ(array[0], 0);
  EXPECT_EQ(array[1], 0);
  EXPECT_EQ(array[2], 0);
  EXPECT_EQ(array[3], 2);
  EXPECT_EQ(array[4], 0);
  EXPECT_EQ(array[5], 0);
  EXPECT_EQ(array[6], 1);
  EXPECT_EQ(array[7], 0);
  EXPECT_EQ(array[8], 0);
  EXPECT_EQ(array[9], 3);
  EXPECT_EQ(array[10], 0);
  EXPECT_EQ(array[11], 0);
}

TEST(util_bitmap, TestConvertToRowMajorArrayGrey) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToRowMajorArray();
  EXPECT_EQ(array.size(), 4);
  EXPECT_EQ(array[0], 0);
  EXPECT_EQ(array[1], 2);
  EXPECT_EQ(array[2], 1);
  EXPECT_EQ(array[3], 3);
}

TEST(util_bitmap, TestConvertToColMajorArrayRGB) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToColMajorArray();
  EXPECT_EQ(array.size(), 12);
  EXPECT_EQ(array[0], 0);
  EXPECT_EQ(array[1], 1);
  EXPECT_EQ(array[2], 2);
  EXPECT_EQ(array[3], 3);
  EXPECT_EQ(array[4], 0);
  EXPECT_EQ(array[5], 0);
  EXPECT_EQ(array[6], 0);
  EXPECT_EQ(array[7], 0);
  EXPECT_EQ(array[8], 0);
  EXPECT_EQ(array[9], 0);
  EXPECT_EQ(array[10], 0);
  EXPECT_EQ(array[11], 0);
}

TEST(util_bitmap, TestConvertToColMajorArrayGrey) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToColMajorArray();
  EXPECT_EQ(array.size(), 4);
  EXPECT_EQ(array[0], 0);
  EXPECT_EQ(array[1], 1);
  EXPECT_EQ(array[2], 2);
  EXPECT_EQ(array[3], 3);
}

TEST(util_bitmap, TestGetAndSetPixelRGB) {
  Bitmap bitmap;
  bitmap.Allocate(1, 1, true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(bitmap.GetPixel(0, 0, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
}

TEST(util_bitmap, TestGetAndSetPixelGrey) {
  Bitmap bitmap;
  bitmap.Allocate(1, 1, false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 2, 3));
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(bitmap.GetPixel(0, 0, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(1, 2, 3));
  EXPECT_TRUE(bitmap.GetPixel(0, 0, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 0, 0));
}

TEST(util_bitmap, TestGetScanlineRGB) {
  Bitmap bitmap;
  bitmap.Allocate(3, 3, true);
  bitmap.Fill(BitmapColor<uint8_t>(1, 2, 3));
  for (size_t r = 0; r < 3; ++r) {
    const uint8_t* scanline = bitmap.GetScanline(r);
    for (size_t c = 0; c < 3; ++c) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(bitmap.GetPixel(r, c, &color));
      EXPECT_EQ(scanline[c * 3], color.r);
      EXPECT_EQ(scanline[c * 3 + 1], color.g);
      EXPECT_EQ(scanline[c * 3 + 2], color.b);
    }
  }
}

TEST(util_bitmap, TestGetScanlineGrey) {
  Bitmap bitmap;
  bitmap.Allocate(3, 3, false);
  bitmap.Fill(BitmapColor<uint8_t>(1, 2, 3));
  for (size_t r = 0; r < 3; ++r) {
    const uint8_t* scanline = bitmap.GetScanline(r);
    for (size_t c = 0; c < 3; ++c) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(bitmap.GetPixel(r, c, &color));
      EXPECT_EQ(scanline[c], color.r);
    }
  }
}

TEST(util_bitmap, TestFill) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  bitmap.Fill(BitmapColor<uint8_t>(1, 2, 3));
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(bitmap.GetPixel(x, y, &color));
      EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
    }
  }
}

TEST(util_bitmap, TestInterpolateNearestNeighbor) {
  Bitmap bitmap;
  bitmap.Allocate(11, 11, true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(5, 5, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(5, 5, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(5.4999, 5.4999, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(5.5, 5.5, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(0, 0, 0));
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(4.5, 5.4999, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
}

TEST(util_bitmap, TestInterpolateBilinear) {
  Bitmap bitmap;
  bitmap.Allocate(11, 11, true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(5, 5, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<float> color;
  EXPECT_TRUE(bitmap.InterpolateBilinear(5, 5, &color));
  EXPECT_EQ(color, BitmapColor<float>(1, 2, 3));
  EXPECT_TRUE(bitmap.InterpolateBilinear(5.5, 5, &color));
  EXPECT_EQ(color, BitmapColor<float>(0.5, 1, 1.5));
  EXPECT_TRUE(bitmap.InterpolateBilinear(5.5, 5.5, &color));
  EXPECT_EQ(color, BitmapColor<float>(0.25, 0.5, 0.75));
}

TEST(util_bitmap, TestSmoothRGB) {
  Bitmap bitmap;
  bitmap.Allocate(50, 50, true);
  for (int x = 0; x < 50; ++x) {
    for (int y = 0; y < 50; ++y) {
      bitmap.SetPixel(x, y,
                      BitmapColor<uint8_t>(y * 50 + x, y * 50 + x, y * 50 + x));
    }
  }
  bitmap.Smooth(1, 1);
  EXPECT_EQ(bitmap.Width(), 50);
  EXPECT_EQ(bitmap.Height(), 50);
  EXPECT_EQ(bitmap.Channels(), 3);
  for (int x = 0; x < 50; ++x) {
    for (int y = 0; y < 50; ++y) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(bitmap.GetPixel(x, y, &color));
      EXPECT_EQ(color.r, color.g);
      EXPECT_EQ(color.r, color.b);
    }
  }
}

TEST(util_bitmap, TestSmoothGrey) {
  Bitmap bitmap;
  bitmap.Allocate(50, 50, false);
  for (int x = 0; x < 50; ++x) {
    for (int y = 0; y < 50; ++y) {
      bitmap.SetPixel(x, y,
                      BitmapColor<uint8_t>(y * 50 + x, y * 50 + x, y * 50 + x));
    }
  }
  bitmap.Smooth(1, 1);
  EXPECT_EQ(bitmap.Width(), 50);
  EXPECT_EQ(bitmap.Height(), 50);
  EXPECT_EQ(bitmap.Channels(), 1);
}

TEST(util_bitmap, TestRescaleRGB) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  Bitmap bitmap1 = bitmap.Clone();
  bitmap1.Rescale(50, 25);
  EXPECT_EQ(bitmap1.Width(), 50);
  EXPECT_EQ(bitmap1.Height(), 25);
  EXPECT_EQ(bitmap1.Channels(), 3);
  Bitmap bitmap2 = bitmap.Clone();
  bitmap2.Rescale(150, 20);
  EXPECT_EQ(bitmap2.Width(), 150);
  EXPECT_EQ(bitmap2.Height(), 20);
  EXPECT_EQ(bitmap2.Channels(), 3);
}

TEST(util_bitmap, TestRescaleGrey) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  Bitmap bitmap1 = bitmap.Clone();
  bitmap1.Rescale(50, 25);
  EXPECT_EQ(bitmap1.Width(), 50);
  EXPECT_EQ(bitmap1.Height(), 25);
  EXPECT_EQ(bitmap1.Channels(), 1);
  Bitmap bitmap2 = bitmap.Clone();
  bitmap2.Rescale(150, 20);
  EXPECT_EQ(bitmap2.Width(), 150);
  EXPECT_EQ(bitmap2.Height(), 20);
  EXPECT_EQ(bitmap2.Channels(), 1);
}

TEST(util_bitmap, TestClone) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  const Bitmap cloned_bitmap = bitmap.Clone();
  EXPECT_EQ(cloned_bitmap.Width(), 100);
  EXPECT_EQ(cloned_bitmap.Height(), 100);
  EXPECT_EQ(cloned_bitmap.Channels(), 3);
  EXPECT_NE(bitmap.Data(), cloned_bitmap.Data());
}

TEST(util_bitmap, TestCloneAsRGB) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  const Bitmap cloned_bitmap = bitmap.CloneAsRGB();
  EXPECT_EQ(cloned_bitmap.Width(), 100);
  EXPECT_EQ(cloned_bitmap.Height(), 100);
  EXPECT_EQ(cloned_bitmap.Channels(), 3);
  EXPECT_NE(bitmap.Data(), cloned_bitmap.Data());
}

TEST(util_bitmap, TestCloneAsGrey) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  const Bitmap cloned_bitmap = bitmap.CloneAsGrey();
  EXPECT_EQ(cloned_bitmap.Width(), 100);
  EXPECT_EQ(cloned_bitmap.Height(), 100);
  EXPECT_EQ(cloned_bitmap.Channels(), 1);
  EXPECT_NE(bitmap.Data(), cloned_bitmap.Data());
}

TEST(util_bitmap, TestOpenImageIORoundTrip) {
  const boost::filesystem::path path =
      boost::filesystem::temp_directory_path() /
      boost::filesystem::unique_path("colmap-bitmap-%%%%-%%%%.png");

  Bitmap written;
  ASSERT_TRUE(written.Allocate(2, 1, true));
  ASSERT_TRUE(written.SetPixel(0, 0, BitmapColor<uint8_t>(1, 2, 3)));
  ASSERT_TRUE(written.SetPixel(1, 0, BitmapColor<uint8_t>(4, 5, 6)));
  ASSERT_TRUE(written.Write(path.string(), BitmapFormat::kPng));

  Bitmap read;
  ASSERT_TRUE(read.Read(path.string(), true));
  EXPECT_TRUE(read.ConvertToRowMajorArray() == written.ConvertToRowMajorArray());
  boost::filesystem::remove(path);
}

TEST(util_bitmap, TestOpenImageIOFormatParity) {
  const boost::filesystem::path directory =
      boost::filesystem::temp_directory_path() /
      boost::filesystem::unique_path("colmap-bitmap-formats-%%%%-%%%%");
  boost::filesystem::create_directories(directory);

  Bitmap written;
  ASSERT_TRUE(written.Allocate(3, 2, true));
  written.Fill(BitmapColor<uint8_t>(17, 34, 51));
  for (const auto& extension : {".png", ".jpg", ".tiff"}) {
    const boost::filesystem::path path = directory / ("bitmap" + std::string(extension));
    ASSERT_TRUE(written.Write(path.string()));
    Bitmap read;
    ASSERT_TRUE(read.Read(path.string(), true));
    EXPECT_EQ(read.Width(), written.Width());
    EXPECT_EQ(read.Height(), written.Height());
    EXPECT_EQ(read.Channels(), written.Channels());
    if (std::string(extension) != ".jpg") {
      EXPECT_TRUE(read.ConvertToRowMajorArray() == written.ConvertToRowMajorArray());
    }
  }
  boost::filesystem::remove_all(directory);
}

TEST(util_bitmap, TestOpenImageIORescaleFilters) {
  Bitmap bitmap;
  ASSERT_TRUE(bitmap.Allocate(4, 4, false));
  bitmap.Fill(BitmapColor<uint8_t>(0));
  ASSERT_TRUE(bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(255)));

  Bitmap bilinear = bitmap.Clone();
  bilinear.Rescale(1, 1, BitmapRescaleFilter::kBilinear);
  Bitmap box = bitmap.Clone();
  box.Rescale(1, 1, BitmapRescaleFilter::kBox);
  BitmapColor<uint8_t> bilinear_color;
  BitmapColor<uint8_t> box_color;
  ASSERT_TRUE(bilinear.GetPixel(0, 0, &bilinear_color));
  ASSERT_TRUE(box.GetPixel(0, 0, &box_color));
  EXPECT_NE(bilinear_color.r, box_color.r);
}

TEST(util_bitmap, TestOpenImageIONumericExif) {
  // The fixture ships inside the shared objects_detection_data release
  // archive (ecvTestDataRepository cache, ~/cloudViewer_data/extract). It is
  // not part of the git tree, so headless environments that never downloaded
  // the dataset skip this case instead of failing the parity gate.
  std::vector<boost::filesystem::path> candidates;
  if (const char* home = std::getenv("HOME")) {
    candidates.push_back(boost::filesystem::path(home) /
                         "cloudViewer_data/extract/objects_detection_data/"
                         "images/deeplsd_examples.jpg");
  }
  candidates.push_back(
      boost::filesystem::path(__FILE__)
          .parent_path()
          .parent_path()
          .parent_path()
          .parent_path()
          .parent_path() /
      "examples/test_data/image/objects_detection_data/images/"
      "deeplsd_examples.jpg");

  boost::filesystem::path path;
  for (const auto& candidate : candidates) {
    if (boost::filesystem::is_regular_file(candidate)) {
      path = candidate;
      break;
    }
  }
  if (path.empty()) {
    std::cout << (
        "Skipping TestOpenImageIONumericExif: deeplsd_examples.jpg is not "
        "available (shared test-data cache not populated)");
    return;
  }

  Bitmap bitmap;
  ASSERT_TRUE(bitmap.Read(path.string(), true));
  double value = 0.0;
  ASSERT_TRUE(bitmap.ExifFocalLength(&value));
  EXPECT_NEAR(value, 35.0 / 43.27 * std::hypot(640.0, 480.0), std::abs(35.0 / 43.27 * std::hypot(640.0, 480.0)) * (0.01) / 100.0);
  std::string camera_model;
  ASSERT_TRUE(bitmap.ExifCameraModel(&camera_model));
  EXPECT_EQ(camera_model, "Panasonic-DMC-LC80-35.000000-640x480");
}
