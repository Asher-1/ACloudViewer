// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#define TEST_NAME "mvs/texture_mapping_test"
#include "util/testing.h"

#include "mvs/texture_mapping.h"

namespace colmap {
namespace mvs {
namespace {

PlyMesh MakeQuad() {
  PlyMesh mesh;
  mesh.vertices = {PlyMeshVertex(0.0f, 0.0f, 0.0f),
                   PlyMeshVertex(1.0f, 0.0f, 0.0f),
                   PlyMeshVertex(1.0f, 1.0f, 0.0f),
                   PlyMeshVertex(0.0f, 1.0f, 0.0f)};
  mesh.faces = {PlyMeshFace(0, 1, 2), PlyMeshFace(0, 2, 3)};
  return mesh;
}

Image MakeImage(float camera_z, const BitmapColor<uint8_t>& color) {
  constexpr int kSize = 128;
  const float K[9] = {kSize, 0, kSize / 2.0f,
                      0, kSize, kSize / 2.0f,
                      0, 0, 1};
  const float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  const float T[3] = {-0.5f, 0.5f, camera_z};
  Image image("test.png", kSize, kSize, K, R, T);
  Bitmap bitmap;
  BOOST_REQUIRE(bitmap.Allocate(kSize, kSize, /*as_rgb=*/true));
  bitmap.Fill(color);
  image.SetBitmap(bitmap);
  return image;
}

MeshTextureMappingOptions TestOptions() {
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  return options;
}

}  // namespace

BOOST_AUTO_TEST_CASE(EndToEnd) {
  const PlyMesh mesh = MakeQuad();
  const BitmapColor<uint8_t> source_color(200, 50, 25);
  const std::vector<Image> images = {MakeImage(5.0f, source_color)};

  const MeshTextureMappingResult result =
      MeshTextureMapping(mesh, images, TestOptions());

  BOOST_CHECK_GT(result.atlas_width, 0);
  BOOST_CHECK_GT(result.atlas_height, 0);
  BOOST_CHECK_EQUAL(result.face_uvs.size(), mesh.faces.size() * 6);
  BOOST_REQUIRE_EQUAL(result.face_view_ids.size(), mesh.faces.size());
  BOOST_CHECK_EQUAL(result.face_view_ids[0], 0);
  BOOST_CHECK_EQUAL(result.face_view_ids[1], 0);
  for (float uv : result.face_uvs) {
    BOOST_CHECK_GE(uv, 0.0f);
    BOOST_CHECK_LE(uv, 1.0f);
  }

  bool found_source_color = false;
  for (int y = 0; y < result.atlas_height && !found_source_color; ++y) {
    for (int x = 0; x < result.atlas_width; ++x) {
      BitmapColor<uint8_t> color;
      if (result.texture_atlas.GetPixel(x, y, &color) && color.r > 150 &&
          color.g < 100 && color.b < 100) {
        found_source_color = true;
        break;
      }
    }
  }
  BOOST_CHECK(found_source_color);
}

BOOST_AUTO_TEST_CASE(FaceBehindCameraIsRejected) {
  const PlyMesh mesh = MakeQuad();
  const std::vector<Image> images = {
      MakeImage(-5.0f, BitmapColor<uint8_t>(128))};
  const MeshTextureMappingResult result =
      MeshTextureMapping(mesh, images, TestOptions());
  BOOST_REQUIRE_EQUAL(result.face_view_ids.size(), mesh.faces.size());
  BOOST_CHECK_EQUAL(result.face_view_ids[0], -1);
  BOOST_CHECK_EQUAL(result.face_view_ids[1], -1);
}

BOOST_AUTO_TEST_CASE(EmptyMeshIsStable) {
  const std::vector<Image> images = {
      MakeImage(5.0f, BitmapColor<uint8_t>(128))};
  const MeshTextureMappingResult result =
      MeshTextureMapping(PlyMesh(), images, TestOptions());
  BOOST_CHECK_EQUAL(result.atlas_width, 0);
  BOOST_CHECK_EQUAL(result.atlas_height, 0);
  BOOST_CHECK(result.face_uvs.empty());
  BOOST_CHECK(result.face_view_ids.empty());
}

}  // namespace mvs
}  // namespace colmap
