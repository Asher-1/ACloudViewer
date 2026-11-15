// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Reference camera centers for the reference-frame anchor of
// MapAnythingReconstruction::anchor_to_reference, from either
//   - a WAI scene_meta.json (the official MapAnything ETH3D test scenes), or
//   - a COLMAP model_aligner --ref_images_path text file (the user's own
//     GPS or surveyed camera positions).
// Shared by the GUI worker and the headless GT_META / REF_IMAGES_PATH
// options so both front ends read a file identically.

#pragma once

#include <QString>
#include <QStringList>
#include <vector>

namespace MapAnythingSceneMeta {

/** WAI layout <scene>/images/<frame>.<ext> next to <scene>/scene_meta.json:
 *  the meta path when every image lies in one directory named "images" and
 *  that file exists beside it; empty otherwise. */
QString discover(const QStringList& images);

/** Reference camera centers, images.size()*3 floats (transform_matrix
 *  column 3 of the frame whose frame_name equals the image's file name
 *  without its last suffix; NaN when absent). false + \p error when the file
 *  is unreadable or has no frames array. */
bool loadCenters(const QString& metaPath,
                 const QStringList& images,
                 std::vector<float>* centers,
                 QString* error);

/** A cartesian axis is re-centred when the first reference line has
 *  |coordinate| >= this (ecvGlobalShiftManager's default threshold: below
 *  it a float keeps ~1 mm). One constant so GUI and headless agree. */
constexpr double kShiftThreshold = 1.0e4;

struct ReferenceCenters {
    /** images.size()*3 floats, NaN where an image has no reference line;
     *  reference coordinates minus \p offset. */
    std::vector<float> centers;
    /** Reference coordinates of the local origin: true = local + offset.
     *  Non-zero only for large cartesian coordinates (UTM, ECEF, national
     *  grids), per axis: the first line's value truncated to hundreds
     *  (ecvGlobalShiftManager::BestShift). */
    double offset[3] = {0.0, 0.0, 0.0};
    int matched = 0;  // input images with a reference line
    /** true: the file held WGS84 lat/lon/alt, converted to local
     *  East-North-Up metres about \p enuOrigin. */
    bool gps = false;
    double enuOrigin[3] = {0.0, 0.0, 0.0};  // lat, lon (deg), alt (m)
};

/** COLMAP model_aligner reference file: one "<image_name> X Y Z" per
 *  non-empty line, whitespace separated, trailing tokens ignored (COLMAP
 *  ReadFileCameraLocations); lines starting with '#' are skipped. A line
 *  names an input image when it equals the image's path or a trailing part
 *  of it ("DSC_1.JPG", "seq1/DSC_1.JPG"); lines naming no input image are
 *  ignored, as in COLMAP.
 *  \p isGps: X Y Z are latitude, longitude (degrees) and altitude (m) on
 *  WGS84, converted to ENU about the FIRST line (model_aligner
 *  --ref_is_gps 1 --alignment_type enu). Otherwise cartesian, kept in the
 *  file's own frame (large axes carried by ReferenceCenters::offset).
 *  false + \p error: unreadable file, malformed line, a line matching two
 *  images, an image named by two lines, or fewer than 3 matched images. */
bool loadColmapReferences(const QString& path,
                          const QStringList& images,
                          bool isGps,
                          ReferenceCenters* out,
                          QString* error);

/** WGS84 geodetic (lat, lon in degrees, alt in m) -> ECEF metres
 *  (COLMAP GPSTransform::EllipsoidToECEF). */
void wgs84ToEcef(const double lla[3], double xyz[3]);
/** WGS84 geodetic -> East-North-Up metres about \p origin (lat, lon, alt)
 *  (COLMAP GPSTransform::EllipsoidToENU). */
void wgs84ToEnu(const double lla[3], const double origin[3], double enu[3]);

}  // namespace MapAnythingSceneMeta
