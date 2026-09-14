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

#include "exe/model.h"

#include "scene/reconstruction_io.h"

#include "geometry/gps.h"
#include "geometry/pose.h"
#include "geometry/similarity_transform.h"
#include "controllers/reconstruction_clustering.h"
#include "estimators/alignment.h"
#include "estimators/coordinate_frame.h"
#include "scene/reconstruction_clustering.h"
#include "util/misc.h"
#include "controllers/option_manager.h"
#include "util/threading.h"

namespace colmap {
namespace {

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
ComputeEqualPartsBounds(const Reconstruction& reconstruction,
                        const Eigen::Vector3i& split) {
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> bounds;
    const auto bbox = reconstruction.ComputeBoundingBox();
    const Eigen::Vector3d extent = bbox.second - bbox.first;
    const Eigen::Vector3d offset(extent(0) / split(0), extent(1) / split(1),
                                 extent(2) / split(2));

    for (int k = 0; k < split(2); ++k) {
        for (int j = 0; j < split(1); ++j) {
            for (int i = 0; i < split(0); ++i) {
                Eigen::Vector3d min_bound(bbox.first(0) + i * offset(0),
                                          bbox.first(1) + j * offset(1),
                                          bbox.first(2) + k * offset(2));
                bounds.emplace_back(min_bound, min_bound + offset);
            }
        }
    }

    return bounds;
}

Eigen::Vector3d TransformLatLonAltToModelCoords(
        const SimilarityTransform3& tform, double lat, double lon, double alt) {
    // Since this is intended for use in ENU aligned models we want to define the
    // altitude along the ENU frame z axis and not the Earth's radius. Thus, we
    // set the altitude to 0 when converting from LLA to ECEF and then we use the
    // altitude at the end, after scaling, to set it as the z coordinate in the
    // ENU frame.
    Eigen::Vector3d xyz = GPSTransform(GPSTransform::WGS84)
                                  .EllToXYZ({Eigen::Vector3d(lat, lon, 0.0)})[0];
    tform.TransformPoint(&xyz);
    xyz(2) = tform.Scale() * alt;
    return xyz;
}

void WriteBoundingBox(const std::string& reconstruction_path,
                      const std::pair<Eigen::Vector3d, Eigen::Vector3d>& bounds,
                      const std::string& suffix = "") {
    const Eigen::Vector3d extent = bounds.second - bounds.first;
    // write axis-aligned bounding box
    {
        const std::string path =
                JoinPaths(reconstruction_path, "bbox_aligned" + suffix + ".txt");
        std::ofstream file(path, std::ios::trunc);
        CHECK(file.is_open()) << path;

        // Ensure that we don't loose any precision by storing in text.
        file.precision(17);
        file << bounds.first.transpose() << std::endl;
        file << bounds.second.transpose() << std::endl;
    }
    // write oriented bounding box
    {
        const std::string path =
                JoinPaths(reconstruction_path, "bbox_oriented" + suffix + ".txt");
        std::ofstream file(path, std::ios::trunc);
        CHECK(file.is_open()) << path;

        // Ensure that we don't loose any precision by storing in text.
        file.precision(17);
        const Eigen::Vector3d center = (bounds.first + bounds.second) * 0.5;
        file << center.transpose() << std::endl << std::endl;
        file << "1 0 0\n0 1 0\n0 0 1" << std::endl << std::endl;
        file << extent.transpose() << std::endl;
    }
}

std::vector<Eigen::Vector3d> ConvertCameraLocations(
        const bool ref_is_gps,
        const std::string& alignment_type,
        const std::vector<Eigen::Vector3d>& ref_locations) {
    if (ref_is_gps) {
        const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
        if (alignment_type != "enu") {
            LOG(INFO) << "Converting Alignment Coordinates from GPS "
                         "(lat/lon/alt) to ECEF.";
            return gps_transform.EllipsoidToECEF(ref_locations);
        } else {
            THROW_CHECK(!ref_locations.empty());
            LOG(INFO) << "Converting Alignment Coordinates from GPS "
                         "(lat/lon/alt) to ENU. Using the first GPS coordinate "
                         "as the ENU origin: "
                      << ref_locations[0].transpose();
            return gps_transform.EllipsoidToENU(ref_locations,
                                                ref_locations[0](0),
                                                ref_locations[0](1),
                                                ref_locations[0](2));
        }
    } else {
        LOG(INFO) << "Cartesian Alignment Coordinates extracted (MUST NOT BE "
                     "GPS coords!).";
        return ref_locations;
    }
}

void ReadFileCameraLocations(const std::filesystem::path& ref_images_path,
                             const bool ref_is_gps,
                             const std::string& alignment_type,
                             std::vector<std::string>* ref_image_names,
                             std::vector<Eigen::Vector3d>* ref_locations) {
    for (const auto& line : ReadTextFileLines(ref_images_path)) {
        std::stringstream line_parser(line);
        line_parser.imbue(std::locale::classic());
        std::string image_name;
        Eigen::Vector3d camera_position;
        THROW_CHECK(line_parser >> image_name >> camera_position[0] >>
                    camera_position[1] >> camera_position[2]);
        ref_image_names->push_back(image_name);
        ref_locations->push_back(camera_position);
    }

    *ref_locations =
            ConvertCameraLocations(ref_is_gps, alignment_type, *ref_locations);
}

void ReadDatabaseCameraLocations(const std::filesystem::path& database_path,
                                 const bool ref_is_gps,
                                 const std::string& alignment_type,
                                 std::vector<std::string>* ref_image_names,
                                 std::vector<Eigen::Vector3d>* ref_locations) {
    auto database = Database::Open(database_path);

    // Index pose priors by their associated data ID.
    NodeHashMap<data_t, PosePrior> pose_priors_by_data_id;
    for (const auto& pose_prior : database->ReadAllPosePriors()) {
        pose_priors_by_data_id.emplace(pose_prior.corr_data_id, pose_prior);
    }

    for (const auto& image : database->ReadAllImages()) {
        const auto it = pose_priors_by_data_id.find(image.DataId());
        if (it != pose_priors_by_data_id.end()) {
            ref_image_names->push_back(image.Name());
            const auto& pose_prior = it->second;
            if (ref_is_gps) {
                THROW_CHECK_EQ(
                        static_cast<int>(pose_prior.coordinate_system),
                        static_cast<int>(PosePrior::CoordinateSystem::WGS84));
            }
            ref_locations->push_back(pose_prior.position);
        }
    }

    *ref_locations =
            ConvertCameraLocations(ref_is_gps, alignment_type, *ref_locations);
}

void WriteComparisonErrorsCSV(
        const std::filesystem::path& path,
        const std::vector<ImageAlignmentError>& errors) {
    std::ofstream file(path, std::ios::trunc);
    THROW_CHECK_FILE_OPEN(file, path);

    file.imbue(std::locale::classic());
    file.precision(17);
    file << "# Model comparison pose errors: one entry per common image\n";
    file << "# <rotation error (deg)>, <proj center error>\n";
    for (size_t i = 0; i < errors.size(); ++i) {
        file << errors[i].rotation_error_deg << ", "
             << errors[i].proj_center_error << '\n';
    }
}

void PrintErrorStats(std::ostream& out,
                     const AlignmentErrorSummary::Statistics& stats) {
    out << "Min:    " << stats.min << '\n';
    out << "Max:    " << stats.max << '\n';
    out << "Mean:   " << stats.mean << '\n';
    out << "Median: " << stats.median << '\n';
    out << "P90:    " << stats.p90 << '\n';
    out << "P99:    " << stats.p99 << '\n';
}

void PrintComparisonSummary(std::ostream& out,
                            const std::vector<ImageAlignmentError>& errors) {
    if (errors.empty()) {
        out << "Cannot extract error statistics from empty input\n";
        return;
    }
    AlignmentErrorSummary summary = AlignmentErrorSummary::Compute(errors);
    out << "\nRotation errors (degrees)\n";
    PrintErrorStats(out, summary.rotation_errors_deg);
    out << "\nProjection center errors\n";
    PrintErrorStats(out, summary.proj_center_errors);
}

}  // namespace

int RunModelAligner(int argc, char** argv) {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::filesystem::path database_path;
    std::filesystem::path ref_model_path;
    std::filesystem::path ref_images_path;
    bool ref_is_gps = true;
    bool merge_origins = false;
    std::filesystem::path transform_path;
    std::string alignment_type = "custom";
    int min_common_images = 3;
    RANSACOptions ransac_options;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddDefaultOption("database_path", &database_path);
    options.AddDefaultOption("ref_model_path", &ref_model_path);
    options.AddDefaultOption("ref_images_path", &ref_images_path);
    options.AddDefaultOption("ref_is_gps", &ref_is_gps);
    options.AddDefaultOption("merge_image_and_ref_origins", &merge_origins);
    options.AddDefaultOption("transform_path", &transform_path);
    options.AddDefaultOption(
            "alignment_type",
            &alignment_type,
            "{plane, ecef, enu, enu-plane, enu-plane-unscaled, custom}");
    options.AddDefaultOption("min_common_images", &min_common_images);
    options.AddDefaultOption("alignment_max_error", &ransac_options.max_error);
    options.Parse(argc, argv);

    StringToLower(&alignment_type);
    const FlatHashSet<std::string> alignment_options{
            "plane", "ecef", "enu", "enu-plane", "enu-plane-unscaled",
            "custom"};
    if (alignment_options.count(alignment_type) == 0) {
        LOG(ERROR) << "Invalid `alignment_type` - supported values are "
                      "{'plane', 'ecef', 'enu', 'enu-plane', "
                      "'enu-plane-unscaled', 'custom'}";
        return EXIT_FAILURE;
    }

    if (ransac_options.max_error <= 0) {
        LOG(ERROR) << "You must provide a maximum alignment error > 0";
        return EXIT_FAILURE;
    }

    if (ref_model_path.empty() && database_path.empty() &&
        ref_images_path.empty() && alignment_type != "plane") {
        LOG(ERROR) << "One of the following arguments must be specified: "
                      "--ref_model_path, --database_path, --ref_images_path, "
                      "--alignment_type=plane";
        return EXIT_FAILURE;
    }

    std::vector<std::string> ref_image_names;
    std::vector<Eigen::Vector3d> ref_locations;
    if (!ref_model_path.empty() && database_path.empty()) {
        Reconstruction ref_reconstruction;
        ref_reconstruction.Read(ref_model_path);
        for (const auto& image : ref_reconstruction.Images()) {
            ref_image_names.push_back(image.second.Name());
            ref_locations.push_back(image.second.ProjectionCenter());
        }
    } else if (!ref_images_path.empty() && database_path.empty()) {
        ReadFileCameraLocations(ref_images_path,
                                ref_is_gps,
                                alignment_type,
                                &ref_image_names,
                                &ref_locations);
    } else if (!database_path.empty() && ref_images_path.empty()) {
        ReadDatabaseCameraLocations(database_path,
                                    ref_is_gps,
                                    alignment_type,
                                    &ref_image_names,
                                    &ref_locations);
    } else if (alignment_type != "plane") {
        LOG(ERROR) << "Use location file or database, not both";
        return EXIT_FAILURE;
    }

    if (alignment_type != "plane" &&
        static_cast<int>(ref_locations.size()) < min_common_images) {
        LOG(ERROR) << "Cannot align with insufficient reference locations.";
        return EXIT_FAILURE;
    }

    Reconstruction reconstruction;
    reconstruction.Read(input_path);
    Sim3d tform;

    if (alignment_type == "plane") {
        PrintHeading2("Aligning reconstruction to principal plane");
        AlignToPrincipalPlane(&reconstruction, &tform);
    } else {
        PrintHeading2("Aligning reconstruction to " + alignment_type);
        LOG(INFO) << StringPrintf("=> Using %d reference images",
                                  ref_image_names.size());

        const bool alignment_success = AlignReconstructionToLocations(
                reconstruction,
                ref_image_names,
                ref_locations,
                min_common_images,
                ransac_options,
                &tform);

        if (!alignment_success) {
            LOG(ERROR) << "=> Alignment failed";
            return EXIT_FAILURE;
        }

        reconstruction.Transform(tform);

        std::vector<double> errors;
        errors.reserve(ref_image_names.size());

        for (size_t i = 0; i < ref_image_names.size(); ++i) {
            const Image* image =
                    reconstruction.FindImageWithName(ref_image_names[i]);
            if (image != nullptr) {
                errors.push_back(
                        (image->ProjectionCenter() - ref_locations[i]).norm());
            }
        }
        LOG(INFO) << StringPrintf("=> Alignment error: %f (mean), %f (median)",
                                  Mean(errors),
                                  Median(errors));

        if (StringStartsWith(alignment_type, "enu-plane")) {
            PrintHeading2("Aligning ECEF aligned reconstruction to ENU plane");
            AlignToENUPlane(
                    &reconstruction, &tform, alignment_type ==
                    "enu-plane-unscaled");
        }
    }

    if (merge_origins) {
        for (size_t i = 0; i < ref_image_names.size(); i++) {
            const Image* first_image =
                    reconstruction.FindImageWithName(ref_image_names[i]);

            if (first_image != nullptr) {
                const Eigen::Vector3d& first_img_position = ref_locations[i];
                const Eigen::Vector3d trans_align =
                        first_img_position - first_image->ProjectionCenter();
                const Sim3d origin_align(
                        1.0, Eigen::Quaterniond::Identity(), trans_align);

                LOG(INFO) << "\n Aligning reconstruction's origin with ref "
                             "origin: "
                          << first_img_position.transpose() << '\n';

                reconstruction.Transform(origin_align);

                // Update the Sim3 transformation in case it is stored next.
                tform = Sim3d(tform.scale(),
                              tform.rotation(),
                              tform.translation() + trans_align);

                break;
            }
        }
    }

    LOG(INFO) << "=> Alignment succeeded";
    reconstruction.Write(output_path);
    if (!transform_path.empty()) {
        tform.ToFile(transform_path);
    }

    return EXIT_SUCCESS;
}

int RunModelAnalyzer(int argc, char** argv) {
    std::string path;

    OptionManager options;
    options.AddRequiredOption("path", &path);
    options.Parse(argc, argv);

    Reconstruction reconstruction;
    reconstruction.Read(path);

    std::cout << StringPrintf("Cameras: %d", reconstruction.NumCameras())
              << std::endl;
    std::cout << StringPrintf("Images: %d", reconstruction.NumImages())
              << std::endl;
    std::cout << StringPrintf("Registered images: %d",
                              reconstruction.NumRegImages())
              << std::endl;
    std::cout << StringPrintf("Points: %d", reconstruction.NumPoints3D())
              << std::endl;
    std::cout << StringPrintf("Observations: %d",
                              reconstruction.ComputeNumObservations())
              << std::endl;
    std::cout << StringPrintf("Mean track length: %f",
                              reconstruction.ComputeMeanTrackLength())
              << std::endl;
    std::cout << StringPrintf("Mean observations per image: %f",
                              reconstruction.ComputeMeanObservationsPerRegImage())
              << std::endl;
    std::cout << StringPrintf("Mean reprojection error: %fpx",
                              reconstruction.ComputeMeanReprojectionError())
              << std::endl;

    return EXIT_SUCCESS;
}

bool CompareModels(const Reconstruction& reconstruction1,
                   const Reconstruction& reconstruction2,
                   const std::string& alignment_error,
                   const double min_inlier_observations,
                   const double max_reproj_error,
                   const double max_proj_center_error,
                   std::vector<ImageAlignmentError>& errors,
                   Sim3d& rec2_from_rec1) {
    PrintHeading1("Reconstruction 1");
    LOG(INFO) << StringPrintf("Frames: %d", reconstruction1.NumRegFrames());
    LOG(INFO) << StringPrintf("Images: %d", reconstruction1.NumRegImages());
    LOG(INFO) << StringPrintf("Points: %d", reconstruction1.NumPoints3D());

    PrintHeading1("Reconstruction 2");
    LOG(INFO) << StringPrintf("Frames: %d", reconstruction2.NumRegFrames());
    LOG(INFO) << StringPrintf("Images: %d", reconstruction2.NumRegImages());
    LOG(INFO) << StringPrintf("Points: %d", reconstruction2.NumPoints3D());

    PrintHeading1("Comparing reconstructed image poses");
    const std::vector<std::pair<image_t, image_t>> common_image_ids =
            reconstruction1.FindCommonRegImageIds(reconstruction2);
    LOG(INFO) << StringPrintf("Common images: %d", common_image_ids.size());

    bool success = false;
    if (alignment_error == "reprojection") {
        success = AlignReconstructionsViaReprojections(
                reconstruction1,
                reconstruction2,
                /*min_inlier_observations=*/min_inlier_observations,
                /*max_reproj_error=*/max_reproj_error,
                &rec2_from_rec1);
    } else if (alignment_error == "proj_center") {
        success = AlignReconstructionsViaProjCenters(
                reconstruction1,
                reconstruction2,
                /*max_proj_center_error=*/max_proj_center_error,
                &rec2_from_rec1);
    } else {
        LOG(ERROR) << "Invalid alignment_error specified.";
        return false;
    }

    if (!success) {
        LOG(INFO) << "=> Reconstruction alignment failed";
        return false;
    }

    LOG(INFO) << "Computed alignment transform:\n"
              << rec2_from_rec1.ToMatrix();

    errors = ComputeImageAlignmentError(
            reconstruction1, reconstruction2, rec2_from_rec1);

    PrintHeading2("Image alignment error summary");
    PrintComparisonSummary(std::cout, errors);

    return true;
}

int RunModelComparer(int argc, char** argv) {
    std::filesystem::path input_path1;
    std::filesystem::path input_path2;
    std::filesystem::path output_path;
    std::string alignment_error = "reprojection";
    double min_inlier_observations = 0.3;
    double max_reproj_error = 8.0;
    double max_proj_center_error = 0.1;

    OptionManager options;
    options.AddRequiredOption("input_path1", &input_path1);
    options.AddRequiredOption("input_path2", &input_path2);
    options.AddDefaultOption("output_path", &output_path);
    options.AddDefaultOption(
            "alignment_error", &alignment_error, "{reprojection, proj_center}");
    options.AddDefaultOption("min_inlier_observations",
                             &min_inlier_observations);
    options.AddDefaultOption("max_reproj_error", &max_reproj_error);
    options.AddDefaultOption("max_proj_center_error", &max_proj_center_error);
    options.Parse(argc, argv);

    if (!output_path.empty() && !ExistsDir(output_path)) {
        LOG(ERROR) << "Provided output path is not a valid directory";
        return EXIT_FAILURE;
    }

    Reconstruction reconstruction1;
    reconstruction1.Read(input_path1);
    Reconstruction reconstruction2;
    reconstruction2.Read(input_path2);
    std::vector<ImageAlignmentError> errors;
    Sim3d rec2_from_rec1;
    const bool success = CompareModels(reconstruction1,
                                       reconstruction2,
                                       alignment_error,
                                       min_inlier_observations,
                                       max_reproj_error,
                                       max_proj_center_error,
                                       errors,
                                       rec2_from_rec1);
    if (!success) {
        return EXIT_FAILURE;
    }
    if (!output_path.empty()) {
        const auto errors_path = output_path / "errors.csv";
        WriteComparisonErrorsCSV(errors_path, errors);
        const auto summary_path = output_path / "errors_summary.txt";
        std::ofstream file(summary_path, std::ios::trunc);
        THROW_CHECK_FILE_OPEN(file, summary_path);
        PrintComparisonSummary(file, errors);
    }
    return EXIT_SUCCESS;
}

int RunModelClusterer(int argc, char** argv) {
    std::filesystem::path input_path;
    std::filesystem::path output_path;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddReconstructionClustererOptions();
    options.Parse(argc, argv);

    if (!ExistsDir(input_path)) {
        LOG(ERROR) << "`input_path` is not a directory";
        return EXIT_FAILURE;
    }

    if (!ExistsDir(output_path)) {
        LOG(ERROR) << "`output_path` is not a directory";
        return EXIT_FAILURE;
    }

    PrintHeading1("Loading model");
    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->Read(input_path);

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();

    ReconstructionClustererController controller(
            *options.reconstruction_clusterer,
            reconstruction,
            reconstruction_manager);
    controller.Run();

    PrintHeading1("Writing clustered model(s)");
    reconstruction_manager->Write(output_path);

    return EXIT_SUCCESS;
}

int RunModelConverter(int argc, char** argv) {
    std::string input_path;
    std::string output_path;
    std::string output_type;
    bool skip_distortion = false;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddRequiredOption("output_type", &output_type,
                              "{BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}");
    options.AddDefaultOption("skip_distortion", &skip_distortion);
    options.Parse(argc, argv);

    Reconstruction reconstruction;
    reconstruction.Read(input_path);

    StringToLower(&output_type);
    if (output_type == "bin") {
        reconstruction.WriteBinary(output_path);
    } else if (output_type == "txt") {
        reconstruction.WriteText(output_path);
    } else if (output_type == "nvm") {
        ExportNVM(reconstruction, output_path, skip_distortion);
    } else if (output_type == "bundler") {
        ExportBundler(reconstruction, output_path + ".bundle.out",
                            output_path + ".list.txt", skip_distortion);
    } else if (output_type == "r3d") {
        ExportRecon3D(reconstruction, output_path, skip_distortion);
    } else if (output_type == "cam") {
        ExportCam(reconstruction, output_path, skip_distortion);
    } else if (output_type == "ply") {
        ExportPLY(reconstruction, output_path);
    } else if (output_type == "vrml") {
        const auto base_path = output_path.substr(0, output_path.find_last_of("."));
        ExportVRML(reconstruction, base_path + ".images.wrl",
                           base_path + ".points3D.wrl", 1,
                           Eigen::Vector3d(1, 0, 0));
    } else {
        std::cerr << "ERROR: Invalid `output_type`" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RunModelCropper(int argc, char** argv) {
    Timer timer;
    timer.Start();

    std::string input_path;
    std::string output_path;
    std::string boundary;
    std::string gps_transform_path;
    bool is_gps = false;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddRequiredOption("boundary", &boundary);
    options.AddDefaultOption("gps_transform_path", &gps_transform_path);
    options.Parse(argc, argv);

    if (!ExistsDir(input_path)) {
        std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
        return EXIT_FAILURE;
    }

    if (!ExistsDir(output_path)) {
        std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<double> boundary_elements = CSVToVector<double>(boundary);
    if (boundary_elements.size() != 2 && boundary_elements.size() != 6) {
        std::cerr << "ERROR: Invalid `boundary` - supported values are "
                     "'x1,y1,z1,x2,y2,z2' or 'p1,p2'."
                  << std::endl;
        return EXIT_FAILURE;
    }

    Reconstruction reconstruction;
    reconstruction.Read(input_path);

    PrintHeading2("Calculating boundary coordinates");
    std::pair<Eigen::Vector3d, Eigen::Vector3d> bounding_box;
    if (boundary_elements.size() == 6) {
        SimilarityTransform3 tform;
        if (!gps_transform_path.empty()) {
            PrintHeading2("Reading model to ECEF transform");
            is_gps = true;
            tform = SimilarityTransform3::FromFile(gps_transform_path).Inverse();
        }
        bounding_box.first =
                is_gps ? TransformLatLonAltToModelCoords(tform, boundary_elements[0],
                                                         boundary_elements[1],
                                                         boundary_elements[2])
                       : Eigen::Vector3d(boundary_elements[0], boundary_elements[1],
                                         boundary_elements[2]);
        bounding_box.second =
                is_gps ? TransformLatLonAltToModelCoords(tform, boundary_elements[3],
                                                         boundary_elements[4],
                                                         boundary_elements[5])
                       : Eigen::Vector3d(boundary_elements[3], boundary_elements[4],
                                         boundary_elements[5]);
    } else {
        bounding_box = reconstruction.ComputeBoundingBox(boundary_elements[0],
                                                         boundary_elements[1]);
    }

    PrintHeading2("Cropping reconstruction");
    reconstruction.Crop(bounding_box).Write(output_path);
    WriteBoundingBox(output_path, bounding_box);

    std::cout << "=> Cropping succeeded" << std::endl;
    timer.PrintMinutes();
    return EXIT_SUCCESS;
}

int RunModelMerger(int argc, char** argv) {
    std::string input_path1;
    std::string input_path2;
    std::string output_path;
    double max_reproj_error = 64.0;

    OptionManager options;
    options.AddRequiredOption("input_path1", &input_path1);
    options.AddRequiredOption("input_path2", &input_path2);
    options.AddRequiredOption("output_path", &output_path);
    options.AddDefaultOption("max_reproj_error", &max_reproj_error);
    options.Parse(argc, argv);

    Reconstruction reconstruction1;
    reconstruction1.Read(input_path1);
    PrintHeading2("Reconstruction 1");
    std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
              << std::endl;
    std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
              << std::endl;

    Reconstruction reconstruction2;
    reconstruction2.Read(input_path2);
    PrintHeading2("Reconstruction 2");
    std::cout << StringPrintf("Images: %d", reconstruction2.NumRegImages())
              << std::endl;
    std::cout << StringPrintf("Points: %d", reconstruction2.NumPoints3D())
              << std::endl;

    PrintHeading2("Merging reconstructions");
    if (reconstruction1.Merge(reconstruction2, max_reproj_error)) {
        std::cout << "=> Merge succeeded" << std::endl;
        PrintHeading2("Merged reconstruction");
        std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
                  << std::endl;
        std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
                  << std::endl;
    } else {
        std::cout << "=> Merge failed" << std::endl;
    }

    reconstruction1.Write(output_path);

    return EXIT_SUCCESS;
}

int RunModelOrientationAligner(int argc, char** argv) {
    std::string input_path;
    std::string output_path;
    std::string method = "MANHATTAN-WORLD";

    ManhattanWorldFrameEstimationOptions frame_estimation_options;

    OptionManager options;
    options.AddImageOptions();
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddDefaultOption("method", &method,
                             "{MANHATTAN-WORLD, IMAGE-ORIENTATION}");
    options.AddDefaultOption("max_image_size",
                             &frame_estimation_options.max_image_size);
    options.Parse(argc, argv);

    StringToLower(&method);
    if (method != "manhattan-world" && method != "image-orientation") {
        std::cout << "WARNING: Invalid `method` - supported values are "
                     "'MANHATTAN-WORLD' or 'IMAGE-ORIENTATION'."
                  << std::endl;
        return EXIT_FAILURE;
    }

    Reconstruction reconstruction;
    reconstruction.Read(input_path);

    PrintHeading1("Aligning Reconstruction");

    Eigen::Matrix3d tform;

    if (method == "manhattan-world") {
        const Eigen::Matrix3d frame = EstimateManhattanWorldFrame(
                frame_estimation_options, reconstruction, *options.image_path);

        if (frame.col(0).nonZeros() == 0) {
            std::cout << "Only aligning vertical axis" << std::endl;
            tform = RotationFromUnitVectors(frame.col(1), Eigen::Vector3d(0, 1, 0));
        } else if (frame.col(1).nonZeros() == 0) {
            tform = RotationFromUnitVectors(frame.col(0), Eigen::Vector3d(1, 0, 0));
            std::cout << "Only aligning horizontal axis" << std::endl;
        } else {
            tform = frame.transpose();
            std::cout << "Aligning horizontal and vertical axes" << std::endl;
        }
    } else if (method == "image-orientation") {
        const Eigen::Vector3d gravity_axis =
                EstimateGravityVectorFromImageOrientation(reconstruction);
        tform = RotationFromUnitVectors(gravity_axis, Eigen::Vector3d(0, 1, 0));
    } else {
        LOG(FATAL) << "Alignment method not supported";
    }

    std::cout << "Using the rotation matrix:" << std::endl;
    std::cout << tform << std::endl;

    reconstruction.Transform(SimilarityTransform3(
            1, RotationMatrixToQuaternion(tform), Eigen::Vector3d(0, 0, 0)));

    std::cout << "Writing aligned reconstruction..." << std::endl;
    reconstruction.Write(output_path);

    return EXIT_SUCCESS;
}

int RunModelSplitter(int argc, char** argv) {
    Timer timer;
    timer.Start();

    std::string input_path;
    std::string output_path;
    std::string split_type;
    std::string split_params;
    std::string gps_transform_path;
    int num_threads = -1;
    int min_reg_images = 10;
    int min_num_points = 100;
    double overlap_ratio = 0.0;
    double min_area_ratio = 0.0;
    bool is_gps = false;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddRequiredOption("split_type", &split_type,
                              "{tiles, extent, parts}");
    options.AddRequiredOption("split_params", &split_params);
    options.AddDefaultOption("gps_transform_path", &gps_transform_path);
    options.AddDefaultOption("num_threads", &num_threads);
    options.AddDefaultOption("min_reg_images", &min_reg_images);
    options.AddDefaultOption("min_num_points", &min_num_points);
    options.AddDefaultOption("overlap_ratio", &overlap_ratio);
    options.AddDefaultOption("min_area_ratio", &min_area_ratio);
    options.Parse(argc, argv);

    if (!ExistsDir(input_path)) {
        std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
        return EXIT_FAILURE;
    }

    if (!ExistsDir(output_path)) {
        std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
        return EXIT_FAILURE;
    }

    if (overlap_ratio < 0) {
        std::cout << "WARN: Invalid `overlap_ratio`; resetting to 0" << std::endl;
        overlap_ratio = 0.0;
    }

    PrintHeading1("Splitting sparse model");
    std::cout << StringPrintf(" => Using \"%s\" split type", split_type.c_str())
              << std::endl;

    Reconstruction reconstruction;
    reconstruction.Read(input_path);

    SimilarityTransform3 tform;
    if (!gps_transform_path.empty()) {
        PrintHeading2("Reading model to ECEF transform");
        is_gps = true;
        tform = SimilarityTransform3::FromFile(gps_transform_path).Inverse();
    }
    const double scale = tform.Scale();

    // Create the necessary number of reconstructions based on the split method
    // and get the bounding boxes for each sub-reconstruction
    PrintHeading2("Computing bound_coords");
    std::vector<std::string> tile_keys;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> exact_bounds;
    StringToLower(&split_type);
    if (split_type == "tiles") {
        std::ifstream file(split_params);
        CHECK(file.is_open()) << split_params;

        double x1, y1, z1, x2, y2, z2;
        std::string tile_key;
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> bounds;
        tile_keys.clear();
        file >> tile_key >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
        while (!file.fail()) {
            tile_keys.push_back(tile_key);
            if (is_gps) {
                exact_bounds.emplace_back(
                        TransformLatLonAltToModelCoords(tform, x1, y1, z1),
                        TransformLatLonAltToModelCoords(tform, x2, y2, z2));
            } else {
                exact_bounds.emplace_back(Eigen::Vector3d(x1, y1, z1),
                                          Eigen::Vector3d(x2, y2, z2));
            }
            file >> tile_key >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
        }
    } else if (split_type == "extent") {
        std::vector<double> parts = CSVToVector<double>(split_params);
        Eigen::Vector3d extent(std::numeric_limits<double>::max(),
                               std::numeric_limits<double>::max(),
                               std::numeric_limits<double>::max());
        for (size_t i = 0; i < parts.size(); ++i) {
            extent(i) = parts[i] * scale;
        }

        const auto bbox = reconstruction.ComputeBoundingBox();
        const Eigen::Vector3d full_extent = bbox.second - bbox.first;
        const Eigen::Vector3i split(
                static_cast<int>(full_extent(0) / extent(0)) + 1,
                static_cast<int>(full_extent(1) / extent(1)) + 1,
                static_cast<int>(full_extent(2) / extent(2)) + 1);

        exact_bounds = ComputeEqualPartsBounds(reconstruction, split);

    } else if (split_type == "parts") {
        auto parts = CSVToVector<int>(split_params);
        Eigen::Vector3i split(1, 1, 1);
        for (size_t i = 0; i < parts.size(); ++i) {
            split(i) = parts[i];
            if (split(i) < 1) {
                std::cerr << "ERROR: Cannot split in less than 1 parts for dim " << i
                          << std::endl;
                return EXIT_FAILURE;
            }
        }
        exact_bounds = ComputeEqualPartsBounds(reconstruction, split);
    } else {
        std::cout << "WARNING: Invalid split type: " << split_type << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> bounds;
    for (const auto& bbox : exact_bounds) {
        const Eigen::Vector3d padding =
                (overlap_ratio * (bbox.second - bbox.first));
        bounds.emplace_back(bbox.first - padding, bbox.second + padding);
    }

    PrintHeading2("Applying split and writing reconstructions");
    const size_t num_parts = bounds.size();
    std::cout << StringPrintf(" => Splitting to %d parts", num_parts)
              << std::endl;

    const bool use_tile_keys = split_type == "tiles";

    auto SplitReconstruction = [&](const int idx) {
        Reconstruction tile_recon = reconstruction.Crop(bounds[idx]);
        // calculate area covered by model as proportion of box area
        auto bbox_extent = bounds[idx].second - bounds[idx].first;
        auto model_bbox = tile_recon.ComputeBoundingBox();
        auto model_extent = model_bbox.second - model_bbox.first;
        double area_ratio =
                (model_extent(0) * model_extent(1)) / (bbox_extent(0) * bbox_extent(1));
        int tile_num_points = tile_recon.NumPoints3D();

        std::string name = use_tile_keys ? tile_keys[idx] : std::to_string(idx);
        const bool include_tile =
                area_ratio >= min_area_ratio &&       //
                tile_num_points >= min_num_points &&  //
                tile_recon.NumRegImages() >= static_cast<size_t>(min_reg_images);

        if (include_tile) {
            std::cout << StringPrintf(
                                 "Writing reconstruction %s with %d images, %d points, "
                                 "and %.2f%% area coverage",
                                 name.c_str(), tile_recon.NumRegImages(), tile_num_points,
                                 100.0 * area_ratio)
                      << std::endl;
            const std::string reconstruction_path = JoinPaths(output_path, name);
            CreateDirIfNotExists(reconstruction_path);
            reconstruction.Write(reconstruction_path);
            WriteBoundingBox(reconstruction_path, bounds[idx]);
            WriteBoundingBox(reconstruction_path, exact_bounds[idx], "_exact");

        } else {
            std::cout << StringPrintf(
                                 "Skipping reconstruction %s with %d images, %d points, "
                                 "and %.2f%% area coverage",
                                 name.c_str(), tile_recon.NumRegImages(), tile_num_points,
                                 100.0 * area_ratio)
                      << std::endl;
        }
    };

    ThreadPool thread_pool(GetEffectiveNumThreads(num_threads));
    for (size_t idx = 0; idx < num_parts; ++idx) {
        thread_pool.AddTask(SplitReconstruction, idx);
    }
    thread_pool.Wait();

    timer.PrintMinutes();
    return EXIT_SUCCESS;
}

int RunModelTransformer(int argc, char** argv) {
    std::string input_path;
    std::string output_path;
    std::string transform_path;
    bool is_inverse = false;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddRequiredOption("transform_path", &transform_path);
    options.AddDefaultOption("is_inverse", &is_inverse);
    options.Parse(argc, argv);

    std::cout << "Reading points input: " << input_path << std::endl;
    Reconstruction recon;
    bool is_dense = false;
    if (HasFileExtension(input_path, ".ply")) {
        is_dense = true;
        recon.ImportPLY(input_path);
    } else if (ExistsDir(input_path)) {
        recon.Read(input_path);
    } else {
        std::cerr << "Invalid model input; not a PLY file or sparse reconstruction "
                     "directory."
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Reading transform input: " << transform_path << std::endl;
    SimilarityTransform3 tform = SimilarityTransform3::FromFile(transform_path);
    if (is_inverse) {
        tform = tform.Inverse();
    }

    std::cout << "Applying transform to recon with " << recon.NumPoints3D()
              << " points" << std::endl;
    recon.Transform(tform);

    std::cout << "Writing output: " << output_path << std::endl;
    if (is_dense) {
        ExportPLY(recon, output_path);
    } else {
        recon.Write(output_path);
    }

    return EXIT_SUCCESS;
}

}  // namespace colmap
