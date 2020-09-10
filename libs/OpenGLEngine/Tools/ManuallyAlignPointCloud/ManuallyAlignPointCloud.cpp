// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include "Tools/ManuallyAlignPointCloud/VisualizerForAlignment.h"

#include <Visualization/Utility/SelectionPolygonVolume.h>
#include <Registration/ColoredICP.h>
#include <thread>

void PrintTransformation(const Eigen::Matrix4d &transformation) {
    using namespace cloudViewer;
    CVLib::utility::LogInfo("Current transformation is:");
    CVLib::utility::LogInfo("\t{:.6f} {:.6f} {:.6f} {:.6f}", transformation(0, 0),
                     transformation(0, 1), transformation(0, 2),
                     transformation(0, 3));
    CVLib::utility::LogInfo("\t{:.6f} {:.6f} {:.6f} {:.6f}", transformation(1, 0),
                     transformation(1, 1), transformation(1, 2),
                     transformation(1, 3));
    CVLib::utility::LogInfo("\t{:.6f} {:.6f} {:.6f} {:.6f}", transformation(2, 0),
                     transformation(2, 1), transformation(2, 2),
                     transformation(2, 3));
    CVLib::utility::LogInfo("\t{:.6f} {:.6f} {:.6f} {:.6f}", transformation(3, 0),
                     transformation(3, 1), transformation(3, 2),
                     transformation(3, 3));
}

void PrintHelp() {
    using namespace cloudViewer;
    // PrintOpen3DVersion();
    // clang-format off
    CVLib::utility::LogInfo("Usage:");
    CVLib::utility::LogInfo("    > ManuallyAlignPointCloud source_file target_file [options]");
    CVLib::utility::LogInfo("      Manually align point clouds in source_file and target_file.");
    CVLib::utility::LogInfo("");
    CVLib::utility::LogInfo("Options:");
    CVLib::utility::LogInfo("    --help, -h                : Print help information.");
    CVLib::utility::LogInfo("    --verbose n               : Set verbose level (0-4).");
    CVLib::utility::LogInfo("    --voxel_size d            : Set downsample voxel size.");
    CVLib::utility::LogInfo("    --max_corres_distance d   : Set max correspondence distance.");
    CVLib::utility::LogInfo("    --without_scaling         : Disable scaling in transformations.");
    CVLib::utility::LogInfo("    --without_dialog          : Disable dialogs. Default files will be used.");
    CVLib::utility::LogInfo("    --without_gui_icp file    : The program runs as a console command. No window");
    CVLib::utility::LogInfo("                                will be created. The program reads an alignment");
    CVLib::utility::LogInfo("                                from file. It does cropping, downsample, ICP,");
    CVLib::utility::LogInfo("                                then saves the alignment session into file.");
    CVLib::utility::LogInfo("    --without_gui_eval file   : The program runs as a console command. No window");
    CVLib::utility::LogInfo("                                will be created. The program reads an alignment");
    CVLib::utility::LogInfo("                                from file. It does cropping, downsample,");
    CVLib::utility::LogInfo("                                evaluation, then saves everything.");
    // clang-format on
}

int main(int argc, char **argv) {
    using namespace cloudViewer;

    if (argc < 3 || CVLib::utility::ProgramOptionExists(argc, argv, "--help") ||
        CVLib::utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 0;
    }

    int verbose = CVLib::utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    CVLib::utility::SetVerbosityLevel((CVLib::utility::VerbosityLevel)verbose);
    double voxel_size =
            CVLib::utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", -1.0);
    double max_corres_distance = CVLib::utility::GetProgramOptionAsDouble(
            argc, argv, "--max_corres_distance", -1.0);
    bool with_scaling =
            !CVLib::utility::ProgramOptionExists(argc, argv, "--without_scaling");
    bool with_dialog =
            !CVLib::utility::ProgramOptionExists(argc, argv, "--without_dialog");
    std::string default_polygon_filename =
            CVLib::utility::filesystem::GetFileNameWithoutExtension(argv[2]) + ".json";
    std::string alignment_filename = CVLib::utility::GetProgramOptionAsString(
            argc, argv, "--without_gui_icp", "");
    std::string eval_filename = CVLib::utility::GetProgramOptionAsString(
            argc, argv, "--without_gui_eval", "");
    std::string default_directory =
            CVLib::utility::filesystem::GetFileParentDirectory(argv[1]);

    auto source_ptr = io::CreatePointCloudFromFile(argv[1]);
    auto target_ptr = io::CreatePointCloudFromFile(argv[2]);
    if (!source_ptr->hasPoints() || !target_ptr->hasPoints()) {
        CVLib::utility::LogWarning("Failed to read one of the point clouds.");
        return 1;
    }

    if (!alignment_filename.empty()) {
        AlignmentSession session;
        if (io::ReadIJsonConvertible(alignment_filename, session) == false) {
            return 0;
        }
        session.voxel_size_ = voxel_size;
        session.max_correspondence_distance_ = max_corres_distance;
        source_ptr->transform(session.transformation_);
        auto polygon_volume =
                std::make_shared<visualization::SelectionPolygonVolume>();
        if (io::ReadIJsonConvertible(default_polygon_filename,
                                     *polygon_volume)) {
            CVLib::utility::LogInfo("Crop point cloud.");
            source_ptr = polygon_volume->CropPointCloud(*source_ptr);
        }
        if (voxel_size > 0.0) {
            CVLib::utility::LogInfo("Downsample point cloud with voxel size {:.4f}.",
                             voxel_size);
            source_ptr = source_ptr->voxelDownSample(voxel_size);
        }
        if (max_corres_distance > 0.0) {
            CVLib::utility::LogInfo("ICP with max correspondence distance {:.4f}.",
                             max_corres_distance);
            auto result = registration::RegistrationICP(
                    *source_ptr, *target_ptr, max_corres_distance,
                    Eigen::Matrix4d::Identity(),
                    registration::TransformationEstimationPointToPoint(true),
                    registration::ICPConvergenceCriteria(1e-6, 1e-6, 30));
            CVLib::utility::LogInfo(
                    "Registration finished with fitness {:.4f} and RMSE "
                    "{:.4f}.",
                    result.fitness_, result.inlier_rmse_);
            if (result.fitness_ > 0.0) {
                session.transformation_ =
                        result.transformation_ * session.transformation_;
                PrintTransformation(session.transformation_);
                source_ptr->transform(result.transformation_);
            }
        }
        io::WriteIJsonConvertible(alignment_filename, session);
        return 1;
    }

    if (!eval_filename.empty()) {
        AlignmentSession session;
        if (io::ReadIJsonConvertible(eval_filename, session) == false) {
            return 0;
        }
        source_ptr->transform(session.transformation_);
        auto polygon_volume =
                std::make_shared<visualization::SelectionPolygonVolume>();
        if (io::ReadIJsonConvertible(default_polygon_filename,
                                     *polygon_volume)) {
            CVLib::utility::LogInfo("Crop point cloud.");
            source_ptr = polygon_volume->CropPointCloud(*source_ptr);
        }
        if (voxel_size > 0.0) {
            CVLib::utility::LogInfo("Downsample point cloud with voxel size {:.4f}.",
                             voxel_size);
            source_ptr = source_ptr->voxelDownSample(voxel_size);
        }
        std::string source_filename =
                CVLib::utility::filesystem::GetFileNameWithoutExtension(
                        eval_filename) +
                ".source.ply";
        std::string target_filename =
                CVLib::utility::filesystem::GetFileNameWithoutExtension(
                        eval_filename) +
                ".target.ply";
        std::string source_binname =
                CVLib::utility::filesystem::GetFileNameWithoutExtension(
                        eval_filename) +
                ".source.bin";
        std::string target_binname =
                CVLib::utility::filesystem::GetFileNameWithoutExtension(
                        eval_filename) +
                ".target.bin";
        FILE *f;

        io::WritePointCloud(source_filename, *source_ptr);
        auto source_dis = source_ptr->computePointCloudDistance(*target_ptr);
        f = CVLib::utility::filesystem::FOpen(source_binname, "wb");
        fwrite(source_dis.data(), sizeof(double), source_dis.size(), f);
        fclose(f);
        io::WritePointCloud(target_filename, *target_ptr);
        auto target_dis = target_ptr->computePointCloudDistance(*source_ptr);
        f = CVLib::utility::filesystem::FOpen(target_binname, "wb");
        fwrite(target_dis.data(), sizeof(double), target_dis.size(), f);
        fclose(f);
        return 1;
    }

    visualization::VisualizerWithEditing vis_source, vis_target;
    VisualizerForAlignment vis_main(vis_source, vis_target, voxel_size,
                                    max_corres_distance, with_scaling,
                                    with_dialog, default_polygon_filename,
                                    default_directory);

    vis_source.CreateVisualizerWindow("Source Point Cloud", 1280, 720, 10, 100);
    vis_source.AddGeometry(source_ptr);
    if (source_ptr->size() > 5000000) {
        vis_source.GetRenderOption().point_size_ = 1.0;
    }
    vis_source.BuildUtilities();
    vis_target.CreateVisualizerWindow("Target Point Cloud", 1280, 720, 10, 880);
    vis_target.AddGeometry(target_ptr);
    if (target_ptr->size() > 5000000) {
        vis_target.GetRenderOption().point_size_ = 1.0;
    }
    vis_target.BuildUtilities();
    vis_main.CreateVisualizerWindow("Alignment", 1280, 1440, 1300, 100);
    vis_main.AddSourceAndTarget(source_ptr, target_ptr);
    vis_main.BuildUtilities();

    while (vis_source.PollEvents() && vis_target.PollEvents() &&
           vis_main.PollEvents()) {
    }

    vis_source.DestroyVisualizerWindow();
    vis_target.DestroyVisualizerWindow();
    vis_main.DestroyVisualizerWindow();
    return 0;
}
