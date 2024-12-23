// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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

#include "CloudViewer.h"

using namespace cloudViewer;

void PrintVoxelGridInformation(const geometry::VoxelGrid& voxel_grid) {
    utility::LogInfo("geometry::VoxelGrid with {:d} voxels",
                     voxel_grid.voxels_.size());
    utility::LogInfo("               origin: [{:f} {:f} {:f}]",
                     voxel_grid.origin_(0), voxel_grid.origin_(1),
                     voxel_grid.origin_(2));
    utility::LogInfo("               voxel_size: {:f}", voxel_grid.voxel_size_);
    return;
}

void PrintHelp() {
    using namespace cloudViewer;

    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Voxelization [pointcloud_filename] [voxel_filename_ply]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace cloudViewer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    auto voxel = geometry::VoxelGrid::CreateFromPointCloud(*pcd, 0.05);
    PrintVoxelGridInformation(*voxel);
    visualization::DrawGeometries({pcd, voxel});
    io::WriteVoxelGrid(argv[2], *voxel, true);

    auto voxel_read = io::CreateVoxelGridFromFile(argv[2]);
    PrintVoxelGridInformation(*voxel_read);
    visualization::DrawGeometries({pcd, voxel_read});
}
