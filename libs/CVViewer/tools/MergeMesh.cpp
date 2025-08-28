// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > MergeMesh source_directory target_file [option]");
    utility::LogInfo("      Merge mesh files under <source_directory>.");
    utility::LogInfo("");
    utility::LogInfo("Options (listed in the order of execution priority):");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4).");
    utility::LogInfo("    --purge                   : Clear duplicated and unreferenced vertices and");
    utility::LogInfo("                                triangles.");
    // clang-format on
}

int main(int argc, char **argv) {
    using namespace cloudViewer;
    using namespace cloudViewer::utility::filesystem;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc <= 2 || utility::ProgramOptionExists(argc, argv, "--help")) {
        PrintHelp();
        return 0;
    }
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);

    std::string directory(argv[1]);
    std::vector<std::string> filenames;
    ListFilesInDirectory(directory, filenames);

	ccPointCloud* merged_baseVertex = new ccPointCloud("vertices");
	assert(merged_baseVertex);
	merged_baseVertex->setEnabled(false);
	merged_baseVertex->setLocked(false);
    auto merged_mesh_ptr = cloudViewer::make_shared<ccMesh>(merged_baseVertex);
	merged_mesh_ptr->addChild(merged_baseVertex);
    for (const auto &filename : filenames) {
		ccPointCloud* baseVertices = new ccPointCloud("vertices");
		assert(baseVertices);
		baseVertices->setEnabled(false);
		// DGM: no need to lock it as it is only used by one mesh!
		baseVertices->setLocked(false);
        auto mesh_ptr = cloudViewer::make_shared<ccMesh>(baseVertices);
		mesh_ptr->addChild(baseVertices);
        if (cloudViewer::io::ReadTriangleMesh(filename, *mesh_ptr)) {
			//do some cleaning
			{
				baseVertices->shrinkToFit();
				mesh_ptr->shrinkToFit();
				NormsIndexesTableType* normals = mesh_ptr->getTriNormsTable();
				if (normals)
				{
					normals->shrink_to_fit();
				}
			}

            *merged_mesh_ptr += *mesh_ptr;
        }
    }

    if (utility::ProgramOptionExists(argc, argv, "--purge")) {
        merged_mesh_ptr->shrinkToFit();
    }
    cloudViewer::io::WriteTriangleMesh(argv[2], *merged_mesh_ptr);

    return 1;
}
