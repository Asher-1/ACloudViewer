// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <iostream>
#include <memory>
#include <thread>

#include "CloudViewer.h"

void PrintUsage() {
    using namespace CVLib;
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Visualizer [mesh|spin|slowspin|pointcloud|rainbow|image|depth|editing|editmesh] [filename]");
    utility::LogInfo("    > Visualizer [animation] [filename] [trajectoryfile]");
    utility::LogInfo("    > Visualizer [rgbd] [color] [depth] [--rgbd_type]");
    // clang-format on
}

/* just for test */
ccPointCloud readPoints(const char* file)
{
	ccPointCloud cloud;
	if (cloudViewer::io::ReadPointCloud(file, cloud)) {
		CVLib::utility::LogInfo("Successfully read {}", file);
	}
	else {
		CVLib::utility::LogWarning("Failed to read {}", file);
		return ccPointCloud();
	}
	return cloud;
}

ccMesh readMeshes(const char* file)
{
	ccPointCloud* baseVertices = new ccPointCloud("vertices");
	assert(baseVertices);
	baseVertices->setEnabled(false);
	// DGM: no need to lock it as it is only used by one mesh!
	baseVertices->setLocked(false);
	ccMesh mesh(baseVertices);
	mesh.addChild(baseVertices);
	if (cloudViewer::io::ReadTriangleMesh(file, mesh)) {
		CVLib::utility::LogInfo("Successfully read {}", file);
	}
	else {
		CVLib::utility::LogWarning("Failed to read {}", file);
		return ccMesh();
	}
	return mesh;
}
/* just for test */

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

    CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Debug);
    if (argc < 3) {
        PrintUsage();
        return 1;
    }

    std::string option(argv[1]);
    if (option == "mesh") {
		ccPointCloud* baseVertices = new ccPointCloud("vertices");
		assert(baseVertices);
		baseVertices->setEnabled(false);
		// DGM: no need to lock it as it is only used by one mesh!
		baseVertices->setLocked(false);
		auto mesh_ptr = std::make_shared<ccMesh>(baseVertices);
		mesh_ptr->addChild(baseVertices);
		if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
			CVLib::utility::LogInfo("Successfully read {}", argv[2]);
		}
		else {
			CVLib::utility::LogWarning("Failed to read {}", argv[2]);
			return 1;
		}

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
        mesh_ptr->computeVertexNormals();
        visualization::DrawGeometries({mesh_ptr}, "Mesh", 1600, 900);
    } 
	else if (option == "editmesh") {
		ccPointCloud* baseVertices = new ccPointCloud("vertices");
		assert(baseVertices);
		baseVertices->setEnabled(false);
		// DGM: no need to lock it as it is only used by one mesh!
		baseVertices->setLocked(false);
		auto mesh_ptr = std::make_shared<ccMesh>(baseVertices);
		mesh_ptr->addChild(baseVertices);
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->computeVertexNormals();
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
        visualization::DrawGeometriesWithVertexSelection(
                {mesh_ptr}, "Edit Mesh", 1600, 900);
    } 
	else if (option == "spin") {
		ccPointCloud* baseVertices = new ccPointCloud("vertices");
		assert(baseVertices);
		baseVertices->setEnabled(false);
		// DGM: no need to lock it as it is only used by one mesh!
		baseVertices->setLocked(false);
		auto mesh_ptr = std::make_shared<ccMesh>(baseVertices);
		mesh_ptr->addChild(baseVertices);
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->computeVertexNormals();
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

        visualization::DrawGeometriesWithAnimationCallback(
                {mesh_ptr},
                [&](visualization::Visualizer *vis) {
                    vis->GetViewControl().Rotate(10, 0);
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    return false;
                }, "Spin", 1600, 900);
    } 
	else if (option == "slowspin") {
		ccPointCloud* baseVertices = new ccPointCloud("vertices");
		assert(baseVertices);
		baseVertices->setEnabled(false);
		// DGM: no need to lock it as it is only used by one mesh!
		baseVertices->setLocked(false);
		auto mesh_ptr = std::make_shared<ccMesh>(baseVertices);
		mesh_ptr->addChild(baseVertices);

        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->computeVertexNormals();
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

        visualization::DrawGeometriesWithKeyCallbacks(
                {mesh_ptr},
                {{GLFW_KEY_SPACE,
                  [&](visualization::Visualizer *vis) {
                      vis->GetViewControl().Rotate(10, 0);
                      std::this_thread::sleep_for(
                              std::chrono::milliseconds(30));
                      return false;
                  }}},
                "Press Space key to spin", 1600, 900);
    } 
	else if (option == "pointcloud") {
        auto cloud_ptr = std::make_shared<ccPointCloud>();
		if (io::ReadPointCloud(argv[2], *cloud_ptr)) {
			CVLib::utility::LogInfo("Successfully read {}", argv[2]);
		}
		else {
			CVLib::utility::LogWarning("Failed to read {}", argv[2]);
			return 1;
		}

        cloud_ptr->normalizeNormals();
		auto obbox = std::make_shared<ecvOrientedBBox>(cloud_ptr->getOrientedBoundingBox());
        visualization::DrawGeometries({ cloud_ptr, obbox }, "PointCloud", 1600, 900);

    } 
	else if (option == "rainbow") {
        auto cloud_ptr = std::make_shared<ccPointCloud>();
        if (io::ReadPointCloud(argv[2], *cloud_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }

        cloud_ptr->normalizeNormals();
        cloud_ptr->resizeTheRGBTable();
        double color_index = 0.0;
        double color_index_step = 0.05;

        auto update_colors_func = [&cloud_ptr](double index) {
            auto color_map_ptr = visualization::GetGlobalColorMap();
			for (unsigned int j = 0; j < cloud_ptr->size(); ++j) {
				cloud_ptr->setPointColor(j, 
					ecvColor::Rgb::FromEigen(color_map_ptr->GetColor(index)));
            }
        };
        update_colors_func(1.0);

        visualization::DrawGeometriesWithAnimationCallback(
                {cloud_ptr},
                [&](visualization::Visualizer *vis) {
                    color_index += color_index_step;
                    if (color_index > 2.0) color_index -= 2.0;
                    update_colors_func(fabs(color_index - 1.0));
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    return true;
                },
                "Rainbow", 1600, 900);
    } 
	else if (option == "image") {
        auto image_ptr = std::make_shared<geometry::Image>();
        if (io::ReadImage(argv[2], *image_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        visualization::DrawGeometries({image_ptr}, "Image", image_ptr->width_,
                                      image_ptr->height_);
    } 
	else if (option == "rgbd") {
        if (argc < 4) {
            PrintUsage();
            return 1;
        }

        int rgbd_type =
                CVLib::utility::GetProgramOptionAsInt(argc, argv, "--rgbd_type", 0);
        auto color_ptr = std::make_shared<geometry::Image>();
        auto depth_ptr = std::make_shared<geometry::Image>();

        if (io::ReadImage(argv[2], *color_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }

        if (io::ReadImage(argv[3], *depth_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[3]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[3]);
            return 1;
        }

        std::shared_ptr<geometry::RGBDImage> (*CreateRGBDImage)(
                const geometry::Image &, const geometry::Image &, bool);
        if (rgbd_type == 0)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromRedwoodFormat;
        else if (rgbd_type == 1)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromTUMFormat;
        else if (rgbd_type == 2)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromSUNFormat;
        else if (rgbd_type == 3)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromNYUFormat;
        else
            CreateRGBDImage = &geometry::RGBDImage::CreateFromRedwoodFormat;
        auto rgbd_ptr = CreateRGBDImage(*color_ptr, *depth_ptr, false);
        visualization::DrawGeometries({rgbd_ptr}, "RGBD", depth_ptr->width_ * 2,
                                      depth_ptr->height_);

    } 
	else if (option == "depth") {
        auto image_ptr = io::CreateImageFromFile(argv[2]);
        camera::PinholeCameraIntrinsic camera;
        camera.SetIntrinsics(640, 480, 575.0, 575.0, 319.5, 239.5);
        auto pointcloud_ptr = ccPointCloud::CreateFromDepthImage(*image_ptr, camera);
        visualization::DrawGeometries(
                {pointcloud_ptr},
                "ccPointCloud from Depth geometry::Image", 1920, 1080);
    } 
	else if (option == "editing") {
        auto pcd = io::CreatePointCloudFromFile(argv[2]);
        visualization::DrawGeometriesWithEditing({pcd}, "Editing", 1920, 1080);
    }
	else if (option == "axisBBox") {
		double extent = CVLib::utility::GetProgramOptionAsDouble(argc, argv, "--extent");
		double minValue = -extent / 2.0;
		double maxValue = extent / 2.0;

		auto abbox = std::make_shared<ccBBox>(
			Eigen::Vector3d(minValue, minValue, minValue), 
			Eigen::Vector3d(maxValue, maxValue, maxValue));
		visualization::DrawGeometries({ abbox }, "Axis aligned bounding box", 1600, 900);
	}
	else if (option == "orientedBBox") {
		double extent = CVLib::utility::GetProgramOptionAsDouble(argc, argv, "--extent");
		double minValue = -extent / 2.0;
		double maxValue = extent / 2.0;
		auto abbox = std::make_shared<ccBBox>(
			Eigen::Vector3d(minValue, minValue, minValue),
			Eigen::Vector3d(maxValue, maxValue, maxValue));

		auto obbox = std::make_shared<ecvOrientedBBox>(
			Eigen::Vector3d(-10, 10, 0),
			abbox->GetRotationMatrixFromXYZ(Eigen::Vector3d(2, 1, 0)),
			Eigen::Vector3d(40, 20, 20));

		visualization::DrawGeometries({ obbox }, "Oriented bounding box", 1600, 900);
	}
	else if (option == "animation") {
		ccPointCloud* baseVertices = new ccPointCloud("vertices");
		assert(baseVertices);
		auto mesh_ptr = std::make_shared<ccMesh>(baseVertices);
		baseVertices->setEnabled(false);
		// DGM: no need to lock it as it is only used by one mesh!
		baseVertices->setLocked(false);
		mesh_ptr->addChild(baseVertices);

        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            CVLib::utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            CVLib::utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }

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

        mesh_ptr->computeVertexNormals();
        if (argc == 3) {
            visualization::DrawGeometriesWithCustomAnimation(
                    {mesh_ptr}, "Animation", 1920, 1080);
        } else {
            visualization::DrawGeometriesWithCustomAnimation(
                    {mesh_ptr}, "Animation", 1600, 900, 50, 50, argv[3]);
        }
    }

    CVLib::utility::LogInfo("End of the test.");

    return 0;
}