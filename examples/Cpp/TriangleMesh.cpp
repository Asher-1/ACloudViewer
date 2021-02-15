// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#include "CloudViewer.h"

#include <ecvHObjectCaster.h>

void PrintHelp() {
    using namespace cloudViewer;
    utility::LogInfo("Usage :");
    utility::LogInfo("    > TriangleMesh sphere");
    utility::LogInfo("    > TriangleMesh merge <file1> <file2>");
    utility::LogInfo("    > TriangleMesh normal <file1> <file2>");
}

void PaintMesh(ccMesh &mesh, const Eigen::Vector3d &color) {
	ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
	if (cloud)
	{
		if (!cloud->hasColors())
		{
			cloud->reserveTheRGBTable();
		}
		
		for (size_t i = 0; i < cloud->size(); i++) {
			cloud->addEigenColor(color);
		}
	}
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    std::string option(argv[1]);
    if (option == "plane") {
        auto mesh = ccMesh::CreatePlane();
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		io::WriteTriangleMesh("plane.ply", *mesh, true, true);
	} else if (option == "cylinder") {
		auto mesh = ccMesh::CreateSphere(0.05);
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		io::WriteTriangleMesh("sphere.ply", *mesh, true, true);
	} else if (option == "cylinder") {
        auto mesh = ccMesh::CreateCylinder(0.5, 2.0);
        mesh->computeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("cylinder.ply", *mesh, true, true);
	} else if (option == "box") {
		auto mesh = ccMesh::CreateBox();
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		io::WriteTriangleMesh("box.ply", *mesh, true, true);
	} else if (option == "torus") {
		auto mesh = ccMesh::CreateTorus();
		mesh->scale(5.0, mesh->getGeometryCenter());
		*mesh += *ccMesh::CreateTorus();
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		io::WriteTriangleMesh("torus.ply", *mesh, true, true);
	} else if (option == "cone") {
        auto mesh = ccMesh::CreateCone(0.5, 2.0, 20, 3);
        mesh->computeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("cone.ply", *mesh, true, true);
    } else if (option == "arrow") {
        auto mesh = ccMesh::CreateArrow();
        mesh->computeVertexNormals();
        utility::LogInfo("Mesh has {:d} vertices, {:d} triangles.",
			mesh->getVerticeSize(), mesh->size());
		mesh->rotate(ccHObject::GetRotationMatrixFromXYZ(Eigen::Vector3d(0.3, 0.5, 0.1)),
			mesh->getGeometryCenter());
		ccBBox aabox = mesh->getAxisAlignedBoundingBox();
        utility::LogInfo("Mesh Axis Aligned bbox volume: {:f}.", aabox.volume());
		ecvOrientedBBox obb = mesh->getOrientedBoundingBox();
        utility::LogInfo("Mesh Oriented bbox volume: {:f}.", obb.volume());

		auto abbox_show = std::make_shared<ccBBox>(aabox);
		auto obbox_show = std::make_shared<ecvOrientedBBox>(obb);
		visualization::DrawGeometries({ mesh, abbox_show, obbox_show });
		aabox.scale(0.5, aabox.getGeometryCenter());
		visualization::DrawGeometries({ mesh->crop(aabox), abbox_show, obbox_show });
        io::WriteTriangleMesh("arrow.ply", *mesh, true, true);
    } else if (option == "frame") {
        if (argc < 3) {
            auto mesh = ccMesh::CreateCoordinateFrame();
            visualization::DrawGeometries({mesh});
            io::WriteTriangleMesh("frame.ply", *mesh, true, true);
        } else {
            auto mesh = io::CreateMeshFromFile(argv[2]);
            mesh->computeVertexNormals();
            auto boundingbox = mesh->getAxisAlignedBoundingBox();
            auto mesh_frame = ccMesh::CreateCoordinateFrame(
                    boundingbox.getMaxExtent() * 0.2, boundingbox.getMinBound());
            visualization::DrawGeometries({mesh, mesh_frame});
        }
    } else if (option == "merge") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh1 = io::CreateMeshFromFile(argv[2]);
        auto mesh2 = io::CreateMeshFromFile(argv[3]);
        utility::LogInfo("Mesh1 has {:d} vertices, {:d} triangles.",
                         mesh1->getVerticeSize(), mesh1->size());
        utility::LogInfo("Mesh2 has {:d} vertices, {:d} triangles.",
                         mesh2->getVerticeSize(), mesh2->size());
        *mesh1 += *mesh2;
        utility::LogInfo(
                "After merge, Mesh1 has {:d} vertices, {:d} triangles.",
                mesh1->getVerticeSize(), mesh1->size());
		mesh1->shrinkToFit();
		mesh1->removeDuplicatedVertices();
		mesh1->removeDuplicatedTriangles();
		mesh1->removeDegenerateTriangles();
		mesh1->removeUnreferencedVertices();
        utility::LogInfo(
                "After purge vertices, Mesh1 has {:d} vertices, {:d} "
                "triangles.",
                mesh1->getVerticeSize(), mesh1->size());
		visualization::DrawGeometries({ mesh1 });
		io::WriteTriangleMesh("temp.ply", *mesh1, true, true);
	} else if (option == "quadric_decimation") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		auto mesh_sub = mesh->subdivideMidpoint(2);
		auto mesh_quadric = mesh_sub->simplifyQuadricDecimation(static_cast<int>(mesh_sub->size() / 2));
		visualization::DrawGeometries({ mesh_quadric });
		if (argc >= 4)
		{
			io::WriteTriangleMesh(argv[3], *mesh_quadric, true, true);
		}
	} else if (option == "subdivide_loop") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		auto mesh_sub = mesh->subdivideLoop(3);
		visualization::DrawGeometries({ mesh_sub });
		if (argc >= 4)
		{
			io::WriteTriangleMesh(argv[3], *mesh_sub, true, true);
		}
	} else if (option == "subdivide_midpoint") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		auto mesh_sub = mesh->subdivideMidpoint(2);
		visualization::DrawGeometries({ mesh_sub });
		if (argc >= 4)
		{
			io::WriteTriangleMesh(argv[3], *mesh_sub, true, true);
		}
    } else if (option == "filter_smooth_laplacian") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		auto mesh_smooth_laplacian = mesh->filterSmoothLaplacian(100, 0.5);
		visualization::DrawGeometries({ mesh_smooth_laplacian });
		if (argc >= 4)
		{
			io::WriteTriangleMesh(argv[3], *mesh_smooth_laplacian, true, true);
		}
    } else if (option == "filter_sharpen") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		auto mesh_sharpen = mesh->filterSharpen(1, 1);
		visualization::DrawGeometries({ mesh_sharpen });
		if (argc >= 4)
		{
			io::WriteTriangleMesh(argv[3], *mesh_sharpen, true, true);
		}
	} else if (option == "ball_reconstruction") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		auto pcd = mesh->samplePointsPoissonDisk(2000);
		visualization::DrawGeometries({ pcd });
		std::vector<double> radii = { 0.005, 0.01, 0.02, 0.04 };
		auto out_mesh = ccMesh::CreateFromPointCloudBallPivoting(*pcd, radii);
		visualization::DrawGeometries({ out_mesh });
		if (argc >=4)
		{
			io::WriteTriangleMesh(argv[3], *out_mesh, true, true);
		}
	} else if (option == "normal") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
		mesh->computeVertexNormals();
		io::WriteTriangleMesh(argv[3], *mesh, true, true);

	} else if (option == "scale") {
		if (argc < 4)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
        double scale = std::stod(argv[4]);
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans(0, 0) = trans(1, 1) = trans(2, 2) = scale;
        mesh->transform(trans);
        io::WriteTriangleMesh(argv[3], *mesh);
    } else if (option == "unify") {
		if (argc < 5)
		{
			PrintHelp();
			return 1;
		}
        // unify into (0, 0, 0) - (scale, scale, scale) box
        auto mesh = io::CreateMeshFromFile(argv[2]);
        auto bbox = mesh->getAxisAlignedBoundingBox();
        double scale1 = std::stod(argv[4]);
        double scale2 = std::stod(argv[5]);
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans(0, 0) = trans(1, 1) = trans(2, 2) = scale1 / bbox.getMaxExtent();
        mesh->transform(trans);
        trans.setIdentity();
        trans.block<3, 1>(0, 3) =
                Eigen::Vector3d(scale2 / 2.0, scale2 / 2.0, scale2 / 2.0) -
                bbox.getGeometryCenter() * scale1 / bbox.getMaxExtent();
        mesh->transform(trans);
        io::WriteTriangleMesh(argv[3], *mesh);
    } else if (option == "distance") {
		if (argc < 4)
		{
			PrintHelp();
			return 1;
		}
        auto mesh1 = io::CreateMeshFromFile(argv[2]);
        auto mesh2 = io::CreateMeshFromFile(argv[3]);
        double scale = std::stod(argv[4]);
		ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(mesh1->getAssociatedCloud());
        if (!cloud || !cloud->resizeTheRGBTable())
        {
			return 1;
        }

		geometry::KDTreeFlann kdtree;
        kdtree.SetGeometry(*mesh2);
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        double r = 0.0;
        for (size_t i = 0; i < mesh1->getVerticeSize(); i++) {
            kdtree.SearchKNN(mesh1->getVertice(i), 1, indices, dists);
            double color = std::min(sqrt(dists[0]) / scale, 1.0);
			cloud->setPointColor(static_cast<unsigned int>(i), 
				ecvColor::Rgb::FromEigen(Eigen::Vector3d(color, color, color)));
            r += sqrt(dists[0]);
        }
        utility::LogInfo("Average distance is {:.6f}.",
                         r / (double)mesh1->getVerticeSize());
        if (argc > 5) {
            io::WriteTriangleMesh(argv[5], *mesh1);
        }
        visualization::DrawGeometries({mesh1});
    } else if (option == "showboth") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh1 = io::CreateMeshFromFile(argv[2]);
        PaintMesh(*mesh1, Eigen::Vector3d(1.0, 0.75, 0.0));
        auto mesh2 = io::CreateMeshFromFile(argv[3]);
        PaintMesh(*mesh2, Eigen::Vector3d(0.25, 0.25, 1.0));
        std::vector<std::shared_ptr<const ccHObject>> meshes;
        meshes.push_back(mesh1);
        meshes.push_back(mesh2);
        visualization::DrawGeometries(meshes);
    } else if (option == "colormapping") {
		if (argc < 3)
		{
			PrintHelp();
			return 1;
		}
        auto mesh = io::CreateMeshFromFile(argv[2]);
        mesh->computeVertexNormals();
        camera::PinholeCameraTrajectory trajectory;
        io::ReadIJsonConvertible(argv[3], trajectory);
        if (utility::filesystem::DirectoryExists("image") == false) {
            utility::LogWarning("No image!");
            return 0;
        }
        int idx = 3000;
        std::vector<std::shared_ptr<const ccHObject>> ptrs;
        ptrs.push_back(mesh);
        auto mesh_sphere = ccMesh::CreateSphere(0.05);
        Eigen::Matrix4d trans;
        trans.setIdentity();
        trans.block<3, 1>(0, 3) = 
			mesh->getVertice(static_cast<size_t>(idx));
        mesh_sphere->transform(trans);
        mesh_sphere->computeVertexNormals();
        ptrs.push_back(mesh_sphere);
        visualization::DrawGeometries(ptrs);

        for (size_t i = 0; i < trajectory.parameters_.size(); i += 10) {
            std::string buffer =
                    fmt::format("image/image_{:06d}.png", (int)i + 1);
            auto image = io::CreateImageFromFile(buffer);
            auto fimage = image->CreateFloatImage();
            Eigen::Vector4d pt_in_camera =
                    trajectory.parameters_[i].extrinsic_ *
                    Eigen::Vector4d(mesh->getVertice(static_cast<size_t>(idx))(0),
                                    mesh->getVertice(static_cast<size_t>(idx))(1),
                                    mesh->getVertice(static_cast<size_t>(idx))(2), 1.0);
            Eigen::Vector3d pt_in_plane =
                    trajectory.parameters_[i].intrinsic_.intrinsic_matrix_ *
                    pt_in_camera.block<3, 1>(0, 0);
            Eigen::Vector3d uv = pt_in_plane / pt_in_plane(2);
            std::cout << pt_in_camera << std::endl;
            std::cout << pt_in_plane << std::endl;
            std::cout << pt_in_plane / pt_in_plane(2) << std::endl;
            auto result = fimage->FloatValueAt(uv(0), uv(1));
            if (result.first) {
                utility::LogInfo("{:.6f}", result.second);
            }
            visualization::DrawGeometries({fimage}, "Test", 1920, 1080);
        }
    }
    return 0;
}
